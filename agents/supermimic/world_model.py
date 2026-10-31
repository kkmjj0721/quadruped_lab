# Copyright (c) 2026-2027 zh
"""内容：提供统计归一化和只学习记录物理转移的动力学网络。"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .interfaces import TaskState


class RunningNorm(nn.Module):
    """按参考实现合并总体方差；前向固定统计，不隐式更新。"""

    def __init__(self, width: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("var", torch.ones(width))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, values: Tensor):
        values = values.reshape(-1, self.mean.numel())
        if values.shape[0] == 0:
            return
        if not torch.isfinite(values).all():
            raise FloatingPointError("Normalization data contains non-finite values")
        variance, mean = torch.var_mean(values, dim=0, correction=0)
        count = values.shape[0]
        total = self.count + count
        difference = mean - self.mean
        self.var.copy_((self.var * self.count + variance * count + difference.square() * self.count * count / total) / total)
        self.mean.add_(difference * count / total)
        self.count.copy_(total)

    def forward(self, values: Tensor) -> Tensor:
        return ((values - self.mean) / (self.var.sqrt() + 1e-6)).clamp(-10.0, 10.0)

    def unnormalize(self, values: Tensor) -> Tensor:
        return values * (self.var.sqrt() + 1e-6) + self.mean


class WorldModel(nn.Module):
    """任务决定残差坐标；网络只学习记录物理转移，不学习奖励或失败头。"""

    def __init__(self, task, hidden):
        super().__init__()
        self.task = task
        self.spec = task.spec
        layers, width = [], task.model_input_dim
        for hidden_width in hidden:
            layers.extend((nn.Linear(width, hidden_width), nn.SiLU()))
            width = hidden_width
        layers.append(nn.Linear(width, task.spec.delta_dim))
        self.net = nn.Sequential(*layers)
        self.input_norm = RunningNorm(task.model_input_dim)
        self.output_norm = RunningNorm(task.spec.delta_dim)
        self._supervised_compile_backend = None
        self._compiled_supervised = None

    def configure_supervised_compile(self, *, enabled: bool, backend: str):
        """只编译监督物理核心；执行设置不进入模型参数或检查点。"""
        self._supervised_compile_backend = backend if enabled else None
        self._compiled_supervised = None

    def _predict_supervised_physical(self, state, action, recorded_current):
        inputs = self.task.supervised_model_input(state, action, recorded_current)
        delta = self.output_norm.unnormalize(self.net(self.input_norm(inputs)))
        return self.task.integrate(state, action, delta)

    @torch.no_grad()
    def update_stats(self, state: TaskState, action: Tensor, target: TaskState):
        self.input_norm.update(self.task.model_input(state, action))
        self.output_norm.update(self.task.model_target(state, target))

    def forward(self, state: TaskState, action: Tensor, recorded_current: TaskState | None = None) -> TaskState:
        if recorded_current is not None and self._supervised_compile_backend is not None:
            if self._compiled_supervised is None:
                arguments = {"backend": self._supervised_compile_backend, "fullgraph": True, "dynamic": False}
                if self._supervised_compile_backend == "inductor":
                    # 保留监督地形relief显式half→float的舍入边界。
                    arguments["options"] = {"emulate_precision_casts": True}
                self._compiled_supervised = torch.compile(self._predict_supervised_physical, **arguments)
            physical = self._compiled_supervised(state, action, recorded_current)
        else:
            inputs = (self.task.model_input(state, action) if recorded_current is None
                      else self.task.supervised_model_input(state, action, recorded_current))
            delta = self.output_norm.unnormalize(self.net(self.input_norm(inputs)))
            physical = self.task.integrate(state, action, delta)
        # 监督推进仍消耗原有命令采样随机数；想象前向不接入该编译核心。
        return self.task.advance(state, action, physical)

    def loss(self, window) -> Tensor:
        current = window.initial
        recorded_current = window.initial
        scale = self.output_norm.var.sqrt() + 1e-6
        total = current.physical.new_zeros(())
        for step in range(window.actions.shape[1]):
            target = window.targets.index((slice(None), step))
            action = window.actions[:, step]
            predicted = self(current, action, recorded_current)
            error = self.task.model_error(current, predicted, target, recorded_current) / scale
            total = total + (error.square().mean(-1) * window.valid[:, step]).sum()
            # 本协议的记忆只含记录动作与外生计时，按日志恢复命令重采样；物理状态始终回喂预测。
            current = TaskState(predicted.physical, target.memory, target.command, target.context)
            recorded_current = target
        return total / window.valid.sum().clamp_min(1)
