# Copyright (c) 2026-2027 zh
"""内容：提供任务侧残差网络及真实物理标签统计，供视觉世界模型独立装配。"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .compilation import compile_kernel
from .tensor_state import TaskState
from .normalization import RunningNorm


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
        self.use_bf16 = False
        self._compiled_supervised = None

    def configure_supervised_compile(self, *, enabled: bool, backend: str):
        """整体编译自回归监督与损失；执行设置不进入模型参数或检查点。"""
        self._compiled_supervised = compile_kernel(self._loss, backend) if enabled else None

    @torch.no_grad()
    def prepare_statistics(self, state: TaskState, action: Tensor, target: TaskState, mask: Tensor | None = None):
        """完整批次计算输入与标签，结束行仅在统计归约中排除。"""
        return (self.input_norm.prepare_update(self.task.model_input(state, action), mask),
                self.output_norm.prepare_update(self.task.model_target(state, target), mask))

    @torch.no_grad()
    def apply_statistics(self, statistics):
        """提交已校验统计；任务子类可在此保持输出尺度约定。"""
        self.input_norm.apply_update(statistics[0])
        self.output_norm.apply_update(statistics[1])

    @torch.no_grad()
    def update_stats(self, state: TaskState, action: Tensor, target: TaskState, mask: Tensor | None = None):
        statistics = self.prepare_statistics(state, action, target, mask)
        if not torch.stack([item.finite for item in statistics]).all():
            raise FloatingPointError("Normalization data contains non-finite values")
        self.apply_statistics(statistics)

    def forward(self, state: TaskState, action: Tensor, recorded_current: TaskState | None = None) -> TaskState:
        inputs = (self.task.model_input(state, action) if recorded_current is None
                  else self.task.supervised_model_input(state, action, recorded_current))
        normalized = self.input_norm(inputs)
        if self.use_bf16:
            # 仅网络参与混合精度；反归一化及任务状态推进保持 FP32。
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                network_delta = self.net(normalized)
            network_delta = network_delta.float()
        else:
            network_delta = self.net(normalized)
        delta = self.output_norm.unnormalize(network_delta)
        physical = self.task.integrate(state, action, delta)
        # 监督推进仍消费命令采样随机数；完整损失编译不得删除这些副作用。
        return self.task.advance(state, action, physical)

    def loss(self, window) -> Tensor:
        return self._loss(window) if self._compiled_supervised is None else self._compiled_supervised(window)

    def loss_scale(self) -> Tensor:
        """默认沿用输出统计；残差模型可独立定义物理状态误差尺度。"""
        return self.output_norm.var.sqrt() + 1e-6

    def _loss(self, window) -> Tensor:
        current = window.initial
        recorded_current = window.initial
        scale = self.loss_scale()
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
