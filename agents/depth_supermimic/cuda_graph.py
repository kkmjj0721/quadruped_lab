# Copyright (c) 2026-2027 zh
"""内容：以固定准备批次捕获深度模型、完整策略反传和价值更新，保留原训练预算。"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import gc

import torch

from .learner import DepthLearner


# ---------- 静态输入与实际更新预热 ----------

@contextmanager
def _capture_without_gc():
    """仅捕获期间暂停循环回收，避免析构编译内核；始终恢复原开关。"""
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def _clone_inputs(value):
    """递归为准备批次保留独立存储，不把可变长度原始前缀带入图。"""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if is_dataclass(value):
        return type(value)(**{field.name: _clone_inputs(getattr(value, field.name)) for field in fields(value)})
    if isinstance(value, tuple):
        return tuple(_clone_inputs(item) for item in value)
    raise TypeError(f"Unsupported CUDA-graph input type: {type(value).__name__}")


@torch.no_grad()
def _copy_inputs(target, source):
    """只复制值，输入结构、形状、类型和设备必须与捕获时一致。"""
    if type(target) is not type(source):
        raise ValueError("CUDA-graph input structure changed")
    if isinstance(target, torch.Tensor):
        if target.shape != source.shape or target.dtype != source.dtype or target.device != source.device:
            raise ValueError("CUDA-graph input shape, dtype or device changed")
        target.copy_(source)
    elif is_dataclass(target):
        for field in fields(target):
            _copy_inputs(getattr(target, field.name), getattr(source, field.name))
    else:
        for destination, current in zip(target, source, strict=True):
            _copy_inputs(destination, current)


class CapturedUpdate:
    """前三次真实更新在侧流预热，第四次捕获并重放；不额外训练。"""

    def __init__(self, function, device: torch.device, *, warmup: int = 3):
        self.function, self.device, self.warmup = function, torch.device(device), warmup
        if self.device.type != "cuda":
            raise ValueError("CUDA-graph updates require a CUDA device")
        self.calls = self.replays = 0
        self.graph = self.inputs = self.outputs = None
        self.stream = torch.cuda.Stream(device=self.device)

    def __call__(self, *inputs):
        with torch.cuda.device(self.device):
            if self.inputs is None:
                self.inputs = _clone_inputs(inputs)
            else:
                _copy_inputs(self.inputs, inputs)
            self.calls += 1
            current = torch.cuda.current_stream(self.device)
            if self.calls <= self.warmup:
                self.stream.wait_stream(current)
                with torch.cuda.stream(self.stream):
                    outputs = self.function(*self.inputs)
                current.wait_stream(self.stream)
                return outputs
            if self.graph is None:
                torch.cuda.synchronize(self.device)
                with _capture_without_gc():
                    self.graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.graph, stream=self.stream):
                        self.outputs = self.function(*self.inputs)
                # 捕获只记录CUDA工作，本次真实更新由下面的replay执行。
                print(f"[INFO] Captured CUDA update graph: {self.function.__name__}", flush=True)
            self.graph.replay()
            self.replays += 1
            return self.outputs


# ---------- 三类更新共享学习器数学核心 ----------

class CudaGraphDepthLearner(DepthLearner):
    """图外准备完整真实前缀，图内保留最新感知与完整八／三十二步更新。"""

    supports_cuda_graph = True

    def __init__(self, task, actor, decoder, model, critic, config, execution=None):
        self.device = next(model.parameters()).device
        if self.device.type != "cuda":
            raise ValueError("CUDA-graph updates require a CUDA device")
        if execution is None or not execution.cuda_graph or not execution.fused_adam:
            raise ValueError("CUDA-graph updates require cuda_graph=True and fused_adam=True")
        super().__init__(task, actor, decoder, model, critic, config, execution)
        self._reset_graphs()

    def _make_optimizer(self, name, *, capturable=True):
        optimizer = super()._make_optimizer(name, capturable=True)
        for group in optimizer.param_groups:
            group["lr"] = torch.tensor(float(group["lr"]), device=self.device)
        return optimizer

    def _reset_graphs(self):
        self._model_update = CapturedUpdate(self._model_core, self.device)
        self._actor_updates = {}
        self._critic_update = None

    def schedule(self, iteration: int):
        references = [(group, group["lr"]) for optimizer in
                      (self.actor_optimizer, self.model_optimizer, self.critic_optimizer)
                      for group in optimizer.param_groups]
        super().schedule(iteration)
        for group, reference in references:
            reference.fill_(group["lr"])
            group["lr"] = reference

    def load_optimizer_state_dict(self, name, data):
        """恢复Adam或Muon复合状态后重建所有图，设备学习率地址保持不变。"""
        super().load_optimizer_state_dict(name, data)
        self._reset_graphs()

    def reset_policy_optimizers(self):
        """策略回滚只清除策略／价值捕获，模型图与模型优化器继续复用。"""
        super().reset_policy_optimizers()
        self._actor_updates = {}
        self._critic_update = None

    def _run_model_core(self, parts):
        return self._model_update(parts)

    def _run_actor_core(self, seed, *, freeze_actor=False):
        if freeze_actor not in self._actor_updates:
            self._actor_updates[freeze_actor] = CapturedUpdate(
                lambda value: self._actor_core(value, freeze_actor=freeze_actor), self.device)
        return self._actor_updates[freeze_actor](seed)

    def _run_critic_core(self, features, targets):
        if self._critic_update is None:
            self._critic_update = CapturedUpdate(self._critic_core, self.device)
        return self._critic_update(features, targets)

    def timing_setup_pending(self, *, control_enabled=False, freeze_actor=False):
        """查询实际使用的图是否仍需预热或捕获，不创建或执行新更新。"""
        if self._model_update.graph is None:
            return True
        if not control_enabled:
            return False
        actor = self._actor_updates.get(freeze_actor)
        return (actor is None or actor.graph is None or self._critic_update is None
                or self._critic_update.graph is None)

    def fit_model(self, window):
        metrics = super().fit_model(window)
        metrics.update({"cuda_graph/model_captured": float(self._model_update.graph is not None),
                        "cuda_graph/model_replays": self._model_update.replays})
        return metrics

    def improve(self, seed, *, freeze_actor=False, timing=None):
        metrics = super().improve(seed, freeze_actor=freeze_actor, timing=timing)
        actor = self._actor_updates[freeze_actor]
        metrics.update({"cuda_graph/actor_captured": float(actor.graph is not None),
                        "cuda_graph/actor_replays": actor.replays,
                        "cuda_graph/critic_captured": float(self._critic_update.graph is not None),
                        "cuda_graph/critic_replays": self._critic_update.replays})
        return metrics
