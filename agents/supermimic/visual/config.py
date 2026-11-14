# Copyright (c) 2026-2027 zh
"""内容：声明视觉辅助学习的优化配置，机器人任务配置由调用方组合。"""

from dataclasses import dataclass
import math

from ..config import ExecutionConfig


@dataclass
class VisualExecutionConfig(ExecutionConfig):
    """扩展视觉两路精度，不向盲走任务配置引入无关字段。"""

    perception_bf16: bool = False  # 感知卷积、MLP和GRU使用CUDA BF16，对外记忆保留FP32。
    planner_bf16: bool = False  # 选点网络和注意力矩阵使用CUDA BF16，几何及概率保留FP32。

    def __post_init__(self):
        super().__post_init__()
        for name in ("perception_bf16", "planner_bf16"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class AuxiliaryConfig:
    """感知优化沿用 MGDP Stage1；选点学习率沿用 ACP Table S5。"""

    perception_lr: float = 1e-4
    perception_weight_decay: float = 1e-5
    planner_lr: float = 1e-4
    grad_clip: float = 1.0

    def __post_init__(self):
        for name in ("perception_lr", "planner_lr", "grad_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.perception_weight_decay) or self.perception_weight_decay < 0:
            raise ValueError("perception_weight_decay must be finite and nonnegative")
