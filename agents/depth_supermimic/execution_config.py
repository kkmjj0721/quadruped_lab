# Copyright (c) 2026-2027 zh
"""内容：声明深度循环策略的微批、编译和混合精度执行参数。"""

from dataclasses import dataclass


RUNNER_RUNTIME_FIELDS = ("max_iterations", "save_interval", "log_interval", "metric_interval", "imag_microbatch",
                         "critic_microbatch", "wm_compile", "fused_adam", "cuda_graph",
                         "wm_bf16", "actor_bf16", "critic_bf16")


@dataclass
class ExecutionConfig:
    """声明独立视觉任务的执行边界，不改变总批量和完整时间反传。"""

    imag_microbatch: int = 16
    critic_microbatch: int = 256
    wm_compile: bool = False  # 完整自回归监督及损失。
    fused_adam: bool = False
    cuda_graph: bool = False  # 捕获完整模型、策略联合重建与价值更新；要求CUDA和fused Adam。
    wm_bf16: bool = False  # 仅世界模型MLP使用CUDA BF16；监督和想象共享该边界。
    actor_bf16: bool = False  # 仅策略MLP使用CUDA BF16；CNN、GRU与归一化保留FP32。
    critic_bf16: bool = False  # 仅价值MLP使用CUDA BF16；输出和返回递推保留FP32。

    def __post_init__(self):
        for name in ("imag_microbatch", "critic_microbatch"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("wm_compile", "fused_adam", "cuda_graph", "wm_bf16", "actor_bf16", "critic_bf16"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")


