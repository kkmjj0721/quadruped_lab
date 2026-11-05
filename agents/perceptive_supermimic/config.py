# Copyright (c) 2026-2027 zh
"""内容：声明同步视觉Latent训练参数和不改变优化语义的分批预算。"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ExecutionConfig:
    """只拆分同一次更新内部的批次，各批累计后统一更新参数。"""

    wm_microbatch: int = 256
    vision_batch: int = 128
    imag_microbatch: int = 256
    critic_microbatch: int = 4096

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class PerceptiveConfig:
    """网络宽度来自本项目设计；8步拟合、32步想象等机制参考指定InstinctMJ。"""

    replay_capacity: int = 1000
    image_capacity: int = 1024
    collect_steps: float = 2.5
    warmup_steps: int = 64
    wm_only_iterations: int = 500
    wm_window: int = 8
    wm_batch: int = 2048
    wm_updates: int = 10
    imag_batch: int = 1024
    horizon: int = 32
    latent_dim: int = 32
    proprio_latent_dim: int = 32
    history_hidden: tuple[int, ...] = (256, 128)
    actor_hidden: tuple[int, ...] = (512, 256, 128)
    critic_hidden: tuple[int, ...] = (512, 256, 128)
    wm_hidden: tuple[int, ...] = (1024, 1024, 1024)
    reconstruction_images: int = 512
    contact_weight: float = 1.0
    latent_weight: float = 1.0
    reconstruction_weight: float = 1.0
    gamma: float = 0.99
    lam: float = 0.95
    learning_rate: float = 2e-4
    lr_decay_iterations: int = 250000
    lr_final_ratio: float = 0.03
    critic_epochs: int = 5
    critic_minibatches: int = 4
    grad_clip: float = 1.0
    actor_grad_value: float = 0.1
    action_noise: float = 0.1
    action_clip: float = 20.0
    log_interval: int = 50
    save_interval: int = 1000

    def __post_init__(self):
        positive = ("replay_capacity", "image_capacity", "warmup_steps", "wm_window", "wm_batch", "wm_updates",
                    "imag_batch", "horizon", "latent_dim", "proprio_latent_dim", "reconstruction_images",
                    "lr_decay_iterations", "critic_epochs", "critic_minibatches", "log_interval", "save_interval")
        for name in positive + ("wm_only_iterations",):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (1 if name in positive else 0):
                raise ValueError(f"Invalid integer budget: {name}")
        for name in ("history_hidden", "actor_hidden", "critic_hidden", "wm_hidden"):
            widths = tuple(getattr(self, name))
            if not widths or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in widths):
                raise ValueError(f"{name} must contain positive integer widths")
            object.__setattr__(self, name, widths)
        for name in ("collect_steps", "learning_rate", "grad_clip", "actor_grad_value", "action_clip", "reconstruction_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.collect_steps < 1:
            raise ValueError("collect_steps must be >= 1")
        for name in ("action_noise", "contact_weight", "latent_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("gamma", "lam", "lr_final_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be within [0,1]")
        if self.gamma == 1 or self.lr_final_ratio == 0:
            raise ValueError("Discount must be < 1 and final learning-rate ratio must be > 0")
        if self.replay_capacity < self.wm_window or self.image_capacity < self.wm_window + 2:
            raise ValueError("Replay and image capacities cannot hold the model window")
        if self.warmup_steps < self.wm_window or self.critic_minibatches > self.imag_batch * self.horizon:
            raise ValueError("Warmup or imagined batch cannot fill the configured windows/minibatches")
