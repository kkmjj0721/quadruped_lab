# Copyright (c) 2026-2027 zh
"""内容：声明混合专家历史的参考训练预算、预热排期和显式资源覆盖参数。"""

from dataclasses import dataclass
import math


@dataclass
class ExecutionConfig:
    """限制同时参与计算的样本数，不改变总训练批量或检查点语义。"""

    imag_microbatch: int = 4096
    critic_microbatch: int = 32768

    def __post_init__(self):
        for name in ("imag_microbatch", "critic_microbatch"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass
class SuperMimicConfig:
    """采集调度按8bfc115，模型批量按论文表A.5，其余预算参考d9eee9b3，专家结构按30e74dc。"""

    replay_capacity: int = 1000
    # 每轮平均真实控制步；2.5对应UTD=0.4，循环累计后取整。
    collect_steps: float = 2.5
    warmup_iterations: int = 300
    wm_rewarmup_iterations: int = 1500
    critic_warmup_iterations: int = 500
    wm_hidden: tuple[int, ...] = (1024, 1024, 1024)
    wm_window: int = 8
    wm_batch: int = 2048
    wm_updates: int = 10
    imag_batch: int = 4096
    horizon: int = 32
    gamma: float = 0.99
    lam: float = 0.95
    actor_hidden: tuple[int, ...] = (512, 256, 128)
    critic_hidden: tuple[int, ...] = (512, 256, 128)
    moe_expert_num: int = 8
    moe_hidden: tuple[int, ...] = (512, 256, 256)
    moe_latent_dim: int = 32
    learning_rate: float = 2e-4
    actor_lr_scale: float = 1.0
    actor_ramp_iterations: int = 1000
    lr_decay_iterations: int = 30000
    lr_final_frac: float = 0.03
    critic_epochs: int = 5
    critic_minibatches: int = 4
    grad_clip: float = 1.0
    grad_clip_element: float = 0.1
    compile_imagination: bool = True
    compile_backend: str = "inductor"
    action_noise: float = 0.1
    explore_frac: float = 0.25
    explore_noise: float = 0.3
    explore_theta: float = 2.0
    wm_prioritized_frac: float = 0.25
    wm_archive_frac: float = 0.25
    archive_size: int = 100000
    archive_keep_prob: float = 0.02
    collapse_falls: float = 3.0
    collapse_interval: int = 50
    driver_frac: float = 0.25
    expert_frac: float = 0.0
    driver_fade: tuple[int, int] = (1_000_000_000, 1_000_000_000)
    driver_vmin: float = 0.5
    driver_vmax: float = 3.0

    def __post_init__(self):
        positive = ("replay_capacity", "wm_window", "wm_batch",
                    "wm_updates", "imag_batch", "horizon", "lr_decay_iterations", "critic_epochs",
                    "critic_minibatches", "collapse_interval", "moe_expert_num", "moe_latent_dim")
        for name in positive + ("warmup_iterations", "wm_rewarmup_iterations", "critic_warmup_iterations", "actor_ramp_iterations", "archive_size"):
            value = getattr(self, name)
            minimum = 1 if name in positive else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if (isinstance(self.collect_steps, bool) or not isinstance(self.collect_steps, (int, float))
                or not math.isfinite(self.collect_steps) or self.collect_steps < 1):
            raise ValueError("collect_steps must be a finite number >= 1")
        for name in ("wm_hidden", "actor_hidden", "critic_hidden", "moe_hidden"):
            values = tuple(getattr(self, name))
            if not values or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in values):
                raise ValueError(f"{name} must contain positive integer widths")
            setattr(self, name, values)
        for name in ("gamma", "lam", "lr_final_frac", "explore_frac", "wm_prioritized_frac", "wm_archive_frac", "archive_keep_prob"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and within [0,1]")
        if self.gamma == 1 or self.lr_final_frac == 0 or self.wm_prioritized_frac + self.wm_archive_frac > 1:
            raise ValueError("Invalid discount, final learning rate or replay fractions")
        for name in ("learning_rate", "actor_lr_scale", "grad_clip", "grad_clip_element", "explore_theta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("action_noise", "explore_noise", "collapse_falls", "driver_vmin", "driver_vmax"):
            if isinstance(getattr(self, name), bool) or not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("driver_frac", "expert_frac"):
            if isinstance(getattr(self, name), bool) or not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be finite and within [0,1]")
        self.driver_fade = tuple(self.driver_fade)
        if (len(self.driver_fade) != 2 or any(isinstance(x, bool) or not isinstance(x, int) for x in self.driver_fade)
                or self.driver_frac + self.expert_frac > 1):
            raise ValueError("Driver fade must be two iterations and driver plus expert share cannot exceed one")
        if self.driver_frac and (self.driver_vmin <= 0 or self.driver_vmax < self.driver_vmin):
            raise ValueError("Driver speed range must be positive and ordered")
        if not isinstance(self.compile_imagination, bool):
            raise ValueError("compile_imagination must be a boolean")
        if not isinstance(self.compile_backend, str) or not self.compile_backend:
            raise ValueError("Invalid compilation settings")
