# Copyright (c) 2026-2027 zh
"""内容：定义深度循环网络尺寸及保留完整八步监督、三十二步想象的训练配置。"""

from dataclasses import dataclass
import math

from .execution_config import RUNNER_RUNTIME_FIELDS


@dataclass
class DepthNetworkConfig:
    """固定图像和状态协议；隐藏层宽度可在独立测试预算中显式覆盖。"""

    image_height: int = 64
    image_width: int = 64
    observation_dim: int = 47
    history_length: int = 5
    action_dim: int = 12
    feature_dim: int = 32
    hidden_dim: int = 128
    latent_dim: int = 32
    physical_input_dim: int = 239
    physical_output_dim: int = 80
    privileged_dim: int = 262
    camera_period: int = 5
    actor_hidden: tuple[int, ...] = (512, 256, 128)
    wm_hidden: tuple[int, ...] = (1024, 1024, 1024)
    critic_hidden: tuple[int, ...] = (512, 256, 128)

    def __post_init__(self):
        if (self.image_height, self.image_width) != (64, 64):
            raise ValueError("Depth images must have height 64 and width 64")
        for name in ("observation_dim", "history_length", "action_dim", "feature_dim", "hidden_dim", "latent_dim",
                     "physical_input_dim", "physical_output_dim", "privileged_dim", "camera_period"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.camera_period < 2:
            raise ValueError("camera_period must be at least two control steps")
        for name in ("actor_hidden", "wm_hidden", "critic_hidden"):
            widths = tuple(getattr(self, name))
            if not widths or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in widths):
                raise ValueError(f"{name} must contain positive integer widths")
            setattr(self, name, widths)


@dataclass
class DepthSuperMimicConfig:
    """独立声明视觉 SuperMimic 实际使用的算法字段，不带盲策略或教师分支。"""

    replay_capacity: int = 1000
    collect_steps: float = 2.5
    warmup_iterations: int = 100
    wm_hidden: tuple[int, ...] = (1024, 1024, 1024)
    wm_window: int = 8
    wm_batch: int = 2048
    wm_updates: int = 5
    imag_batch: int = 2048
    horizon: int = 32
    gamma: float = 0.99
    lam: float = 0.95
    actor_hidden: tuple[int, ...] = (512, 256, 128)
    critic_hidden: tuple[int, ...] = (512, 256, 128)
    learning_rate: float = 2e-4
    actor_lr_scale: float = 1.0
    actor_optimizer: str = "adam"  # Go1配方显式选择Muon；视觉感知和解码器仍由Adam更新。
    wm_optimizer: str = "adam"
    muon_weight_decay: float = 0.1
    actor_ramp_iterations: int = 1000
    lr_decay_iterations: int = 30000
    lr_final_frac: float = 0.03
    critic_epochs: int = 5
    critic_minibatches: int = 4
    grad_clip: float = 1.0
    grad_clip_element: float = 0.1
    compile_imagination: bool = False
    compile_backend: str = "inductor"
    action_noise: float = 0.3
    collect_det_frac: float = 0.0
    wm_prioritized_frac: float = 0.25
    wm_archive_frac: float = 0.25
    archive_size: int = 100000
    archive_keep_prob: float = 0.02
    reconstruction_coef: float = 1.2
    feature_coef: float = 1.0
    collapse_falls: float = 3.0
    collapse_interval: int = 50

    def __post_init__(self):
        for name in ("actor_optimizer", "wm_optimizer"):
            if getattr(self, name) not in ("adam", "muon_adam"):
                raise ValueError(f"{name} must be adam or muon_adam")
        positive = ("replay_capacity", "wm_window", "wm_batch", "wm_updates", "imag_batch", "horizon",
                    "lr_decay_iterations", "critic_epochs", "critic_minibatches", "collapse_interval")
        for name in positive + ("warmup_iterations", "actor_ramp_iterations", "archive_size"):
            value = getattr(self, name)
            minimum = 1 if name in positive else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if (isinstance(self.collect_steps, bool) or not isinstance(self.collect_steps, (int, float))
                or not math.isfinite(self.collect_steps) or self.collect_steps < 1):
            raise ValueError("collect_steps must be a finite number >= 1")
        for name in ("wm_hidden", "actor_hidden", "critic_hidden"):
            widths = tuple(getattr(self, name))
            if not widths or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in widths):
                raise ValueError(f"{name} must contain positive integer widths")
            setattr(self, name, widths)
        for name in ("gamma", "lam", "lr_final_frac", "collect_det_frac", "wm_prioritized_frac",
                     "wm_archive_frac", "archive_keep_prob"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be finite and within [0,1]")
        if self.gamma == 1 or self.lr_final_frac == 0 or self.wm_prioritized_frac + self.wm_archive_frac > 1:
            raise ValueError("Invalid discount, final learning rate or replay fractions")
        for name in ("learning_rate", "actor_lr_scale", "grad_clip", "grad_clip_element"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("action_noise", "reconstruction_coef", "feature_coef", "muon_weight_decay", "collapse_falls"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(self.compile_imagination, bool):
            raise ValueError("compile_imagination must be a boolean")
        if self.compile_backend not in ("inductor", "eager"):
            raise ValueError("compile_backend must be inductor or eager")
