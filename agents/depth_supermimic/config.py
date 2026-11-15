# Copyright (c) 2026-2027 zh
"""内容：定义 Parkour Latent 通用算法参数及独立执行开关，机器人具体值在任务配置声明。"""

from dataclasses import dataclass
import math


@dataclass
class DepthSuperMimicConfig:
    """保留 Parkour 配方；想象总批量按用户要求采用本项目 Go2 的2048。"""

    replay_capacity: int = 500
    collect_steps: float = 2.5
    warmup_steps: int = 64
    wm_window: int = 8
    wm_updates: int = 5
    wm_batch: int = 2048
    wm_hidden: tuple[int, ...] = (1024, 1024, 1024)
    wm_failure_frac: float = 0.25
    wm_failure_horizon: int = 16
    wm_holdout_envs: float = 0.1
    warmup_iterations: int = 0
    latent_windows: int = 2048
    latent_weight: float = 1.0
    latent_recon_images: int = 512
    latent_recon_weight: float = 1.0
    auxiliary_weight: float = 1.0
    fail_windows: int = 512
    fail_head_warmup_iterations: int = 300
    fail_clip: float = 0.95
    fail_pos_weight: float = 1.0
    horizon: int = 32
    imag_batch: int = 2048
    gamma: float = 0.99
    lam: float = 0.95
    command_margin_steps: int = 2
    actor_hidden: tuple[int, ...] = (256, 128, 64)
    actor_experts: int = 4
    critic_hidden: tuple[int, ...] = (256, 128)
    actor_optimizer: str = "muon_adam"
    wm_optimizer: str = "muon_adam"
    muon_weight_decay: float = 0.1
    learning_rate: float = 4e-4
    actor_lr_scale: float = 4.0
    actor_ramp_iterations: int = 1000
    lr_decay_iterations: int = 30000
    lr_final_frac: float = 0.03
    critic_epochs: int = 5
    critic_minibatches: int = 4
    action_noise: float = 0.3
    collect_det_frac: float = 0.25
    standing_zero_noise: bool = True
    grad_clip: float = 1.0
    grad_clip_element: float = 0.1

    def __post_init__(self):
        positive = ("replay_capacity", "warmup_steps", "wm_window", "wm_updates", "wm_batch", "wm_failure_horizon",
                    "latent_windows", "latent_recon_images", "fail_windows", "horizon", "imag_batch", "actor_experts",
                    "lr_decay_iterations", "critic_epochs", "critic_minibatches")
        for name in positive + ("warmup_iterations", "fail_head_warmup_iterations", "command_margin_steps", "actor_ramp_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (1 if name in positive else 0):
                raise ValueError(f"Invalid positive integer configuration: {name}")
        if self.replay_capacity < max(self.warmup_steps, self.wm_window, self.horizon):
            raise ValueError("Replay capacity must cover warmup, supervision window and imagination horizon")
        for name in ("collect_steps", "learning_rate", "actor_lr_scale", "grad_clip", "grad_clip_element"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Configuration must be finite and positive: {name}")
        for name in ("gamma", "lam", "wm_failure_frac", "wm_holdout_envs", "fail_clip", "lr_final_frac"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) < 1:
                raise ValueError(f"Configuration must be in [0,1): {name}")
        for name in ("latent_weight", "latent_recon_weight", "auxiliary_weight", "action_noise", "fail_pos_weight", "muon_weight_decay"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"Configuration must be finite and nonnegative: {name}")
        if not math.isfinite(self.collect_det_frac) or not 0 <= self.collect_det_frac <= 1:
            raise ValueError("Deterministic collection fraction must be in [0,1]")
        if not isinstance(self.standing_zero_noise, bool):
            raise ValueError("Standing zero noise must be a boolean")
        for name in ("wm_hidden", "actor_hidden", "critic_hidden"):
            value = tuple(getattr(self, name))
            if not value or any(isinstance(width, bool) or not isinstance(width, int) or width < 1 for width in value):
                raise ValueError(f"Hidden widths must be positive integers: {name}")
            setattr(self, name, value)

        for name in ("actor_optimizer", "wm_optimizer"):
            if getattr(self, name) not in ("adam", "muon_adam"):
                raise ValueError(f"Unsupported optimizer: {name}={getattr(self, name)}")

    @classmethod
    def from_checkpoint(cls, values):
        """历史检查点缺失新增字段时保留原优化器、学习率和采集行为。"""
        return cls(**{"actor_optimizer": "adam", "wm_optimizer": "adam", "actor_lr_scale": 1.0,
                      "actor_ramp_iterations": 0, "collect_det_frac": 0.0, "standing_zero_noise": False, **values})


@dataclass
class DepthExecutionConfig:
    """分批保持总样本数、完整 BPTT 时间链和优化器步数。"""

    compile: bool = True
    compile_backend: str = "inductor"
    fused_adam: bool = True
    cuda_graph: bool = True
    actor_bf16: bool = True
    encoder_bf16: bool = True
    wm_bf16: bool = True
    critic_bf16: bool = True
    decoder_bf16: bool = True
    imag_microbatch: int = 4096
    critic_microbatch: int = 32768
    metric_interval: int = 50
    replay_history_storage: str = "frames"

    def __post_init__(self):
        for name in ("imag_microbatch", "critic_microbatch", "metric_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Execution value must be a positive integer: {name}")
        if self.compile_backend not in ("inductor", "eager"):
            raise ValueError("Compile backend must be inductor or eager")
        if self.replay_history_storage not in ("full", "frames"):
            raise ValueError("Replay history storage must be full or frames")
        if self.cuda_graph and not self.fused_adam:
            raise ValueError("CUDA graphs require fused Adam")
