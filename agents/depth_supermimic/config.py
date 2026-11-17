# Copyright (c) 2026-2027 zh
"""内容：声明四足查询深度策略的训练预算、预热排期和显式资源覆盖参数。"""

from dataclasses import dataclass
import math


RUNNER_RUNTIME_FIELDS = ("max_iterations", "save_interval", "log_interval", "imag_microbatch",
                         "critic_microbatch", "wm_compile", "fused_adam", "cuda_graph",
                         "detailed_timing", "timing_interval", "replay_history_storage",
                         "wm_bf16", "actor_bf16", "critic_bf16", "metric_interval")


@dataclass
class ExecutionConfig:
    """声明分批和默认s101执行方式，不改变总训练批量或检查点语义。"""

    imag_microbatch: int = 4096
    critic_microbatch: int = 32768
    wm_compile: bool = True  # 完整自回归监督及损失。
    fused_adam: bool = True
    cuda_graph: bool = True  # CUDA捕获采样与更新，融合指标累计；CPU由runner解析为False。
    detailed_timing: bool = False  # 图外记录阶段事件，原有计时指标保持原义。
    timing_interval: int = 1  # 详细计时的采样轮数间隔；首次使用和图准备轮强制记录。
    metric_interval: int = 1  # 设备指标汇总及TensorBoard写入间隔；护栏和learn返回强制完成窗口。
    replay_history_storage: str = "frames"  # 默认只保存实际带噪单帧；full保留完整历史作显式对照。
    wm_bf16: bool = False  # 仅世界模型MLP使用CUDA BF16；监督和想象共享该边界。
    actor_bf16: bool = False  # 仅策略MLP使用CUDA BF16；深度卷积、查询与归一化保留FP32。
    critic_bf16: bool = False  # 仅价值MLP使用CUDA BF16；输出和返回递推保留FP32。

    def __post_init__(self):
        if self.replay_history_storage not in ("full", "frames"):
            raise ValueError("replay_history_storage must be full or frames")
        for name in ("imag_microbatch", "critic_microbatch", "timing_interval", "metric_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("wm_compile", "fused_adam", "cuda_graph", "detailed_timing", "wm_bf16", "actor_bf16", "critic_bf16"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")


@dataclass
class DepthSuperMimicConfig:
    """共享算法字段、通用默认与校验；各机器人在自己的algo段声明实际训练配方。"""

    replay_capacity: int = 1000
    # 每轮平均真实控制步；2.5对应UTD=0.4，循环累计后取整。
    collect_steps: float = 2.5
    warmup_iterations: int = 0
    wm_rewarmup_iterations: int = 1500
    critic_warmup_iterations: int = 500
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
    learning_rate: float = 4e-4
    actor_lr_scale: float = 4.0
    actor_optimizer: str = "adam"  # muon_adam将策略全部Linear权重交给Muon，其余参数使用Adam。
    wm_optimizer: str = "adam"  # muon_adam仅将世界模型中间Linear权重交给Muon。
    muon_weight_decay: float = 0.1  # PartyMimic默认值，仅作用于Muon参数组。
    actor_ramp_iterations: int = 1000
    lr_decay_iterations: int = 30000
    lr_final_frac: float = 0.03
    critic_epochs: int = 5
    critic_minibatches: int = 4
    grad_clip: float = 1.0
    grad_clip_element: float = 0.1
    compile_imagination: bool = True  # 想象单步、完整lambda-return及critic损失。
    compile_backend: str = "inductor"
    # 论文表A.5：仅真实采集时逐步逐维独立添加固定高斯噪声，单位为动作缩放前单位。
    action_noise: float = 0.3
    collect_det_frac: float = 0.25  # 无额外动作探索噪声的环境比例；机器人配方显式选择。
    wm_prioritized_frac: float = 0.25
    wm_archive_frac: float = 0.25
    archive_size: int = 100000
    archive_keep_prob: float = 0.02
    collapse_falls: float = 3.0
    collapse_interval: int = 50

    auxiliary_batch: int = 128
    auxiliary_updates: int = 1
    visual_refresh_steps: int = 8
    read_loss_coef: float = 0.01
    frame_window: int = 4
    camera_period: int = 5
    encoder_microbatch: int = 256

    def __post_init__(self):
        for name in ("auxiliary_batch", "auxiliary_updates", "visual_refresh_steps", "encoder_microbatch"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.frame_window != 4 or self.camera_period != 5:
            raise ValueError("Depth query protocol requires four frames and five control steps per camera frame")
        if not math.isfinite(self.read_loss_coef) or self.read_loss_coef < 0:
            raise ValueError("read_loss_coef must be finite and nonnegative")
        for name in ("actor_optimizer", "wm_optimizer"):
            if getattr(self, name) not in ("adam", "muon_adam"):
                raise ValueError(f"{name} must be adam or muon_adam")
        if (isinstance(self.muon_weight_decay, bool) or not isinstance(self.muon_weight_decay, (int, float))
                or not math.isfinite(self.muon_weight_decay) or self.muon_weight_decay < 0):
            raise ValueError("muon_weight_decay must be finite and nonnegative")
        positive = ("replay_capacity", "wm_window", "wm_batch",
                    "wm_updates", "imag_batch", "horizon", "lr_decay_iterations", "critic_epochs",
                    "critic_minibatches", "collapse_interval")
        for name in positive + ("warmup_iterations", "wm_rewarmup_iterations", "critic_warmup_iterations", "actor_ramp_iterations", "archive_size"):
            value = getattr(self, name)
            minimum = 1 if name in positive else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if (isinstance(self.collect_steps, bool) or not isinstance(self.collect_steps, (int, float))
                or not math.isfinite(self.collect_steps) or self.collect_steps < 1):
            raise ValueError("collect_steps must be a finite number >= 1")
        for name in ("wm_hidden", "actor_hidden", "critic_hidden"):
            values = tuple(getattr(self, name))
            if not values or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in values):
                raise ValueError(f"{name} must contain positive integer widths")
            setattr(self, name, values)
        for name in ("gamma", "lam", "lr_final_frac", "wm_prioritized_frac", "wm_archive_frac", "archive_keep_prob"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and within [0,1]")
        if (isinstance(self.collect_det_frac, bool) or not isinstance(self.collect_det_frac, (int, float))
                or not math.isfinite(self.collect_det_frac) or not 0 <= self.collect_det_frac <= 1):
            raise ValueError("collect_det_frac must be finite and within [0,1]")
        if self.gamma == 1 or self.lr_final_frac == 0 or self.wm_prioritized_frac + self.wm_archive_frac > 1:
            raise ValueError("Invalid discount, final learning rate or replay fractions")
        for name in ("learning_rate", "actor_lr_scale", "grad_clip", "grad_clip_element"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("action_noise", "collapse_falls"):
            if isinstance(getattr(self, name), bool) or not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(self.compile_imagination, bool):
            raise ValueError("compile_imagination must be a boolean")
        if not isinstance(self.compile_backend, str) or not self.compile_backend:
            raise ValueError("Invalid compilation settings")
