# Copyright (c) 2026-2027 zh
"""
内容：
    声明 MGDP f43139d 原生两阶段 Go1 任务的时间、传感和噪声配置。
    阶段一使用 mix，阶段二使用 parkour；训练算法配置由算法模块负责。
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Go1MGDPConfig:
    stage: int = 1
    num_envs: int = 20
    seed: int = 7
    physics_integrator: str = "implicit"
    timestep: float = 0.005
    decimation: int = 4
    episode_length_s: float = 20.0
    depth_size: int = 16
    depth_raw_size: int = 120
    depth_near: float = 0.0
    depth_far: float = 5.0
    depth_clip_near: float = 0.1
    depth_clip_far: float = 2.5
    depth_noise_std: float = 0.03
    depth_dropout: float = 0.1
    depth_noise: bool = True
    height_noise: bool = True
    observation_noise: bool = True
    action_clip: float = 100.0

    @property
    def family(self) -> str:
        return "mix" if self.stage == 1 else "parkour"

    def __post_init__(self):
        if type(self.stage) is not int or self.stage not in (1, 2):
            raise ValueError("stage must be 1 or 2.")
        if self.physics_integrator not in ("implicit", "implicitfast"):
            raise ValueError("physics_integrator must be implicit or implicitfast.")
        if type(self.num_envs) is not int or self.num_envs <= 0:
            raise ValueError("num_envs must be a positive integer.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        for name in ("depth_noise", "height_noise", "observation_noise"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean.")
        # 原生采集与动作延迟依赖这些离散常量；变更必须同步对应执行协议。
        for name, expected in (("decimation", 4), ("depth_size", 16), ("depth_raw_size", 120)):
            if type(getattr(self, name)) is not int or getattr(self, name) != expected:
                raise ValueError(f"Native MGDP requires {name}={expected}.")
        for name in ("timestep", "episode_length_s", "action_clip", "depth_far", "depth_clip_far"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("depth_near", "depth_clip_near", "depth_noise_std"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not self.depth_near <= self.depth_clip_near < self.depth_clip_far <= self.depth_far:
            raise ValueError("Depth clipping must lie inside the camera range.")
        if not math.isfinite(self.depth_dropout) or not 0 <= self.depth_dropout <= 1:
            raise ValueError("depth_dropout must be in [0, 1].")
        native_values = {
            "timestep": 0.005, "episode_length_s": 20.0, "action_clip": 100.0,
            "depth_near": 0.0, "depth_far": 5.0, "depth_clip_near": 0.1,
            "depth_clip_far": 2.5, "depth_noise_std": 0.03, "depth_dropout": 0.1,
        }
        for name, expected in native_values.items():
            if type(getattr(self, name)) is bool or getattr(self, name) != expected:
                raise ValueError(f"Native MGDP requires {name}={expected}.")
