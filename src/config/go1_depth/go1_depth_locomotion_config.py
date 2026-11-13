# Copyright (c) 2026-2027 zh
"""内容：声明Go1同步深度速度任务的仿真、观测、地形及奖励参数。"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Go1LocomotionConfig:
    """本项目视觉迁移配方；未启用资产文件中尚未执行的动力学域随机化。"""

    num_envs: int = 16
    seed: int = 7
    terrain: str = "vision"
    terrain_rows: int = 2
    terrain_resolution: float = 0.05
    episode_length_s: float = 20.0
    timestep: float = 0.005
    decimation: int = 4
    depth_size: int = 32
    depth_near: float = 0.1
    depth_far: float = 3.0
    depth_noise: bool = True
    depth_noise_std: float = 0.03
    depth_dropout: float = 0.1
    action_clip: float = 4.0
    command_forward: tuple[float, float] = (0.0, 0.8)
    command_lateral: tuple[float, float] = (-0.2, 0.2)
    command_yaw: tuple[float, float] = (-0.5, 0.5)
    standing_probability: float = 0.1
    observation_noise: bool = True
    minimum_height: float = 0.14
    maximum_tilt: float = 1.0
    terrain_margin: float = 1.5
    tracking_linear_weight: float = 1.0
    tracking_yaw_weight: float = 0.5
    orientation_weight: float = -0.2
    vertical_velocity_weight: float = -1.0
    angular_velocity_weight: float = -0.05
    action_rate_weight: float = -0.01
    torque_weight: float = -1e-5
    joint_limit_weight: float = -10.0
    failure_weight: float = -2.0

    def __post_init__(self):
        for name in ("num_envs", "terrain_rows", "decimation", "depth_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        if self.terrain not in ("flat", "vision") or self.depth_size < 8:
            raise ValueError("Terrain must be flat or vision and depth size must be >= 8")
        for name in ("episode_length_s", "timestep", "terrain_resolution", "depth_near", "depth_far", "action_clip", "minimum_height", "maximum_tilt", "terrain_margin"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.depth_near >= self.depth_far or self.terrain_margin >= 4.0 or self.maximum_tilt >= math.pi:
            raise ValueError("Invalid depth interval, terrain margin or tilt threshold")
        if not math.isfinite(self.depth_noise_std) or self.depth_noise_std < 0:
            raise ValueError("Depth noise standard deviation must be finite and nonnegative")
        for name in ("standing_probability", "depth_dropout"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be within [0,1]")
        for name in ("depth_noise", "observation_noise"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in ("command_forward", "command_lateral", "command_yaw"):
            pair = getattr(self, name)
            if len(pair) != 2 or not all(math.isfinite(x) for x in pair) or pair[0] > pair[1]:
                raise ValueError(f"{name} must be a finite ordered pair")
        for name, value in vars(self).items():
            if name.endswith("_weight") and (not math.isfinite(value) or (value <= 0 if name.startswith("tracking") else value > 0)):
                raise ValueError(f"Invalid reward weight: {name}")
