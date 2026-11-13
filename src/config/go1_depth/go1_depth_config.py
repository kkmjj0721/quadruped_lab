# Copyright (c) 2026-2027 zh
"""
内容：
    集中定义 Go1 Depth 的机器人、任务与训练配方配置。
    控制与姿态参考 wmp_go1@65cf1afd 的 a1_amp_config.py；机械上限来自 go1_new_2.urdf。
    域随机化参考同版本 legged_gym/envs/base/legged_robot_config.py:128–158。
"""

from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING

from src.config.base.base_config import BaseConfig

if TYPE_CHECKING:
    from agents.perceptive_supermimic.config import ExecutionConfig, PerceptiveConfig


# ---------- 机器人与原生相机 ----------


class Go1DepthCfg(BaseConfig):
    """机器人参数；相机安装位姿与模型惯量由 Go1 MJCF 定义。"""

    class control:
        """位置 PD；角度为 rad、力矩为 N·m，decimation 以物理步计。"""

        stiffness = {"hip": 40.0, "thigh": 40.0, "calf": 40.0}
        damping = {"hip": 1.0, "thigh": 1.0, "calf": 1.0}
        effort_limit = {"hip": 33.5, "thigh": 33.5, "calf": 33.5}
        armature = 0.0
        frictionloss = 0.0
        viscous_damping = 0.0
        action_scale = 0.25  # 目标角 = 默认关节角 + action_scale × 动作。
        decimation = 4
        soft_joint_pos_limit_factor = 0.9
        # URDF 的机械速度上限，rad/s；内置位置执行器不实施硬速度限幅。
        velocity_limit = 30.0

    class collision:
        """保留现有 MJCF 的接触设置；摩擦依次为滑动、扭转和滚动系数。"""

        enable_self_collision = True  # WMP 的 self_collisions=0 表示启用自碰撞。
        condim = 6
        priority = 1
        friction = (1.0, 0.005, 0.0001)
        solref = (0.02, 1.0)
        solimp = (0.9, 0.95, 0.001, 0.5, 2.0)

    class init_state:
        """WMP 默认出生姿态；位置为 m，四元数为 wxyz，关节角为 rad。"""

        pos = [0.0, 0.0, 0.35]
        rot = (1.0, 0.0, 0.0, 0.0)
        lin_vel = (0.0, 0.0, 0.0)
        ang_vel = (0.0, 0.0, 0.0)
        joint_vel = {".*": 0.0}
        default_joint = {
            **{f"{leg}_hip_joint": 0.1 if leg in ("FL", "RL") else -0.1 for leg in ("FL", "FR", "RL", "RR")},
            **{f"{leg}_thigh_joint": 0.8 if leg in ("FL", "FR") else 1.0 for leg in ("FL", "FR", "RL", "RR")},
            **{f"{leg}_calf_joint": -1.5 for leg in ("FL", "FR", "RL", "RR")},
        }
        # 独立保留 WMP 的站姿目标，供任务选择；资产默认使用上面的出生姿态。
        stand_joint = {
            **{f"{leg}_hip_joint": 0.0 for leg in ("FL", "FR", "RL", "RR")},
            **{f"{leg}_thigh_joint": 0.55 for leg in ("FL", "FR", "RL", "RR")},
            **{f"{leg}_calf_joint": -1.38 for leg in ("FL", "FR", "RL", "RR")},
        }

    class domain_rand:
        """沿用 WMP 的随机化开关与范围；仅声明配置，尚未接入 MjLab 执行逻辑。"""

        randomize_friction = True
        friction_range = [0.5, 2.0]  # 无量纲；机器人全部碰撞体的摩擦系数。
        randomize_restitution = True
        restitution_range = [0.0, 0.0]  # 无量纲；保留 WMP 的零恢复系数。

        randomize_base_mass = True
        added_mass_range = [0.0, 3.0]  # kg；基座附加质量，不是质量倍率。

        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]  # 除基座外各连杆的标称质量倍率。

        randomize_com_pos = True
        # m；基座质心沿各轴的加性偏移。
        com_x_pos_range = [-0.05, 0.05]
        com_y_pos_range = [-0.05, 0.05]
        com_z_pos_range = [-0.05, 0.05]

        push_robots = True
        push_interval_s = 15  # s；推扰间隔。
        min_push_interval_s = 15  # s；保留 WMP 字段，上游推扰调度未读取。
        max_push_vel_xy = 1.0  # m/s；重设基座 x、y 速度时，各轴对称均匀采样的半幅。

        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]  # 标称 PD 位置增益倍率。
        damping_multiplier_range = [0.8, 1.2]  # 标称 PD 速度增益倍率。

        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]  # PD 输出力矩倍率，不是力矩上限倍率。

    class depth:
        """原生相机输出尺寸与水平视场；安装外参沿用 XML，视场单位为度。"""

        width = 64
        height = 64
        horizontal_fov = 58.0
        use_textures = False
        use_shadows = False
        enabled_geom_groups = (0, 1, 2)  # 地形和视觉网格；模型碰撞体位于组 3。


# ---------- 同步深度速度任务 ----------


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


# ---------- MGDP 原生两阶段任务 ----------


@dataclass(frozen=True)
class Go1MGDPConfig:
    """MGDP f43139d 的原生时间、传感和噪声配置；阶段一 mix，阶段二 parkour。"""

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


# ---------- 训练配方与执行预算 ----------


IMAGINATION_HORIZON = 32


def make_algorithm_config(*, perception: bool = True) -> "PerceptiveConfig":
    """保持现有网络宽度，只按本地显存缩小回放、监督和想象批量。"""
    from agents.perceptive_supermimic.config import PerceptiveConfig

    return PerceptiveConfig(
        replay_capacity=128, image_capacity=160, warmup_steps=64, wm_only_iterations=0,
        wm_window=8, wm_batch=32, wm_updates=2, imag_batch=16, horizon=IMAGINATION_HORIZON,
        reconstruction_images=64, critic_epochs=2, critic_minibatches=2,
        action_clip=4.0, log_interval=50, save_interval=1000,
        reconstruction_weight=1.2 if perception else 1.0,
        depth_reconstruction_loss="mse" if perception else "l1",
        height_reconstruction_weight=1.0 if perception else 0.0,
        contrastive_weight=0.3 if perception else 0.0,
        contrastive_temperature=0.1, height_shape=(17, 11),
    )


def make_execution_config() -> "ExecutionConfig":
    """分块改变峰值显存，不改变逻辑批次和32步展开长度。"""
    from agents.perceptive_supermimic.config import ExecutionConfig

    return ExecutionConfig(wm_microbatch=16, vision_batch=16, imag_microbatch=8, critic_microbatch=128)


def make_mgdp_algorithm_config() -> "PerceptiveConfig":
    """两阶段共用的 MGDP 感知与 SuperMimic 配置；阶段只改变任务分布。"""
    return replace(make_algorithm_config(), perception_model="mgdp", latent_dim=64, action_clip=100.0)
