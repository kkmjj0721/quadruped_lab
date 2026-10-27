# Copyright (c) 2026-2027 zh
"""内容：声明指定 Go2 参考模型、单 MLP 训练预算与论文表 A.6 速度任务。"""

import math

from agents.supermimic.single.config import SingleConfig
from src.config.base.base_config import BaseConfig


# 参考 go2-supermimic d9eee9b3 robot/go2.py；膝关节传动比额外乘1.5。
_HIP_INERTIA = 0.000111842 * 6.33 ** 2
_KNEE_INERTIA = 0.000111842 * (6.33 * 1.5) ** 2
_FREQUENCY = 10 * 2 * math.pi


class Go2ReferenceCfg(BaseConfig):
    """新训练默认；原论文没有本配置所需的 Go2 动作库先验。"""

    algorithm = "single_mlp"
    single = SingleConfig

    class env:
        """固定任务协议的维度、环境规模及回合预算。"""

        num_envs = 4096
        num_actions = 12
        scene_extent = 2.0
        seed = 7
        episode_length_s = 20.0
        num_obs_history = 5
        history_size = num_obs_history
        num_observations = 45
        num_obs = num_observations * num_obs_history
        num_privileged_obs = 179
        num_critic_obs = 179

    class sim:
        """平地和粗糙地形共用的仿真预算；物理步长以秒计。"""

        nconmax = None
        njmax = 300
        broadphase = "nxn"
        solver = "cg"
        timestep = 0.005
        iterations = 10
        ls_iterations = 20
        ccd_iterations = 50  # 凸碰撞迭代上限；EPA临时显存随该值线性增长。

    class reset:
        """回合重置的位姿与速度扰动；不改变真实课程分配的出生地块。"""

        base_offset = {
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0),
                           "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
        }
        joint_offset = {"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)}

    class viewer:
        """Go2跟随相机；距离以米计，角度以度计。"""

        distance = 1.5
        elevation = -10.0
        azimuth = 90.0

    class noise:
        """只扰动新生成的盲本体帧，历史帧不重复加噪。"""

        add_noise = True
        noise_level = 1.0

        class noise_scales:
            """对称均匀噪声的半幅，单位与对应观测相同。"""

            base_ang_vel = 0.2  # rad/s。
            projected_gravity = 0.05
            joint_pos = 0.01  # rad。
            joint_vel = 1.5  # rad/s。

    class runner:
        enable_control_updates = True

    class control:
        """参考直流电机与0～2物理步延迟，动作比例为0.25×力矩上限/Kp。"""

        stiffness = {"hip": _HIP_INERTIA * _FREQUENCY ** 2,
                     "thigh": _HIP_INERTIA * _FREQUENCY ** 2,
                     "calf": _KNEE_INERTIA * _FREQUENCY ** 2}
        damping = {"hip": 4 * _HIP_INERTIA * _FREQUENCY,
                   "thigh": 4 * _HIP_INERTIA * _FREQUENCY,
                   "calf": 4 * _KNEE_INERTIA * _FREQUENCY}
        effort_limit = {"hip": 23.7, "thigh": 23.7, "calf": 45.43}
        velocity_limit = {"hip": 30.1, "thigh": 30.1, "calf": 20.07}
        armature = {"hip": _HIP_INERTIA, "thigh": _HIP_INERTIA, "calf": _KNEE_INERTIA}
        action_scales = {"hip": 0.25 * 23.7 / (_HIP_INERTIA * _FREQUENCY ** 2),
                         "thigh": 0.25 * 23.7 / (_HIP_INERTIA * _FREQUENCY ** 2),
                         "calf": 0.25 * 45.43 / (_KNEE_INERTIA * _FREQUENCY ** 2)}
        decimation = 4
        soft_joint_pos_limit_factor = 0.9
        delay_min_lag = 0
        delay_max_lag = 2
        delay_hold_prob = 0.0
        delay_update_period = 1

    class collision:
        foot_condim = 6
        body_condim = 1
        foot_priority = 1
        foot_friction = (1.0, 5e-3, 5e-4)
        solref = (0.01, 1.0)
        enable_self_collision = True

    class init_state:
        pos = [0.0, 0.0, 0.29]
        default_joint = {
            **{f"{leg}_hip_joint": 0.1 if leg in ("FL", "RL") else -0.1 for leg in ("FL", "FR", "RL", "RR")},
            **{f"{leg}_thigh_joint": 0.9 for leg in ("FL", "FR", "RL", "RR")},
            **{f"{leg}_calf_joint": -1.8 for leg in ("FL", "FR", "RL", "RR")},
        }

    class terrain:
        """指定LPACRL项目的地形几何；课程由当前任务单独实现。"""

        mode = "flat"
        seed = 0
        patch_size_m = (8.0, 8.0)
        core_rows = 5
        surrounding_rings = 1
        outer_flat_width_m = 16.0
        max_init_level = 0
        curriculum = False
        boundary_margin_m = 1.1
        levels = {
            "random_rough": {f"l{i}": {"height_max_m": height}
                             for i, height in enumerate((0.02, 0.03, 0.045, 0.06))},
            **{f"stairs_{label}_outward_{direction}": {
                f"l{i}": {"step_height_m": height, "step_width": depth, "border_width": border - 1e-9}
                for i, height in enumerate((0.05, 0.08, 0.10, 0.12, 0.16))}
               for label, depth, border in (("018", 0.18, 0.95), ("021", 0.21, 1.07),
                                           ("025", 0.25, 1.00), ("028", 0.28, 1.07))
               for direction in ("up", "down")},
            "slope_outward_up": {f"l{i}": {"gradient": gradient}
                                 for i, gradient in enumerate((0.0, 0.4 / 3, 0.8 / 3, 0.4))},
            "slope_outward_down": {f"l{i}": {"gradient": gradient}
                                   for i, gradient in enumerate((0.0, 0.4 / 3, 0.8 / 3, 0.4))},
        }
        defaults = {
            "random_rough": {"noise_step": 0.005, "horizontal_scale": 0.1, "vertical_scale": 0.005,
                             "downsampled_scale": None, "border_width": 0.25},
            "stairs": {"platform_width": 2.5, "holes": False},
            "slope": {"platform_width": 2.0, "border_width": 0.25,
                      "horizontal_scale": 0.1, "vertical_scale": 0.001},
        }
        scan_size_m = (1.2, 0.72)
        scan_resolution_m = 0.12

    class domain_randomization:
        """摩擦取指定Go2参考；编码器和COM幅度取论文表A.3，无外力push。"""

        enabled = True
        friction_range = (0.3, 1.5)
        encoder_bias_range = (-0.01, 0.01)
        com_range = {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}

    class commands:
        """论文附录G的均匀速度命令支路；坐标为heading平移与world yaw。"""

        reference_steps_per_iteration = 1
        initial_ranges = ((-1.5, 2.0), (-1.32, 1.32), (-3.35, 3.35))
        stages = []
        zero_probability_start = 0.1
        zero_probability_end = 0.1
        zero_probability_iterations = 1
        resampling_time = (5.0, 5.0)
        terrain_limits = {name: (2.0, 1.32, 3.35) for name in ("flat", "stairs", "slope", "random_rough")}
        heading_command = False

    class reward:
        """表A.6的六项奖励，Go2倾倒阈值和软化宽度取参考实现。"""

        tracking_linear_variance = 0.5 ** 2
        tracking_angular_variance = 0.5
        fall_angle = math.radians(70)
        fall_softness_deg = 5.0

        class scales:
            track_linear_velocity = 2.0
            track_angular_velocity = 2.0
            action_rate_l2 = -0.3
            joint_pos_limits = -2.0
            torso_sideways_lean = -4.0
            body_ang_vel = -0.05
