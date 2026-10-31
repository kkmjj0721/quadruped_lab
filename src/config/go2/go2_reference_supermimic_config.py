# Copyright (c) 2026-2027 zh
"""内容：声明指定Go2参考模型、混合专家策略、HIM命令及二十一项速度奖励。"""

import math

from agents.supermimic.config import SuperMimicConfig
from src.config.base.base_config import BaseConfig


class Go2ReferenceCfg(BaseConfig):
    """新训练默认；原论文没有本配置所需的 Go2 动作库先验。"""

    algorithm = "moe_history"
    algo = SuperMimicConfig

    class env:
        """固定任务协议的维度、环境规模及回合预算。"""

        num_envs = 4096
        num_actions = 12
        scene_extent = 2.0
        seed = 0
        episode_length_s = 20.0
        num_obs_history = 5
        history_size = num_obs_history
        num_observations = 47
        num_obs = num_observations * num_obs_history
        num_privileged_obs = 262
        num_critic_obs = 262

    class experiment:
        """步态协议；段名沿用已验证策略快照，旧配置在读取边界恢复原开关。"""

        contact_source = "model"  # 想象端使用受监督接触概率；height仅用于旧快照。
        actor_phase = True  # 原45维末尾追加sin／cos，相位与步态奖励共用时钟。

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
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.01, 0.05),
                           "yaw": (-3.14, 3.14)},
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

        class driver:
            """外部策略在真实采集中代替当前策略行动，只贡献数据而不参与更新。"""

            policy = ""
            expert_policy = ""
            # 与参考 driver_frac=0.25 一致；未设置 policy 时该占比不生效。
            frac = 0.25
            expert_frac = 0.0
            fade_iterations = (1_000_000_000, 1_000_000_000)
            speed_min = 0.5
            speed_max = 3.0

    class control:
        """内置位置执行器对齐 legged_wbc_mjlab Go2；动作比例按关节类型显式设置。"""

        stiffness = {"hip": 20.0, "thigh": 20.0, "calf": 40.0}
        damping = {"hip": 0.5, "thigh": 0.5, "calf": 1.0}
        effort_limit = {"hip": 23.7, "thigh": 23.7, "calf": 45.43}
        # 保留旧快照字段；内置位置执行器不使用直流电机的速度限制。
        velocity_limit = {"hip": 30.1, "thigh": 30.1, "calf": 20.07}
        armature = {"hip": 0.01, "thigh": 0.01, "calf": 0.01}
        action_scales = {"hip": 0.25, "thigh": 0.5, "calf": 0.5}
        decimation = 4
        soft_joint_pos_limit_factor = 0.9
        delay_min_lag = 0
        delay_max_lag = 4
        delay_hold_prob = 0.5
        delay_update_period = 10

    class collision:
        foot_condim = 3
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
        """Ri-5036 HIM地形；难度采用Go2 HIM行走表现课程。"""

        mode = "flat"
        seed = None
        patch_size_m = (8.0, 8.0)
        core_rows = 6
        num_cols = 20  # 沿用参考配置；MjLab课程模式按十种子地形实际生成十列。
        outer_flat_width_m = 20.0
        max_init_level = 0
        curriculum = True
        curriculum_iterations = 12000  # 保留旧配置字段；行走表现课程不使用该值。
        field_resolution_m = 0.02
        field_half_m = 6.5
        field_cache_dir = "logs/heightfields"
        scan_size_m = (1.2, 0.72)
        scan_resolution_m = 0.12

    class domain_randomization:
        """按指定Go2参考的MjLab基类随机化，不加入外力push。"""

        enabled = True
        friction_range = (0.3, 1.5)
        spin_friction_range = (1e-4, 2e-2)
        roll_friction_range = (1e-5, 5e-3)
        encoder_bias_range = (-0.015, 0.015)
        com_range = {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "z": (-0.03, 0.03)}

    class commands:
        """HIM采样及heading反馈；新奖励评价body平移与world yaw。"""

        reference_steps_per_iteration = 1
        initial_ranges = ((-1.0, 1.0), (-1.0, 1.0), (-2.0, 2.0))
        stages = []
        zero_probability_start = 0.1
        zero_probability_end = 0.1
        zero_probability_iterations = 1
        resampling_time = (4.0, 12.0)
        heading_command = True
        heading_range = (-3.14, 3.14)
        heading_control_stiffness = 0.5
        rel_heading_envs = 1.0
        rel_standing_envs = 0.1
        rel_forward_envs = 0.05

    class reward:
        """机体系平移及指定HIM奖励；各项先返回奖励率。"""

        tracking_linear_frame = "body"
        body_ang_vel_frame = "body"  # 新训练按HIM约束机体系xy角速度；旧完整快照保留heading。
        tracking_linear_variance = 0.25
        tracking_angular_variance = 0.25
        fall_angle = math.radians(70)
        fall_softness_deg = 5.0
        stand_still_command_threshold = 0.1  # 命令平移模长加偏航绝对值不超过阈值时启用。
        foot_slip_command_threshold = 0.1  # 命令平移模长加偏航绝对值超过阈值时启用。
        stumble_ratio = 5.0  # 水平接触力超过竖直接触力绝对值的该倍数时判为绊脚。
        leg_collision_force_threshold = 1.0  # 任一腿部几何体对地接触力模严格超过该值时惩罚，N。
        hip_pos_command_threshold = 0.1  # 侧移及转向命令绝对值均不超过阈值时约束髋角。
        soft_landing_command_threshold = 0.1  # 命令平移模长加偏航绝对值超过阈值时约束落地力。

        class gait:
            """相位按来源足名配置；软接触只用于想象近似。"""

            period = 0.6
            offsets = {"FR": 0.0, "FL": 0.5, "RR": 0.5, "RL": 0.0}
            stance_fraction = 0.56
            command_threshold = 0.1
            contact_height = 0.035  # 指定参考train.py稀有窗口采用的足端中心触地阈值，米。
            contact_softness = 0.005  # 想象接触过渡宽度，米；属于本次软化适配。

        class posture:
            walking_threshold = 0.1
            running_threshold = 1.5
            std_standing = {"hip": 0.05, "thigh": 0.1, "calf": 0.15}
            std_walking = {"hip": 0.15, "thigh": 0.35, "calf": 0.5}
            std_running = {"hip": 0.15, "thigh": 0.35, "calf": 0.5}

        class scales:
            track_linear_velocity = 2.0
            track_angular_velocity = 2.0
            action_rate_l2 = -0.1
            joint_pos_limits = -10.0
            body_ang_vel = -0.05
            foot_gait = 0.5
            feet_air_time = 0.2
            pose = 1.0
            lin_vel_z = -2.0
            body_orientation_l2 = -0.2
            smoothness = -0.01
            stand_still = -2.0
            stand_still_joint_vel_l2 = -0.5  # 仅在站立命令下约束关节速度，权重是待训练验证的初值。
            foot_slip = -0.1
            joint_acc_l2 = -2.5e-7
            stumble = -0.1
            joint_torques_l2 = -2.0e-5
            leg_collision = -1.0
            angular_momentum = -0.025
            hip_pos = -1.0
            soft_landing = -2.0e-3
