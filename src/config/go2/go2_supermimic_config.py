# Copyright (c) 2026-2027 zh
"""
内容：
    定义Go2 SuperMimic＋CTS-MoE的任务、固定连通地形、课程和学习参数。
"""

import math

from src.config.base.base_config import BaseConfig


class Go2SuperMimicCfg(BaseConfig):
    """独立声明本任务参数；真实与想象共享同一份任务定义。"""

    # ---------- 环境与物理控制 ----------

    class env:
        """固定任务协议的维度、环境规模及回合预算。"""

        num_envs = 4096
        num_actions = 12
        scene_extent = 2.0
        seed = 7
        episode_length_s = 20.0
        num_obs_history = 5
        history_size = num_obs_history
        num_observations = 47
        num_obs = num_observations * num_obs_history
        num_privileged_obs = 298
        num_critic_obs = 567

    class sim:
        """平地和粗糙地形共用的仿真预算；物理步长以秒计。"""

        # 参考RI-4438 HIM的有限容量；接触池按环境数分配，约束上限逐环境生效。
        nconmax = 48
        njmax = 300
        broadphase = "nxn"
        solver = "cg"
        timestep = 0.005
        iterations = 10
        ls_iterations = 20
        ccd_iterations = 50  # 凸碰撞迭代上限；EPA临时显存随该值线性增长。
        contact_sensor_maxmatch = 64

    class control:
        """关节位置执行器；动作是相对默认姿态的无量纲偏移。"""

        stiffness = {"hip": 20.0, "thigh": 20.0, "calf": 40.0}
        damping = {"hip": 0.5, "thigh": 0.5, "calf": 1.0}
        effort_limit = {"hip": 23.7, "thigh": 23.7, "calf": 45.43}  # N·m。
        armature = 0.01
        action_scale = 0.5  # rad；髋关节额外乘hip_reduction。
        hip_reduction = 0.5
        decimation = 4
        soft_joint_pos_limit_factor = 0.9
        # 首版只支持无延迟名义执行器，避免向确定性模型隐藏队列状态。
        delay_min_lag = 0
        delay_max_lag = 0
        delay_hold_prob = 0.0
        delay_update_period = 10

    class collision:
        """本任务的足端接触材料与软约束，碰撞过滤由机器人资产装配。"""

        foot_condim = 3
        body_condim = 1
        foot_priority = 1
        foot_friction = (0.6,)
        foot_solimp = (0.9, 0.95, 0.023)
        enable_self_collision = True

    class init_state:
        """相对出生点的位置以米计；零动作对应的关节角以弧度计。"""

        pos = [0.0, 0.0, 0.32]
        default_joint = {
            "FL_hip_joint": -0.1,
            "RL_hip_joint": -0.1,
            "FR_hip_joint": 0.1,
            "RR_hip_joint": 0.1,
            "FL_thigh_joint": 0.9,
            "RL_thigh_joint": 0.9,
            "FR_thigh_joint": 0.9,
            "RR_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RL_calf_joint": -1.8,
            "FR_calf_joint": -1.8,
            "RR_calf_joint": -1.8,
        }

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

    # ---------- 地形、命令与任务信号 ----------

    class terrain:
        """指定LPACRL项目的地形几何；课程由当前任务单独实现。"""

        mode = "rough"
        seed = 0
        patch_size_m = (8.0, 8.0)
        core_rows = 5
        surrounding_rings = 1
        outer_flat_width_m = 16.0
        max_init_level = 0
        curriculum = True
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
        scan_size_m = (1.6, 1.0)
        scan_resolution_m = 0.1

    class commands:
        """课程进度使用真实控制步／24，阶段范围在reset应用。"""

        reference_steps_per_iteration = 24
        initial_ranges = ((-0.5, 0.5), (-0.5, 0.5), (-1.0, 1.0))
        stages = [
            {"iteration": 20000, "ranges": ((-1.0, 1.0), (-1.0, 1.0), (-1.5, 1.5))},
            {"iteration": 50000, "ranges": ((-2.0, 2.0), (-1.0, 1.0), (-2.0, 2.0))},
        ]
        zero_probability_start = 0.0
        zero_probability_end = 0.1
        zero_probability_iterations = 1500
        resampling_time = (5.0, 5.0)
        terrain_limits = {"stairs": (1.0, 1.0, 1.5), "slope": (1.5, 1.0, 1.5),
                          "random_rough": (1.0, 1.0, 1.5), "flat": (2.0, 1.0, 2.0)}
        heading_command = False

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

    class reward:
        """迁入RI-4438 HIM实际启用的22项；原始率只在聚合末尾乘dt。"""

        tracking_linear_std = math.sqrt(0.25)
        tracking_angular_std = math.sqrt(0.25)
        command_threshold = 0.1
        gait_period = 0.6
        gait_offset = (0.0, 0.5, 0.5, 0.0)  # FL、FR、RL、RR。
        gait_stance_fraction = 0.56
        # 以下容差按hip、thigh、calf重复到四腿，单位为rad。
        pose_std_standing = (0.05, 0.10, 0.15)
        pose_std_walking = (0.25, 0.35, 0.35)
        pose_std_running = (0.30, 0.35, 0.40)
        pose_walking_threshold = 0.1
        pose_running_threshold = 1.0
        pose_terrain_radius = 0.35  # m。
        pose_terrain_height_std = (0.01, 0.08)  # m。
        pose_terrain_tolerance = (0.15, 0.30, 0.35)
        pose_scan_max_distance = 5.0  # m；与参考地形扫描的有效射线范围一致。
        collision_force_threshold = 1.0  # N；腿碰地与自碰采用同一参考阈值。
        stumble_ratio = 5.0
        fall_height = 0.12
        fall_angle = math.radians(70)
        state_error_scales = (0.05, 0.10, 0.5, 1.0, 0.10, 2.0)

        class scales:
            track_linear_velocity = 2.0
            track_angular_velocity = 1.0
            lin_vel_z = -2.0
            body_ang_vel = -0.05
            body_orientation_l2 = -0.2
            pose = 0.5
            angular_momentum = -0.025
            joint_acc_l2 = -2.5e-7
            action_rate_l2 = -0.015
            smoothness = -0.015
            joint_pos_limits = -10.0
            joint_torques_l2 = -2.0e-5
            hip_pos = -0.5
            foot_gait = 0.5
            feet_air_time = 0.2
            foot_slip = -0.1
            soft_landing = -2e-3
            stand_still = -0.5
            leg_collision = -1.0
            self_collision = -100.0
            stumble = -0.1
            is_terminated = -100.0

    # ---------- 网络与学习调度 ----------

    class policy:
        actor_hidden_dims = (512, 256, 128)
        critic_hidden_dims = (512, 256, 128)
        teacher_hidden_dims = (512, 256)
        student_hidden_dims = (512, 256, 256)
        normalize_observations = True
        action_limit = 1.0  # 无量纲；平滑限制最终动作，近零导数保持1。

    class model:
        hidden_dims = (512, 512, 512)
        output_scales = (0.3, 0.5, 2.0, 0.01, 0.02, 0.02)
        learning_rate = 2e-4
        max_grad_norm = 1.0
        termination_weight = 1.0
        # 10项传感器奖励率随后接四足腾空／接触时长；仅用于预测及损失归一化。
        auxiliary_scales = (1.0, 1e6, 1e3, 1.0, 1.0, 1.0, 100.0, 1.0, 4.0, 1.0,
                            0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6)
        auxiliary_loss_weight = 1.0

    class learner:
        learning_rate = 2e-4
        student_learning_rate = 1e-3
        gamma = 0.99
        lam = 0.95
        horizon = 32
        critic_epochs = 5
        critic_minibatches = 4
        student_epochs = 1
        student_minibatches = 4
        balance_weight = 0.01
        teacher_weight = 0.5
        max_grad_norm = 1.0
        actor_grad_clip = 0.1
        observation_noise = True
        actor_ramp_updates = 1000  # 控制更新次数；只渐增actor／教师优化器学习率。

    class runner:
        replay_capacity = 256
        initial_collection_steps = 100
        collection_steps = 8
        world_model_updates = 4
        model_warmup_updates = 200
        window_length = 8
        world_model_batch_size = 256
        imagination_batch_size = 256
        student_batch_size = 256
        teacher_fraction = 0.75
        action_noise_std = 0.1
        enable_control_updates = False
