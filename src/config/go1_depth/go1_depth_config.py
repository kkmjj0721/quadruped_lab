# Copyright (c) 2026-2027 zh
"""
内容：
    集中定义 Go1 资产、静态深度相机及纯深度 SuperMimic 任务配置。
"""

from dataclasses import dataclass
import math

from mjlab.sensor import CameraSensorCfg

from agents.depth_supermimic.config import DepthSuperMimicConfig

from src.config.base.base_config import BaseConfig


class Go1DepthCfg(BaseConfig):
    """统一声明资产、相机与训练参数，各实例之间的可变数据相互独立。"""

    algorithm = "depth_rssm"

    @dataclass
    class algo(DepthSuperMimicConfig):
        """用户指定的大规模训练配方；完整八步监督和三十二步想象保持不变。"""

        wm_batch: int = 2048
        imag_batch: int = 2048
        archive_size: int = 100000
        learning_rate: float = 4e-4
        actor_lr_scale: float = 4.0
        actor_optimizer: str = "muon_adam"
        wm_optimizer: str = "muon_adam"
        warmup_iterations: int = 0
        collect_det_frac: float = 0.25
        compile_imagination: bool = True  # 首轮编译预热单独统计，完整32步反向不截断。

    class control:
        """保留现有位置执行器的增益、限幅、惯量和指令延迟。"""

        stiffness = {"hip": 40.0, "thigh": 40.0, "calf": 40.0}  # N·m/rad。
        damping = {"hip": 1.0, "thigh": 1.0, "calf": 1.0}  # N·m·s/rad。
        effort_limit = {"hip": 33.5, "thigh": 33.5, "calf": 33.5}  # N·m。
        armature = {"hip": 0.01, "thigh": 0.01, "calf": 0.01}  # kg·m²。
        soft_joint_pos_limit_factor = 0.9
        delay_min_lag = 0
        delay_max_lag = 4
        delay_hold_prob = 0.5
        delay_update_period = 10
        # 按用户要求对齐 Go2 SuperMimic 的分关节动作缩放。
        action_scales = {"hip": 0.25, "thigh": 0.5, "calf": 0.5}
        decimation = 4

    class collision:
        """保留足端接触材料、全身碰撞及自碰撞。"""

        foot_condim = 3
        body_condim = 1
        foot_priority = 1
        foot_friction = (0.6,)
        foot_solimp = (0.9, 0.95, 0.023)
        enable_self_collision = True

    class init_state:
        """机器人初始位姿；位置为米，关节角为弧度。"""

        pos = [0.0, 0.0, 0.35]
        default_joint = {
            "FL_hip_joint": 0.1,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "FR_hip_joint": -0.1,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,
            "RL_hip_joint": 0.1,
            "RL_thigh_joint": 1.0,
            "RL_calf_joint": -1.5,
            "RR_hip_joint": -0.1,
            "RR_thigh_joint": 1.0,
            "RR_calf_joint": -1.5,
        }

    class sensors:
        """相机名称与基础渲染参数；挂载位姿及视场由 XML 定义。"""

        class depth:
            camera_name = "depth_camera"
            width = 64
            height = 64
            data_types = ("depth",)
            enabled_geom_groups = (0, 3)
            use_textures = False
            use_shadows = False
            period = 5  # 每5个50Hz控制步拍摄一帧，重置后首帧立即拍摄。
            near = 0.0
            far = 5.0
            noise_std = 0.03  # 米制高斯噪声；固定量程归一化之前添加。
            dropout_probability = 0.1

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
        """声明步态相位观测与想象端接触来源。"""

        contact_source = "model"  # 想象端默认使用受监督接触概率；height可显式选择足高近似对照。
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
        """Go1跟随相机；距离以米计，角度以度计。"""

        distance = 1.5
        elevation = -10.0
        azimuth = 90.0

    class noise:
        """只扰动新生成的本体帧，历史帧不重复加噪。"""

        add_noise = True
        noise_level = 1.0

        class noise_scales:
            """对称均匀噪声的半幅，单位与对应观测相同。"""

            base_ang_vel = 0.2  # rad/s。
            projected_gravity = 0.05
            joint_pos = 0.01  # rad。
            joint_vel = 1.5  # rad/s。

    class terrain:
        """Ri-5036 HIM地形；难度采用Go1 HIM行走表现课程。"""

        mode = "rough"
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
        """按指定Go1参考的MjLab基类随机化，不加入外力push。"""

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
        body_ang_vel_frame = "body"  # 默认按HIM约束机身系xy角速度；heading可显式用于对照。
        tracking_linear_variance = 0.25
        tracking_angular_variance = 0.25
        pose_tracking_gate = True  # 移动时以实际速度跟踪门控姿态奖励；站立保留原评分。
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
            is_terminated = -100.0  # 失败终止奖励率；想象端使用软失败概率，正常超时不罚。
            action_rate_l2 = -0.1
            joint_pos_limits = -10.0
            body_ang_vel = -0.05
            foot_gait = 0.5
            feet_air_time = 0.2
            pose = 0.5
            lin_vel_z = -2.0
            body_orientation_l2 = -0.2
            smoothness = -0.01
            stand_still = -2.0
            stand_still_joint_vel_l2 = -0.1  # 仅在站立命令下约束关节速度，权重是待训练验证的初值。
            stuck = -1.0  # 沿用 WMP：有 x 轴指令但机身 x 轴速度过低时惩罚。
            foot_slip = -0.1
            joint_acc_l2 = -3.0e-7
            stumble = -0.1
            joint_torques_l2 = -2.0e-5
            leg_collision = -1.0
            angular_momentum = -0.025
            hip_pos = -1.0
            soft_landing = -3.0e-3

    class runner:
        """视觉执行配置仅声明当前实现支持的选项；预算可由入口覆盖。"""

        max_iterations = 20000
        save_interval = 10000
        log_interval = 1
        metric_interval = 10
        imag_microbatch = 4096
        critic_microbatch = 32768
        wm_compile = True  # 完整八步监督编译；正确性对照可显式关闭。
        fused_adam = True
        cuda_graph = True  # 三类更新固定地址捕获，有限RSSM准备、感知监督和相机采集在图外。
        wm_bf16 = True  # 仅MLP矩阵，物理状态/返回/CNN/GRU仍FP32。
        actor_bf16 = True
        critic_bf16 = True
        enable_control_updates = True
        standing_zero_noise = True


def make_go1_depth_camera_cfg(go1_cfg: Go1DepthCfg | None = None, *, entity_name: str = "robot") -> CameraSensorCfg:
    """装配已命名实体的深度传感器，沿用资产中的相机位姿和视场。"""
    cfg = go1_cfg if go1_cfg is not None else Go1DepthCfg()
    depth = cfg.sensors.depth
    return CameraSensorCfg(
        name="depth",
        camera_name=f"{entity_name}/{depth.camera_name}",
        width=depth.width,
        height=depth.height,
        data_types=tuple(depth.data_types),
        enabled_geom_groups=tuple(depth.enabled_geom_groups),
        use_textures=depth.use_textures,
        use_shadows=depth.use_shadows,
    )
