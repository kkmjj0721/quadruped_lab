# Copyright (c) 2026-2027 zh
"""
内容：
    定义 Go1 Depth 视觉任务统一使用的机器人、仿真、相机、地形、奖励和训练配置。
    控制与初始姿态取自 wmp_go1@65cf1afd 的实际 Go1 URDF 训练配置；模型以项目 XML 为准。
    地形、命令、奖励与训练参数沿用现有任务值，在本类中直接声明。
"""

from dataclasses import dataclass
import math

from mjlab.sensor import CameraSensorCfg

from agents.supermimic.config import SuperMimicConfig
from src.config.base.base_config import BaseConfig


class Go1DepthCfg(BaseConfig):
    """Go1 深度视觉任务、机器人资产与相机预览共用的完整配置。"""

    algorithm = "visual_foothold"

    @dataclass
    class algo(SuperMimicConfig):
        """本机器人的实际训练配方；驱动策略参数在runner.driver中维护。"""

        # ---------- 采集与回放 ----------
        replay_capacity: int = 1000
        collect_steps: float = 2.5  # 每轮平均真实控制步，2／3步交替。
        wm_prioritized_frac: float = 0.25
        wm_archive_frac: float = 0.25
        archive_size: int = 100000
        archive_keep_prob: float = 0.02

        # ---------- 世界模型与想象 ----------
        wm_hidden: tuple[int, ...] = (1024, 1024, 1024)
        wm_window: int = 8
        wm_batch: int = 2048
        wm_updates: int = 5
        imag_batch: int = 2048
        horizon: int = 32
        gamma: float = 0.99
        lam: float = 0.95

        # ---------- MLP 策略与价值网络 ----------
        # 历史估计器固定为 256／128／64 的 MLP，输出 3 维速度与 32 维归一化 latent。
        actor_hidden: tuple[int, ...] = (512, 256, 128)
        critic_hidden: tuple[int, ...] = (512, 256, 128)
        velocity_loss_coef: float = 1.0  # 真实同一时刻机身线速度的MSE监督权重。

        # ---------- 优化与排期 ----------
        learning_rate: float = 4e-4
        actor_lr_scale: float = 4.0
        actor_optimizer: str = "muon_adam"
        wm_optimizer: str = "muon_adam"
        muon_weight_decay: float = 0.1
        warmup_iterations: int = 0
        wm_rewarmup_iterations: int = 1500
        critic_warmup_iterations: int = 500
        actor_ramp_iterations: int = 1000
        lr_decay_iterations: int = 30000
        lr_final_frac: float = 0.03
        critic_epochs: int = 5
        critic_minibatches: int = 4
        grad_clip: float = 1.0
        grad_clip_element: float = 0.1

        # ---------- 探索噪声 ----------
        action_noise: float = 0.3  # 采用本项目消融配方；标准差为动作缩放前单位，论文表A.5默认0.1。
        collect_det_frac: float = 0.25  # 各地形按比例分配无额外动作探索噪声的环境。

        # ---------- 编译与回滚 ----------
        compile_imagination: bool = True  # 编译可微张量单步与返回递推；异步视觉刷新仍在图外。
        compile_backend: str = "inductor"
        collapse_falls: float = 3.0
        collapse_interval: int = 50

    class env:
        """环境规模、视觉任务观测维度及回合预算。"""

        num_envs = 1024
        num_actions = 12
        env_spacing = 3.0
        scene_extent = 2.0
        episode_length_s = 20.0
        seed = 42
        num_obs_history = 5
        history_size = 5
        num_observations = 45
        num_obs = 225
        # 视觉 critic 的实际尺寸由任务字段计算；不以该字段选择物理状态布局。
        num_privileged_obs = 1330
        num_critic_obs = 1330

    class experiment:
        contact_source = "model"
        actor_phase = False

    class sim:
        """物理步长以秒计；求解器预算沿用本项目 Go2 SuperMimic。"""

        timestep = 0.005
        nconmax = None
        njmax = 300
        broadphase = "nxn"
        solver = "cg"
        iterations = 10
        ls_iterations = 20
        ccd_iterations = 50

    class control:
        """位置控制初值；借鉴 wmp_go1，不代表完整复现其执行器随机化。"""

        control_type = "P"
        stiffness = {"hip": 40.0, "thigh": 40.0, "calf": 40.0}  # N·m/rad。
        damping = {"hip": 1.0, "thigh": 1.0, "calf": 1.0}  # N·m·s/rad。
        # 现有 XML 和源 go1_new_2.urdf 的全部关节限幅；不套用旧参考 XML 的电机值。
        effort_limit = {"hip": 33.5, "thigh": 33.5, "calf": 33.5}  # N·m。
        armature = {"hip": 0.0, "thigh": 0.0, "calf": 0.0}  # kg·m²。
        # 目标角 = 默认角 + action_scales[关节组] * 动作；无额外髋关节缩放。
        action_scales = {"hip": 0.25, "thigh": 0.25, "calf": 0.25}  # rad。
        decimation = 4  # 控制周期 0.02 s，50 Hz。
        soft_joint_pos_limit_factor = 0.9
        # 此处关闭延迟；参考中的物理子步延迟不等同于 MjLab 历史队列。
        delay_min_lag = 0
        delay_max_lag = 0
        delay_hold_prob = 0.0
        delay_update_period = 1

    class collision:
        """MuJoCo 接触材料沿用本项目 Go2 初值，保留 Go1 全身碰撞及自碰撞。"""

        foot_condim = 3
        body_condim = 1
        foot_priority = 1
        foot_friction = (0.6,)
        foot_solimp = (0.9, 0.95, 0.023)
        enable_self_collision = True

    class init_state:
        """wmp_go1 的 Go1 出生位姿；米、弧度，前后腿采用不同默认角。"""

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

    class reset:
        """相对于所分配地块应用出生位姿及关节重置范围。"""

        base_offset = {
            "pose_range": {"x": (-0.5, 0.5),
                           "y": (-0.5, 0.5),
                           "z": (0.0, 0.0),
                           "roll": (0.0, 0.0),
                           "pitch": (0.0, 0.0),
                           "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        }
        joint_offset = {"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)}

    class sensors:
        """传感器采样协议；不在配置层执行更新或预处理。"""

        contact_num_slots = 1
        nonfoot_history_length = 4

        class depth:
            """绑定资产相机，使用 MGDP Stage1 的米制噪声、固定量程与双帧协议。"""

            camera_name = "depth_camera"
            width = 64
            height = 64
            data_types = ("depth",)
            # 碰撞几何与地形查询同源，保留机体及载荷遮挡；相机自身外壳由资产移到第 4 组。
            enabled_geom_groups = (0, 3)
            use_textures = False
            use_shadows = False
            # 以下由视觉任务消费；CameraSensorCfg 不提供刷新、双帧或缩放实现。
            resized = (16, 16)  # 高、宽；MGDP 论文与 Stage1 共同的目标尺寸。
            history_length = 2  # 按采集时间从旧到新排列，重置不跨回合拼帧。
            update_interval = 5  # 控制步；50 Hz 控制对应 10 Hz 相机刷新。
            noise_gaussian = 0.03
            noise_dropout = 0.1
            near_clip = 0.0
            far_clip = 5.0

        class height_scan:
            """仅训练侧的米制 yaw 系地图；x 主序、y 次序，高度为地表 z 减根部 z。"""

            measured_points_x = tuple(round(-0.4 + 0.1 * i, 2) for i in range(17))
            measured_points_y = tuple(round(-0.5 + 0.1 * i, 2) for i in range(11))

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
        """保留 Go2 的22项配方，增加离地锁存目标的首次触地跟踪。"""

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

        foothold_tracking_sigma = 0.1  # 球面到锁存目标的距离核尺度，米；沿当前地图采样量级。

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
            foot_slip = -0.1
            joint_acc_l2 = -3.0e-7
            stumble = -0.1
            joint_torques_l2 = -2.0e-5
            leg_collision = -1.0
            angular_momentum = -0.025
            hip_pos = -1.0
            soft_landing = -3.0e-3
            track_foothold = 1.0  # 四足平均后统一乘控制周期；设为0可作原配方对照。

    class domain_randomization:
        """启动时随机标定与动力学，控制步施加外扰；范围为工程初值。"""

        enabled = True
        friction_range = (0.3, 1.5)
        spin_friction_range = (1e-4, 2e-2)
        roll_friction_range = (1e-5, 5e-3)
        encoder_bias_range = (-0.015, 0.015)
        com_range = {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "z": (-0.03, 0.03)}

        pd_gains_enabled = True
        kp_scale_range = (0.8, 1.2)  # wmp_go1的PD倍率；相对名义40 N·m/rad。
        kd_scale_range = (0.8, 1.2)  # 相对名义1 N·m·s/rad。
        motor_strength_enabled = True
        motor_strength_range = (0.8, 1.2)  # 乘随机PD的饱和前输出，与限幅分别采样。
        effort_limit_enabled = True
        effort_limit_scale_range = (0.8, 1.0)  # 不高于名义33.5 N·m。
        armature_enabled = True
        armature_range = (0.0, 0.01)  # 绝对值，kg·m²；名义为0，不能用乘法随机化。
        joint_friction_enabled = True
        joint_friction_range = (0.0, 0.1)  # 绝对被动库仑摩擦，N·m。
        joint_damping_enabled = True
        joint_damping_range = (0.0, 0.05)  # 绝对被动粘性阻尼，N·m·s/rad。

        class camera:
            """相对XML挂载体采样光学位姿；时间单位为50 Hz控制步。"""

            enabled = True
            position_range = (-0.01, 0.01)  # 各轴平移，米；借鉴wmp_go1。
            rotation_range = {"roll": (-math.radians(2), math.radians(2)),
                              "pitch": (-math.radians(5), math.radians(5)),
                              "yaw": (-math.radians(2), math.radians(2))}
            fovy_range = (56.0, 60.0)  # 度；先推导基础焦距，再施加残余标定误差。
            focal_scale_range = (0.95, 1.05)
            principal_offset_range = (-1 / 64, 1 / 64)  # MuJoCo归一化主点偏移，原始图±1像素。
            update_interval_range = (4, 6)
            frame_drop_probability = 0.05
            latency_range = (0, 3)  # 每帧独立通信延迟0～60ms；首帧即时启动。

        class push:
            """在基座COM施加持续世界系外力与力矩，触发间隔和持续时间独立采样。"""

            enabled = True
            force_range = {"x": (-20.0, 20.0), "y": (-20.0, 20.0), "z": (0.0, 0.0)}  # N。
            torque_range = {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "z": (-2.0, 2.0)}  # N·m。
            duration_s = (0.1, 0.3)
            interval_s = (8.0, 12.0)

        class impulse:
            """冲量除以控制周期转成单周期wrench，与持续外力相加。"""

            enabled = True
            linear_range = {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.0, 0.0)}  # N·s。
            angular_range = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)}  # N·m·s。
            interval_s = (10.0, 15.0)

    class runner:
        """本机器人的训练目标、保存节奏与执行预算；命令行可显式覆盖。"""

        max_iterations = 20000
        save_interval = 10000
        log_interval = 1
        metric_interval = 10  # 设备端汇总展示指标，每10轮回读并写TensorBoard。
        imag_microbatch = 4096
        critic_microbatch = 32768
        wm_compile = True  # 完整8步自回归监督编译；默认BF16网络保留FP32物理及损失。
        fused_adam = False
        cuda_graph = False
        detailed_timing = False  # 可显式开启图外采样／更新细分，原有阶段计时继续保留。
        timing_interval = 1  # 正整数；图准备及首次阶段强制采样，其余按此轮数间隔记录。
        replay_history_storage = "frames"  # 默认压缩带噪历史；full保留完整存储作显式对照。
        wm_bf16 = True  # 世界模型MLP默认使用CUDA BF16，监督与想象一致。
        actor_bf16 = True  # 历史估计器与动作 MLP 默认使用 CUDA BF16。
        critic_bf16 = True  # 价值MLP默认使用CUDA BF16，输出与返回递推仍为FP32。
        enable_control_updates = True
        standing_zero_noise = True  # 站立命令采集关闭额外动作噪声，保留固定无噪声组。

        perception_bf16 = True
        planner_bf16 = True
        perception_batch = 8
        planner_batch = 32
        perception_steps = 16
        render_batch_size = 64  # 独立预测渲染按批处理，避免为整个回放分配渲染世界。

        class driver:
            """外部策略在真实采集中代替当前策略行动，只贡献数据而不参与更新。"""

            policy = ""
            expert_policy = ""
            # 与参考 driver_frac=0.25 一致；未设置 policy 时该占比不生效。
            frac = 0.25
            expert_frac = 0.0
            fade_iterations = (1_000_000_000, 1_000_000_000)

    class visual:
        """任务侧接触与相机协议；单位为米或控制步。"""

        contact_threshold = 0.5

    class noise:
        """本体观测噪声沿用 Go2，深度噪声在 sensors.depth 中声明。"""

        add_noise = True
        noise_level = 1.0

        class noise_scales:
            """对称均匀噪声半幅；任务只对新生成的本体帧加噪。"""

            base_ang_vel = 0.2  # rad/s。
            projected_gravity = 0.05
            joint_pos = 0.01  # rad。
            joint_vel = 1.5  # rad/s。

    class viewer:
        """沿用 Go2 跟随视角；距离为米，角度为度，不改变机载深度相机。"""

        distance = 1.5
        elevation = -10.0
        azimuth = 90.0


def make_go1_depth_camera_cfg(go1_cfg: Go1DepthCfg | None = None, *, entity_name: str = "robot") -> CameraSensorCfg:
    """装配已命名实体的深度传感器；挂载位姿和 VFOV 直接沿用 XML。"""
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
