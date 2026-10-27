# Copyright (c) 2026-2027 zh
"""
内容：
    定义 Unitree Go2 的环境、命令、控制、初始状态、重置和训练默认参数。

    配置采用嵌套类组织，既支持直接读取类属性，也支持通过 BaseConfig
    实例化后独立修改可变参数。机器人模型和执行器装配由
    `src/assets/robots/go2_constants.py` 负责。
"""

import math
from copy import deepcopy

from mjlab.terrains.config import ROUGH_TERRAINS_CFG

from src.config.base.base_config import BaseConfig


class Go2Cfg(BaseConfig):
    """Unitree Go2 基础运动控制配置。"""

    class env:
        """环境规模、观测维度和回合设置。"""

        num_envs = 1024
        num_actions = 12
        num_action = num_actions  # 兼容参考项目的旧字段名。

        # 单帧 actor 观测：角速度、投影重力、三维速度命令、步态相位、关节位置、
        # 关节速度和上一动作。
        num_obs_history = 1
        num_critic_obs_history = 1
        num_obs_one_step = 3 + 3 + 3 + 2 + num_actions * 3
        history_size = num_obs_history
        num_observations = num_obs_one_step
        num_obs = num_observations * num_obs_history

        # critic 观测在 actor 观测后追加基座线速度、高度扫描和足端状态。
        num_height_scan = 187
        num_critic_obs_flat = num_obs_one_step + 3 + 4 + 4 + 4 + 12
        num_critic_obs_rough = num_critic_obs_flat + num_height_scan
        # 兼容旧调用方：默认维度代表粗糙地形 critic；平地请使用 num_critic_obs_flat。
        num_privileged_obs = num_critic_obs_rough
        num_critic_obs = num_critic_obs_rough
        env_spacing = 3.0
        scene_extent = 2.0
        send_timeouts = True
        episode_length_s = 20.0
        seed = 42

    class sim:
        """仿真器的基础数值参数。"""

        nconmax = 35
        njmax = 1500
        timestep = 0.005
        iterations = 10
        ls_iterations = 20
        ccd_iterations = 500
        contact_sensor_maxmatch = 500

        class flat:
            """平地模式的接触计算预算；None 表示由框架分配接触容量。"""

            nconmax = None
            njmax = 300
            ccd_iterations = 50
            contact_sensor_maxmatch = 64

    class sensors:
        """地形扫描范围、距离和分辨率以米计；接触历史以控制步计。"""

        terrain_scan_size = (1.6, 1.0)
        terrain_scan_resolution = 0.1
        terrain_scan_max_distance = 5.0
        terrain_scan_debug_vis = True
        terrain_scan_show_normals = True
        contact_num_slots = 1
        nonfoot_history_length = 4

    class terrain:
        """保留 MjLab 1.6.0 粗糙地形预设；字典及子地形在实例间独立。"""

        # 保留各子地形的配置类型，避免 asdict 将其递归转换为普通字典。
        generator = {**deepcopy(vars(ROUGH_TERRAINS_CFG)), "curriculum": True}
        max_init_terrain_level = 5
        curriculum_promotion_fraction = 0.8
        curriculum_min_commanded_distance = 1.0  # 米；站立或极小位移命令不晋级。

    class viewer:
        """Go2 跟随相机；距离以米计，角度以度计。"""

        distance = 1.5
        elevation = -10.0
        azimuth = 90.0

    class play:
        """播放模式的回合、地形网格和速度命令覆盖。"""

        episode_length_s = int(1e9)
        terrain_num_cols = 5
        terrain_num_rows = 5
        terrain_border_width = 10.0
        lin_vel_x = (-0.5, 1.0)
        lin_vel_y = (-0.5, 0.5)
        ang_vel_yaw = (-0.5, 0.5)

    class commands:
        """速度命令采样和课程参数。"""

        curriculum = True
        curriculum_threshold = 0.8
        curriculum_increment = 0.25  # vx 每次向两侧扩展的速度，单位 m/s；0 表示冻结。
        curriculum_update_interval_s = None  # 默认每个最大回合时长评估一次。
        num_commands = 4  # 顺序为 x 方向线速度、y 方向线速度、偏航角速度和航向角。
        resampling_time = [3.0, 8.0]
        heading_command = True
        heading_control_stiffness = 0.5
        rel_standing_envs = 0.05
        debug_vis = True

        class initial_ranges:
            """启用课程时的 vx 起始范围，自动裁剪到 ranges 指定的目标范围内。"""

            lin_vel_x = [-0.5, 1.0]

        class ranges:
            """vx 的课程目标范围及其余命令的固定范围，单位为 m/s、rad/s 和 rad。"""

            lin_vel_x = [-1.0, 2.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-math.pi, math.pi]

    class control:
        """关节位置控制和动作延迟参数。"""

        control_type = "P"
        stiffness = {"hip": 20.0, "thigh": 20.0, "calf": 40.0}
        damping = {"hip": 0.5, "thigh": 0.5, "calf": 1.0}
        action_scale = 0.5
        decimation = 4
        hip_reduction = 0.5
        effort_limit = {"hip": 23.7, "thigh": 23.7, "calf": 45.43}
        armature = 0.01
        # 参考任务未将 friction 接入 frictionloss，保留该旧字段但不改变动力学。
        friction = 0.01
        soft_joint_pos_limit_factor = 0.9
        delay_min_lag = 0
        delay_max_lag = 4
        delay_hold_prob = 0.5
        delay_update_period = 10

    class collision:
        """足端接触材料与软约束；其余碰撞规则由资产模块装配。"""

        foot_condim = 3
        body_condim = 1
        foot_priority = 1
        foot_friction = (0.6,)
        foot_solimp = (0.9, 0.95, 0.023)

    class init_state:
        """机器人出生位姿和零动作对应的关节目标。"""

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
        """回合重置时施加的基座和关节扰动范围。"""

        base_offset = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {},
        }
        joint_offset = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }

    class domain_rand:
        """Go2 域随机化范围；新增项参考 RI-4438 HIM，并按 Go2 标称模型适配。"""

        foot_friction_range = [0.3, 1.6]
        encoder_bias_range = [-0.015, 0.015]  # rad；每环境固定，仅用于 actor。
        base_com_range = [-0.05, 0.05]  # m；基座各轴质心偏移。
        # 质量与惯量共用对数均匀倍率，通过伪惯量联合采样；基座与腿部分开。
        base_mass_range = [0.85, 1.15]
        link_mass_range = [0.9, 1.1]
        kp_range = [0.8, 1.2]  # 标称位置增益倍率。
        kd_range = [0.8, 1.2]  # 标称速度增益倍率。
        effort_limit_range = [0.8, 1.0]  # 标称力矩上限倍率。
        joint_armature_range = [0.8, 1.2]  # 标称转子等效惯量倍率。
        # Go2 标称被动摩擦和阻尼为零，使用绝对值，避免倍率缩放无效。
        joint_friction_range = [0.005, 0.015]  # N·m；库仑摩擦。
        joint_damping_range = [0.0, 0.05]  # N·m·s/rad；被动阻尼，独立于 PD kd。
        imu_gyro_bias_range = [-0.02, 0.02]  # rad/s；每环境固定。
        imu_mount_angle_range = [-math.radians(1.0), math.radians(1.0)]  # 各轴安装角误差。
        foot_contact_time_constant_range = [0.015, 0.03]  # s；不小于两倍物理步长。
        external_force_range = [-6.0, 6.0]  # N；世界坐标系各轴，作用于基座质心。
        external_force_duration_s = [0.1, 0.3]
        external_force_cooldown_s = [6.0, 10.0]
        push_interval_s = [5.0, 6.0]
        push_velocity_range = {
            "x": [-0.5, 0.5],
            "y": [-0.5, 0.5],
            "z": [-0.4, 0.4],
            "roll": [-0.52, 0.52],
            "pitch": [-0.52, 0.52],
            "yaw": [-0.78, 0.78],
        }

    class noise:
        """参考 Go2 PPO actor/critic 观测噪声参数。"""

        add_noise = True
        noise_level = 1.0

        class noise_scales:
            """对称均匀噪声的半幅；单位与对应观测相同。"""

            base_ang_vel = 0.2
            projected_gravity = 0.05
            joint_pos = 0.01
            joint_vel = 1.5
            base_lin_vel = 0.5

    class reward:
        """参考 Go2 PPO 速度任务的奖励权重和形状参数。"""

        class scales:
            """奖励项权重；实际奖励函数由任务模块注册。"""

            track_linear_velocity = 1.0
            track_angular_velocity = 1.0
            body_orientation_l2 = -1.0
            body_ang_vel = -0.05
            angular_momentum = -0.025
            is_terminated = -200.0
            joint_acc_l2 = -2.5e-7
            joint_pos_limits = -10.0
            action_rate_l2 = -0.05
            foot_gait = 0.5
            foot_clearance = -1.0
            foot_slip = -0.25
            soft_landing = -1.0e-3
            stand_still = -1.0

        tracking_linear_std = math.sqrt(0.25)
        tracking_angular_std = math.sqrt(0.5)
        command_threshold = 0.1

        # 相位观测和步态奖励共用这些参数，避免两处使用不同周期。
        gait_period = 0.6
        gait_offset = [0.0, 0.5, 0.5, 0.0]
        gait_stance_fraction = 0.56
        foot_clearance_height = 0.10
        illegal_contact_force_threshold = 10.0
        fall_angle = math.radians(70.0)


class GO2CfgPPO(BaseConfig):
    """Go2 基础 PPO 网络、优化器和运行器配置。"""

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"
        obs_normalization = True
        distribution_std_type = "scalar"

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        seed = 42
        clip_actions = None
        num_steps_per_env = 24
        max_iterations = 10001
        save_interval = 100
        experiment_name = "go2_ppo"
        logger = "tensorboard"


# 保留更符合 Python 命名习惯的导入名，同时不影响参考配置的旧接口。
Go2CfgPPO = GO2CfgPPO


def make_go2_ppo_cfg(ppo_cfg: Go2CfgPPO | None = None) -> dict:
    """生成本项目 ``agents/base`` 使用的 PPO 配置字典。"""
    cfg = ppo_cfg if ppo_cfg is not None else GO2CfgPPO()
    return {
        "num_steps_per_env": cfg.runner.num_steps_per_env,
        "save_interval": cfg.runner.save_interval,
        "obs_groups": {"actor": ["actor"], "critic": ["critic"]},
        "check_for_nan": True,
        "actor": {
            "class_name": "agents.base.models:MLPModel",
            "hidden_dims": list(cfg.policy.actor_hidden_dims),
            "activation": cfg.policy.activation,
            "obs_normalization": cfg.policy.obs_normalization,
            "distribution_cfg": {
                "class_name": "agents.base.modules:GaussianDistribution",
                "init_std": cfg.policy.init_noise_std,
                "std_type": cfg.policy.distribution_std_type,
            },
        },
        "critic": {
            "class_name": "agents.base.models:MLPModel",
            "hidden_dims": list(cfg.policy.critic_hidden_dims),
            "activation": cfg.policy.activation,
            "obs_normalization": cfg.policy.obs_normalization,
        },
        "algorithm": {
            "class_name": "agents.base.algorithms:PPO",
            "value_loss_coef": cfg.algorithm.value_loss_coef,
            "use_clipped_value_loss": cfg.algorithm.use_clipped_value_loss,
            "clip_param": cfg.algorithm.clip_param,
            "entropy_coef": cfg.algorithm.entropy_coef,
            "num_learning_epochs": cfg.algorithm.num_learning_epochs,
            "num_mini_batches": cfg.algorithm.num_mini_batches,
            "learning_rate": cfg.algorithm.learning_rate,
            "schedule": cfg.algorithm.schedule,
            "gamma": cfg.algorithm.gamma,
            "lam": cfg.algorithm.lam,
            "desired_kl": cfg.algorithm.desired_kl,
            "max_grad_norm": cfg.algorithm.max_grad_norm,
        },
    }
