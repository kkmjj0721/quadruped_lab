# Copyright (c) 2026-2027 zh
"""
内容：
    定义 Go1 Depth 的标称控制、接触材料、初始姿态、域随机化和深度相机参数。
    控制与姿态参考 wmp_go1@65cf1afd 的 a1_amp_config.py；机械上限来自 go1_new_2.urdf。
    域随机化参考同版本 legged_gym/envs/base/legged_robot_config.py:128–158。
"""

from src.config.base.base_config import BaseConfig


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
