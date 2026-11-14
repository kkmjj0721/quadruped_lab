# Copyright (c) 2026-2027 zh
"""
内容：
    定义 Go1 Depth 的机器人、仿真、相机和基础地形配置，参考 Go2 的嵌套配置组织。
    控制与初始姿态取自 wmp_go1@65cf1afd 的实际 Go1 URDF 训练配置；模型以项目 XML 为准。
    本模块不注册任务；MGDP 基准、视觉策略、奖励和训练配方待对应任务实现时确定。
"""

from copy import deepcopy

from mjlab.sensor import CameraSensorCfg

from src.config.base.base_config import BaseConfig
from src.terrain.vision import make_vision_terrain_cfg


class Go1DepthCfg(BaseConfig):
    """Go1 机载深度相机的基础接入参数；不是可直接训练的完整任务配置。"""

    class env:
        """环境规模和基础控制维度；视觉 actor／critic 总维度由后续任务声明。"""

        num_envs = 1024
        num_actions = 12
        env_spacing = 3.0
        scene_extent = 2.0
        episode_length_s = 20.0
        seed = 42

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
        """沿用 Go2 基础重置范围；任务接入后相对于所分配地块应用。"""

        base_offset = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0),
                           "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        }
        joint_offset = {"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)}

    class sensors:
        """传感器及后续任务的采样协议；不在配置层执行更新或预处理。"""

        contact_num_slots = 1
        nonfoot_history_length = 4

        class depth:
            """绑定现有相机；原始输出为米制深度，归一化与噪声基准尚未选定。"""

            camera_name = "depth_camera"
            width = 64
            height = 64
            data_types = ("depth",)
            # 碰撞几何与地形查询同源，保留机体及载荷遮挡；相机自身外壳由资产移到第 4 组。
            enabled_geom_groups = (0, 3)
            use_textures = False
            use_shadows = False
            # 以下由后续任务消费；CameraSensorCfg 不提供刷新、双帧或缩放实现。
            resized = (16, 16)  # 高、宽；MGDP 论文与 Stage1 共同的目标尺寸。
            history_length = 2  # 按采集时间从旧到新排列，重置不跨回合拼帧。
            update_interval = 5  # 控制步；50 Hz 控制对应 10 Hz 相机刷新。

        class height_scan:
            """仅训练侧的米制 yaw 系地图；x 主序、y 次序，高度为地表 z 减根部 z。"""

            measured_points_x = tuple(round(-0.4 + 0.1 * i, 2) for i in range(17))
            measured_points_y = tuple(round(-0.5 + 0.1 * i, 2) for i in range(11))

    class terrain:
        """复用现有四列视觉地形；具体子地形类型及可变值在实例之间独立。"""

        # 与 Go2 配置一样保留对象类型；不能用 asdict 递归抹掉子地形配置类。
        generator = deepcopy(vars(make_vision_terrain_cfg()))
        max_init_terrain_level = 0
        curriculum_promotion_fraction = 0.8
        curriculum_min_commanded_distance = 1.0  # 米；沿用 Go2 行走表现课程阈值。

    class noise:
        """仅本体观测噪声的候选初值，沿用 Go2；深度噪声待 MGDP 基准选定。"""

        add_noise = True
        noise_level = 1.0

        class noise_scales:
            """对称均匀噪声半幅；后续任务只对新生成的本体帧加噪。"""

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
