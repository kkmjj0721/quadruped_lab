# Copyright (c) 2026-2027 zh
"""
内容：
    定义 Go1 Depth 机器人资产与静态深度相机配置。
"""

from mjlab.sensor import CameraSensorCfg

from src.config.base.base_config import BaseConfig


class Go1DepthCfg(BaseConfig):
    """集中配置资产装配参数，各实例之间的可变数据相互独立。"""

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
            width = 32
            height = 32
            data_types = ("depth",)
            enabled_geom_groups = (0, 3)
            use_textures = False
            use_shadows = False


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
