# Copyright (c) 2026-2027 zh
"""
内容：
    以现有 Go1 XML 装配 MGDP 原版标称关节姿态、显式力矩执行器和相机。
    参考 MGDP f43139d 的 random_dog_config_stage1.py、sensor_config.py，
    相机坐标转换来自 warp_sensor/warp_sensor/warp_manager.py:109–111。
"""

import math

import mujoco
from mjlab.actuator import BuiltinMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.utils.spec_config import CollisionCfg

from .go1_depth_constants import FOOT_NAMES, GO1_XML, JOINT_NAMES


LEG_CONTACT_NAMES = tuple(f"{leg}_{segment}_collision" for leg in FOOT_NAMES for segment in ("thigh", "calf"))
FOOT_CONTACT_NAMES = tuple(f"{leg}_foot_collision" for leg in FOOT_NAMES)
FEET_SENSOR_NAME = "feet_contact"
BASE_SENSOR_NAME = "base_contact"
LEG_SENSOR_NAME = "leg_contact"
CAMERA_NAME = "mgdp_depth"
CAMERA_SENSOR_NAME = "depth"
CAMERA_TRANSLATION = (0.34, 0.0, 0.07)
CAMERA_HORIZONTAL_FOV = 67.0
CAMERA_PITCH_DEGREES = 30.0
# Preview4实际导入后的props[1:4]是FL三段，URDF文本顺序并非导入刚体顺序。
RANDOMIZED_LINK_NAMES = ("FL_hip", "FL_thigh", "FL_calf")
# 官方 go1.urdf:132,174,202；速度限值由任务适配器在 PD 后、积分前执行获准的迁移。
JOINT_RANGES = {
    "hip": (-1.0471975512, 1.0471975512),
    "thigh": (-0.663225115758, 2.96705972839),
    "calf": (-2.72271363311, -0.837758040957),
}
JOINT_VELOCITY_LIMIT = 30.1

_cos = math.cos(math.radians(CAMERA_PITCH_DEGREES) / 2)
_sin = math.sin(math.radians(CAMERA_PITCH_DEGREES) / 2)
# 原 warp 光学 +Z 朝前、+Y 朝下，MuJoCo 为 -Z 朝前、+Y 朝上。
CAMERA_QUATERNION = ((_cos + _sin) / 2, (_cos - _sin) / 2, (_sin - _cos) / 2, -(_cos + _sin) / 2)


def get_go1_mgdp_spec() -> mujoco.MjSpec:
    """复用现有网格，设置官方关节行程与限矩并新增基座相机。"""
    spec = mujoco.MjSpec.from_file(str(GO1_XML))
    for name in JOINT_NAMES:
        spec.joint(name).actfrcrange = (-23.7, 23.7)
        spec.joint(name).range = JOINT_RANGES[name.split("_")[1]]
    spec.body("base_link").add_camera(
        name=CAMERA_NAME, pos=CAMERA_TRANSLATION, quat=CAMERA_QUATERNION,
        fovy=CAMERA_HORIZONTAL_FOV, resolution=(120, 120),
    )
    return spec


def get_go1_mgdp_robot_cfg() -> EntityCfg:
    """动作与关节均按 FL、FR、RL、RR 的 hip/thigh/calf 顺序。"""
    nominal = {
        **{f"{leg}_hip_joint": 0.0 for leg in FOOT_NAMES},
        **{f"{leg}_thigh_joint": 0.8 if leg in ("FL", "FR") else 1.0 for leg in FOOT_NAMES},
        **{f"{leg}_calf_joint": -1.5 for leg in FOOT_NAMES},
    }
    return EntityCfg(
        spec_fn=get_go1_mgdp_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.45), joint_pos=nominal, joint_vel={".*": 0.0}),
        collisions=(CollisionCfg(
            geom_names_expr=(".*_collision",), contype=1, conaffinity=0, condim=6, priority=1,
            friction=(1.0, 0.005, 0.0001), solref=(0.02, 1.0), solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
        ),),
        sort_actuators=True,
        articulation=EntityArticulationInfoCfg(
            actuators=(BuiltinMotorActuatorCfg(
                target_names_expr=JOINT_NAMES, effort_limit=23.7, armature=0.0,
                frictionloss=0.0, viscous_damping=0.0,
            ),),
            soft_joint_pos_limit_factor=0.9,
        ),
    )


def get_go1_mgdp_camera_cfg() -> CameraSensorCfg:
    """raw 120²、67°视场；只显示地形组 0，输出米制光轴深度。"""
    return CameraSensorCfg(
        name=CAMERA_SENSOR_NAME, camera_name=f"robot/{CAMERA_NAME}",
        width=120, height=120, fovy=CAMERA_HORIZONTAL_FOV, data_types=("depth",),
        enabled_geom_groups=(0,), use_textures=False, use_shadows=False,
    )
