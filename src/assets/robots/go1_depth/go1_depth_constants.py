# Copyright (c) 2026-2027 zh
"""
内容：
    按集中配置装配 Go1 Depth 的 MJCF、位置执行器、碰撞及原生深度相机。
    直接运行时使用配置中的初始姿态打开 MuJoCo 资产查看器。

用法：
    .venv/bin/python src/assets/robots/go1_depth/go1_depth_constants.py
"""

from copy import deepcopy
from functools import partial
import math
from pathlib import Path
import sys

# 直接执行文件时，从当前项目解析 src 包。
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.utils.spec_config import CollisionCfg

from src import SRC_PATH
from src.config.go1_depth.go1_depth_config import Go1DepthCfg


GO1_XML: Path = SRC_PATH / "assets" / "robots" / "go1_depth" / "xmls" / "go1.xml"
FOOT_NAMES = ("FL", "FR", "RL", "RR")
JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in FOOT_NAMES for joint in ("hip", "thigh", "calf"))


def get_spec(go1_cfg: Go1DepthCfg | None = None) -> mujoco.MjSpec:
    """读取现有模型，并将关节合力上限与配置的执行器限矩保持一致。"""
    cfg = go1_cfg if go1_cfg is not None else Go1DepthCfg()
    spec = mujoco.MjSpec.from_file(str(GO1_XML))
    for name in JOINT_NAMES:
        limit = cfg.control.effort_limit[name.split("_")[1]]
        spec.joint(name).actfrcrange = (-limit, limit)
    return spec


# ---------- 执行器与碰撞 ----------

def _make_actuator_cfg(control_cfg, joint_type: str) -> BuiltinPositionActuatorCfg:
    """按关节组创建位置执行器，保留标称摩擦与被动阻尼。"""
    return BuiltinPositionActuatorCfg(
        target_names_expr=(f".*_{joint_type}_joint",),
        stiffness=control_cfg.stiffness[joint_type],
        damping=control_cfg.damping[joint_type],
        effort_limit=control_cfg.effort_limit[joint_type],
        armature=control_cfg.armature,
        frictionloss=control_cfg.frictionloss,
        viscous_damping=control_cfg.viscous_damping,
    )


def _make_collision_cfg(collision_cfg) -> CollisionCfg:
    """装配完整碰撞、自碰撞开关和接触材料。"""
    return CollisionCfg(
        geom_names_expr=(".*_collision",),
        contype=1,
        conaffinity=int(collision_cfg.enable_self_collision),
        condim=collision_cfg.condim,
        priority=collision_cfg.priority,
        friction=tuple(collision_cfg.friction),
        solref=tuple(collision_cfg.solref),
        solimp=tuple(collision_cfg.solimp),
    )


# ---------- 资产工厂 ----------

def get_go1_depth_robot_cfg(go1_cfg: Go1DepthCfg | None = None) -> EntityCfg:
    """创建独立机器人配置；关节和动作槽位均按 FL、FR、RL、RR 逐腿排列。"""
    cfg = deepcopy(go1_cfg) if go1_cfg is not None else Go1DepthCfg()
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=tuple(cfg.init_state.pos),
            rot=tuple(cfg.init_state.rot),
            lin_vel=tuple(cfg.init_state.lin_vel),
            ang_vel=tuple(cfg.init_state.ang_vel),
            joint_pos=dict(cfg.init_state.default_joint),
            joint_vel=dict(cfg.init_state.joint_vel),
        ),
        collisions=(_make_collision_cfg(cfg.collision),),
        spec_fn=partial(get_spec, cfg),
        sort_actuators=True,
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(_make_actuator_cfg(cfg.control, name) for name in ("hip", "thigh", "calf")),
            soft_joint_pos_limit_factor=cfg.control.soft_joint_pos_limit_factor,
        ),
    )


# ---------- 相机工厂 ----------

def get_go1_depth_camera_cfg(
    go1_cfg: Go1DepthCfg | None = None,
    *,
    entity_name: str = "robot",
    name: str = "depth",
) -> CameraSensorCfg:
    """包装 XML 相机，返回供场景加入 sensors 的配置；输出为米制光轴深度。"""
    cfg = go1_cfg if go1_cfg is not None else Go1DepthCfg()
    depth = cfg.depth
    if any(type(value) is not int or value <= 0 for value in (depth.width, depth.height)):
        raise ValueError("Camera width and height must be positive integers")
    if not 0.0 < depth.horizontal_fov < 180.0:
        raise ValueError("Camera horizontal field of view must be between 0 and 180 degrees")
    fovy = math.degrees(
        2 * math.atan(math.tan(math.radians(depth.horizontal_fov) / 2) * depth.height / depth.width)
    )
    return CameraSensorCfg(
        name=name,
        camera_name=f"{entity_name}/depth_camera",
        width=depth.width,
        height=depth.height,
        fovy=fovy,
        data_types=("depth",),
        use_textures=depth.use_textures,
        use_shadows=depth.use_shadows,
        enabled_geom_groups=tuple(depth.enabled_geom_groups),
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer

    robot = get_go1_depth_robot_cfg().build()
    model = robot.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_forward(model, data)
    viewer.launch(model, data)
