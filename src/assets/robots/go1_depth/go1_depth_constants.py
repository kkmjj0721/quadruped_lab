# Copyright (c) 2026-2027 zh
"""
内容：
    定义 Go1 Depth 的模型名称映射，按配置装配执行器、初始状态和碰撞规则。
    直接运行时使用配置中的初始姿态打开 MuJoCo 资产查看器。

用法：
    .venv/bin/python src/assets/robots/go1_depth/go1_depth_constants.py
"""

from pathlib import Path
import sys

# 直接执行文件时，从当前项目解析 src 包。
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from src import SRC_PATH
from src.config.go1_depth.go1_depth_config import Go1DepthCfg


GO1_DEPTH_XML: Path = SRC_PATH / "assets" / "robots" / "go1_depth" / "xmls" / "go1.xml"
ROOT_BODY_NAME = "base_link"
IMU_SITE_NAME = "imu"
DEPTH_CAMERA_NAME = "depth_camera"
DEPTH_CAMERA_HOUSING_GEOM_NAME = "base2_collision"

# 顺序与本模型的 MJCF 关节、足端定义一致；不能直接套用其他机器人任务的排列。
LEG_NAMES = ("FL", "FR", "RL", "RR")
JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in LEG_NAMES for joint in ("hip", "thigh", "calf"))
FOOT_SITE_NAMES = LEG_NAMES
FOOT_BODY_NAMES = tuple(f"{leg}_calf" for leg in LEG_NAMES)  # 足端惯量合并于小腿，没有独立足刚体。
FOOT_GEOM_NAMES = tuple(f"{leg}_foot_collision" for leg in LEG_NAMES)


def get_spec() -> mujoco.MjSpec:
    """保留 XML 几何与相机参数，仅把包围光心的相机壳分至独立渲染组。"""
    spec = mujoco.MjSpec.from_file(str(GO1_DEPTH_XML))
    # 光心位于该实心碰撞盒内部；深度相机排除第 4 组，其余机身仍保留自遮挡。
    # geom.group 仅控制渲染分组，不改变相机壳的物理碰撞位掩码与接触材料。
    spec.geom(DEPTH_CAMERA_HOUSING_GEOM_NAME).group = 4
    return spec


# ---------- 执行器与碰撞 ----------

def _make_actuator_cfg(control_cfg, joint_type: str) -> BuiltinPositionActuatorCfg:
    """按关节组创建位置执行器；未覆盖的被动动力学仍取 XML。"""
    return BuiltinPositionActuatorCfg(
        target_names_expr=(f".*_{joint_type}_joint",),
        stiffness=control_cfg.stiffness[joint_type],
        damping=control_cfg.damping[joint_type],
        effort_limit=control_cfg.effort_limit[joint_type],
        armature=control_cfg.armature[joint_type],
        delay_min_lag=control_cfg.delay_min_lag,
        delay_max_lag=control_cfg.delay_max_lag,
        delay_hold_prob=control_cfg.delay_hold_prob,
        delay_update_period=control_cfg.delay_update_period,
    )


def _make_collision_cfg(collision_cfg) -> CollisionCfg:
    """覆盖机身、相机盒、载荷及各腿碰撞，视觉网格不参与碰撞。"""
    foot_regex = "^[FR][LR]_foot_collision$"
    return CollisionCfg(
        geom_names_expr=(".*_collision",),
        condim={foot_regex: collision_cfg.foot_condim, ".*": collision_cfg.body_condim},
        priority={foot_regex: collision_cfg.foot_priority, ".*": 0},
        friction={foot_regex: tuple(collision_cfg.foot_friction)},
        solimp={foot_regex: tuple(collision_cfg.foot_solimp)},
        contype=1,
        conaffinity=int(collision_cfg.enable_self_collision),
    )


# ---------- 资产工厂 ----------

def get_go1_depth_robot_cfg(go1_cfg: Go1DepthCfg | None = None) -> EntityCfg:
    """从配置创建独立资产；多次调用不共享可变初始状态、执行器或碰撞配置。"""
    cfg = go1_cfg if go1_cfg is not None else Go1DepthCfg()
    return EntityCfg(
        sort_actuators=True,  # ctrl 与 JOINT_NAMES 一致，不采用关节组配置的添加顺序。
        init_state=EntityCfg.InitialStateCfg(
            pos=tuple(cfg.init_state.pos),
            joint_pos=dict(cfg.init_state.default_joint),
            joint_vel={".*": 0.0},
        ),
        collisions=(_make_collision_cfg(cfg.collision),),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(_make_actuator_cfg(cfg.control, name) for name in ("hip", "thigh", "calf")),
            soft_joint_pos_limit_factor=cfg.control.soft_joint_pos_limit_factor,
        ),
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer

    robot = get_go1_depth_robot_cfg().build()
    model = robot.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_forward(model, data)
    viewer.launch(model, data)
