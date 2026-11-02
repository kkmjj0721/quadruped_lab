# Copyright (c) 2026-2027 zh
"""
内容：
    按参考 Go2 配置装配独立的 MJCF 资产、执行器、初始状态和碰撞规则。
"""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from src import SRC_PATH


GO2_XML: Path = SRC_PATH / "assets" / "robots" / "go2" / "xmls" / "go2_reference.xml"
assert GO2_XML.exists()


def get_spec() -> mujoco.MjSpec:
    """读取项目内的参考 Go2 MJCF。"""
    return mujoco.MjSpec.from_file(str(GO2_XML))


# ---------- 执行器与碰撞 ----------

def _make_actuator_cfg(control_cfg, joint_type: str) -> BuiltinPositionActuatorCfg:
    """按关节组创建参考模型的位置执行器。"""
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
    """装配参考模型的完整碰撞，保留自碰撞和带数字后缀的几何体。"""
    foot_regex = "^[FR][LR]_foot_collision$"
    collision_regex = r".*_collision\d*"
    return CollisionCfg(
        geom_names_expr=(collision_regex,),
        condim={foot_regex: collision_cfg.foot_condim, collision_regex: collision_cfg.body_condim},
        priority={foot_regex: collision_cfg.foot_priority, ".*": 0},
        friction={foot_regex: tuple(collision_cfg.foot_friction)},
        solref=tuple(collision_cfg.solref),
        contype=1,
        conaffinity=1,
    )


# ---------- 资产工厂 ----------


def get_reference_go2_robot_cfg(cfg) -> EntityCfg:
    """从本次配置创建参考资产，返回值与输入及其他调用互相独立。"""
    return EntityCfg(
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
