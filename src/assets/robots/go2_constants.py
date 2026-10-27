# Copyright (c) 2026-2027 zh
"""
内容：
    按 Go2 配置装配独立的 MJCF 资产、执行器、初始状态和碰撞规则。
"""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from src import SRC_PATH
from src.config.go2.go2_config import Go2Cfg


GO2_XML: Path = SRC_PATH / "assets" / "robots" / "go2" / "xmls" / "go2.xml"
assert GO2_XML.exists()


def get_spec() -> mujoco.MjSpec:
    """读取项目内的 Go2 MJCF。"""
    return mujoco.MjSpec.from_file(str(GO2_XML))


# ---------- 执行器与碰撞 ----------

def _make_actuator_cfg(control_cfg, joint_type: str) -> BuiltinPositionActuatorCfg:
    """按关节组创建位置执行器，保留参考模型的摩擦设置。"""
    return BuiltinPositionActuatorCfg(
        target_names_expr=(f".*{joint_type}_.*",),
        stiffness=control_cfg.stiffness[joint_type],
        damping=control_cfg.damping[joint_type],
        effort_limit=control_cfg.effort_limit[joint_type],
        armature=control_cfg.armature,
        delay_min_lag=control_cfg.delay_min_lag,
        delay_max_lag=control_cfg.delay_max_lag,
        delay_hold_prob=control_cfg.delay_hold_prob,
        delay_update_period=control_cfg.delay_update_period,
    )


def _make_collision_cfg(collision_cfg, *, feet_only: bool = False) -> CollisionCfg:
    """装配接触维度和材料；碰撞过滤沿用参考 Go2 规则。"""
    foot_regex = "^[FR][LR]_foot_collision$"
    if feet_only:
        return CollisionCfg(
            geom_names_expr=(foot_regex,),
            contype=0,
            conaffinity=1,
            condim=collision_cfg.foot_condim,
            priority=collision_cfg.foot_priority,
            friction=tuple(collision_cfg.foot_friction),
            solimp=tuple(collision_cfg.foot_solimp),
        )
    return CollisionCfg(
        geom_names_expr=(".*_collision",),
        condim={foot_regex: collision_cfg.foot_condim, ".*_collision": collision_cfg.body_condim},
        priority={foot_regex: collision_cfg.foot_priority, ".*": 0},
        friction={foot_regex: tuple(collision_cfg.foot_friction)},
        solimp={foot_regex: tuple(collision_cfg.foot_solimp)},
        contype=1,
        conaffinity=int(getattr(collision_cfg, "enable_self_collision", False)),
    )


# ---------- 资产工厂 ----------

def get_go2_robot_cfg(go2_cfg: Go2Cfg | None = None) -> EntityCfg:
    """从本次配置创建资产，返回值与输入及其他调用互相独立。"""
    cfg = go2_cfg if go2_cfg is not None else Go2Cfg()
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


# 保留旧资产常量的导入接口；工厂每次重新装配，不使用这些可变默认对象。
_DEFAULT_ROBOT_CFG = get_go2_robot_cfg()
INIT_STATE = _DEFAULT_ROBOT_CFG.init_state
FULL_COLLISION = _DEFAULT_ROBOT_CFG.collisions[0]
GO2_ARTICULATION = _DEFAULT_ROBOT_CFG.articulation
GO2_HIP, GO2_THIGH, GO2_CALF = GO2_ARTICULATION.actuators
FEET_ONLY_COLLISION = _make_collision_cfg(Go2Cfg().collision, feet_only=True)
