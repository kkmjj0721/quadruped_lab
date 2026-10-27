# Copyright (c) 2026-2027 zh
"""内容：装配指定 go2-supermimic 参考 MJCF、直流电机与完整碰撞。"""

from pathlib import Path

import mujoco
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(Path(__file__).parent / "go2/xmls/go2_reference.xml"))


def get_reference_go2_robot_cfg(cfg) -> EntityCfg:
    """参数来自集中配置；与旧 PPO、CTS、MoE 资产装配相互独立。"""
    control = cfg.control
    actuators = tuple(DcMotorActuatorCfg(
        target_names_expr=(f".*_{name}_joint",), stiffness=control.stiffness[name], damping=control.damping[name],
        effort_limit=control.effort_limit[name], saturation_effort=2 * control.effort_limit[name],
        velocity_limit=control.velocity_limit[name], armature=control.armature[name],
        delay_min_lag=control.delay_min_lag, delay_max_lag=control.delay_max_lag,
        delay_hold_prob=control.delay_hold_prob, delay_update_period=control.delay_update_period,
    ) for name in ("hip", "thigh", "calf"))
    foot_pattern, collision_pattern = "^[FR][LR]_foot_collision$", r".*_collision\d*"
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=tuple(cfg.init_state.pos), joint_pos=dict(cfg.init_state.default_joint),
                                           joint_vel={".*": 0.0}),
        collisions=(CollisionCfg(
            geom_names_expr=(collision_pattern,), contype=1, conaffinity=1, solref=tuple(cfg.collision.solref),
            condim={foot_pattern: cfg.collision.foot_condim, collision_pattern: cfg.collision.body_condim},
            priority={foot_pattern: cfg.collision.foot_priority, ".*": 0},
            friction={foot_pattern: tuple(cfg.collision.foot_friction)},
        ),),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(actuators=actuators,
                                             soft_joint_pos_limit_factor=control.soft_joint_pos_limit_factor),
    )
