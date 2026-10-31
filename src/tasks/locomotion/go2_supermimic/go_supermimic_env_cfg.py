# Copyright (c) 2026-2027 zh
"""
内容：
    装配Go2 SuperMimic基础物理环境，具体地形模式由config/env_cfgs.py选择。
"""

from copy import deepcopy
import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import SceneEntityCfg, TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from src.assets.robots.go2_reference import get_reference_go2_robot_cfg
from src.config.go2.go2_reference_supermimic_config import Go2ReferenceCfg

from .mdp.state_spec import FOOT_NAMES
from .mdp.terrains import TerrainMap


def validate_config(cfg: Go2ReferenceCfg) -> None:
    """在构造仿真前统一校验任务入口参数。"""
    if cfg.algorithm != "moe_history":
        raise ValueError("SuperMimic requires the moe_history algorithm")
    if not cfg.commands.heading_command or cfg.commands.rel_heading_envs != 1.0:
        raise ValueError("The HIM adaptation requires heading commands for all moving environments")
    if cfg.commands.stages:
        raise ValueError("Velocity curriculum is disabled for the HIM adaptation")
    if isinstance(cfg.env.num_envs, bool) or not isinstance(cfg.env.num_envs, int) or cfg.env.num_envs < 1:
        raise ValueError("Environment count must be a positive integer")
    if not math.isfinite(cfg.env.episode_length_s) or cfg.env.episode_length_s <= 0:
        raise ValueError("Episode duration must be positive and finite")
    if not math.isfinite(cfg.sim.timestep) or cfg.sim.timestep <= 0:
        raise ValueError("Simulation timestep must be positive and finite")
    if isinstance(cfg.control.decimation, bool) or not isinstance(cfg.control.decimation, int) or cfg.control.decimation < 1:
        raise ValueError("Control decimation must be a positive integer")
    if cfg.env.num_obs_history != 5:
        raise ValueError("The Go2 actor requires exactly five history frames")
    if cfg.experiment.contact_source not in ("height", "model"):
        raise ValueError("The Go2 contact source must be height or model")
    if not isinstance(cfg.experiment.actor_phase, bool):
        raise ValueError("The Go2 actor phase flag must be boolean")
    observation_dim = 47 if cfg.experiment.actor_phase else 45
    if (cfg.env.num_observations, cfg.env.num_obs) != (observation_dim, observation_dim * cfg.env.num_obs_history):
        raise ValueError("The Go2 observation dimensions must match the actor phase flag")
    critic_dimensions = (cfg.env.num_critic_obs, cfg.env.num_privileged_obs)
    if critic_dimensions not in ((234, 234), (258, 258), (259, 259), (262, 262)):
        raise ValueError("The Go2 state layout requires 262 critic observations or the legacy 234/258/259 layouts")
    if cfg.reward.tracking_linear_frame not in ("body", "heading"):
        raise ValueError("Linear velocity tracking frame must be body or heading")
    if cfg.reward.body_ang_vel_frame not in ("body", "heading"):
        raise ValueError("Body angular velocity penalty frame must be body or heading")
    for name in ("angular_momentum", "hip_pos", "soft_landing"):
        value = getattr(cfg.reward.scales, name)
        if not math.isfinite(value) or value > 0:
            raise ValueError(f"Reward weight must be finite and nonpositive: {name}")
        if critic_dimensions != (262, 262) and value != 0:
            raise ValueError("The new HIM rewards require a 262-observation task configuration")
    for name in ("stumble", "joint_torques_l2"):
        value = getattr(cfg.reward.scales, name)
        if not math.isfinite(value) or value > 0:
            raise ValueError(f"Reward weight must be finite and nonpositive: {name}")
        if critic_dimensions == (234, 234) and value != 0:
            raise ValueError("Force rewards require a 258/259/262-observation task configuration")
    collision_weight = cfg.reward.scales.leg_collision
    if not math.isfinite(collision_weight) or collision_weight > 0:
        raise ValueError("Leg collision reward weight must be finite and nonpositive")
    if critic_dimensions not in ((259, 259), (262, 262)) and collision_weight != 0:
        raise ValueError("Leg collision reward requires a 259/262-observation task configuration")
    if not math.isfinite(cfg.reward.leg_collision_force_threshold) or cfg.reward.leg_collision_force_threshold <= 0:
        raise ValueError("Leg collision force threshold must be positive and finite")
    if not math.isfinite(cfg.reward.stumble_ratio) or cfg.reward.stumble_ratio <= 0:
        raise ValueError("Stumble force ratio must be positive and finite")
    standing_velocity_weight = cfg.reward.scales.stand_still_joint_vel_l2
    if not math.isfinite(standing_velocity_weight) or standing_velocity_weight > 0:
        raise ValueError("Standing joint velocity weight must be finite and nonpositive")
    for name in ("stand_still_command_threshold", "foot_slip_command_threshold",
                 "hip_pos_command_threshold", "soft_landing_command_threshold"):
        value = getattr(cfg.reward, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Reward command threshold must be finite and nonnegative: {name}")
    if cfg.terrain.mode not in ("flat", "rough"):
        raise ValueError("Unsupported SuperMimic terrain mode")
    if cfg.reward.gait.period <= 0 or not 0 < cfg.reward.gait.stance_fraction < 1 or cfg.reward.gait.contact_softness <= 0:
        raise ValueError("Gait period, stance fraction and contact softness are invalid")
    if not all(math.isfinite(value) for value in (
        cfg.reward.gait.period, cfg.reward.gait.stance_fraction, cfg.reward.gait.contact_height,
        cfg.reward.gait.contact_softness, cfg.reward.gait.command_threshold,
    )) or set(cfg.reward.gait.offsets) != set(FOOT_NAMES):
        raise ValueError("Gait parameters must be finite and define every foot by name")
    command_ranges = (*cfg.commands.initial_ranges, cfg.commands.heading_range, cfg.commands.resampling_time)
    if len(cfg.commands.initial_ranges) != 3 or any(
        len(bounds) != 2 or not all(math.isfinite(value) for value in bounds) or bounds[0] > bounds[1]
        for bounds in command_ranges
    ):
        raise ValueError("Command ranges must be finite and ordered pairs")
    for value in (cfg.commands.rel_standing_envs, cfg.commands.rel_forward_envs):
        if not 0 <= value <= 1:
            raise ValueError("Command mode probabilities must be in [0,1]")
    if len(cfg.commands.resampling_time) != 2 or not 0 < cfg.commands.resampling_time[0] <= cfg.commands.resampling_time[1]:
        raise ValueError("Command resampling times must be positive and ordered")


def make_go2_supermimic_env_cfg(
    cfg: Go2ReferenceCfg | None = None, *, terrain_map: TerrainMap | None = None,
) -> ManagerBasedRlEnvCfg:
    """返回可直接构造MjLab环境的配置，不复用任何PPO任务内部实现。"""
    cfg = deepcopy(cfg) if cfg is not None else Go2ReferenceCfg()
    validate_config(cfg)
    terrain_map = terrain_map if terrain_map is not None else TerrainMap(cfg.terrain)

    # ---------- 场景 ----------

    feet = tuple(f"{leg}_foot_collision" for leg in FOOT_NAMES)
    sensors = (ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=feet, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True,
    ),)
    if cfg.env.num_critic_obs in (259, 262):
        sensors += (ContactSensorCfg(
            name="leg_ground_contact",
            primary=ContactMatch(
                mode="geom", pattern=r"^(FR|FL|RR|RL)_(hip|thigh|calf).*_collision\d*$", entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="none", num_slots=1, history_length=1,
        ),)
    scene = SceneCfg(
        entities={"robot": get_reference_go2_robot_cfg(cfg)},
        terrain=terrain_map.entity_cfg(),
        sensors=sensors,
        num_envs=cfg.env.num_envs,
        extent=cfg.env.scene_extent,
    )

    # ---------- 动作 ----------

    action_scales = {rf".*_{name}_joint": value for name, value in cfg.control.action_scales.items()}
    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            use_default_offset=True,
            scale=action_scales,
        ),
    }

    # ---------- 重置事件 ----------

    # 只使用MjLab环境层的通用重置原语；任务课程与记忆重置由adapter协调。
    events = {
        "reset_root": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params=deepcopy(cfg.reset.base_offset),
        ),
        "reset_joints": EventTermCfg(
            func=envs_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "position_range": tuple(cfg.reset.joint_offset["position_range"]),
                "velocity_range": tuple(cfg.reset.joint_offset["velocity_range"]),
            },
        ),
    }
    if cfg.domain_randomization.enabled:
        domain = cfg.domain_randomization
        for name, axis, distribution, bounds in (
            ("slide", 0, "uniform", domain.friction_range),
            ("spin", 1, "log_uniform", domain.spin_friction_range),
            ("roll", 2, "log_uniform", domain.roll_friction_range),
        ):
            events[f"foot_friction_{name}"] = EventTermCfg(func=envs_mdp.dr.geom_friction, mode="startup", params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=feet), "operation": "abs",
                "distribution": distribution, "ranges": tuple(bounds), "axes": [axis], "shared_random": True,
            })
        events.update({
            "encoder_bias": EventTermCfg(func=envs_mdp.dr.encoder_bias, mode="startup", params={
                "asset_cfg": SceneEntityCfg("robot"), "bias_range": tuple(domain.encoder_bias_range),
            }),
            "base_com": EventTermCfg(func=envs_mdp.dr.body_com_offset, mode="startup", params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("base",)), "operation": "add",
                "ranges": {axis: tuple(domain.com_range[name]) for axis, name in enumerate(("x", "y", "z"))},
            }),
        })

    # ---------- 终止与任务逻辑边界 ----------

    # 原生环境只判定超时；跌倒等失败由任务模型在reset前统一计算。
    terminations = {"time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True)}
    # 命令、奖励与课程使用本任务TaskModel和adapter，真实／想象共用同一套实现。
    # 保持这些manager为空，防止原生环境重复推进任务或使用另一套奖励。
    commands = {}
    rewards = {}
    curriculum = {}
    metrics = {}

    # ---------- 仿真与视图 ----------

    sim = SimulationCfg(
        nconmax=cfg.sim.nconmax,
        njmax=cfg.sim.njmax,
        broadphase=cfg.sim.broadphase,
        mujoco=MujocoCfg(
            timestep=cfg.sim.timestep,
            solver=cfg.sim.solver,
            iterations=cfg.sim.iterations,
            ls_iterations=cfg.sim.ls_iterations,
            ccd_iterations=cfg.sim.ccd_iterations,
            impratio=1.0,
            cone="pyramidal",
        ),
    )
    viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name="base",
        distance=cfg.viewer.distance,
        elevation=cfg.viewer.elevation,
        azimuth=cfg.viewer.azimuth,
    )

    # ---------- 组装 ----------

    return ManagerBasedRlEnvCfg(
        scene=scene,
        # adapter在真实步进后提取任务状态；本体、特权观测和历史只有任务模型一处定义。
        observations={},
        actions=actions,
        events=events,
        commands=commands,
        rewards=rewards,
        curriculum=curriculum,
        metrics=metrics,
        terminations=terminations,
        auto_reset=False,
        decimation=cfg.control.decimation,
        episode_length_s=cfg.env.episode_length_s,
        seed=cfg.env.seed,
        sim=sim,
        viewer=viewer,
    )
