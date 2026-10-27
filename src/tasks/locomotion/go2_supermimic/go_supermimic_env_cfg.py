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
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from src.assets.robots import get_go2_robot_cfg
from src.config.go2.go2_supermimic_config import Go2SuperMimicCfg

from .mdp.observations import physical_observation
from .mdp.state_spec import FOOT_NAMES
from .mdp.terrains import TerrainMap


def validate_config(cfg: Go2SuperMimicCfg) -> None:
    """拒绝尚未建模的执行器时序及不一致的任务参数。"""
    if cfg.control.delay_min_lag != 0 or cfg.control.delay_max_lag != 0:
        raise ValueError("SuperMimic v2 requires zero actuator delay")
    if cfg.commands.heading_command:
        raise ValueError("SuperMimic v2 requires direct yaw-rate commands")
    if isinstance(cfg.env.num_envs, bool) or not isinstance(cfg.env.num_envs, int) or cfg.env.num_envs < 1:
        raise ValueError("Environment count must be a positive integer")
    if not math.isfinite(cfg.env.episode_length_s) or cfg.env.episode_length_s <= 0:
        raise ValueError("Episode duration must be positive and finite")
    if not math.isfinite(cfg.sim.timestep) or cfg.sim.timestep <= 0:
        raise ValueError("Simulation timestep must be positive and finite")
    if isinstance(cfg.control.decimation, bool) or not isinstance(cfg.control.decimation, int) or cfg.control.decimation < 1:
        raise ValueError("Control decimation must be a positive integer")
    if cfg.env.num_obs_history != 5:
        raise ValueError("The Go2 student requires exactly five history frames")
    if cfg.terrain.mode not in ("flat", "rough"):
        raise ValueError("Unsupported SuperMimic terrain mode")


def make_go2_supermimic_env_cfg(
    cfg: Go2SuperMimicCfg | None = None, *, terrain_map: TerrainMap | None = None,
) -> ManagerBasedRlEnvCfg:
    """返回可直接构造MjLab环境的配置，不复用任何PPO任务内部实现。"""
    cfg = deepcopy(cfg) if cfg is not None else Go2SuperMimicCfg()
    validate_config(cfg)
    terrain_map = terrain_map if terrain_map is not None else TerrainMap(cfg.terrain)

    # ---------- 场景与传感器 ----------

    feet = tuple(f"{leg}_foot_collision" for leg in FOOT_NAMES)
    contact = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", entity="robot", pattern=feet),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    # 自碰历史覆盖本控制周期的全部物理子步，保持参考奖励的计数语义。
    leg_contact = ContactSensorCfg(
        name="leg_ground_contact",
        primary=ContactMatch(mode="geom", entity="robot", pattern=r"^(FR|FL|RR|RL)_(hip|thigh|calf).*_collision$"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), reduce="none", num_slots=1, history_length=1,
    )
    self_contact = ContactSensorCfg(
        name="self_collision_contact",
        primary=ContactMatch(mode="geom", entity="robot", pattern=r".*_collision$"),
        secondary=ContactMatch(mode="subtree", entity="robot", pattern="base_link"),
        fields=("found", "force"), reduce="maxforce", num_slots=1,
        history_length=cfg.control.decimation, secondary_policy="error",
    )
    scene = SceneCfg(
        entities={"robot": get_go2_robot_cfg(cfg)},
        terrain=terrain_map.entity_cfg(),
        sensors=(() if getattr(cfg, "algorithm", "cts") == "single_moe" else (contact, leg_contact, self_contact)),
        num_envs=cfg.env.num_envs,
        extent=cfg.env.scene_extent,
    )

    # ---------- 观测 ----------

    # 物理环境只输出真值；本体帧、特权观测与历史由本任务MDP和算法runner生成。
    observations = {
        "state": ObservationGroupCfg(
            terms={"physical": ObservationTermCfg(func=physical_observation)},
            enable_corruption=False,
            concatenate_terms=True,
        ),
    }

    # ---------- 动作 ----------

    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            use_default_offset=True,
            scale={
                r".*_hip_joint": cfg.control.action_scale * cfg.control.hip_reduction,
                r".*_thigh_joint": cfg.control.action_scale,
                r".*_calf_joint": cfg.control.action_scale,
            },
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

    # ---------- 终止与任务逻辑边界 ----------

    # 原生环境只判定超时；跌倒等失败由mdp/terminations.py在reset前统一计算。
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
        contact_sensor_maxmatch=cfg.sim.contact_sensor_maxmatch,
        broadphase=cfg.sim.broadphase,
        mujoco=MujocoCfg(
            timestep=cfg.sim.timestep,
            solver=cfg.sim.solver,
            iterations=cfg.sim.iterations,
            ls_iterations=cfg.sim.ls_iterations,
            ccd_iterations=cfg.sim.ccd_iterations,
        ),
    )
    viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name="base_link",
        distance=cfg.viewer.distance,
        elevation=cfg.viewer.elevation,
        azimuth=cfg.viewer.azimuth,
    )

    # ---------- 组装 ----------

    return ManagerBasedRlEnvCfg(
        scene=scene,
        observations=observations,
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
