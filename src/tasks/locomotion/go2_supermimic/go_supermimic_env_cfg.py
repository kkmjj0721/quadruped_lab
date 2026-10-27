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
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from src.assets.robots.go2_reference import get_reference_go2_robot_cfg
from src.config.go2.go2_reference_supermimic_config import Go2ReferenceCfg

from .mdp.state_spec import FOOT_NAMES
from .mdp.terrains import TerrainMap


def validate_config(cfg: Go2ReferenceCfg) -> None:
    """在构造仿真前统一校验任务入口参数。"""
    if cfg.algorithm != "single_mlp":
        raise ValueError("SuperMimic requires the single_mlp algorithm")
    if cfg.commands.heading_command:
        raise ValueError("SuperMimic requires direct yaw-rate commands")
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
    if cfg.terrain.mode not in ("flat", "rough"):
        raise ValueError("Unsupported SuperMimic terrain mode")


def make_go2_supermimic_env_cfg(
    cfg: Go2ReferenceCfg | None = None, *, terrain_map: TerrainMap | None = None,
) -> ManagerBasedRlEnvCfg:
    """返回可直接构造MjLab环境的配置，不复用任何PPO任务内部实现。"""
    cfg = deepcopy(cfg) if cfg is not None else Go2ReferenceCfg()
    validate_config(cfg)
    terrain_map = terrain_map if terrain_map is not None else TerrainMap(cfg.terrain)

    # ---------- 场景 ----------

    feet = tuple(f"{leg}_foot_collision" for leg in FOOT_NAMES)
    scene = SceneCfg(
        entities={"robot": get_reference_go2_robot_cfg(cfg)},
        terrain=terrain_map.entity_cfg(),
        sensors=(),
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
        events.update({
            "foot_friction": EventTermCfg(func=envs_mdp.dr.geom_friction, mode="startup", params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=feet), "operation": "abs",
                "distribution": "uniform", "ranges": tuple(domain.friction_range),
                "axes": [0], "shared_random": True,
            }),
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
            impratio=10,
            cone="elliptic",
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
        # adapter在真实步进后提取73维状态；本体、特权观测和历史只有任务模型一处定义。
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
