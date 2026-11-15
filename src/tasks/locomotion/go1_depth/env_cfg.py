# Copyright (c) 2026-2027 zh
"""内容：装配 Go1 资产、Go2 rough 地形、四足接触和真实机载深度相机。"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import SceneEntityCfg, TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from src.assets.robots.go1_depth.go1_depth_constants import FOOT_GEOM_NAMES, ROOT_BODY_NAME, get_go1_depth_robot_cfg
from src.config.go1_depth.go1_depth_config import Go1DepthCfg, make_go1_depth_camera_cfg
from src.tasks.locomotion.go2_supermimic.mdp.terrains import TerrainMap


def make_go1_depth_env_cfg(cfg=None, *, terrain_map=None, play=False):
    """原生环境只做物理和超时，指令、奖励与手动 reset 由视觉 adapter 统一推进。"""
    cfg = deepcopy(cfg) if cfg is not None else Go1DepthCfg()
    if cfg.algorithm != "depth_supermimic" or cfg.env.num_envs < 1:
        raise ValueError("Go1 depth requires depth_supermimic and a positive environment count")
    if cfg.terrain.mode not in ("flat", "rough"):
        raise ValueError("Go1 depth terrain mode must be flat or rough")
    if (cfg.sensors.depth.width, cfg.sensors.depth.height) != (32, 32):
        raise ValueError("The Go1 depth task requires a 32x32 raw camera")
    terrain_map = terrain_map if terrain_map is not None else TerrainMap(cfg.terrain)
    sensors = (
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(mode="geom", pattern=FOOT_GEOM_NAMES, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True,
        ),
        ContactSensorCfg(
            name="leg_ground_contact",
            primary=ContactMatch(mode="geom", pattern=r"^(FR|FL|RR|RL)_(hip|thigh|calf).*_collision\d*$", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="none", num_slots=1,
            history_length=cfg.sensors.nonfoot_history_length,
        ),
        make_go1_depth_camera_cfg(cfg),
    )
    scene = SceneCfg(entities={"robot": get_go1_depth_robot_cfg(cfg)}, terrain=terrain_map.entity_cfg(),
                     sensors=sensors, num_envs=cfg.env.num_envs, extent=cfg.env.scene_extent)
    actions = {"joint_pos": JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",), use_default_offset=True,
        scale={rf".*_{name}_joint": value for name, value in cfg.control.action_scales.items()},
    )}
    events = {
        "reset_root": EventTermCfg(func=envs_mdp.reset_root_state_uniform, mode="reset", params=deepcopy(cfg.reset.base_offset)),
        "reset_joints": EventTermCfg(func=envs_mdp.reset_joints_by_offset, mode="reset", params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "position_range": tuple(cfg.reset.joint_offset["position_range"]),
            "velocity_range": tuple(cfg.reset.joint_offset["velocity_range"]),
        }),
    }
    if cfg.domain_randomization.enabled:
        domain = cfg.domain_randomization
        for name, axis, distribution, bounds in (
            ("slide", 0, "uniform", domain.friction_range),
            ("spin", 1, "log_uniform", domain.spin_friction_range),
            ("roll", 2, "log_uniform", domain.roll_friction_range),
        ):
            events[f"foot_friction_{name}"] = EventTermCfg(func=envs_mdp.dr.geom_friction, mode="startup", params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES), "operation": "abs",
                "distribution": distribution, "ranges": tuple(bounds), "axes": [axis], "shared_random": True,
            })
        events.update({
            "encoder_bias": EventTermCfg(func=envs_mdp.dr.encoder_bias, mode="startup", params={
                "asset_cfg": SceneEntityCfg("robot"), "bias_range": tuple(domain.encoder_bias_range),
            }),
            "base_com": EventTermCfg(func=envs_mdp.dr.body_com_offset, mode="startup", params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(ROOT_BODY_NAME,)), "operation": "add",
                "ranges": {axis: tuple(domain.com_range[name]) for axis, name in enumerate(("x", "y", "z"))},
            }),
        })
    sim = SimulationCfg(
        nconmax=cfg.sim.nconmax, njmax=cfg.sim.njmax, broadphase=cfg.sim.broadphase,
        mujoco=MujocoCfg(timestep=cfg.sim.timestep, solver=cfg.sim.solver, iterations=cfg.sim.iterations,
                        ls_iterations=cfg.sim.ls_iterations, ccd_iterations=cfg.sim.ccd_iterations,
                        impratio=1.0, cone="pyramidal"),
    )
    viewer = ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name=ROOT_BODY_NAME,
                          distance=cfg.viewer.distance, elevation=cfg.viewer.elevation, azimuth=cfg.viewer.azimuth)
    terminations = {} if play else {"time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True)}
    return ManagerBasedRlEnvCfg(scene=scene, observations={}, actions=actions, events=events, commands={}, rewards={},
                               curriculum={}, metrics={}, terminations=terminations, auto_reset=False,
                               decimation=cfg.control.decimation, episode_length_s=cfg.env.episode_length_s,
                               seed=cfg.env.seed, sim=sim, viewer=viewer)
