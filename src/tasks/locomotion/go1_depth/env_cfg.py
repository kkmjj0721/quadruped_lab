# Copyright (c) 2026-2027 zh
"""内容：装配Go1、同步原生深度相机及视觉地形；任务奖励和重置由适配器维护。"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import SceneEntityCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from src.assets.robots.go1_depth.go1_depth_constants import FOOT_NAMES, get_go1_depth_camera_cfg, get_go1_depth_robot_cfg
from src.terrain.vision.config import make_vision_terrain_cfg


def make_env_cfg(cfg):
    """终止manager保持为空，适配器可先读取同一时刻的物理量与图像再重置。"""
    terrain = (TerrainEntityCfg(terrain_type="plane") if cfg.terrain == "flat" else
               TerrainEntityCfg(terrain_type="generator", max_init_terrain_level=0,
                                terrain_generator=make_vision_terrain_cfg(seed=cfg.seed, num_rows=cfg.terrain_rows)))
    feet = ContactSensorCfg(name="feet_contact", primary=ContactMatch(mode="geom", entity="robot",
                             pattern=tuple(f"{foot}_foot_collision" for foot in FOOT_NAMES)),
                            secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"),
                            reduce="netforce", num_slots=1)
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(num_envs=cfg.num_envs, entities={"robot": get_go1_depth_robot_cfg()}, terrain=terrain,
                       sensors=(get_go1_depth_camera_cfg(), feet)),
        actions={"joint_pos": JointPositionActionCfg(entity_name="robot", actuator_names=(".*",),
                                                    use_default_offset=True, scale=0.25)},
        observations={}, rewards={}, commands={}, terminations={}, curriculum={}, metrics={},
        events={
            "root": EventTermCfg(func=mdp.reset_root_state_uniform, mode="reset", params={
                "pose_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15), "yaw": (-0.15, 0.15)},
                "velocity_range": {}}),
            "joints": EventTermCfg(func=mdp.reset_joints_by_offset, mode="reset", params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "position_range": (-0.02, 0.02), "velocity_range": (0.0, 0.0)})},
        auto_reset=False, decimation=cfg.decimation, episode_length_s=cfg.episode_length_s, seed=cfg.seed,
        sim=SimulationCfg(nconmax=64, njmax=256, mujoco=MujocoCfg(timestep=cfg.timestep, iterations=10,
                            ls_iterations=10, solver="newton", cone="pyramidal")),
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="base_link",
                            distance=3.0, elevation=-20.0, azimuth=90.0))
