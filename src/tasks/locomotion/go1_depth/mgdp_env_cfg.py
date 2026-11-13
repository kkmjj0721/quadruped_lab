# Copyright (c) 2026-2027 zh
"""
内容：
    装配 MGDP 原生两阶段 Go1 场景；采集、命令、课程和手动重置由适配器负责。
"""

from dataclasses import replace
import math

import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers import SceneEntityCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from src.assets.robots.go1_depth.go1_mgdp_constants import (
    BASE_SENSOR_NAME, CAMERA_NAME, FEET_SENSOR_NAME, FOOT_CONTACT_NAMES,
    JOINT_NAMES, LEG_CONTACT_NAMES, LEG_SENSOR_NAME, RANDOMIZED_LINK_NAMES,
    get_go1_mgdp_camera_cfg, get_go1_mgdp_robot_cfg,
)
from src.config.go1_depth.go1_depth_config import Go1MGDPConfig
from src.terrain.mgdp import MgdpTerrainData, make_mgdp_terrain_cfg
from .mdp.mgdp_randomization import MGDPJointActionCfg, make_mgdp_dynamics_event


class MGDPImplicitCfg(MujocoCfg):
    """MjLab配置表尚未暴露MJWarp已实现的完整隐式积分，在模型构建边界接入。

    保留基类全部物理选项，仅使速度雅可比包含多刚体科里奥利和陀螺项。
    对照 mujoco_warp/_src/forward.py:579 的 IMPLICIT 分支。
    """

    def apply(self, model):
        MujocoCfg.apply(replace(self, integrator="implicitfast"), model)
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT


def make_mgdp_env_cfg(cfg: Go1MGDPConfig, data: MgdpTerrainData) -> ManagerBasedRlEnvCfg:
    """返回无自动重置的场景；传感器净力顺序匹配 MGDPTask 的物理状态。"""
    if len(data.families) != 1 or data.families[0].name != cfg.family or data.seed != cfg.seed:
        raise ValueError("Terrain family and seed must match the selected MGDP stage.")
    sensors = [get_go1_mgdp_camera_cfg()]
    for name, mode, primary in (
        (FEET_SENSOR_NAME, "geom", FOOT_CONTACT_NAMES),
        (BASE_SENSOR_NAME, "body", "base_link"),
        (LEG_SENSOR_NAME, "geom", LEG_CONTACT_NAMES),
    ):
        sensors.append(ContactSensorCfg(
            name=name, primary=ContactMatch(mode=mode, entity="robot", pattern=primary),
            secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"),
            reduce="netforce", num_slots=1,
        ))
    camera = SceneEntityCfg("robot", camera_names=(CAMERA_NAME,))
    events = {
        "dynamics": make_mgdp_dynamics_event(link_names=RANDOMIZED_LINK_NAMES, scale_inertia=True),
        "camera_translation": EventTermCfg(func=dr.cam_pos, mode="startup", params={
            "asset_cfg": camera, "ranges": (-0.02, 0.02), "operation": "add", "axes": [0, 1, 2],
        }),
        "camera_pitch": EventTermCfg(func=dr.cam_quat, mode="startup", params={
            "asset_cfg": camera, "pitch_range": (-math.radians(2.0), math.radians(2.0)),
        }),
    }
    physics_type = MGDPImplicitCfg if cfg.physics_integrator == "implicit" else MujocoCfg
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=cfg.num_envs, entities={"robot": get_go1_mgdp_robot_cfg()},
            terrain=TerrainEntityCfg(
                terrain_type="generator", terrain_generator=make_mgdp_terrain_cfg(data),
                # 复合地形仅有一个装配块；真实课程原点随后由适配器赋值。
                max_init_terrain_level=0,
            ),
            sensors=tuple(sensors),
        ),
        actions={"joint_pos": MGDPJointActionCfg(
            entity_name="robot", actuator_names=JOINT_NAMES, latency_seed=cfg.seed, decimation=cfg.decimation,
        )},
        observations={}, rewards={}, commands={}, terminations={}, curriculum={}, metrics={}, events=events,
        auto_reset=False, decimation=cfg.decimation, episode_length_s=cfg.episode_length_s, seed=cfg.seed,
        sim=SimulationCfg(
            # 每接触最多10个金字塔约束，再加12个关节限位；256不足以覆盖64接触。
            nconmax=64, njmax=1024,
            mujoco=physics_type(integrator=cfg.physics_integrator, timestep=cfg.timestep, iterations=10, ls_iterations=10, solver="newton", cone="pyramidal"),
        ),
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="base_link",
            distance=3.0, elevation=-20.0, azimuth=90.0,
        ),
    )
