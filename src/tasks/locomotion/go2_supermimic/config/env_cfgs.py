# Copyright (c) 2026-2027 zh
"""内容：从集中参数创建独立的Go2粗糙地形、平地及播放模式配置。"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg

from src.config.go2.go2_reference_supermimic_config import Go2ReferenceCfg

from ..go_supermimic_env_cfg import make_go2_supermimic_env_cfg, validate_config


def resolve_go2_supermimic_cfg(
    go2_cfg: Go2ReferenceCfg | None = None, *, play: bool = False, terrain_mode: str | None = None,
) -> Go2ReferenceCfg:
    """覆盖完整任务参数；返回副本以便adapter与纯Tensor任务使用同一语义。"""
    cfg = deepcopy(go2_cfg) if go2_cfg is not None else Go2ReferenceCfg()
    if terrain_mode is not None:
        cfg.terrain.mode = terrain_mode
    if play:
        cfg.noise.add_noise = False
        cfg.terrain.curriculum = False
    validate_config(cfg)
    return cfg


def _make_env_cfg(play, go2_cfg, terrain_mode, terrain_map):
    cfg = resolve_go2_supermimic_cfg(go2_cfg, play=play, terrain_mode=terrain_mode)
    if terrain_map is not None and terrain_map.plane != (terrain_mode == "flat"):
        raise ValueError("Terrain map and environment configuration modes differ")
    env_cfg = make_go2_supermimic_env_cfg(cfg, terrain_map=terrain_map)
    if play:
        # 播放持续运行，跌倒重置仍由adapter的任务终止判定负责。
        del env_cfg.terminations["time_out"]
    return env_cfg


def unitree_go2_rough_env_cfg(
    play: bool = False, *, go2_cfg: Go2ReferenceCfg | None = None, terrain_map=None,
) -> ManagerBasedRlEnvCfg:
    """创建保留完整固定连通地图的粗糙地形环境配置。"""
    return _make_env_cfg(play, go2_cfg, "rough", terrain_map)


def unitree_go2_flat_env_cfg(
    play: bool = False, *, go2_cfg: Go2ReferenceCfg | None = None, terrain_map=None,
) -> ManagerBasedRlEnvCfg:
    """创建平地配置，沿用同一套任务协议与名义控制参数。"""
    return _make_env_cfg(play, go2_cfg, "flat", terrain_map)
