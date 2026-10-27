# Copyright (c) 2026-2027 zh
"""
内容：
    组合六类盲走地形的独立配置，供场景装配和预览工具使用。
"""

from numbers import Integral
from typing import Literal

from mjlab.terrains import TerrainGeneratorCfg

from .blind.common import validate_patch, validate_range
from .blind.flat_terrain import FlatTerrainCfg
from .blind.rough_terrain import RoughTerrainCfg
from .blind.slope_terrain import SlopeTerrainCfg
from .blind.stairs_terrain import InvertedPyramidStairsTerrainCfg, PyramidStairsTerrainCfg


BlindTerrainName = Literal[
    "all", "flat", "rough", "pyramid_stairs", "pyramid_stairs_inv", "slope_up", "slope_down",
    "stairs_up", "stairs_down",
]
BLIND_TERRAIN_NAMES = (
    "flat", "rough", "pyramid_stairs", "pyramid_stairs_inv", "slope_up", "slope_down",
)
_TERRAIN_ALIASES = {"stairs_up": "pyramid_stairs", "stairs_down": "pyramid_stairs_inv"}


def make_blind_terrain_cfg(
    terrain: BlindTerrainName = "all", *, seed: int = 42, size: tuple[float, float] = (8.0, 8.0),
    num_rows: int = 5, num_cols: int = 6, curriculum: bool = True,
    difficulty_range: tuple[float, float] = (0.0, 1.0),
) -> TerrainGeneratorCfg:
    """返回新配置；课程模式每类一列，随机模式使用指定的列数。"""
    terrain = _TERRAIN_ALIASES.get(terrain, terrain)
    if terrain != "all" and terrain not in BLIND_TERRAIN_NAMES:
        raise ValueError(f"Unknown blind terrain: {terrain!r}.")
    for name, count in (("num_rows", num_rows), ("num_cols", num_cols)):
        if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("Seed must be a non-negative integer.")
    validate_range(difficulty_range, "Difficulty range")
    if difficulty_range[1] > 1.0:
        raise ValueError("Difficulty range must lie in [0, 1].")
    sub_terrains = {
        "flat": FlatTerrainCfg(size=size),
        "rough": RoughTerrainCfg(size=size),
        "pyramid_stairs": PyramidStairsTerrainCfg(size=size),
        "pyramid_stairs_inv": InvertedPyramidStairsTerrainCfg(size=size),
        "slope_up": SlopeTerrainCfg(size=size),
        "slope_down": SlopeTerrainCfg(size=size, descending=True),
    }
    if terrain != "all":
        sub_terrains = {terrain: sub_terrains[terrain]}
    for cfg in sub_terrains.values():
        validate_patch(cfg, difficulty_range[0])
    return TerrainGeneratorCfg(
        seed=int(seed), size=size, num_rows=int(num_rows), num_cols=int(num_cols),
        curriculum=curriculum, difficulty_range=difficulty_range, sub_terrains=sub_terrains,
        border_width=0.0, color_scheme="height",
    )
