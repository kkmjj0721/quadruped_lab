# Copyright (c) 2026-2027 zh
"""
内容：
    按固定列顺序组合视觉地形的上下坡、粗糙高度场和上下楼梯课程。
"""

import math
from numbers import Integral

from mjlab.terrains import TerrainGeneratorCfg
from mjlab.terrains.config import random_rough

from ..blind.stairs_terrain import InvertedPyramidStairsTerrainCfg, PyramidStairsTerrainCfg
from .slope_terrain import PyramidSlopeTerrainCfg


VISION_TERRAIN_NAMES = ("slope", "rough", "stairs_up", "stairs_down")


def make_vision_terrain_cfg(
    *, seed: int = 42, size: tuple[float, float] = (8.0, 8.0), num_rows: int = 10,
) -> TerrainGeneratorCfg:
    """返回四列独立配置；行难度包含两端，单行使用最低难度。"""
    if isinstance(num_rows, bool) or not isinstance(num_rows, Integral) or num_rows <= 0:
        raise ValueError("num_rows must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if len(size) != 2 or any(not math.isfinite(value) or value <= 0 for value in size):
        raise ValueError("size must contain two finite positive lengths.")

    sub_terrains = {
        "slope": PyramidSlopeTerrainCfg(size=size),
        # 沿用 Go2 SuperMimic 原始 rough 列，保留不随难度缩放的上游行为。
        "rough": random_rough(size=size, noise_range=(0.01, 0.04), noise_step=0.01),
        "stairs_up": PyramidStairsTerrainCfg(size=size, step_width=0.20, step_height_range=(0.05, 0.20)),
        "stairs_down": InvertedPyramidStairsTerrainCfg(size=size, step_width=0.20, step_height_range=(0.05, 0.20)),
    }
    return TerrainGeneratorCfg(
        seed=int(seed), size=size, num_rows=int(num_rows), num_cols=len(sub_terrains),
        curriculum=True, difficulty_range=(0.0, 1.0), sub_terrains=sub_terrains,
        border_width=0.0, color_scheme="height",
    )
