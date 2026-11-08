# Copyright (c) 2026-2027 zh
"""
内容：
    提供视觉地形课程的配置入口和按角度控制的上下坡配置。
"""

from .config import VISION_TERRAIN_NAMES, make_vision_terrain_cfg
from .slope_terrain import PyramidSlopeTerrainCfg

__all__ = ["VISION_TERRAIN_NAMES", "make_vision_terrain_cfg", "PyramidSlopeTerrainCfg"]
