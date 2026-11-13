# Copyright (c) 2026-2027 zh
"""
内容：
    导出 MGDP 原版地形数据、MuJoCo 装配配置和独立监督标签查询。
"""

from .config import make_mgdp_terrain_cfg
from .generator import (
    MGDP_COMMIT, MIX_CFG, PARKOUR_CFG, MgdpBox, MgdpFamilyData,
    MgdpTerrainData, generate_mgdp_terrain,
)
from .query import MgdpTerrainQuery

__all__ = [
    "MGDP_COMMIT", "MIX_CFG", "PARKOUR_CFG", "MgdpBox", "MgdpFamilyData",
    "MgdpTerrainData", "MgdpTerrainQuery", "generate_mgdp_terrain", "make_mgdp_terrain_cfg",
]
