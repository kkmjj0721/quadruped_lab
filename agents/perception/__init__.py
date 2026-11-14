# Copyright (c) 2026-2027 zh
"""内容：公开 MGDP 感知、显式时序地图和有效标签监督接口。"""

from .config import PerceptionConfig
from .mgdp import DepthPerception, MGDPPerception, map_loss

__all__ = ["PerceptionConfig", "DepthPerception", "MGDPPerception", "map_loss"]
