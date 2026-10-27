# Copyright (c) 2026-2027 zh
"""
内容：
    生成可拼入 MjLab 地形网格的有限平地。
"""

from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import TerrainOutput

from .common import BlindTerrainCfg, add_box, validate_patch


@dataclass(kw_only=True)
class FlatTerrainCfg(BlindTerrainCfg):
    """地表高度为零的有限平地。"""

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        validate_patch(self, difficulty)
        geometry = add_box(spec, self, 0.0, self.size[0], 0.0, -self.base_thickness, (0.48, 0.56, 0.63, 1.0))
        origin = np.array([self.platform_length / 2, self.size[1] / 2, 0.0])
        return TerrainOutput(origin=origin, geometries=[geometry])
