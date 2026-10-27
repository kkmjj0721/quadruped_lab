# Copyright (c) 2026-2027 zh
"""
内容：
    用双线性插值随机高度图生成具有平坦起终点的不规则地面。
"""

from dataclasses import dataclass
import math

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import TerrainOutput

from .common import BlindTerrainCfg, add_heightfield, heightfield_axes, validate_patch, validate_range


@dataclass(kw_only=True)
class RoughTerrainCfg(BlindTerrainCfg):
    """起伏幅值以米计，粗网格间距控制不规则表面的空间尺度。"""

    amplitude_range: tuple[float, float] = (0.0, 0.10)
    horizontal_scale: float = 0.05
    noise_spacing: float = 0.25

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        validate_patch(self, difficulty)
        validate_range(self.amplitude_range, "Amplitude range")
        x, y = heightfield_axes(self.size, self.horizontal_scale)
        if not math.isfinite(self.noise_spacing) or self.noise_spacing < self.horizontal_scale:
            raise ValueError("Noise spacing must be finite and at least horizontal_scale.")
        coarse_x, coarse_y = heightfield_axes(self.size, self.noise_spacing)
        amplitude = self.amplitude_range[0] + difficulty * (self.amplitude_range[1] - self.amplitude_range[0])
        noise = rng.uniform(-amplitude, amplitude, (len(coarse_y), len(coarse_x)))
        along_x = np.array([np.interp(x, coarse_x, row) for row in noise])
        heights = np.array([np.interp(y, coarse_y, column) for column in along_x.T]).T
        # 平台边界向内部对齐采样点，非整除分辨率也不侵入指定的平坦平台。
        start = x[np.searchsorted(x, self.platform_length, side="left")]
        end = x[np.searchsorted(x, self.size[0] - self.platform_length, side="right") - 1]
        if end <= start:
            raise ValueError("horizontal_scale leaves no rough surface between the flat platforms.")
        # 用一个粗网格间距渐变，避免平台与起伏区之间高度突变。
        taper = np.clip(np.minimum(x - start, end - x) / self.noise_spacing, 0.0, 1.0)
        heights *= taper[None, :]
        geometry = add_heightfield(spec, self, heights, (0.58, 0.48, 0.32, 1.0))
        origin = np.array([self.platform_length / 2, self.size[1] / 2, 0.0])
        return TerrainOutput(origin=origin, geometries=[geometry])
