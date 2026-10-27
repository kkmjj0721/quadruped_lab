# Copyright (c) 2026-2027 zh
"""
内容：
    生成 MjLab 风格的中心平台正/倒金字塔高度图斜坡。
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import TerrainOutput

from .common import (
    BlindTerrainCfg,
    add_heightfield,
    heightfield_axes,
    validate_patch,
    validate_range,
)


@dataclass(kw_only=True)
class SlopeTerrainCfg(BlindTerrainCfg):
    """中心平台向四周连续变化的正/倒金字塔坡配置。"""

    descending: bool = False
    """为真时中心平台下沉，生成 MjLab 的 inverted 金字塔坡。"""
    slope_range: tuple[float, float] = (0.0, 1.0)
    """难度从 0 到 1 对应的坡度范围，坡度为高差/水平距离，不是角度。"""
    platform_width: float = 2.0
    """中心方形平台的边长，单位为米。"""
    border_width: float = 0.25
    """地块外围零高度边框的宽度，单位为米。"""
    horizontal_scale: float = 0.05
    """高度图网格的最大水平间距，单位为米。"""

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        """按 MjLab 高度图公式生成中心平台正/倒金字塔坡。"""
        del rng
        validate_patch(self, difficulty)
        validate_range(self.slope_range, "slope_range")

        if not np.isfinite(self.platform_width) or self.platform_width <= 0.0:
            raise ValueError("platform_width must be finite and positive.")
        if not np.isfinite(self.border_width) or self.border_width < 0.0:
            raise ValueError("border_width must be finite and non-negative.")
        if not np.isfinite(self.horizontal_scale) or self.horizontal_scale <= 0.0:
            raise ValueError("horizontal_scale must be finite and positive.")
        if self.border_width > 0.0 and self.border_width < self.horizontal_scale:
            raise ValueError("border_width must be zero or at least horizontal_scale.")

        x, y = heightfield_axes(self.size, self.horizontal_scale)
        x_indices = np.flatnonzero(
            (x >= self.border_width) & (x <= self.size[0] - self.border_width)
        )
        y_indices = np.flatnonzero(
            (y >= self.border_width) & (y <= self.size[1] - self.border_width)
        )
        if x_indices.size < 3 or y_indices.size < 3:
            raise ValueError(
                "horizontal_scale must leave at least three interior heightfield samples."
            )

        inner_x, inner_y = x[x_indices], y[y_indices]
        center_x, center_y = self.size[0] / 2.0, self.size[1] / 2.0
        platform_x_min = center_x - self.platform_width / 2.0
        platform_x_max = center_x + self.platform_width / 2.0
        platform_y_min = center_y - self.platform_width / 2.0
        platform_y_max = center_y + self.platform_width / 2.0
        if not np.any((inner_x >= platform_x_min) & (inner_x <= platform_x_max)):
            raise ValueError("horizontal_scale must sample the center platform along X.")
        if not np.any((inner_y >= platform_y_min) & (inner_y <= platform_y_max)):
            raise ValueError("horizontal_scale must sample the center platform along Y.")

        # 高度图必须保持均匀网格；非整除尺寸时，边框实际量化到第一个内部采样点。
        inner_width, inner_length = inner_x[-1] - inner_x[0], inner_y[-1] - inner_y[0]
        if self.platform_width >= min(inner_width, inner_length):
            raise ValueError("platform_width must fit inside the sampled terrain area.")

        slope = self.slope_range[0] + difficulty * (self.slope_range[1] - self.slope_range[0])
        if self.descending:
            slope = -slope

        inner_half_x, inner_half_y = inner_width / 2.0, inner_length / 2.0
        height_max = slope * inner_width / 2.0

        # MjLab 使用两个三角剖面相乘，得到四面向中心收拢的连续金字塔坡。
        x_factor = (inner_half_x - np.abs(inner_x - center_x)) / inner_half_x
        y_factor = (inner_half_y - np.abs(inner_y - center_y)) / inner_half_y
        raw = height_max * y_factor[:, None] * x_factor[None, :]

        platform_x = center_x - self.platform_width / 2.0
        platform_y = center_y - self.platform_width / 2.0
        platform_x_factor = (inner_half_x - abs(platform_x - center_x)) / inner_half_x
        platform_y_factor = (inner_half_y - abs(platform_y - center_y)) / inner_half_y
        platform_height = height_max * platform_x_factor * platform_y_factor
        raw = np.clip(raw, min(0.0, platform_height), max(0.0, platform_height))

        heights = np.zeros((len(y), len(x)), dtype=np.float64)
        heights[np.ix_(y_indices, x_indices)] = raw
        geometry = add_heightfield(spec, self, heights, (0.57, 0.65, 0.44, 1.0))
        origin = np.array((center_x, center_y, platform_height))
        return TerrainOutput(origin=origin, geometries=[geometry])
