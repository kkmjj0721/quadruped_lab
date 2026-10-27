# Copyright (c) 2026-2027 zh
"""
内容：
    生成有限正金字塔和倒金字塔同心方环楼梯，并保留直线楼梯兼容配置。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput

from .common import BlindTerrainCfg, add_box, validate_patch, validate_range


_BLUE = (0.20, 0.45, 0.95)
_RED = (0.90, 0.30, 0.30)
_MIN_GEOM_SIZE = 1e-6


def _add_rectangular_box(
    spec: mujoco.MjSpec,
    cfg: BlindTerrainCfg,
    x_start: float,
    x_end: float,
    y_start: float,
    y_end: float,
    top: float,
    bottom: float,
    color: tuple[float, float, float, float],
) -> TerrainGeometry:
    """添加覆盖指定 XY 矩形的有限盒体。"""
    x_length = x_end - x_start
    y_length = y_end - y_start
    z_length = top - bottom
    if min(x_length, y_length, z_length) <= 0.0:
        raise ValueError("Rectangular terrain box must have positive dimensions.")
    geom = spec.body("terrain").add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=((x_start + x_end) / 2.0, (y_start + y_end) / 2.0, (top + bottom) / 2.0),
        size=(
            max(x_length / 2.0, _MIN_GEOM_SIZE),
            max(y_length / 2.0, _MIN_GEOM_SIZE),
            max(z_length / 2.0, _MIN_GEOM_SIZE),
        ),
        friction=cfg.friction,
        rgba=color,
    )
    return TerrainGeometry(geom=geom, color=color)


def _add_border(
    spec: mujoco.MjSpec,
    cfg: BlindTerrainCfg,
    border: float,
    bottom: float,
    color: tuple[float, float, float, float],
) -> list[TerrainGeometry]:
    """添加不与内部台阶重叠的外围平框。"""
    if border <= 0.0:
        return []
    size_x, size_y = cfg.size
    return [
        _add_rectangular_box(spec, cfg, 0.0, border, 0.0, size_y, 0.0, bottom, color),
        _add_rectangular_box(
            spec, cfg, size_x - border, size_x, 0.0, size_y, 0.0, bottom, color
        ),
        _add_rectangular_box(spec, cfg, border, size_x - border, 0.0, border, 0.0, bottom, color),
        _add_rectangular_box(
            spec, cfg, border, size_x - border, size_y - border, size_y,
            0.0, bottom, color,
        ),
    ]


def _validate_pyramid_parameters(cfg: "PyramidStairsTerrainCfg", difficulty: float) -> float:
    """检查金字塔参数并返回当前难度对应的单级高度。"""
    validate_patch(cfg, difficulty)
    validate_range(cfg.step_height_range, "step_height_range")
    size_x, size_y = cfg.size
    if not math.isfinite(cfg.border_width) or cfg.border_width < 0.0:
        raise ValueError("border_width must be finite and non-negative.")
    if 2.0 * cfg.border_width >= min(size_x, size_y):
        raise ValueError("border_width must leave a positive inner terrain area.")
    if not math.isfinite(cfg.platform_width) or cfg.platform_width <= 0.0:
        raise ValueError("platform_width must be finite and positive.")
    inner_size = min(size_x - 2.0 * cfg.border_width, size_y - 2.0 * cfg.border_width)
    if cfg.platform_width > inner_size:
        raise ValueError("platform_width must fit inside the bordered terrain area.")
    if not math.isfinite(cfg.step_width) or cfg.step_width <= 0.0:
        raise ValueError("step_width must be finite and positive.")
    maximum_step_width = (inner_size - cfg.platform_width) / 2.0
    if cfg.step_width > maximum_step_width:
        raise ValueError("step_width must leave room for at least one stair ring.")
    return cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )


def _make_pyramid_stairs(
    cfg: "PyramidStairsTerrainCfg",
    difficulty: float,
    spec: mujoco.MjSpec,
    inverted: bool,
) -> TerrainOutput:
    """用四条矩形边带和中心盒体构造一类金字塔楼梯。"""
    step_height = _validate_pyramid_parameters(cfg, difficulty)
    size_x, size_y = cfg.size
    inner_x = size_x - 2.0 * cfg.border_width
    inner_y = size_y - 2.0 * cfg.border_width
    num_steps = max(
        0,
        int(
            min(
                (inner_x - cfg.platform_width) / (2.0 * cfg.step_width),
                (inner_y - cfg.platform_width) / (2.0 * cfg.step_width),
            )
        ),
    )
    total_height = (num_steps + 1) * step_height
    lowest_surface = -total_height if inverted else 0.0
    bottom = lowest_surface - cfg.base_thickness
    color_base = _RED if inverted else _BLUE
    geometries = _add_border(spec, cfg, cfg.border_width, bottom, (*color_base, 1.0))

    for ring_index in range(num_steps):
        outer_x_start = cfg.border_width + ring_index * cfg.step_width
        outer_x_end = size_x - cfg.border_width - ring_index * cfg.step_width
        outer_y_start = cfg.border_width + ring_index * cfg.step_width
        outer_y_end = size_y - cfg.border_width - ring_index * cfg.step_width
        inner_ring_x_start = outer_x_start + cfg.step_width
        inner_ring_x_end = outer_x_end - cfg.step_width
        inner_ring_y_start = outer_y_start + cfg.step_width
        inner_ring_y_end = outer_y_end - cfg.step_width
        surface = (ring_index + 1) * step_height
        if inverted:
            surface = -surface
        shade = 0.75 + 0.25 * (ring_index % 2)
        color = (*tuple(channel * shade for channel in color_base), 1.0)
        geometries.extend(
            (
                _add_rectangular_box(
                    spec, cfg, outer_x_start, outer_x_end, outer_y_start,
                    inner_ring_y_start, surface, bottom, color,
                ),
                _add_rectangular_box(
                    spec, cfg, outer_x_start, outer_x_end, inner_ring_y_end,
                    outer_y_end, surface, bottom, color,
                ),
                _add_rectangular_box(
                    spec, cfg, outer_x_start, inner_ring_x_start, inner_ring_y_start,
                    inner_ring_y_end, surface, bottom, color,
                ),
                _add_rectangular_box(
                    spec, cfg, inner_ring_x_end, outer_x_end, inner_ring_y_start,
                    inner_ring_y_end, surface, bottom, color,
                ),
            )
        )

    center_x_start = cfg.border_width + num_steps * cfg.step_width
    center_x_end = size_x - center_x_start
    center_y_start = cfg.border_width + num_steps * cfg.step_width
    center_y_end = size_y - center_y_start
    center_surface = total_height if not inverted else -total_height
    center_color = (*color_base, 1.0)
    geometries.append(
        _add_rectangular_box(
            spec, cfg, center_x_start, center_x_end, center_y_start, center_y_end,
            center_surface, bottom, center_color,
        )
    )
    origin = np.array((size_x / 2.0, size_y / 2.0, center_surface))
    return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class PyramidStairsTerrainCfg(BlindTerrainCfg):
    """中心升高、向外围收缩的同心方环楼梯配置。"""

    step_height_range: tuple[float, float] = (0.0, 0.18)
    """难度从 0 到 1 对应的单级高度范围，单位为米。"""
    step_width: float = 0.35
    """每一圈台阶的宽度，单位为米。"""
    platform_width: float = 1.5
    """中心平台的最小边长，单位为米。"""
    border_width: float = 0.5
    """外围零高度平框的宽度，单位为米。"""

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        """生成中心真实表面为正高度的有限金字塔楼梯。"""
        del rng
        return _make_pyramid_stairs(self, difficulty, spec, inverted=False)


@dataclass(kw_only=True)
class InvertedPyramidStairsTerrainCfg(PyramidStairsTerrainCfg):
    """中心下沉、向外围升高的同心方环楼梯配置。"""

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        """生成中心真实表面为负高度的有限倒金字塔楼梯。"""
        del rng
        return _make_pyramid_stairs(self, difficulty, spec, inverted=True)


@dataclass(kw_only=True)
class StairsTerrainCfg(BlindTerrainCfg):
    """直线楼梯配置；长度、高度与步宽的单位均为米。"""

    descending: bool = False
    """为真时沿 +X 从高平台下行到零高度，否则从零高度上行。"""
    step_height_range: tuple[float, float] = (0.0, 0.18)
    """难度从 0 到 1 对应的非负单级高度范围。"""
    step_width: float = 0.35
    """每级踏面的水平长度；末端平台吸收不足一整级的余量。"""

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        """用相邻有限盒体构建完整地块，并返回起点表面原点。"""
        del rng
        validate_patch(self, difficulty)
        validate_range(self.step_height_range, "step_height_range")
        run = self.size[0] - 2 * self.platform_length
        if not math.isfinite(self.step_width) or not 0 < self.step_width <= run:
            raise ValueError("step_width must be finite, positive, and no larger than the stair run.")

        step_count = math.floor(run / self.step_width)
        step_height = self.step_height_range[0] + difficulty * (
            self.step_height_range[1] - self.step_height_range[0]
        )
        total_height = step_count * step_height
        start_height = total_height if self.descending else 0.0
        base_color = np.array((0.30, 0.55, 0.66))
        geometries = []

        # 每段只覆盖自己的 X 区间，共用底面；即使级高为零也保留实体厚度。
        edges = [0.0, self.platform_length]
        edges.extend(self.platform_length + np.arange(1, step_count + 1) * self.step_width)
        edges.append(self.size[0])
        levels = [0, *range(1, step_count + 1), step_count]
        for x_start, x_end, level in zip(edges[:-1], edges[1:], levels, strict=True):
            height = level * step_height
            if self.descending:
                height = total_height - height
            shade = 0.84 + 0.16 * (level % 2)
            color = (*map(float, base_color * shade), 1.0)
            geometries.append(
                add_box(spec, self, x_start, x_end, height, -self.base_thickness, color)
            )

        origin = np.array((self.platform_length / 2, self.size[1] / 2, start_height))
        return TerrainOutput(origin=origin, geometries=geometries)
