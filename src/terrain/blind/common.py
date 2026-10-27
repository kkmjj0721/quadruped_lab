# Copyright (c) 2026-2027 zh
"""
内容：
    定义盲走地形的公共参数、边界检查与有限碰撞几何构造。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry


@dataclass(kw_only=True)
class BlindTerrainCfg(SubTerrainCfg):
    """沿局部 +X 前进的地块，尺寸和厚度单位为米。"""

    size: tuple[float, float] = (8.0, 8.0)
    platform_length: float = 1.5
    base_thickness: float = 0.15
    friction: tuple[float, float, float] = (0.8, 0.005, 0.0001)


def validate_patch(cfg: BlindTerrainCfg, difficulty: float) -> None:
    """在生成前检查公共尺寸和难度，防止无效碰撞几何。"""
    if not math.isfinite(difficulty) or not 0.0 <= difficulty <= 1.0:
        raise ValueError("Difficulty must be finite and in [0, 1].")
    if len(cfg.size) != 2 or any(not math.isfinite(v) or v <= 0 for v in cfg.size):
        raise ValueError("Terrain size must contain two finite positive lengths.")
    if not math.isfinite(cfg.platform_length) or not 0 < 2 * cfg.platform_length < cfg.size[0]:
        raise ValueError("Platform length must be positive and leave space between both platforms.")
    if not math.isfinite(cfg.base_thickness) or cfg.base_thickness <= 0:
        raise ValueError("Base thickness must be finite and positive.")
    if len(cfg.friction) != 3 or any(not math.isfinite(v) or v < 0 for v in cfg.friction):
        raise ValueError("Friction must contain three finite non-negative values.")
    if not math.isfinite(cfg.proportion) or cfg.proportion < 0:
        raise ValueError("Terrain proportion must be finite and non-negative.")
    if cfg.flat_patch_sampling is not None:
        raise ValueError("Blind terrains use a fixed start platform; flat_patch_sampling is not supported.")


def validate_range(values: tuple[float, float], name: str) -> None:
    """检查由难度插值的非负数值区间。"""
    if len(values) != 2 or any(not math.isfinite(v) for v in values) or not 0 <= values[0] <= values[1]:
        raise ValueError(f"{name} must contain two finite values with 0 <= lower <= upper.")


def heightfield_axes(size: tuple[float, float], horizontal_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """生成包含边缘的采样轴，实际间距不大于指定分辨率。"""
    if not math.isfinite(horizontal_scale) or horizontal_scale <= 0:
        raise ValueError("Horizontal scale must be finite and positive.")
    counts = [max(2, math.ceil(length / horizontal_scale) + 1) for length in size]
    if counts[0] * counts[1] > 4_000_000:
        raise ValueError("Heightfield exceeds 4,000,000 samples; increase horizontal_scale.")
    return np.linspace(0.0, size[0], counts[0]), np.linspace(0.0, size[1], counts[1])


def add_box(
    spec: mujoco.MjSpec, cfg: BlindTerrainCfg, x_start: float, x_end: float,
    top: float, bottom: float, color: tuple[float, float, float, float],
) -> TerrainGeometry:
    """添加覆盖整个地块宽度的有限盒体，避免无限平面遮盖低处。"""
    geom = spec.body("terrain").add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=((x_start + x_end) / 2, cfg.size[1] / 2, (top + bottom) / 2),
        size=((x_end - x_start) / 2, cfg.size[1] / 2, (top - bottom) / 2),
        friction=cfg.friction, rgba=color,
    )
    return TerrainGeometry(geom=geom, color=color)


def add_heightfield(
    spec: mujoco.MjSpec, cfg: BlindTerrainCfg, heights: np.ndarray,
    color: tuple[float, float, float, float],
) -> TerrainGeometry:
    """将行 Y、列 X 的米制高度转换为保持物理高度的 MuJoCo 高度图。"""
    heights = np.asarray(heights, dtype=np.float64)
    if heights.ndim != 2 or min(heights.shape) < 2 or not np.isfinite(heights).all():
        raise ValueError("Heights must be a finite 2D array with at least two rows and columns.")
    minimum = float(heights.min())
    # MuJoCo 会再次归一化非恒定数据，仅常高图可用正数占位，避免放大小幅起伏。
    elevation = float(np.ptp(heights)) or 1e-6
    normalized = (heights - minimum) / elevation
    # 使用确定性命名，同一 spec 内新增多地块时不发生资源重名。
    name = f"blind_hfield_{len(spec.hfields)}"
    field = spec.add_hfield(
        name=name, nrow=heights.shape[0], ncol=heights.shape[1],
        size=(cfg.size[0] / 2, cfg.size[1] / 2, elevation, cfg.base_thickness),
        userdata=normalized.astype(np.float32).ravel().tolist(),
    )
    geom = spec.body("terrain").add_geom(
        type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=name,
        pos=(cfg.size[0] / 2, cfg.size[1] / 2, minimum),
        friction=cfg.friction, rgba=color,
    )
    return TerrainGeometry(geom=geom, hfield=field, color=color)
