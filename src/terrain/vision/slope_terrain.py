# Copyright (c) 2026-2027 zh
"""
内容：
    按角度插值生成中心平台升高、同一地块包含上下坡的四面凸碰撞网格。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput

from ..blind.common import BlindTerrainCfg, add_box, validate_patch, validate_range


@dataclass(kw_only=True)
class PyramidSlopeTerrainCfg(BlindTerrainCfg):
    """四面平坡；角度单位为度，其余尺寸单位为米。"""

    angle_range_deg: tuple[float, float] = (0.0, 40.0)
    """难度两端的坡角，先线性插值角度，再换算为高差与水平距离之比。"""
    platform_width: float = 2.0
    """中心平坦平台的最小边长。"""
    border_width: float = 0.25
    """地块外围的零高度平框宽度，不参与坡角计算。"""

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        """生成真实斜面角度受配置控制的上坡、平台和下坡。"""
        del rng
        validate_patch(self, difficulty)
        validate_range(self.angle_range_deg, "angle_range_deg")
        if self.angle_range_deg[1] >= 90.0:
            raise ValueError("Slope angles must be less than 90 degrees.")
        if not math.isfinite(self.platform_width) or self.platform_width <= 0.0:
            raise ValueError("platform_width must be finite and positive.")
        if not math.isfinite(self.border_width) or self.border_width < 0.0:
            raise ValueError("border_width must be finite and non-negative.")

        half_x = self.size[0] / 2.0 - self.border_width
        half_y = self.size[1] / 2.0 - self.border_width
        run = min(half_x, half_y) - self.platform_width / 2.0
        if run <= 0.0:
            raise ValueError("platform_width must leave space for slopes inside the border.")

        angle = self.angle_range_deg[0] + difficulty * (self.angle_range_deg[1] - self.angle_range_deg[0])
        height = math.tan(math.radians(angle)) * run
        color = (0.57, 0.65, 0.44, 1.0)
        origin = np.array((self.size[0] / 2.0, self.size[1] / 2.0, height))
        if angle == 0.0:
            geometry = add_box(spec, self, 0.0, self.size[0], 0.0, -self.base_thickness, color)
            return TerrainOutput(origin=origin, geometries=[geometry])

        # 三圈顶点依次为底面、坡脚、平台；每个梯形的两个三角面严格共面。
        vertices = []
        for radius_x, radius_y, z in (
            (half_x, half_y, -self.base_thickness),
            (half_x, half_y, 0.0),
            (half_x - run, half_y - run, height),
        ):
            vertices.extend(((-radius_x, -radius_y, z), (radius_x, -radius_y, z), (radius_x, radius_y, z), (-radius_x, radius_y, z)))
        faces = [(0, 2, 1), (0, 3, 2), (8, 9, 10), (8, 10, 11)]
        for level in (0, 4):
            for side in range(4):
                first, second = level + side, level + (side + 1) % 4
                faces.extend(((first, second, second + 4), (first, second + 4, first + 4)))
        mesh = spec.add_mesh(
            name=f"vision_slope_{len(spec.meshes)}", uservert=np.asarray(vertices).ravel().tolist(),
            userface=np.asarray(faces).ravel().tolist(),
        )
        body = spec.body("terrain")
        geom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name,
            pos=(self.size[0] / 2.0, self.size[1] / 2.0, 0.0), friction=self.friction, rgba=color,
        )
        geometries = [TerrainGeometry(geom=geom, color=color)]
        if self.border_width > 0.0:
            size_x, size_y = self.size
            border = self.border_width
            for x0, x1, y0, y1 in (
                (0.0, border, 0.0, size_y), (size_x - border, size_x, 0.0, size_y),
                (border, size_x - border, 0.0, border), (border, size_x - border, size_y - border, size_y),
            ):
                geom = body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=((x0 + x1) / 2.0, (y0 + y1) / 2.0, -self.base_thickness / 2.0),
                    size=((x1 - x0) / 2.0, (y1 - y0) / 2.0, self.base_thickness / 2.0),
                    friction=self.friction, rgba=color,
                )
                geometries.append(TerrainGeometry(geom=geom, color=color))
        return TerrainOutput(origin=origin, geometries=geometries)
