# Copyright (c) 2026-2027 zh
"""
内容：
    将 MGDP 原始整数高程与独立悬空盒体装配为 MjLab 复合地形。
    高度场采用 MuJoCo 三角化碰撞；不宣称与 PhysX 简化网格接触等价。
"""

from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import (
    SubTerrainCfg, TerrainGeneratorCfg, TerrainGeometry, TerrainOutput,
)

from .generator import MgdpTerrainData


@dataclass(kw_only=True)
class MgdpCompositeTerrainCfg(SubTerrainCfg):
    data: MgdpTerrainData

    def function(self, difficulty, spec, rng) -> TerrainOutput:
        body = spec.body("terrain")
        geometries = []
        # MjLab 会统一减去半个复合场景尺寸；预先抵消以保留 data 的世界坐标。
        correction = np.array([self.size[0] / 2, self.size[1] / 2, 0.0])
        for family in self.data.families:
            raw = family.collision_height_raw
            low, high = int(raw.min()), int(raw.max())
            extent = max(high - low, 1) * family.vertical_scale
            # 上游数组按 [x,y]；MuJoCo hfield 按 [y,x] 存储。
            normalized = ((raw.T.astype(np.float64) - low) / max(high - low, 1)).astype(np.float32)
            span = (np.array(raw.shape) - 1) * family.horizontal_scale
            hfield = spec.add_hfield(
                name=f"mgdp_{family.name}_hfield", nrow=raw.shape[1], ncol=raw.shape[0],
                size=[span[0] / 2, span[1] / 2, extent, 0.1],
                userdata=normalized.ravel().tolist(),
            )
            position = family.world_offset + np.array([
                span[0] / 2 - family.border_size, span[1] / 2 - family.border_size,
                low * family.vertical_scale,
            ])
            geom = body.add_geom(
                name=f"mgdp_{family.name}_surface", type=mujoco.mjtGeom.mjGEOM_HFIELD,
                hfieldname=hfield.name, pos=position + correction,
                friction=[1.0, 0.005, 0.0001], rgba=[0.45, 0.48, 0.45, 1.0],
            )
            geometries.append(TerrainGeometry(geom=geom, hfield=hfield))
            for index, box in enumerate(family.boxes):
                geom = body.add_geom(
                    name=f"mgdp_{family.name}_{box.kind}_{index}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=box.size / 2, pos=box.center + correction,
                    friction=[1.0, 0.005, 0.0001], rgba=[0.5, 0.46, 0.4, 1.0],
                )
                geometries.append(TerrainGeometry(geom=geom))
        return TerrainOutput(origin=self.data.origins[0, 0] + correction, geometries=geometries)


def make_mgdp_terrain_cfg(data: MgdpTerrainData) -> TerrainGeneratorCfg:
    """用一个复合块装配选定地图；任务必须覆写真实 env origins 和课程。"""
    lower = np.stack([family.world_offset[:2] - family.border_size for family in data.families]).min(axis=0)
    upper = np.stack([
        family.world_offset[:2] + [family.cfg.num_rows * family.cfg.terrain_length, family.cfg.num_cols * family.cfg.terrain_width]
        + family.border_size for family in data.families
    ]).max(axis=0)
    return TerrainGeneratorCfg(
        seed=data.seed, size=tuple(upper - lower), num_rows=1, num_cols=1, curriculum=True,
        border_width=0.0, color_scheme="none", sub_terrains={"mgdp": MgdpCompositeTerrainCfg(data=data)},
    )
