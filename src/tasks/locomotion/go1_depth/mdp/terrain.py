# Copyright (c) 2026-2027 zh
"""内容：由真实碰撞几何采样静态高度栅格，以可微双线性插值服务Go1想象与高度监督。"""

import math

import mujoco
import numpy as np
import torch
import torch.nn.functional as functional


class Go1Terrain:
    """栅格近似包含mesh坡、楼梯与高度场；地图外采用边界延拓且单独判定越界失败。"""

    def __init__(self, model, cfg, device):
        self.flat = cfg.terrain == "flat"
        self.device = torch.device(device)
        self.margin = cfg.terrain_margin
        self.resolution = cfg.terrain_resolution
        self.half_extent = None
        self.grid = None
        if self.flat:
            return
        half = np.asarray((cfg.terrain_rows * 4.0, 16.0))
        self.half_extent = torch.as_tensor(half, dtype=torch.float32, device=device)
        self.sample_extent = self.half_extent - 1e-4
        xs = np.linspace(-half[0] + 1e-4, half[0] - 1e-4, math.ceil(2 * half[0] / self.resolution) + 1)
        ys = np.linspace(-half[1] + 1e-4, half[1] - 1e-4, math.ceil(2 * half[1] / self.resolution) + 1)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        terrain_ids = [index for index in range(model.ngeom)
                       if model.body(model.geom_bodyid[index]).name in ("terrain", "terrain/terrain")]
        if not terrain_ids:
            raise ValueError("Compiled Go1 scene has no terrain collision geometry")
        if any(model.geom_group[index] != 0 for index in terrain_ids):
            raise ValueError("Go1 terrain ray sampling requires terrain geometry group 0")
        self.ray_top = float(np.max(data.geom_xpos[terrain_ids, 2] + model.geom_rbound[terrain_ids]) + 1)
        direction = np.asarray((0.0, 0.0, -1.0))
        groups = np.asarray((1, 0, 0, 0, 0, 0), dtype=np.uint8)
        geom_id = np.zeros(1, dtype=np.int32)
        point = np.asarray((0.0, 0.0, self.ray_top))
        heights = np.empty((len(ys), len(xs)), dtype=np.float32)
        for row, y in enumerate(ys):
            point[1] = y
            for column, x in enumerate(xs):
                point[0] = x
                distance = mujoco.mj_ray(model, data, point, direction, groups, True, -1, geom_id)
                if distance < 0:
                    raise ValueError(f"Terrain ray missed inside the declared map at ({x}, {y})")
                heights[row, column] = self.ray_top - distance
        self.grid = torch.from_numpy(heights).to(device)[None, None]

    def height(self, xy):
        if self.flat:
            return xy.sum(-1) * 0
        normalized = (xy / self.sample_extent).clamp(-1, 1)
        values = functional.grid_sample(self.grid, normalized.reshape(1, 1, -1, 2), mode="bilinear",
                                        padding_mode="border", align_corners=True)
        return values.reshape(xy.shape[:-1])

    def boundary_margin(self, xy):
        if self.flat:
            return torch.ones_like(xy[..., 0]) * 1000.0
        return (self.half_extent - xy.abs()).amin(-1) - self.margin
