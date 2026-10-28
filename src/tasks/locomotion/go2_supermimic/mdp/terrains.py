# Copyright (c) 2026-2027 zh
"""
内容：
    装配指定参考地形，从碰撞几何生成0.02米栅格并执行可微双线性查询。
    初始化几何查询使用MuJoCo高度场三角表面；运行时与来源一样在栅格域外夹取。
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy

import mujoco
import numpy as np
import torch
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg


def reference_generator(cfg):
    """逐项复现go2-supermimic env/terrains.py的ROUGH_TERRAINS。"""
    from mjlab.terrains.config import (
        flat, pyramid_stairs, pyramid_stairs_inv, box_random_grid, random_rough,
        hf_pyramid_slope, hf_pyramid_slope_inv,
    )
    return TerrainGeneratorCfg(
        seed=cfg.seed, size=tuple(cfg.patch_size_m), border_width=cfg.outer_flat_width_m,
        num_rows=cfg.core_rows, num_cols=cfg.num_cols, curriculum=True, color_scheme="none",
        sub_terrains={
            "flat": flat(proportion=0.10),
            "stairs": pyramid_stairs(proportion=0.20, step_height_range=(0.02, 0.10), step_width=0.4, platform_width=1.5),
            "stairs_inv": pyramid_stairs_inv(proportion=0.20, step_height_range=(0.02, 0.10), step_width=0.4, platform_width=1.5),
            "boxes": box_random_grid(proportion=0.15, grid_width=0.45, grid_height_range=(0.02, 0.05), platform_width=1.0),
            "rough": random_rough(proportion=0.15, noise_range=(0.01, 0.04), noise_step=0.01),
            "slope": hf_pyramid_slope(proportion=0.10, slope_range=(0.0, 0.3)),
            "slope_inv": hf_pyramid_slope_inv(proportion=0.10, slope_range=(0.0, 0.3)),
        }, add_lights=False,
    )


def clean_look(terrain):
    """保留参考地面的材质、背景与补光。"""
    from mjlab.utils.spec_config import LightCfg, MaterialCfg, TextureCfg
    terrain.textures = (
        TextureCfg(name="sky", type="skybox", builtin="gradient", rgb1=(.93,.95,.98), rgb2=(.62,.74,.87), width=512, height=3072),
        TextureCfg(name="grid", type="2d", builtin="checker", mark="edge", rgb1=(.66,.66,.65), rgb2=(.62,.62,.61),
                   markrgb=(.46,.47,.50), width=512, height=512),
    )
    terrain.materials = (MaterialCfg(name="ground", texture="grid", texuniform=True, texrepeat=(1.,1.), reflectance=0., geom_names_expr=(".*",)),)
    terrain.lights = (LightCfg(name="fill", type="directional", pos=(0.,0.,10.), dir=(.3,.2,-1.), diffuse=(.12,.12,.12),
                              specular=(0.,0.,0.), ambient=(.12,.13,.15), castshadow=False),)
    return terrain


class TerrainMap:
    """参考的七种地形各占一列，五行按生成器difficulty递增。"""

    def __init__(self, cfg):
        self.cfg = deepcopy(cfg)
        self.plane = cfg.mode == "flat"
        self.size = tuple(cfg.patch_size_m)
        self.generator = None if self.plane else reference_generator(cfg)
        self.family_names = ("flat",) if self.plane else tuple(self.generator.sub_terrains)
        self.proportions = (1.,) if self.plane else tuple(term.proportion for term in self.generator.sub_terrains.values())
        self.rows, self.cols = (1, 1) if self.plane else (cfg.core_rows, cfg.num_cols)
        self.level_counts = (self.rows,) * self.cols
        self.row_offset = self.rings = 0
        self.support_half_extent = np.asarray((self.rows, self.cols)) * self.size / 2 + cfg.outer_flat_width_m
        self.origins = None

    def entity_cfg(self):
        if self.plane:
            return clean_look(TerrainEntityCfg(terrain_type="plane"))
        return clean_look(TerrainEntityCfg(terrain_type="generator", terrain_generator=deepcopy(self.generator), max_init_terrain_level=None))

    def bind(self, origins):
        if self.plane:
            return
        if origins is None or tuple(origins.shape) != (self.rows, self.cols, 3) or not torch.isfinite(origins).all():
            raise ValueError("Compiled terrain origins differ from the reference map")
        self.origins = origins.detach().clone()

    def spawn_origins(self, family_ids, levels):
        if self.plane:
            raise ValueError("Plane origins are allocated by the environment")
        return self.origins[levels, family_ids].clone()


class PlaneSurface:
    """无界平面，前向保留数值为零的位置导数。"""

    def __init__(self, device="cpu", height=0.0):
        if not math.isfinite(height):
            raise ValueError("Plane height must be finite")
        self.device = torch.device(device)
        self.elevation = float(height)
        self.geometry_hash = hashlib.sha256(f"plane/{self.elevation}".encode()).hexdigest()

    def contains(self, xy, margin=0.0):
        if margin < 0 or not math.isfinite(margin):
            raise ValueError("Terrain query margin must be finite and nonnegative")
        return torch.isfinite(xy).all(dim=-1)

    def height(self, xy):
        if xy.shape[-1] != 2 or not torch.isfinite(xy).all():
            raise ValueError("Terrain query must contain finite XY points")
        return self.height_unchecked(xy)

    def height_unchecked(self, xy):
        """由调用边界保证输入有效的纯张量版本，供编译想象使用。"""
        return xy.sum(-1) * 0 + self.elevation


class TerrainGeometry:
    """按地块选择静态碰撞表面；查询热路径只使用Torch。"""

    QUERY_CHUNK_SIZE = 8192

    @classmethod
    def from_model(cls, model, device="cpu", terrain_map=None):
        if terrain_map is None:
            raise ValueError("TerrainSurface requires the compiled terrain map")
        if terrain_map.plane:
            return PlaneSurface(device)
        self = cls()
        self.device = torch.device(device)
        self.rows, self.cols = terrain_map.rows, terrain_map.cols
        self.size = torch.tensor(terrain_map.size, dtype=torch.float32, device=device)
        self.half_grid = self.size * self.size.new_tensor((self.rows, self.cols)) / 2
        self.half_support = self.size.new_tensor(terrain_map.support_half_extent)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        terrain_ids = []
        for geom in range(model.ngeom):
            body = model.body(model.geom_bodyid[geom]).name
            if body == "terrain" or body.endswith("/terrain"):
                terrain_ids.append(geom)
        if not terrain_ids:
            raise ValueError("Compiled model has no terrain collision geometry")
        bucket_count = self.rows * self.cols + 1
        boxes = [[] for _ in range(bucket_count)]
        fields = [None] * bucket_count
        digest = hashlib.sha256()
        digest.update(json.dumps({"rows": self.rows, "cols": self.cols, "size": terrain_map.size,
                                  "support": terrain_map.support_half_extent.tolist()}, sort_keys=True).encode())
        half = np.asarray((self.rows, self.cols)) * terrain_map.size / 2
        for geom in terrain_ids:
            kind = model.geom_type[geom]
            position = data.geom_xpos[geom].copy()
            matrix = data.geom_xmat[geom].reshape(3, 3)
            if not np.allclose(matrix, np.eye(3), atol=1e-6):
                raise ValueError("Static terrain query only supports axis-aligned boxes and heightfields")
            dimensions = model.geom_size[geom].copy()
            digest.update(np.asarray([kind], dtype=np.int64).tobytes())
            digest.update(position.tobytes())
            digest.update(dimensions.tobytes())
            if kind == mujoco.mjtGeom.mjGEOM_BOX:
                lower, upper = position[:2] - dimensions[:2], position[:2] + dimensions[:2]
                first = np.floor((lower + half + 1e-8) / terrain_map.size).astype(int)
                last = np.floor((upper + half - 1e-8) / terrain_map.size).astype(int)
                value = (*position[:2], *dimensions[:2], position[2] + dimensions[2])
                for row in range(max(0, first[0]), min(self.rows - 1, last[0]) + 1):
                    for col in range(max(0, first[1]), min(self.cols - 1, last[1]) + 1):
                        boxes[row * self.cols + col].append(value)
                if (lower < -half).any() or (upper > half).any():
                    boxes[-1].append(value)
            elif kind == mujoco.mjtGeom.mjGEOM_HFIELD:
                field_id = model.geom_dataid[geom]
                nrow, ncol = int(model.hfield_nrow[field_id]), int(model.hfield_ncol[field_id])
                address = model.hfield_adr[field_id]
                heights = model.hfield_data[address:address + nrow * ncol].copy()
                field_size = model.hfield_size[field_id].copy()
                digest.update(field_size.tobytes())
                digest.update(heights.tobytes())
                cell = np.floor((position[:2] + half) / terrain_map.size).astype(int)
                if ((cell < 0) | (cell >= (self.rows, self.cols))).any():
                    raise ValueError("Heightfield centre lies outside the terrain map")
                bucket = cell[0] * self.cols + cell[1]
                if fields[bucket] is not None:
                    raise ValueError("Each terrain tile must contain at most one heightfield")
                fields[bucket] = (position, field_size, nrow, ncol, heights)
            else:
                raise ValueError(f"Unsupported terrain collision geometry: {kind}")
        self.geometry_hash = digest.hexdigest()
        max_boxes = max(max(map(len, boxes)), 1)
        box_data = np.zeros((bucket_count, max_boxes, 5), dtype=np.float64)
        box_data[..., 2:4] = -1
        box_data[..., 4] = -np.inf
        for bucket, values in enumerate(boxes):
            if values:
                box_data[bucket, :len(values)] = values
        self.boxes = torch.from_numpy(box_data).to(device)
        max_samples = max([entry[2] * entry[3] for entry in fields if entry is not None] + [4])
        field_values = np.zeros((bucket_count, max_samples), dtype=np.float64)
        field_info = np.zeros((bucket_count, 7), dtype=np.float64)
        field_info[:, 2:4] = 1
        field_rows = np.full((bucket_count, 2), 2, dtype=np.int64)
        field_present = np.zeros(bucket_count, dtype=bool)
        for bucket, entry in enumerate(fields):
            if entry is None:
                continue
            position, sizes, nrow, ncol, heights = entry
            field_values[bucket, :len(heights)] = heights
            field_info[bucket] = (*position[:2], *sizes[:2], position[2], sizes[2], 1)
            field_rows[bucket] = nrow, ncol
            field_present[bucket] = True
        self.field_values = torch.from_numpy(field_values).to(device)
        self.field_info = torch.from_numpy(field_info).to(device)
        self.field_shapes = torch.from_numpy(field_rows).to(device)
        self.field_present = torch.from_numpy(field_present).to(device)
        return self

    def contains(self, xy, margin=0.0):
        if margin < 0 or not math.isfinite(margin):
            raise ValueError("Terrain query margin must be finite and nonnegative")
        return torch.isfinite(xy).all(-1) & (xy.abs() <= self.half_support - margin).all(-1)

    def rasterize(self, xs, ys):
        """按轴对齐碰撞几何填充来源射线栅格，不逐点广播全部箱体。"""
        heights = torch.full((ys.numel(), xs.numel()), -torch.inf, dtype=torch.float64)
        # 同一箱体可同时登记在相邻地块，仅在初始化去重一次。
        boxes = torch.unique(self.boxes.reshape(-1, 5), dim=0)
        boxes = boxes[boxes[:, 2] >= 0]
        for box in boxes:
            cx, cy, hx, hy, top = box.tolist()
            x0 = int(torch.searchsorted(xs, torch.tensor(cx - hx, dtype=xs.dtype)))
            x1 = int(torch.searchsorted(xs, torch.tensor(cx + hx, dtype=xs.dtype), right=True))
            y0 = int(torch.searchsorted(ys, torch.tensor(cy - hy, dtype=ys.dtype)))
            y1 = int(torch.searchsorted(ys, torch.tensor(cy + hy, dtype=ys.dtype), right=True))
            region = heights[y0:y1, x0:x1]
            region.copy_(torch.maximum(region, region.new_tensor(top)))
        for bucket in self.field_present.nonzero().flatten().tolist():
            info = self.field_info[bucket]
            x0, y0 = (int(torch.searchsorted(axis, info[dim] - info[dim + 2])) for dim, axis in enumerate((xs, ys)))
            x1, y1 = (int(torch.searchsorted(axis, info[dim] + info[dim + 2], right=True)) for dim, axis in enumerate((xs, ys)))
            gx, gy = torch.meshgrid(xs[x0:x1], ys[y0:y1], indexing="xy")
            points = torch.stack((gx.flatten(), gy.flatten()), -1)
            # 此处查询同样使用三角插值，对应MuJoCo高度场碰撞的表面。
            values = torch.cat([self._field_heights(chunk, bucket) for chunk in points.split(65536)])
            region = heights[y0:y1, x0:x1]
            region.copy_(torch.maximum(region, values.reshape(region.shape)))
        return torch.where(torch.isfinite(heights), heights, 0).float()

    def _field_heights(self, points, bucket):
        info, sizes = self.field_info[bucket], self.field_shapes[bucket]
        uv = ((points - info[:2] + info[2:4]) / (2 * info[2:4])).clamp(0, 1)
        coordinate = uv * (sizes.flip(-1) - 1)
        index = torch.minimum(coordinate.floor().long(), sizes.flip(-1) - 2)
        fx, fy = (coordinate - index).unbind(-1)
        c, r = index.unbind(-1)
        width = sizes[1]
        z00, z01 = self.field_values[bucket, r * width + c], self.field_values[bucket, r * width + c + 1]
        z10, z11 = self.field_values[bucket, (r + 1) * width + c], self.field_values[bucket, (r + 1) * width + c + 1]
        value = torch.where(fx >= fy, z00 + fx * (z01 - z00) + fy * (z11 - z01),
                            z00 + fx * (z11 - z10) + fy * (z10 - z00))
        return info[4] + info[5] * value

    def height(self, xy):
        if xy.shape[-1] != 2 or xy.device != self.device:
            raise ValueError("Terrain XY points must use the surface device")
        if not self.contains(xy).all():
            raise ValueError("Terrain query lies outside the supported surface")
        result = self.height_unchecked(xy)
        if not torch.isfinite(result).all():
            raise ValueError("Terrain geometry does not cover a requested XY point")
        return result

    def height_unchecked(self, xy):
        """已验证表面上的纯张量查询；单策略任务在外层屏蔽无支持点。"""
        shape = xy.shape[:-1]
        points = xy.reshape(-1, 2)
        if points.shape[0] == 0:
            return points.sum(-1).reshape(shape)
        # 分块限制箱体临时张量，保持原严格查询的数值和位置梯度。
        result = torch.cat([self._height_points(chunk) for chunk in points.split(self.QUERY_CHUNK_SIZE)], dim=0)
        return result.reshape(shape)

    def _height_points(self, points):
        """计算一块已验证的平面坐标，不重复执行全局边界检查或设备同步。"""
        cell = torch.floor((points + self.half_grid) / self.size).long()
        in_grid = ((cell >= 0) & (cell < cell.new_tensor((self.rows, self.cols)))).all(-1)
        bucket = torch.where(in_grid, cell[:, 0] * self.cols + cell[:, 1], self.rows * self.cols)
        boxes = self.boxes[bucket]
        box_inside = ((points[:, None, :] - boxes[..., :2]).abs() <= boxes[..., 2:4]).all(-1)
        box_height = torch.where(box_inside, boxes[..., 4], -torch.inf).amax(-1)
        info = self.field_info[bucket]
        sizes = self.field_shapes[bucket]
        uv = (points - info[:, :2] + info[:, 2:4]) / (2 * info[:, 2:4])
        field_inside = self.field_present[bucket] & ((uv >= 0) & (uv <= 1)).all(-1)
        # 高度场ncol对应x，nrow对应y；仅地址取整，插值分数保留位置导数。
        coordinate = uv.clamp(0, 1) * (sizes.flip(-1) - 1)
        index = torch.floor(coordinate).long()
        index = torch.minimum(index, sizes.flip(-1) - 2)
        fraction = coordinate - index
        c, r = index.unbind(-1)
        fx, fy = fraction.unbind(-1)
        width = sizes[:, 1]
        z00 = self.field_values[bucket, r * width + c]
        z01 = self.field_values[bucket, r * width + c + 1]
        z10 = self.field_values[bucket, (r + 1) * width + c]
        z11 = self.field_values[bucket, (r + 1) * width + c + 1]
        interpolated = torch.where(fx >= fy, z00 + fx * (z01 - z00) + fy * (z11 - z01),
                                   z00 + fx * (z11 - z10) + fy * (z10 - z00))
        field_height = torch.where(field_inside, info[:, 4] + info[:, 5] * interpolated, -torch.inf)
        return torch.maximum(box_height, field_height) + points.sum(-1) * 0


class TerrainSurface:
    """参考0.02米高度栅格：双线性插值及域外clamp，与来源采样同义。"""

    @classmethod
    def from_model(cls, model, device="cpu", terrain_map=None):
        if terrain_map.plane:
            return PlaneSurface(device)
        self = cls()
        self.device = torch.device(device)
        cfg = terrain_map.cfg
        origins = terrain_map.origins.reshape(-1, 3).cpu().numpy()
        self.origin = (float(origins[:, 0].min() - cfg.field_half_m), float(origins[:, 1].min() - cfg.field_half_m))
        upper = (float(origins[:, 0].max() + cfg.field_half_m), float(origins[:, 1].max() + cfg.field_half_m))
        self.res = cfg.field_resolution_m
        geometry = TerrainGeometry.from_model(model, terrain_map=terrain_map)
        payload = f"reference_heightfield_v1/{geometry.geometry_hash}/{self.origin}/{upper}/{self.res}"
        self.geometry_hash = hashlib.sha256(payload.encode()).hexdigest()
        from pathlib import Path
        path = Path(cfg.field_cache_dir) / f"heightfield_blind_{self.geometry_hash[:12]}.pt"
        if path.exists():
            self.h = torch.load(path, map_location=self.device, weights_only=True)["h"]
            return self
        xs = torch.from_numpy(np.arange(self.origin[0], upper[0] + 1e-9, self.res))
        ys = torch.from_numpy(np.arange(self.origin[1], upper[1] + 1e-9, self.res))
        heights = geometry.rasterize(xs, ys)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"h": heights, "origin": self.origin, "res": self.res}, path)
        self.h = heights.to(self.device)
        return self

    def contains(self, xy, margin=0.0):
        """来源高度图对所有有限坐标有效，域外取边缘值。"""
        return torch.isfinite(xy).all(-1)

    def height(self, xy):
        if xy.shape[-1] != 2 or not torch.isfinite(xy).all():
            raise ValueError("Terrain queries must contain finite XY coordinates")
        return self.height_unchecked(xy)

    def height_unchecked(self, xy):
        gx = (xy[..., 0] - self.origin[0]) / self.res
        gy = (xy[..., 1] - self.origin[1]) / self.res
        height, width = self.h.shape
        gx, gy = gx.clamp(0, width - 1.001), gy.clamp(0, height - 1.001)
        x0, y0 = gx.floor().long(), gy.floor().long()
        fx, fy = gx - x0, gy - y0
        h00, h01 = self.h[y0, x0], self.h[y0, x0 + 1]
        h10, h11 = self.h[y0 + 1, x0], self.h[y0 + 1, x0 + 1]
        return (h00 * (1 - fx) + h01 * fx) * (1 - fy) + (h10 * (1 - fx) + h11 * fx) * fy
