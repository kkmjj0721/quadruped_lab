# Copyright (c) 2026-2027 zh
"""
内容：
    装配固定家族／等级的连通地形，并从编译几何建立可微静态表面查询。
    高度场使用三角插值；台阶边界为真实离散边界，不承诺边界处的光滑导数。
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from mjlab.terrains import (
    BoxFlatTerrainCfg, BoxInvertedPyramidStairsTerrainCfg, BoxPyramidStairsTerrainCfg,
    HfPyramidSlopedTerrainCfg, HfRandomUniformTerrainCfg, TerrainEntityCfg, TerrainGeneratorCfg,
)
from mjlab.terrains.terrain_generator import SubTerrainCfg


def _stable_seed(seed: int, key: str) -> int:
    """仅依据语义地址生成种子，不受Python进程随机hash影响。"""
    return int.from_bytes(hashlib.sha256(f"{seed}/{key}".encode()).digest()[:8], "little")


def _make_patch(family, parameters, size):
    params = dict(parameters)
    if family == "random_rough":
        amplitude = params.pop("height_max_m")
        return HfRandomUniformTerrainCfg(size=size, noise_range=(0.0, amplitude),
                                        scale_with_difficulty=False, **params)
    if family.startswith("stairs_"):
        height = params.pop("step_height_m")
        terrain_type = BoxInvertedPyramidStairsTerrainCfg if family.endswith("_up") else BoxPyramidStairsTerrainCfg
        return terrain_type(size=size, step_height_range=(height, height), **params)
    if family.startswith("slope_"):
        gradient = params.pop("gradient")
        if gradient == 0:
            return BoxFlatTerrainCfg(size=size)
        return HfPyramidSlopedTerrainCfg(size=size, slope_range=(gradient, gradient),
                                       inverted=family.endswith("_up"), **params)
    raise ValueError(f"Unsupported terrain family: {family}")


@dataclass(kw_only=True)
class ConnectedColumnCfg(SubTerrainCfg):
    """difficulty只选择物理行，几何参数自身已固定等级。"""

    tiles: tuple = ()
    seed: int = 0
    column: int = 0

    def function(self, difficulty, spec, rng):
        del rng
        row = round(difficulty * max(len(self.tiles) - 1, 1))
        if not 0 <= row < len(self.tiles) or not math.isclose(difficulty, row / max(len(self.tiles) - 1, 1), abs_tol=1e-7):
            raise ValueError("Invalid connected terrain row address")
        family, level, parameters = self.tiles[row]
        generator = np.random.default_rng(_stable_seed(self.seed, f"tile/{row}/{self.column}/{family}/{level}"))
        output = _make_patch(family, parameters, self.size).function(0.0, spec, generator)
        # 出生点须位于实际中心表面，不能沿用粗糙高度场的局部最大高度。
        for geometry in output.geometries:
            if geometry.hfield is None:
                continue
            heightfield = geometry.hfield
            data = np.asarray(heightfield.userdata).reshape(heightfield.nrow, heightfield.ncol)
            y, x = (heightfield.nrow - 1) / 2, (heightfield.ncol - 1) / 2
            r, c = int(y), int(x)
            r1, c1 = min(r + 1, heightfield.nrow - 1), min(c + 1, heightfield.ncol - 1)
            fy, fx = y - r, x - c
            z00, z01, z10, z11 = data[r, c], data[r, c1], data[r1, c], data[r1, c1]
            value = (z00 + fx * (z01 - z00) + fy * (z11 - z01) if fx >= fy else
                     z00 + fx * (z11 - z10) + fy * (z10 - z00))
            output.origin[2] = geometry.geom.pos[2] + heightfield.size[2] * value
        return output


class TerrainMap:
    """课程以家族逻辑等级索引核心区，不把外扩行或重复最高行当成新等级。"""

    def __init__(self, cfg):
        self.cfg = deepcopy(cfg)
        self.plane = cfg.mode == "flat"
        if cfg.mode not in ("flat", "rough"):
            raise ValueError("Terrain mode must be flat or rough")
        self.size = tuple(float(value) for value in cfg.patch_size_m)
        if len(self.size) != 2 or any(not math.isfinite(value) or value <= 0 for value in self.size):
            raise ValueError("Terrain patch dimensions must be finite and positive")
        self.family_names = ("flat",) if self.plane else tuple(cfg.levels)
        self.level_counts = (1,) if self.plane else tuple(len(cfg.levels[family]) for family in self.family_names)
        self.rings = self.row_offset = 0 if self.plane else cfg.surrounding_rings
        self.rows = 1 if self.plane else cfg.core_rows + 2 * self.rings
        self.cols = 1 if self.plane else len(self.family_names) + 2 * self.rings
        if self.rings < 0 or any(count <= 0 for count in self.level_counts) or (not self.plane and cfg.core_rows < max(self.level_counts)):
            raise ValueError("Terrain rows cannot omit configured family levels")
        self.support_half_extent = np.asarray((self.rows, self.cols)) * self.size / 2 + cfg.outer_flat_width_m
        if not math.isfinite(cfg.outer_flat_width_m) or cfg.outer_flat_width_m < 0:
            raise ValueError("Outer flat width must be finite and nonnegative")
        self.origins = None
        self.tiles = []
        if not self.plane:
            for row in range(self.rows):
                for col in range(self.cols):
                    family = self.family_names[(col - self.rings) % len(self.family_names)]
                    names = tuple(cfg.levels[family])
                    level = names[min(max(row - self.rings, 0), len(names) - 1)]
                    category = "stairs" if family.startswith("stairs_") else "slope" if family.startswith("slope_") else family
                    if category not in cfg.defaults:
                        raise ValueError(f"Missing terrain defaults for {family}")
                    parameters = {**deepcopy(cfg.defaults[category]), **deepcopy(cfg.levels[family][level])}
                    self._validate_parameters(family, parameters)
                    self.tiles.append({"row": row, "col": col, "family": family, "level": level, "parameters": parameters})

    def _validate_parameters(self, family, parameters):
        for name, value in parameters.items():
            if value is None or isinstance(value, bool):
                continue
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid terrain parameter {family}/{name}")
            if value == 0 and name not in ("border_width", "gradient"):
                raise ValueError(f"Terrain parameter must be positive: {family}/{name}")
        if 2 * parameters.get("border_width", 0) + parameters.get("platform_width", 0) >= min(self.size):
            raise ValueError("Terrain platform and border do not fit the patch")
        if "horizontal_scale" in parameters:
            horizontal = parameters["horizontal_scale"]
            border = parameters["border_width"]
            downsample = parameters.get("downsampled_scale") or horizontal
            if (border and border < horizontal) or downsample < horizontal or min(self.size) - 2 * border < 4 * downsample:
                raise ValueError("Invalid heightfield sampling scales")
        if family == "random_rough" and parameters["noise_step"] < parameters["vertical_scale"]:
            raise ValueError("Noise step must cover a vertical quantization unit")

    def entity_cfg(self):
        if self.plane:
            return TerrainEntityCfg(terrain_type="plane")
        columns = {
            f"connected_column_{col}": ConnectedColumnCfg(
                size=self.size, seed=self.cfg.seed, column=col,
                tiles=tuple((tile["family"], tile["level"], tile["parameters"])
                            for tile in self.tiles if tile["col"] == col))
            for col in range(self.cols)
        }
        return TerrainEntityCfg(
            terrain_type="generator", max_init_terrain_level=self.rows - 1,
            terrain_generator=TerrainGeneratorCfg(
                seed=self.cfg.seed, curriculum=True, size=self.size, num_rows=self.rows, num_cols=self.cols,
                difficulty_range=(0.0, 1.0), border_width=self.cfg.outer_flat_width_m, border_height=1.0,
                sub_terrains=columns))

    def bind(self, origins):
        if self.plane:
            if origins is not None:
                raise ValueError("Plane terrain must use per-environment grid origins")
            return
        if origins is None or tuple(origins.shape) != (self.rows, self.cols, 3) or not torch.isfinite(origins).all():
            raise ValueError("Compiled terrain origins do not match the connected map")
        expected = origins.new_tensor([[(row + 0.5 - self.rows / 2) * self.size[0],
                                       (col + 0.5 - self.cols / 2) * self.size[1]]
                                      for row in range(self.rows) for col in range(self.cols)])
        if not torch.allclose(origins[..., :2].reshape(-1, 2), expected, atol=1e-5):
            raise ValueError("Compiled terrain centres do not match the connected map")
        self.origins = origins.detach().clone()

    def spawn_origins(self, family_ids, levels):
        if self.plane:
            raise ValueError("Plane origins must be allocated by the environment")
        if self.origins is None:
            raise RuntimeError("Bind compiled terrain origins before sampling spawns")
        if family_ids.dtype != torch.long or levels.dtype != torch.long or family_ids.shape != levels.shape:
            raise ValueError("Terrain family and level indices must be matching int64 tensors")
        if family_ids.device != self.origins.device or levels.device != self.origins.device:
            raise ValueError("Terrain indices and origins must share a device")
        if ((family_ids < 0) | (family_ids >= len(self.family_names))).any():
            raise ValueError("Terrain family index out of range")
        counts = levels.new_tensor(self.level_counts)[family_ids]
        if ((levels < 0) | (levels >= counts)).any():
            raise ValueError("Terrain logical level out of range")
        return self.origins[levels + self.rings, family_ids + self.rings].clone()


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


class TerrainSurface:
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
        box_data = np.zeros((bucket_count, max_boxes, 5), dtype=np.float32)
        box_data[..., 2:4] = -1
        box_data[..., 4] = -np.inf
        for bucket, values in enumerate(boxes):
            if values:
                box_data[bucket, :len(values)] = values
        self.boxes = torch.from_numpy(box_data).to(device)
        max_samples = max([entry[2] * entry[3] for entry in fields if entry is not None] + [4])
        field_values = np.zeros((bucket_count, max_samples), dtype=np.float32)
        field_info = np.zeros((bucket_count, 7), dtype=np.float32)
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
        box_inside = ((points[:, None, :] - boxes[..., :2]).abs() <= boxes[..., 2:4] + 1e-6).all(-1)
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
