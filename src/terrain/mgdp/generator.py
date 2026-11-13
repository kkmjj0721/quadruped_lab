# Copyright (c) 2026-2027 zh
"""
内容：
    按 MGDP f43139d 原版列选择、难度和随机流生成 mix 与 parkour 地形。
    原碰撞高程、悬空实体和感知标签独立保留，供 MjLab 适配器装配。
"""

from dataclasses import dataclass
from numbers import Integral
import random

import numpy as np
from scipy.ndimage import binary_dilation

from ._parkour import ParkourGeometry
from ._primitives import HeightGeometry, HeightPatch


MGDP_COMMIT = "f43139d1d97849819a0dce609dc0b59faf577b3f"


@dataclass(frozen=True)
class NativeTerrainCfg:
    name: str
    num_rows: int
    terrain_length: float
    terrain_width: float
    horizontal_scale: float
    border_size: float
    proportions: tuple[float, ...]
    num_cols: int = 10
    vertical_scale: float = 0.005
    slope_threshold: float = 0.75
    edge_width: float = 0.05


MIX_CFG = NativeTerrainCfg(
    "mix", 20, 8.0, 8.0, 0.1, 25.0,
    (0.2, 0.2, 0.2, 0.2, 1.1, 0.2, 1.2, 1.1, 0.0, 0.3, 1.0),
)
PARKOUR_CFG = NativeTerrainCfg(
    "parkour", 10, 10.0, 4.0, 0.05, 5.0,
    (0.0, 0.0, 0.0, 0.002, 0.101, 0.101, 0.0, 0.101, 0.101,
     0.0, 0.0, 0.0, 0.0, 0.101, 0.101, 0.101, 0.101, 1.1),
)


@dataclass(frozen=True)
class MgdpBox:
    kind: str
    vertices: np.ndarray
    """原版 box mesh 的八个顶点，已经转换到最终世界坐标。"""
    center: np.ndarray
    size: np.ndarray


@dataclass(frozen=True)
class MgdpFamilyData:
    cfg: NativeTerrainCfg
    world_offset: np.ndarray
    collision_height_raw: np.ndarray
    label_height_raw: np.ndarray
    edge_mask: np.ndarray
    origins: np.ndarray
    classes: np.ndarray
    boxes: tuple[MgdpBox, ...]
    goals_narrow: np.ndarray

    @property
    def name(self):
        return self.cfg.name

    @property
    def horizontal_scale(self):
        return self.cfg.horizontal_scale

    @property
    def vertical_scale(self):
        return self.cfg.vertical_scale

    @property
    def border_size(self):
        return self.cfg.border_size


@dataclass(frozen=True)
class MgdpTerrainData:
    seed: int
    families: tuple[MgdpFamilyData, ...]
    origins: np.ndarray
    family_by_column: np.ndarray
    class_by_column: np.ndarray
    level_counts: np.ndarray
    patch_sizes: np.ndarray


def native_edge_mask(height_raw: np.ndarray, cfg: NativeTerrainCfg) -> np.ndarray:
    """只计算官方网格转换器的 move_x，再沿 x 做原版宽度膨胀。"""
    threshold = cfg.slope_threshold * cfg.horizontal_scale / cfg.vertical_scale
    move_x = np.zeros(height_raw.shape)
    move_x[:-1] += height_raw[1:] - height_raw[:-1] > threshold
    move_x[1:] -= height_raw[:-1] - height_raw[1:] > threshold
    half_width = int(cfg.edge_width / cfg.horizontal_scale)
    return binary_dilation(move_x != 0, structure=np.ones((2 * half_width + 1, 1)))


class NativeGeometry(HeightGeometry, ParkourGeometry):
    """每家族使用独立同种子 RNG；不污染调用者的全局随机流。"""

    def __init__(self, seed: int):
        self.np_random = np.random.RandomState(seed)
        self.py_random = random.Random(seed)

    def make_patch(self, cfg: NativeTerrainCfg, row: int, col: int) -> HeightPatch:
        difficulty = row / cfg.num_rows
        choice = col / cfg.num_cols + 0.001
        # 保留官方逐前缀 np.sum，既不归一化，也不调整不可达配置权重。
        proportions = [np.sum(cfg.proportions[:i + 1]) for i in range(len(cfg.proportions))]
        terrain = HeightPatch(
            int(cfg.terrain_length / cfg.horizontal_scale),
            int(cfg.terrain_width / cfg.horizontal_scale), cfg.horizontal_scale, cfg.vertical_scale,
        )
        if cfg.name == "mix":
            self._make_mix(terrain, choice, difficulty, proportions)
        else:
            self._make_parkour(terrain, choice, difficulty, proportions, cfg.num_rows)
        return terrain

    def _make_mix(self, terrain, choice, difficulty, proportions):
        self.py_random.uniform(0.5, 0.5)
        slope, rise = difficulty * 0.4, 0.05 + 0.18 * difficulty
        if choice < proportions[0]:
            self.pyramid_sloped_terrain(terrain, -slope if choice < proportions[0] / 2 else slope, 3.0)
            terrain.idx = 0
        elif choice < proportions[1]:
            self.pyramid_sloped_terrain(terrain, slope, 3.0)
            self.random_uniform_terrain(terrain, -0.05, 0.05, 0.005, 0.2)
            terrain.idx = 1
        elif choice < proportions[3]:
            terrain.idx = 2 if choice < proportions[2] else 3
            self.pyramid_stairs_terrain(terrain, 0.31, -rise if terrain.idx == 2 else rise, 3.0)
        else:
            # 固定原版十列只会到达 id 4；其余权重仍保存在配置中。
            self.discrete_obstacles_terrain(terrain, 0.05 + difficulty * 0.2, 1.0, 2.5, 20, 3.0)
            terrain.idx = 4

    def _make_parkour(self, terrain, choice, difficulty, proportions, num_rows):
        depth = self.py_random.uniform(0.6, 0.6)
        idx = next(i for i, threshold in enumerate(proportions) if choice < threshold)
        terrain.idx = idx
        rise = 0.05 + 0.18 * difficulty
        stone_size = 0.8 if difficulty < 0.2 else -0.5 * difficulty + 0.8
        distance = 0.1 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10
        if idx == 3:
            self.parkour_step_gap_terrain(terrain, difficulty, depth, 2)
        elif idx == 4:
            size = 0.7 if difficulty < 0.2 else -0.5 * difficulty * difficulty + 0.7
            gap = 0.05 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10
            self.stepping_stones_terrain(terrain, size, gap, rise, 2.0, depth)
        elif idx == 5:
            self.stepping_two_stones_terrain(terrain, stone_size, distance, rise, 2.0, depth)
        elif idx == 7:
            self.stepping_one_stones_terrain(terrain, difficulty, stone_size, distance, rise, 2.0, depth)
        elif idx == 8:
            size = 0.6 if difficulty < 0.2 else -0.6 * difficulty + 0.8
            self.stepping_one_bridge_terrain(terrain, size, difficulty, 2.0, depth)
        elif idx == 13:
            size = 0.35 if difficulty < 0.2 else -0.1 * difficulty + 0.35
            self.stepping_air_breams_terrain(terrain, size, distance, rise, 2.0, depth)
            self.add_breams_terrain(terrain, num_rows)
        elif idx == 14:
            self.air_stone(terrain, difficulty, depth)
            self.add_stone_terrain(terrain, num_rows)
        elif idx == 15:
            minimum = rise if difficulty < 0.1 else 0.1 + 0.40 * difficulty
            maximum = 0.1 + rise if difficulty < 0.1 else 0.2 + 0.50 * difficulty
            self.parkour_hurdle_terrain(terrain, difficulty, stone_size, distance, rise, [minimum, maximum], 2.0, depth)
        elif idx == 16:
            self.half_sloped_terrain(terrain, level_index=difficulty * 2.5, depth=depth)
        elif idx == 17:
            self.narrow_corridor(terrain, difficulty, depth)
            self.add_narrow_terrain(terrain, num_rows)
        else:
            raise ValueError(f"Unreachable terrain class in the native ten-column recipe: {idx}")
        if idx not in (14, 17):
            # 官方 add_roughness 不传 difficulty，始终使用默认 1。
            height = self.py_random.uniform(0.01, 0.04)
            self.random_uniform_terrain(terrain, -height, height, 0.005, 0.5)


def _generate_family(cfg: NativeTerrainCfg, seed: int, world_offset: np.ndarray) -> MgdpFamilyData:
    geometry = NativeGeometry(seed)
    nx = int(cfg.terrain_length / cfg.horizontal_scale)
    ny = int(cfg.terrain_width / cfg.horizontal_scale)
    border = int(cfg.border_size / cfg.horizontal_scale)
    raw = np.zeros((cfg.num_rows * nx + 2 * border, cfg.num_cols * ny + 2 * border), dtype=np.int16)
    origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
    classes = np.zeros((cfg.num_rows, cfg.num_cols), dtype=np.int64)
    goals = np.zeros((cfg.num_rows, 3))
    goals_stone, goals_narrow = goals.copy(), goals.copy()
    for col in range(cfg.num_cols):
        for row in range(cfg.num_rows):
            patch = geometry.make_patch(cfg, row, col)
            raw[border + row * nx:border + (row + 1) * nx, border + col * ny:border + (col + 1) * ny] = patch.height_field_raw
            x1, x2 = [int((cfg.terrain_length / 2 + delta) / cfg.horizontal_scale) for delta in (-1, 1)]
            y1, y2 = [int((cfg.terrain_width / 2 + delta) / cfg.horizontal_scale) for delta in (-1, 1)]
            origins[row, col] = [
                (row + 0.5) * cfg.terrain_length, (col + 0.5) * cfg.terrain_width,
                np.max(patch.height_field_raw[x1:x2, y1:y2]) * cfg.vertical_scale,
            ]
            classes[row, col] = patch.idx
            if cfg.name == "parkour":
                if patch.idx == 13:
                    goals[row] = patch.center_position[row] + [row * cfg.terrain_length, col * cfg.terrain_width, 0]
                elif patch.idx == 14:
                    goals_stone[row] = patch.center_position_stone[row] + [row * cfg.terrain_length, col * cfg.terrain_width, 0]
                elif patch.idx == 17:
                    goals_narrow[row] = patch.center_position[row] + [
                        (row + 1) * cfg.terrain_length, col * cfg.terrain_width + cfg.terrain_width / 2, 0.32,
                    ]
    edge_mask = native_edge_mask(raw, cfg)
    collision_raw, boxes = raw.copy(), []
    if cfg.name == "parkour":
        for kind, centers, function, indices in (
            ("beam", goals, geometry.get_muti_beam_trimeshes, [0, 1, 4, 5]),
            ("overhead", goals_stone, geometry.get_muti_stone_trimeshes, [2, 3, 6, 7]),
        ):
            vertices, _ = function(
                centers, cfg.num_rows, cfg.num_cols, cfg.terrain_length, cfg.terrain_width,
                cfg.horizontal_scale, cfg.vertical_scale, cfg.border_size, box_size=np.zeros(3),
            )
            for vertices_raw in vertices:
                face = vertices_raw[indices]
                xi = [int(x / cfg.horizontal_scale) for x in face[:, 0]]
                yi = [int(y / cfg.horizontal_scale) for y in face[:, 1]]
                raw[xi[0]:xi[1], yi[0]:yi[2]] = face[0, 2] / cfg.vertical_scale
                world_vertices = vertices_raw.astype(np.float64) - [cfg.border_size, cfg.border_size, 0] + world_offset
                low, high = world_vertices.min(axis=0), world_vertices.max(axis=0)
                boxes.append(MgdpBox(kind, world_vertices, (low + high) / 2, high - low))
    return MgdpFamilyData(
        cfg, world_offset, collision_raw, raw, edge_mask, origins + world_offset,
        classes, tuple(boxes), goals_narrow + world_offset,
    )


def generate_mgdp_terrain(seed: int = 42, *, family: str = "mix") -> MgdpTerrainData:
    """生成指定阶段的原生地图；combined 保留为显式诊断选项。"""
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    if family not in ("mix", "parkour", "combined"):
        raise ValueError("family must be 'mix', 'parkour', or 'combined'.")
    configs = (MIX_CFG, PARKOUR_CFG) if family == "combined" else (MIX_CFG if family == "mix" else PARKOUR_CFG,)
    families = tuple(
        _generate_family(cfg, int(seed), np.array([210.0 if family == "combined" and cfg.name == "parkour" else 0.0, 0.0, 0.0]))
        for cfg in configs
    )
    origins = np.empty((max(cfg.num_rows for cfg in configs), 10 * len(configs), 3))
    for index, item in enumerate(families):
        origins[:item.cfg.num_rows, index * 10:(index + 1) * 10] = item.origins
        origins[item.cfg.num_rows:, index * 10:(index + 1) * 10] = item.origins[-1]
    return MgdpTerrainData(
        int(seed), families, origins,
        np.repeat(np.array([0 if cfg.name == "mix" else 1 for cfg in configs], dtype=np.int64), 10),
        np.concatenate([item.classes[0] for item in families]),
        np.repeat(np.array([cfg.num_rows for cfg in configs], dtype=np.int64), 10),
        np.repeat(np.array([[cfg.terrain_length, cfg.terrain_width] for cfg in configs]), 10, axis=0),
    )
