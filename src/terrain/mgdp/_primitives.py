# Copyright (c) 2026-2027 zh
"""
内容：
    使用独立实现复现 MGDP 所用整数高度场基础运算和随机抽样顺序。
    参考版本及旧版 SciPy 插值的等价替换说明见 NOTICE.md。
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RectBivariateSpline


@dataclass
class HeightPatch:
    width: int
    length: int
    horizontal_scale: float
    vertical_scale: float = 0.005
    height_field_raw: np.ndarray = field(init=False)
    idx: int = -1

    def __post_init__(self):
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)


class HeightGeometry:
    """保持离散高度与随机流，不调用 IsaacGym。"""

    def random_uniform_terrain(self, terrain, min_height, max_height, step, downsampled_scale):
        spacing = downsampled_scale
        scale = terrain.vertical_scale
        lo, hi, stride = int(min_height / scale), int(max_height / scale), int(step / scale)
        coarse_shape = (
            int(terrain.width * terrain.horizontal_scale / spacing),
            int(terrain.length * terrain.horizontal_scale / spacing),
        )
        coarse = self.np_random.choice(np.arange(lo, hi + stride, stride), coarse_shape)
        end_x = terrain.width * terrain.horizontal_scale
        end_y = terrain.length * terrain.horizontal_scale
        x = np.linspace(0, end_x, coarse_shape[0])
        y = np.linspace(0, end_y, coarse_shape[1])
        # interp2d(y, x, coarse) 的线性规则网格等价式；轴顺序保持不变。
        interpolator = RectBivariateSpline(x, y, coarse, kx=1, ky=1)
        upsampled = interpolator(np.linspace(0, end_x, terrain.width), np.linspace(0, end_y, terrain.length))
        terrain.height_field_raw += np.rint(upsampled).astype(np.int16)

    def pyramid_sloped_terrain(self, terrain, slope, platform_size):
        nx, ny = terrain.width, terrain.length
        cx, cy = nx // 2, ny // 2
        x = ((cx - np.abs(cx - np.arange(nx))) / cx).reshape(nx, 1)
        y = ((cy - np.abs(cy - np.arange(ny))) / cy).reshape(1, ny)
        peak = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * (nx / 2))
        terrain.height_field_raw += (peak * x * y).astype(np.int16)
        radius = int(platform_size / terrain.horizontal_scale / 2)
        boundary = terrain.height_field_raw[cx - radius, cy - radius]
        terrain.height_field_raw = np.clip(terrain.height_field_raw, min(boundary, 0), max(boundary, 0))

    def pyramid_stairs_terrain(self, terrain, step_width, step_height, platform_size):
        stride = int(step_width / terrain.horizontal_scale)
        rise = int(step_height / terrain.vertical_scale)
        platform = int(platform_size / terrain.horizontal_scale)
        x0, x1, y0, y1, height = 0, terrain.width, 0, terrain.length, 0
        while x1 - x0 > platform and y1 - y0 > platform:
            x0, x1, y0, y1 = x0 + stride, x1 - stride, y0 + stride, y1 - stride
            height += rise
            terrain.height_field_raw[x0:x1, y0:y1] = height

    def discrete_obstacles_terrain(self, terrain, max_height, min_size, max_size, num_rects, platform_size):
        peak = int(max_height / terrain.vertical_scale)
        sizes = range(int(min_size / terrain.horizontal_scale), int(max_size / terrain.horizontal_scale), 4)
        # 负数整除顺序与原版一致，奇数高度时不是 -(peak // 2)。
        heights = [-peak, -peak // 2, peak // 2, peak]
        for _ in range(num_rects):
            nx = self.np_random.choice(sizes)
            ny = self.np_random.choice(sizes)
            x0 = self.np_random.choice(range(0, terrain.width - nx, 4))
            y0 = self.np_random.choice(range(0, terrain.length - ny, 4))
            terrain.height_field_raw[x0:x0 + nx, y0:y0 + ny] = self.np_random.choice(heights)
        width = int(platform_size / terrain.horizontal_scale)
        x0, x1 = (terrain.width - width) // 2, (terrain.width + width) // 2
        y0, y1 = (terrain.length - width) // 2, (terrain.length + width) // 2
        terrain.height_field_raw[x0:x1, y0:y1] = 0
