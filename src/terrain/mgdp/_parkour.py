# Copyright (c) 2026-2027 zh
"""
内容：
    MGDP f43139d 的 parkour 纯几何函数；不依赖 IsaacGym。
    来源及迁移边界见同目录 NOTICE.md。仅把随机数源改为实例局部状态。
"""

import numpy as np


class ParkourGeometry:
    """由生成器注入 np_random 与 py_random，保留官方调用顺序。"""
    def air_stone(self, terrain, difficulty, depth=2.0):
        """等价移植官方 add_extreme_gap_terrain.py:217；保留原始量化与随机抽样。"""
        start_y, end_y = (20, terrain.length - 20)
        terrain.height_field_raw[:] = int(-depth / terrain.vertical_scale)
        terrain.height_field_raw[:, start_y:end_y] = 0

    def get_muti_stone_trimeshes(
        self, center_position, num_rows, num_cols, terrain_length, terrain_width,
        horizontal_scale, vertical_scale, border_size,
        center_rpy=np.zeros(3), box_size=np.zeros(3), platform_size=2,
    ):
        """等价移植官方 add_extreme_gap_terrain.py:223；保留原始量化与随机抽样。"""
        frame_vertices, frame_triangles = ([], [])
        terrain_length = terrain_length / horizontal_scale
        terrain_width = terrain_width / horizontal_scale
        max_height = 0.55
        min_height = 0.18
        for i in range(num_rows):
            difficulty = i / (num_rows - 1) if num_rows > 1 else 0
            height = max_height - (max_height - min_height) * difficulty
            box_size[0] = 4.5
            box_size[1] = self.np_random.randint(18, 25) / 10
            box_size[2] = self.np_random.randint(1, 5) / 10
            center_x = (terrain_length - box_size[0]) // 20
            center_y = (terrain_width - box_size[1] + terrain_width) // 20
            center = center_position[i, 0:3] + np.array([center_x, center_y, box_size[2]])
            pos_x = self.np_random.randint(1, 2) / 10
            center += np.array([pos_x, 0, height])
            center = np.round(center, 3)
            vertices_cur, triangles_cur = self.stone_trimesh(box_size, center, center_rpy)
            frame_vertices.append(vertices_cur)
            frame_triangles.append(triangles_cur)
            center -= np.array([pos_x, 0, height])
        return (frame_vertices, frame_triangles)

    def stone_trimesh(self, size, center_position, rpy=np.zeros(3)):
        """等价移植官方 add_extreme_gap_terrain.py:265；保留原始量化与随机抽样。"""
        if not (rpy == 0).all():
            raise NotImplementedError('Only axis-aligned box triangle mesh is implemented')
        vertices = np.empty((8, 3), dtype=np.float32)
        vertices[:] = center_position
        vertices[[0, 4, 2, 6], 0] -= size[0] / 2
        vertices[[1, 5, 3, 7], 0] += size[0] / 2
        vertices[[0, 1, 2, 3], 1] -= size[1] / 2
        vertices[[4, 5, 6, 7], 1] += size[1] / 2
        vertices[[2, 3, 6, 7], 2] -= size[2] / 2
        vertices[[0, 1, 4, 5], 2] += size[2] / 2
        vertices = np.round(vertices, 3)
        triangles = -np.ones((12, 3), dtype=np.uint32)
        triangles[0] = [0, 2, 1]
        triangles[1] = [1, 2, 3]
        triangles[2] = [0, 4, 2]
        triangles[3] = [2, 4, 6]
        triangles[4] = [4, 5, 6]
        triangles[5] = [5, 7, 6]
        triangles[6] = [1, 3, 5]
        triangles[7] = [3, 7, 5]
        triangles[8] = [0, 1, 4]
        triangles[9] = [1, 5, 4]
        triangles[10] = [2, 6, 3]
        triangles[11] = [3, 6, 7]
        return (vertices, triangles)

    def add_stone_terrain(self, terrain, num_rows):
        """等价移植官方 add_extreme_gap_terrain.py:302；保留原始量化与随机抽样。"""
        terrain.center_position_stone = np.zeros((num_rows, 3))

    def add_narrow_terrain(self, terrain, num_rows):
        """等价移植官方 add_extreme_gap_terrain.py:305；保留原始量化与随机抽样。"""
        terrain.center_position = np.zeros((num_rows, 3))

    def narrow_corridor(self, terrain, difficulty, depth=2.0):
        """等价移植官方 add_extreme_gap_terrain.py:308；保留原始量化与随机抽样。"""
        platform_size = int(2 / terrain.horizontal_scale)
        base_height = int(-depth / terrain.vertical_scale)
        wall_abs_height = self.np_random.randint(200, 251)
        start_y, end_y = (20, terrain.length - 20)
        terrain.height_field_raw[:] = base_height
        terrain.height_field_raw[:, start_y:end_y] = 0
        center_y = terrain.length // 2
        min_gap = 2
        max_gap = 8
        narrow_gap_size = max_gap - (max_gap - min_gap) * difficulty
        narrow_gap = int(narrow_gap_size)
        narrow_gap = np.clip(narrow_gap, min_gap, max_gap)
        terrain.height_field_raw[platform_size + 20:, center_y + narrow_gap:end_y] = wall_abs_height
        terrain.height_field_raw[platform_size + 20:, start_y:center_y - narrow_gap] = wall_abs_height

    def half_sloped_terrain(self, terrain, level_index, platform_size=2.0, slope_strength=None, final_platform_length=5, depth=2.0):
        """等价移植官方 add_extreme_gap_terrain.py:338；保留原始量化与随机抽样。"""
        platform_size = int(platform_size / terrain.horizontal_scale)
        if slope_strength is None:
            slope_strength = 2.0 + 2.5 * level_index
        base_height = int(-depth / terrain.vertical_scale)
        start_y, end_y = (20, terrain.length - 20)
        mid_region = (slice(None), slice(start_y, end_y))
        terrain.height_field_raw[:] = base_height
        terrain.height_field_raw[mid_region] = 0
        positions = {
            'initial_platform': (0, platform_size),
            'up_slope': (platform_size, 2 * platform_size),
            'mid_platform': (2 * platform_size, 3 * platform_size),
            'down_slope': (3 * platform_size, 4 * platform_size),
            'final_platform': (4 * platform_size, terrain.width),
        }
        positions['final_platform'] = (positions['down_slope'][1], min(positions['down_slope'][1] + final_platform_length, terrain.width))
        up_start, up_end = positions['up_slope']
        xs = np.arange(up_start, up_end)
        max_height = slope_strength * (up_end - up_start)
        up_heights = (slope_strength * (xs - up_start)).astype(np.int16)
        terrain.height_field_raw[up_start:up_end, start_y:end_y] = up_heights[:, None]
        mid_start, mid_end = positions['mid_platform']
        terrain.height_field_raw[mid_start:mid_end, start_y:end_y] = max_height
        down_start, down_end = positions['down_slope']
        down_end = min(down_start + (up_end - up_start), positions['final_platform'][0])
        xs = np.arange(down_start, down_end)
        down_heights = (max_height - slope_strength * (xs - down_start)).astype(np.int16)
        terrain.height_field_raw[down_start:down_end, start_y:end_y] = down_heights[:, None]

    def box_trimesh(self, size, center_position, rpy=np.zeros(3)):
        """等价移植官方 add_extreme_gap_terrain.py:412；保留原始量化与随机抽样。"""
        if not (rpy == 0).all():
            raise NotImplementedError('Only axis-aligned box triangle mesh is implemented')
        vertices = np.empty((8, 3), dtype=np.float32)
        vertices[:] = center_position
        vertices[[0, 4, 2, 6], 0] -= size[0] / 2
        vertices[[1, 5, 3, 7], 0] += size[0] / 2
        vertices[[0, 1, 2, 3], 1] -= size[1] / 2
        vertices[[4, 5, 6, 7], 1] += size[1] / 2
        vertices[[2, 3, 6, 7], 2] -= size[2] / 2
        vertices[[0, 1, 4, 5], 2] += size[2] / 2
        vertices = np.round(vertices, 3)
        triangles = -np.ones((12, 3), dtype=np.uint32)
        triangles[0] = [0, 2, 1]
        triangles[1] = [1, 2, 3]
        triangles[2] = [0, 4, 2]
        triangles[3] = [2, 4, 6]
        triangles[4] = [4, 5, 6]
        triangles[5] = [5, 7, 6]
        triangles[6] = [1, 3, 5]
        triangles[7] = [3, 7, 5]
        triangles[8] = [0, 1, 4]
        triangles[9] = [1, 5, 4]
        triangles[10] = [2, 6, 3]
        triangles[11] = [3, 6, 7]
        return (vertices, triangles)

    def parkour_step_gap_terrain(self, terrain, difficulty, depth, platform_size=2.0):
        """等价移植官方 add_extreme_gap_terrain.py:449；保留原始量化与随机抽样。"""
        gap_size = int(difficulty / terrain.horizontal_scale)
        gap_size = np.clip(int(gap_size), 2, 12)
        depth = int(depth / terrain.vertical_scale)
        platform_size = int(platform_size / terrain.horizontal_scale)
        start_y = 0
        end_y = int(terrain.length - platform_size / 8)
        start_x = platform_size
        center_x = terrain.width // 2
        terrain.height_field_raw[start_x:center_x, start_y:end_y] = -depth
        terrain.height_field_raw[start_x + gap_size:center_x - gap_size, start_y + gap_size:end_y - gap_size] = 0
        start_x = center_x + int(platform_size / 2)
        end_x = int(terrain.width)
        terrain.height_field_raw[start_x:end_x, start_y:end_y] = -depth
        terrain.height_field_raw[start_x + gap_size:end_x - gap_size, start_y + gap_size:end_y - gap_size] = 0

    def stepping_stones_terrain(self, terrain, stone_size, stone_distance, max_height, platform_size=1.0, depth=1):
        """等价移植官方 add_extreme_gap_terrain.py:473；保留原始量化与随机抽样。"""
        stone_size = int(stone_size / terrain.horizontal_scale)
        stone_distance_x = int(stone_distance / terrain.horizontal_scale)
        stone_distance_y = min(int(stone_distance / terrain.horizontal_scale), 1)
        max_height = min(int(max_height / terrain.vertical_scale), 40)
        platform_size = int(platform_size / terrain.horizontal_scale)
        height_range = np.arange(1, max_height, step=4)
        start_x = 0
        terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)
        if terrain.width > terrain.length:
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                start_y = self.np_random.randint(0, stone_size)
                stop_y = max(0, start_y - stone_distance_y)
                terrain.height_field_raw[start_x:stop_x, 0:stop_y] = self.np_random.choice(height_range)
                while start_y < terrain.length - 10:
                    stop_y = min(terrain.length - 10, start_y + stone_size)
                    terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = self.np_random.choice(height_range)
                    start_y += stone_size + stone_distance_y
                start_x += stone_size + stone_distance_x
        x1 = 0
        x2 = platform_size
        y1 = (terrain.length - platform_size) // 2
        y2 = (terrain.length + platform_size) // 2
        terrain.height_field_raw[x1:x2, y1:y2] = 0
        return terrain

    def stepping_two_stones_terrain(self, terrain, stone_size, stone_distance, max_height, platform_size=1.0, depth=1):
        """等价移植官方 add_extreme_gap_terrain.py:507；保留原始量化与随机抽样。"""
        stone_size = int(stone_size / terrain.horizontal_scale)
        max_height = int(max_height / terrain.vertical_scale)
        max_height = np.clip(max_height, 0, 30)
        platform_size = int(platform_size / terrain.horizontal_scale)
        height_range = np.arange(1, max_height, step=3)
        stone_size = np.clip(stone_size, 6, 16)
        stone_distance_gap = int(stone_distance / terrain.horizontal_scale)
        stone_distance_gap = np.clip(stone_distance_gap, 1, 6)
        stone_distance_range = np.arange(0, stone_distance_gap, step=1) + 1
        platform_y = (terrain.length - platform_size) // 2
        row1_y = int(platform_y + platform_size / 2 - stone_size - 0.1)
        row2_y = int(platform_y + platform_size / 2 + 0.1)
        terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)
        start_x = 0
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            terrain.height_field_raw[start_x:stop_x, row1_y:row1_y + stone_size] = self.np_random.choice(height_range)
            terrain.height_field_raw[start_x:stop_x, row2_y:row2_y + stone_size] = self.np_random.choice(height_range)
            stone_distance_gap = self.np_random.choice(stone_distance_range)
            start_x += stone_size + stone_distance_gap
        x1 = 0
        x2 = platform_size
        terrain.height_field_raw[x1:x2, platform_y:platform_y + platform_size] = 0
        return terrain

    def stepping_one_stones_terrain(self, terrain, difficulty, stone_size, stone_distance, max_height, platform_size=1.0, depth=1):
        """等价移植官方 add_extreme_gap_terrain.py:597；保留原始量化与随机抽样。"""
        stone_size = int(stone_size / terrain.horizontal_scale)
        stone_distance = int(stone_distance / terrain.horizontal_scale)
        max_height = int(max_height / terrain.vertical_scale)
        max_height = np.clip(max_height, 0, 20)
        platform_size = int(platform_size / terrain.horizontal_scale)
        if difficulty < 0.3:
            height_range = np.arange(1, max_height, step=3)
        elif 0.3 <= difficulty < 0.7:
            height_range = np.arange(1, max_height, step=4)
        else:
            height_range = np.arange(1, max_height, step=12)
        platform_y = (terrain.length - platform_size) // 2
        terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)
        stone_distance_gap = np.clip(int(difficulty / terrain.horizontal_scale), 4, 16)
        start_x = 0
        while start_x < terrain.width:
            stone_size_x = np.clip(stone_size, 12, 30)
            stone_distance_range = np.arange(12, 30, step=1)
            stone_size_y = self.np_random.choice(stone_distance_range)
            row1_y = int(platform_y + platform_size / 2 - stone_size_y / 2)
            stop_x = min(terrain.width, start_x + stone_size_x)
            terrain.height_field_raw[start_x:stop_x, row1_y:row1_y + stone_size_y] = self.np_random.choice(height_range)
            start_x += stone_size + stone_distance_gap
        x1 = 0
        x2 = platform_size
        terrain.height_field_raw[x1:x2, platform_y:platform_y + platform_size] = 0
        return terrain

    def stepping_one_bridge_terrain(self, terrain, stone_size, difficulty, platform_size=1.0, depth=1):
        """等价移植官方 add_extreme_gap_terrain.py:644；保留原始量化与随机抽样。"""
        stone_size = int(stone_size / terrain.horizontal_scale)
        platform_size = int(platform_size / terrain.horizontal_scale)
        platform_y = (terrain.length - platform_size) // 2
        row1_y = int(platform_y + platform_size / 2 - stone_size / 2)
        terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)
        height_range = np.arange(0, 10, step=2)
        start_x = 0
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            terrain.height_field_raw[start_x:stop_x, row1_y:row1_y + stone_size] = self.np_random.choice(height_range)
            start_x += stone_size
        x1 = 0
        x2 = platform_size
        terrain.height_field_raw[x1:x2, platform_y:platform_y + platform_size] = 0
        return terrain

    def stepping_air_breams_terrain(self, terrain, stone_size, stone_distance, max_height, platform_size=1.0, depth=1):
        """等价移植官方 add_extreme_gap_terrain.py:916；保留原始量化与随机抽样。"""
        platform_size = int(platform_size / terrain.horizontal_scale)
        platform_y = (terrain.length - platform_size) // 2
        terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)
        x1 = 0
        x2 = platform_size
        terrain.height_field_raw[x1:x2, platform_y:platform_y + platform_size] = 0
        return terrain

    def get_muti_beam_trimeshes(
        self, center_position, num_rows, num_cols, terrain_length, terrain_width,
        horizontal_scale, vertical_scale, border_size,
        center_rpy=np.zeros(3), box_size=np.zeros(3), platform_size=2,
    ):
        """等价移植官方 add_extreme_gap_terrain.py:929；保留原始量化与随机抽样。"""
        frame_vertices, frame_triangles = ([], [])
        platform_size = int(platform_size / horizontal_scale)
        box_size[2] = 0.1
        min_bream_width = 7
        max_bream_width = 15
        terrain_length = terrain_length / horizontal_scale
        terrain_width = terrain_width / horizontal_scale
        for i in range(num_rows):
            difficulty = i / (num_rows - 1)
            stone_size = 0.35 if difficulty < 0.2 else -0.1 * difficulty + 0.35
            stone_distance = 0.1 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10
            bream_length = int(stone_size / horizontal_scale)
            stone_distance = int(stone_distance / horizontal_scale)
            box_size[0] = bream_length / 20
            step_height = 0.04 if difficulty < 0.2 else 0.05 + 0.18 * difficulty
            max_height = int(step_height / vertical_scale) / 100
            height_range = np.arange(0, max_height, step=0.04)
            center = center_position[i, 0:3] + np.array([5 + box_size[0] / 2, 5, 0])
            start_y = (terrain_width - platform_size) // 2
            start_x = platform_size
            while start_x < terrain_length:
                bream_width = 2 * self.py_random.randint(min_bream_width, max_bream_width)
                box_size[1] = bream_width / 20
                height = self.np_random.choice(height_range)
                center += np.array([start_x / 20, start_y / 10, height])
                center = np.round(center, 3)
                vertices_cur, triangles_cur = self.box_trimesh(box_size, center, center_rpy)
                frame_vertices.append(vertices_cur)
                frame_triangles.append(triangles_cur)
                center -= np.array([start_x / 20, start_y / 10, height])
                start_x += bream_length + stone_distance
        return (frame_vertices, frame_triangles)

    def add_breams_terrain(self, terrain, num_rows):
        """等价移植官方 add_extreme_gap_terrain.py:978；保留原始量化与随机抽样。"""
        goals = np.zeros((num_rows, 3))
        terrain.center_position = goals

    def parkour_hurdle_terrain(self, terrain, difficulty, stone_size, stone_distance, max_height, hurdle_height_range, platform_size=1.0, depth=1):
        """等价移植官方 add_extreme_gap_terrain.py:1070；保留原始量化与随机抽样。"""
        stone_size = int(stone_size / terrain.horizontal_scale)
        platform_size = int(platform_size / terrain.horizontal_scale)
        platform_y = (terrain.length - platform_size) // 2
        terrain.height_field_raw[:, :] = int(-0.6 / terrain.vertical_scale)
        row0_y = int(platform_y + platform_size / 2 - 17 / 2)
        terrain.height_field_raw[:, row0_y:row0_y + 18] = 0
        hurdle_height_max = round(hurdle_height_range[1] / terrain.vertical_scale)
        hurdle_height_min = round(hurdle_height_range[0] / terrain.vertical_scale)
        step_height = self.np_random.randint(hurdle_height_min, hurdle_height_max)
        start_x = platform_size + 20
        while start_x < terrain.width - 20:
            stone_size_x = np.clip(stone_size, 12, 30)
            stone_distance_range = np.arange(17, 30, step=1)
            stop_x = min(terrain.width, start_x + stone_size_x)
            stone_size_y = self.np_random.choice(stone_distance_range)
            row1_y = int(platform_y + platform_size / 2 - stone_size_y / 2)
            terrain.height_field_raw[start_x:stop_x, row1_y:row1_y + stone_size_y] = step_height
            stone_distance_gap = np.clip(int(difficulty / terrain.horizontal_scale), 60, 80)
            start_x += stone_size + stone_distance_gap
        x1 = 0
        x2 = int(platform_size / 2.5)
        return terrain
