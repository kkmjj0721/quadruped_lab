# Copyright (c) 2026-2027 zh
"""内容：保持固定HIM速度范围，按参考训练进度开放地形行。"""

from copy import deepcopy

import torch
import numpy as np
from mjlab.terrains.terrain_entity import _proportional_counts


class CommandCurriculum:
    """只提供固定范围与进度计数，不进行速度课程。"""

    def __init__(self, cfg, families, device="cpu"):
        self.cfg = deepcopy(cfg)
        self.device = torch.device(device)
        self.control_steps = 0

    @property
    def iteration(self):
        return self.control_steps

    def parameters(self, family_ids):
        bounds = torch.tensor(self.cfg.initial_ranges, dtype=torch.float32, device=self.device)
        return (bounds.flatten().expand(family_ids.numel(), -1),
                torch.full((family_ids.numel(), 1), self.cfg.rel_standing_envs, device=self.device))

    def state_dict(self):
        return {"control_steps": self.control_steps}

    def load_state_dict(self, state):
        self.control_steps = state["control_steps"]


class TerrainCurriculum:
    """复现terrain_levels_by_progress：每次reset均匀抽样已开放行。"""

    def __init__(self, terrain_map, num_envs, max_init_level=0, *, enabled=True, episode_length_s=20.0, device="cpu"):
        self.map = terrain_map
        self.enabled = enabled and not terrain_map.plane
        self.device = torch.device(device)
        self.control_steps = 0
        # MjLab按环境编号连续分列，而非逐环境循环交错分配。
        counts = torch.from_numpy(_proportional_counts(num_envs, np.asarray(terrain_map.proportions))).to(device)
        self.family_ids = torch.repeat_interleave(torch.arange(terrain_map.cols, device=device), counts)
        self.level_counts = torch.full((terrain_map.cols,), terrain_map.rows, device=device, dtype=torch.long)
        self.levels = torch.zeros(num_envs, device=device, dtype=torch.long)
        if not self.enabled and not terrain_map.plane:
            self.levels = torch.randint(terrain_map.rows, (num_envs,), device=device)

    @property
    def max_level(self):
        progress = min(1.0, self.control_steps / max(1, self.map.cfg.curriculum_iterations))
        return min(self.map.rows - 1, int(progress * self.map.rows))

    def sample(self, ids):
        if self.enabled and ids.numel():
            self.levels[ids] = torch.randint(self.max_level + 1, (ids.numel(),), device=self.device)

    def origins(self, ids):
        return self.map.spawn_origins(self.family_ids[ids], self.levels[ids])

    def state_dict(self):
        return {"family_ids": self.family_ids.clone(), "levels": self.levels.clone(), "control_steps": self.control_steps}

    def load_state_dict(self, state):
        self.family_ids.copy_(state["family_ids"].to(self.device))
        self.levels.copy_(state["levels"].to(self.device))
        self.control_steps = state["control_steps"]
