# Copyright (c) 2026-2027 zh
"""
内容：
    按真实控制步推进命令阶段，按同家族有序等级更新真实回合的地形出生位置。
"""

from __future__ import annotations

from copy import deepcopy
import math

import torch


class CommandCurriculum:
    """参考go2_rl_gym的阶段范围与零命令概率，不使用想象回报晋级。"""

    def __init__(self, cfg, families, device="cpu"):
        self.cfg = deepcopy(cfg)
        self.families = tuple(families)
        self.device = torch.device(device)
        self.control_steps = 0
        if cfg.reference_steps_per_iteration < 1 or cfg.zero_probability_iterations <= 0:
            raise ValueError("Command curriculum progress units must be positive")
        if not 0 <= cfg.zero_probability_start <= 1 or not 0 <= cfg.zero_probability_end <= 1:
            raise ValueError("Invalid standing command probabilities")
        stages = [(-1, cfg.initial_ranges), *((stage["iteration"], stage["ranges"]) for stage in cfg.stages)]
        last = -2
        for iteration, ranges in stages:
            if iteration <= last or len(ranges) != 3:
                raise ValueError("Command stages must be ordered and contain three axes")
            for bounds in ranges:
                if len(bounds) != 2 or not all(math.isfinite(value) for value in bounds) or bounds[0] > bounds[1]:
                    raise ValueError("Invalid command stage bounds")
            last = iteration
        for limits in cfg.terrain_limits.values():
            if len(limits) != 3 or any(not math.isfinite(value) or value <= 0 for value in limits):
                raise ValueError("Terrain velocity limits must be positive")

    @property
    def iteration(self):
        return self.control_steps // self.cfg.reference_steps_per_iteration

    def parameters(self, family_ids):
        ranges = self.cfg.initial_ranges
        for stage in self.cfg.stages:
            if self.iteration >= stage["iteration"]:
                ranges = stage["ranges"]
        bounds = torch.tensor(ranges, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(family_ids.numel(), 1, 1)
        limits = []
        for family in self.families:
            key = "stairs" if family.startswith("stairs") else "slope" if family.startswith("slope") else family
            limits.append(self.cfg.terrain_limits[key])
        limit = torch.tensor(limits, dtype=torch.float32, device=self.device)[family_ids]
        bounds[:, :, 0] = torch.maximum(bounds[:, :, 0], -limit)
        bounds[:, :, 1] = torch.minimum(bounds[:, :, 1], limit)
        progress = min(self.iteration / self.cfg.zero_probability_iterations, 1.0)
        probability = self.cfg.zero_probability_start + progress * (self.cfg.zero_probability_end - self.cfg.zero_probability_start)
        return bounds.flatten(1), torch.full((family_ids.numel(), 1), probability, dtype=torch.float32, device=self.device)

    def state_dict(self):
        return {"control_steps": self.control_steps}

    def load_state_dict(self, state):
        value = state["control_steps"]
        if not isinstance(value, int) or value < 0:
            raise ValueError("Invalid command curriculum control steps")
        self.control_steps = value


class TerrainCurriculum:
    """逻辑level只在同family递增；实际origin映射交给TerrainMap。"""

    def __init__(self, terrain_map, num_envs, max_init_level=0, *, enabled=True, episode_length_s=20.0, device="cpu"):
        if not isinstance(max_init_level, int) or max_init_level < 0:
            raise ValueError("Initial terrain level must be a nonnegative integer")
        self.map = terrain_map
        self.enabled = enabled and terrain_map.family_names != ("flat",)
        self.episode_length_s = episode_length_s
        self.device = torch.device(device)
        self.family_ids = torch.arange(num_envs, device=device) % len(terrain_map.family_names)
        self.level_counts = torch.tensor(terrain_map.level_counts, device=device, dtype=torch.long)
        maximum = torch.minimum(self.level_counts[self.family_ids] - 1, torch.full_like(self.family_ids, max_init_level))
        self.levels = torch.floor(torch.rand(num_envs, device=device) * (maximum + 1)).long()
        self.start_xy = torch.zeros(num_envs, 2, device=device)
        self.max_distance = torch.zeros(num_envs, device=device)
        self.commanded_displacement = torch.zeros(num_envs, 2, device=device)
        self.elapsed = torch.zeros(num_envs, device=device)

    def observe(self, world_xy, commanded_velocity_world, dt):
        self.max_distance = torch.maximum(self.max_distance, (world_xy - self.start_xy).norm(dim=-1))
        self.commanded_displacement += commanded_velocity_world * dt
        self.elapsed += dt

    def complete(self, ids, current_command):
        if not self.enabled or ids.numel() == 0:
            return
        distance = self.max_distance[ids]
        # 早摔保留剩余预算，避免短回合因尚未累计足够命令而逃过降级。
        required = self.commanded_displacement[ids].norm(dim=-1)
        required += current_command[ids, :2].norm(dim=-1) * (self.episode_length_s - self.elapsed[ids]).clamp_min(0)
        valid = self.elapsed[ids] > 0
        up = (distance > self.map.size[0] / 2) & valid
        down = (distance < required * 0.5) & ~up & valid
        candidate = (self.levels[ids] + up.long() - down.long()).clamp_min(0)
        counts = self.level_counts[self.family_ids[ids]]
        random_level = torch.floor(torch.rand(ids.numel(), device=self.device) * counts).long()
        self.levels[ids] = torch.where(candidate >= counts, random_level, candidate)

    def reset_episodes(self, ids, world_xy):
        self.start_xy[ids] = world_xy[ids]
        self.max_distance[ids] = 0
        self.commanded_displacement[ids] = 0
        self.elapsed[ids] = 0

    def origins(self, ids):
        return self.map.spawn_origins(self.family_ids[ids], self.levels[ids])

    def state_dict(self):
        return {"family_ids": self.family_ids.clone(), "levels": self.levels.clone()}

    def load_state_dict(self, state):
        families = state["family_ids"].to(self.device)
        levels = state["levels"].to(self.device)
        if families.shape != self.family_ids.shape or levels.shape != self.levels.shape:
            raise ValueError("Terrain curriculum environment count mismatch")
        if families.dtype != torch.long or levels.dtype != torch.long or not torch.equal(families, self.family_ids):
            raise ValueError("Terrain curriculum family assignment mismatch")
        if (levels < 0).any() or (levels >= self.level_counts[families]).any():
            raise ValueError("Invalid saved terrain levels")
        self.levels.copy_(levels)
