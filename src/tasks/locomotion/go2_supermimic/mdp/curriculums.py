# Copyright (c) 2026-2027 zh
"""内容：保持固定HIM速度范围，按Go2 HIM行走表现升降地形难度。"""

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
    """复现Go2 HIM地形升降规则，保留现有地形布局与训练起始等级。"""

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
        self.episode_start_pos_w = torch.zeros(num_envs, 2, device=device)
        self.commanded_displacement_w = torch.zeros_like(self.episode_start_pos_w)
        if not self.enabled and not terrain_map.plane:
            self.levels = torch.randint(terrain_map.rows, (num_envs,), device=device)

    @property
    def max_level(self):
        return self.map.rows - 1

    def sample(self, ids):
        """重置保留已由行走表现决定的等级，不按训练进度重新抽样。"""
        del ids

    def reset(self, ids, positions):
        """以实际出生位置开始新回合，手动重置不参与升降。"""
        self.episode_start_pos_w[ids] = positions
        self.commanded_displacement_w[ids] = 0

    def advance(self, command, yaw, dt):
        """按来源时序累计已执行命令的世界坐标位移向量。"""
        if not self.enabled:
            return
        vx, vy = command[:, :2].unbind(-1)
        velocity = torch.stack((yaw.cos() * vx - yaw.sin() * vy, yaw.sin() * vx + yaw.cos() * vy), -1)
        self.commanded_displacement_w.add_(velocity * dt)

    def update(self, env, ids, command):
        """结算已结束回合；当前结束步的命令积分在此之后推进。"""
        if not self.enabled or ids.numel() == 0:
            return
        terrain = env.scene.terrain
        position = env.scene["robot"].data.root_link_pos_w[ids, :2]
        distance = (position - self.episode_start_pos_w[ids]).norm(dim=-1)
        time_left = env.max_episode_length_s - env.episode_length_buf[ids] * env.step_dt
        commanded_distance = (self.commanded_displacement_w[ids].norm(dim=-1)
                              + command[ids, :2].norm(dim=-1) * time_left.clamp_min(0))
        terrain_distance = (position - env.scene.env_origins[ids, :2]).norm(dim=-1)
        move_up = (terrain_distance > self.map.size[0] / 2) & (commanded_distance >= 1.0)
        move_down = (distance < commanded_distance * 0.5) & ~move_up
        has_episode = (env.episode_length_buf[ids] > 0) & (env.common_step_counter > 0)
        move_up &= has_episode
        move_down &= has_episode
        # 使用原生API保留最高级随机回落及最低等级截零的来源行为。
        terrain.terrain_levels[ids] = self.levels[ids]
        terrain.update_env_origins(ids, move_up, move_down)
        self.levels[ids] = terrain.terrain_levels[ids]

    def origins(self, ids):
        return self.map.spawn_origins(self.family_ids[ids], self.levels[ids])

    def state_dict(self):
        return {"family_ids": self.family_ids.clone(), "levels": self.levels.clone(), "control_steps": self.control_steps}

    def load_state_dict(self, state):
        self.family_ids.copy_(state["family_ids"].to(self.device))
        self.levels.copy_(state["levels"].to(self.device))
        self.control_steps = state["control_steps"]
