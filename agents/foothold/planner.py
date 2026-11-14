# Copyright (c) 2026-2027 zh
"""内容：按 ACP 公开网络结构独立实现四足地图候选注意力与硬选点。"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn

from .config import FootholdConfig


class FootholdPrediction(NamedTuple):
    """足顺序为 FL、FR、RL、RR；目标为调用任务所定义的米制 yaw 系坐标。"""

    logits: Tensor
    probabilities: Tensor
    targets: Tensor
    indices: Tensor


class FootholdPlanner(nn.Module):
    """从 17×11 地图中为每足选一个点；不生成接触事件或维护锁存状态。

    各头注意力概率取均值，返回其对数作为 logits；不包含已取消的
    占用预测头，也不保留只服务该头的 value／输出投影。
    """

    def __init__(self, config: FootholdConfig, candidate_xy: Tensor):
        super().__init__()
        if candidate_xy.shape != (17, 11, 2) or not candidate_xy.is_floating_point():
            raise ValueError("candidate_xy must be a floating-point tensor of shape [17, 11, 2]")
        if not torch.isfinite(candidate_xy).all():
            raise ValueError("candidate_xy must contain finite coordinates")
        self.proprio_dim = config.proprio_dim
        self.history_length = config.history_length
        self.command_dim = config.command_dim
        self.use_bf16 = False
        self.register_buffer("candidate_xy", candidate_xy.detach().clone().float())
        self.point_encoder = nn.Sequential(nn.Linear(3, 16), nn.ELU())
        self.height_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.ELU(), nn.Conv2d(16, 48, 5, padding=2), nn.ELU())
        self.terrain_projection = nn.Linear(64, 96)
        self.context_encoder = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        input_width = config.proprio_dim * config.history_length + config.command_dim + 64
        self.history_encoder = nn.Sequential(
            nn.Linear(input_width, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 96))
        self.foot_embedding = nn.Embedding(4, 96)
        self.film = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 192))
        self.query_projection = nn.Linear(96, 96)
        self.key_projection = nn.Linear(96, 96)

    def _check_inputs(self, history: Tensor, command: Tensor, height_map: Tensor):
        if (history.dim() != 3 or history.size(1) != self.history_length
                or history.size(2) != self.proprio_dim):
            raise ValueError("history must have shape [B, history_length, proprio_dim]")
        if command.dim() != 2 or command.size(0) != history.size(0) or command.size(1) != self.command_dim:
            raise ValueError("command must have shape [B, command_dim]")
        if (height_map.dim() != 3 or height_map.size(0) != history.size(0)
                or height_map.size(1) != 17 or height_map.size(2) != 11):
            raise ValueError("height_map must have shape [B, 17, 11]")

    def candidate_points(self, height_map: Tensor) -> Tensor:
        """按 x 主序、y 次序拼出候选 xyz；调用者已经检查地图形状。"""
        xy = self.candidate_xy.reshape(1, 187, 2).expand(height_map.size(0), -1, -1)
        return torch.cat((xy, height_map.reshape(-1, 187, 1)), dim=-1)

    def _attention_scores(self, history: Tensor, command: Tensor, height_map: Tensor, candidates: Tensor) -> Tensor:
        """集中网络及注意力矩阵计算，坐标构造和概率归一化留在精度边界外。"""
        batch = history.size(0)
        points = self.point_encoder(candidates)
        local = self.height_encoder(height_map.unsqueeze(1)).flatten(2).transpose(1, 2)
        terrain = self.terrain_projection(torch.cat((points, local), dim=-1))
        context = self.context_encoder(terrain).amax(dim=1)
        shared = self.history_encoder(torch.cat((history.flatten(1), command, context), dim=-1))
        scale, shift = self.film(self.foot_embedding.weight).chunk(2, dim=-1)
        queries = shared.unsqueeze(1) * scale.unsqueeze(0) + shift.unsqueeze(0)
        query = self.query_projection(queries).reshape(batch, 4, 32, 3).transpose(1, 2)
        key = self.key_projection(terrain).reshape(batch, 187, 32, 3).transpose(1, 2)
        return torch.matmul(query, key.transpose(-2, -1))

    def forward(self, history: Tensor, command: Tensor, height_map: Tensor) -> Tensor:
        self._check_inputs(history, command, height_map)
        candidates = self.candidate_points(height_map)
        if self.use_bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                scores = self._attention_scores(history, command, height_map, candidates)
        else:
            scores = self._attention_scores(history, command, height_map, candidates)
        scores = scores.float() / math.sqrt(3.0)
        # 在对数域对各头的概率求均值，保持与多头平均注意力相同的分布。
        return torch.logsumexp(scores.log_softmax(dim=-1), dim=1) - math.log(32.0)

    @torch.jit.export
    def predict(self, history: Tensor, command: Tensor, height_map: Tensor) -> FootholdPrediction:
        """硬选一个完整三维候选；梯度归属由监督学习器与推理调用边界控制。"""
        logits = self(history, command, height_map)
        probabilities = logits.softmax(dim=-1)
        indices = logits.argmax(dim=-1)
        targets = self.candidate_points(height_map).gather(1, indices.unsqueeze(-1).expand(-1, -1, 3))
        return FootholdPrediction(logits, probabilities, targets, indices)
