# Copyright (c) 2026-2027 zh
"""内容：五帧本体历史、四足视觉查询与单路动作MLP，以及历史条件价值网络。"""

import math

import torch
from torch import nn

from .world_model import RunningNorm


class Mlp(nn.Module):
    """矩阵乘可选BF16，网络边界与后续物理计算保持FP32。"""

    def __init__(self, widths, activation=nn.ELU):
        super().__init__()
        layers = []
        for index, (source, target) in enumerate(zip(widths, widths[1:])):
            layers.append(nn.Linear(source, target))
            if index < len(widths) - 2:
                layers.append(activation())
        self.network = nn.Sequential(*layers)
        self.use_bf16 = False

    def forward(self, values):
        if self.use_bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                result = self.network(values)
            return result.float()
        return self.network(values)


class DepthActor(nn.Module):
    """查询参数归策略优化器，CNN与落足预测器由独立感知模块持有。"""

    def __init__(self, hidden=(512, 256, 128)):
        super().__init__()
        self.normalizer = RunningNorm(235)
        self.query = nn.Linear(241, 16)
        self.actor = Mlp([323, *hidden, 12])
        self.register_buffer("foot_ids", torch.eye(4))
        frame, y, x = torch.meshgrid(torch.arange(4), torch.arange(4), torch.arange(4), indexing="ij")
        coordinates = torch.stack((frame / 3, y / 3, x / 3), -1).reshape(64, 3)
        frequency = 2 ** torch.arange(3).float() * math.pi
        codes = torch.cat((torch.sin(coordinates[..., None] * frequency),
                           torch.cos(coordinates[..., None] * frequency)), -1).flatten(1)
        self.register_buffer("position_code", codes[:, :16])

    def normalized_history(self, history):
        return self.normalizer(history.flatten(1))

    def attention(self, history, tokens, points):
        batch = history.shape[0]
        body = self.normalized_history(history)
        query_input = torch.cat((body[:, None].expand(-1, 4, -1), points.reshape(batch, 4, 2),
                                 self.foot_ids[None].expand(batch, -1, -1)), -1)
        query = self.query(query_input)
        weights = torch.softmax(query @ (tokens + self.position_code).transpose(1, 2) / 4, dim=-1)
        return body, weights

    def forward(self, history, tokens, points):
        body, weights = self.attention(history, tokens, points)
        context = (weights @ tokens).flatten(1)
        return self.actor(torch.cat((body, context, tokens.mean(1), points.flatten(1)), -1))

    def read_loss_sum(self, history, tokens, points, targets, valid):
        """返回有效标签的交叉熵之和；由学习器按完整seed批量统一归一化。"""
        _, weights = self.attention(history, tokens.detach(), points.detach())
        errors = -(targets * weights.clamp_min(1e-8).log()).sum(-1)
        return (errors * valid).sum()

    def policy_parameters(self):
        return self.parameters()

    def update_statistics(self, history):
        self.normalizer.update(history.flatten(1))

    def prepare_statistics(self, history):
        return (self.normalizer.prepare_update(history.flatten(1)),)

    def apply_statistics(self, statistics):
        self.normalizer.apply_update(statistics[0])


class Critic(nn.Module):
    """价值条件包含本体、视觉、动作ring与相机年龄，部署时移除。"""

    def __init__(self, width=1582, hidden=(512, 256, 128)):
        super().__init__()
        self.normalizer = RunningNorm(width)
        self.net = Mlp([width, *hidden, 1])

    def forward(self, values):
        return self.net(self.normalizer(values)).squeeze(-1)
