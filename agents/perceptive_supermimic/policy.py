# Copyright (c) 2026-2027 zh
"""内容：融合本体历史与截断梯度的视觉特征，提供动作策略和训练用价值网络。"""

from dataclasses import asdict
from itertools import chain
import json

import torch
from torch import Tensor, nn

from agents.base.modules.mlp import MLP
from agents.supermimic.world_model import RunningNorm


class PerceptiveActor(nn.Module):
    """CNN由重建损失训练，动作目标只训练本体编码器与actor。"""

    def __init__(self, spec, config, encoder):
        super().__init__()
        self.encoder = encoder
        self.observation_dim, self.history_length = spec.observation_dim, spec.history_length
        self.action_dim, self.latent_dim = spec.action_dim, config.latent_dim
        self.action_clip = float(config.action_clip)
        self.spec_json = json.dumps(asdict(spec), sort_keys=True)
        self.spec_hash, self.semantic_id = spec.hash, spec.semantic_id
        self.depth_semantic_id, self.model_id = spec.depth_semantic_id, spec.model_id
        self.depth_shape = spec.depth_shape
        self.normalizer = RunningNorm(spec.history_length * spec.observation_dim)
        self.frame_normalizer = RunningNorm(spec.observation_dim)
        self.history_encoder = MLP(spec.history_length * spec.observation_dim, config.proprio_latent_dim, config.history_hidden)
        self.actor = MLP(spec.observation_dim + config.proprio_latent_dim + config.latent_dim, spec.action_dim, config.actor_hidden)

    def forward(self, history: Tensor, depth: Tensor) -> Tensor:
        return self.act_latent(history, self.encoder(depth))

    @torch.jit.export
    def act_latent(self, history: Tensor, latent: Tensor) -> Tensor:
        flattened = history.reshape(-1, self.history_length * self.observation_dim)
        proprio = self.history_encoder(self.normalizer(flattened))
        current = self.frame_normalizer(flattened[:, -self.observation_dim:])
        return self.actor(torch.cat((current, proprio, latent.detach()), dim=-1)).clamp(-self.action_clip, self.action_clip)

    @torch.no_grad()
    def update_statistics(self, history: Tensor):
        flattened = history.reshape(-1, self.history_length * self.observation_dim)
        self.normalizer.update(flattened)
        self.frame_normalizer.update(flattened[:, -self.observation_dim:])

    def policy_parameters(self):
        return chain(self.history_encoder.parameters(), self.actor.parameters())


class Critic(nn.Module):
    """价值输入由任务构造，允许使用训练时可得的物理与场景信息。"""

    def __init__(self, input_dim: int, hidden: tuple[int, ...]):
        super().__init__()
        self.normalizer = RunningNorm(input_dim)
        self.net = MLP(input_dim, 1, hidden)

    def forward(self, features: Tensor) -> Tensor:
        return self.net(self.normalizer(features)).squeeze(-1)
