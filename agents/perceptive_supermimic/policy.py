# Copyright (c) 2026-2027 zh
"""内容：以普通MLP融合本体历史、机身速度、当前指令及视觉投影，提供策略与价值网络。"""

from dataclasses import asdict
from itertools import chain
import json

import torch
from torch import Tensor, nn

from agents.base.modules.mlp import MLP
from agents.supermimic.world_model import RunningNorm


class PerceptiveActor(nn.Module):
    """采用WMP的历史与指令融合结构；视觉先截断梯度，再经策略专用投影。"""

    def __init__(self, spec, config, encoder):
        super().__init__()
        self.encoder = encoder
        self.observation_dim, self.history_length = spec.observation_dim, spec.history_length
        self.action_dim, self.latent_dim = spec.action_dim, config.latent_dim
        self.command_dim, self.proprio_latent_dim = spec.command_dim, config.proprio_latent_dim
        self.visual_policy_dim = config.visual_policy_dim
        self.history_hidden, self.visual_hidden = tuple(config.history_hidden), tuple(config.visual_hidden)
        self.actor_hidden = tuple(config.actor_hidden)
        self.velocity_dim = 3
        self.input_order = "history_latent_body_velocity_command_projected_visual"
        self.action_clip = float(config.action_clip)
        self.spec_json = json.dumps(asdict(spec), sort_keys=True)
        self.spec_hash, self.semantic_id = spec.hash, spec.semantic_id
        self.depth_semantic_id, self.model_id = spec.depth_semantic_id, spec.model_id
        self.depth_shape = spec.depth_shape
        self.register_buffer("command_indices", torch.tensor(spec.command_indices, dtype=torch.long))
        self.register_buffer("proprio_indices", torch.tensor(
            [index for index in range(spec.observation_dim) if index not in spec.command_indices], dtype=torch.long))
        history_dim = spec.history_length * (spec.observation_dim - spec.command_dim)
        self.normalizer = RunningNorm(history_dim)
        self.history_encoder = MLP(history_dim, self.proprio_latent_dim + self.velocity_dim, self.history_hidden)
        self.visual_projector = MLP(self.latent_dim, self.visual_policy_dim, self.visual_hidden)
        self.actor = MLP(self.proprio_latent_dim + self.velocity_dim + self.command_dim + self.visual_policy_dim,
                         self.action_dim, self.actor_hidden)

    def forward(self, history: Tensor, depth: Tensor) -> Tensor:
        return self.act_latent(history, self.encoder(depth))

    def _proprio_history(self, history: Tensor) -> Tensor:
        frames = history.reshape(-1, self.history_length, self.observation_dim)
        return frames.index_select(-1, self.proprio_indices).flatten(1)

    @torch.jit.export
    def encode_history(self, history: Tensor) -> tuple[Tensor, Tensor]:
        """从包含当前帧的本体历史估计潜特征及同一时刻的机身xyz速度，速度保留m/s量纲。"""
        encoded = self.history_encoder(self.normalizer(self._proprio_history(history)))
        return encoded[:, :self.proprio_latent_dim], encoded[:, self.proprio_latent_dim:]

    @torch.jit.export
    def act_latent(self, history: Tensor, latent: Tensor) -> Tensor:
        frames = history.reshape(-1, self.history_length, self.observation_dim)
        proprio, velocity = self.encode_history(frames)
        command = frames[:, -1].index_select(-1, self.command_indices)
        visual = self.visual_projector(latent.detach())
        features = torch.cat((proprio, velocity, command, visual), dim=-1)
        return self.actor(features).clamp(-self.action_clip, self.action_clip)

    @torch.no_grad()
    def update_statistics(self, history: Tensor):
        self.normalizer.update(self._proprio_history(history))

    def policy_parameters(self):
        return chain(self.history_encoder.parameters(), self.visual_projector.parameters(), self.actor.parameters())


class Critic(nn.Module):
    """价值输入由任务构造，允许使用训练时可得的物理与场景信息。"""

    def __init__(self, input_dim: int, hidden: tuple[int, ...]):
        super().__init__()
        self.normalizer = RunningNorm(input_dim)
        self.net = MLP(input_dim, 1, hidden)

    def forward(self, features: Tensor) -> Tensor:
        return self.net(self.normalizer(features)).squeeze(-1)
