# Copyright (c) 2026-2027 zh
"""
内容：
    组合 MGDP 双帧编码器、固定 GRU 与 Parkour Latent 的四专家策略和增量世界模型。
    来源版本和许可见 NOTICE.md；状态推进与奖励由任务协议提供。
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn


def mlp(input_dim, hidden, output_dim, activation=nn.ELU, layer_norm=False):
    """保留参考实现每层 Linear、可选 LayerNorm、激活的顺序。"""
    layers, width = [], input_dim
    for size in hidden:
        layers.append(nn.Linear(width, size))
        if layer_norm:
            layers.append(nn.LayerNorm(size))
        layers.append(activation())
        width = size
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """使用参考总体方差、零初始计数和 eps²；原位更新以保持图地址。"""

    def __init__(self, dim, max_count=2e8, eps=1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps

    @torch.no_grad()
    def update(self, values):
        values = values.reshape(-1, values.shape[-1]).float()
        size = values.shape[0]
        mean, variance = values.mean(0), values.var(0, unbiased=False)
        total = self.count + size
        difference = mean - self.mean
        active = self.count < self.max_count
        # Graph重放也按当前设备计数冻结，保持原有整批更新边界。
        updated_mean = self.mean + difference * size / total
        updated_var = (self.var * self.count + variance * size + difference.square() * self.count * size / total) / total
        self.mean.copy_(torch.where(active, updated_mean, self.mean))
        self.var.copy_(torch.where(active, updated_var, self.var))
        self.count.copy_(torch.where(active, total, self.count))

    def forward(self, values):
        return (values - self.mean) / self.std()

    def std(self):
        return torch.sqrt(self.var + self.eps**2)


def run_network(network, values, bf16):
    """仅矩阵或卷积使用局部 BF16，返回 FP32 后才进行任务计算。"""
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False) if bf16 else nullcontext():
        return network(values).float()


class DoubleConv(nn.Module):
    """沿用 MGDP 的无偏置卷积、八组归一化与 SiLU，连续执行两次。"""

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                                   nn.GroupNorm(8, output_channels), nn.SiLU(inplace=True),
                                   nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                                   nn.GroupNorm(8, output_channels), nn.SiLU(inplace=True))

    def forward(self, values):
        return self.block(values)


class DepthEncoder(nn.Module):
    """按 MGDP 固定版本提取双帧 CNN token；GRU 记忆由策略显式维护。"""

    def __init__(self, depth_shape=(2, 32, 32), latent_dim=64, *, bf16=False):
        super().__init__()
        self.conv1 = DoubleConv(depth_shape[0], 32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = DoubleConv(64, 96)
        self.bottleneck = DoubleConv(96, 96)
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(nn.Linear(96 * 4 * 4, 192), nn.LayerNorm(192), nn.SiLU(inplace=True),
                                nn.Linear(192, latent_dim))
        self.bf16 = bf16

    def forward(self, depth):
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False) if self.bf16 else nullcontext():
            visual = self.conv1(depth)
            visual = self.conv2(self.pool1(visual))
            visual = self.bottleneck(self.conv3(self.pool2(visual)))
            return self.fc(self.global_pool(visual).flatten(1)).float()


class DepthActor(nn.Module):
    """显式推进固定 GRU，将记忆与本体历史送入四个独立动作专家。"""

    def __init__(self, spec, config, *, bf16=False, encoder_bf16=False):
        super().__init__()
        self.term_dims = tuple(spec.observation_term_dims)
        self.depth_shape = tuple(spec.depth_shape)
        self.encoder = DepthEncoder(spec.depth_shape, spec.latent_dim, bf16=encoder_bf16)
        self.memory = nn.GRUCell(spec.latent_dim, spec.latent_dim).requires_grad_(False)
        input_dim = spec.history_length * spec.observation_dim + spec.latent_dim
        self.gate = nn.Linear(input_dim, config.actor_experts)
        self.experts = nn.ModuleList([mlp(input_dim, config.actor_hidden, spec.action_dim) for _ in range(config.actor_experts)])
        self.bf16 = bf16

    def act_latent(self, history, latent):
        terms = [part.flatten(1) for part in history.split(self.term_dims, -1)]
        values = torch.cat((*terms, latent), -1)
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False) if self.bf16 else nullcontext():
            logits = self.gate(values)
            outputs = torch.stack([expert(values) for expert in self.experts], 1)
        weights, outputs = logits.float().softmax(-1), outputs.float()
        return torch.einsum("be,beo->bo", weights, outputs)

    @torch.no_grad()
    def remember(self, token, hidden):
        """保留参考版本的有效梯度边界；GRU 权重和递推状态均不反传。"""
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False) if self.encoder.bf16 else nullcontext():
            return self.memory(token, hidden).float()

    def forward(self, proprio_history, depth_stack, hidden):
        hidden = self.remember(self.encoder(depth_stack), hidden)
        return self.act_latent(proprio_history, hidden), hidden

    def policy_parameters(self):
        """编码器只由重建优化器负责；固定 GRU 不进入策略参数集合。"""
        return [*self.gate.parameters(), *self.experts.parameters()]


class LatentDecoder(nn.Module):
    """仅训练使用；从 8×8 起点经两层转置卷积重建双帧 32×32。"""

    def __init__(self, latent_dim=64, channels=2, *, bf16=False):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, 32 * 8 * 8), nn.ELU())
        self.dec = nn.Sequential(nn.ConvTranspose2d(32, 32, 3, 2, 1, output_padding=1), nn.ELU(),
                                 nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1), nn.ELU(),
                                 nn.Conv2d(16, channels, 3, 1, 1))
        self.bf16 = bf16

    def forward(self, latent):
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False) if self.bf16 else nullcontext():
            return self.dec(self.fc(latent).view(-1, 32, 8, 8)).sigmoid().float()


class WorldModel(nn.Module):
    """全状态独立增量、接触及视觉增量；aux40 仅适配指定 Go2 奖励所需真值。"""

    def __init__(self, task, hidden=(1024, 1024, 1024), *, bf16=False):
        super().__init__()
        self.task, self.spec, self.bf16 = task, task.spec, bf16
        spec = self.spec
        self.net = mlp(spec.model_input_dim + spec.latent_dim, hidden,
                       spec.delta_dim + spec.contact_dim + spec.latent_dim + spec.auxiliary_dim, nn.SiLU, True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[spec.delta_dim:spec.delta_dim + spec.contact_dim] = -3.0
        self.in_norm = RunningNorm(spec.model_input_dim)
        self.out_norm = RunningNorm(spec.delta_dim, eps=1e-4)
        self.aux_norm = RunningNorm(spec.auxiliary_dim, eps=1e-4)
        self.register_buffer("z_mean", torch.zeros(spec.latent_dim))
        self.register_buffer("z_std", torch.ones(spec.latent_dim))
        self.register_buffer("dz_std", torch.ones(spec.latent_dim))

    @torch.no_grad()
    def update_statistics(self, state, action, target):
        self.in_norm.update(self.task.model_input(state, action))
        self.out_norm.update(self.task.model_target(state, target))
        self.aux_norm.update(self.task.auxiliary(target))

    @torch.no_grad()
    def update_z_statistics(self, target, difference, valid, rate=0.01):
        target, difference, valid = target.flatten(0, 1), difference.flatten(0, 1), valid.reshape(-1, 1)
        self.z_mean.lerp_(target.mean(0), rate)
        self.z_std.lerp_(target.std(0).clamp_min(1e-3), rate)
        self.dz_std.lerp_((difference.square().mul(valid).sum(0) / valid.sum().clamp_min(1)).sqrt().clamp_min(1e-3), rate)

    def forward(self, state, action, latent):
        inputs = self.in_norm(self.task.model_input(state, action))
        output = run_network(self.net, torch.cat((inputs, (latent - self.z_mean) / self.z_std), -1), self.bf16)
        delta, logits, latent_delta, auxiliary = output.split((self.spec.delta_dim, self.spec.contact_dim,
                                                              self.spec.latent_dim, self.spec.auxiliary_dim), -1)
        delta = delta * self.out_norm.std() + self.out_norm.mean
        auxiliary = auxiliary * self.aux_norm.std() + self.aux_norm.mean
        predicted = self.task.integrate(state, action, delta, logits.sigmoid(), auxiliary)
        return predicted, logits, latent + latent_delta * self.dz_std


class Critic(nn.Module):
    def __init__(self, input_dim, hidden=(256, 128), *, bf16=False):
        super().__init__()
        self.norm = RunningNorm(input_dim)
        self.net = mlp(input_dim, hidden, 1)
        self.bf16 = bf16

    def forward(self, features):
        return run_network(self.net, self.norm(features), self.bf16).squeeze(-1)


class FailureHead(Critic):
    """与参考 FailureHead 使用相同归一化、256/128 MLP 和未激活 logit。"""
