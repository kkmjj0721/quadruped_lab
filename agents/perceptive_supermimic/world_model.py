# Copyright (c) 2026-2027 zh
"""内容：学习物理残差、接触和视觉潜变量的可微转移，任务负责坐标与状态积分。"""

import torch
from torch import Tensor, nn
import torch.nn.functional as functional

from agents.supermimic.world_model import RunningNorm

from .interfaces import ModelPrediction, TaskState, VisualWindow


class WorldModel(nn.Module):
    """参考Latent机制联合推进物理状态和视觉；奖励与终止由任务提供。"""

    def __init__(self, task, config):
        super().__init__()
        self.task, self.spec = task, task.spec
        self.latent_dim = config.latent_dim
        layers, width = [], self.spec.model_input_dim + self.latent_dim
        for hidden_width in config.wm_hidden:
            layers.extend((nn.Linear(width, hidden_width), nn.LayerNorm(hidden_width), nn.SiLU()))
            width = hidden_width
        layers.append(nn.Linear(width, self.spec.delta_dim + self.spec.contact_dim + self.latent_dim))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[self.spec.delta_dim:self.spec.delta_dim + self.spec.contact_dim] = -3.0
        self.input_norm = RunningNorm(self.spec.model_input_dim)
        self.output_norm = RunningNorm(self.spec.delta_dim)
        self.register_buffer("z_mean", torch.zeros(self.latent_dim))
        self.register_buffer("z_std", torch.ones(self.latent_dim))
        self.register_buffer("dz_std", torch.ones(self.latent_dim))

    def forward(self, state: TaskState, latent: Tensor, action: Tensor) -> ModelPrediction:
        features = self.input_norm(self.task.model_input(state, action))
        output = self.net(torch.cat((features, (latent - self.z_mean) / self.z_std), dim=-1))
        delta, contacts, dz = output.split((self.spec.delta_dim, self.spec.contact_dim, self.latent_dim), dim=-1)
        predicted = self.task.integrate(state, action, self.output_norm.unnormalize(delta))
        updated = latent + dz * self.dz_std
        if self.spec.visual_interval > 1:
            capture = state.memory[:, self.spec.visual_phase_index] == self.spec.visual_interval - 1
            updated = torch.where(capture[:, None], updated, latent)
        return ModelPrediction(predicted, updated, contacts)

    @torch.no_grad()
    def update_stats(self, window: VisualWindow, initial_latent: Tensor, target_latents: Tensor):
        valid = window.valid
        current, target = window.states.index(valid), window.targets.index(valid)
        self.input_norm.update(self.task.model_input(current, window.actions[valid]))
        self.output_norm.update(self.task.model_target(current, target))
        latent = target_latents[valid]
        previous = torch.cat((initial_latent[:, None], target_latents[:, :-1]), dim=1)
        capture = valid
        if self.spec.visual_interval > 1:
            capture = valid & (window.states.memory[..., self.spec.visual_phase_index] == self.spec.visual_interval - 1)
        dz = (target_latents - previous)[capture]
        # 本次完整窗口的统计固定后再分批反传，潜变量逐维归一化不依赖分批大小。
        self.z_mean.lerp_(latent.mean(0), 0.01)
        self.z_std.lerp_(latent.std(0, correction=0).clamp_min(1e-3), 0.01)
        if dz.shape[0]:
            self.dz_std.lerp_(dz.square().mean(0).sqrt().clamp_min(1e-3), 0.01)

    def loss(self, window: VisualWindow, initial_latent: Tensor, target_latents: Tensor) -> dict[str, Tensor]:
        current, latent = window.initial, initial_latent.detach()
        total = {name: latent.new_zeros(()) for name in ("physical", "contact", "latent")}
        scale = self.output_norm.var.sqrt() + 1e-6
        for step in range(window.actions.shape[1]):
            # 只推进仍有效的行，避免终止后的无监督轨迹继续增长并污染梯度。
            active = window.valid[:, step].nonzero().flatten()
            if active.numel() == 0:
                break
            predicted = self(current.index(active), latent[active], window.actions[active, step])
            recorded = window.states.index((active, step))
            target = window.targets.index((active, step))
            error = self.task.model_error(recorded, predicted.state, target) / scale
            total["physical"] = total["physical"] + error.square().mean(-1).sum()
            if self.spec.contact_dim:
                contact = functional.binary_cross_entropy_with_logits(predicted.contact_logits, window.contacts[active, step], reduction="none")
                total["contact"] = total["contact"] + contact.mean(-1).sum()
            z_error = (predicted.latent - target_latents[active, step].detach()) / self.z_std
            total["latent"] = total["latent"] + z_error.square().mean(-1).sum()
            # 物理量与视觉潜变量回喂预测；外生量及控制记忆恢复记录值。
            recorded_target = window.targets.index((slice(None), step))
            current = TaskState(current.physical.index_copy(0, active, predicted.state.physical),
                                recorded_target.memory, recorded_target.command, recorded_target.context)
            latent = latent.index_copy(0, active, predicted.latent)
        count = window.valid.sum().clamp_min(1)
        return {name: value / count for name, value in total.items()}
