# Copyright (c) 2026-2027 zh
"""内容：执行深度重建、多步动力学、失败头与解析策略梯度更新，保持各自梯度边界。"""

from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F

from agents.supermimic.compilation import compile_kernel
from agents.supermimic.interfaces import TaskState
from agents.supermimic.learner import _step_if_finite
from agents.supermimic.optim import MuonAdam

from .imagine import Imagination, lambda_returns


@contextmanager
def frozen(*modules):
    parameters = [parameter for module in modules for parameter in module.parameters()]
    flags = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)


class DepthLearner:
    def __init__(self, task, actor, model, decoder, critic, failure, config, execution):
        self.task, self.actor, self.model, self.decoder = task, actor, model, decoder
        self.critic, self.failure, self.config, self.execution = critic, failure, config, execution
        self.device = next(model.parameters()).device
        self.model_parameters = [*model.parameters(), *actor.encoder.parameters(), *decoder.parameters()]
        self.actor_parameters = actor.policy_parameters()
        self._device_counts = torch.zeros(5, dtype=torch.int64, device=self.device)
        fused = True if execution.fused_adam and self.device.type == "cuda" else None
        graph = execution.cuda_graph and self.device.type == "cuda"
        # 各优化器持有独立LR地址，防止Actor倍率被其他优化器的基础率覆盖。
        rates = {name: torch.tensor(config.learning_rate, device=self.device) if graph else config.learning_rate
                 for name in ("model", "actor", "critic", "failure")}
        if config.wm_optimizer == "muon_adam":
            # 临时容器只汇总参数，不改变各网络保存的模块键。
            model_modules = nn.ModuleList((model, actor.encoder, decoder))
            self.opt_model = MuonAdam(model_modules, [model.net], config.learning_rate, betas=(0.9, 0.999),
                                      fused=fused, capturable=graph, io=False, weight_decay=config.muon_weight_decay)
        else:
            self.opt_model = torch.optim.Adam(self.model_parameters, lr=rates["model"], fused=fused, capturable=graph)
        if config.actor_optimizer == "muon_adam":
            # encoder 仅归模型重建更新，不能进入策略优化器的 Adam 余项。
            policy_modules = nn.ModuleList((actor.gate, *actor.experts))
            self.opt_actor = MuonAdam(policy_modules, [policy_modules], config.learning_rate, betas=(0.9, 0.95),
                                      fused=fused, capturable=graph, io=True, weight_decay=config.muon_weight_decay)
        else:
            self.opt_actor = torch.optim.AdamW(self.actor_parameters, lr=rates["actor"], betas=(0.9, 0.95), weight_decay=0,
                                              fused=fused, capturable=graph)
        self.opt_critic = torch.optim.Adam(critic.parameters(), lr=rates["critic"], fused=fused, capturable=graph)
        self.opt_failure = torch.optim.Adam(failure.parameters(), lr=rates["failure"], fused=fused, capturable=graph)
        self.error_floor = torch.tensor([3e-3] * 3 + [5e-3] * 3 + [2e-2] * 3 + [5e-2] * 3
                                        + [3e-3] * task.spec.action_dim + [5e-2] * task.spec.action_dim, device=self.device)
        self.imagination = Imagination(task, actor, model, critic, failure, config, execution)
        self.model_loss = compile_kernel(self._model_loss, execution.compile_backend) if execution.compile else self._model_loss
        self.critic_loss = compile_kernel(self._critic_loss, execution.compile_backend) if execution.compile else self._critic_loss
        self.return_kernel = compile_kernel(lambda_returns, execution.compile_backend) if execution.compile else lambda_returns
        self.reset_graphs()
        self.model_updates = self.actor_updates = self.critic_updates = self.failure_updates = 0
        self.actor_skipped = 0

    # 计数留在设备，恢复时沿用原位赋值；只在日志和检查点边界回读。
    @property
    def model_updates(self):
        return self._device_counts[0]

    @model_updates.setter
    def model_updates(self, value):
        self._device_counts[0].copy_(torch.as_tensor(value, device=self.device))

    @property
    def actor_updates(self):
        return self._device_counts[1]

    @actor_updates.setter
    def actor_updates(self, value):
        self._device_counts[1].copy_(torch.as_tensor(value, device=self.device))

    @property
    def critic_updates(self):
        return self._device_counts[2]

    @critic_updates.setter
    def critic_updates(self, value):
        self._device_counts[2].copy_(torch.as_tensor(value, device=self.device))

    @property
    def failure_updates(self):
        return self._device_counts[3]

    @failure_updates.setter
    def failure_updates(self, value):
        self._device_counts[3].copy_(torch.as_tensor(value, device=self.device))

    @property
    def actor_skipped(self):
        return self._device_counts[4]

    @actor_skipped.setter
    def actor_skipped(self, value):
        self._device_counts[4].copy_(torch.as_tensor(value, device=self.device))

    def reset_graphs(self):
        """优化器恢复后重建捕获状态，禁止旧图继续引用被替换的优化器状态。"""
        self.model_update, self.failure_update, self.critic_update = self._model_core, self._failure_core, self._critic_core
        self.graphs = {}
        if self.execution.cuda_graph and self.device.type == "cuda":
            from agents.supermimic.cuda_graph import CapturedUpdate

            for name, function in (("model", self._model_core), ("failure", self._failure_core), ("critic", self._critic_core)):
                self.graphs[name] = CapturedUpdate(function, self.device)
            self.model_update, self.failure_update, self.critic_update = (self.graphs[name] for name in ("model", "failure", "critic"))

    def schedule(self, iteration):
        cfg = self.config
        progress = min(1.0, iteration / cfg.lr_decay_iterations)
        rate = cfg.learning_rate * (cfg.lr_final_frac + (1 - cfg.lr_final_frac) * 0.5 * (1 + math.cos(math.pi * progress)))
        ramp = min(1.0, max(0.0, iteration - cfg.warmup_iterations) / max(1, cfg.actor_ramp_iterations)) if cfg.actor_ramp_iterations else 1.0
        for optimizer in (self.opt_model, self.opt_actor, self.opt_critic, self.opt_failure):
            scheduled = rate * cfg.actor_lr_scale * ramp if optimizer is self.opt_actor else rate
            for group in optimizer.param_groups:
                if isinstance(group["lr"], torch.Tensor):
                    group["lr"].fill_(scheduled)
                else:
                    group["lr"] = scheduled
        return rate

    def _model_loss(self, window, initial_latent, target_latent, recon_stack, recon_target):
        cfg, task, model = self.config, self.task, self.model
        batch, length = window.valid.shape
        state, latent = window.states.index((slice(None), 0)), initial_latent
        states, contacts, latents = [], [], []
        for step in range(length):
            # 上游逐步使用真实窗口的上下文，地形扫描仍在预测状态处查询。
            state = TaskState(state.physical, state.memory, window.states.command[:, step], window.states.context[:, step])
            state, logits, latent = model(state, window.actions[:, step], latent)
            states.append(state)
            contacts.append(logits)
            latents.append(latent)
        predicted = TaskState(*(torch.stack([getattr(item, name) for item in states], 1)
                                for name in ("physical", "memory", "command", "context")))
        flat_predicted = predicted.map(lambda value: value.flatten(0, 1))
        flat_target = window.targets.map(lambda value: value.flatten(0, 1))
        error = (task.model_error(flat_predicted, flat_target) / torch.maximum(model.out_norm.std(), self.error_floor)).square().mean(-1)
        error = error.view(batch, length)
        bce = F.binary_cross_entropy_with_logits(torch.stack(contacts, 1), window.contact, reduction="none").sum(-1)
        dynamics_loss = ((error + 0.1 * bce) * window.valid).sum() / batch
        auxiliary_error = ((task.auxiliary(flat_predicted) - task.auxiliary(flat_target)) / model.aux_norm.std()).square().mean(-1)
        auxiliary_loss = (auxiliary_error.reshape(batch, length) * window.valid).sum() / batch
        latent_count = target_latent.shape[0]
        latent_scale = target_latent.var((0, 1)).mean().clamp_min(1e-6)
        latent_error = (torch.stack(latents, 1)[:latent_count] - target_latent).square().mean(-1) / latent_scale
        latent_mask = window.valid[:latent_count]
        latent_loss = (latent_error * latent_mask).sum() / latent_count
        # 干净图只作为标签；编码器和潜变量动力学始终消费实际带噪观测。
        clean = recon_target.detach()
        valid_depth = clean > 0
        valid_count = valid_depth.sum().clamp_min(1)
        reconstruction = ((self.decoder(self.actor.encoder(recon_stack)) - clean).abs() * valid_depth).sum() / valid_count
        noisy_input = ((recon_stack.detach() - clean).abs() * valid_depth).sum() / valid_count
        total = dynamics_loss + cfg.auxiliary_weight * auxiliary_loss + cfg.latent_weight * latent_loss + cfg.latent_recon_weight * reconstruction
        metrics = torch.stack((dynamics_loss / length, (error[:, -1] * window.valid[:, -1]).sum() / window.valid[:, -1].sum().clamp_min(1),
                               (bce * window.valid).sum() / batch, auxiliary_loss / length,
                               (latent_error * latent_mask).sum() / latent_mask.sum().clamp_min(1),
                               (latent_error[:, -1] * latent_mask[:, -1]).sum() / latent_mask[:, -1].sum().clamp_min(1),
                               reconstruction, noisy_input))
        return total, predicted, metrics

    def _model_core(self, window, initial_latent, target_latent, recon_stack, recon_target):
        self.opt_model.zero_grad(set_to_none=True)
        total, predicted, metrics = self.model_loss(window, initial_latent, target_latent, recon_stack, recon_target)
        total.backward()
        gradient = nn.utils.clip_grad_norm_(self.model_parameters, self.config.grad_clip)
        self.opt_model.step()
        return predicted.map(lambda value: value.detach()), metrics.detach(), gradient.detach()

    def _failure_core(self, true_features, predicted_features, label, weight):
        positive = torch.where(label > 0.5, self.config.fail_pos_weight, 1.0)
        first = F.binary_cross_entropy_with_logits(self.failure(true_features), label, reduction="none")
        second = F.binary_cross_entropy_with_logits(self.failure(predicted_features), label, reduction="none")
        loss = ((first + second) * positive * weight).sum() / weight.sum().clamp_min(1e-6)
        self.opt_failure.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.failure.parameters(), self.config.grad_clip)
        self.opt_failure.step()
        return loss.detach() / 2

    def fit_model(self, window):
        cfg = self.config
        count, length = min(cfg.latent_windows, window.valid.shape[0]), window.valid.shape[1]
        with torch.no_grad():
            initial_latent = self.actor.encoder(window.depth[:, 0].float() / 255)
            target_latent = self.actor.encoder(window.next_depth[:count].flatten(0, 1).float() / 255).view(count, length, -1)
            difference = target_latent - torch.cat((initial_latent[:count, None], target_latent[:, :-1]), 1)
            self.model.update_z_statistics(target_latent, difference, window.valid[:count])
            recon_count = min(cfg.latent_recon_images, count * length)
            recon_stack = window.next_depth[:count].flatten(0, 1)[:recon_count].float() / 255
            recon_target = window.next_clean_depth[:count].flatten(0, 1)[:recon_count].float() / 255
        predicted, metrics, gradient = self.model_update(window, initial_latent, target_latent, recon_stack, recon_target)
        failure_count = min(cfg.fail_windows, window.valid.shape[0])
        with torch.no_grad():
            recorded = window.targets.index(slice(0, failure_count)).map(lambda value: value.flatten(0, 1))
            predicted = predicted.index(slice(0, failure_count)).map(lambda value: value.flatten(0, 1))
            command = window.states.command[:failure_count].flatten(0, 1)
            # R1 两路失败监督共享实际执行指令及终止前计时，预测物理状态仍保持独立。
            true_features = self.task.failure_features(recorded, recorded, command)
            predicted_features = self.task.failure_features(predicted, recorded, command)
            self.failure.norm.update(true_features)
            label = window.terminated[:failure_count].float().flatten()
            weight = (window.valid[:failure_count] * window.weight[:failure_count, None]).flatten()
        failure_loss = self.failure_update(true_features, predicted_features, label, weight)
        self.model_updates.add_(1)
        self.failure_updates.add_(1)
        names = ("wm/loss", "wm/err_last", "wm/contact_bce", "wm/auxiliary_loss", "latent/err_window", "latent/err_last",
                 "latent/reconstruction_l1", "latent/noisy_input_l1")
        return {**dict(zip(names, metrics.clone().unbind())), "wm/grad_norm": gradient.clone(), "failure/loss": failure_loss.clone()}

    def _critic_loss(self, features, targets):
        return (self.critic(features) - targets).square().mean()

    def _actor_core(self, seed, observation_noise, *, use_failure_head):
        cfg = self.config
        batch = seed.history.shape[0]
        self.opt_actor.zero_grad(set_to_none=True)
        feature_batches, reward_batches, failure_batches, losses, survivals = [], [], [], [], []
        with frozen(self.model, self.critic, self.failure):
            for start in range(0, batch, self.execution.imag_microbatch):
                part = seed.index(slice(start, start + self.execution.imag_microbatch))
                data = self.imagination.rollout(part, use_failure_head=use_failure_head,
                                                observation_noise=observation_noise[:, start:start + part.history.shape[0]])
                loss = -data.objective.mean() * (part.history.shape[0] / batch)
                loss.backward()
                losses.append(loss.detach())
                feature_batches.append(data.features)
                reward_batches.append(data.rewards)
                failure_batches.append(data.failures)
                survivals.append(data.survival)
            gradient = nn.utils.clip_grad_norm_(self.actor_parameters, cfg.grad_clip)
            nn.utils.clip_grad_value_(self.actor_parameters, cfg.grad_clip_element)
            finite = torch.isfinite(gradient)
            # R1 只在 actor 梯度有限时执行策略更新。
            _step_if_finite(self.opt_actor, finite)
        metrics = (torch.stack(losses).sum(), gradient.detach(),
                   torch.cat(reward_batches).mean() / self.task.dt, torch.cat(survivals).mean())
        return torch.stack(metrics), torch.cat(feature_batches), torch.cat(reward_batches), torch.cat(failure_batches), finite

    def _critic_core(self, features, rewards, failures):
        cfg = self.config
        with torch.no_grad():
            # R1 先用 H+1 个状态更新统计，再固定当前 critic 的 λ-return 目标。
            self.critic.norm.update(features.transpose(0, 1))
            values = self.critic(features)
            targets = self.return_kernel(rewards, failures, values, cfg.gamma, cfg.lam)
            flat_features = features[:, :-1].transpose(0, 1).reshape(-1, features.shape[-1])
            flat_targets = targets.transpose(0, 1).reshape(-1)
        count = flat_features.shape[0]
        width = count // cfg.critic_minibatches
        value_losses = []
        for _ in range(cfg.critic_epochs):
            permutation = torch.randperm(count, device=self.device)
            for index in range(cfg.critic_minibatches):
                selected = permutation[index * width:(index + 1) * width]
                self.opt_critic.zero_grad(set_to_none=True)
                total = flat_features.new_zeros(())
                # 尾部微批按样本占比加权，始终只产生本次小批的一次优化器更新。
                for start in range(0, selected.numel(), self.execution.critic_microbatch):
                    part = selected[start:start + self.execution.critic_microbatch]
                    loss = self.critic_loss(flat_features[part], flat_targets[part])
                    weighted = loss * (part.numel() / selected.numel())
                    weighted.backward()
                    total += weighted.detach()
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
                self.opt_critic.step()
                value_losses.append(total)
        return torch.stack(value_losses).mean()

    def improve(self, seed, iteration):
        batch = seed.history.shape[0]
        use_failure_head = iteration >= self.config.fail_head_warmup_iterations
        observation_noise = self.imagination.sample_noise(batch, seed.history)
        if self.execution.cuda_graph and self.device.type == "cuda":
            from agents.supermimic.cuda_graph import CapturedUpdate

            name = "actor_true" if use_failure_head else "actor_false"
            if name not in self.graphs:
                self.graphs[name] = CapturedUpdate(
                    lambda value, noise: self._actor_core(value, noise, use_failure_head=use_failure_head), self.device)
            actor_update = self.graphs[name]
        else:
            actor_update = lambda value, noise: self._actor_core(value, noise, use_failure_head=use_failure_head)
        metrics, features, rewards, failures, finite = actor_update(seed, observation_noise)
        self.actor_updates.add_(finite)
        self.actor_skipped.add_(~finite)
        value_loss = self.critic_update(features, rewards, failures)
        self.critic_updates.add_(self.config.critic_epochs * self.config.critic_minibatches)
        names = ("policy/loss", "policy/grad_norm", "imagination/reward", "imagination/survival")
        return {**dict(zip(names, metrics.clone().unbind())), "critic/loss": value_loss.clone()}
