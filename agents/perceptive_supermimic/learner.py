# Copyright (c) 2026-2027 zh
"""内容：分别优化模型与视觉重建、想象策略和固定目标的价值回归。"""

from itertools import chain
import math

import torch
from torch import nn

from agents.supermimic.learner import frozen

from .config import ExecutionConfig
from .imagine import Imagination
from .terrain_perception import contrastive_loss


def _backward(loss):
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Training loss is non-finite")
    loss.backward()


def _clip(parameters, norm, value=None):
    gradient = nn.utils.clip_grad_norm_(parameters, norm)
    if not bool(torch.isfinite(gradient)):
        raise FloatingPointError("Training gradients are non-finite")
    if value is not None:
        nn.utils.clip_grad_value_(parameters, value)
    return gradient.detach()


class Learner:
    """微批只降低激活峰值；整批监督、目标与优化器步数保持一致。"""

    def __init__(self, task, actor, model, critic, decoder, config, execution=None, *, terrain_perception=None):
        self.task, self.actor, self.model, self.critic, self.decoder = task, actor, model, critic, decoder
        self.config, self.execution = config, execution or ExecutionConfig()
        self.terrain_perception = terrain_perception
        self.model_parameters = list(chain(model.parameters(), actor.encoder.parameters(), decoder.parameters()))
        if terrain_perception is not None:
            self.model_parameters.extend(terrain_perception.parameters())
        self.actor_parameters = list(actor.policy_parameters())
        self.critic_parameters = list(critic.parameters())
        self.model_optimizer = torch.optim.Adam(self.model_parameters, lr=config.learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor_parameters, lr=config.learning_rate, betas=(0.9, 0.95))
        self.critic_optimizer = torch.optim.Adam(self.critic_parameters, lr=config.learning_rate)
        self.imagination = Imagination(task, actor, model, critic, config)
        self.model_updates = self.actor_updates = self.critic_updates = 0

    def schedule(self, iteration):
        progress = min(1.0, iteration / self.config.lr_decay_iterations)
        ratio = self.config.lr_final_ratio + (1 - self.config.lr_final_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        for optimizer in (self.model_optimizer, self.actor_optimizer, self.critic_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = self.config.learning_rate * ratio

    @torch.no_grad()
    def _encode(self, images):
        return torch.cat([self.actor.encoder(part) for part in images.split(self.execution.vision_batch)], 0)

    def _fit_perception(self, images, clean, heights):
        """先求整批对比特征梯度，再微批重算；负样本池和累计梯度不随微批大小改变。"""
        config, execution = self.config, self.execution
        count = images.shape[0]
        metrics = {name: images.new_zeros(()) for name in ("reconstruction", "height_reconstruction", "contrastive")}
        if config.contrastive_weight:
            visual = self._encode(images).requires_grad_()
            with torch.no_grad():
                terrain = torch.cat([self.terrain_perception.encoder(part) for part in heights.split(execution.vision_batch)], 0)
            terrain.requires_grad_()
            contrastive = contrastive_loss(visual, terrain, config.contrastive_temperature)
            visual_gradient, terrain_gradient = torch.autograd.grad(contrastive, (visual, terrain))
            metrics["contrastive"] = contrastive.detach()
        for start in range(0, count, execution.vision_batch):
            stop = min(count, start + execution.vision_batch)
            weight = (stop - start) / count
            visual = self.actor.encoder(images[start:stop])
            error = self.decoder(visual) - clean[start:stop]
            reconstruction = error.abs().mean() if config.depth_reconstruction_loss == "l1" else error.square().mean()
            loss = config.reconstruction_weight * reconstruction * weight
            metrics["reconstruction"] += reconstruction.detach() * weight
            if self.terrain_perception is not None:
                terrain = self.terrain_perception.encoder(heights[start:stop])
                height_loss = (self.terrain_perception.reconstruct(terrain) - heights[start:stop]).square().mean()
                loss = loss + config.height_reconstruction_weight * height_loss * weight
                metrics["height_reconstruction"] += height_loss.detach() * weight
                if config.contrastive_weight:
                    # 特征梯度已含整批均值，不能再乘微批权重；此内积只用于恢复参数梯度。
                    loss = loss + config.contrastive_weight * (
                        (visual * visual_gradient[start:stop]).sum() + (terrain * terrain_gradient[start:stop]).sum())
            _backward(loss)
        return metrics

    def fit_model(self, window):
        config, execution = self.config, self.execution
        batch, steps = window.valid.shape
        # 图像与高度都取有效终态；静态场景由任务按同一位姿重算，标签不参与反传。
        indices = window.valid.flatten().nonzero().flatten()
        indices = indices[torch.randperm(indices.numel(), device=indices.device)[:config.reconstruction_images]]
        images = window.depth_next.flatten(0, 1)[indices]
        clean = window.depth_next_clean.flatten(0, 1)[indices]
        heights = None
        if self.terrain_perception is not None:
            with torch.no_grad():
                heights = self.task.terrain_height(window.targets.index((indices // steps, indices % steps))).detach()
            if heights.shape != (indices.numel(), 1, *config.height_shape) or heights.dtype != torch.float32 or heights.device != images.device:
                raise ValueError("Terrain height targets must match [batch,1,*height_shape], float32 dtype and the image device")
            if not torch.isfinite(heights).all() or ((heights < -1) | (heights > 1)).any():
                raise ValueError("Terrain height targets must be finite and normalized to [-1,1]")
        # 同一次优化的潜变量目标由同一版CNN产生，统计在任何反传之前统一更新。
        initial = self._encode(window.depth)
        targets = self._encode(window.depth_next.flatten(0, 1)).reshape(batch, steps, -1)
        self.model.update_stats(window, initial, targets)
        self.model_optimizer.zero_grad(set_to_none=True)
        total_steps = window.valid.sum()
        metrics = {name: window.actions.new_zeros(()) for name in ("physical", "contact", "latent")}
        for start in range(0, batch, execution.wm_microbatch):
            stop = min(batch, start + execution.wm_microbatch)
            part = window.index(slice(start, stop))
            weight = part.valid.sum() / total_steps
            losses = self.model.loss(part, initial[start:stop], targets[start:stop])
            loss = losses["physical"] + config.contact_weight * losses["contact"] + config.latent_weight * losses["latent"]
            _backward(loss * weight)
            for name, value in losses.items():
                metrics[name] += value.detach() * weight
        metrics.update(self._fit_perception(images, clean, heights))
        metrics["model_grad_norm"] = _clip(self.model_parameters, config.grad_clip)
        self.model_optimizer.step()
        self.model_updates += 1
        metrics["model_loss"] = (metrics["physical"] + config.contact_weight * metrics["contact"]
                                 + config.latent_weight * metrics["latent"] + config.reconstruction_weight * metrics["reconstruction"]
                                 + config.height_reconstruction_weight * metrics["height_reconstruction"]
                                 + config.contrastive_weight * metrics["contrastive"])
        return metrics

    def improve(self, seed):
        config, execution = self.config, self.execution
        batch = seed.history.shape[0]
        # 任务输出是监督边界，整批校验一次后各微批沿用同一组真实标签。
        with torch.no_grad():
            target_velocity = self.task.body_linear_velocity(seed.state).detach()
        if target_velocity.shape != (batch, 3) or target_velocity.dtype != torch.float32 or target_velocity.device != seed.history.device:
            raise ValueError("Body velocity targets must have shape [batch, 3], float32 dtype and the seed device")
        if not torch.isfinite(target_velocity).all():
            raise FloatingPointError("Body velocity targets are non-finite")
        self.model_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        features, targets = [], []
        control_loss = seed.history.new_zeros(())
        reward_mean = control_loss.clone()
        velocity_mse_axes = seed.history.new_zeros(3)
        with frozen(self.model, self.actor.encoder, self.critic):
            for start in range(0, batch, execution.imag_microbatch):
                stop = min(batch, start + execution.imag_microbatch)
                part = seed.index(slice(start, stop))
                imagined = self.imagination.rollout(part)
                weight = (stop - start) / batch
                policy_loss = -imagined.objective.mean()
                _, velocity = self.actor.encode_history(part.history)
                # 标签来自回放当前真实状态；不以想象速度或下一状态代替同一时刻监督。
                velocity_error = (velocity - target_velocity[start:stop]).square()
                loss = policy_loss + config.velocity_loss_coef * velocity_error.mean()
                _backward(loss * weight)
                control_loss += policy_loss.detach() * weight
                velocity_mse_axes += velocity_error.detach().mean(0) * weight
                reward_mean += imagined.rewards.mean() * weight
                features.append(imagined.features[:, :-1].flatten(0, 1))
                targets.append(self.imagination.lambda_returns(imagined).flatten())
                del imagined, loss, policy_loss, velocity, velocity_error, part
            actor_gradient = _clip(self.actor_parameters, config.grad_clip, config.actor_grad_value)
            self.actor_optimizer.step()
            self.actor_updates += 1
        features, targets = torch.cat(features), torch.cat(targets)
        # 价值快照和lambda目标已经固定，之后再更新critic输入统计和参数。
        self.critic.normalizer.update(features)
        value_loss = features.new_zeros(())
        minibatches = config.critic_epochs * config.critic_minibatches
        for _ in range(config.critic_epochs):
            indices = torch.randperm(features.shape[0], device=features.device)
            for selected in torch.tensor_split(indices, config.critic_minibatches):
                self.critic_optimizer.zero_grad(set_to_none=True)
                minibatch_loss = features.new_zeros(())
                for part in selected.split(execution.critic_microbatch):
                    loss = (self.critic(features[part]) - targets[part]).square().mean()
                    weight = part.numel() / selected.numel()
                    _backward(loss * weight)
                    minibatch_loss += loss.detach() * weight
                _clip(self.critic_parameters, config.grad_clip)
                self.critic_optimizer.step()
                self.critic_updates += 1
                value_loss += minibatch_loss / minibatches
        velocity_loss = velocity_mse_axes.mean()
        return {"control_loss": control_loss, "velocity_loss": velocity_loss,
                "actor_loss": control_loss + config.velocity_loss_coef * velocity_loss,
                "velocity_rmse": velocity_loss.sqrt(),
                **{f"velocity_rmse_{axis}": value for axis, value in zip(("x", "y", "z"), velocity_mse_axes.sqrt())},
                "critic_loss": value_loss, "actor_grad_norm": actor_gradient, "imagined_reward": reward_mean}
