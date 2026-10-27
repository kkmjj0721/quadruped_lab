# Copyright (c) 2026-2027 zh
"""内容：更新物理模型、直接优化历史策略，并拟合固定想象价值目标。"""

from contextlib import contextmanager
from dataclasses import asdict
import math

import torch
from torch import nn

from .config import SingleExecutionConfig
from .imagination import Imagination
from .replay import Seed


@contextmanager
def frozen(*modules):
    parameters = [parameter for module in modules for parameter in module.parameters()]
    previous = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in zip(parameters, previous):
            parameter.requires_grad_(enabled)


def clip_gradients(parameters, norm: float, element: float | None = None):
    parameters = list(parameters)
    value = nn.utils.clip_grad_norm_(parameters, norm, error_if_nonfinite=True)
    if element is not None:
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.clamp_(-element, element)
    return value


class Learner:
    def __init__(self, task, actor, model, critic, config, execution=None):
        self.actor, self.model, self.critic, self.config = actor, model, critic, config
        self.execution = SingleExecutionConfig(**asdict(execution if execution is not None else SingleExecutionConfig()))
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate, betas=(0.9, 0.95))
        self.model_optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.learning_rate)
        self.imagination = Imagination(task, actor, model, critic, config)
        self.actor_updates = 0
        self.model_updates = 0
        self.critic_updates = 0
        self.lr_multiplier = 1.0
        self.ramp_start = 0
        self.actor_frozen_until = 0
        self.policy_start = config.warmup_iterations

    def schedule(self, iteration: int):
        """策略斜率从冻结结束、预热结束和回滚点三者最晚处起算。"""
        cfg = self.config
        progress = min(1.0, iteration / cfg.lr_decay_iterations)
        rate = cfg.learning_rate * (cfg.lr_final_frac + (1 - cfg.lr_final_frac) * .5 * (1 + math.cos(math.pi * progress)))
        rate *= self.lr_multiplier
        start = max(self.ramp_start, self.actor_frozen_until, self.policy_start)
        ramp = min(1.0, max(0.0, iteration - start) / max(1, cfg.actor_ramp_iterations))
        for group in self.actor_optimizer.param_groups:
            group["lr"] = rate * cfg.actor_lr_scale * ramp
        for optimizer in (self.model_optimizer, self.critic_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = rate

    def fit_model(self, window):
        self.model_optimizer.zero_grad(set_to_none=True)
        loss = self.model.loss(window)
        if not torch.isfinite(loss):
            raise FloatingPointError("World-model loss is non-finite")
        loss.backward()
        gradient = clip_gradients(self.model.parameters(), self.config.grad_clip)
        self.model_optimizer.step()
        self.model_updates += 1
        return {"wm_loss": loss.detach(), "wm_grad_norm": gradient.detach(), "wm_valid_steps": window.valid.sum()}

    def improve(self, seed, *, freeze_actor=False):
        cfg = self.config
        batch = seed.history.shape[0]
        if batch < 1:
            raise ValueError("Imagination seed batch must be nonempty")
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.model_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        feature_batches, target_batches = [], []
        metrics = {}
        with frozen(self.model, self.critic):
            # 负载均衡按整个想象批次统计门控使用率，与参考实现的总批量语义一致。
            balance, usage = self.actor.load_balance_loss(seed.history)
            if not torch.isfinite(balance):
                raise FloatingPointError("Load-balance loss is non-finite")
            (balance * cfg.load_balance_coef).backward()
            for start in range(0, batch, self.execution.imag_microbatch):
                end = min(start + self.execution.imag_microbatch, batch)
                weight = (end - start) / batch
                part = Seed(seed.state.index(slice(start, end)), seed.history[start:end])
                data = self.imagination.rollout(part)
                targets = self.imagination.lambda_returns(data)
                loss = -data.objective.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Imagined actor objective is non-finite")
                # 每段保留完整时间反传，尾段按样本数加权；整批仅更新一次策略。
                (loss * weight).backward()
                feature_batches.append(data.features[:, :-1].flatten(0, 1).detach())
                target_batches.append(targets.flatten().detach())
                values = {"control_loss": loss.detach(), "control/load_balance_loss": balance.detach(),
                          "control/gate_usage_max": usage.detach(),
                          "imagination/survival_h": data.survival.detach().mean(),
                          "imagination/discounted_reward": data.discounted_reward.detach().mean(),
                          "imagination/bootstrap": data.bootstrap.detach().mean(),
                          "imagination/reward": data.rewards.mean()}
                for name, value in values.items():
                    metrics[name] = metrics.get(name, 0) + value * weight
                # 下段前释放已反传的计算图，只保留价值监督需要的脱离梯度张量。
                del data, targets, loss, values, part
            gradient = clip_gradients(self.actor.parameters(), cfg.grad_clip, cfg.grad_clip_element)
            if not freeze_actor:
                self.actor_optimizer.step()
            del balance, usage
        if not freeze_actor:
            self.actor_updates += 1
        features = torch.cat(feature_batches)
        targets = torch.cat(target_batches)
        del feature_batches, target_batches
        # 目标已经固定；统计更新不能改变同一次监督的目标值。
        self.critic.normalizer.update(features)
        losses = []
        for _ in range(cfg.critic_epochs):
            indices = torch.randperm(features.shape[0], device=features.device)
            for selected in indices.tensor_split(min(cfg.critic_minibatches, features.shape[0])):
                self.critic_optimizer.zero_grad(set_to_none=True)
                value_loss = features.new_zeros(())
                # 保持原小批边界和优化器步数，仅分拆小批内部的前向与反向峰值。
                for part in selected.split(self.execution.critic_microbatch):
                    partial_loss = (self.critic(features[part]) - targets[part]).square().mean()
                    if not torch.isfinite(partial_loss):
                        raise FloatingPointError("Critic loss is non-finite")
                    weighted_loss = partial_loss * (part.numel() / selected.numel())
                    weighted_loss.backward()
                    value_loss += weighted_loss.detach()
                    del partial_loss, weighted_loss
                clip_gradients(self.critic.parameters(), cfg.grad_clip)
                self.critic_optimizer.step()
                self.critic_updates += 1
                losses.append(value_loss)
        metrics.update({"control_grad_norm": gradient.detach(), "critic_loss": torch.stack(losses).mean(),
                        "control_learning_rate": self.actor_optimizer.param_groups[0]["lr"],
                        "execution/imag_batch": batch,
                        "execution/imag_microbatch": min(batch, self.execution.imag_microbatch),
                        "execution/critic_microbatch": min(self.execution.critic_microbatch,
                                                           math.ceil(features.shape[0] / min(cfg.critic_minibatches, features.shape[0])))})
        return metrics

    def state_dict(self):
        return {"actor": self.actor_optimizer.state_dict(), "model": self.model_optimizer.state_dict(),
                "critic": self.critic_optimizer.state_dict(), "actor_updates": self.actor_updates,
                "model_updates": self.model_updates, "critic_updates": self.critic_updates,
                "lr_multiplier": self.lr_multiplier, "ramp_start": self.ramp_start,
                "actor_frozen_until": self.actor_frozen_until, "policy_start": self.policy_start}

    def load_state_dict(self, data):
        for name in ("actor", "model", "critic"):
            getattr(self, name + "_optimizer").load_state_dict(data[name])
        for name in ("actor_updates", "model_updates", "critic_updates", "lr_multiplier", "ramp_start",
                     "actor_frozen_until", "policy_start"):
            setattr(self, name, data[name])
