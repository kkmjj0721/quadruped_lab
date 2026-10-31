# Copyright (c) 2026-2027 zh
"""内容：更新物理模型、直接优化历史策略，并拟合固定想象价值目标。"""

from contextlib import contextmanager
from dataclasses import asdict
import math

import torch
from torch import nn

from .buffer import Seed
from .config import ExecutionConfig
from .imagine import Imagination


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
    value = nn.utils.clip_grad_norm_(parameters, norm)
    if element is not None:
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.clamp_(-element, element)
    return value


def _finite_metrics(metrics):
    """非有限更新由计数明确记录，不把其NaN标量送入要求有限值的日志接口。"""
    return {name: value for name, value in metrics.items() if bool(torch.isfinite(torch.as_tensor(value)).all())}


class Learner:
    def __init__(self, task, actor, model, critic, config, execution=None):
        self.actor, self.model, self.critic, self.config = actor, model, critic, config
        self.execution = ExecutionConfig(**asdict(execution if execution is not None else ExecutionConfig()))
        # 关闭时沿用原默认自动选择foreach；显式False会改为单张量更新。
        fused = True if self.execution.fused_adam else None
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate, betas=(0.9, 0.95),
                                                fused=fused)
        self.model_optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, fused=fused)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.learning_rate, fused=fused)
        self.imagination = Imagination(task, actor, model, critic, config)
        self.actor_updates = 0
        self.model_updates = 0
        self.critic_updates = 0
        self.nonfinite_steps = 0
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
        loss.backward()
        gradient = clip_gradients(self.model.parameters(), self.config.grad_clip)
        finite = bool(torch.isfinite(gradient))
        if finite:
            self.model_optimizer.step()
            self.model_updates += 1
        else:
            self.nonfinite_steps += 1
        return _finite_metrics({"wm_loss": loss.detach(), "wm_grad_norm": gradient.detach(),
                                "wm_valid_steps": window.valid.sum(), "wm_skipped": float(not finite)})

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
            for start in range(0, batch, self.execution.imag_microbatch):
                end = min(start + self.execution.imag_microbatch, batch)
                weight = (end - start) / batch
                part = Seed(seed.state.index(slice(start, end)), seed.history[start:end])
                data = self.imagination.rollout(part)
                targets = self.imagination.lambda_returns(data)
                loss = -data.objective.mean()
                # 每段保留完整时间反传，尾段按样本数加权；整批仅更新一次策略。
                (loss * weight).backward()
                feature_batches.append(data.features[:, :-1].flatten(0, 1).detach())
                target_batches.append(targets.flatten().detach())
                values = {"control_loss": loss.detach(),
                          "imagination/survival_h": data.survival.detach().mean(),
                          "imagination/discounted_reward": data.discounted_reward.detach().mean(),
                          "imagination/bootstrap": data.bootstrap.detach().mean(),
                          "imagination/reward": data.rewards.mean()}
                for name, value in values.items():
                    metrics[name] = metrics.get(name, 0) + value * weight
                # 下段前释放已反传的计算图，只保留价值监督需要的脱离梯度张量。
                del data, targets, loss, values, part
            gradient = clip_gradients(self.actor.parameters(), cfg.grad_clip, cfg.grad_clip_element)
            finite = bool(torch.isfinite(gradient)) and bool(torch.isfinite(metrics["control_loss"]))
            if not finite:
                self.nonfinite_steps += 1
            if not freeze_actor and finite:
                self.actor_optimizer.step()
        if not freeze_actor and finite:
            self.actor_updates += 1
        features = torch.cat(feature_batches)
        targets = torch.cat(target_batches)
        del feature_batches, target_batches
        # 目标已经固定；统计更新不能改变同一次监督的目标值。
        self.critic.normalizer.update(features)
        losses = []
        skipped = 0
        minibatch = features.shape[0] // cfg.critic_minibatches
        if minibatch == 0:
            raise ValueError("Imagined transitions must fill at least one row per critic minibatch")
        for _ in range(cfg.critic_epochs):
            indices = torch.randperm(features.shape[0], device=features.device)
            # 参考按整除的小批宽度取片，余数不参与本轮critic监督。
            for index in range(cfg.critic_minibatches):
                selected = indices[index * minibatch:(index + 1) * minibatch]
                self.critic_optimizer.zero_grad(set_to_none=True)
                value_loss = features.new_zeros(())
                # 保持原小批边界和优化器步数，仅分拆小批内部的前向与反向峰值。
                for part in selected.split(self.execution.critic_microbatch):
                    partial_loss = (self.critic(features[part]) - targets[part]).square().mean()
                    weighted_loss = partial_loss * (part.numel() / selected.numel())
                    weighted_loss.backward()
                    value_loss += weighted_loss.detach()
                    del partial_loss, weighted_loss
                critic_gradient = clip_gradients(self.critic.parameters(), cfg.grad_clip)
                if bool(torch.isfinite(critic_gradient)):
                    self.critic_optimizer.step()
                    self.critic_updates += 1
                else:
                    skipped += 1
                losses.append(value_loss)
        metrics.update({"control_grad_norm": gradient.detach(), "critic_loss": torch.stack(losses).mean(),
                        "control_skipped": float(not finite), "critic_skipped": skipped,
                        "nonfinite_steps": self.nonfinite_steps,
                        "control_learning_rate": self.actor_optimizer.param_groups[0]["lr"],
                        "execution/imag_batch": batch,
                        "execution/imag_microbatch": min(batch, self.execution.imag_microbatch),
                        "execution/critic_microbatch": min(self.execution.critic_microbatch, minibatch)})
        return _finite_metrics(metrics)

    def state_dict(self):
        return {"actor": self.actor_optimizer.state_dict(), "model": self.model_optimizer.state_dict(),
                "critic": self.critic_optimizer.state_dict(), "actor_updates": self.actor_updates,
                "model_updates": self.model_updates, "critic_updates": self.critic_updates,
                "nonfinite_steps": self.nonfinite_steps}

    def load_optimizer_state_dict(self, name, data):
        """沿用当前融合执行选项，加载原步数、动量和其余优化器参数。"""
        # 必须在加载前覆盖：PyTorch据此把fused Adam的step张量放到参数设备。
        fused = True if self.execution.fused_adam else None
        state = {**data, "param_groups": [{**group, "fused": fused} for group in data["param_groups"]]}
        getattr(self, name + "_optimizer").load_state_dict(state)

    def load_state_dict(self, data):
        for name in ("actor", "model", "critic"):
            self.load_optimizer_state_dict(name, data[name])
        for name in ("actor_updates", "model_updates", "critic_updates"):
            setattr(self, name, data[name])
