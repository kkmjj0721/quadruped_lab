# Copyright (c) 2026-2027 zh
"""
内容：
    对真实连续窗口自回归拟合动力学、失败及可选辅助头，不将超时作为跌倒标签。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from agents.supermimic.interfaces import ReplayWindow, TaskState, validate_policy_input
from agents.supermimic.models.dynamics import DynamicsModel


class WorldModelTrainer:
    """只更新世界模型参数，记录动作和辅助状态均作为固定监督数据。"""

    def __init__(self, model: DynamicsModel, learning_rate: float = 2e-4,
                 max_grad_norm: float = 1.0, termination_weight: float = 1.0):
        if not math.isfinite(learning_rate) or learning_rate <= 0 or not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError("Learning rate and gradient norm must be positive and finite")
        if not math.isfinite(termination_weight) or termination_weight < 0:
            raise ValueError("Termination weight must be finite and nonnegative")
        self.model = model
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.termination_weight = termination_weight
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def state_dict(self) -> dict:
        """保存独立优化器及训练定义，不包含仿真和回放状态。"""
        return {"optimizer": self.optimizer.state_dict(), "spec_hash": self.model.spec.hash,
                "config": {"learning_rate": self.learning_rate, "max_grad_norm": self.max_grad_norm,
                           "termination_weight": self.termination_weight}}

    def load_state_dict(self, state: dict) -> None:
        expected = self.state_dict()
        if state.get("spec_hash") != expected["spec_hash"] or state.get("config") != expected["config"]:
            raise ValueError("World-model trainer checkpoint configuration mismatch")
        self.optimizer.load_state_dict(state["optimizer"])

    def _loss_components(self, window: ReplayWindow) -> tuple[Tensor, Tensor, Tensor, int]:
        validate_policy_input(window.initial, self.model.spec)
        if window.actions.ndim != 3:
            raise ValueError("World-model actions must have time, batch and feature axes")
        horizon, batch = window.actions.shape[:2]
        spec = self.model.spec
        if window.actions.shape[-1] != spec.action_dim or window.initial.observation.shape[0] != batch:
            raise ValueError("World-model window batch or action dimension mismatch")
        if horizon <= 0 or window.valid.shape != (horizon, batch) or window.valid.dtype != torch.bool:
            raise ValueError("Invalid world-model window mask")
        if (window.terminated.shape != window.valid.shape or window.truncated.shape != window.valid.shape
                or window.terminated.dtype != torch.bool or window.truncated.dtype != torch.bool):
            raise ValueError("Invalid world-model termination shapes")
        device = window.initial.observation.device
        if any(tensor.device != device for tensor in (window.actions, window.valid, window.terminated, window.truncated)):
            raise ValueError("World-model window must use one device")
        for name, width in (("physical", spec.state_dim), ("memory", spec.memory_dim),
                            ("command", spec.command_dim), ("context", spec.context_dim)):
            target = getattr(window.next_states, name)
            if target.shape != (horizon, batch, width) or target.dtype != torch.float32 or target.device != device:
                raise ValueError("Invalid world-model target state")
        if window.actions.dtype != torch.float32:
            raise ValueError("World-model actions must use float32")
        if (window.valid[1:] & ~window.valid[:-1]).any():
            raise ValueError("Window validity cannot restart after padding")
        if (window.valid[1:] & (window.terminated[:-1] | window.truncated[:-1])).any():
            raise ValueError("World-model window cannot continue after an episode boundary")
        state = window.initial.state.detached_clone()
        count = int(window.valid.sum())
        state_loss = window.actions.new_zeros(())
        term_loss = window.actions.new_zeros(())
        auxiliary_loss = window.actions.new_zeros(())
        for index in range(horizon):
            active = window.valid[index]
            if not active.any():
                break
            target_all = window.next_states.index(index).detached_clone()
            target = target_all.index(active)
            current = state.index(active)
            action = window.actions[index, active].detach()
            predicted = self.model.supervised_step(current, action, target)
            error = self.model.task.state_error(predicted, target)
            logits = self.model.termination_logits(predicted)
            auxiliary_error = self.model.supervised_auxiliary_loss(current, action, predicted, target)
            active_count = int(active.sum())
            if any(value.shape != (active_count,) for value in (error, logits, auxiliary_error)):
                raise ValueError("State, termination and auxiliary predictions must be batch vectors")
            state_loss = state_loss + error.sum()
            auxiliary_loss = auxiliary_loss + auxiliary_error.sum()
            term_loss = term_loss + F.binary_cross_entropy_with_logits(
                logits, window.terminated[index, active].to(logits.dtype), reduction="sum",
            )
            # 仅有效行进入网络，补齐数据即使含NaN也不能污染有效梯度。
            rows = torch.nonzero(active, as_tuple=False).squeeze(-1)
            # 学习得到的物理及辅助状态均回喂预测；无效行保留原值，不带入补齐标签。
            state = TaskState(**{name: getattr(state, name).index_copy(0, rows, getattr(predicted, name))
                                 for name in ("physical", "memory", "command", "context")})
        denominator = max(count, 1)
        return state_loss / denominator, term_loss / denominator, auxiliary_loss / denominator, count

    def _losses(self, window: ReplayWindow) -> tuple[Tensor, Tensor, int]:
        """保留既有内部调用的三元组，将可选辅助项计入第一项。"""
        state_loss, term_loss, auxiliary_loss, count = self._loss_components(window)
        return state_loss + auxiliary_loss, term_loss, count

    def update(self, window: ReplayWindow) -> dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        state_loss, term_loss, auxiliary_loss, count = self._loss_components(window)
        loss = state_loss + self.termination_weight * term_loss + auxiliary_loss
        if not count:
            return {"loss": 0.0, "state_loss": 0.0, "termination_loss": 0.0, "auxiliary_loss": 0.0, "grad_norm": 0.0,
                    "valid_steps": 0.0, "skipped": 1.0}
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite world-model loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm, error_if_nonfinite=True)
        self.optimizer.step()
        return {"loss": float(loss.detach()), "state_loss": float(state_loss.detach()),
                "termination_loss": float(term_loss.detach()), "auxiliary_loss": float(auxiliary_loss.detach()), "grad_norm": float(norm),
                "valid_steps": float(count), "skipped": 0.0}

    @torch.no_grad()
    def evaluate(self, window: ReplayWindow) -> dict[str, float]:
        state_loss, term_loss, auxiliary_loss, count = self._loss_components(window)
        return {"loss": float(state_loss + self.termination_weight * term_loss + auxiliary_loss),
                "state_loss": float(state_loss), "termination_loss": float(term_loss),
                "auxiliary_loss": float(auxiliary_loss), "valid_steps": float(count)}
