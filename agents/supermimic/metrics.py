# Copyright (c) 2026-2027 zh
"""
内容：
    累计真实环境回合回报与长度，维护整体及教师／学生的完成回合窗口。
"""

from __future__ import annotations

import math
from collections import deque
from numbers import Real

import torch


class EpisodeStatistics:
    """未结束回合跨采集轮次保留，窗口只接收真实已结束回合。"""

    def __init__(self, num_envs: int, device, *, window_size: int = 100, step_dt: float | None = None):
        for name, value in (("num_envs", num_envs), ("window_size", window_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if step_dt is not None:
            if isinstance(step_dt, bool) or not isinstance(step_dt, Real) or not math.isfinite(step_dt) or step_dt <= 0:
                raise ValueError("step_dt must be finite and positive, or None")
            step_dt = float(step_dt)
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.step_dt = step_dt
        self._returns = torch.zeros(num_envs, dtype=torch.float64, device=self.device)
        self._lengths = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._episodes = deque(maxlen=window_size)
        self._teacher_returns = deque(maxlen=window_size)
        self._student_returns = deque(maxlen=window_size)
        self.begin_collection()

    def reset(self) -> None:
        """显式重置采集或参数续训时，清空所有未结束累计与完成窗口。"""
        self._returns.zero_()
        self._lengths.zero_()
        self._episodes.clear()
        self._teacher_returns.clear()
        self._student_returns.clear()
        self.begin_collection()

    def begin_collection(self) -> None:
        """只清除当前采集轮次的完成计数。"""
        self._completed = 0
        self._teacher_completed = 0
        self._student_completed = 0

    def _validate(self, name: str, value: torch.Tensor, *, boolean: bool = False) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.shape != (self.num_envs,):
            raise ValueError(f"{name} must have shape ({self.num_envs},)")
        if value.device != self._returns.device:
            raise ValueError(f"{name} must use device {self._returns.device}")
        if boolean and value.dtype != torch.bool:
            raise TypeError(f"{name} must have boolean dtype")
        if not boolean and not value.is_floating_point():
            raise TypeError(f"{name} must have floating-point dtype")
        if not boolean and not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")

    @torch.no_grad()
    def update(self, reward: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor,
               branch: torch.Tensor) -> None:
        """输入为本次控制步转移，branch为True的环境计入教师组。"""
        self._validate("reward", reward)
        for name, value in (("terminated", terminated), ("truncated", truncated), ("branch", branch)):
            self._validate(name, value, boolean=True)
        accumulated = self._returns + reward.detach().to(dtype=self._returns.dtype)
        if not bool(torch.isfinite(accumulated).all()):
            raise ValueError("Accumulated episode returns must remain finite")
        self._returns.copy_(accumulated)
        self._lengths.add_(1)
        done = terminated | truncated
        if not bool(done.any()):
            return
        # 只把完成行拷贝到CPU窗口，避免保留环境张量或其计算图。
        completed_returns = self._returns[done].cpu().tolist()
        completed_lengths = self._lengths[done].cpu().tolist()
        completed_branches = branch[done].cpu().tolist()
        for episode_return, length, teacher in zip(completed_returns, completed_lengths, completed_branches):
            self._episodes.append((episode_return, length))
            if teacher:
                self._teacher_returns.append(episode_return)
                self._teacher_completed += 1
            else:
                self._student_returns.append(episode_return)
                self._student_completed += 1
        self._completed += len(completed_returns)
        self._returns[done] = 0
        self._lengths[done] = 0

    def metrics(self) -> dict[str, float]:
        """没有完成回合时不生成均值，调用方可据此显示n/a。"""
        values = {
            "episode/completed": float(self._completed),
            "episode/window_count": float(len(self._episodes)),
            "episode/teacher_completed": float(self._teacher_completed),
            "episode/student_completed": float(self._student_completed),
            "episode/teacher_window_count": float(len(self._teacher_returns)),
            "episode/student_window_count": float(len(self._student_returns)),
        }
        if self._episodes:
            count = len(self._episodes)
            values["episode/return_mean"] = sum(item[0] / count for item in self._episodes)
            values["episode/length_steps_mean"] = sum(item[1] / count for item in self._episodes)
            if self.step_dt is not None:
                values["episode/length_seconds_mean"] = values["episode/length_steps_mean"] * self.step_dt
        for name, window in (("teacher", self._teacher_returns), ("student", self._student_returns)):
            if window:
                values[f"episode/{name}_return_mean"] = sum(value / len(window) for value in window)
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Episode statistics must remain finite")
        return values
