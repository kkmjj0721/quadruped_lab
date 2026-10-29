# Copyright (c) 2026-2027 zh
"""
内容：
    独立格式化SuperMimic真实采集、模型训练和想象控制日志，并逐轮写入TensorBoard标量。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from datetime import timedelta


# 保留原始tag供已有分析工具读取，同时提供与PPO一致的常用面板分组。
_SCALAR_GROUPS = {
    "wm_loss": "Loss/world_model",
    "critic_loss": "Loss/value",
    "control_loss": "Loss/control",
    "world_model_learning_rate": "Loss/world_model_learning_rate",
    "critic_learning_rate": "Loss/value_learning_rate",
    "control_learning_rate": "Loss/control_learning_rate",
    "wm_grad_norm": "Grad/world_model",
    "control_grad_norm": "Grad/control",
    "performance/real_total_fps": "Perf/total_fps",
    "timing/collection_seconds": "Perf/collection_time",
    "timing/learning_seconds": "Perf/learning_time",
    "episode/return_mean": "Train/mean_reward",
    "episode/length_steps_mean": "Train/mean_episode_length",
}


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _numeric_metrics(metrics: Mapping[str, Real]) -> dict[str, int | float]:
    """复制并验证可写入TensorBoard的有限数字，不修改调用方的映射。"""
    if not isinstance(metrics, Mapping):
        raise TypeError("Training metrics must be a flat mapping of string keys to finite numbers")
    values = {}
    for name, value in metrics.items():
        if not isinstance(name, str):
            raise TypeError("Training metric names must be strings")
        if not isinstance(value, Real):
            raise TypeError(f"Training metric {name!r} must be a finite number")
        try:
            finite = math.isfinite(value)
        except (OverflowError, ValueError):
            finite = False
        if not finite:
            raise ValueError(f"Training metric {name!r} must be finite")
        values[name] = int(value) if isinstance(value, int) else float(value)
    return values


def format_training_log(metrics: Mapping[str, Real], *, num_envs: int, device) -> str:
    """按Go2 HIM的80列单块样式显示训练摘要和回合奖励分项。"""
    _positive_integer("num_envs", num_envs)
    values = _numeric_metrics(metrics)

    def number(key: str, *, decimals: int = 4, count: bool = False) -> str:
        if key not in values:
            return "n/a"
        return f"{values[key]:.0f}" if count else f"{values[key]:.{decimals}f}"

    def duration(key: str) -> str:
        if key not in values:
            return "n/a"
        return str(timedelta(seconds=int(max(0, values[key]))))

    def row(label: str, value: str) -> str:
        return f"{label + ':':>40} {value}"

    iteration = number("iteration", count=True)
    target = number("training_target_iteration", count=True)
    progress = iteration if "training_target_iteration" not in values else f"{iteration}/{target}"
    model_ready = not bool(values.get("model_not_ready", 0))
    control_available = model_ready and values.get("control_enabled", 1) != 0
    model_available = model_ready and values.get("wm_skipped", 0) < 1
    learning_keys = ("timing/world_model_seconds", "timing/control_seconds")
    learning_time = f"{sum(values[key] for key in learning_keys):.3f}s" if all(key in values for key in learning_keys) else "n/a"
    collection_time = number("timing/collection_seconds", decimals=3)
    if "timing/collection_seconds" in values:
        collection_time += "s"
    lines = ["#" * 80, f" Learning iteration {progress} ".center(80), "",
             row("Total steps", number("real_steps", count=True)),
             row("Steps per second", number("performance/real_total_fps", count=True)),
             row("Collection time", collection_time), row("Learning time", learning_time)]
    if not model_ready:
        lines.append(row("Training phase", "replay warmup"))
    elif values.get("control_enabled") == 0:
        lines.append(row("Training phase", "world-model only"))
    for label, key, available in (
        ("Mean world model loss", "wm_loss", model_available),
        ("Mean value loss", "critic_loss", control_available),
        ("Mean control loss", "control_loss", control_available),
    ):
        if available and key in values:
            lines.append(row(label, number(key)))
    if "control_learning_rate" in values:
        lines.append(row("Control learning rate", f"{values['control_learning_rate']:.2e}"))
    for label, key in (("Mean reward", "episode/return_mean"), ("Mean episode length", "episode/length_steps_mean")):
        if key in values:
            lines.append(row(label, number(key, decimals=2)))
    for label, key, count in (
        ("Metrics/failures", "failures", True), ("Metrics/timeouts", "timeouts", True),
        ("Metrics/terrain_level", "task/terrain_level", False),
    ):
        if key in values:
            lines.append(row(label, number(key, count=count)))
    for label, key in (("Metrics/linear_velocity_rmse", "task/tracking_linear_mse"),
                       ("Metrics/yaw_rate_rmse", "task/tracking_angular_mse")):
        if key in values:
            lines.append(row(label, f"{math.sqrt(max(0, values[key])):.4f}"))
    if "task/action_rms" in values:
        lines.append(row("Metrics/action_rms", number("task/action_rms")))
    for key in values:
        if key.startswith("Episode_Reward/"):
            lines.append(row(key, number(key)))
    iteration_time = number("timing/iteration_seconds", decimals=2)
    if "timing/iteration_seconds" in values:
        iteration_time += "s"
    lines.extend(["-" * 80, row("Iteration time", iteration_time),
                  row("Time elapsed", duration("timing/session_elapsed_seconds")),
                  row("ETA", duration("timing/eta_seconds")), ""])
    return "\n".join(lines)


class TrainingLogger:
    """逐轮写入TensorBoard标量，终端按迭代间隔显示并保留首轮与目标末轮。"""

    def __init__(self, log_dir: str | Path | None, *, num_envs: int, device, log_interval: int = 1):
        _positive_integer("num_envs", num_envs)
        _positive_integer("log_interval", log_interval)
        self.num_envs = num_envs
        self.device = str(device)
        self.log_interval = log_interval
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self._records = 0
        self._writer = None
        if self.log_dir is not None:
            # 逐轮指标集中写入同一事件文件，tag按"/"分组成TensorBoard面板。
            from torch.utils.tensorboard import SummaryWriter

            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(str(self.log_dir), flush_secs=10)

    def add_text(self, tag: str, text: str) -> None:
        """把一次性的运行来源信息写入同一事件文件，不新增运行目录文件。"""
        if self._writer is not None:
            self._writer.add_text(tag, text, 0)

    def record(self, metrics: Mapping[str, Real], *, force: bool = False) -> None:
        values = _numeric_metrics(metrics)
        self._records += 1
        iteration = values.get("iteration", self._records)
        if self._writer is not None:
            for name, value in values.items():
                self._writer.add_scalar(name, value, iteration)
                if name in _SCALAR_GROUPS:
                    self._writer.add_scalar(_SCALAR_GROUPS[name], value, iteration)
            for source, tag in (("task/tracking_linear_mse", "Metrics/tracking/linear_velocity_rmse"),
                                ("task/tracking_angular_mse", "Metrics/tracking/yaw_rate_rmse")):
                if source in values:
                    self._writer.add_scalar(tag, math.sqrt(max(0, values[source])), iteration)
            if "timing/session_elapsed_seconds" in values:
                seconds = int(values["timing/session_elapsed_seconds"])
                for source in ("episode/return_mean", "episode/length_steps_mean"):
                    if source in values:
                        self._writer.add_scalar(_SCALAR_GROUPS[source] + "/time", values[source], seconds)
        target = values.get("training_target_iteration")
        if force or self._records == 1 or iteration % self.log_interval == 0 or iteration == target:
            print(format_training_log(values, num_envs=self.num_envs, device=self.device), flush=True)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
