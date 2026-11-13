# Copyright (c) 2026-2027 zh
"""内容：按 Go2 SuperMimic 的窗口口径与80列格式打印 Go1 视觉训练进度。"""

import time

from agents.supermimic.logging import format_training_log
from agents.supermimic.metrics import MetricWindow, read_metrics


class TrainingProgress:
    """只汇总已回读的逐轮标量，不拥有事件文件或影响训练状态。"""

    def __init__(self, config, *, num_envs, device, initial_iteration, initial_seconds,
                 target_iteration, minimum_training_seconds=0.0):
        self.config, self.num_envs, self.device = config, num_envs, device
        self.initial_iteration, self.initial_seconds = initial_iteration, initial_seconds
        self.target_iteration, self.minimum_training_seconds = target_iteration, minimum_training_seconds
        self.window = MetricWindow("cpu")
        self.started = self.window_started = None
        self.latest = None
        self.records = 0

    def record(self, metrics):
        values = dict(metrics)
        if "model_loss" in values:
            values["wm_loss"] = values["model_loss"]
        for name, weight in (("reconstruction", self.config.reconstruction_weight),
                             ("height_reconstruction", self.config.height_reconstruction_weight),
                             ("contrastive", self.config.contrastive_weight)):
            if weight == 0:
                values.pop(name, None)
        now = time.perf_counter()
        if self.started is None:
            # 第一轮训练时长不含回放预填；随后会话墙钟包含日志与周期保存。
            self.started = self.window_started = now - values["timing/iteration_seconds"]
        self.latest = values
        self.window.add(values)
        complete = (values["iteration"] >= self.target_iteration
                    and values["training_seconds"] - self.initial_seconds >= self.minimum_training_seconds)
        if self.records == 0 or values["iteration"] % self.config.log_interval == 0 or complete:
            self.flush(now=now)

    def flush(self, *, now=None):
        if not len(self.window):
            return
        now = time.perf_counter() if now is None else now
        values, validity = self.window.aggregate()
        values = read_metrics(values, "cpu", validity=validity)
        iteration = int(self.latest["iteration"])
        elapsed = now - self.started
        window_seconds = now - self.window_started
        trained_seconds = self.latest["training_seconds"] - self.initial_seconds
        remaining = max(0, self.target_iteration - iteration)
        eta = remaining * elapsed / (iteration - self.initial_iteration)
        if trained_seconds > 0:
            eta = max(eta, max(0, self.minimum_training_seconds - trained_seconds) * elapsed / trained_seconds)
        values["iteration"] = iteration
        if iteration < self.target_iteration or trained_seconds >= self.minimum_training_seconds:
            values["training_target_iteration"] = max(self.target_iteration, iteration)
        values.update({"timing/session_elapsed_seconds": elapsed, "timing/eta_seconds": eta,
                       "timing/iteration_seconds": window_seconds / len(self.window)})
        if "real_steps_delta" in values:
            values["performance/real_total_fps"] = values["real_steps_delta"] / max(window_seconds, 1e-9)
        print(format_training_log(values, num_envs=self.num_envs, device=self.device), flush=True)
        self.records += 1
        self.window.clear()
        self.window_started = now
