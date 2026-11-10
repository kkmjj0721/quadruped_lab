# Copyright (c) 2026-2027 zh
"""内容：用流内事件标记训练阶段，在日志边界之后读取已完成时间。"""

from contextlib import contextmanager
import math
import statistics
import time

import torch


class PhaseTimer:
    """CUDA 只记录事件，不在阶段边界同步；CPU 使用已完成操作的墙钟。"""

    def __init__(self, device):
        self.device = torch.device(device)
        self.marks = {}
        self.mark("start")

    def mark(self, name):
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            self.marks[name] = event
        else:
            self.marks[name] = time.perf_counter()

    def seconds(self, start, end):
        """调用方必须已读取最后一个事件之后排队的设备标量。"""
        if self.device.type == "cuda":
            return self.marks[start].elapsed_time(self.marks[end]) / 1000
        return self.marks[end] - self.marks[start]


def _statistics(values):
    """以实际观察次数汇总；P90取最近秩，不将未执行项填零。"""
    ordered = sorted(values)
    if not ordered:
        return {"calls": 0, "total_seconds": 0.0, "mean_seconds": None,
                "median_seconds": None, "p90_seconds": None}
    return {"calls": len(ordered), "total_seconds": math.fsum(ordered),
            "mean_seconds": statistics.fmean(ordered), "median_seconds": statistics.median(ordered),
            "p90_seconds": ordered[math.ceil(.9 * len(ordered)) - 1]}


class TrainingTiming:
    """采样训练阶段并汇总会话墙钟；事件读取依赖调用方原有的设备回读边界。"""

    def __init__(self, device, *, enabled=False, interval=1):
        self.device = torch.device(device)
        self.enabled, self.interval = enabled, interval
        self._events = {}
        self._samples, self._session = {}, {}
        self._iterations, self._seen_phases = [], set()
        self._pending, self._indices = [], {}
        self._iteration = None
        self._sampled = False
        self._collected = False

    def begin_iteration(self, iteration, phase, *, setup=False):
        if not self.enabled:
            return
        first_use = phase not in self._seen_phases
        self._seen_phases.add(phase)
        self._iteration, self._phase = iteration, phase
        self._setup = bool(setup or first_use)
        self._sampled = self._setup or iteration % self.interval == 0
        self._pending, self._indices = [], {}
        self._collected = False

    @contextmanager
    def section(self, name, *, cuda=True):
        """主机调用区间与CUDA流区间分别计量，不在阶段末等待GPU。"""
        if not self.enabled or not self._sampled:
            yield
            return
        index = self._indices.get(name, 0)
        self._indices[name] = index + 1
        events = None
        started = time.perf_counter()
        if cuda and self.device.type == "cuda":
            key = name, index
            if key not in self._events:
                self._events[key] = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events = self._events[key]
            events[0].record(torch.cuda.current_stream(self.device))
        try:
            yield
        finally:
            if events is not None:
                events[1].record(torch.cuda.current_stream(self.device))
            self._pending.append((name, time.perf_counter() - started, events))

    def device_metrics(self):
        """仅在原标量回读完成后取事件值；重复调用不重复累计。"""
        if not self.enabled:
            return {}
        metrics = {"timing/detail/sampled": float(self._sampled), "timing/detail/setup": float(self._setup)}
        if self._collected or not self._sampled:
            return metrics
        group = self._phase + ("/setup" if self._setup else "/steady")
        for name, host_seconds, events in self._pending:
            values = {"host": host_seconds}
            if events is not None:
                values["cuda"] = events[0].elapsed_time(events[1]) / 1000
            for clock, seconds in values.items():
                key = f"timing/detail/{name}_{clock}_seconds"
                metrics[key] = metrics.get(key, 0.0) + seconds
                self._samples.setdefault((group, name, clock), []).append(seconds)
        self._collected = True
        return metrics

    def previous_metrics(self):
        """上一轮日志完成后才知道的时间带原轮号延后记录。"""
        if not self.enabled or not self._iterations:
            return {}
        previous = self._iterations[-1]
        return {"timing/previous_iteration": previous["iteration"],
                "timing/previous_iteration_wall_seconds": previous["wall_seconds"],
                "timing/previous_logging_seconds": previous["logging_seconds"]}

    def end_iteration(self, wall_seconds, log_seconds):
        if not self.enabled:
            return
        self._iterations.append({"iteration": self._iteration, "phase": self._phase, "setup": self._setup,
                                 "sampled": self._sampled, "wall_seconds": wall_seconds,
                                 "logging_seconds": log_seconds})
        self._iteration = None

    def record_wall(self, name, seconds):
        """会话级墙钟在详细开关两种状态均保留；包容父项与子项分别命名。"""
        self._session.setdefault(name, []).append(seconds)

    def summary(self):
        stages, groups = {}, {}
        for (group, name, clock), samples in self._samples.items():
            stages.setdefault(group, {}).setdefault(name, {})[clock] = _statistics(samples)
        for row in self._iterations:
            group = row["phase"] + ("/setup" if row["setup"] else "/steady")
            groups.setdefault(group, []).append(row)
        return {"format": "supermimic_timing_v1", "device": str(self.device),
                "detailed_timing": self.enabled, "timing_interval": self.interval,
                "units": "seconds", "session": {name: _statistics(values) for name, values in self._session.items()},
                "iterations": {"completed": len(self._iterations),
                               "sampled": sum(row["sampled"] for row in self._iterations),
                               "first": self._iterations[0]["iteration"] if self._iterations else None,
                               "last": self._iterations[-1]["iteration"] if self._iterations else None,
                               "unfinished": self._iteration,
                               "groups": {group: {"iterations": len(rows),
                                                   "sampled": sum(row["sampled"] for row in rows),
                                                   "wall": _statistics([row["wall_seconds"] for row in rows]),
                                                   "logging": _statistics([row["logging_seconds"] for row in rows])}
                                          for group, rows in groups.items()}},
                "stages": stages,
                "scope": {"host": "Host call intervals; CUDA work may still be queued",
                          "cuda": "Stream event intervals including possible host submission gaps; not kernel-only time",
                          "iteration_wall": "Host wall through original metric read, guard and logging; excludes outer save",
                          "sampling": "Stage statistics cover sampled calls only; no extrapolation to unsampled iterations",
                          "nesting": "Coarse model/control and save_total contain their respective detail/export children"}}
