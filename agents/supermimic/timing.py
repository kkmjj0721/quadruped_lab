# Copyright (c) 2026-2027 zh
"""内容：用流内事件标记训练阶段，在日志边界之后读取已完成时间。"""

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
