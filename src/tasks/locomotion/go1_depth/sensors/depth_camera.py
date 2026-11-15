# Copyright (c) 2026-2027 zh
"""
内容：
    直接对原始32×32深度归一化并加噪，生成双帧输入并模拟整数控制步延迟。
    相机原始数据由 MjLab CameraSensor 的 Warp 场景渲染产生，深度为光轴方向距离。
    带噪输入与干净重建标签共享历史索引，噪声只在新帧入队时采样。
"""

import math

import torch


class DepthCameraHistory:
    """保存回合内连续图像，按同一索引输出带噪双帧输入与干净标签。"""

    def __init__(self, cfg, num_envs, device="cpu", *, noisy=True):
        if cfg.protocol_version != 3:
            raise ValueError("Depth camera requires protocol version 3")
        if cfg.update_interval != 1:
            raise ValueError("Depth camera update_interval must be one control step")
        if not math.isfinite(cfg.far_clip) or cfg.far_clip <= 0:
            raise ValueError("Depth camera far_clip must be finite and positive")
        if cfg.history_length != 2:
            raise ValueError("Depth camera requires two frames")
        lower, upper = cfg.delay_range
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (lower, upper)) or not 0 <= lower <= upper:
            raise ValueError("Depth delay_range must contain ordered nonnegative integer control steps")
        if cfg.history_skip_frames <= 0:
            raise ValueError("Depth history stride must be positive")
        if cfg.sensor_history_length < (cfg.history_length - 1) * cfg.history_skip_frames + upper + 1:
            raise ValueError("Depth sensor history does not cover the frame span and maximum delay")
        if not math.isfinite(cfg.noise_gaussian) or cfg.noise_gaussian < 0:
            raise ValueError("Depth Gaussian noise must be finite and nonnegative")
        if not math.isfinite(cfg.noise_dropout) or not 0 <= cfg.noise_dropout <= 1:
            raise ValueError("Depth dropout probability must be finite and in [0,1]")
        self.cfg, self.device = cfg, torch.device(device)
        self.noisy = noisy
        height, width = cfg.height, cfg.width
        self.frames = torch.zeros(num_envs, cfg.sensor_history_length, height, width, device=device)
        self.clean_frames = torch.zeros_like(self.frames)
        self.pointer = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.delay = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.episode = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.frame_sequence = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.indices = torch.arange(num_envs, device=device)
        self.offsets = torch.arange(cfg.history_length - 1, -1, -1, device=device) * cfg.history_skip_frames

    def process(self, depth):
        """逐像素保留原始深度；无效为零，合法最远量程为一。"""
        depth = depth.squeeze(-1)
        valid = torch.isfinite(depth) & (depth > 0) & (depth <= self.cfg.far_clip)
        return torch.where(valid, depth / self.cfg.far_clip, 0.)

    @torch.no_grad()
    def reset(self, ids):
        """清空所选行，首个新图像填满历史，延迟只在回合开始重新采样。"""
        self.frames[ids] = 0
        self.clean_frames[ids] = 0
        self.pointer[ids] = -1
        self.count[ids] = 0
        self.episode[ids] += 1
        lower, upper = self.cfg.delay_range
        self.delay[ids] = torch.randint(lower, upper + 1, (ids.numel(),), device=self.device)

    @torch.no_grad()
    def append(self, depth, ids=None):
        """新图像只采一次噪声，局部重置仅更新所选环境的历史。"""
        ids = self.indices if ids is None else ids
        clean = self.process(depth)
        frame = clean
        if self.noisy and (self.cfg.noise_gaussian > 0 or self.cfg.noise_dropout > 0):
            # 按 MGDP 顺序在米制命中上加高斯噪声、掉点；缺失标签始终保持零。
            frame = clean * self.cfg.far_clip
            if self.cfg.noise_gaussian > 0:
                frame = frame + torch.randn_like(frame) * self.cfg.noise_gaussian
            if self.cfg.noise_dropout > 0:
                frame = frame * (torch.rand_like(frame) > self.cfg.noise_dropout)
            frame = torch.where(clean > 0, frame.clamp(0, self.cfg.far_clip) / self.cfg.far_clip, 0.)
        self.pointer[ids] = (self.pointer[ids] + 1) % self.cfg.sensor_history_length
        first = self.count[ids] == 0
        first_ids = ids[first]
        self.frames[first_ids] = frame[first, None]
        self.clean_frames[first_ids] = clean[first, None]
        self.frames[ids, self.pointer[ids]] = frame
        self.clean_frames[ids, self.pointer[ids]] = clean
        self.count[ids] += 1
        self.frame_sequence[ids] += 1

    def stack(self, *, clean=False):
        """两路深度共享整数延迟与环形地址，读取时不重新采噪。"""
        length = self.cfg.sensor_history_length
        address = (self.pointer[:, None] - self.offsets[None] - self.delay[:, None]) % length
        frames = self.clean_frames if clean else self.frames
        return frames[self.indices[:, None], address].clone()
