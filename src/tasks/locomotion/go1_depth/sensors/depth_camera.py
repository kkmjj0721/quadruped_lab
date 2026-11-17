# Copyright (c) 2026-2027 zh
"""内容：持有低频深度帧、同帧 clean 标签与实际光学相机投影；仅新帧采样噪声。"""

from __future__ import annotations

from dataclasses import fields
import math

import torch

from agents.depth_supermimic.interfaces import DepthFrame


def clone_frame(frame: DepthFrame) -> DepthFrame:
    """复制帧协议，避免底层零拷贝渲染或局部 reset 覆盖原生终态。"""
    return DepthFrame(**{field.name: getattr(frame, field.name).detach().clone() for field in fields(frame)})


def camera_intrinsics(model, camera_id: int, width: int, height: int, *, device="cpu"):
    """按当前 Warp compute_ray 的实际视锥返回以像素中心为整数坐标的 K。

    Warp 在显式 sensorsize 与输出比例不同时居中截取视锥；这里采用相同规则。
    相机光学坐标为右、下、前，与 MuJoCo 的右、上、后通过 diag(1,-1,-1) 转换。
    """
    sensor_w, sensor_h = map(float, model.cam_sensorsize[camera_id])
    if sensor_h:
        aspect = width / height
        if aspect > sensor_w / sensor_h:
            sensor_h = sensor_w / aspect
        elif aspect < sensor_w / sensor_h:
            sensor_w = sensor_h * aspect
        focal_x, focal_y, center_x, center_y = map(float, model.cam_intrinsic[camera_id])
        fx, fy = width * focal_x / sensor_w, height * focal_y / sensor_h
        cx, cy = (width - 1) / 2 - width * center_x / sensor_w, (height - 1) / 2 - height * center_y / sensor_h
    else:
        fy = height / (2 * math.tan(math.radians(float(model.cam_fovy[camera_id])) / 2))
        fx, cx, cy = fy, (width - 1) / 2, (height - 1) / 2
    return torch.tensor(((fx, 0., cx), (0., fy, cy), (0., 0., 1.)), device=device, dtype=torch.float32)


def world_to_optical(camera_position, camera_rotation):
    """由实际 forward 后的 cam_xpos/cam_xmat 构造世界到光学正 z 的变换。"""
    rotation = camera_rotation.reshape(-1, 3, 3).transpose(-1, -2)
    signs = rotation.new_tensor((1., -1., -1.))
    rotation = rotation * signs[None, :, None]
    result = torch.zeros((rotation.shape[0], 4, 4), device=rotation.device, dtype=rotation.dtype)
    result[:, :3, :3] = rotation
    result[:, :3, 3] = -torch.bmm(rotation, camera_position.unsqueeze(-1)).squeeze(-1)
    result[:, 3, 3] = 1
    return result


class DepthCameraState:
    """每行有自己的 age；reset 首帧立即拍摄并清零该行 age。"""

    def __init__(self, cfg, num_envs, *, device="cpu", noisy=True):
        self.cfg, self.device, self.noisy = cfg, torch.device(device), noisy
        shape = (num_envs, 1, cfg.height, cfg.width)
        self.frame = DepthFrame(
            noisy=torch.zeros(shape, device=self.device), clean=torch.zeros(shape, device=self.device),
            valid=torch.zeros(shape, device=self.device, dtype=torch.bool),
            fresh=torch.zeros(num_envs, device=self.device, dtype=torch.bool),
            age=torch.zeros(num_envs, device=self.device, dtype=torch.long),
            world_to_camera=torch.eye(4, device=self.device).expand(num_envs, -1, -1).clone(),
            intrinsics=torch.eye(3, device=self.device).expand(num_envs, -1, -1).clone())

    def tick(self):
        """控制步推进只增加年龄，不改变图像、标签、噪声或相机投影。"""
        self.frame.fresh.zero_()
        self.frame.age.add_(1)
        return (self.frame.age >= self.cfg.period).nonzero(as_tuple=False).flatten()

    @torch.no_grad()
    def capture(self, metric_depth, world_to_camera, intrinsics, ids):
        """只更新指定行；clean 的真实有效掩码不受人工 dropout 影响。"""
        depth = metric_depth[ids].permute(0, 3, 1, 2)
        valid = torch.isfinite(depth) & (depth > self.cfg.near) & (depth <= self.cfg.far)
        clean = torch.where(valid, depth / self.cfg.far, 0.)
        noisy = depth
        if self.noisy:
            if self.cfg.noise_std:
                noisy = noisy + torch.randn_like(noisy) * self.cfg.noise_std
            if self.cfg.dropout_probability:
                noisy = noisy * (torch.rand_like(noisy) >= self.cfg.dropout_probability)
        noisy = torch.where(valid, noisy.clamp(self.cfg.near, self.cfg.far) / self.cfg.far, 0.)
        self.frame.clean[ids], self.frame.noisy[ids], self.frame.valid[ids] = clean, noisy, valid
        self.frame.world_to_camera[ids] = world_to_camera[ids]
        self.frame.intrinsics[ids] = intrinsics.expand(self.frame.age.shape[0], -1, -1)[ids]
        self.frame.age[ids] = 0
        self.frame.fresh[ids] = True
