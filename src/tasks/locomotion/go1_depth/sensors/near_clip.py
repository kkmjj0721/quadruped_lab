# Copyright (c) 2026-2027 zh
"""
内容：
    以共享场景 BVH 为深度相机补充 InstinctMJ 03563ee 的近命中逐跳续投射。
    对应 grouped_ray_caster.py 的 _apply_hit_filter_and_continue；默认六跳、epsilon=1e-4。
    保留原始远处命中，仅修正欧氏距离不超过 min_distance 的像素。
"""

import torch
import warp as wp
from mujoco_warp import rays


def continue_near_hits(distance, origin, direction, cast_active, *, min_distance, max_distance, epsilon, max_hops):
    """跳过近命中并继续查询；cast_active 只返回活跃环境／光线的距离。"""
    rejected = (distance >= 0) & (distance <= min_distance)
    if not rejected.any():
        return distance
    final = torch.where(rejected, -torch.ones_like(distance), distance)
    traveled = torch.where(rejected, distance + epsilon, torch.zeros_like(distance))
    current = origin + direction * traveled[..., None]
    active = rejected & (traveled < max_distance)
    for _ in range(max_hops):
        if not active.any():
            break
        env, ray, hit = cast_active(current, direction, active)
        total = traveled[env, ray] + hit
        valid = (hit >= 0) & (hit <= max_distance - traveled[env, ray])
        accepted = valid & (total > min_distance)
        final[env[accepted], ray[accepted]] = total[accepted]
        continuing = valid & ~accepted
        next_env, next_ray = env[continuing], ray[continuing]
        traveled[next_env, next_ray] = total[continuing] + epsilon
        current[next_env, next_ray] = origin[next_env, next_ray] + direction[next_env, next_ray] * traveled[next_env, next_ray, None]
        active = torch.zeros_like(active)
        active[next_env, next_ray] = traveled[next_env, next_ray] < max_distance
    return final


class NearClipContinuation:
    """沿实际相机渲染光线复用 mujoco_warp.rays，近命中才进入续投射。"""

    def __init__(self, env, cfg):
        self.env, self.cfg = env, cfg
        camera = env.scene["depth"]
        self.camera_idx = camera.camera_idx
        self.context = camera._ctx
        render_context = self.context.render_context
        # 本任务只有一台相机，未随机化内参；直接使用渲染器的逐像素归一化方向。
        self.local_direction = wp.to_torch(render_context.ray).reshape(cfg.height * cfg.width, 3).clone()
        self.cosine = -self.local_direction[:, 2]
        vec6 = wp.types.vector(length=6, dtype=float)
        self.geomgroup = vec6(*[-1 if index in cfg.enabled_geom_groups else 0 for index in range(6)])

    def _cast_active(self, origin, direction, active):
        env, ray = active.nonzero(as_tuple=True)
        slot = active.long().cumsum(-1)[env, ray] - 1
        width = int(slot.max().item()) + 1
        batch = active.shape[0]
        points = origin.new_zeros(batch, width, 3)
        vectors = direction.new_zeros(batch, width, 3)
        vectors[..., 2] = 1
        points[env, slot], vectors[env, slot] = origin[env, ray], direction[env, ray]
        distance = origin.new_empty(batch, width)
        geom = torch.empty(batch, width, dtype=torch.int32, device=origin.device)
        normal = origin.new_empty(batch, width, 3)
        exclude = torch.full((width,), -1, dtype=torch.int32, device=origin.device)
        # 上游 rays 使用 Warp 默认设备；查询必须与仿真模型和相机数据位于同一设备。
        with wp.ScopedDevice(self.env.sim.wp_device):
            rays(m=self.env.sim.wp_model, d=self.env.sim.wp_data,
                 pnt=wp.from_torch(points, dtype=wp.vec3), vec=wp.from_torch(vectors, dtype=wp.vec3),
                 geomgroup=self.geomgroup, flg_static=True, bodyexclude=wp.from_torch(exclude, dtype=wp.int32),
                 dist=wp.from_torch(distance, dtype=wp.float32), geomid=wp.from_torch(geom, dtype=wp.int32),
                 normal=wp.from_torch(normal, dtype=wp.vec3), rc=self.context.render_context)
        return env, ray, distance[env, slot]

    @torch.no_grad()
    def correct(self, raw_depth):
        """输入和输出均为相机光轴深度，最小距离判定在欧氏射线距离上执行。"""
        flat = raw_depth.squeeze(-1).flatten(1)
        distance = torch.where(flat > 0, flat / self.cosine, -torch.ones_like(flat))
        # 上游先在射线距离上限制 max_distance，再投影到相机光轴。
        distance = torch.where(distance <= self.cfg.far_clip, distance, -torch.ones_like(distance))
        corrected = distance
        if ((distance >= 0) & (distance <= self.cfg.near_clip)).any():
            data = self.env.sim.data
            position = data.cam_xpos[:, self.camera_idx]
            rotation = data.cam_xmat[:, self.camera_idx].reshape(-1, 3, 3)
            direction = torch.einsum("bij,nj->bni", rotation, self.local_direction)
            origin = position[:, None].expand_as(direction)
            corrected = continue_near_hits(
                distance, origin, direction, self._cast_active, min_distance=self.cfg.near_clip,
                max_distance=self.cfg.far_clip, epsilon=self.cfg.near_epsilon, max_hops=self.cfg.near_max_hops)
        planar = torch.where(corrected >= 0, corrected * self.cosine, torch.zeros_like(corrected))
        return planar.reshape(raw_depth.shape)
