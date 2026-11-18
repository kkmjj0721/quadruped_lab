# Copyright (c) 2026-2027 zh
"""内容：按环境相机时钟发射深度射线，并生成同视点的干净／带噪样本。

逐射线求交基准为 uv.lock 的 mjlab 1.6.0、mujoco-warp 3.11.0、Warp 1.18.0。
复用上游 render._make_cast_ray 私有接口及同一 BVH 与光轴深度定义；
只为到期环境发射像素射线，避免原生 sense 在每个控制步渲染全部相机。
"""

import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.sensor.sensor_context import SensorContext
from mujoco_warp._src.render import _make_cast_ray

from src.config.go1_depth.go1_depth_config import make_go1_depth_camera_cfg


def _depth_kernel(geom_ray_types, cull_backfaces, use_fast_math):
    """仅选择环境索引；求交及米制光轴投影与上游 render 保持一致。"""
    cast_ray = _make_cast_ray(geom_ray_types, first_hit=False)

    @wp.kernel(module="unique", enable_backward=False, module_options={"fast_math": use_fast_math})
    def render_selected(
        world_ids: wp.array[wp.int32],
        geom_type: wp.array[int],
        geom_dataid: wp.array2d[int],
        geom_size: wp.array2d[wp.vec3],
        flex_vertadr: wp.array[int],
        flex_edge: wp.array[wp.vec2i],
        flex_radius: wp.array[float],
        geom_xpos: wp.array2d[wp.vec3],
        geom_xmat: wp.array2d[wp.mat33],
        flexvert_xpos: wp.array2d[wp.vec3],
        camera_position: wp.array2d[wp.vec3],
        camera_rotation: wp.array2d[wp.mat33],
        camera_id: int,
        ray: wp.array[wp.vec3],
        bvh_id: wp.uint64,
        group_root: wp.array[int],
        bvh_ngeom: int,
        bvh_nflexgeom: int,
        enabled_geom_ids: wp.array[int],
        mesh_bvh_id: wp.array[wp.uint64],
        hfield_bvh_id: wp.array[wp.uint64],
        flex_geom_flexid: wp.array[int],
        flex_geom_edgeid: wp.array[int],
        flex_bvh_id: wp.array[wp.uint64],
        flex_group_root: wp.array2d[int],
        depth: wp.array2d[float],
        far: float,
    ):
        row, pixel = wp.tid()
        world = world_ids[row]
        local_ray = ray[pixel]
        world_ray = camera_rotation[world, camera_id] @ local_ray
        geom, distance, normal, u, v, face, mesh = cast_ray(
            geom_type, geom_dataid, geom_size, flex_vertadr, flex_edge, flex_radius,
            geom_xpos, geom_xmat, flexvert_xpos, bvh_id, group_root[world], world,
            bvh_ngeom, bvh_nflexgeom, enabled_geom_ids, mesh_bvh_id, hfield_bvh_id,
            flex_geom_flexid, flex_geom_edgeid, flex_bvh_id, flex_group_root,
            camera_position[world, camera_id], world_ray, far / (-local_ray[2]), wp.static(cull_backfaces),
        )
        value = float(0.0)
        if geom != -1:
            value = distance * (-local_ray[2])
        depth[world, pixel] = value

    return render_selected


class SelectedDepthRenderer:
    """独占低频采集；相机不注册到环境的自动 sense 流程。

    本任务相机内参固定，SensorContext 仅含此一台相机，直接复用其预计算射线。
    BVH refit 仍覆盖所有 world；像素射线仅覆盖 env_ids，未选择行的缓存不写。
    接触数据与时长仍由 scene.update 每物理子步独立更新。
    """

    def __init__(self, raw_env, cfg):
        self.env = raw_env
        self.cfg = cfg.sensors.depth
        camera_cfg = make_go1_depth_camera_cfg(cfg)
        self.sensor = camera_cfg.build()
        sim = raw_env.sim
        self.sensor.initialize(sim.mj_model, sim.wp_model, sim.wp_data, raw_env.device)
        self.context = SensorContext(sim.mj_model, sim.wp_model, sim.wp_data, [self.sensor], (), raw_env.device)
        rc = self.context.render_context
        self.kernel = _depth_kernel(rc.geom_ray_types, rc.enable_backface_culling, rc.use_fast_math)
        self.rendered_frames = torch.zeros(raw_env.num_envs, dtype=torch.long, device=raw_env.device)

    @torch.no_grad()
    def capture(self, env_ids):
        """只更新到期行的像素；调用方在下次渲染前复制需要保留的帧。"""
        sim = self.env.sim
        model, data, rc = sim.wp_model, sim.wp_data, self.context.render_context
        selected = env_ids.to(dtype=torch.int32).contiguous()
        with wp.ScopedDevice(sim.wp_device):
            mjwarp.refit_bvh(model, data, rc)
            wp.launch(self.kernel, dim=(selected.numel(), self.cfg.height * self.cfg.width), inputs=[
                wp.from_torch(selected, dtype=wp.int32), model.geom_type, model.geom_dataid, model.geom_size,
                model.flex_vertadr, model.flex_edge, model.flex_radius, data.geom_xpos, data.geom_xmat,
                data.flexvert_xpos, data.cam_xpos, data.cam_xmat, self.sensor.camera_idx, rc.ray,
                rc.bvh_id, rc.group_root, rc.bvh_ngeom, rc.bvh_nflexgeom, rc.enabled_geom_ids,
                rc.mesh_bvh_id, rc.hfield_bvh_id, rc.flex_geom_flexid, rc.flex_geom_edgeid,
                rc.flex_bvh_id, rc.flex_group_root, rc.depth_data, self.cfg.far,
            ], device=sim.wp_device)
        self.rendered_frames[env_ids] += 1
        return self.context.get_depth(self.sensor.camera_idx)
