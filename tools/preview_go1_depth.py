# Copyright (c) 2026-2027 zh
"""
内容：
    预览现有 Go1 机器人、黄色相机视锥和实际机载深度渲染，保存截图及米制深度。
    默认打开可旋转观察的静态查看器，不执行行走策略或训练。

用法：
    python tools/preview_go1_depth.py
    python tools/preview_go1_depth.py --headless
    python tools/preview_go1_depth.py --terrain flat --output-dir outputs/go1_depth_flat
    python tools/preview_go1_depth.py --terrain stepping_stones --output-dir outputs/go1_depth_stepping_stones
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
TERRAIN_TITLES = {
    "flat": "平地", "slope": "上下坡", "rough": "粗糙地形", "stairs_up": "上楼梯", "stairs_down": "下楼梯",
    "stepping_stones": "梅花桩",
}


# ---------- 模型与相机 ----------

def build_scene(terrain_name: str, difficulty: float):
    """复用机器人资产与单个现有地块，机器人面朝地块入口。"""
    import mujoco
    from src.assets.robots.go1_depth import get_go1_depth_robot_cfg
    from src.config.go1_depth import Go1DepthCfg

    cfg = Go1DepthCfg()
    spec = get_go1_depth_robot_cfg(cfg).build().spec
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1440, 900
    spec.visual.map.znear, spec.visual.map.zfar = 0.001, 20.0
    spec.visual.headlight.ambient = (0.35, 0.35, 0.35)
    spec.visual.headlight.diffuse = (0.6, 0.6, 0.6)
    spec.worldbody.add_light(name="preview_light", pos=(0, -3, 5), dir=(0.3, 0.4, -1),
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=(0.7, 0.7, 0.7))
    spec.add_texture(name="preview_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, width=256, height=1536,
                     rgb1=(0.16, 0.25, 0.36), rgb2=(0.04, 0.06, 0.10))
    spec.add_texture(name="preview_grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=256, height=256,
                     rgb1=(0.18, 0.26, 0.32), rgb2=(0.25, 0.35, 0.42))
    material = spec.add_material(name="preview_floor", texrepeat=(4, 4), texuniform=True)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "preview_grid"
    # 地块前的有限平台在入口处结束，避免无限平面填平下楼梯和负高度表面。
    flat = terrain_name == "flat"
    spec.worldbody.add_geom(name="preview_floor", type=mujoco.mjtGeom.mjGEOM_PLANE if flat else mujoco.mjtGeom.mjGEOM_BOX,
                            size=(0, 0, 0.05) if flat else (1.4, 4, 0.05),
                            pos=(0, 0, -0.001) if flat else (-0.6, 0, -0.051), material="preview_floor", group=0)
    if terrain_name != "flat":
        if terrain_name == "stepping_stones":
            from mjlab.terrains.config import stepping_stones

            # 仅用于观察相机的静态样例，不改变训练课程；桩顶与起步平台等高。
            terrain = stepping_stones(size=(4.0, 4.0), stone_size_range=(0.25, 0.4),
                                      stone_distance_range=(0.2, 0.3), stone_height=0.0,
                                      stone_height_variation=0.0, stone_size_variation=0.0,
                                      displacement_range=0.0, floor_depth=0.8, platform_width=0.6, border_width=0.0)
        else:
            terrain = cfg.terrain.generator["sub_terrains"][terrain_name]
        spec.worldbody.add_body(name="terrain", pos=(0.8, -terrain.size[1] / 2, 0))
        output = terrain.function(difficulty, spec, np.random.default_rng(cfg.env.seed))
        # 上游部分生成器通过输出元数据返回颜色，需要在直接装配时应用。
        for geometry in output.geometries:
            if geometry.geom is not None and geometry.color is not None:
                geometry.geom.rgba = geometry.color
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_forward(model, data)
    return cfg, model, data


def observer_camera():
    """主视图采用侧前方斜视角，便于同时看清机器人和前方地形。"""
    import mujoco

    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.75, 0, 0.35)
    camera.distance, camera.azimuth, camera.elevation = 3.2, 125.0, -24.0
    return camera


def frustum_points(model, data, camera_id: int, width: int, height: int, length: float = 1.5):
    """由真实相机旋转、VFOV 和宽高比计算四条边界射线的端点。"""
    half_height = length * math.tan(math.radians(model.cam_fovy[camera_id]) / 2)
    half_width = half_height * width / height
    local = np.array([[-half_width, -half_height, -length], [half_width, -half_height, -length],
                      [half_width, half_height, -length], [-half_width, half_height, -length]])
    origin = data.cam_xpos[camera_id].copy()
    return origin, local @ data.cam_xmat[camera_id].reshape(3, 3).T + origin


def add_frustum(scene, model, data, camera_id: int, depth_cfg) -> None:
    """仅向主观察场景添加黄色视锥线，不修改模型或相机深度场景。"""
    import mujoco

    origin, corners = frustum_points(model, data, camera_id, depth_cfg.width, depth_cfg.height)
    segments = [(origin, corner) for corner in corners]
    segments += [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    for start, end in segments:
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                           np.eye(3).ravel(), np.array([1.0, 0.9, 0.12, 1.0]))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2.0, start, end)
        scene.ngeom += 1


def render_depth(renderer, data, depth_cfg):
    """渲染碰撞表面的米制光轴深度，保留机体遮挡并排除相机壳。"""
    import mujoco

    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[list(depth_cfg.enabled_geom_groups)] = 1
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=depth_cfg.camera_name, scene_option=option)
    return renderer.render().copy()


# ---------- 图像与交互 ----------

def depth_panel(depth: np.ndarray, depth_max: float) -> Image.Image:
    """固定米制显示范围；近黑远白，放大时使用最近邻以保留真实像素。"""
    gray = np.rint(np.clip(depth / depth_max, 0, 1) * 255).astype(np.uint8)
    panel = Image.new("RGB", (280, 334), (246, 248, 250))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 7), "机载深度图", font=ImageFont.truetype(FONT_PATH, 20), fill=(25, 35, 48))
    panel.paste(Image.fromarray(gray).resize((256, 256), Image.Resampling.NEAREST).convert("RGB"), (12, 39))
    draw.text((12, 301), f"{depth.shape[1]}×{depth.shape[0]}  ·  0–{depth_max:g} m  ·  近黑远白",
              font=ImageFont.truetype(FONT_PATH, 15), fill=(65, 75, 85))
    return panel


def title_panel(terrain_name: str) -> Image.Image:
    """将中文预览说明栅格化，以便原生查看器正确显示。"""
    panel = Image.new("RGB", (480, 82), (20, 30, 42))
    draw = ImageDraw.Draw(panel)
    draw.text((14, 7), f"Go1 深度相机预览 · {TERRAIN_TITLES[terrain_name]}",
              font=ImageFont.truetype(FONT_PATH, 24), fill=(240, 245, 250))
    draw.text((14, 46), "静态姿态 · 黄色线为相机视锥 · 鼠标旋转 / 缩放",
              font=ImageFont.truetype(FONT_PATH, 16), fill=(190, 205, 220))
    return panel


def draw_image_panel(image: Image.Image, rectangle, context) -> None:
    """先建立二维绘制状态，再将面板覆盖到当前帧缓冲，避免被三维深度测试丢弃。"""
    import mujoco

    mujoco.mjr_rectangle(rectangle, 1.0, 1.0, 1.0, 1.0)
    mujoco.mjr_drawPixels(np.ascontiguousarray(np.flipud(image)).ravel(), None, rectangle, context)


def save_preview(model, data, cfg, depth, panel, title, output_dir: Path, terrain_name: str, difficulty: float, depth_max: float):
    """保存真实渲染截图、原始深度及可追溯相机参数。"""
    import mujoco

    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[[0, 2]] = 1
    camera_id = model.camera(cfg.sensors.depth.camera_name).id
    with mujoco.Renderer(model, height=900, width=1440) as renderer:
        renderer.update_scene(data, camera=observer_camera(), scene_option=option)
        add_frustum(renderer.scene, model, data, camera_id, cfg.sensors.depth)
        image = Image.fromarray(renderer.render().copy())
    image.paste(panel, (1440 - panel.width - 20, 20))
    image.paste(title, (20, 20))
    output_dir.mkdir(parents=True, exist_ok=True)
    image.save(output_dir / "go1_depth_preview.png")
    panel.crop((12, 39, 268, 295)).resize((depth.shape[1], depth.shape[0]), Image.Resampling.NEAREST).convert("L").save(output_dir / "depth.png")
    np.save(output_dir / "depth_m.npy", depth)
    record = {
        "preview_type": "static_robot_and_camera", "terrain": terrain_name, "difficulty": difficulty,
        "mujoco_version": mujoco.__version__, "depth_type": "optical_axis_z_m", "depth_shape": list(depth.shape),
        "display_range_m": [0, depth_max], "display_mapping": "near_black_far_white",
        "depth_min_m": float(depth.min()), "depth_max_m": float(depth.max()),
        "camera_name": cfg.sensors.depth.camera_name, "camera_world_position_m": data.cam_xpos[camera_id].tolist(),
        "camera_world_rotation": data.cam_xmat[camera_id].reshape(3, 3).tolist(),
        "camera_vfov_deg": float(model.cam_fovy[camera_id]), "visible_geom_groups": list(cfg.sensors.depth.enabled_geom_groups),
        "robot_qpos": data.qpos.tolist(), "policy_loaded": False,
    }
    (output_dir / "preview.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved robot preview and metric depth to: {output_dir}", flush=True)


def run_viewer(model, data, cfg, title, depth_max: float, duration: float | None) -> None:
    """在同一线程管理 GLFW 窗口与深度上下文，支持旋转、平移和缩放。"""
    import glfw
    import mujoco

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW; use --headless to save images without a desktop.")
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
    window = glfw.create_window(1440, 900, "Go1 Depth Camera Preview", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("Failed to create the preview window.")
    try:
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        camera = observer_camera()
        scene = mujoco.MjvScene(model, maxgeom=5000)
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        try:
            previous = list(glfw.get_cursor_pos(window))

            def move_mouse(window, x, y):
                """左键旋转、右键平移；位移按窗口高度归一化。"""
                dx, dy = x - previous[0], y - previous[1]
                previous[:] = [x, y]
                height = max(1, glfw.get_window_size(window)[1])
                if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
                    mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ROTATE_V, dx / height, dy / height, camera)
                elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
                    mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_MOVE_V, dx / height, dy / height, camera)

            glfw.set_cursor_pos_callback(window, move_mouse)
            # 正向滚动拉近相机；本机 MuJoCo 接口不再接受 scene 参数。
            glfw.set_scroll_callback(window, lambda window, x, y: mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, 0.05 * y, camera))
            option = mujoco.MjvOption()
            option.geomgroup[:] = 0
            option.geomgroup[[0, 2]] = 1
            depth_cfg = cfg.sensors.depth
            with mujoco.Renderer(model, height=depth_cfg.height, width=depth_cfg.width) as renderer:
                start = time.monotonic()
                print("Viewer ready. Left-drag to rotate, right-drag to pan, scroll to zoom, and close the window to exit.", flush=True)
                while not glfw.window_should_close(window) and (duration is None or time.monotonic() - start < duration):
                    frame_start = time.monotonic()
                    panel = depth_panel(render_depth(renderer, data, depth_cfg), depth_max)
                    # 深度渲染会切换到自己的离屏上下文，主视图绘制前必须切回窗口。
                    glfw.make_context_current(window)
                    width, height = glfw.get_framebuffer_size(window)
                    mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
                    add_frustum(scene, model, data, model.camera(depth_cfg.camera_name).id, depth_cfg)
                    mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), scene, context)
                    for image, left in ((panel, max(0, width - panel.width - 20)), (title, 20)):
                        rectangle = mujoco.MjrRect(left, max(0, height - image.height - 20), image.width, image.height)
                        draw_image_panel(image, rectangle, context)
                    glfw.swap_buffers(window)
                    glfw.poll_events()
                    time.sleep(max(0, 0.1 - (time.monotonic() - frame_start)))
        finally:
            glfw.make_context_current(window)
            context.free()
    finally:
        glfw.destroy_window(window)
        glfw.terminate()


def main() -> None:
    """处理预览参数，先保存图片，再按需打开桌面窗口。"""
    parser = argparse.ArgumentParser(description="Preview the Go1 robot, camera frustum, and real depth rendering.")
    parser.add_argument("--headless", action="store_true", help="Save images without opening a viewer.")
    parser.add_argument("--terrain", choices=tuple(TERRAIN_TITLES), default="stairs_up", help="Existing terrain to preview.")
    parser.add_argument("--difficulty", type=float, default=0.35, help="Terrain difficulty in [0, 1].")
    parser.add_argument("--depth-max", type=float, default=3.0, help="Maximum displayed depth in meters; does not alter raw depth.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/go1_depth_preview", help="Directory for preview artifacts.")
    parser.add_argument("--viewer-seconds", type=float, default=None, help="Optional duration limit for viewer checks.")
    args = parser.parse_args()
    if not math.isfinite(args.difficulty) or not 0 <= args.difficulty <= 1:
        parser.error("difficulty must be finite and within [0, 1]")
    if not math.isfinite(args.depth_max) or args.depth_max <= 0:
        parser.error("depth-max must be finite and positive")
    if args.viewer_seconds is not None and (not math.isfinite(args.viewer_seconds) or args.viewer_seconds <= 0):
        parser.error("viewer-seconds must be finite and positive")
    if not args.headless and os.environ.get("MUJOCO_GL", "glfw") != "glfw":
        parser.error("interactive preview requires MUJOCO_GL=glfw; use --headless for other backends")
    os.environ.setdefault("MUJOCO_GL", "egl" if args.headless else "glfw")
    import mujoco

    cfg, model, data = build_scene(args.terrain, args.difficulty)
    depth_cfg = cfg.sensors.depth
    with mujoco.Renderer(model, height=depth_cfg.height, width=depth_cfg.width) as renderer:
        depth = render_depth(renderer, data, depth_cfg)
        panel, title = depth_panel(depth, args.depth_max), title_panel(args.terrain)
        save_preview(model, data, cfg, depth, panel, title, args.output_dir, args.terrain, args.difficulty, args.depth_max)
    if not args.headless:
        run_viewer(model, data, cfg, title, args.depth_max, args.viewer_seconds)


if __name__ == "__main__":
    main()
