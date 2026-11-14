# Copyright (c) 2026-2027 zh
"""
内容：
    装配视觉地形的四列课程，渲染完整网格和最高难度地块，导出灰度高度场与复现记录。

用法：
    uv run --no-sync python -B tools/preview_vision_terrain.py --output-dir outputs/terrain/vision
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
BACKGROUND = (18, 25, 36)
TEXT = (235, 242, 249)
MUTED = (167, 187, 205)
TITLES = {"slope": "上下坡", "rough": "随机高度场", "stairs_up": "上楼梯", "stairs_down": "下楼梯"}


# ---------- 场景与相机 ----------

def configure_visuals(spec) -> None:
    """仅设置截图灯光与离屏画布，不改变碰撞几何。"""
    import mujoco

    spec.visual.global_.offwidth = 1800
    spec.visual.global_.offheight = 1800
    spec.visual.headlight.ambient = (0.25, 0.25, 0.25)
    spec.visual.headlight.diffuse = (0.35, 0.35, 0.35)
    spec.visual.map.znear = 0.01
    spec.visual.map.zfar = 300.0
    spec.worldbody.add_light(
        name="preview_key", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=(0.0, -20.0, 40.0), dir=(0.3, 0.4, -1.0),
        diffuse=(0.65, 0.65, 0.65), specular=(0.0, 0.0, 0.0), castshadow=True,
    )


def render(model, *, lookat, distance, azimuth, elevation, size):
    """使用实际编译模型按真实物理比例生成像素。"""
    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    with mujoco.Renderer(model, height=size[1], width=size[0]) as renderer:
        renderer.update_scene(data, camera)
        pixels = renderer.render().copy()
    record = {
        "lookat_m": list(lookat), "distance_m": distance,
        "azimuth_deg": azimuth, "elevation_deg": elevation, "render_size_px": list(size),
    }
    return pixels, record


def build_scene(cfg):
    """通过真实的 MjLab 场景装配链路编译课程地形。"""
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainEntityCfg

    scene_cfg = SceneCfg(
        num_envs=cfg.num_rows * len(cfg.sub_terrains),
        terrain=TerrainEntityCfg(terrain_type="generator", terrain_generator=cfg, debug_vis=False),
        spec_fn=configure_visuals,
    )
    scene = Scene(scene_cfg, device="cpu")
    return scene, scene.compile()


# ---------- 中文标注与灰度数据 ----------

def make_top_view(model, cfg, fonts):
    """俯视展示固定列顺序，在左侧对应标注各行的角度和级高。"""
    from PIL import Image, ImageDraw

    render_size = (1120, 1500)
    half_x = cfg.num_rows * cfg.size[0] / 2
    half_y = len(cfg.sub_terrains) * cfg.size[1] / 2
    tangent = math.tan(math.radians(model.vis.global_.fovy / 2))
    distance = 1.12 * max(half_x / tangent, half_y / (tangent * render_size[0] / render_size[1]))
    pixels, camera = render(
        model, lookat=(0.0, 0.0, 0.0), distance=distance,
        azimuth=180.0, elevation=-89.99, size=render_size,
    )
    image = Image.new("RGB", (1500, 1740), BACKGROUND)
    image.paste(Image.fromarray(pixels), (350, 160))
    draw = ImageDraw.Draw(image)
    draw.text((28, 20), "视觉地形 · 前四列课程", font=fonts["title"], fill=TEXT)
    draw.text((30, 76), f"{cfg.num_rows} 行 × 4 列 · 每块 {cfg.size[0]:g} × {cfg.size[1]:g} m · 真实 MuJoCo 碰撞几何", font=fonts["body"], fill=MUTED)
    draw.text((28, 142), "行 / 坡角 / 台阶级高", font=fonts["body"], fill=TEXT)
    scale = render_size[1] / (2 * distance * tangent)
    for col, name in enumerate(cfg.sub_terrains):
        center_x = 350 + render_size[0] / 2 + ((col + 0.5) * cfg.size[1] - half_y) * scale
        draw.text((center_x, 122), f"{col + 1} {TITLES[name]}", font=fonts["body"], fill=TEXT, anchor="mt")
    slope = cfg.sub_terrains["slope"]
    stairs = cfg.sub_terrains["stairs_up"]
    for row in range(cfg.num_rows):
        difficulty = row / max(cfg.num_rows - 1, 1)
        angle = slope.angle_range_deg[0] + difficulty * (slope.angle_range_deg[1] - slope.angle_range_deg[0])
        height_cm = 100 * (stairs.step_height_range[0] + difficulty * (stairs.step_height_range[1] - stairs.step_height_range[0]))
        center_y = 160 + render_size[1] / 2 + ((row + 0.5) * cfg.size[0] - half_x) * scale
        draw.text((28, center_y), f"{row + 1:02d}  /  {angle:4.1f}°  /  {height_cm:4.1f} cm", font=fonts["body"], fill=TEXT, anchor="lm")
    draw.text((30, 1680), "难度由上向下递增；楼梯踏面 20 cm；高度场保留 Go2 参考配方，各行粗糙度相同。", font=fonts["body"], fill=MUTED)
    return image, camera


def make_oblique_view(model, cfg, fonts):
    """斜视显示完整地图的立体关系，不放大垂直高度。"""
    from PIL import Image, ImageDraw

    map_x, map_y = cfg.num_rows * cfg.size[0], len(cfg.sub_terrains) * cfg.size[1]
    pixels, camera = render(
        model, lookat=(0.0, 0.0, 0.0), distance=1.65 * math.hypot(map_x, map_y),
        azimuth=125.0, elevation=-52.0, size=(1800, 1100),
    )
    image = Image.new("RGB", (1800, 1230), BACKGROUND)
    image.paste(Image.fromarray(pixels), (0, 115))
    draw = ImageDraw.Draw(image)
    draw.text((32, 16), "完整课程地形 · 斜视", font=fonts["title"], fill=TEXT)
    draw.text((34, 71), "列顺序：上下坡 → 随机高度场 → 上楼梯 → 下楼梯；难度沿行递增。", font=fonts["body"], fill=MUTED)
    return image, camera


def make_maximum_view(cfg, seed, output_dir, fonts):
    """独立生成难度一的四类地块，并导出真实高度场的灰度数据。"""
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1600, 1240), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((30, 14), "四类地块 · 最高难度", font=fonts["title"], fill=TEXT)
    draw.text((32, 68), "坡角 40°；楼梯踏面 20 cm、级高 20 cm；所有图像保持真实比例。", font=fonts["body"], fill=MUTED)
    records = []
    for index, (name, sub_cfg) in enumerate(cfg.sub_terrains.items()):
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name="terrain")
        configure_visuals(spec)
        output = sub_cfg.function(1.0, spec, np.random.default_rng(seed))
        model = spec.compile()
        pixels, camera = render(
            model, lookat=(cfg.size[0] / 2, cfg.size[1] / 2, 0.0),
            distance=1.55 * math.hypot(*cfg.size), azimuth=130.0,
            elevation=-38.0, size=(800, 500),
        )
        left, top = (index % 2) * 800, 120 + (index // 2) * 560
        image.paste(Image.fromarray(pixels), (left, top))
        draw.text((left + 24, top + 504), f"第 {index + 1} 列  {TITLES[name]}", font=fonts["body"], fill=TEXT)
        record = {"name": name, "difficulty": 1.0, "origin_m": output.origin.tolist(), "camera": camera}
        if name == "rough":
            field_id = int(model.geom_dataid[np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD)[0]])
            shape = (int(model.hfield_nrow[field_id]), int(model.hfield_ncol[field_id]))
            address = int(model.hfield_adr[field_id])
            normalized = model.hfield_data[address:address + math.prod(shape)].reshape(shape)
            gray = np.rint(normalized * 255).astype(np.uint8)
            # 直接保存编译高度场矩阵，不添加文字、不翻转或增强灰度对比。
            Image.fromarray(gray).save(output_dir / "rough_heightmap.png")
            record["heightmap"] = {
                "file": "rough_heightmap.png", "shape_yx": list(shape),
                "gray_mapping": "gray / 255 * elevation_z + geom_z; white is high",
                "elevation_z_m": float(model.hfield_size[field_id, 2]),
                "geom_z_m": float(model.geom_pos[0, 2]),
                "note_zh": "编译后碰撞高度数据的八位可视化，未改变实际地形；图像首行对应局部最小 Y。",
            }
        records.append(record)
    return image, records


# ---------- 元数据与主入口 ----------

def source_record():
    """记录参与本次场景生成的源码，包括本地未跟踪文件。"""
    paths = [
        PROJECT_ROOT / "src/terrain/__init__.py",
        *sorted((PROJECT_ROOT / "src/terrain/vision").glob("*.py")),
        PROJECT_ROOT / "src/terrain/blind/common.py",
        PROJECT_ROOT / "src/terrain/blind/stairs_terrain.py",
        Path(__file__).resolve(),
    ]
    return {
        "git_head": subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "git_status_short": subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "status", "--short"], text=True),
        "source_sha256": {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }


def main() -> None:
    """执行课程装配、真实渲染与产物记录。"""
    parser = argparse.ArgumentParser(description="Render the first four vision terrain columns and maximum-difficulty patches.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/terrain/vision"), help="Output directory for PNG, MJB, and JSON files.")
    parser.add_argument("--num-rows", type=int, default=10, help="Number of difficulty rows; one row uses the lowest difficulty.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(PROJECT_ROOT))

    import mujoco
    import torch
    from PIL import ImageFont
    from src.terrain.vision import VISION_TERRAIN_NAMES, make_vision_terrain_cfg

    cfg = make_vision_terrain_cfg(seed=args.seed, num_rows=args.num_rows)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fonts = {"title": ImageFont.truetype(str(FONT_PATH), 38), "body": ImageFont.truetype(str(FONT_PATH), 23)}
    scene, model = build_scene(cfg)
    top, top_camera = make_top_view(model, cfg, fonts)
    top.save(args.output_dir / "vision_curriculum_top.png")
    oblique, oblique_camera = make_oblique_view(model, cfg, fonts)
    oblique.save(args.output_dir / "vision_curriculum_oblique.png")
    maximum, maximum_records = make_maximum_view(cfg, args.seed, args.output_dir, fonts)
    maximum.save(args.output_dir / "vision_max_difficulty.png")
    mujoco.mj_saveModel(model, str(args.output_dir / "vision_curriculum.mjb"))
    artifact_names = (
        "vision_curriculum_top.png", "vision_curriculum_oblique.png", "vision_max_difficulty.png",
        "rough_heightmap.png", "vision_curriculum.mjb",
    )
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "preview_type": "terrain_geometry_only",
        "seed": args.seed, "scene_chain": ["SceneCfg", "TerrainEntityCfg", "TerrainGeneratorCfg"],
        "terrain_columns": list(VISION_TERRAIN_NAMES), "config": asdict(cfg),
        "curriculum_difficulties": [row / max(cfg.num_rows - 1, 1) for row in range(cfg.num_rows)],
        "terrain_origins": scene.terrain.terrain_origins.detach().cpu().tolist(),
        "env_origins": scene.terrain.env_origins.detach().cpu().tolist(),
        "model": {"geoms": int(model.ngeom), "heightfields": int(model.nhfield), "envs": cfg.num_rows * len(cfg.sub_terrains)},
        "cameras": {"top": top_camera, "oblique": oblique_camera},
        "maximum_difficulty_patches": maximum_records,
        "notes_zh": ["未启动机器人或训练。", "粗糙列保留上游估计出生高度，不代表平坦平台或精确地表。", "各地块独立，不保证跨块高度连续。"],
        "versions": {name: version(name) for name in ("mjlab", "mujoco", "numpy", "pillow", "torch")},
        "source": source_record(),
        "artifacts": {name: {"sha256": hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest(), "bytes": (args.output_dir / name).stat().st_size} for name in artifact_names},
    }
    (args.output_dir / "vision_curriculum.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Rendered vision terrain: {cfg.num_rows} rows, {len(cfg.sub_terrains)} columns.")
    print(f"Saved images, compiled model, and metadata to: {args.output_dir}")


if __name__ == "__main__":
    main()
