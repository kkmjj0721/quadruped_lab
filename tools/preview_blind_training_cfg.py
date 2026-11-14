# Copyright (c) 2026-2027 zh
"""
内容：
    通过完整的盲走课程配置装配 MjLab 训练场景，渲染整张 MuJoCo 地形网格，
    并保存 PNG、编译后的 MJB 模型和可复现元数据。

用法：
    uv run --no-sync python -B tools/preview_blind_training_cfg.py \
        --output-dir outputs/terrain/blind_training
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SIZE = (1800, 1280)
RENDER_SIZE = (1800, 820)
HEADER_HEIGHT = 300
BACKGROUND = (16, 23, 34)
PANEL = (24, 34, 48)
PANEL_ALT = (31, 45, 62)
TEXT_COLOR = (236, 244, 250)
MUTED_COLOR = (164, 184, 202)
ACCENT = (92, 202, 155)
BLUE = (106, 158, 236)

TERRAIN_TITLES = {
    "flat": ("平地", "flat"),
    "rough": ("不规则高度图", "rough"),
    "pyramid_stairs": ("正金字塔楼梯", "pyramid_stairs"),
    "pyramid_stairs_inv": ("倒金字塔楼梯", "pyramid_stairs_inv"),
    "slope_up": ("正金字塔坡", "slope_up"),
    "slope_down": ("倒金字塔坡", "slope_down"),
}


# ---------- 参数与复现记录 ----------

def parse_args() -> argparse.Namespace:
    """读取场景预览参数。"""
    parser = argparse.ArgumentParser(description="渲染完整盲走课程训练场景装配预览。")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/terrain/blind_training"),
        help="PNG、MJB 和 JSON 的输出目录。",
    )
    parser.add_argument("--seed", type=int, default=42, help="地形生成随机种子。")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed 必须是非负整数。")
    return args


def git_output(*arguments: str) -> str | None:
    """读取 Git 信息；工作区不是 Git 仓库时返回空值。"""
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def source_record() -> dict[str, Any]:
    """记录本次预览涉及源码的摘要，包含尚未提交的脚本。"""
    paths = [
        PROJECT_ROOT / "src" / "terrain" / "config.py",
        *sorted((PROJECT_ROOT / "src" / "terrain" / "blind").glob("*.py")),
        Path(__file__).resolve(),
    ]
    return {
        "git_head": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short"),
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
    }


def package_versions() -> dict[str, str]:
    """收集渲染、场景和数值计算依赖版本。"""
    package_names = ("mjlab", "mujoco", "numpy", "pillow", "torch")
    versions = {}
    for name in package_names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


# ---------- 场景装配 ----------

def configure_visuals(spec, mujoco) -> None:
    """配置仅用于截图的背景、灯光和可视化范围。"""
    spec.visual.headlight.ambient = (0.16, 0.16, 0.16)
    spec.visual.headlight.diffuse = (0.28, 0.28, 0.28)
    spec.visual.headlight.specular = (0.04, 0.04, 0.04)
    spec.visual.map.znear = 0.05
    spec.visual.map.zfar = 150.0
    spec.worldbody.add_light(
        name="preview_key",
        pos=(-18.0, -24.0, 45.0),
        dir=(0.35, 0.48, -1.0),
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=(0.48, 0.52, 0.58),
        specular=(0.08, 0.08, 0.08),
        castshadow=True,
    )
    spec.worldbody.add_light(
        name="preview_fill",
        pos=(28.0, 16.0, 32.0),
        dir=(-0.45, -0.3, -1.0),
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=(0.16, 0.19, 0.24),
        specular=(0.0, 0.0, 0.0),
        castshadow=False,
    )


def build_scene(seed: int, torch) -> tuple[Any, Any, Any]:
    """按训练场景的真实装配链路构造配置、Scene 和编译模型。"""
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainEntityCfg
    from src.terrain.config import make_blind_terrain_cfg
    import mujoco

    torch.manual_seed(seed)
    terrain_generator_cfg = make_blind_terrain_cfg(
        seed=seed,
        size=(8.0, 8.0),
        num_rows=5,
        num_cols=6,
        curriculum=True,
        difficulty_range=(0.0, 1.0),
    )
    terrain_entity_cfg = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=terrain_generator_cfg,
        debug_vis=True,
    )
    scene_cfg = SceneCfg(
        num_envs=30,
        terrain=terrain_entity_cfg,
        spec_fn=lambda spec: configure_visuals(spec, mujoco),
    )
    scene = Scene(scene_cfg, device="cpu")
    model = scene.compile()
    return scene, scene_cfg, model


# ---------- 离屏渲染 ----------

def load_fonts(ImageFont) -> tuple[dict[str, Any], bool]:
    """优先加载中文字体；没有中文字体时回退到英文标注。"""
    candidates = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    )
    font_path = next((path for path in candidates if path.exists()), None)
    chinese = font_path is not None
    if font_path is None:
        print("No CJK font found; using English image captions.")
        font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    sizes = {
        "title": 38,
        "subtitle": 21,
        "chain": 20,
        "column": 22,
        "column_code": 15,
        "body": 18,
        "small": 16,
    }
    fonts = {
        name: ImageFont.truetype(str(font_path), size)
        for name, size in sizes.items()
    }
    return fonts, chinese


def make_camera(mujoco):
    """创建覆盖 5×6 地形课程网格的自由相机。"""
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.35)
    camera.azimuth = 136.0
    camera.elevation = -48.0
    camera.distance = 65.0
    return camera


def render_scene(model, mujoco):
    """使用真实编译模型和地形碰撞几何生成离屏像素。"""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = make_camera(mujoco)
    width, height = RENDER_SIZE
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera)
        pixels = renderer.render().copy()
    return pixels, camera, data


def draw_centered(draw, box: tuple[int, int, int, int], text: str, font, fill) -> None:
    """在矩形中水平居中文本。"""
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=font)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    draw.text(
        ((left + right - text_width) / 2, (top + bottom - text_height) / 2 - bounds[1]),
        text,
        font=font,
        fill=fill,
    )


def annotate_image(pixels, fonts, chinese: bool, Image, ImageDraw):
    """添加装配链路、六列名称、难度行和预览边界说明。"""
    image = Image.new("RGB", IMAGE_SIZE, BACKGROUND)
    render_image = Image.fromarray(pixels).convert("RGB")
    image.paste(render_image, (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(image)

    title = "训练场景装配预览（不是已运行 PPO 训练）" if chinese else (
        "Training-scene assembly preview (not a running PPO training)"
    )
    subtitle = (
        "真实 MuJoCo 离屏渲染 · 5 行课程 × 6 列地形 · 30 个环境出生点"
        if chinese else
        "Real MuJoCo offscreen render · 5 curriculum rows × 6 terrain columns · 30 env origins"
    )
    draw.text((36, 22), title, font=fonts["title"], fill=TEXT_COLOR)
    draw.text((38, 76), subtitle, font=fonts["subtitle"], fill=MUTED_COLOR)

    chain_y = 122
    chain_items = (
        ("SceneCfg", ACCENT),
        ("TerrainEntityCfg", BLUE),
        ("TerrainGeneratorCfg", (234, 180, 92)),
    )
    chain_x = 38
    for index, (label, color) in enumerate(chain_items):
        bounds = draw.textbbox((chain_x, chain_y), label, font=fonts["chain"])
        box = (chain_x - 12, chain_y - 7, bounds[2] + 12, chain_y + 34)
        draw.rounded_rectangle(box, radius=10, fill=PANEL_ALT, outline=color, width=2)
        draw.text((chain_x, chain_y), label, font=fonts["chain"], fill=TEXT_COLOR)
        chain_x = bounds[2] + 38
        if index != len(chain_items) - 1:
            draw.text((chain_x - 22, chain_y), "→", font=fonts["chain"], fill=MUTED_COLOR)

    map_left, map_top = 700, 112
    map_width, map_height = 1060, 160
    draw.rounded_rectangle(
        (map_left, map_top, map_left + map_width, map_top + map_height),
        radius=12, fill=PANEL, outline=(58, 76, 96), width=1,
    )
    map_title = "课程列（sub_terrains）" if chinese else "Curriculum columns (sub_terrains)"
    draw.text((map_left + 20, map_top + 12), map_title, font=fonts["body"], fill=MUTED_COLOR)
    column_width = (map_width - 40) // 6
    for index, name in enumerate(TERRAIN_TITLES):
        x0 = map_left + 10 + index * column_width
        x1 = x0 + column_width - 6
        y0, y1 = map_top + 48, map_top + 138
        draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=PANEL_ALT, outline=BLUE, width=1)
        title_text = TERRAIN_TITLES[name][0] if chinese else TERRAIN_TITLES[name][1]
        draw_centered(draw, (x0 + 4, y0 + 7, x1 - 4, y0 + 51), title_text, fonts["column"], TEXT_COLOR)
        draw_centered(draw, (x0 + 4, y0 + 53, x1 - 4, y1 - 7), name, fonts["column_code"], MUTED_COLOR)

    footer_top = HEADER_HEIGHT + RENDER_SIZE[1]
    draw.rectangle((0, footer_top, IMAGE_SIZE[0], IMAGE_SIZE[1]), fill=BACKGROUND)
    footer_title = "课程行 / 难度" if chinese else "Curriculum rows / difficulty"
    draw.text((34, footer_top + 16), footer_title, font=fonts["body"], fill=MUTED_COLOR)
    row_left = 230
    row_width = 250
    gap = 15
    for row in range(5):
        x0 = row_left + row * (row_width + gap)
        x1 = x0 + row_width
        difficulty = row / 4.0
        label = f"第 {row + 1} 行 · 难度 {difficulty:.2f}" if chinese else (
            f"row {row + 1} · difficulty {difficulty:.2f}"
        )
        draw.rounded_rectangle(
            (x0, footer_top + 12, x1, footer_top + 56),
            radius=8, fill=PANEL, outline=(58, 76, 96), width=1,
        )
        draw_centered(draw, (x0 + 6, footer_top + 13, x1 - 6, footer_top + 55), label, fonts["small"], TEXT_COLOR)
    marker_text = (
        "绿色：环境出生点（debug_vis） · 蓝色：地形原点 · 课程难度沿行从 0 → 1"
        if chinese else
        "Green: env origins (debug_vis) · Blue: terrain origins · difficulty increases 0 → 1 by row"
    )
    draw.text((34, footer_top + 72), marker_text, font=fonts["small"], fill=MUTED_COLOR)
    return image


# ---------- 元数据与主入口 ----------

def tensor_record(value) -> dict[str, Any]:
    """把 MjLab 的张量原样记录为形状和数值。"""
    array = value.detach().cpu().numpy()
    return {"shape": list(array.shape), "values": array.tolist()}


def make_metadata(args, cfg, scene_cfg, scene, model, camera, image_path: Path, model_path: Path) -> dict[str, Any]:
    """生成包含实际场景原点、相机和源码摘要的 JSON 记录。"""
    terrain = scene.terrain
    if terrain is None or terrain.terrain_origins is None or terrain.env_origins is None:
        raise RuntimeError("Compiled scene has no terrain origins.")
    terrain_origins = tensor_record(terrain.terrain_origins)
    env_origins = tensor_record(terrain.env_origins)
    cfg_dict = asdict(cfg)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "preview_type": "scene_assembly_only",
        "training_status": "not_running_ppo",
        "seed": args.seed,
        "scene_chain": ["SceneCfg", "TerrainEntityCfg", "TerrainGeneratorCfg"],
        "scene_config": {
            "num_envs": scene_cfg.num_envs,
            "env_spacing_m": scene_cfg.env_spacing,
            "terrain_type": scene_cfg.terrain.terrain_type,
            "debug_vis": scene_cfg.terrain.debug_vis,
        },
        "terrain_generator_config": cfg_dict,
        "terrain_columns": [
            {"index": index, "name": name, "title_zh": TERRAIN_TITLES[name][0]}
            for index, name in enumerate(cfg.sub_terrains)
        ],
        "curriculum_difficulties": [row / max(cfg.num_rows - 1, 1) for row in range(cfg.num_rows)],
        "terrain_origins": terrain_origins,
        "env_origins": env_origins,
        "camera": {
            "type": "free",
            "lookat": camera.lookat.tolist(),
            "azimuth": float(camera.azimuth),
            "elevation": float(camera.elevation),
            "distance": float(camera.distance),
            "image_size_px": list(IMAGE_SIZE),
            "render_size_px": list(RENDER_SIZE),
        },
        "versions": {
            **package_versions(),
            "python": sys.version,
        },
        "artifacts": {
            "image": image_path.name,
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "compiled_model": model_path.name,
            "compiled_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "num_geoms": int(model.ngeom),
            "num_heightfields": int(model.nhfield),
            "num_sites": int(model.nsite),
        },
        "source": source_record(),
    }


def main() -> None:
    """装配、编译、渲染并保存完整训练场景预览产物。"""
    args = parse_args()
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(PROJECT_ROOT))

    import mujoco
    import torch
    from PIL import Image, ImageDraw, ImageFont

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fonts, chinese = load_fonts(ImageFont)
    scene, scene_cfg, model = build_scene(args.seed, torch)
    pixels, camera, _ = render_scene(model, mujoco)
    image = annotate_image(pixels, fonts, chinese, Image, ImageDraw)

    image_path = args.output_dir / "blind_training_scene.png"
    model_path = args.output_dir / "blind_training_scene.mjb"
    metadata_path = args.output_dir / "blind_training_scene.json"
    image.save(image_path)
    mujoco.mj_saveModel(model, str(model_path))
    metadata_record = make_metadata(
        args, scene_cfg.terrain.terrain_generator, scene_cfg, scene, model,
        camera, image_path, model_path,
    )
    metadata_path.write_text(
        json.dumps(metadata_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(f"Rendered training scene: {image_path}")
    print(f"Saved compiled model: {model_path}")
    print(f"Saved scene metadata: {metadata_path}")
    print(
        "Scene summary: 5 curriculum rows, 6 terrain columns, "
        f"30 envs, {model.ngeom} geoms, {model.nhfield} heightfields.",
    )


if __name__ == "__main__":
    main()
