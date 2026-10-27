# Copyright (c) 2026-2027 zh
"""
内容：
    使用 MuJoCo 离屏渲染盲走地形，保存单图、总览、模型和复现记录。

用法：
    uv run --no-sync python -B tools/preview_blind_terrain.py --output-dir outputs/terrain/blind
    uv run --no-sync python -B tools/preview_blind_terrain.py --terrain rough --difficulty 1 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TERRAINS = ("flat", "rough", "pyramid_stairs", "pyramid_stairs_inv", "slope_up", "slope_down")
TITLES = {
    "flat": ("平地", "Flat ground"),
    "rough": ("不规则高度图", "Irregular heightfield"),
    "pyramid_stairs": ("正金字塔楼梯", "Pyramid stairs"),
    "pyramid_stairs_inv": ("倒金字塔楼梯", "Inverted pyramid stairs"),
    "slope_up": ("正金字塔坡", "Pyramid slope"),
    "slope_down": ("倒金字塔坡", "Inverted pyramid slope"),
}
IMAGE_SIZE = (960, 640)
BACKGROUND = (16, 23, 34)
TEXT_COLOR = (232, 240, 248)
MUTED_COLOR = (159, 180, 199)


# ---------- 参数与复现记录 ----------

def parse_args() -> argparse.Namespace:
    """读取预览范围和可复现的随机参数。"""
    parser = argparse.ArgumentParser(description="Render blind terrains with MuJoCo and save reproducible previews.")
    parser.add_argument("--terrain", choices=("all", *TERRAINS), default="all", help="Terrain type to render.")
    parser.add_argument("--difficulty", type=float, default=0.75, help="Terrain difficulty in [0, 1].")
    parser.add_argument("--seed", type=int, default=42, help="Non-negative random seed.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/terrain/blind"), help="Output directory.")
    args = parser.parse_args()
    if not math.isfinite(args.difficulty) or not 0.0 <= args.difficulty <= 1.0:
        parser.error("--difficulty must be finite and in [0, 1].")
    if args.seed < 0:
        parser.error("--seed must be non-negative.")
    return args


def source_record() -> dict:
    """记录版本及实际源文件摘要，覆盖尚未提交的新文件。"""
    def git_output(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *arguments], capture_output=True, text=True, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    paths = sorted((PROJECT_ROOT / "src" / "terrain").rglob("*.py")) + [Path(__file__).resolve()]
    return {
        "git_head": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short"),
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
    }


# ---------- 场景与渲染 ----------

def make_spec(mujoco):
    """设置只影响显示的背景与灯光，不额外添加碰撞地面。"""
    width, height = IMAGE_SIZE
    return mujoco.MjSpec.from_string(f"""
    <mujoco model="blind_terrain_preview">
      <visual>
        <global offwidth="{width}" offheight="{height}"/>
        <quality shadowsize="2048" offsamples="4"/>
        <headlight ambient="0.18 0.18 0.18" diffuse="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
        <map znear="0.01" zfar="20"/>
      </visual>
      <asset>
        <texture name="background" type="skybox" builtin="gradient" width="256" height="1536"
                 rgb1="0.063 0.09 0.133" rgb2="0.09 0.13 0.19"/>
      </asset>
      <worldbody>
        <light name="key" type="directional" pos="-3 -4 8" dir="0.25 0.6 -1"
               diffuse="0.8 0.8 0.8" specular="0.12 0.12 0.12" castshadow="true"/>
        <light name="fill" type="directional" pos="4 3 5" dir="-0.4 -0.6 -1"
               diffuse="0.3 0.35 0.4" specular="0 0 0" castshadow="false"/>
      </worldbody>
    </mujoco>
    """)


def add_start_markers(spec, origin, width: float, mujoco) -> None:
    """用不参与碰撞的站点表示起点与 +X，保留真实地形表面。"""
    x, y, z = map(float, origin)
    spec.worldbody.add_site(
        name="start_marker", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=(x, y, z + 0.012), size=(0.13, 0.01, 0.01), rgba=(0.3, 0.95, 0.65, 1.0),
    )
    # 箭头紧靠起点一侧，三个胶囊站点均无碰撞属性。
    arrow_y, arrow_z = y - width * 0.32, z + 0.035
    segments = (
        (x - 0.3, arrow_y, x + 0.5, arrow_y),
        (x + 0.5, arrow_y, x + 0.28, arrow_y - 0.15),
        (x + 0.5, arrow_y, x + 0.28, arrow_y + 0.15),
    )
    for index, (x0, y0, x1, y1) in enumerate(segments):
        spec.worldbody.add_site(
            name=f"forward_marker_{index}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=(x0, y0, arrow_z, x1, y1, arrow_z), size=(0.025, 0.025, 0.025),
            rgba=(0.3, 0.95, 0.65, 1.0),
        )


def surface_range(model, data, mujoco) -> tuple[float, float]:
    """从编译模型取得实际表面高度范围，排除底座及显示站点。"""
    heights = []
    for index, geom_type in enumerate(model.geom_type):
        z = float(data.geom_xpos[index, 2])
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            heights.append(z + float(model.geom_size[index, 2]))
        elif geom_type == mujoco.mjtGeom.mjGEOM_HFIELD:
            field_id = model.geom_dataid[index]
            start = model.hfield_adr[field_id]
            count = model.hfield_nrow[field_id] * model.hfield_ncol[field_id]
            elevation = model.hfield_data[start:start + count] * model.hfield_size[field_id, 2] + z
            heights.extend((float(elevation.min()), float(elevation.max())))
    if not heights:
        raise RuntimeError("The compiled terrain contains no supported surface geometry.")
    return min(heights), max(heights)


def render_terrain(name: str, args, fonts, chinese: bool, mujoco, Image, ImageDraw) -> tuple:
    """装配单个真实地形、渲染图片并导出对应编译模型。"""
    from mjlab.terrains import TerrainGenerator
    from src.terrain.config import make_blind_terrain_cfg

    cfg = make_blind_terrain_cfg(
        name, seed=args.seed, num_rows=1, num_cols=1,
        difficulty_range=(args.difficulty, args.difficulty),
    )
    spec = make_spec(mujoco)
    generator = TerrainGenerator(cfg)
    generator.compile(spec)
    origin = generator.terrain_origins[0, 0]
    add_start_markers(spec, origin, cfg.size[1], mujoco)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    minimum, maximum = surface_range(model, data, mujoco)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    # 所有类别共用视角、焦点和距离，避免把低起伏视觉放大。
    camera.lookat[:] = (0.0, 0.0, 0.45)
    camera.azimuth = 135.0
    camera.elevation = -28.0
    camera.distance = 16.0
    width, height = IMAGE_SIZE
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera)
        pixels = renderer.render().copy()

    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 108), fill=BACKGROUND)
    draw.text((34, 16), TITLES[name][0 if chinese else 1], fill=TEXT_COLOR, font=fonts["title"])
    detail = terrain_detail(name, cfg.sub_terrains[name], args.difficulty, chinese)
    draw.text((35, 64), detail, fill=MUTED_COLOR, font=fonts["body"])
    draw.rectangle((0, height - 72, width, height), fill=BACKGROUND)
    footnote = (
        "绿色标记：起点与 +X 前进方向  ·  真实几何比例"
        if chinese else "Green markers: start and +X travel direction  |  True geometry scale"
    )
    draw.text((35, height - 60), footnote, fill=MUTED_COLOR, font=fonts["small"])
    scale = (
        f"地块 {cfg.size[0]:g} × {cfg.size[1]:g} 米  ·  难度 {args.difficulty:g}  ·  种子 {args.seed}"
        if chinese else f"Patch {cfg.size[0]:g} x {cfg.size[1]:g} m  |  Difficulty {args.difficulty:g}  |  Seed {args.seed}"
    )
    draw.text((35, height - 33), scale, fill=MUTED_COLOR, font=fonts["small"])
    image_path = args.output_dir / f"{name}.png"
    model_path = args.output_dir / f"{name}.mjb"
    image.save(image_path)
    mujoco.mj_saveModel(model, str(model_path))
    print(f"Rendered {name}: {image_path} (surface z={minimum:.4f}..{maximum:.4f} m)")
    record = {
        "terrain": name,
        "generator_config": asdict(cfg),
        "origin": origin.tolist(),
        "surface_height_range_m": [minimum, maximum],
        "camera": {"lookat": camera.lookat.tolist(), "azimuth": camera.azimuth,
                   "elevation": camera.elevation, "distance": camera.distance},
        "image": image_path.name,
        "compiled_model": model_path.name,
        "compiled_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "num_geoms": int(model.ngeom),
        "num_heightfields": int(model.nhfield),
    }
    return image, record


# ---------- 中文标注与总览 ----------

def load_fonts(ImageFont) -> tuple[dict, bool]:
    """优先使用系统中文字体，缺失时显式使用英文标注。"""
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
    sizes = {"title": 34, "body": 22, "small": 18, "overview": 38}
    fonts = {
        key: ImageFont.truetype(str(font_path), size) if font_path.exists() else ImageFont.load_default(size=size)
        for key, size in sizes.items()
    }
    return fonts, chinese


def terrain_detail(name: str, cfg, difficulty: float, chinese: bool) -> str:
    """根据当前配置标注物理参数，不把范围端点误写成实测高度。"""
    def interpolate(values):
        return values[0] + difficulty * (values[1] - values[0])

    if name == "flat":
        return "有限平面  ·  表面高度 0 米" if chinese else "Finite flat surface  |  Surface height 0 m"
    if name == "rough":
        amplitude = interpolate(cfg.amplitude_range)
        return (
            f"随机起伏上限 ±{amplitude:.3f} 米  ·  网格 {cfg.horizontal_scale:g} 米"
            if chinese else f"Noise bound +/-{amplitude:.3f} m  |  Grid {cfg.horizontal_scale:g} m"
        )
    if name.startswith("pyramid_stairs"):
        height = interpolate(cfg.step_height_range)
        return (
            f"台阶高度 {height:.3f} 米  ·  踏面 {cfg.step_width:g} 米  ·  中心平台 {cfg.platform_width:g} 米"
            if chinese else f"Step height {height:.3f} m  |  Step width {cfg.step_width:g} m  |  Center platform {cfg.platform_width:g} m"
        )
    slope = interpolate(cfg.slope_range)
    degrees = math.degrees(math.atan(slope))
    direction = "中心下沉" if cfg.descending else "中心升高"
    direction_en = "Center down" if cfg.descending else "Center up"
    return (
        f"坡度 {slope:.4g}  ·  倾角 {degrees:.1f}°  ·  {direction}  ·  "
        f"中心平台 {cfg.platform_width:g} 米"
        if chinese
        else f"Rise/run {slope:.4g}  |  Angle {degrees:.1f} deg  |  "
        f"{direction_en}  |  Center platform {cfg.platform_width:g} m"
    )


def save_overview(images: list, args, fonts, chinese: bool, Image, ImageDraw) -> Path:
    """以统一尺寸并列展示实际截图，不拉伸地形或改变其高度。"""
    columns = min(2, len(images))
    rows = math.ceil(len(images) / columns)
    gap, margin, header = 20, 28, 130
    width, height = IMAGE_SIZE
    canvas = Image.new(
        "RGB", (margin * 2 + columns * width + (columns - 1) * gap,
                header + rows * height + (rows - 1) * gap + margin), BACKGROUND,
    )
    draw = ImageDraw.Draw(canvas)
    title = "盲走基础地形" if chinese else "Blind locomotion terrains"
    subtitle = (
        "MuJoCo 实际渲染  /  平地、随机高度图、正/倒金字塔楼梯与正/倒金字塔坡"
        if chinese else "MuJoCo render  /  Flat ground, heightfield, pyramid stairs and pyramid slopes"
    )
    draw.text((margin + 6, 17), title, fill=TEXT_COLOR, font=fonts["overview"])
    draw.text((margin + 6, 76), subtitle, fill=MUTED_COLOR, font=fonts["body"])
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        canvas.paste(image, (margin + column * (width + gap), header + row * (height + gap)))
    path = args.output_dir / "overview.png"
    canvas.save(path)
    return path


# ---------- 主入口 ----------

def main() -> None:
    """批量渲染并写入完整复现记录，失败时保留异常退出状态。"""
    args = parse_args()
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(PROJECT_ROOT))
    # OpenGL 后端必须先于 MuJoCo 的导入确定；--help 无需加载仿真依赖。
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fonts, chinese = load_fonts(ImageFont)
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "seed": args.seed,
        "difficulty": args.difficulty,
        "image_size_px": list(IMAGE_SIZE),
        "renderer": "MuJoCo offscreen",
        "opengl_backend": os.environ.get("MUJOCO_GL", "default"),
        "versions": {name: metadata.version(name) for name in ("mjlab", "mujoco", "numpy", "pillow")},
        "python_version": sys.version,
        "code": source_record(),
        "terrains": [],
    }
    names = TERRAINS if args.terrain == "all" else (args.terrain,)
    images = []
    for name in names:
        image, terrain_record = render_terrain(name, args, fonts, chinese, mujoco, Image, ImageDraw)
        images.append(image)
        record["terrains"].append(terrain_record)
    overview_path = save_overview(images, args, fonts, chinese, Image, ImageDraw)
    record["overview"] = overview_path.name
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved overview: {overview_path}")
    print(f"Saved reproduction metadata: {metadata_path}")


if __name__ == "__main__":
    main()
