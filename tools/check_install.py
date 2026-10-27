# Copyright (c) 2026-2027 zh
"""
内容：
    检查核心依赖、MuJoCo 基础步进以及 PyTorch 和 Warp 的指定计算设备。

用法：
    uv run --no-sync python -B tools/check_install.py --device cuda
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from importlib import metadata
from pathlib import Path


def check_install(device: str) -> None:
    """验证安装结果，不创建机器人任务或启动训练。"""
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"Python 3.11 is required; found {sys.version.split()[0]}.")
    print(f"Python {sys.version.split()[0]}: {sys.executable}")

    modules = {}
    for distribution, module_name in (
        ("mjlab", "mjlab"),
        ("mujoco", "mujoco"),
        ("mujoco-warp", "mujoco_warp"),
        ("warp-lang", "warp"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("rsl-rl-lib", "rsl_rl"),
        ("rsl-rl-lib", "rsl_rl.models"),
        ("rsl-rl-lib", "rsl_rl.runners"),
    ):
        modules[module_name] = importlib.import_module(module_name)
        print(f"Imported {module_name} ({distribution} {metadata.version(distribution)})")

    # 脚本从 tools/ 启动时显式加入仓库根目录，并检查算法确实来自本地源码。
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    for module_name in ("agents.base", "agents.base.algorithms", "agents.base.models", "agents.base.runners"):
        module = importlib.import_module(module_name)
        if not Path(module.__file__).resolve().is_relative_to(project_root / "agents"):
            raise RuntimeError(f"Local algorithm module resolved outside this project: {module_name}.")
        print(f"Imported local algorithm module {module_name}: {module.__file__}")

    mujoco = modules["mujoco"]
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body pos="0 0 1"><freejoint/>'
        '<geom type="sphere" size="0.1" mass="1"/>'
        '</body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    for _ in range(5):
        mujoco.mj_step(model, data)
    if data.time <= 0 or not all(math.isfinite(float(value)) for value in data.qpos):
        raise RuntimeError("MuJoCo produced an invalid simulation state.")
    print("MuJoCo model loading and stepping passed.")

    torch = modules["torch"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device auto-detection selected {device}.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device.")
    torch_device = "cuda:0" if device == "cuda" else "cpu"
    values = torch.tensor([1.0, 2.0], device=torch_device)
    if (values * 2).cpu().tolist() != [2.0, 4.0]:
        raise RuntimeError(f"PyTorch arithmetic failed on {torch_device}.")
    print(f"PyTorch arithmetic passed on {torch_device} (CUDA runtime: {torch.version.cuda}).")
    if device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # 同时检查 torchvision 的原生算子，捕获后端或二进制版本不匹配。
    torchvision = modules["torchvision"]
    for nms_device in dict.fromkeys(("cpu", torch_device)):
        boxes = torch.tensor(
            [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]], device=nms_device
        )
        scores = torch.tensor([0.9, 0.8, 0.7], device=nms_device)
        if torchvision.ops.nms(boxes, scores, 0.5).cpu().tolist() != [0, 2]:
            raise RuntimeError(f"Torchvision NMS failed on {nms_device}.")
        print(f"Torchvision NMS passed on {nms_device}.")

    warp = modules["warp"]
    warp_device = warp.get_device(torch_device)
    if device == "cuda" and not warp_device.is_cuda:
        raise RuntimeError("Warp did not select a CUDA device.")
    warp_values = warp.array([1.0, 2.0], dtype=warp.float32, device=warp_device)
    if warp_values.numpy().tolist() != [1.0, 2.0]:
        raise RuntimeError(f"Warp memory transfer failed on {warp_device}.")
    print(f"Warp device allocation and memory transfer passed on {warp_device}.")
    print("Installation check passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the project installation without starting training.")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto", help="Compute device to verify (default: auto)."
    )
    args = parser.parse_args()
    try:
        check_install(args.device)
    except Exception as exc:
        print(f"Installation check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
