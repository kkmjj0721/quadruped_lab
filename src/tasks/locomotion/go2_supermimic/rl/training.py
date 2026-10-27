# Copyright (c) 2026-2027 zh
"""
内容：
    装配Go2 SuperMimic训练、参数续训、独立产物目录和周期检查点保存。
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import torch
from mjlab.utils.torch import configure_torch_backends

from agents.supermimic.single.config import SingleExecutionConfig

from ..config.loading import (
    apply_settings as _apply_settings, read_settings as _read_settings,
    configuration_for_settings,
)
from ..config.train_cfg import SuperMimicTrainConfig, TASK_MODES
from ..go_supermimic_env_cfg import validate_config
from .runner import make_go2_supermimic_runner
from .single_runner import algorithm_config


# ---------- 参数与任务配置 ----------

def _integer(value, minimum: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _validate_arguments(task_id: str, args: SuperMimicTrainConfig) -> None:
    if task_id not in TASK_MODES:
        raise ValueError(f"Unknown Go2 SuperMimic task: {task_id}")
    _integer(args.max_iterations, 1, "max_iterations")
    _integer(args.save_interval, 1, "save_interval")
    _integer(args.log_interval, 1, "log_interval")
    SingleExecutionConfig(imag_microbatch=args.imag_microbatch, critic_microbatch=args.critic_microbatch)
    if args.num_envs is not None:
        _integer(args.num_envs, 2, "num_envs")
    if args.seed is not None:
        _integer(args.seed, 0, "seed")
    if args.model_only is not None and not isinstance(args.model_only, bool):
        raise ValueError("model_only must be a boolean or None")
    if not isinstance(args.device, str) or re.fullmatch(r"cpu|cuda(?::\d+)?", args.device) is None:
        raise ValueError("SuperMimic device must be cpu or one CUDA device such as cuda:0")
    if not isinstance(args.log_root, (str, Path)) or not str(args.log_root).strip():
        raise ValueError("log_root must be a nonempty path")
    for name, value in (("experiment_name", args.experiment_name), ("run_name", args.run_name)):
        if not isinstance(value, str) or value in (".", "..") or "/" in value or "\\" in value:
            raise ValueError(f"{name} must be a single directory component")
    if not args.experiment_name.strip():
        raise ValueError("experiment_name must not be empty")
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as error:
        raise ValueError("WORLD_SIZE must be 1 for SuperMimic training") from error
    if world_size != 1:
        raise ValueError("Distributed SuperMimic training is not supported; WORLD_SIZE must be 1")


def _resolve_task_config(task_id: str, args: SuperMimicTrainConfig):
    resume = Path(args.resume).expanduser().resolve() if args.resume is not None else None
    if resume is not None and not resume.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {resume}")
    source = Path(args.task_config).expanduser().resolve() if args.task_config is not None else None
    if source is None and resume is not None:
        source = resume.parent / "task_config.json"
        if not source.is_file():
            raise FileNotFoundError(f"Resume requires {source} or an explicit --task-config")
    settings, snapshot = _read_settings(source) if source is not None else ({}, False)
    cfg = configuration_for_settings(settings)
    if source is not None:
        sections = set(cfg.to_dict()) - {"init_member_classes", "to_dict"}
        # 完整快照可能主动删去地形家族等字典键，恢复时不得补回默认内容。
        complete = snapshot or sections.issubset(settings)
        _apply_settings(cfg, settings, replace_mappings=complete)
    if resume is not None and cfg.terrain.mode != TASK_MODES[task_id]:
        raise ValueError("Resume task terrain mode differs from the saved configuration")
    cfg.terrain.mode = TASK_MODES[task_id]
    for field, value in (("num_envs", args.num_envs), ("seed", args.seed)):
        if value is not None:
            if resume is not None and value != getattr(cfg.env, field):
                raise ValueError(f"Cannot change {field} when resuming a task checkpoint")
            setattr(cfg.env, field, value)
    if args.model_only is not None or resume is None:
        enabled = not bool(args.model_only)
        if resume is not None and enabled != cfg.runner.enable_control_updates:
            raise ValueError("Cannot change model_only when resuming a task checkpoint")
        cfg.runner.enable_control_updates = enabled
    _integer(cfg.env.num_envs, 2, "Task environment count")
    _integer(cfg.env.seed, 0, "Task seed")
    validate_config(cfg)
    algorithm_config(cfg)
    return cfg, resume, source


def _resolve_device(requested: str) -> str:
    if requested == "cpu":
        return requested
    device = torch.device(requested)
    index = device.index if device.index is not None else 0
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise ValueError(f"Requested CUDA device is unavailable: cuda:{index}; use --device cpu for CPU")
    return f"cuda:{index}"


# ---------- 运行快照与训练装配 ----------

def _write_run_config(log_dir: Path, task_id: str, args: SuperMimicTrainConfig,
                      device: str, resume: Path | None, source: Path | None) -> None:
    root = Path(__file__).resolve().parents[5]
    workspace = {}
    for name, command in (("revision", ["git", "rev-parse", "HEAD"]), ("status", ["git", "status", "--short"])):
        try:
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=10, check=False)
            workspace[name] = result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            workspace[name] = None
    snapshot = {
        "task_id": task_id, "arguments": asdict(args), "device": device, "resume_from": resume,
        "task_config_source": source, "command": list(sys.argv), "python": sys.version,
        "torch": torch.__version__, "workspace": workspace,
    }
    (log_dir / "run_config.json").write_text(json.dumps(snapshot, default=str, indent=2) + "\n", encoding="utf-8")
    # 大量工作区源码尚未跟踪，单独保存实际运行源码，避免Git版本号无法重建实验。
    sources = [path for folder in ("agents/supermimic", "src", "scripts") for path in (root / folder).rglob("*.py")]
    sources += list((root / "src/assets").rglob("*.xml")) + [root / "pyproject.toml"]
    manifest = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    (log_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(log_dir / "sources.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, arcname=str(path.relative_to(root)))


def launch_training(task_id: str, args: SuperMimicTrainConfig) -> Path:
    """调用既有runner追加训练轮数，周期保存并返回本次独立产物目录。"""
    _validate_arguments(task_id, args)
    cfg, resume, source = _resolve_task_config(task_id, args)
    device = _resolve_device(args.device)
    log_root = (Path(args.log_root).expanduser() / args.experiment_name).resolve()
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    if args.run_name:
        name += f"_{args.run_name}"
    log_dir = log_root / name
    log_dir.mkdir(parents=True, exist_ok=False)
    _write_run_config(log_dir, task_id, args, device, resume, source)
    configure_torch_backends()
    print(f"[INFO] Logging experiment in directory: {log_dir}", flush=True)
    print(f"[INFO] Training with: task={task_id}, device={device}, seed={cfg.env.seed}", flush=True)
    print(f"[INFO] Control updates enabled: {cfg.runner.enable_control_updates}", flush=True)
    execution = SingleExecutionConfig(imag_microbatch=args.imag_microbatch, critic_microbatch=args.critic_microbatch)
    print(f"[INFO] SuperMimic {cfg.algorithm} training: envs={cfg.env.num_envs}, "
          f"wm_batch={cfg.single.wm_batch}, imag_batch={cfg.single.imag_batch}, "
          f"replay={cfg.single.replay_capacity}, compile={cfg.single.compile_imagination}, "
          f"imag_microbatch={execution.imag_microbatch}, critic_microbatch={execution.critic_microbatch}", flush=True)
    runner = make_go2_supermimic_runner(cfg, device=device, log_dir=log_dir, execution=execution)
    try:
        if resume is not None:
            print(f"[INFO] Loading model checkpoint from: {resume}", flush=True)
            runner.load(resume)
        target_iteration = runner.iteration + args.max_iterations
        runner.configure_logging(target_iteration=target_iteration, log_interval=args.log_interval)
        while runner.iteration < target_iteration:
            remaining = target_iteration - runner.iteration
            until_checkpoint = args.save_interval - runner.iteration % args.save_interval
            runner.learn(min(remaining, until_checkpoint))
            checkpoint = log_dir / f"model_{runner.iteration}.pt"
            runner.save(checkpoint)
            print(f"[INFO] Saved checkpoint: {checkpoint}", flush=True)
    finally:
        runner.env.close()
    return log_dir
