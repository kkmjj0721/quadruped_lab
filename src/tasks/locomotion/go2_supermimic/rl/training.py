# Copyright (c) 2026-2027 zh
"""
内容：
    装配Go2 SuperMimic训练、参数续训、独立产物目录和周期检查点保存。
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import torch
import yaml
from mjlab.utils.torch import configure_torch_backends

from agents.supermimic.config import ExecutionConfig

from ..config.loading import (
    apply_settings as _apply_settings, read_settings as _read_settings,
    configuration_for_settings,
)
from ..config.train_cfg import SuperMimicTrainConfig, TASK_MODES
from ..go_supermimic_env_cfg import validate_config
from .runner import algorithm_config, make_go2_supermimic_runner


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
    ExecutionConfig(imag_microbatch=args.imag_microbatch, critic_microbatch=args.critic_microbatch)
    if args.num_envs is not None:
        _integer(args.num_envs, 2, "num_envs")
    if args.seed is not None:
        _integer(args.seed, 0, "seed")
    if args.model_only is not None and not isinstance(args.model_only, bool):
        raise ValueError("model_only must be a boolean or None")
    if not isinstance(args.continue_run, bool) or not isinstance(args.reset_critic, bool):
        raise ValueError("continue_run and reset_critic must be booleans")
    if args.resume is None and (args.continue_run or args.reset_critic):
        raise ValueError("continue_run and reset_critic require a resume checkpoint")
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
    cfg.terrain.mode = TASK_MODES[task_id]
    for field, value in (("num_envs", args.num_envs), ("seed", args.seed)):
        if value is not None:
            setattr(cfg.env, field, value)
    if args.model_only is not None or resume is None:
        enabled = not bool(args.model_only)
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

def _run_origin(task_id: str, args: SuperMimicTrainConfig, device: str, resume: Path | None, source: Path | None) -> dict:
    """收集命令、依赖版本与代码状态，供事件文本和params快照共同使用。"""
    root = Path(__file__).resolve().parents[5]
    workspace = {}
    for name, command in (("revision", ["git", "rev-parse", "HEAD"]),
                          ("status", ["git", "status", "--short"]),
                          ("diff", ["git", "diff", "HEAD", "--"])):
        try:
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=10, check=False)
            workspace[name] = (result.stdout if name == "diff" else result.stdout.strip()) if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            workspace[name] = None
    return {"task_id": task_id, "started_at": datetime.now().astimezone().isoformat(),
            "arguments": asdict(args), "device": device, "resume_from": resume,
            "task_config_source": source, "command": list(sys.argv), "python": sys.version,
            "torch": torch.__version__,
            "packages": {name: version(name) for name in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "tensorboard")},
            "workspace": workspace}


def launch_training(task_id: str, args: SuperMimicTrainConfig) -> Path:
    """调用既有runner训练至目标轮次，周期保存模型和ONNX并返回独立产物目录。"""
    _validate_arguments(task_id, args)
    cfg, resume, source = _resolve_task_config(task_id, args)
    device = _resolve_device(args.device)
    log_root = (Path(args.log_root).expanduser() / args.experiment_name).resolve()
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    if args.run_name:
        name += f"_{args.run_name}"
    log_dir = log_root / name
    log_dir.mkdir(parents=True, exist_ok=False)
    configure_torch_backends()
    print(f"[INFO] Logging experiment in directory: {log_dir}", flush=True)
    print(f"[INFO] Training with: task={task_id}, device={device}, seed={cfg.env.seed}", flush=True)
    print(f"[INFO] Control updates enabled: {cfg.runner.enable_control_updates}", flush=True)
    execution = ExecutionConfig(imag_microbatch=args.imag_microbatch, critic_microbatch=args.critic_microbatch)
    print(f"[INFO] SuperMimic {cfg.algorithm} training: envs={cfg.env.num_envs}, "
          f"collect_avg={cfg.algo.collect_steps:g}, utd={1.0 / cfg.algo.collect_steps:g}, "
          f"wm_batch={cfg.algo.wm_batch}, imag_batch={cfg.algo.imag_batch}, "
          f"replay={cfg.algo.replay_capacity}, compile={cfg.algo.compile_imagination}, "
          f"imag_microbatch={execution.imag_microbatch}, critic_microbatch={execution.critic_microbatch}", flush=True)
    runner = make_go2_supermimic_runner(cfg, device=device, log_dir=log_dir, execution=execution)
    try:
        # 快照反映JSON与CLI覆盖后的实际配置，普通YAML类型便于直接查看和安全回读。
        origin = json.loads(json.dumps(_run_origin(task_id, args, device, resume, source), default=str))
        origin.update(seed=cfg.env.seed, num_envs=cfg.env.num_envs)
        params = log_dir / "params"
        params.mkdir()
        diff = origin["workspace"].pop("diff")
        origin["workspace"]["diff_file"] = "params/git.diff" if diff is not None else None
        if diff is not None:
            (params / "git.diff").write_text(diff, encoding="utf-8")
        settings = cfg.to_dict()
        agent = {"algorithm": settings.pop("algorithm"), "config": asdict(algorithm_config(cfg)),
                 "execution": asdict(execution), "runner": settings.pop("runner")}
        for name in ("algo", "init_member_classes", "to_dict"):
            settings.pop(name)
        for filename, values in (("env.yaml", settings), ("agent.yaml", agent), ("run.yaml", origin)):
            (params / filename).write_text(yaml.safe_dump(values, sort_keys=False, allow_unicode=True), encoding="utf-8")
        runner.logger.add_text("run/origin", "```json\n" + json.dumps(
            origin, indent=2) + "\n```")
        if resume is not None:
            print(f"[INFO] Loading model checkpoint from: {resume}", flush=True)
            runner.load(resume, continue_run=args.continue_run, reset_critic=args.reset_critic)
        target_iteration = args.max_iterations
        if target_iteration <= runner.iteration:
            raise ValueError(f"Target iteration {target_iteration} must exceed checkpoint iteration {runner.iteration}")
        runner.configure_logging(target_iteration=target_iteration, log_interval=args.log_interval)
        while runner.iteration < target_iteration:
            remaining = target_iteration - runner.iteration
            until_checkpoint = args.save_interval - runner.iteration % args.save_interval
            runner.learn(min(remaining, until_checkpoint))
            # 参考在塌缩护栏首次预警时跳过该保存点，不发布可疑策略及其ONNX。
            if runner._collapse_strikes == 0:
                checkpoint = log_dir / f"model_{runner.iteration}.pt"
                runner.save(checkpoint)
                print(f"[INFO] Saved checkpoint: {checkpoint}", flush=True)
    finally:
        runner.close()
        runner.env.close()
    return log_dir
