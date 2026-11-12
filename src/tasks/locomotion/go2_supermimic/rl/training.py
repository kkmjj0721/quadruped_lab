# Copyright (c) 2026-2027 zh
"""
内容：
    装配Go2 SuperMimic训练、参数续训、独立产物目录和周期检查点保存。
"""

from __future__ import annotations

from dataclasses import asdict, fields
from datetime import datetime
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import torch
import yaml
from mjlab.utils.torch import configure_torch_backends

from agents.supermimic.config import ExecutionConfig, RUNNER_RUNTIME_FIELDS
from agents.supermimic.timing import TrainingTiming
from src.runtime import configure_cuda_libdevice

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
    for name in ("max_iterations", "save_interval", "log_interval"):
        value = getattr(args, name)
        if value is not None:
            _integer(value, 1, name)
    ExecutionConfig(**{field.name: getattr(args, field.name) for field in fields(ExecutionConfig)
                       if getattr(args, field.name) is not None})
    if args.num_envs is not None:
        _integer(args.num_envs, 2, "num_envs")
    if args.seed is not None:
        _integer(args.seed, 0, "seed")
    if args.model_only is not None and not isinstance(args.model_only, bool):
        raise ValueError("model_only must be a boolean or None")
    if not all(isinstance(getattr(args, name), bool) for name in ("continue_run", "reset_critic")):
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
    for name in RUNNER_RUNTIME_FIELDS:
        value = getattr(args, name)
        if value is not None:
            setattr(cfg.runner, name, value)
    for name in ("max_iterations", "save_interval", "log_interval"):
        _integer(getattr(cfg.runner, name), 1, name)
    ExecutionConfig(**{field.name: getattr(cfg.runner, field.name) for field in fields(ExecutionConfig)})
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
            "triton_libdevice_path": os.environ.get("TRITON_LIBDEVICE_PATH"),
            "packages": {name: version(name) for name in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "tensorboard")},
            "workspace": workspace}


def launch_training(task_id: str, args: SuperMimicTrainConfig) -> Path:
    """调用既有runner训练至目标轮次，周期保存模型和ONNX并返回独立产物目录。"""
    entry_started = time.perf_counter()
    _validate_arguments(task_id, args)
    cfg, resume, source = _resolve_task_config(task_id, args)
    runtime = cfg.runner
    device = _resolve_device(args.device)
    if torch.device(device).type == "cuda":
        configure_cuda_libdevice()
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
    execution = ExecutionConfig(**{field.name: getattr(runtime, field.name) for field in fields(ExecutionConfig)})
    timing = TrainingTiming(device, enabled=execution.detailed_timing, interval=execution.timing_interval)
    runner = None
    initialized = training_completed = False
    starting_steps = starting_iteration = starting_prefill_steps = 0
    try:
        runner = make_go2_supermimic_runner(cfg, device=device, log_dir=log_dir, execution=execution)
        timing, execution = runner.timing, runner.execution
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        timing.record_wall("initialization", time.perf_counter() - entry_started)
        initialized = True
        print(f"[INFO] SuperMimic {cfg.algorithm} training: envs={cfg.env.num_envs}, "
              f"collect_avg={cfg.algo.collect_steps:g}, utd={1.0 / cfg.algo.collect_steps:g}, "
              f"wm_batch={cfg.algo.wm_batch}, imag_batch={cfg.algo.imag_batch}, "
              f"replay={cfg.algo.replay_capacity}, compile={cfg.algo.compile_imagination}, "
              f"imag_microbatch={execution.imag_microbatch}, critic_microbatch={execution.critic_microbatch}, "
              f"wm_compile={execution.wm_compile}, fused_adam={execution.fused_adam}, "
              f"cuda_graph={execution.cuda_graph}, detailed_timing={execution.detailed_timing}, "
              f"wm_bf16={execution.wm_bf16}, actor_bf16={execution.actor_bf16}, critic_bf16={execution.critic_bf16}, "
              f"timing_interval={execution.timing_interval}, metric_interval={execution.metric_interval}", flush=True)
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
            restore_started = time.perf_counter()
            try:
                runner.load(resume, continue_run=args.continue_run, reset_critic=args.reset_critic)
            finally:
                timing.record_wall("restore", time.perf_counter() - restore_started)
        starting_steps, starting_iteration = runner.real_steps, runner.iteration
        starting_prefill_steps = runner.prefill_real_steps
        target_iteration = runtime.max_iterations
        if target_iteration <= runner.iteration:
            raise ValueError(f"Target iteration {target_iteration} must exceed checkpoint iteration {runner.iteration}")
        runner.configure_logging(target_iteration=target_iteration, log_interval=runtime.log_interval)
        while runner.iteration < target_iteration:
            remaining = target_iteration - runner.iteration
            until_checkpoint = runtime.save_interval - runner.iteration % runtime.save_interval
            learn_started = time.perf_counter()
            try:
                runner.learn(min(remaining, until_checkpoint))
            finally:
                timing.record_wall("learn_blocks", time.perf_counter() - learn_started)
            # 参考在塌缩护栏首次预警时跳过该保存点，不发布可疑策略及其ONNX。
            if runner._collapse_strikes == 0:
                checkpoint = log_dir / f"model_{runner.iteration}.pt"
                runner.save(checkpoint)
                print(f"[INFO] Saved checkpoint: {checkpoint}", flush=True)
        training_completed = True
    finally:
        primary_error = sys.exception()
        cleanup_error = None
        if not initialized:
            timing.record_wall("initialization", time.perf_counter() - entry_started)
        if runner is not None:
            # 分别关闭两项资源；已有训练异常优先传播，清理异常作为上下文保留。
            for name, close in (("logger_close", runner.close), ("environment_close", runner.env.close)):
                close_started = time.perf_counter()
                try:
                    close()
                except BaseException as error:
                    if primary_error is not None:
                        primary_error.add_note(f"Cleanup failed in {name}: {error!r}")
                    elif cleanup_error is None:
                        cleanup_error = error
                    else:
                        cleanup_error.add_note(f"Cleanup failed in {name}: {error!r}")
                finally:
                    timing.record_wall(name, time.perf_counter() - close_started)
        try:
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
        except BaseException as error:
            if primary_error is not None:
                primary_error.add_note(f"Final CUDA synchronization failed: {error!r}")
            elif cleanup_error is None:
                cleanup_error = error
            else:
                cleanup_error.add_note(f"Final CUDA synchronization failed: {error!r}")
        entry_seconds = time.perf_counter() - entry_started
        timing.record_wall("entry_total", entry_seconds)
        error = primary_error if primary_error is not None else cleanup_error
        summary = timing.summary()
        added_steps = runner.real_steps - starting_steps if runner is not None else 0
        prefill_steps = runner.prefill_real_steps - starting_prefill_steps if runner is not None else 0
        training_steps = added_steps - prefill_steps
        summary["run"] = {
            "task_id": task_id, "seed": cfg.env.seed, "num_envs": cfg.env.num_envs,
            "initialized": initialized, "training_completed": training_completed,
            "success": training_completed and error is None,
            "error": {"type": type(error).__name__, "message": str(error)} if error is not None else None,
            "start_iteration": starting_iteration, "target_iteration": runtime.max_iterations,
            "final_iteration": runner.iteration if runner is not None else None,
            "new_real_steps_including_prefill": added_steps,
            "new_prefill_real_steps": prefill_steps, "new_training_real_steps": training_steps,
            "entry_real_transitions_per_second": added_steps / entry_seconds,
            "entry_training_transitions_per_second": training_steps / entry_seconds,
            "runner_internal_seconds": runner.elapsed if runner is not None else None,
        }
        summary["scope"].update({
            "entry_total": "launch_training entry through logger/environment close and final target-device synchronization; excludes this summary write",
            "learn_blocks": "Inclusive host wall of runner.learn calls; includes prefill, setup and per-iteration logging, excludes outer checkpoint save",
            "save_total": "Inclusive parent of checkpoint_prepare, onnx_export, checkpoint_publish and temporary-directory cleanup; do not sum parent and children",
            "process_wall": "Not measured here; the explicit training test launcher measures child-process wall externally",
            "transitions": "New steps subtract restored counters; entry_real_transitions_per_second includes prefill, entry_training_transitions_per_second excludes it",
        })
        try:
            (log_dir / "timing_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        except (OSError, TypeError, ValueError) as summary_error:
            if error is None:
                raise
            error.add_note(f"Timing summary write failed: {summary_error!r}")
        if primary_error is None and cleanup_error is not None:
            raise cleanup_error
    return log_dir
