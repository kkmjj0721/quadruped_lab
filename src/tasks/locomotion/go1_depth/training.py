# Copyright (c) 2026-2027 zh
"""内容：装配 Go1 视觉训练、独立配置快照、参数续训与周期保存。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime
import json
import os
from pathlib import Path
import time

import torch

from agents.depth_supermimic.config import DepthSuperMimicConfig
from agents.depth_supermimic.runner import DepthSuperMimicRunner
from agents.depth_supermimic.execution_config import ExecutionConfig
from src.config.go1_depth.go1_depth_config import Go1DepthCfg
from src.runtime import configure_cuda_libdevice
from .config_loading import apply_settings, read_settings
from .world_model import Go1WorldModel


TASK_MODES = {"Unitree-Go1-Depth-SuperMimic-Rough": "rough", "Unitree-Go1-Depth-SuperMimic-Flat": "flat"}


@dataclass(frozen=True)
class DepthTrainArguments:
    """任务和网络默认值来自 Go1 配置，JSON 支持明确覆盖全部已声明字段。"""

    device: str = "cuda:0"
    num_envs: int | None = None
    seed: int | None = None
    max_iterations: int | None = None
    save_interval: int | None = None
    log_interval: int | None = None
    metric_interval: int | None = None
    imag_microbatch: int | None = None
    critic_microbatch: int | None = None
    wm_compile: bool | None = None
    compile_imagination: bool | None = None
    wm_bf16: bool | None = None
    actor_bf16: bool | None = None
    critic_bf16: bool | None = None
    fused_adam: bool | None = None
    cuda_graph: bool | None = None
    task_config: Path | None = None
    resume: Path | None = None
    continue_run: bool = False
    model_only: bool | None = None
    log_root: str = "logs/supermimic"
    experiment_name: str = "go1_depth_cnn_gru"


def resolve_config(task_id, args):
    if task_id not in TASK_MODES:
        raise ValueError(f"Unknown Go1 depth task: {task_id}")
    cfg = Go1DepthCfg()
    source = args.task_config
    if source is None and args.resume is not None:
        source = args.resume.parent / "task_config.json"
    if source is not None:
        settings, complete = read_settings(source)
        apply_settings(cfg, settings, replace_mappings=complete)
    cfg.terrain.mode = TASK_MODES[task_id]
    for name in ("num_envs", "seed"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg.env, name, value)
    for name in ("max_iterations", "save_interval", "log_interval", "metric_interval", "imag_microbatch", "critic_microbatch",
                 "wm_compile", "wm_bf16", "actor_bf16", "critic_bf16", "fused_adam", "cuda_graph"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg.runner, name, value)
    if args.compile_imagination is not None:
        cfg.algo.compile_imagination = args.compile_imagination
    if args.model_only is not None:
        cfg.runner.enable_control_updates = not args.model_only
    return cfg


def make_runner(cfg, *, device="cuda:0", log_dir=None, control_enabled=None, metric_interval=None):
    from .environment import make_go1_depth_env

    algorithm = DepthSuperMimicConfig(**{item.name: getattr(cfg.algo, item.name) for item in fields(DepthSuperMimicConfig)})
    execution = ExecutionConfig(**{item.name: getattr(cfg.runner, item.name) for item in fields(ExecutionConfig)})
    if execution.cuda_graph and (torch.device(device).type != "cuda" or not execution.fused_adam):
        raise ValueError("Go1 depth CUDA Graph updates require a CUDA device and fused Adam")
    bf16 = execution.wm_bf16 or execution.actor_bf16 or execution.critic_bf16
    if bf16 and (torch.device(device).type != "cuda" or not torch.cuda.is_bf16_supported(including_emulation=False)):
        raise ValueError("Depth MLP BF16 requires a CUDA device with native BF16 support")
    torch.manual_seed(cfg.env.seed)
    configure_cuda_libdevice()
    env = make_go1_depth_env(cfg, device=device)
    try:
        runner = DepthSuperMimicRunner(env, algorithm, Go1WorldModel, log_dir=log_dir, execution=execution,
                                       control_enabled=cfg.runner.enable_control_updates if control_enabled is None else control_enabled,
                                       metric_interval=cfg.runner.metric_interval if metric_interval is None else metric_interval)
        runner.logger.log_interval = cfg.runner.log_interval
        return runner
    except BaseException:
        env.close()
        raise


def launch_training(task_id, args):
    if args.continue_run and args.resume is None:
        raise ValueError("continue_run requires a resume checkpoint")
    if torch.device(args.device).type != "cuda":
        raise ValueError("Go1 depth rendering requires a CUDA device")
    if not args.experiment_name or Path(args.experiment_name).name != args.experiment_name:
        raise ValueError("experiment_name must be a nonempty directory component")
    cfg = resolve_config(task_id, args)
    for name in ("save_interval", "log_interval", "metric_interval"):
        value = getattr(cfg.runner, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    target = cfg.runner.max_iterations
    if isinstance(target, bool) or not isinstance(target, int) or target < 1:
        raise ValueError("max_iterations must be a positive integer")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Go1 depth training currently requires one process and one CUDA device")
    destination = Path(args.log_root).expanduser().resolve() / args.experiment_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    destination.mkdir(parents=True, exist_ok=False)
    snapshot = {"config": cfg.to_dict(), "task_id": task_id}
    (destination / "task_config.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    runner = make_runner(cfg, device=args.device, log_dir=destination)
    training_started = None
    completed = False
    initial_iteration = 0
    initial_steps = 0
    try:
        if args.resume is not None:
            runner.load(args.resume, continue_run=args.continue_run)
        initial_iteration, initial_steps = runner.iteration, runner.real_steps
        if target <= runner.iteration:
            raise ValueError("Training target must exceed the restored iteration")
        runner.target_iteration = target
        (destination / "memory_budget.json").write_text(json.dumps(runner.memory_budget(), indent=2) + "\n", encoding="utf-8")
        (destination / "algorithm_config.json").write_text(json.dumps(asdict(runner.config), indent=2) + "\n", encoding="utf-8")
        print(f"[INFO] Go1 depth training initialized: output={destination}, device={args.device}", flush=True)
        training_started = time.perf_counter()
        while runner.iteration < target:
            block = min(target - runner.iteration, cfg.runner.save_interval - runner.iteration % cfg.runner.save_interval)
            runner.learn(block)
            runner.save(destination / f"model_{runner.iteration}.pt")
            print(f"[INFO] Saved depth checkpoint: iteration={runner.iteration}", flush=True)
        completed = True
    finally:
        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(args.device)
        summary = {"success": completed, "training_seconds": runner.elapsed,
                   "training_block_seconds": time.perf_counter() - training_started if training_started else 0.,
                   "start_iteration": initial_iteration, "final_iteration": runner.iteration,
                   "new_real_steps": runner.real_steps - initial_steps, "imagined_steps": runner.imagined_steps,
                   "seed": cfg.env.seed, "task": task_id, "horizon": runner.config.horizon,
                   "model_window": runner.config.wm_window}
        (destination / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        try:
            runner.close()
        finally:
            runner.env.close()
    return destination
