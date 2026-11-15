# Copyright (c) 2026-2027 zh
"""内容：装配 Go1 Depth 环境与通用算法，保存配置和源码状态，按明确预算启动训练。"""

from copy import deepcopy
from dataclasses import asdict, fields
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import torch
from mjlab.utils.torch import configure_torch_backends

from agents.depth_supermimic.config import DepthExecutionConfig, DepthSuperMimicConfig
from src.config.go1_depth.go1_depth_config import Go1DepthCfg, Go1DepthTrainConfig
from src.runtime import configure_cuda_libdevice


def _override_config(config, values, prefix=""):
    """仅在 JSON 外部边界接受已有字段，拼写错误直接报错。"""
    for name, value in values.items():
        if name.startswith("_") or not hasattr(config, name) or callable(getattr(config, name)):
            raise ValueError(f"Unknown configuration field: {prefix}{name}")
        current = getattr(config, name)
        if hasattr(current, "__dict__") and isinstance(value, dict):
            _override_config(current, value, f"{prefix}{name}.")
        else:
            setattr(config, name, value)


def resolve_config(task_id: str, args: Go1DepthTrainConfig, cfg=None, *, training=True):
    """恢复完整任务快照后再应用显式入口覆盖，轮次始终表示目标总轮数。"""
    from ..cli import TASK_MODES

    if task_id not in TASK_MODES:
        raise ValueError(f"Unknown Go1 Depth SuperMimic task: {task_id}")
    config = deepcopy(cfg) if cfg is not None else Go1DepthCfg()
    source = args.task_config
    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {args.resume}")
        source = source or args.resume.parent / "task_config.json"
    if source is not None:
        saved = json.loads(Path(source).read_text(encoding="utf-8"))
        values = saved.get("config", saved)
        complete = _task_snapshot(Go1DepthCfg()).keys() <= values.keys()
        version = values.get("sensors", {}).get("depth", {}).get("protocol_version", None if complete else 3)
        if version != 3:
            raise ValueError("Go1 depth requires a v3 two-frame GRU configuration; legacy inputs cannot be converted")
        _override_config(config, values)
    config.terrain.mode = TASK_MODES[task_id]
    for name in ("num_envs", "seed"):
        if (value := getattr(args, name)) is not None:
            setattr(config.env, name, value)
    for name in ("max_iterations", "save_interval", "log_interval", "compile", "cuda_graph", "imag_microbatch"):
        if (value := getattr(args, name)) is not None:
            setattr(config.runner, name, value)
    minimum_envs = 2 if training else 1
    if config.env.num_envs < minimum_envs or config.env.seed < 0:
        raise ValueError(f"Depth task requires at least {minimum_envs} environments and a nonnegative seed")
    if config.runner.max_iterations < 1 or config.runner.save_interval < 1 or config.runner.log_interval < 1:
        raise ValueError("Training target, checkpoint interval and logging interval must be positive")
    algorithm = DepthSuperMimicConfig(**{item.name: getattr(config.algo, item.name) for item in fields(DepthSuperMimicConfig)})
    execution = DepthExecutionConfig(**{item.name: getattr(config.runner, item.name) for item in fields(DepthExecutionConfig)})
    return config, algorithm, execution


def _task_snapshot(cfg):
    """递归序列化配置字段，排除各层配置对象的方法。"""
    values = {name: getattr(cfg, name) for name in dir(cfg) if not name.startswith("_")}
    return {name: _task_snapshot(value) if hasattr(value, "__dict__") else value
            for name, value in values.items() if not callable(value)}


def _code_state(root):
    """本地读取 Git；远程源码包使用同步时保存的实际代码状态。"""
    if (root / ".git").exists():
        commands = {"revision": ["git", "rev-parse", "HEAD"],
                    "status": ["git", "status", "--short", "--untracked-files=all"],
                    "diff": ["git", "diff", "HEAD", "--"]}
        return {name: subprocess.check_output(command, cwd=root, text=True) for name, command in commands.items()}
    return json.loads((root / "source_state.json").read_text(encoding="utf-8"))


def make_runner(cfg, *, device, log_dir=None, execution=None):
    """仿真成功后交给算法；算法构造失败时及时关闭已分配环境。"""
    from agents.depth_supermimic.runner import DepthSuperMimicRunner
    from .. import make_go1_depth_env

    config = DepthSuperMimicConfig(**{item.name: getattr(cfg.algo, item.name) for item in fields(DepthSuperMimicConfig)})
    if execution is None:
        execution = DepthExecutionConfig(**{item.name: getattr(cfg.runner, item.name) for item in fields(DepthExecutionConfig)})
    torch.manual_seed(cfg.env.seed)
    env = make_go1_depth_env(cfg, device=device)
    try:
        return DepthSuperMimicRunner(env, config, log_dir=log_dir, execution=execution,
                                    collection_groups=env.terrain_curriculum.family_ids)
    except BaseException:
        env.close()
        raise


def launch_training(task_id: str, args: Go1DepthTrainConfig, *, cfg=None) -> Path:
    """真实训练循环由通用 runner 完成；本层只负责装配与运行证据。"""
    config, algorithm, execution = resolve_config(task_id, args, cfg)
    if args.run_name and re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name) is None:
        raise ValueError("Run name must contain only letters, digits, underscores, dots or hyphens")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA training requested but CUDA is unavailable")
    execution.cuda_graph = execution.cuda_graph and device.type == "cuda"
    for field in ("actor_bf16", "encoder_bf16", "wm_bf16", "decoder_bf16", "critic_bf16"):
        setattr(execution, field, getattr(execution, field) and device.type == "cuda")
    configure_cuda_libdevice()
    configure_torch_backends()
    os.environ.setdefault("MUJOCO_GL", "egl")
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f") + (f"_{args.run_name}" if args.run_name else "")
    directory = Path(args.log_root).expanduser().resolve() / "go1_depth" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[5]
    snapshot = {"task_id": task_id, "config": _task_snapshot(config), "algorithm": asdict(algorithm),
                "execution": asdict(execution), "reference": "InstinctMJ@03563ee02665f327425106372264070315673748"}
    (directory / "task_config.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (directory / "run_source.json").write_text(json.dumps({"command": sys.argv, "torch": torch.__version__,
        "arguments": asdict(args), "code": _code_state(root)}, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[INFO] Logging Go1 Depth training in {directory}", flush=True)
    print(f"[INFO] Device={device}, seed={config.env.seed}, environments={config.env.num_envs}, "
          f"target_iterations={config.runner.max_iterations}, save_interval={config.runner.save_interval}", flush=True)
    runner = make_runner(config, device=str(device), log_dir=directory, execution=execution)
    started = time.perf_counter()
    try:
        if args.resume is not None:
            runner.load(args.resume)
        if runner.iteration >= config.runner.max_iterations:
            raise ValueError("The training target must exceed the restored checkpoint iteration")
        previous_training_seconds = runner.training_seconds
        runner.learn(config.runner.max_iterations, save_interval=config.runner.save_interval,
                     log_interval=config.runner.log_interval)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result = {"status": "completed", "iteration": runner.iteration,
                  "elapsed_seconds": runner.training_seconds - previous_training_seconds,
                  "learn_wall_seconds": time.perf_counter() - started, "nonfinite_steps": runner.nonfinite_steps,
                  "updates": {name: int(getattr(runner.learner, name)) for name in
                              ("model_updates", "actor_updates", "critic_updates", "failure_updates", "actor_skipped")},
                  "execution": asdict(runner.execution),
                  "update_graphs": {name: {"captured": item.graph is not None, "calls": item.calls, "replays": item.replays}
                                    for name, item in runner.learner.graphs.items()},
                  "sampling_graphs": {name: {"captured": item.graph is not None, "calls": item.calls, "replays": item.replays}
                                      for name, item in (("model", runner._model_sampling), ("seed", runner._seed_sampling))
                                      if item is not None},
                  "num_envs": config.env.num_envs, "seed": config.env.seed,
                  "checkpoint": str(directory / f"model_{runner.iteration}.pt")}
        (directory / "training_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    finally:
        runner.close()
    print(f"[INFO] Go1 Depth training completed at iteration {runner.iteration}", flush=True)
    return directory
