# Copyright (c) 2026-2027 zh
"""
内容：
    统一训练入口：先解析任务ID，再按该任务的参数结构解析并启动训练。

用法：
    python scripts/train.py Unitree-Go2-Flat --env.scene.num-envs 64
    python scripts/train.py Unitree-Go2-SuperMimic-Flat --device cuda:0 --num-envs 4096
    python -m scripts.train Unitree-Go2-Flat --gpu-ids None --env.scene.num-envs 1
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
import logging
import os
from pathlib import Path
import re
import sys
from typing import Literal

# 直接执行脚本时也从当前项目导入任务。
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mjlab
import tyro
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

SUPERMIMIC_PREFIX = "Unitree-Go2-SuperMimic-"


@dataclass(frozen=True)
class TrainConfig:
    """训练配置；环境和算法字段可通过命令行逐项覆盖。"""

    env: ManagerBasedRlEnvCfg
    agent: RslRlBaseRunnerCfg
    gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
    """可见 GPU 的索引；None 表示 CPU。"""
    video: bool = False
    video_length: int = 200
    """每段录像的控制步数。"""
    video_interval: int = 2000
    """录像触发间隔，单位为控制步。"""
    enable_nan_guard: bool = False
    log_root: str = "logs/rsl_rl"
    torchrunx_log_dir: str | None = None

    @staticmethod
    def from_task(task_id: str) -> TrainConfig:
        """读取任务注册表中的独立配置副本。"""
        from src.tasks import register_tasks
        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

        register_tasks()
        return TrainConfig(env=load_env_cfg(task_id), agent=load_rl_cfg(task_id))


def load_runner_cls(task_id: str):
    """仅在PPO训练分支读取MjLab注册表，保留既有入口的调用接口。"""
    from mjlab.tasks.registry import load_runner_cls as load

    return load(task_id)


def _validate_config(cfg: TrainConfig) -> None:
    """在分配仿真资源前检查入口参数。"""
    if cfg.env.scene.num_envs <= 0:
        raise ValueError("Environment count must be positive.")
    if cfg.agent.max_iterations < 0:
        raise ValueError("Training iterations must be non-negative.")
    if cfg.agent.num_steps_per_env <= 0 or cfg.agent.save_interval <= 0:
        raise ValueError("Rollout length and checkpoint interval must be positive.")
    if cfg.agent.seed < 0:
        raise ValueError("Training seed must be non-negative.")
    if cfg.video and (cfg.video_length <= 0 or cfg.video_interval <= 0):
        raise ValueError("Video length and interval must be positive.")
    if isinstance(cfg.gpu_ids, list) and (not cfg.gpu_ids or any(index < 0 for index in cfg.gpu_ids)):
        raise ValueError("GPU indices must be a non-empty list of non-negative integers.")


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
    """在单个 worker 中创建环境、恢复检查点并启动 PPO 更新。"""
    from src.tasks import register_tasks

    register_tasks()
    _validate_config(cfg)
    cfg = deepcopy(cfg)
    log_dir = Path(log_dir)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    resume_path = None
    if cfg.agent.resume:
        resume_path = get_checkpoint_path(log_dir.parent, cfg.agent.load_run, cfg.agent.load_checkpoint)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    rank = int(os.environ.get("RANK", "0")) if cuda_visible else 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if cuda_visible else 0
    device = f"cuda:{local_rank}" if cuda_visible else "cpu"
    if cuda_visible:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    configure_torch_backends()
    cfg.agent.seed += rank
    cfg.env.seed = cfg.agent.seed
    if cfg.enable_nan_guard:
        cfg.env.sim.nan_guard.enabled = True
        print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}", flush=True)
    if rank == 0:
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Logging experiment in directory: {log_dir}", flush=True)

    print(f"[INFO] Training with: device={device}, seed={cfg.agent.seed}, rank={rank}", flush=True)
    print(f"[INFO] Creating simulation on {device}; the first run may compile Warp kernels...", flush=True)
    record_video = cfg.video and rank == 0
    env = ManagerBasedRlEnv(cfg=cfg.env, device=device, render_mode="rgb_array" if record_video else None)
    try:
        if record_video:
            env = VideoRecorder(
                env,
                video_folder=log_dir / "videos" / "train",
                step_trigger=lambda step: step % cfg.video_interval == 0,
                video_length=cfg.video_length,
                disable_logger=True,
            )
            print("[INFO] Recording videos during training.", flush=True)
        env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
        agent_cfg = asdict(cfg.agent)
        # 运行器会原地调整配置字典，因此先保存未被修改的配置。
        if rank == 0:
            dump_yaml(log_dir / "params" / "env.yaml", asdict(cfg.env))
            dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)
        runner = runner_cls(env, agent_cfg, str(log_dir), device)
        runner.add_git_repo_to_log(__file__)
        if resume_path is not None:
            print(f"[INFO] Loading model checkpoint from: {resume_path}", flush=True)
            runner.load(str(resume_path))
        runner.learn(num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True)
    finally:
        env.close()


def launch_training(task_id: str, args: TrainConfig | None = None) -> None:
    """为一次训练选择设备和独立目录，按 GPU 数量启动 worker。"""
    args = deepcopy(args) if args is not None else TrainConfig.from_task(task_id)
    _validate_config(args)
    log_root = (Path(args.log_root).expanduser() / args.agent.experiment_name).resolve()
    if args.agent.resume:
        # 先固定所有 worker 的续训来源，避免新目录影响其他 worker 的正则查找。
        resume_path = get_checkpoint_path(log_root, args.agent.load_run, args.agent.load_checkpoint)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
        args.agent.load_run = re.escape(resume_path.parent.name) + "$"
        args.agent.load_checkpoint = re.escape(resume_path.name) + "$"
    log_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    if args.agent.run_name:
        log_name += f"_{args.agent.run_name}"
    log_dir = log_root / log_name
    try:
        selected_gpus, num_gpus = select_gpus(args.gpu_ids)
    except IndexError as error:
        raise ValueError(
            "Requested GPU indices exceed the available devices; use --gpu-ids None for CPU."
        ) from error
    # 在启动日志或 worker 前保留唯一目录，时间戳冲突时防止覆盖已有实验。
    log_dir.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if selected_gpus is None else ",".join(map(str, selected_gpus))
    os.environ.setdefault("MUJOCO_GL", "egl")
    if num_gpus <= 1:
        run_train(task_id, args, log_dir)
        return

    import torchrunx

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    logging.basicConfig(level=logging.INFO)
    if "TORCHRUNX_LOG_DIR" not in os.environ:
        os.environ["TORCHRUNX_LOG_DIR"] = (
            args.torchrunx_log_dir if args.torchrunx_log_dir is not None else str(log_dir / "torchrunx")
        )
    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
        hostnames=["localhost"],
        workers_per_host=num_gpus,
        backend=None,
        copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*", "TMPDIR"),
    ).run(run_train, task_id, args, log_dir)


def supermimic_task(task_id: str | None) -> bool:
    """SuperMimic 任务使用自己的参数、装配与产物目录，不进入PPO注册表。"""
    if task_id is None or not task_id.startswith(SUPERMIMIC_PREFIX):
        return False
    from src.tasks.locomotion.go2_supermimic.config.train_cfg import TASK_MODES

    if task_id not in TASK_MODES:
        raise ValueError(f"Unknown Go2 SuperMimic task: {task_id}")
    return True


def main() -> None:
    """先解析任务 ID，再根据该任务配置解析命令行覆盖。"""
    # 提前读取任务ID，避免任何分支都导入PPO注册表。
    if supermimic_task(sys.argv[1] if len(sys.argv) > 1 else None):
        from src.tasks.locomotion.go2_supermimic.cli import train_main

        train_main(sys.argv[1], sys.argv[2:])
        return

    from src.tasks import register_tasks
    from mjlab.tasks.registry import list_tasks

    register_tasks()
    tasks = list_tasks()
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        from src.tasks.locomotion.go2_supermimic.config.train_cfg import TASK_MODES

        print(f"usage: {sys.argv[0]} <TASK> [OPTIONS]")
        print(f"Run '{sys.argv[0]} <TASK> --help' for task-specific options.")
        print("Available tasks:\n  " + "\n  ".join(sorted([*tasks, *TASK_MODES])))
        return
    task_id, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    args = tyro.cli(
        TrainConfig,
        args=remaining_args,
        default=TrainConfig.from_task(task_id),
        prog=f"{sys.argv[0]} {task_id}",
        config=mjlab.TYRO_FLAGS,
    )
    launch_training(task_id, args)


if __name__ == "__main__":
    main()
