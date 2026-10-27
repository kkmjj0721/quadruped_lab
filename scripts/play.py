# Copyright (c) 2026-2027 zh
"""
内容：
    按任务 ID 加载本地策略检查点，或使用零动作、随机动作查看仿真环境。

用法：
    python scripts/play.py Unitree-Go2-Flat --checkpoint-file logs/rsl_rl/go2_ppo/实验目录/model_100.pt
    python scripts/play.py Unitree-Go2-SuperMimic-Rough --checkpoint-file logs/supermimic/go2_supermimic/实验目录/model_100.pt
    python -m scripts.play Unitree-Go2-Flat --agent zero --num-envs 1 --viewer viser
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

# 直接执行脚本时也从当前项目导入任务，不依赖外部 PYTHONPATH。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mjlab
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
    """策略、设备、录像和查看器参数。"""

    agent: Literal["zero", "random", "trained"] = "trained"
    checkpoint_file: str | None = None
    num_envs: int | None = None
    device: str | None = None
    video: bool = False
    video_length: int = 200
    video_height: int | None = None
    video_width: int | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    no_terminations: bool = False
    """关闭所有终止条件，便于持续观察。"""


def load_runner_cls(task_id: str):
    """仅PPO分支读取注册表，保持SuperMimic任务独立。"""
    from mjlab.tasks.registry import load_runner_cls as load

    return load(task_id)


def _run_viewer(env, policy, cfg: PlayConfig, device: str) -> None:
    viewer = cfg.viewer
    if viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        viewer = "native" if has_display else "viser"
    print(f"[INFO] Playing with: device={device}, viewer={viewer}", flush=True)
    viewer_cls = NativeMujocoViewer if viewer == "native" else ViserPlayViewer
    viewer_cls(env, policy).run()


def _run_supermimic(task_id: str, cfg: PlayConfig, checkpoint: Path | None, device: str) -> None:
    """只装配任务侧播放接口、可选录像和查看器。"""
    from src.tasks.locomotion.go2_supermimic.rl.playback import SuperMimicVideoRecorder, make_playback

    record_video = checkpoint is not None and cfg.video
    if cfg.video and checkpoint is None:
        print("[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir).")
    env, policy = make_playback(
        task_id, checkpoint_file=checkpoint, agent=cfg.agent, num_envs=cfg.num_envs, device=device,
        render_mode="rgb_array" if record_video else None, video_height=cfg.video_height,
        video_width=cfg.video_width, no_terminations=cfg.no_terminations,
    )
    try:
        if record_video:
            env = SuperMimicVideoRecorder(env, video_folder=checkpoint.parent / "videos" / "play",
                                          step_trigger=lambda step: step == 0, video_length=cfg.video_length,
                                          disable_logger=True)
            print("[INFO] Recording videos during play.")
        if cfg.no_terminations:
            print("[INFO] Terminations disabled.")
        _run_viewer(env, policy, cfg, device)
    finally:
        env.close()


def run_play(task_id: str, cfg: PlayConfig) -> None:
    """装配播放环境和策略，并在查看器退出或失败后关闭环境。"""
    if cfg.agent not in ("zero", "random", "trained"):
        raise ValueError(f"Unsupported agent mode: {cfg.agent}")
    if cfg.viewer not in ("auto", "native", "viser"):
        raise ValueError(f"Unsupported viewer backend: {cfg.viewer}")
    for name in ("num_envs", "video_height", "video_width"):
        value = getattr(cfg, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive.")
    if cfg.video and cfg.video_length <= 0:
        raise ValueError("video_length must be positive when recording video.")

    trained = cfg.agent == "trained"
    checkpoint: Path | None = None
    if trained:
        if cfg.checkpoint_file is None:
            raise ValueError("Trained playback requires --checkpoint-file.")
        checkpoint = Path(cfg.checkpoint_file).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
        print(f"[INFO] Loading checkpoint: {checkpoint}", flush=True)

    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if task_id.startswith("Unitree-Go2-SuperMimic"):
        _run_supermimic(task_id, cfg, checkpoint, device)
        return

    from src.tasks import register_tasks
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    register_tasks()
    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    if cfg.no_terminations:
        env_cfg.terminations = {}
        print("[INFO] Terminations disabled.")
    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width

    record_video = trained and cfg.video
    if cfg.video and not trained:
        print("[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir).")

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array" if record_video else None)
    try:
        if record_video:
            assert checkpoint is not None
            env = VideoRecorder(
                env,
                video_folder=checkpoint.parent / "videos" / "play",
                step_trigger=lambda step: step == 0,
                video_length=cfg.video_length,
                disable_logger=True,
            )
            print("[INFO] Recording videos during play.")
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        if trained:
            runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
            runner = runner_cls(env, asdict(agent_cfg), device=device)
            runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
            policy = runner.get_inference_policy(device=device)
        else:
            action_shape = env.unwrapped.action_space.shape

            def policy(obs) -> torch.Tensor:
                """按实际动作空间生成零动作或均匀随机动作。"""
                del obs
                if cfg.agent == "zero":
                    return torch.zeros(action_shape, device=env.unwrapped.device)
                return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

        _run_viewer(env, policy, cfg, device)
    finally:
        env.close()


def main() -> None:
    """按任务协议分派，SuperMimic不经过PPO注册表。"""
    from src.tasks.locomotion.go2_supermimic.config.train_cfg import TASK_MODES

    if len(sys.argv) > 1 and sys.argv[1].startswith("Unitree-Go2-SuperMimic"):
        task_id = sys.argv[1]
        if task_id not in TASK_MODES:
            raise ValueError(f"Unknown Go2 SuperMimic task: {task_id}")
        cfg = tyro.cli(PlayConfig, args=sys.argv[2:], prog=f"{sys.argv[0]} {task_id}", config=mjlab.TYRO_FLAGS)
        run_play(task_id, cfg)
        return

    from src.tasks import register_tasks
    from mjlab.tasks.registry import list_tasks

    register_tasks()
    tasks = list_tasks()
    if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
        print(f"usage: {sys.argv[0]} TASK [OPTIONS]")
        print(f"Run '{sys.argv[0]} TASK --help' for playback options.")
        print("Available tasks:")
        for task in sorted(set(tasks) | set(TASK_MODES)):
            print(f"  {task}")
        return

    task_id, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    cfg = tyro.cli(
        PlayConfig,
        args=remaining,
        default=PlayConfig(),
        prog=f"{sys.argv[0]} {task_id}",
        config=mjlab.TYRO_FLAGS,
    )
    run_play(task_id, cfg)


if __name__ == "__main__":
    main()
