# Copyright (c) 2026-2027 zh
"""内容：装配Go1单阶段视觉训练，保存来源、指标、32步留出验证和可恢复检查点。"""

from dataclasses import asdict
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from agents.perceptive_supermimic import PerceptiveRunner
from agents.perceptive_supermimic.export import export_policy
from src.config.go1_depth.go1_depth_supermimic_config import IMAGINATION_HORIZON

from .adapter import Go1DepthEnv
from .evaluation import compare_trajectories, plot_trajectory_comparison, replay_heldout_trajectories


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def save_sources(output: Path) -> None:
    """保留实际源码文本和Git状态，不计算文件哈希，不改动工作区。"""
    root = Path(__file__).resolve().parents[5]
    status = subprocess.check_output(["git", "status", "--short", "--untracked-files=all"], cwd=root, text=True)
    _write_json(output / "code_state.json", {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(), "status": status})
    (output / "workspace.diff").write_text(subprocess.check_output(["git", "diff", "HEAD"], cwd=root, text=True), encoding="utf-8")
    sources = []
    for directory in ("agents/perceptive_supermimic", "src/tasks/locomotion/go1_depth", "src/config/go1_depth",
                      "src/assets/robots/go1_depth", "src/terrain/vision", "src/terrain/blind"):
        sources.extend((root / directory).rglob("*.py"))
    sources += [root / name for name in ("scripts/train_go1_depth.py", "scripts/play_go1_depth.py",
                                        "tests/test_training/go1_depth_experiment.py",
                                        "agents/supermimic/policy.py", "agents/supermimic/world_model.py",
                                        "agents/supermimic/learner.py", "agents/base/modules/mlp.py",
                                        "src/tasks/locomotion/go2_supermimic/mdp/terrains.py",
                                        "src/tasks/locomotion/go2_supermimic/mdp/reference_rotations.py",
                                        "src/assets/robots/go1_depth/xmls/go1.xml")]
    for source in sources:
        target = output / "source" / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


@torch.no_grad()
def verify_export(runner, output):
    """对实际Go1观测验证CPU策略、TorchScript和ONNX双输入一致性。"""
    from onnx.reference import ReferenceEvaluator

    paths = export_policy(runner.actor, output, iteration=runner.iteration, metadata={
        "task": "go1_depth", "training_horizon": IMAGINATION_HORIZON,
        "raw_camera_processing": json.loads(runner.spec.depth_semantic_id),
    })
    module = deepcopy(runner.actor).cpu().eval()
    scripted = torch.jit.load(str(paths["pt"]))
    evaluator = ReferenceEvaluator(str(paths["onnx"]))
    history = runner.history.detach().cpu()
    depth = runner.current.observation.depth.detach().cpu()
    errors = {"torchscript_max_error": 0.0, "onnx_max_error": 0.0}
    for batch in sorted({1, min(3, history.shape[0])}):
        expected = module(history[:batch], depth[:batch])
        scripted_output = scripted(history[:batch], depth[:batch])
        onnx_output = torch.from_numpy(evaluator.run(None, {
            "obs_history": history[:batch].numpy(), "depth": depth[:batch].numpy()})[0])
        torch.testing.assert_close(scripted_output, expected, rtol=0, atol=0)
        torch.testing.assert_close(onnx_output, expected, rtol=4e-4, atol=4e-5)
        errors["torchscript_max_error"] = max(errors["torchscript_max_error"], float((scripted_output - expected).abs().max()))
        errors["onnx_max_error"] = max(errors["onnx_max_error"], float((onnx_output - expected).abs().max()))
    return errors


def train(env_config, algorithm, execution, *, device, output, iterations: int,
          minimum_training_seconds: float = 0.0, timeout_seconds: float = 1800.0,
          checkpoint=None, verify: bool = False):
    """完成指定迭代与最少实际训练时间；额外验证只在显式请求时执行。"""
    if algorithm.horizon != IMAGINATION_HORIZON or algorithm.wm_window != 8 or algorithm.wm_only_iterations != 0:
        raise ValueError("Go1 requires horizon=32, model window=8 and single-stage updates")
    if algorithm.action_clip != env_config.action_clip:
        raise ValueError("Actor and environment action clipping must match")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("Training iterations must be a positive integer")
    if not math.isfinite(minimum_training_seconds) or minimum_training_seconds < 0:
        raise ValueError("Minimum training time must be finite and nonnegative")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= minimum_training_seconds:
        raise ValueError("Wall timeout must be finite and exceed minimum training time")
    torch.manual_seed(env_config.seed)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "configuration.json", {
        "environment": asdict(env_config), "algorithm": asdict(algorithm), "execution": asdict(execution),
        "device": str(device), "iterations": iterations, "minimum_training_seconds": minimum_training_seconds,
        "timeout_seconds": timeout_seconds, "checkpoint": str(checkpoint) if checkpoint else None,
        "torch_version": str(torch.__version__), "command": sys.argv,
        "scope": "go1_mjlab_single_stage_perceptive_supermimic", "verify": verify,
    })
    save_sources(output)
    print(f"Initializing Go1: device={device}, environments={env_config.num_envs}, horizon=32", flush=True)
    env = Go1DepthEnv(env_config, device=device, latent_dim=algorithm.latent_dim)
    try:
        runner = PerceptiveRunner(env, algorithm, execution=execution, log_dir=output)
        if checkpoint is not None:
            runner.load(checkpoint)
        _write_json(output / "task_spec.json", asdict(runner.spec))
        initial_iterations, initial_seconds = runner.iteration, runner.training_seconds
        before_parameters = {name: value.detach().cpu().clone() for name, value in runner.actor.named_parameters()}
        terrain_before = ({name: value.detach().cpu().clone() for name, value in runner.terrain_perception.named_parameters()}
                          if runner.terrain_perception is not None else {})
        before_evaluation = compare_trajectories(env, runner.actor, runner.model, output / "evaluation_before", batches=2) if verify else None
        # 评估样本不写入训练回放，之后重新reset并填充真实训练数据。
        runner.reset_collection()
        start_wall = time.perf_counter()
        from torch.utils.tensorboard import SummaryWriter
        with SummaryWriter(log_dir=str(output / "tensorboard")) as writer, (output / "metrics.jsonl").open("w", encoding="utf-8") as stream:
            def record(metrics):
                values = dict(metrics)
                # 环境附加诊断明确标为本轮最后一个控制步，不伪装成整轮均值。
                for name, value in env.last_metrics.items():
                    number = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
                    if not math.isfinite(number):
                        raise FloatingPointError(f"Environment diagnostic is non-finite: {name}")
                    values[f"last_control_step/{name}"] = number
                stream.write(json.dumps(values, allow_nan=False) + "\n")
                stream.flush()
                for name, value in values.items():
                    writer.add_scalar(name, value, runner.iteration)

            latest = runner.learn(1, callback=record)
            _write_json(output / "ready.json", {"initialized": True, "iteration": runner.iteration,
                                                "horizon": algorithm.horizon, "actor_updates": runner.learner.actor_updates})
            print("Training started: real Go1 collection and all optimizer updates completed", flush=True)
            while runner.iteration - initial_iterations < iterations or runner.training_seconds - initial_seconds < minimum_training_seconds:
                if time.perf_counter() - start_wall > timeout_seconds:
                    raise TimeoutError("Go1 training exceeded the configured wall-clock budget")
                latest = runner.learn(1, callback=record)
        actual_seconds = runner.training_seconds - initial_seconds
        checkpoint_path = runner.save(output / "model_final.pt")
        changes = {}
        for name, prefix in (("depth_encoder", "encoder."), ("history_encoder", "history_encoder."),
                             ("visual_projector", "visual_projector."), ("actor", "actor.")):
            changes[name] = max(float((value.detach().cpu() - before_parameters[key]).abs().max())
                                for key, value in runner.actor.named_parameters() if key.startswith(prefix))
        if terrain_before:
            changes["terrain_perception"] = max(float((value.detach().cpu() - terrain_before[name]).abs().max())
                                                for name, value in runner.terrain_perception.named_parameters())
        summary = {
            "actual_training_seconds": actual_seconds, "training_iterations": runner.iteration - initial_iterations,
            "total_iterations": runner.iteration, "horizon": algorithm.horizon, "model_window": algorithm.wm_window,
            "wm_only_iterations": algorithm.wm_only_iterations, "last_metrics": latest,
            "model_updates": runner.learner.model_updates, "actor_updates": runner.learner.actor_updates,
            "critic_updates": runner.learner.critic_updates, "parameter_changes": changes,
            "checkpoint": str(checkpoint_path), "full_training_duration_met": actual_seconds >= 600,
            "robot_control_convergence_verified": False, "sim_to_real_verified": False,
            "latent_comparison_uses_each_checkpoints_encoder_and_scale": True,
        }
        if verify:
            if any(change <= 0 for change in changes.values()):
                raise AssertionError(f"Missing parameter update: {changes}")
            if runner.learner.actor_updates != runner.iteration:
                raise AssertionError("Single-stage actor update count differs from iteration count")
            summary["export_errors"] = verify_export(runner, output / "export")
            heldout = replay_heldout_trajectories(env.task, runner.actor, runner.model,
                                                 output / "evaluation_before/trajectories.npz", output / "evaluation_heldout_after")
            after = compare_trajectories(env, runner.actor, runner.model, output / "evaluation_policy_after", batches=2)
            summary["heldout_evaluation"] = heldout
            summary["policy_evaluation"] = after
            plot_trajectory_comparison(before_evaluation, heldout, output / "trajectory_comparison.png")
            # 同一对象显式重载检查点并重新采集，验证恢复后仍能完成32步策略更新。
            runner.load(checkpoint_path)
            previous_updates = runner.learner.actor_updates
            runner.learn(1)
            summary["resume_verified"] = runner.learner.actor_updates == previous_updates + 1
            if not summary["resume_verified"]:
                raise AssertionError("Resume did not update the actor")
        _write_json(output / "summary.json", summary)
        print(json.dumps({name: summary[name] for name in ("actual_training_seconds", "training_iterations", "horizon", "checkpoint")}), flush=True)
        return summary
    finally:
        env.close()
