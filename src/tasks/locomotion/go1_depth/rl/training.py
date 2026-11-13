# Copyright (c) 2026-2027 zh
"""内容：装配Go1分阶段视觉训练，保存来源、指标、32步留出验证和可恢复检查点。"""

from dataclasses import asdict, replace
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
from src.config.go1_depth.go1_depth_config import IMAGINATION_HORIZON

from .adapter import Go1DepthEnv
from .evaluation import compare_trajectories, evaluate_closed_loop, plot_trajectory_comparison, replay_heldout_trajectories


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
                      "src/assets/robots/go1_depth", "src/terrain/vision", "src/terrain/blind", "src/terrain/mgdp", "src/sim"):
        sources.extend((root / directory).rglob("*.py"))
    sources += [root / name for name in ("scripts/train_go1_depth.py", "scripts/play_go1_depth.py",
                                        "tests/test_training/go1_depth_experiment.py", "tests/test_training/go1_mgdp_experiment.py",
                                        "tests/test_training/go1_mgdp_diagnostic.py", "tests/test_training/go1_mgdp_snapshot.py",
                                        "tests/test_training/go1_mgdp_backend_probe.py",
                                        "agents/supermimic/policy.py", "agents/supermimic/world_model.py",
                                        "agents/supermimic/learner.py", "agents/base/modules/mlp.py",
                                        "src/tasks/locomotion/go2_supermimic/mdp/terrains.py",
                                        "src/tasks/locomotion/go2_supermimic/mdp/reference_rotations.py",
                                        "src/assets/robots/go1_depth/xmls/go1.xml")]
    for source in sources:
        target = output / "source" / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _validate_best_selection(runner, source, spec, *, maximum_iteration):
    """源目录的最佳感知必须与描述配套，且不能来自所恢复策略之后。"""
    selection = json.loads((source / "perception_best.json").read_text())
    payload = torch.load(source / "perception_best.pt", map_location=runner.device, weights_only=False)
    runner._validate_payload(payload, spec)
    if payload.get("metadata", {}).get("perception_selection") != selection:
        raise ValueError("Best perception checkpoint and selection metadata do not match")
    if selection["iteration"] != payload["counters"]["iteration"] or selection["iteration"] > maximum_iteration:
        raise ValueError("Best perception checkpoint is inconsistent or newer than the resumed policy")
    if not math.isfinite(selection["score"]):
        raise ValueError("Best perception selection score must be finite")
    return selection


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
          checkpoint=None, stage1_run=None, verify: bool = False):
    """完成指定迭代与最少实际训练时间；额外验证只在显式请求时执行。"""
    if algorithm.horizon != IMAGINATION_HORIZON or algorithm.wm_window != 8 or algorithm.wm_only_iterations != 0:
        raise ValueError("Go1 requires horizon=32, model window=8 and joint updates within each stage")
    if algorithm.action_clip != env_config.action_clip:
        raise ValueError("Actor and environment action clipping must match")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("Training iterations must be a positive integer")
    if not math.isfinite(minimum_training_seconds) or minimum_training_seconds < 0:
        raise ValueError("Minimum training time must be finite and nonnegative")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= minimum_training_seconds:
        raise ValueError("Wall timeout must be finite and exceed minimum training time")
    native = algorithm.perception_model == "mgdp"
    if stage1_run is not None and (not native or env_config.stage != 2 or checkpoint is not None):
        raise ValueError("Stage transfer requires MGDP stage 2 and cannot be combined with resume")
    if native and env_config.stage == 2 and checkpoint is None and stage1_run is None:
        raise ValueError("MGDP stage 2 requires a stage-1 run or a same-stage checkpoint")
    torch.manual_seed(env_config.seed)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    configuration = {
        "environment": asdict(env_config), "algorithm": asdict(algorithm), "execution": asdict(execution),
        "device": str(device), "iterations": iterations, "minimum_training_seconds": minimum_training_seconds,
        "timeout_seconds": timeout_seconds, "checkpoint": str(checkpoint) if checkpoint else None,
        "torch_version": str(torch.__version__), "command": sys.argv,
        "scope": "go1_mgdp_two_stage_supermimic" if native else "go1_legacy_perceptive_supermimic",
        "stage1_run": str(stage1_run) if stage1_run else None, "verify": verify,
    }
    _write_json(output / "configuration.json", configuration)
    save_sources(output)
    print(f"Initializing Go1: device={device}, environments={env_config.num_envs}, horizon=32", flush=True)
    if native:
        from .mgdp_adapter import MGDPGo1DepthEnv
        env = MGDPGo1DepthEnv(env_config, device=device, latent_dim=algorithm.latent_dim)
    else:
        env = Go1DepthEnv(env_config, device=device, latent_dim=algorithm.latent_dim)
    try:
        runner = PerceptiveRunner(env, algorithm, execution=execution, log_dir=output)
        transfer = None
        if checkpoint is not None:
            runner.load(checkpoint)
        elif stage1_run is not None:
            from ..mdp.mgdp_task import MGDPTask
            source = Path(stage1_run).resolve()
            source_configuration = json.loads((source / "configuration.json").read_text())
            expected_environment = asdict(replace(env_config, stage=1))
            if source_configuration["environment"] != expected_environment:
                raise ValueError("Stage transfer source must match the stage-1 environment configuration")
            # 任务构造无需启动第二套仿真；源语义由已验证的stage1配置和同一机器人确定。
            source_task = MGDPTask(replace(env_config, stage=1), env.task.nominal, env.task.limits,
                                   env.surface, device=env.device, latent_dim=algorithm.latent_dim)
            transfer = runner.load_components(source / "model_final.pt", source / "perception_best.pt", source_spec=source_task.spec)
            transfer["selection"] = _validate_best_selection(runner, source, source_task.spec,
                                                              maximum_iteration=transfer["policy_iteration"])
            _write_json(output / "stage_transfer.json", transfer)
        best = {"score": None, "iteration": None, "selection": "0.6*depth_mse+0.4*height_mse"}
        if native and checkpoint is not None:
            source = Path(checkpoint).resolve().parent
            best = _validate_best_selection(runner, source, runner.spec, maximum_iteration=runner.iteration)
            shutil.copy2(source / "perception_best.pt", output / "perception_best.pt")
            _write_json(output / "perception_best.json", best)

        def select_perception(metrics):
            if not native:
                return
            score = 0.6 * metrics["reconstruction"] + 0.4 * metrics["height_reconstruction"]
            if best["score"] is None or score < best["score"]:
                best.update(score=score, iteration=runner.iteration)
                runner.save(output / "perception_best.pt", metadata={"perception_selection": dict(best)})
                _write_json(output / "perception_best.json", best)
        _write_json(output / "task_spec.json", asdict(runner.spec))
        initial_iterations, initial_seconds = runner.iteration, runner.training_seconds
        stage_offset = initial_iterations if stage1_run is not None else 0
        if native and checkpoint is not None:
            source_configuration = json.loads((Path(checkpoint).resolve().parent / "configuration.json").read_text())
            if "stage_iteration_offset" not in source_configuration:
                raise ValueError("Resume source lacks the native perception-save clock; keep the historical run separate.")
            stage_offset = source_configuration["stage_iteration_offset"]
            if type(stage_offset) is not int or not 0 <= stage_offset <= initial_iterations:
                raise ValueError("Stage iteration offset must be an integer within the restored iteration range.")
        configuration["stage_iteration_offset"] = stage_offset
        _write_json(output / "configuration.json", configuration)
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
                # 官方it从0开始；阶段切换重新计时，普通resume沿用该阶段的保存相位。
                if (runner.iteration - stage_offset - 1) % algorithm.save_interval == 0:
                    select_perception(metrics)

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
            "stage": env_config.stage if native else None, "stage_transfer": transfer,
            "best_perception": best if native else None,
        }
        if verify:
            if any(change <= 0 for change in changes.values()):
                raise AssertionError(f"Missing parameter update: {changes}")
            if runner.learner.actor_updates != runner.iteration:
                raise AssertionError("Actor update count differs from iteration count")
            summary["export_errors"] = verify_export(runner, output / "export")
            heldout = replay_heldout_trajectories(env.task, runner.actor, runner.model,
                                                 output / "evaluation_before/trajectories.npz", output / "evaluation_heldout_after")
            after = compare_trajectories(env, runner.actor, runner.model, output / "evaluation_policy_after", batches=2)
            summary["heldout_evaluation"] = heldout
            summary["policy_evaluation"] = after
            summary["closed_loop_evaluation"] = evaluate_closed_loop(env, runner.actor, output / "evaluation_closed_loop")
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
