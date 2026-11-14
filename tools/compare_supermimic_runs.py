# Copyright (c) 2026-2027 zh
"""
内容：
    对齐两次已结束、单个learn块的SuperMimic窗口日志，比较含记录开销的速度与训练指标。
    校验窗口连续覆盖、累计步数和配置；生成JSON与独立曲线，不运行训练。

用法：
    .venv/bin/python -B tools/compare_supermimic_runs.py --before <改前日志目录> --after <改后日志目录> --output <分析产物目录>
"""

import argparse
import json
import math
from pathlib import Path
import statistics

import yaml
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader


def differences(left, right, prefix=""):
    """保留字段路径，区分配置差异与运行时间等来源元数据。"""
    result = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(left.keys() | right.keys()):
            result.extend(differences(left.get(key), right.get(key), f"{prefix}.{key}" if prefix else key))
    elif left != right:
        result.append({"field": prefix, "before": left, "after": right})
    return result


def read_run(path):
    summary = json.loads((path / "timing_summary.json").read_text())
    assert summary["run"]["success"] and summary["run"]["start_iteration"] == 0
    assert summary["session"]["learn_blocks"]["calls"] == 1, "Only a single completed learn block is supported"
    files = list(path.glob("events.out.tfevents.*"))
    assert len(files) == 1, "Expected one event file per run"
    rows, scalar_count = {}, 0
    for event in LegacyEventFileLoader(str(files[0])).Load():
        for value in event.summary.value:
            if not value.HasField("simple_value"):
                continue
            scalar_count += 1
            assert math.isfinite(value.simple_value), "Nonfinite logged scalar"
            if not value.tag.endswith("/time"):
                row = rows.setdefault(event.step, {})
                assert value.tag not in row, "Duplicate scalar at the same iteration"
                row[value.tag] = value.simple_value
    rows = {step: row for step, row in sorted(rows.items()) if "timing/iteration_seconds" in row}
    windows = []
    for row in rows.values():
        if "timing/previous_window_wall_seconds" in row:
            windows.append({"start": int(row["timing/previous_window_start_iteration"]),
                            "end": int(row["timing/previous_window_end_iteration"]),
                            "seconds": row["timing/previous_window_wall_seconds"]})
    last = rows[max(rows)]
    windows.append({"start": int(last["metrics/window_start_iteration"]), "end": max(rows),
                    "seconds": summary["session"]["metrics_windows"]["total_seconds"]
                    - math.fsum(window["seconds"] for window in windows)})
    previous = 0
    for window in windows:
        row = rows[window["end"]]
        count = window["end"] - window["start"] + 1
        assert window["start"] == previous + 1 and window["seconds"] > 0
        assert row["metrics/window_start_iteration"] == window["start"]
        assert row["metrics/window_iterations"] == count
        window["iterations"] = count
        previous = window["end"]
    assert previous == summary["run"]["final_iteration"]
    assert sum(row["real_steps_delta"] for row in rows.values()) == summary["run"]["new_training_real_steps"]
    snapshots = {}
    for name in ("task_config.json", "params/agent.yaml", "params/env.yaml", "params/run.yaml"):
        load = json.loads if name.endswith(".json") else yaml.safe_load
        snapshots[name] = load((path / name).read_text())
    return {"path": str(path), "rows": rows, "summary": summary, "windows": windows,
            "snapshots": snapshots, "scalar_count": scalar_count}


def summarize(run, lo, hi):
    windows = [w for w in run["windows"] if lo <= w["start"] and w["end"] <= hi]
    count = sum(w["iterations"] for w in windows)
    assert count == hi - lo + 1, "Range must align with complete log windows"
    rows = [(w, run["rows"][w["end"]]) for w in windows]
    seconds = math.fsum(w["seconds"] for w in windows)
    transitions = sum(row["real_steps_delta"] for _, row in rows)
    result = {"iterations": count, "records": len(rows), "complete_seconds": seconds,
              "seconds_per_iteration": seconds / count, "real_transitions_per_second": transitions / seconds}
    # P90描述窗口平均耗时的分布，不能误标为逐轮尾延迟。
    means = sorted(w["seconds"] / w["iterations"] for w in windows)
    result["window_mean_median_seconds"] = statistics.median(means)
    result["window_mean_p90_seconds"] = means[math.ceil(.9 * len(means)) - 1]
    keys = ("timing/collection_seconds", "timing/world_model_seconds", "timing/control_seconds",
            "wm_loss", "critic_loss", "velocity_rmse", "episode/return_mean", "episode/length_steps_mean",
            "task/terrain_level", "task/action_rms", "real_reward",
            "task/tracking_linear_mse", "task/tracking_angular_mse")
    for key in keys:
        accepted = [(row["collection/control_steps"] if key.startswith("task/") or key == "real_reward"
                     else w["iterations"], row[key]) for w, row in rows if key in row]
        if accepted:
            result[key] = math.fsum(weight * value for weight, value in accepted) / sum(weight for weight, _ in accepted)
    for axis in ("linear", "angular"):
        result[f"tracking_{axis}_rmse"] = math.sqrt(result[f"task/tracking_{axis}_mse"])
    for key in ("failures", "timeouts", "reset_steps", "episode/completed", "collection/control_steps"):
        result[key] = sum(row.get(key, 0) for _, row in rows)
    result["failure_fraction"] = (result["failures"] / result["episode/completed"]
                                  if result["episode/completed"] else None)
    result["failures_per_million_transitions"] = result["failures"] * 1e6 / transitions
    result["reset_steps_per_iteration"] = result["reset_steps"] / count
    result["guard_rollback_records"] = sum(row.get("guard/rollback", 0) for _, row in rows)
    return result


def comparison(before, after):
    final = before["summary"]["run"]["final_iteration"]
    assert final == after["summary"]["run"]["final_iteration"] == 5000
    result = {"before": before["path"], "after": after["path"], "configuration_differences": {},
              "sessions": {}, "ranges": {}, "details": {}, "final_metrics": {}, "validation": {},
              "collection_reset_correlations": {}}
    for name in before["snapshots"]:
        result["configuration_differences"][name] = differences(before["snapshots"][name], after["snapshots"][name])
    for label, run in (("before", before), ("after", after)):
        result["sessions"][label] = {key: value["total_seconds"] for key, value in run["summary"]["session"].items()}
        result["validation"][label] = {"success": True, "scalar_count": run["scalar_count"],
                                        "complete_window_count": len(run["windows"]), "run": run["summary"]["run"]}
        result["final_metrics"][label] = {key: value for key, value in run["rows"][final].items()
                                          if not key.startswith(("timing/", "Loss/", "Perf/", "Train/", "Grad/", "Metrics/"))}
        result["details"][label] = run["summary"]["stages"]
        steady = [(w, run["rows"][w["end"]]) for w in run["windows"] if w["start"] >= 51]
        result["collection_reset_correlations"][label] = statistics.correlation(
            [row["timing/collection_seconds"] for _, row in steady],
            [row["reset_steps"] / window["iterations"] for window, row in steady])
    for lo, hi in ((1, 1), (1, 50), (51, 500), (501, 1000), (1001, 2000),
                   (2001, 5000), (4001, 5000), (4951, 5000), (51, 5000), (1, 5000)):
        left, right = summarize(before, lo, hi), summarize(after, lo, hi)
        result["ranges"][f"{lo}-{hi}"] = {"before": left, "after": right,
            "duration_change_percent": (right["complete_seconds"] / left["complete_seconds"] - 1) * 100,
            "throughput_change_percent": (left["complete_seconds"] / right["complete_seconds"] - 1) * 100}
    result["complete_windows"] = {label: run["windows"] for label, run in (("before", before), ("after", after))}
    result["limits"] = ["Single paired training run; no repeated seeds or controlled concurrent GPU load",
                        "Run provenance does not contain source revision/diff or GPU UUID",
                        "Complete window wall includes synchronization, guard and logging; last window uses session residual",
                        "Detailed steady sampling covers iterations divisible by 50, all with 3 collection steps",
                        "Host section durations can include waiting for preceding GPU work and must not be added to CUDA durations",
                        "Tracking RMSE is sqrt of control-step-weighted MSE; episode means are averages of last-100 snapshots",
                        "Training trajectories and RNG consumption differ; metrics are not fixed-checkpoint evaluation"]
    return result


def plot(before, after, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "Noto Sans CJK SC", "axes.unicode_minus": False, "font.size": 10})
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for label, run, color in (("改前 16:36", before, "#31688e"), ("改后 18:09", after, "#d56b30")):
        windows = [w for w in run["windows"] if w["start"] >= 51]
        steps = [w["end"] for w in windows]
        durations = [w["seconds"] / w["iterations"] * 1000 for w in windows]
        axes[0, 0].plot(steps, durations, color=color, label=label, linewidth=1.4)
        elapsed = 0
        cumulative = []
        for window in windows:
            elapsed += window["seconds"]
            cumulative.append(elapsed)
        axes[0, 1].plot(steps, cumulative, color=color, label=label)
        for axis, key in ((axes[1, 0], "linear"), (axes[1, 1], "angular")):
            axis.plot(steps, [math.sqrt(run["rows"][step][f"task/tracking_{key}_mse"]) for step in steps],
                      color=color, label=label, linewidth=1.4)
    titles = ("含日志的窗口平均每轮耗时", "从第51轮开始的累计训练时间", "线速度跟踪误差", "偏航角速度跟踪误差")
    units = ("毫秒 / 轮", "秒", "RMSE（m/s）", "RMSE（rad/s）")
    for axis, title, unit in zip(axes.flat, titles, units):
        axis.set(title=title, xlabel="迭代", ylabel=unit)
        axis.grid(alpha=.22)
        axis.legend(frameon=False)
    figure.suptitle("同 cuda:1 · 4096 env · seed 0 · 5000轮；曲线从第51轮开始\n单次训练记录对比，不能替代多种子与固定检查点评估", fontsize=13)
    for suffix in ("png", "svg"):
        figure.savefig(output / f"comparison.{suffix}", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two completed SuperMimic runs")
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    before, after = read_run(args.before), read_run(args.after)
    report = comparison(before, after)
    (args.output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    plot(before, after, args.output)
    print("[PASS] Both runs cover all 5000 iterations with finite scalars and matching transition totals")
    print(json.dumps({"configuration_differences": report["configuration_differences"], "ranges": report["ranges"]}, indent=2))
