# Copyright (c) 2026-2027 zh
"""
内容：
    定义Go2 SuperMimic训练任务标识及命令行启动参数。
"""

from dataclasses import dataclass
from pathlib import Path


TASK_MODES = {
    "Unitree-Go2-SuperMimic-Flat": "flat",
    "Unitree-Go2-SuperMimic-Rough": "rough",
}


@dataclass(frozen=True)
class SuperMimicTrainConfig:
    """启动参数独立于任务语义；任务与网络细节从集中配置或JSON读取。"""

    device: str = "cuda:0"
    """单个运行设备，例如cpu或cuda:0。"""
    num_envs: int | None = None
    """覆盖并行环境数量；训练至少使用两个并行环境。"""
    seed: int | None = None
    """覆盖任务及网络随机种子。"""
    max_iterations: int = 10000
    """本次追加的训练轮数，续训时也使用追加语义。"""
    save_interval: int = 500
    """按累计训练轮数保存检查点的间隔；最后一轮始终保存。"""
    log_interval: int = 1
    """终端分区日志间隔；JSONL仍逐轮保存。"""
    imag_microbatch: int = 256
    """单策略想象反传同时处理的种子数；不改变总想象批量。"""
    critic_microbatch: int = 4096
    """单策略每次critic反传的样本数；不改变critic优化器更新次数。"""
    log_root: str = "logs/supermimic"
    experiment_name: str = "go2_supermimic"
    run_name: str = ""
    resume: Path | None = None
    """本地参数检查点；默认读取其同目录task_config.json。"""
    task_config: Path | None = None
    """任务参数JSON或runner的task_config.json快照，可覆盖网络与学习预算。"""
    model_only: bool | None = None
    """True只拟合世界模型；新训练默认False，续训默认保持原快照模式。"""


def normalize_train_arguments(arguments: list[str]) -> list[str]:
    """兼容普通HIM入口的常用参数名，统一转为本任务的单设备启动参数。"""
    aliases = {
        "--env.scene.num-envs": "--num-envs", "--agent.seed": "--seed",
        "--agent.max-iterations": "--max-iterations", "--agent.save-interval": "--save-interval",
        "--agent.experiment-name": "--experiment-name", "--agent.run-name": "--run-name",
    }
    normalized = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        flag, separator, value = argument.partition("=")
        if flag == "--gpu-ids":
            if not separator:
                index += 1
                if index >= len(arguments):
                    raise ValueError("--gpu-ids requires one GPU index or None")
                value = arguments[index]
            if value == "None":
                device = "cpu"
            elif value.isdecimal():
                device = f"cuda:{int(value)}"
            else:
                raise ValueError("SuperMimic supports one GPU index or None for --gpu-ids")
            normalized.extend(("--device", device))
        else:
            mapped = aliases.get(flag, flag)
            normalized.append(mapped + ("=" + value if separator else ""))
        index += 1
    return normalized
