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
    max_iterations: int = 40000
    """训练的目标总轮数；continue_run时从检查点轮次继续至此目标，与参考一致。"""
    save_interval: int = 1000
    """按累计轮数保存模型及ONNX；最后一轮也保存，参考塌缩护栏预警时跳过该保存点。"""
    log_interval: int = 1
    """终端分区日志间隔；TensorBoard仍逐轮保存。"""
    imag_microbatch: int = 4096
    """单次想象反传的种子数上限；默认覆盖整个想象批量。"""
    critic_microbatch: int = 32768
    """单次critic反传的样本数上限；默认覆盖整个minibatch，保持每轮20次更新。"""
    wm_compile: bool = True
    """编译世界模型监督核心；后端沿用algo.compile_backend，默认s101开启。"""
    fused_adam: bool = True
    """三个优化器及护栏重建使用fused Adam；默认s101开启。"""
    log_root: str = "logs/supermimic"
    experiment_name: str = "go2_supermimic"
    run_name: str = ""
    resume: Path | None = None
    """本地参数检查点；默认读取其同目录task_config.json。"""
    continue_run: bool = False
    """与参考一致：False只热启动参数并从第1轮训练，True接续检查点轮次。"""
    reset_critic: bool = False
    """加载检查点时重新初始化critic，并按参考冻结actor以预热critic。"""
    task_config: Path | None = None
    """任务参数JSON或runner的task_config.json快照，可覆盖网络与学习预算。"""
    model_only: bool | None = None
    """True只拟合世界模型；新训练默认False，续训默认保持原快照模式。"""
