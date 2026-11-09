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
    """仅声明启动参数及显式覆盖；训练默认由机器人Python配置提供。"""

    device: str = "cuda:0"
    """单个运行设备，例如cpu或cuda:0。"""
    num_envs: int | None = None
    """覆盖并行环境数量；训练至少使用两个并行环境。"""
    seed: int | None = None
    """覆盖任务及网络随机种子。"""
    max_iterations: int | None = None
    """覆盖runner目标总轮数；未指定时沿用配置，续训仍解释为目标总轮数。"""
    save_interval: int | None = None
    """覆盖runner保存间隔；最后一轮也保存，塌缩护栏预警时跳过该保存点。"""
    log_interval: int | None = None
    """覆盖终端分区日志间隔；TensorBoard仍逐轮保存。"""
    imag_microbatch: int | None = None
    """覆盖单次想象反传的种子数上限；未指定时沿用runner配置。"""
    critic_microbatch: int | None = None
    """覆盖单次critic反传的样本数上限；未指定时沿用runner配置。"""
    wm_compile: bool | None = None
    """覆盖世界模型监督核心的编译开关；后端沿用algo.compile_backend。"""
    fused_adam: bool | None = None
    """覆盖全部优化器及护栏重建的fused Adam开关；未指定时沿用runner配置。"""
    cuda_graph: bool | None = None
    """覆盖CUDA更新图开关；CUDA默认启用且需要fused Adam，CPU使用原执行方式。"""
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
