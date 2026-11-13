# Copyright (c) 2026-2027 zh
"""内容：组合Go1本地单阶段视觉训练预算，固定32步想象与8步模型监督。"""

from agents.perceptive_supermimic.config import ExecutionConfig, PerceptiveConfig


IMAGINATION_HORIZON = 32


def make_algorithm_config(*, perception: bool = True) -> PerceptiveConfig:
    """保持现有网络宽度，只按本地显存缩小回放、监督和想象批量。"""
    return PerceptiveConfig(
        replay_capacity=128, image_capacity=160, warmup_steps=64, wm_only_iterations=0,
        wm_window=8, wm_batch=32, wm_updates=2, imag_batch=16, horizon=IMAGINATION_HORIZON,
        reconstruction_images=64, critic_epochs=2, critic_minibatches=2,
        action_clip=4.0, log_interval=50, save_interval=1000,
        reconstruction_weight=1.2 if perception else 1.0,
        depth_reconstruction_loss="mse" if perception else "l1",
        height_reconstruction_weight=1.0 if perception else 0.0,
        contrastive_weight=0.3 if perception else 0.0,
        contrastive_temperature=0.1, height_shape=(17, 11),
    )


def make_execution_config() -> ExecutionConfig:
    """分块改变峰值显存，不改变逻辑批次和32步展开长度。"""
    return ExecutionConfig(wm_microbatch=16, vision_batch=16, imag_microbatch=8, critic_microbatch=128)
