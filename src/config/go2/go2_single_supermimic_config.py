# Copyright (c) 2026-2027 zh
"""内容：定义参考训练预算下的单一盲MoE策略及Go2解析速度任务参数。"""

import math

from agents.supermimic.single.config import SingleConfig
from src.config.base.base_config import BaseConfig
from .go2_supermimic_config import Go2SuperMimicCfg


class Go2SingleActorCfg(BaseConfig):
    """新训练配置；旧CTS类仅用于显式旧配置和历史检查点。"""

    algorithm = "single_moe"
    # 保留历史快照字段及任务指纹；仅作元数据，不参与预算选择。
    profile = "reference"
    sim = Go2SuperMimicCfg.sim
    control = Go2SuperMimicCfg.control
    init_state = Go2SuperMimicCfg.init_state
    reset = Go2SuperMimicCfg.reset
    viewer = Go2SuperMimicCfg.viewer
    commands = Go2SuperMimicCfg.commands
    noise = Go2SuperMimicCfg.noise
    single = SingleConfig

    class env(Go2SuperMimicCfg.env):
        num_observations = 45
        num_obs = 225
        num_privileged_obs = 130
        num_critic_obs = 130

    class collision(Go2SuperMimicCfg.collision):
        enable_self_collision = False

    class terrain(Go2SuperMimicCfg.terrain):
        scan_size_m = (1.2, 0.72)
        scan_resolution_m = 0.12

    class reward:
        """状态解析的速度目标与正则；权重在每步聚合时仅乘一次控制dt。"""

        tracking_std = 0.5
        upright_std = 0.4
        base_height_min = 0.18
        fall_height = 0.12
        fall_angle = math.radians(70)
        fall_softness_deg = 5.0
        height_softness_m = 0.02

        class scales:
            track_linear_velocity = 2.0
            track_angular_velocity = 1.0
            upright = 1.0
            lin_vel_z = -2.0
            body_ang_vel = -0.05
            joint_deviation_l2 = -0.1
            base_height_low = -10.0
            action_rate_l2 = -0.05
            joint_pos_limits = -1.0
            joint_torques_l2 = -2e-5
            joint_acc_l2 = -2.5e-7
            is_terminated = -200.0

    class runner:
        enable_control_updates = True
