# Copyright (c) 2026-2027 zh
"""内容：
    提供参考 Go2 PPO 网络、优化器与运行器配置。
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from src.config.go2.go2_config import Go2Cfg

go2cfg = Go2Cfg()

def unitree_go2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """创建 Unitree Go2 速度任务的 PPO 运行器配置。"""
  return RslRlOnPolicyRunnerCfg(
    actor = RslRlModelCfg(
      hidden_dims = (512, 256, 128),
      activation = "elu",
      obs_normalization = True,
      distribution_cfg = {
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),

    critic = RslRlModelCfg(
      hidden_dims = (512, 256, 128),
      activation = "elu",
      obs_normalization = True,
    ),

    algorithm = RslRlPpoAlgorithmCfg(
      value_loss_coef = 1.0,
      use_clipped_value_loss = True,
      clip_param = 0.2,
      entropy_coef = 0.01,
      num_learning_epochs = 5,
      num_mini_batches = 4,
      learning_rate = 1.0e-3,
      schedule = "adaptive",
      gamma = 0.99,
      lam = 0.95,
      desired_kl = 0.01,
      max_grad_norm = 1.0,
    ),

    logger = "tensorboard",         # 日志后端可选 tensorboard 或 wandb。
    experiment_name = "go2_ppo",
    save_interval = 100,
    num_steps_per_env = 24,
    max_iterations = 10001,
  )
