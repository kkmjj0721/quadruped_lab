# Copyright (c) 2026-2027 zh
"""
内容：
    将 Go2 PPO 集中参数装配为 MjLab 运行器配置。
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from src.config.go2.go2_config import Go2CfgPPO

def unitree_go2_ppo_runner_cfg(ppo_cfg: Go2CfgPPO | None = None) -> RslRlOnPolicyRunnerCfg:
  """创建 Unitree Go2 速度任务的 PPO runner 配置。"""
  go2_ppo_cfg = ppo_cfg if ppo_cfg is not None else Go2CfgPPO()
  return RslRlOnPolicyRunnerCfg(
    actor = RslRlModelCfg(
      hidden_dims = tuple(go2_ppo_cfg.policy.actor_hidden_dims),
      activation = go2_ppo_cfg.policy.activation,
      obs_normalization = go2_ppo_cfg.policy.obs_normalization,
      distribution_cfg = {
        "class_name": "GaussianDistribution",
        "init_std": go2_ppo_cfg.policy.init_noise_std,
        "std_type": go2_ppo_cfg.policy.distribution_std_type,
      },
    ),

    critic = RslRlModelCfg(
      hidden_dims = tuple(go2_ppo_cfg.policy.critic_hidden_dims),
      activation = go2_ppo_cfg.policy.activation,
      obs_normalization = go2_ppo_cfg.policy.obs_normalization,
    ),

    algorithm = RslRlPpoAlgorithmCfg(
      value_loss_coef = go2_ppo_cfg.algorithm.value_loss_coef,
      use_clipped_value_loss = go2_ppo_cfg.algorithm.use_clipped_value_loss,
      clip_param = go2_ppo_cfg.algorithm.clip_param,
      entropy_coef = go2_ppo_cfg.algorithm.entropy_coef,
      num_learning_epochs = go2_ppo_cfg.algorithm.num_learning_epochs,
      num_mini_batches = go2_ppo_cfg.algorithm.num_mini_batches,
      learning_rate = go2_ppo_cfg.algorithm.learning_rate,
      schedule = go2_ppo_cfg.algorithm.schedule,
      gamma = go2_ppo_cfg.algorithm.gamma,
      lam = go2_ppo_cfg.algorithm.lam,
      desired_kl = go2_ppo_cfg.algorithm.desired_kl,
      max_grad_norm = go2_ppo_cfg.algorithm.max_grad_norm,
    ),

    logger = go2_ppo_cfg.runner.logger,
    experiment_name = go2_ppo_cfg.runner.experiment_name,
    save_interval = go2_ppo_cfg.runner.save_interval,
    num_steps_per_env = go2_ppo_cfg.runner.num_steps_per_env,
    max_iterations = go2_ppo_cfg.runner.max_iterations,
    seed = go2_ppo_cfg.runner.seed,
    clip_actions = go2_ppo_cfg.runner.clip_actions,
  )
