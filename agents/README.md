# 算法目录

本项目维护的算法统一放在 `agents/`。基础 PPO 位于 `agents/base/`；新增算法按职责建立独立子目录，任务观测、奖励和终止规则留在 `src/tasks/`，机器人与训练配置组合留在 `src/config/`。

```text
agents/
├── base/
│   ├── algorithms/   # PPO 与 GAE
│   ├── env/          # 向量化环境接口
│   ├── models/       # 前馈 actor、critic
│   ├── modules/      # MLP、高斯分布、观测归一化
│   ├── runners/      # 采样、更新、日志和检查点
│   ├── storage/      # rollout 与小批量
│   ├── utils.py      # 类与观测组配置解析
│   ├── LICENSE
│   └── NOTICE.md
├── supermimic/       # 世界模型、连续回放与BPTT学习
│   ├── config.py         # SuperMimicConfig、ExecutionConfig
│   ├── interfaces.py     # TaskState、EnvStep、AlgorithmSpec
│   ├── world_model.py    # 归一化与世界模型
│   ├── policy.py         # 混合专家 actor、critic
│   ├── buffer.py         # 种子、窗口、回放与归档
│   ├── imagine.py        # 想象轨迹与回报
│   ├── learner.py        # 梯度裁剪与更新
│   ├── runner.py         # 训练循环、保存与续训
│   ├── export.py         # TorchScript／ONNX 导出
│   ├── logging.py        # TensorBoard 与终端日志
│   └── NOTICE.md
└── __init__.py
```

## 基础 PPO

从仓库根目录导入：

```python
from agents.base import OnPolicyRunner

# env 为已初始化、实现 VecEnv 接口的向量化环境。
train_cfg = {
    "num_steps_per_env": 24,
    "save_interval": 50,
    "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
    "actor": {
        "class_name": "MLPModel",
        "hidden_dims": [256, 256, 256],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "log"},
    },
    "critic": {
        "class_name": "MLPModel",
        "hidden_dims": [256, 256, 256],
        "activation": "elu",
        "obs_normalization": True,
    },
    "algorithm": {
        "class_name": "PPO",
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "clip_param": 0.2,
        "gamma": 0.99,
        "lam": 0.95,
        "learning_rate": 0.001,
        "schedule": "adaptive",
        "desired_kl": 0.01,
        "entropy_coef": 0.01,
        "value_loss_coef": 1.0,
        "max_grad_norm": 1.0,
    },
}
runner = OnPolicyRunner(env, train_cfg, log_dir="outputs/ppo/run", device="cuda:0")
runner.learn(1000)
runner.save("outputs/ppo/run/resume.pt", infos={"seed": 7})
```

此段用于说明组件装配，需先提供真实环境；当前仓库尚无机器人任务及 `scripts/train.py`。可以先运行下面的合成环境测试验证完整训练链路。

`env.get_observations()` 返回 batch 为 `[N]` 的 `TensorDict`，每组为浮点 `[N, D]`；actor、critic 各自按 `obs_groups` 顺序拼接输入。未指定时优先使用同名观测组，其次使用 `policy`，critic 最后可复用 actor 组。

`env.step(actions)` 接收 `[N, A]`，返回 `(observations, rewards, dones, extras)`；rewards、dones 均为 `[N]`，环境负责自动重置终止实例。超时应同时设置 dones 和可选 `extras["time_outs"]`，后者使用 `[N]` 掩码；算法沿用参考实现，以当前状态价值补偿超时奖励。runner 将观测、奖励和终止张量移动到算法设备，动作移动到 `env.device`。

训练支持单 CPU 或单 CUDA 设备，使用前馈高斯策略、GAE、裁剪策略目标、可选裁剪价值损失、熵奖励和自适应 KL 学习率。观测归一化在整轮更新后刷新，推理时冻结；标准差支持 `scalar`（直接参数）和 `log`（对数参数）。不包含来源仓库的算法扩展、循环网络、多 GPU、导出或 TensorBoard 集成。

## 保存、恢复与推理

```python
runner.load("outputs/ppo/run/resume.pt")
runner.learn(100)
policy = runner.get_inference_policy()
# 使用 torch.no_grad() 调用 policy(observations)，默认输出确定性动作。
```

检查点保存 actor、critic、优化器、归一化统计和已完成更新次数。`learn(k)` 每次执行额外 k 次更新，自动保存 `model_<已完成更新次数>.pt`；指定 log_dir 时追加 `metrics.jsonl`，不指定时仅输出英文控制台日志。load 只应用于可信的本项目基础 PPO 检查点；不恢复环境内部状态或随机数状态，不保证逐步复现来源仓库训练轨迹。指定 `load_cfg={"actor": True}` 可仅加载 actor 权重。

## 验证与来源

```bash
uv run --no-sync python -B -m unittest discover -s tests/test_agents -v
```

测试在 CPU 合成环境验证采样、GAE、参数更新和检查点读写，不证明机器人训练收敛。来源、许可与适配范围见 [基础 PPO 说明](base/NOTICE.md)。

## SuperMimic 混合专家历史策略

默认 Go2 任务使用 `agents.supermimic`：go2_rl_gym 的八专家历史编码器加 ELU 动作网络、SiLU 世界模型和短时域 BPTT。无教师或蒸馏，部署输入只有五帧本体历史，不含线速度；线速度由历史隐式估计。配置与训练、播放、验证入口统一见 [Go2任务说明](../docs/go2_supermimic.md)，来源与迁移差异见 [supermimic/NOTICE.md](supermimic/NOTICE.md)。
