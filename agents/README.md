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
├── supermimic/       # 世界模型、CTS-MoE、连续回放与BPTT学习
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

## SuperMimic＋CTS-MoE

`agents.supermimic`实现世界模型自回归拟合、教师／八专家学生闭环想象、共享动作网络／critic、学生表征监督与外层调度。来源及与原论文的差异见[NOTICE](supermimic/NOTICE.md)，任务语义与验收见[迁移设计](../docs/design/go2_blind_supermimic.md#12-算法层实施契约2026-10-26)。

装配需要实现`interfaces.py`的`TaskModel`和`CollectionEnv`。前者提供可微物理特征／积分、记忆与命令推进、观测、特权、奖励及归一化状态误差；后者必须返回reset前复制的`final_state`与reset后的`next_state`。`AlgorithmSpec.semantic_id`必须覆盖坐标、布局、动作缩放、时间步与任务语义，修改这些含义须升级标识；所有Tensor使用同设备float32，标志bool，episode／step／context版本int64。

```python
from agents.supermimic import (
    ActorCritic, DynamicsModel, LearnerConfig, RunnerConfig,
    SuperMimicLearner, SuperMimicRunner, WorldModelTrainer,
)

# task与env由具体任务提供，并共享同一设备。
policy = ActorCritic(task.spec).to(env.device)
world_model = DynamicsModel(task).to(env.device)
learner = SuperMimicLearner(policy, world_model, LearnerConfig())
trainer = WorldModelTrainer(world_model)
runner = SuperMimicRunner(env, learner, trainer, RunnerConfig(), log_dir="logs/supermimic/run")
runner.learn(1)
```

默认`enable_control_updates=False`仅采集和拟合模型。通过任务侧的留出预测／物理梯度验收后，显式设置`RunnerConfig(enable_control_updates=True)`，也可传`control_gate(metrics)`控制是否执行策略更新；算法不会根据训练误差自行认定物理导数可靠。教师比例与分支损失权重分别配置；控制只更新共享actor与教师encoder，学生只通过真实学生状态的latent监督更新。想象冻结模型、critic和学生参数，但保留输入导数。

回放不保存逐转移完整历史，采样时恢复同回合前缀；环形覆盖丢失前缀的起点拒采。失败与超时均截断真实窗口，超时不作为失败标签。`runner.save/load`保存模型、四个优化器、RMS、spec、分组、计数和RNG，格式明确为**参数续训**；加载后reset仿真、清空回放并重新预热，不保证与连续训练逐步等价。`get_inference_policy()`接收`(observation, history)`，仅执行学生，调用者维护旧→新且包含当前帧的历史；reset时重复首帧。

```bash
# 常规CPU行为检查，不自动执行显式训练冒烟。
.venv/bin/python -B -m unittest discover -s tests/test_agents -v

# 默认完整CTS-MoE网络的合成训练／参数续训，预算180秒。
RUN_SUPERMIMIC_TRAINING=1 timeout 180 .venv/bin/python -B -m unittest tests.test_training.test_supermimic_training -v
```

Go2的`TaskModel`、真实仿真adapter及runner装配工厂已提供，见[Go2任务说明](../docs/go2_supermimic.md)；命令行训练入口尚未新增。算法合成冒烟产物位于`.artifacts/supermimic/training-smoke/`，不证明Go2收敛、部署或性能优势；真实Go2轻量验证的范围在任务说明中单独维护。
