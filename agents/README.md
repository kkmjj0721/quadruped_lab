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
│   ├── visual/           # 视觉策略、显式感知记忆、辅助学习与想象接口
│   └── NOTICE.md
├── perception/      # MGDP 重建／对齐与 GRU 米制地图
├── foothold/        # 四足注意力选点与监督损失
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

默认 Go2 任务使用 `agents.supermimic`：go2_rl_gym 的八专家历史编码器加 ELU 动作网络、SiLU 世界模型和短时域 BPTT。无教师或蒸馏，部署只输入五帧本体历史；编码器输出32维潜特征和3维当前机身线速度估计，仅潜特征做L2归一化，速度由真实回放同一时刻真值监督。配置与训练、播放、验证入口统一见 [Go2任务说明](../docs/go2_supermimic.md)，来源与迁移差异见 [supermimic/NOTICE.md](supermimic/NOTICE.md)。

## 视觉感知与落足算法

| 入口 | 输入与职责 |
| --- | --- |
| `agents.perception.MGDPPerception` | 双帧 `[B,2,16,16]`、本体历史与显式 GRU 状态；预测米制 `[B,17,11]` 地图，另提供训练重建损失 |
| `agents.foothold.FootholdPlanner` | 本体历史、指令、重建地图与构造时提供的候选 xy；输出四足候选分布和 top-1 三维点 |
| `agents.supermimic.visual.VisualActor` | 本体历史、地图、任务锁存的四足目标与有效位；MLP 估计 32 维 latent 与 3 维机身速度，输出关节动作 |
| `agents.supermimic.visual.VisualLearner` | 感知序列监督、选点监督、已有世界模型拟合和视觉 BPTT；不执行环境采集或生成接触标签 |

`AlgorithmSpec.model_id` 使用 `supermimic_visual_foothold_v2`，旧视觉 MoE 检查点不可载入。历史估计器隐藏层按用户指定采用 256、128、64，结构为 `5D→256→128→64→35`、ELU、线性输出；前 3 维为速度、后 32 维 latent 做 L2 归一化。`D=45` 时输入 225 维，动作 MLP 输入 283 维。视觉估计器不使用 `SuperMimicConfig.moe_*`，动作 MLP 宽度继续由 `actor_hidden` 指定；来源与训练差异见 [NOTICE.md](supermimic/NOTICE.md#视觉感知与落足接口)。

视觉入口使用 `SuperMimicConfig(compile_imagination=False)`；未传执行配置时，actor、世界模型、critic、感知及选点五路网络默认全部开启 CUDA BF16，需要原生支持 BF16 的 CUDA 设备。感知包括深度／高度编解码器、GRU 和地图头，选点包括卷积、MLP 与注意力矩阵乘法；网络对外输出、状态、记忆、几何坐标、概率归一化、损失及返回递推保持 FP32。显式 `VisualExecutionConfig`（从 `agents.supermimic.visual` 导入）的 `wm_bf16`、`actor_bf16`、`critic_bf16`、`perception_bf16`、`planner_bf16` 分别控制五路；通用默认均为 False，CPU 检查传入 `ExecutionConfig(cuda_graph=False, wm_compile=False, fused_adam=False)`。请求 BF16 时不自动回退。CUDA Graph／想象编译仍关闭；原 `SuperMimicRunner` 和 `make_actor` 不装配视觉链。

任务实现 [VisualTask](supermimic/visual/interfaces.py) 的 `update_camera`、`update_visual_targets`、`visual_privileged`，并保留已有世界模型任务接口。相机函数负责预测位姿渲染、噪声、双帧和相位；目标函数负责锁存与坐标转换。算法在相机刷新时更新深度 token、每控制步推进 GRU，奖励及终止规则由任务维护。

`PerceptionPrefix` 必须为同回合、从起点到当前决策步之前的完整前缀，允许尾部填充；首个有效步必须刷新相机。`rebuild_seed()` 按当前权重重算 token 和隐藏状态，返回当前步感知更新前的 `VisualSeed`。`fit_perception()` 或 `load()` 后须重新构造种子，旧种子会报错。回放采集与相机事件压缩由后续任务运行器接入。

`PerceptionBatch.height_normalized` 是供 MGDP 自重建的有限完整标签；`height_metric` 和 `height_valid` 是独立米制地图监督。`FootholdBatch.candidate_valid` 标记 GT 候选的几何有效性，只进入损失。无效米制标签可以是 NaN，全部无效的监督批次跳过相应优化器更新。

`VisualLearner.save/load` 保存和恢复五个网络、优化器、归一化统计和学习排期，同时记录执行配置来源；恢复采用当前执行精度，不保存环境、回放或随机数状态。`export()` 生成 FP32 TorchScript 部署组合，仅包含深度编码器、GRU、地图头、选点和动作策略，不改变训练对象精度；调用者通过 `encode_depth`、`perceive_step`、`propose` 和 `forward` 显式管理帧保持与目标锁存。

算法级检查：

```bash
OMP_NUM_THREADS=1 .venv/bin/python -B -m unittest tests.test_agents.test_perception tests.test_agents.test_foothold tests.test_agents.test_visual_supermimic -v
```

来源与适配见 [感知说明](perception/NOTICE.md)、[选点说明](foothold/NOTICE.md)和[视觉 SuperMimic 说明](supermimic/NOTICE.md#视觉感知与落足接口)。这些模块尚未接入 Go1 实际采集与射线场景，算法级检查不代表机器人行走效果。
