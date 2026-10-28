<!-- Copyright (c) 2026-2027 zh -->

# 本项目 Go2 SuperMimic 与 go2_him 的任务层对齐评估

> 本文保留下方本轮实施前的分析及当时建议。2026-10-28用户已明确要求加入HIM步态／腾空奖励、Go2 PPO姿态奖励及heading命令，并已确认想象端采用足高软接触和连续计时。原文“不采纳heading／接触奖励”等判断的前置条件已改变，不能再作为当前实现约束。新的来源核查、授权差异与验收以[当前设计](../design/go2_blind_supermimic.md#当前实施基准参考训练全对齐him-命令与三项奖励2026-10-28)为准。

本文回答一个具体问题：本项目的 Go2 SuperMimic 任务，是否应该在**命令采样**、**环境与 MDP 组装方式**、**奖励与终止**这三处对齐相邻项目 `legged_wbc_mjlab` 的 `src/tasks/locomotion/go2_him`（HIM 任务），以及除此之外还有哪些地方值得借用它的做法。

结论先行：**三处都应当"部分对齐"，但没有任何一处应当整套照搬**。原因是两个任务分属不同算法族——本项目是 SuperMimic（无 PPO，用可微世界模型在想象中反传），`go2_him` 是 HIM PPO（带特权估计器）。`go2_him` 只能提供**任务层**做法，其中凡依赖真实接触、真实传感器或非可微量的部分，迁入想象回路就会失效。

## 0. 阅读约定与依据分层

依据标记沿用 [Go2 SuperMimic 子系统详细分析](go2_supermimic_analysis.md#0-阅读约定与依据分层)：

- **本项目已验证事实**：本机运行或读码可直接复核。
- **论文依据**：SuperMimic 英文 v1（[本地 PDF](../../paper/SuperMimic_2610.28480v1.pdf)）正文或附录。
- **参考项目依据**：本机仓库实际读到的代码。
- **推断／待验证假设**：由资料推导，附最小验证方法。

对比基线：

| 项 | 本项目 | `go2_him` |
| --- | --- | --- |
| 路径 | `/home/sunteng/projects/quadruped_lab` | `/home/sunteng/projects/legged_wbc_mjlab` |
| 提交 | `dfee759a635dcf7a1cb3d4ed357d052017d4601a`，工作区有未提交改动 | `e6209ec3d910b57673d5cbba1d3ca798a7fa3113`，工作区 2 项未提交差异（与本文件相关源文件无关） |
| 算法 | SuperMimic：真实采集 → 8 步窗口拟合可微世界模型 → 32 步想象 BPTT ＋ λ-return critic，**无 PPO** | HIM PPO ＋ 特权估计器 |
| 观测 | 45×5＝225（纯本体） | 47×6＝282（含 2 维相位）＋特权 244 |

行号简写：`design:<n>` 指本项目 `docs/design/go2_blind_supermimic.md`（仅在无歧义处简写为 `:n` 的引用同样指它）；`config:<n>` 指本项目 `src/config/go2/go2_reference_supermimic_config.py`；无路径前缀的 `.py` 文件名（如 `transition_adapter.py`、`reference_task.py`、`reference_rewards.py`、`go_supermimic_env_cfg.py`）指本项目 `src/tasks/locomotion/go2_supermimic/` 下同名文件，`supermimic/mdp/...`、`supermimic/rl/...` 是它的子目录简写；`go2_ppo/...` 指本项目 `src/tasks/locomotion/go2_ppo/`。`go2_him_cfg:<n>` 指 `legged_wbc_mjlab` 的 `src/tasks/locomotion/go2_him/go2_him_env_cfg.py`；`him_config:<n>` 指同项目 `src/config/go2/go2_him_config.py`；`him_layouts.py` 指同项目 `src/config/base/him_layouts.py`；`him_env_cfgs:<n>`、`him_curriculum:<n>`、`him_rewards:<n>`、`numerics.py` 分别指 `src/tasks/locomotion/go2_him/config/env_cfgs.py`、`src/tasks/locomotion/ri_4438_him/mdp/curriculums.py`、`src/tasks/locomotion/ri_4438_him/mdp/rewards.py`、`src/tasks/locomotion/ri_4438_him/mdp/numerics.py`。`go2_him` 的观测、奖励和课程函数实际定义在 `ri_4438_him` 的 `mdp/` 下并由 `go2_him` 复用，引用时按实际文件给出。`mjlab/...` 与 `mujoco_warp/...` 指本项目 `.venv/lib/python3.11/site-packages/` 下的安装版本；`go2_supermimic/...`、`env/...` 指参考仓库 `/home/sunteng/projects/go2-supermimic@d9eee9b3` 下的路径。

本文只做任务层评估，不改代码；算法层对照见[与上游参考仓库的逐项对比](go2_supermimic_vs_reference.md)，实现细节见[子系统详细分析](go2_supermimic_analysis.md)，设计决策见[Go2 盲速度跟踪设计](../design/go2_blind_supermimic.md)。

## 1. 一句话结论

三处对齐的判断各不相同：

| 议题 | 判断 | 一句话理由 |
| --- | --- | --- |
| 命令采样 | **保留机制，只采纳前向专用环境** | 采样必须留在可微任务模型内作为 context 的函数，不能换成 MjLab 的 `UniformVelocityCommand`；唯一值得借的是 `rel_forward_envs`，周期区间化会偏离论文且不建议 |
| 环境／MDP 组装 | **不改成参考项目那样，反向借用两项** | 本项目的"空 manager ＋ adapter 唯一来源"是为了真实与想象共用一套实现；参考项目的 manager 组装在想象回路不可用。可借 `metrics` 项与仿真容量预算核对 |
| 奖励与终止 | **奖励不增加，终止结论已被交叉印证** | 22 项奖励里绝大多数不可微；70° 倾倒由参考项目独立复现，0.35 m 高度终止经实测对本项目不适用 |
| 其他 | **动作 `clip` 是唯一高优先级缺口** | 实测 1.99% 关节目标越过 MJCF 硬限位，最大越界 0.776 rad |

## 2. 命令采样：要不要对齐

### 2.1 两边做法

本项目（**本项目已验证事实**，`config:177-188`、`mdp/reference_task.py:166-173`）：

- 三轴独立均匀采样，范围 `((-1.5, 2.0), (-1.32, 1.32), (-3.35, 3.35))`，取论文附录 G 的非参考命令支路。
- 站立概率 0.1，`zero_probability_start = zero_probability_end = 0.1` 且 `zero_probability_iterations = 1`，因此**没有爬升过程**。
- `resampling_time = (5.0, 5.0)` 为固定值；`resample` 由 `advance` 内的 `timer <= 1e-7` 触发，新命令与新 timer 在**每回合重置**时也会重采（`transition_adapter.py:150-168`）。
- 采样参数来自 26 维 context 向量的切片 `CONTEXT_RANGES`／`CONTEXT_ZERO_PROBABILITY`，即**采样是任务模型的纯函数**，真实采集与想象共用同一实现。

`go2_him`（**参考项目依据**，`go2_him_cfg:166-187`、`him_config:36-48`、MjLab `mjlab/tasks/velocity/mdp/velocity_command.py:74-98`）：

- `resampling_time_range = (4.0, 12.0)`，三轴各自独立均匀采样。
- `rel_standing_envs = 0.1`，另有 `rel_forward_envs = 0.05`、`rel_heading_envs = 1.0`、`heading_command = True`、`heading_control_stiffness = 0.5`。
- forward 环境把 `vx` 取 `abs().clamp(min=0.3)`、`vy` 与 `ωz` 清零（`mjlab/tasks/velocity/mdp/velocity_command.py:90-98`）；heading 环境每步按航向误差闭环改写 `ωz`（同文件 `:118-125`）。
- MjLab 另有 `init_velocity_prob` 可在重置时直接写入基座速度（`mjlab/tasks/velocity/mdp/velocity_command.py:100-113`）。

### 2.2 逐项判断

| 维度 | 本项目 | `go2_him` | 是否对齐 | 依据与理由 |
| --- | --- | --- | --- | --- |
| 采样实现位置 | 任务模型内，context 驱动 | MjLab `CommandManager` ＋ `CommandTerm` | **不对齐** | 换成 manager 会让采样读机器人实时状态，想象回路无对应状态；会破坏"真实／想象单一来源"这一既有约束（见 `design:518`、`design:740`）。**推断**：改造代价高且无收益 |
| 采样周期 | 固定 5.0 s（论文 250 控制步） | (4.0, 12.0) s | **不对齐** | 论文附录 G 明确"每 250 控制步"；改区间即偏离复现基准。**本项目已验证事实**：实测首次切换发生在第 251 步（5.02 s），float32 计时与论文的 250 步差 1 步，属计时实现细节，可接受 |
| 采样范围 | 论文附录 G 支路 | vx ±0.5、vy ±1、ωz ±2（训练） | **不对齐** | 范围差异来自任务定位；本项目范围由用户确认（`design:101`） |
| 站立概率 | 0.1，无爬升 | 0.1 | **已一致** | — |
| 前向专用环境 | 无 | 0.05，`vx` 夹到 ≥0.3 | **可采纳（P1）** | 强行前进能压制"原地踏步骗跟踪奖励"。**推断**：对本项目 yaw 跟踪偏弱的问题无直接帮助，但能改善 vx 单一的样本分布；实现上只需在 context 增加一位掩码并在 `sample_commands` 内夹取，代价小 |
| heading 命令 | 关闭，且配置校验直接拒绝（`go_supermimic_env_cfg.py:30-31`） | 开启（stiffness 0.5） | **不对齐** | `ωz` 命令每步随航向误差变化就无法在 32 步想象中冻结，`design:279` 已把这一点列为必须显式处理的 L1 问题 |
| 世界系命令环境 | 无（`rel_world_envs` 概念不存在） | 支持，默认 0.0 | **不对齐** | 本项目命令已明确为"航向系平移＋世界系 yaw"，再叠世界系支路只会增加语义分叉 |
| 重置时给初速度 | 无 | `init_velocity_prob`，默认 0.0 | **不对齐** | 任务模型的 `TaskState` 没有速度注入通道，重置只能由 `reset_root_state_uniform` 写物理状态；**推断**：若要做，必须让 `read_physical` 在重置后重读一次，风险高于收益 |
| 命令课程 | 阶段表为空，仅由 terrain_limits 限速 | `RewardBasedVelocityCurriculum`（线性门槛 0.75、增量 0.1、10 s 检查） | **不对齐** | 该课程消费 PPO 奖励管理器统计（`him_curriculum:185-194`），本项目奖励来自 adapter，统计口径不同。`design:281` 已记录这一迁移风险 |

### 2.3 采纳清单

1. **前向专用环境**（P1）：在 context 增加 1 位掩码，`sample_commands` 内对掩码行取 `vx = max(|vx|, 0.3)`、`vy = ωz = 0`。需同步 `CONTEXT_DIM` 26→27 并升 spec 版本。最小验证：重置后统计 5% 行满足掩码语义，且想象 rollout 与该行命令一致。
2. **周期区间化**（可选，P2，需用户确认）：`resampling_time` 改为 `(4.0, 12.0)` 可让各环境相位错开，但会偏离论文的固定 250 步。**不建议**在没有对照实验前改动。

**明确不建议**：整体替换为 MjLab 命令管理器、开启 heading 命令、引入 `init_velocity_prob`。

## 3. 环境与 MDP 组装方式：要不要照参考项目写

### 3.1 两种组装哲学

| 维度 | 本项目 | `go2_him` |
| --- | --- | --- |
| 观测 | `observations={}` 空，观测由任务模型 `observe()` 单点生成（`go_supermimic_env_cfg.py:154`） | 两个观测组（actor 带历史、critic 单帧），逐项配置噪声／延迟／历史（`go2_him_cfg:110-143`） |
| 奖励／命令／课程 | `rewards`、`commands`、`curriculum`、`metrics` 全为空 dict（`go_supermimic_env_cfg.py:119-122`） | 全部由 manager 装配（`go2_him_cfg:351-510`） |
| 终止 | 仅 `time_out`（`go_supermimic_env_cfg.py:116`），失败由 adapter 在 post-forward 状态上算 | `time_out` ＋ `bad_orientation(70°)`（`go2_him_cfg:491-496`） |
| 自动重置 | `auto_reset=False`（`go_supermimic_env_cfg.py:162`） | `auto_reset=False`（`go2_him_cfg:594`），但 rough 配置里被改成 `True`（`him_env_cfgs:59`） |
| 指标 | `metrics={}`，指标由 adapter 产出 `task/*` 键 | `metrics={"mean_action_acc": ...}`（`go2_him_cfg:582`） |

### 3.2 判断

**不改成参考项目那样。** 依据是三方共同的约束，而不是偏好：

1. **论文依据**：SuperMimic 的奖励必须对**想象出的下一状态**可求值，且要与真实采集用同一套解析式（论文附录 A 表 A.6 及"analytic reward"表述）。MjLab manager 形态的奖励项函数签名是 `(env, **params)`，只读 `env.scene["robot"].data`，无法接受想象状态（`design:518`、`design:740`）。
2. **本项目已验证事实**：空 manager 的构造与 reset 行为已被测试覆盖，且 `auto_reset=False` 让 adapter 能在 reset 前复制终态；这是奖励读取 post-forward 状态的前提（`transition_adapter.py:193-205`）。
3. **参考项目依据**：`go2_him` 的观测项延迟是"每项一组 lag 缓冲"，与 MjLab 观测管理器强绑定；本项目的延迟是按物理子步的执行器延迟（`config:112-115`），语义不同，不能互换。

**可以反向借用的两项**（这两处是参考项目的工程优点，且不破坏上面的架构）：

| 项 | `go2_him` 做法 | 本项目可做什么 | 优先级 |
| --- | --- | --- | --- |
| 动作加速度指标 | `metrics={"mean_action_acc"}`（`go2_him_cfg:582`，实现 `mjlab/envs/mdp/metrics.py:11-25`：`a_t − 2a_{t−1} + a_{t−2}` 的绝对值按关节取均值） | adapter 目前只记 `action_rms`（`transition_adapter.py:226`），无法区分"幅值大但平滑"与"抖动"。建议在 `last_metrics` 增加 `action_acc` 一项 | P1 |
| 观测数值护栏 | `checked_observation`（`numerics.py:10-24`）：isfinite ＋ 原始上限检查，保留证据、不做主机同步 | 本项目已有等价保护（`transition_adapter.py:189-196` 对动作和物理状态抛 `FloatingPointError`）。**无需改造**，但可借鉴其"保留坏样本证据"的做法用于事后诊断 | P2 |

### 3.3 顺带发现：仿真容量预算与设计文档不一致

这是本轮审查发现的**文档-代码漂移**，与对齐无关但影响 `go2_him` 对比的可信度：

- **本项目已验证事实**：`config:40` 现为 `nconmax = None`，测试也断言为 `None`（`tests/test_tasks/test_go2_supermimic_config.py:86`）。
- **本项目已验证事实**：`design:783` 写的是"在集中配置 `sim` 段显式维护 `nconmax=48`…`contact_sensor_maxmatch=64`"。历史上该值确为 48（`416d05b:src/config/go2/go2_supermimic_config.py:36`），在任务目录整理（`291995e`）重写配置时变为 `None`，`contact_sensor_maxmatch` 字段整体消失。
- **本项目已验证事实**：`None` 会走 MuJoCo Warp 的启发式（`mujoco_warp/_src/io.py:1282-1292`）。实测 flat 得 48（与设计值相同），**rough 得 128**；按 `ccd_iterations=50` 估算 EPA 数组在 4096 环境下 flat 约 0.604 GB、rough 约 1.611 GB，而设计文档的 0.5625 GiB 只对应 48 接触。
- 影响：rough 的显存预算比设计文档描述的高约 2.7 倍，仍未回到最初的 14.68 GiB 失败值，所以训练可跑；但文档陈述与代码不符。**建议**（P1，纯文档＋一行配置）：要么把 `nconmax` 显式写回 48，要么修正`design:776-784` 的表述并说明按地形模式取启发式。`contact_sensor_maxmatch` 因本任务 `sensors=()` 无接触传感器，属无效字段，文档应删去。

## 4. 奖励与终止

### 4.1 为什么不能照搬 22 项

`go2_him` 的奖励有 **22 项**（`go2_him_cfg:351-489`，本文按 `RewardTermCfg` 计数核对）。本项目锁定论文表 A.6 的 **6 项**（`config:190-204`、`reference_rewards.py`、[Go2 SuperMimic 文档](../go2_supermimic.md) 第 44 行）。差距不是"保守"，而是**可微性**：

| `go2_him` 奖励类别 | 代表项 | 想象回路的可用性 |
| --- | --- | --- |
| 读接触／力传感器 | `foot_gait`、`feet_air_time`、`foot_slip`、`soft_landing`、`leg_collision`、`self_collision`、`stumble` | **不可用**：世界模型不预测接触，想象中无此信号 |
| 读真实力矩／雅可比 | `joint_torques_l2`、`angular_momentum` | **不可用**：需真实逆动力学 |
| 读 `is_terminated` | `is_terminated`（−100） | 可换成软存活，但那是论文的 `1−p_fail` 语义，不是 −100 惩罚 |
| 读预测状态 | `lin_vel_z`、`body_orientation_l2`、`action_rate_l2`、`smoothness`、`joint_acc_l2`、`joint_pos_limits`、`stand_still`、`hip_pos`、`pose` | **可用**，但多数会偏离表 A.6 |

论文附录 A 已明确把自碰撞惩罚从模仿任务中移除，并说明推扰会"注入确定性世界模型无法表达的状态不连续"。**推断**：同一逻辑适用于全部接触类奖励。

### 4.2 可微候选的逐项判断

| 候选 | `go2_him` 权重／定义 | 本项目现状 | 判断 |
| --- | --- | --- | --- |
| `lin_vel_z` | −2.0，**机体系** `root_link_lin_vel_b[:,2]²`（`him_rewards:41-47`） | 无；只能用 `VELOCITY_HEADING[:,2]`（heading 系 z 分量）近似，**两者不严格等价**（roll/pitch 非零时不同） | **不采纳**：表 A.6 无此项；本项目 `torso_sideways_lean`（−4.0）已覆盖侧倾，垂向速度惩罚主要服务于 PPO 的步态稳定，非表 A.6 语义 |
| `body_orientation_l2` | −0.2，重力 xy 平方 | 无（`sideways_lean` 只罚侧倾，**已验证**与 `sin²(roll)` 的最大差 8.9e-8） | **不采纳**：并入会与 −4.0 侧倾项重复计权，且改变"纯前倾不罚"的既有设计（`reference_rewards.py:19`） |
| `smoothness`（动作二阶差） | −0.015，`action_acc_l2` | 无；只有一阶 `action_rate_l2`（−0.3） | **不采纳为奖励**，但**采纳为指标**（见 §3.2） |
| `joint_acc_l2` | −2.5e-7 | 无（状态里无关节加速度） | **不采纳**：需对预测状态做有限差分，噪声放大，性价比低 |
| `joint_pos_limits` | **−10.0** | −2.0，且用 `soft_joint_pos_limits`（0.9×硬限位） | **保留本项目值**：−10 是 PPO 为避免硬限位冲击的设置；本项目已用软限位缓冲，且论文表 A.6 的 −2 有明确出处 |
| `track_angular_velocity` | **1.0**，std=√0.25 | 2.0，方差 0.5 | **保留本项目值**：表 A.6 明确 2.0；参考项目的 1.0 反映其 heading 命令下的权重平衡，前提不同 |
| `stand_still` / `hip_pos` | −0.5 / −0.5，按命令阈值 0.1 门控（`him_rewards:393-431`） | 无 | **不采纳**：站立行为已由 10% 零命令样本隐式学习；加门控惩罚会在想象中引入命令阈值这一分段非线性 |

**结论**：奖励项一项都不加。唯一值得做的是把 `smoothness` 的**度量**引入日志。

### 4.3 终止

| 项 | 本项目 | `go2_him` | 参考仓库 | 论文表 A.6 |
| --- | --- | --- | --- | --- |
| 倾倒判定 | 70°（`config:195`），想象用 5° sigmoid 软化 | 70°（`go2_him_cfg:493-495`） | 70°（`go2_supermimic/env/env_cfg.py:121-125`） | `1+u_z > 0.7`（≈73°） |
| 高度终止 | 无 | 无 | 无 | `h_root < 0.35 m` |
| 越界 | 支持（`reference_task.py:213-221`） | 无 | `outside_arena` 惩罚 | 无 |
| 非法接触 | 不适用 | `illegal_contact`（`him_env_cfgs:172-175`） | 无 | 无 |

三点结论：

1. **70° 得到独立交叉印证**（**参考项目依据**）：`go2_him` 与 `go2-supermimic` 各自独立取 70°，本项目取 70° 有双来源支撑。表 A.6 的 ≈73° 是 G1 速度接口的数值，不适用于 Go2。
2. **0.35 m 高度终止对本项目不适用，且已实测**（**本项目已验证事实**）：用归档 `model_3000.pt` 播放 200 步 × 16 环境，基座高度中位数 **0.2874 m**、95 分位 **0.3362 m**，**98.1% 的步低于 0.35 m**。若照搬该阈值，任务会在正常行走时几乎全程终止。这是 Go2（0.29 m 站立高度）与 G1（更高）的体型差异，`design:100` 的判断由此得到定量支持。
3. `illegal_contact` 不适用（**推断**）：本项目无接触传感器、无自碰撞终止，且想象中无接触信号。

### 4.4 顺带核实：两项奖励实现正确

对照 `go2_him` 的实现复核本项目代理量的数学等价性，结论是**本项目实现正确，不需要改**（**本项目已验证事实**）：

- `tracking_velocity` 的 yaw 分量 `−(g·ω)` 与真实世界系 `ω_z` 的差为 **0.0**（2 万组随机姿态解析核验）；因此"世界系 yaw 角速度"语义与表 A.6 一致。
- `tilt_rate` 的 `|ω|² − ω_z²` 与 heading 系 roll/pitch 角速度平方和的最大差 **3.3e-6**（解析）／**9.1e-6**（实机播放）。
- 注意二者的**参考实现不同**：`go2-supermimic` 的 `tilt_rate_l2` 直接用机体系 `w[:, :2]` 平方和（`go2_supermimic/env/mdp.py:132-136`），与"heading 系"在俯仰/横滚不为零时**不同**（实测均差 0.24、最大差 4.71 rad²/s²，折算每步奖励差均值 2.4e-4、最大 4.7e-3）。本项目按表 A.6 的 heading 系实现，属**正确的偏离**，不要按参考仓库改回机体系。

## 5. 其他部分的审计

### 5.1 动作空间没有限幅（唯一高优先级缺口）

| 项 | 本项目 | `go2_him` |
| --- | --- | --- |
| 动作配置 | `JointPositionActionCfg` 未设 `clip`（`go_supermimic_env_cfg.py:68-75`） | 按关节设 `clip`（`go2_him_cfg:145-164`） |
| clip 取值 | — | hip ±1.0472、前大腿 (−1.5708, 3.4907)、后大腿 (−0.5236, 4.5379)、小腿 (−2.7227, −0.83776) |
| 来源 | — | "绝对目标限位取自本 Go2 MJCF"（同处注释） |

**本项目已验证事实**：

- MjLab 的 `BaseActionCfg.clip` 默认 `None`（`mjlab/managers/action_manager.py:29-30`），`process_actions` 仅在非 `None` 时 clamp（`mjlab/envs/mdp/actions/actions.py:154-163`）。因此未设 `clip` 就是**完全不限幅**。
- 本项目编译后的执行器 `actuator_ctrlrange` 是**力矩**限位（±23.7／±45.43），不会代替关节位置限幅。
- `go2_him` 的 clip 数值与本项目 `robot.data.joint_pos_limits` 逐项相同（hip ±1.0472、前大腿 (−1.5708, 3.4907)、后大腿 (−0.5236, 4.5379)、小腿 (−2.7227, −0.83776)）。
- 用归档 `model_3000.pt` 播放 200 步 × 16 环境共 38 400 个关节目标：**1.99%（764 个）越过 MJCF 硬限位，最大越界 0.776 rad**；按软限位计为 3.14%、最大越界 0.881 rad。越界集中在 `RR_calf`（7.9%）、`RR_hip`（4.8%）、`FR_calf`／`RL_calf`（各约 4%），大腿关节为 0。
- 动作原始值 `max|a| = 5.74`，即策略确实在自由使用未限幅区间。

**建议**（P0）：给 `JointPositionActionCfg` 增加 `clip`，取值直接复用 `robot.data.joint_pos_limits` 的符号表达式或写成配置里的显式区间。收益有两处：真实仿真不再靠 MuJoCo 硬限位被动截断；想象回路里 `integrate` 使用的动作空间被约束在物理可达集内。最小验证：现有检查点上重跑上述统计，硬限位越界率应为 0；再用同一策略播放对比跟踪误差是否变化。

### 5.2 其余逐项

| 项 | 本项目 | `go2_him` | 判断 |
| --- | --- | --- | --- |
| 观测维度 | 45×5＝225（含命令 3） | 47×6＝282（含命令 3、相位 2） | **不加相位**：相位是 PPO 周期步态的辅助输入；本项目无步态奖励、无 gait 项，相位无对应监督信号 |
| 历史帧数 | 5（`config:30`） | 6（`him_layouts.py:10` 默认值，`him_config:21` 引用） | **保留 5**：来自参考仓库 `ACTOR_HISTORY=5`（`go2_supermimic/env/env_cfg.py:30`）；论文正文与附录 A 写"last 10 observations"（论文第 675、1021 行），本项目已记录为已知偏离（`design:27`、`design:895`） |
| `joint_vel` 噪声 | 1.5 | 1.5（`go2_him_cfg:80-83`） | **保留 1.5**：与参考仓库（`go2_supermimic/env/env_cfg.py:179-181`）和 `go2_him` 双重一致；论文表 A.1 的 ±0.5 是 G1 设置。**不要"修正"成 0.5** |
| 动作延迟 | 0～2 物理子步（`config:112-115`），经 `DcMotorActuator` 生效 | 观测侧同步延迟 0／1／hold 0.3／period 10（`him_config:84-88`）＋ 逐关节动作 clip | **保留本项目**：观测延迟属"传感器分组滞后"，与执行器延迟语义不同，本项目模型显式只接收当前与上一动作 |
| 播放时延迟 | **保留** 0～2 子步（实测 instance 上 `delay_max_lag=2` 仍生效） | 播放时全部清零（`him_env_cfgs:187-195`） | **可采纳（P2）**：参考项目在播放时清零延迟是"播放用名义动力学"的明确选择；本项目播放保留了随机延迟，会造成同一策略在播放与评估间有 0～20 ms 差异。**推断**：评估口径应固定，建议评估时清零并在文档记录 |
| 播放时长 | 20 s（`episode_length_s` 不覆盖） | `int(1e9)`（`him_env_cfgs:182`） | **保留 20 s**：本项目评估协议明确用 20 s 回合（`rl/evaluation.py:29-30`、`design:645`），超时重置是评估语义的一部分 |
| 重置随机化 | x/y ±0.1、yaw ±π，无 roll/pitch、无关节偏移、无初速度（`config:49-57`） | x/y ±0.5、roll/pitch ±0.3、yaw ±π（`him_config:50-57`） | **部分可采纳（P2）**：小幅 roll/pitch 能缓解"总能从完全水平姿态起步"的过拟合。**推断**：任务模型从仿真读状态，应能容忍；最小验证是加 ±0.1 rad 后跑短训练，观察 `falls_per_iteration` 是否异常上升 |
| 域随机化范围 | 摩擦 (0.3, 1.5)、编码器 ±0.01、COM ±(0.025/0.05/0.05)，全部 `startup` | 摩擦 (0.25, 1.6)、编码器 ±0.02、`pseudo_inertia`（质量＋惯量＋COM 联合）、PD、力矩、armature、关节摩擦/阻尼、IMU 误差、接触软硬 | **不扩展**：每新增一项物理 DR 都要扩 context 并升 spec（`design:278`）。摩擦与编码器幅度已取论文或参考值；`pseudo_inertia` 的"质量＋惯量＋COM 联合"超出本项目 26 维 context 表达能力 |
| 推扰 | 无（正确） | `push_robot` 每 4～6 s（`go2_him_cfg:214-228`）＋ 短时外力 | **明确不对齐**：论文要求移除；参考仓库也主动 `cfg.events.pop("push_robot")`（`go2_supermimic/env/env_cfg.py:118-119`）。确定性世界模型无法表达速度突变 |
| 地形课程晋级门槛 | `distance > map.size/2`，**无最小命令距离门**（`supermimic/mdp/curriculums.py:103`） | 额外要求 `commanded_distance >= 1.0`（`him_curriculum:78-82`） | **建议采纳（P1）**：本项目零命令／站立回合也可能因残余移动晋级。本项目 PPO 任务已有该门（`src/tasks/locomotion/go2_ppo/mdp/curriculums.py:57`，`min_commanded_distance=1.0`），属仓库内已有做法。注意：verified run 中地形课程**整体关闭**（`terrain.mode=flat`、`curriculum=False`、`task/terrain_level` 全程 0），该建议未经训练验证 |
| 地形课程测距基准 | 从回合起点算**最远**净移动 | 从 `env.scene.env_origins` 算当前位移 | **保留本项目**：最远净移动对"走出去又走回来"更宽容，且已写在 `design:730` |
| 指标命名 | `task/tracking_linear_mse` 等 | `log/...` ＋ `mean_action_acc` | **仅补 `action_acc`**，命名体系不改 |
| 自碰撞 | 开启（`config:123`），仅用于真实仿真 | 对本任务显式开启并调 `conaffinity`／`solref`（`him_env_cfgs:49-58`） | **保留**：本项目自碰撞只影响真实仿真，想象不建模，与论文一致 |

## 6. 优先级汇总

| 优先级 | 事项 | 位置 | 最小验证 |
| --- | --- | --- | --- |
| **P0** | 动作加入关节限位 `clip` | `go_supermimic_env_cfg.py:68-75` | 现有检查点上重跑"越界率"统计应为 0；同策略播放对比跟踪误差 |
| **P1** | 仿真容量预算与文档对齐（`nconmax` 显式 48 或改文档） | `config:40`／`design:783` | flat／rough 分别打印编译后 `Data.overflow` 与启发式取值 |
| **P1** | 地形课程加 `min_commanded_distance=1.0` 门 | `supermimic/mdp/curriculums.py:103` | 构造零命令回合，断言不晋级；既存地形课程测试 |
| **P1** | 日志增加 `action_acc` 指标 | `transition_adapter.py:221-229` | 用现有检查点跑 100 步，确认键出现在 metrics |
| **P1** | 命令前向环境（context 26→27） | `supermimic/mdp/reference_task.py:166-173` | 断言 5% 行满足 `vx≥0.3, vy=ωz=0`；spec 版本升级 |
| **P2** | 评估时清零执行器延迟 | `src/tasks/locomotion/go2_supermimic/config/env_cfgs.py:20-23` | 对比清零前后 `evaluate` 的跟踪误差与跌倒率 |
| **P2** | 重置增加小幅 roll/pitch（±0.1 rad） | `config:49-57` | 短训练观察 `falls_per_iteration` 是否异常上升 |
| **P2** | 借鉴坏样本证据保留（数值护栏） | `transition_adapter.py:189-196` | 仅在诊断脚本中使用，不进想象回路 |

## 7. 明确不采纳及理由

| 不采纳项 | 理由 |
| --- | --- |
| 整套 MjLab 命令管理器 | 采样必须可微且不依赖想象中不存在的机器人状态 |
| heading 命令与 `heading_control_stiffness` | 命令逐帧变化无法在 32 步想象中冻结（`design:279`） |
| 22 项奖励中的接触／力矩／速度类 | 世界模型不预测接触与力矩，梯度不存在 |
| `is_terminated`（−100） | 论文用软存活概率，不是固定惩罚 |
| `push_robot`／短时外力 | 论文明确移除；确定性世界模型无法表达速度突变 |
| `weights = joint_pos_limits −10` | 本项目已用软限位缓冲，且表 A.6 的 −2 有出处 |
| `track_angular_velocity` 权重 1.0 | 表 A.6 明确 2.0，参考项目值的前提是 heading 命令 |
| 0.35 m 高度终止 | 实测 98.1% 正常运行步低于该阈值 |
| 相位观测与步态奖励 | 无步态监督信号，且表 A.6 不含 gait 项 |
| `init_velocity_prob` | 任务模型无速度注入通道 |
| 照参考项目把容量写成固定小值而不修文档 | 见 §3.3：要么显式写回 48，要么保留启发式并修正文档，不能两者不一致 |

## 8. 证据缺口与限制

- `go2_him` 的奖励、观测、课程均**未在本机运行**；本文对其数值的判断来自读码，不构成其效果结论。
- 动作越界统计、基座高度分布、`tilt_rate` 等价性均用归档 `model_3000.pt`（iteration 3000，约 2400 轮后才具备行走能力，未收敛）在 CPU 上测得；不代表收敛策略的行为。
- 地形课程相关建议（`min_commanded_distance`）在 verified run 中课程处于关闭状态，**未经验证**。
- 训练目录已不在工作区：`logs/` 整体被移至 `/home/sunteng/.local/share/Trash/files/logs/`，路径为 `logs/supermimic/go2_supermimic/2026-10-27_19-28-11_197179/`。本文提到"归档检查点"均指该目录下的 `model_3000.pt`。
- `nconmax` 的启发式取值（flat 48／rough 128）由本机 `mujoco_warp/_src/io.py:_default_nconmax` 实测；EPA 字节数为按 `4096 × nconmax × (6 + 5 × 50) × 12` 的估算，不是实测显存峰值。
