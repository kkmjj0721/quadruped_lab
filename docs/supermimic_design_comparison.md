<!-- Copyright (c) 2026-2027 zh -->

# GitHub SuperMimic 实现详解与配置对比（含本项目）

> 调研时间：2026-10-28（Asia/Shanghai）。数据来源为 19 个仓库 main 分支源码快照的**实际阅读**（下载后逐文件打开，记录 `路径:行号`），未运行其训练。本项目列来自当前工作区代码与 `logs/supermimic/go2_supermimic/` 运行快照。仅索引见 [SuperMimic 相关开源项目调研](supermimic_projects.md)。

## 0. 阅读约定

| 标记 | 含义 |
| --- | --- |
| **源码事实** | 仓库文件中直接读到的数值或结构，附 `路径:行号` |
| **本项目事实** | 当前工作区代码或运行快照中读到、跑出的结果 |
| **未注明** | 源码中确实没有该配置项，不是漏读 |
| **推断** | 由源码推导、未实测，文中显式标注 |

数值一律取**代码默认值**；README 的作者自报成绩与耗时未采信，也未复跑。各仓库的作者实跑命令可能与默认值不同（例如 `SuperMimic-MuJoCo` 的 CLI 默认 16 环境、`supermimic-rtx5080` 实跑 2048 环境），表中已尽量标注。

多数仓库的默认预算来自论文 Table A.5（4096 环境、1000 步回放、`wm_batch 2048 × 10`、`pi_batch 4096`、`horizon 32`），因此真正的设计差异集中在 **① 世界模型的状态与宽度 ② 策略是否有参考动作编码器 ③ 终止用解析软化还是学习头 ④ 配置如何组织**。

## 1. 总览表

### 1.1 底座、本体与任务

| # | 仓库 | 仿真/后端 | 机器人 | 任务 | Actor 输入 |
| --- | --- | --- | --- | --- | --- |
| 1 | [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) | Isaac Lab 3 kitless + Newton/MuJoCo-Warp | G1 29 DoF | LAFAN1 40 条 | 99 维×10 帧历史 + 730 维目标 |
| 2 | [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) | mjlab + MuJoCo Warp | G1 29 DoF | LAFAN1 / BONES-SEED | 93 维×10 帧 + 680 维目标 |
| 3 | [Motphys/Uni-SuperMimic](https://github.com/Motphys/Uni-SuperMimic) | UniLab（mjwarp/MuJoCo/Motrix） | G1 29 DoF | LAFAN1 40 条 | 99 维×10 帧 + 730 维目标 |
| 4 | [Zhengtianjin/go2-supermimic](https://github.com/Zhengtianjin/go2-supermimic) | mjlab + MuJoCo Warp | **Go2 12 DoF** | 追逐 + 闪避 | **70 维×5 帧 = 350**，无参考动作 |
| 5 | [lzyang2000/morphmimic](https://github.com/lzyang2000/morphmimic) | mjlab 1.6 | G1 / X2 Ultra / T2 | BONES-SEED 跟踪 | 93 维×10 帧 + 670 维目标 + 形态 |
| 6 | [btx0424/rethinking-supermimic](https://github.com/btx0424/rethinking-supermimic) | mjlab 1.6 | G1 29 DoF | LAFAN1 40 条 | 228 维×10 帧 + 730 维目标 |
| 7 | [Baojch/SuperMimic_Dyn](https://github.com/Baojch/SuperMimic_Dyn) | mjlab + MuJoCo Warp | G1 29 DoF | 空翻/侧手翻/爬行/起身 | 93 维×10 帧 + 680 维目标 |
| 8 | [panhainan2020/SuperMimic-MuJoCo](https://github.com/panhainan2020/SuperMimic-MuJoCo) | MuJoCo 3.8.1 原生 CPU 或 MJLab | G1 29 DoF | LAFAN1 + 100STYLE OOD | 99/93 维×10 帧 + 730 维目标 |
| 9 | [Shimingyi/allmimic](https://github.com/Shimingyi/allmimic) | mjlab + MuJoCo Warp | G1 29 DoF | LAFAN1/BONES-SEED，held-out | 93 维×10 帧 + 730 维目标 |
| 10 | [Renforce-Dynamics/trackerlab-supermimic](https://github.com/Renforce-Dynamics/trackerlab-supermimic) | Isaac Lab 2.3.2 / PhysX | G1/Go2/A3/T1/Bumi/Pi+ | LAFAN1 跟踪 + go2 变体 | 99 维×10 帧 + 730 维目标 |
| 11 | [HighTorque-Dev/supermimic-colab](https://github.com/HighTorque-Dev/supermimic-colab) | Isaac Lab 3 kitless（Docker） | G1 29 DoF | LAFAN1 40 条 | 同 #1（99 维×10 + 730） |
| 12 | [project-instinct/InstinctMJ](https://github.com/project-instinct/InstinctMJ)（`supermimic` 分支） | mjlab | G1 29 DoF | 深度视觉 parkour / shadowing | **1990 维**：目标 670 + 深度 576 + 本体 744 |
| 13 | [CumulusAlpha/Dex-Supermimic](https://github.com/CumulusAlpha/Dex-Supermimic) | mjlab 1.6 | Wuji Hand 2（20 关节） | 掌内重定向 / 转笔 | 155 维单帧，无参考动作历史 |
| 14 | [emoPointer/supermimic_manipulation](https://github.com/emoPointer/supermimic_manipulation) | mjlab 1.5.3（20 Hz） | ARX5 臂 + 夹爪（7 动作） | box_in_bin / lemon / mug_hang | **195 维（39×5）**，无参考动作 |
| 15 | [JzKosann/supermimic-isaaclab-h1_2](https://github.com/JzKosann/supermimic-isaaclab-h1_2) | Isaac Lab / Isaac Sim 5.1 | H1-2 27 DoF | LAFAN1（v0/v1 双配方） | 87 维×10 帧 + 640 维目标 |
| 16 | [Alchedata/supermimic-locomujoco](https://github.com/Alchedata/supermimic-locomujoco) | LocoMuJoCo + JAX | H1 | LAFAN1 40 条 + ODE 变体 | **217 维×10 帧 + 2170 维目标**（无噪声） |
| 17 | [rzc156/supermimic-rtx5080](https://github.com/rzc156/supermimic-rtx5080) | mjlab + MuJoCo Warp | G1 29 DoF | LAFAN1 / BONES-SEED | 93 维×10 帧 + 680 维目标 |
| 18 | [BarryLi159159/T2_supermimic](https://github.com/BarryLi159159/T2_supermimic) | mjlab（8bb2bc9） | G1 / Booster T2 / T2+Sharpa | 跟踪 + loco 速度任务 | 93 维×10 帧 + 680 维目标 |
| 19 | [OMEGA-i/SuperMimicKit](https://github.com/OMEGA-i/SuperMimicKit) | MuJoCo Warp 主 + Isaac/PhysX | G1 / H2 / A3 | LaFAN 重定向（train40/test37） | 99 维×10 帧 + 730 维目标 |
| **本** | **quadruped_lab `go2_supermimic`** | **mjlab + MuJoCo Warp** | **Go2 12 DoF** | **盲本体感知速度跟踪** | **45 维×5 帧 = 225**，无参考动作、无线速度 |

**三条最直接的结论**：

1. 19 个实现里 **17 个是参考动作跟踪**，actor 都带未来目标帧；只有 #4、#13、#14 与本项目不吃参考动作。但 #4 是追逐/闪避（目标物体观测）、#13/#14 是操作任务，**没有一个与本项目的"纯速度指令 + 盲四足"同构**。
2. 观测维度呈两极：#12 的 1990 维（含 576 维深度潜变量）、#16 的 2170 维（含全 body 位姿）属于"给足信息"，而 #13 的 155 维、#14 的 195 维、#4 的 350 维与本项目的 225 维属于"尽量少给"。
3. **只有 #4 与本项目是同机器人 + 同仿真底座**，其余本体的差异决定了它们的调参经验不能直接搬。

### 1.2 训练设计与网络

| # | 仓库 | WM 隐层 | WM 窗口/批量 | 想象批量×视界 | Actor | Critic | 配置组织 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | vibed | 2048×3 SiLU + LN，末层零初始化 | 8 / 2048×10 | 4096 × 32 | 编码 2048,1024,512,256→z32(tanh)＋解码 2048,1024,512,256→29 | 256,128 ELU | argparse |
| 2 | Reproduced | 2048×3 SiLU + LN，末层零初始化 | 8 / 2048×10 | 4096 × 32 | 编码/解码各 2048,1024,512,256，latent 32 | (256,128) | 嵌套 dataclass + tyro |
| 3 | Uni-SuperMimic | 2048×3 SiLU + LN | 8 / 2048×10 | 4096 × 32 | 2048,1024,512,256 ELU→z32→29 | 256,128 ELU | **Hydra YAML + dataclass** |
| 4 | **go2-supermimic** | **1024×3 SiLU** | **8 / 2048×2** | **8192 × 32** | **MLP 512,256,128 ELU（5 帧）** | 512,256,128 ELU | dataclass + 自动 argparse |
| 5 | morphmimic | 2048×3 ELU + LN | 8 / 2048×10 | 4096 × 32 | 2048,1024,512,256→z32→29（+形态条件） | 256,128 ELU + LN | 单文件 dataclass + tyro |
| 6 | rethinking | 2048×3 SiLU + RMSNorm | 8 / 2048×10 | 4096 × 32 | 2048,1024,512,256（split/trunk/film 三选） | **集成 2 头** 512,256,256 | **Hydra YAML** |
| 7 | Dyn | 2048×3 SiLU | 8 / 2048×10 | 4096 × 32 | 2048,1024,512,256→z32→29 | 256,128 SiLU | 嵌套 dataclass + tyro |
| 8 | MuJoCo | 2048×3 SiLU + LN | 8 / 2048×10 | 4096 × 32 | 2048,1024,512,256 ELU→z32→29 | 256,128 ELU | **纯 argparse** |
| 9 | allmimic | 2048×3 SiLU | 8 / 2048×10 | 4096 × 32 | 2048,1024,512,256 ELU→z32→29 | 256,128（可多头） | dataclass + argparse + PRESETS |
| 10 | trackerlab | 2048×3 SiLU + LN | 8 / 2048×10 | 4096 × 32 | 2048,1024,512,256 ELU→z32→29 | 256,128 ELU + Norm | argparse + JSON 预设 |
| 11 | colab | 同 #1 | 8 / 2048×10 | 4096 × 32 | 同 #1 | 256,128 | argparse |
| 12 | InstinctMJ | 2048×3 SiLU + LN（+深度编码器） | 8 / 2048×10 | **1024** × 32 | 视觉编码器 + 8 帧本体历史 | 256,128 | dataclass（并入 Isaac Lab agent cfg） |
| 13 | Dex | **2048×3 SiLU + LN** | 8 / 2048×10 | **1024** × 32 | **MLP 512,256,128 + tanh** | **512,512,256,128** | 单 dataclass + tyro |
| 14 | manipulation | **2048×5** SiLU + LN | 8 / 2048 | 4096 × **16** | 2048,1024,512,256 ELU | 256,128 ELU | frozen dataclass + argparse |
| 15 | h1_2 | 2048×3 SiLU + 终止头 64,64 | 8 / 2048×10 | 4096 × 32 | 2048,1024,512,256 SiLU→z32→27 | 256,128 SiLU | dataclass + Isaac configclass |
| 16 | locomujoco | **ODE 向量场**（3×2048） | 8 / **512**×10 | 4096 × 32 | 2048,1024,512,256 | 256,128 | argparse + `--idea` 变体 |
| 17 | rtx5080 | 2048×3 SiLU + 终止头 64,64 | 8 / 2048×10 | 4096 × 32 | 2048,1024,512,256 SiLU→z32→29 | 256,128 SiLU | 嵌套 dataclass + tyro |
| 18 | T2 | 2048×3 SiLU + 终止头 | 8 / 2048×10 | 4096 × 32 | 2048,1024,512,256 SiLU→z32→29 | 256,128 SiLU | 嵌套 dataclass + tyro |
| 19 | SuperMimicKit | 2048×3 SiLU + LN | 8 / 2048×10 | 4096 × 32 | 2048,1024,512,256 ELU→z32→29 | 256,128 ELU | **机型 JSON + argparse** |
| **本** | **go2_supermimic** | **1024×3 SiLU，无 LN** | **8 / 2048×2** | **8192 × 32** | **MoE 8 专家×32（骨干 512,256,256）→ 头 512,256,128 ELU** | 512,256,128 ELU | **dataclass 树 + JSON 覆盖 + CLI** |

### 1.3 优化、预算与训练循环

| # | 仓库 | envs | 预热 / 回放 | lr / γ / λ | 裁剪 | PPO/教师 | 特有机制 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | vibed | 4096 | 64 步 / 1000 | 2e-4 / .99 / .95 | norm 1.0 + elem 0.1 | 无 | 单卡 1.47 h 自报复现 |
| 2 | Reproduced | 4096 | 40 步 / 1000 | 2e-4 / .99 / .95 | norm 1.0 + elem 0.1 | 无 | 分布式 all-reduce、BONES-SEED 全量 |
| 3 | Uni | 4096 | 64 步 / 1000 | 2e-4 / .99 / .95 | 同上 | 无 | 三后端切换、确定性全量评测 |
| 4 | **go2-supermimic** | **4096** | **300 轮 / 1000** | 2e-4 / .99 / .95 | norm 1.0 + elem 0.1 | 无（外部策略只驱动采集） | OU 探索 25%、稀有归档 100k、塌缩护栏 |
| 5 | morphmimic | 4096 | 100 步 / 1000 | 2e-4 / .99 / .95 | 同上 | 无 | 形态条件 WM、5 点稀疏目标 |
| 6 | rethinking | 4096 | 50 轮 lr 预热 / 1000 | **pi 5e-4** / .99 / .95 | norm 1.0 | 无 | critic 集成、Muon 可选、goal 编码消融 |
| 7 | Dyn | 4096 | 64 步 / 1000 | 2e-4 / .99 / .95 | 同上 | 无 | 学习终止头、高动态动作 |
| 8 | MuJoCo | 4096（CLI 16） | 64 步 + 500 轮 / 1000 | 2e-4 / .99 / .95 | 同上 | 无 | CPU 原生 MuJoCo、100STYLE OOD |
| 9 | allmimic | 4096 | 64 步 / 1000 | 2e-4 / .99 / .95 | pi_clip 1.0 + elem 0.1 | **另有 PPO 基线** | value clip、Muon、Penn 检测 |
| 10 | trackerlab | 4096（预设 8096） | 64 步 + 500 轮 / 1000 | 2e-4 / .99 / .95 | 同上 | 仅 go2 变体有蒸馏 | Isaac/PhysX 全链路、多本体 |
| 11 | colab | 4096 | 同 #1 | 同 #1 | 同 #1 | 无 | Colab 打包 + 逐里程碑日志 |
| 12 | InstinctMJ | 2048 | 64 步 + 500 轮 | 2e-4 / .99 / .95 | 同上 | 无 | 深度潜变量进 WM、失败头、高程图 |
| 13 | Dex | **1024** | 64 步 + 500 轮 / 1000 | 2e-4 / .99 / .95 | 同上（actor AdamW） | 可选 PPO 基线 | 接触 BCE 头、holdout 环境 |
| 14 | manipulation | 4096 | 64 步 + 1000 轮 / 1000 | 2e-4 / .99 / .95 | 同上 | 无 | 视界 16、WM 5 层、5% 留出 |
| 15 | h1_2 | 4096 | 40 步 + 500 轮 / 1000 | 2e-4 / .99 / .95 | 同上 | 无 | v0/v1 双配方、URDF-FK 解析头 |
| 16 | locomujoco | **256**（CLI） | 64 步 / 512 | 2e-4 / .99 / .95 | 同上 | 无 | **ODE 世界模型**、反饱和正则、JAX |
| 17 | rtx5080 | 4096 | 40 步 / 1000 | 2e-4 / .99 / .95 | 同上 | 无 | 单卡 5080 适配 |
| 18 | T2 | 4096 | 40 步 / 1000 | 2e-4 / .99 / .95 | 同上 | 无 | 新本体 + 腕部负载 DR、T2 有 0–8 步延迟 |
| 19 | SuperMimicKit | 4096 | 64 步 + 500 轮 / 1000 | 2e-4 / .99 / .95 | 同上 | 无 | 三本体共享、FP32 ONNX 导出 |
| **本** | **go2_supermimic** | **4096（实测 1024）** | **300 轮 / 1000** | 2e-4 / .99 / .95 | norm 1.0 + elem 0.1 | **无** | **OU 探索 25%、稀有归档 100k、塌缩护栏、微批反传** |

## 2. 世界模型：状态定义与损失

这是整个方法中差异最大、也最影响收敛的部分。

| 仓库 | WM 状态内容 | 状态维 | 输出 | 损失形式 |
| --- | --- | --- | --- | --- |
| vibed / colab | 根高度、6D 倾斜、体坐标速度、关节角/速 | 71 | 帧局部增量 | 平方误差 + 终止 BCE |
| Reproduced | canonical 状态 | 71 | delta 70 | 归一化平方误差 + 终止 BCE |
| Uni | canonical 状态 | 71 | delta 70 | 同上 |
| **go2-supermimic** | 高度 1 + 重力 3 + 航向系线速 3 + 体坐标角速 3 + 关节 24 + **髋/膝/足 36** | **70**（+3 平面位姿） | 状态增量 + 位姿增量 | 平方误差（无终止头） |
| morphmimic | canonical 状态 + 14 body 高度 | 71+ | delta | 平方误差 + hazard BCE |
| rethinking | canonical 状态 | 71 | delta | 带规范化权重的平方误差 + 终止 BCE |
| Dyn | canonical 状态 | 71 | delta | 平方误差 + 终止 logit BCE |
| MuJoCo | canonical 状态 | 71 | delta | 平方误差 |
| allmimic | canonical 状态 | 71 | delta | 平方误差 + 可选残差项 |
| trackerlab | canonical 状态 | 71 | delta | 平方误差 |
| InstinctMJ | 状态 + **深度潜变量 z** | 71+z | delta + 接触 + z | 平方 + latent + 重建 |
| Dex | 动力学维 + 接触维 | 未注明 | delta + 接触 | 平方 + 接触 BCE + 失败 BCE |
| manipulation | 关节 + 物体位姿 | 未注明 | delta | 平方误差 |
| h1_2 | canonical 状态 | 71 | delta + 终止 logit | 平方 + BCE |
| locomujoco | **广义加速度**（25 维）+ 终止 logit | 267 输入 | 加速度 | 物理状态误差 + BCE |
| rtx5080 | canonical 状态 | 71 | delta + 终止 logit | 平方 + BCE |
| T2 | canonical 状态 | 71 | delta + 终止 logit | 平方 + BCE |
| SuperMimicKit | 根 + 关节最小状态 | 未注明 | delta | 平方 + 终止 |
| **本** | 70 维物理 + 动作 12 + 上动作 12 + 上下文 16 + 扫描 77 | **输入 187，输出 73** | 状态增量 + 平面位姿 | 平方误差（任务侧解析奖励/终止） |

**关键分歧（源码事实）**：

- **是否给世界模型足端/肢体点**：只有 #4 显式把髋、膝、足坐标（36 维）放进状态，目的是让想象能精确判定触地/腾空和与障碍物的碰撞。多数实现靠解析 FK 从关节角重算，效果等价但计算在图内。
- **是否在 WM 里学终止**：#15/#17/#18/#13 用独立 BCE 头；#4 与本项目把它留给任务侧的解析判据；#5 用 4 参数 hazard 头。
- **是否在 WM 里学奖励**：**没有一个实现学奖励**，全部用解析奖励穿过可微 WM。#13 的 `reward` 也是解析式，这与本项目一致。

## 3. 逐仓库剖析

### 3.1 zita-ch/supermimic_vibed（最早的独立复现）

- **底座**：Isaac Lab 3 kitless + Newton/MuJoCo-Warp，不用 Kit/PhysX。约 1650 行，无配置系统，超参就是 argparse 默认值。
- **观测**：99 维 = 关节位置偏移 29 + 关节速度 29 + 体坐标角速度 3 + 投影重力 3 + 上一步动作 29 + **相对参考根朝向 6D**。噪声半宽 0.01/0.5/0.2/0.05/0/0.05。`OBS_SCALE` 常量定义了但**从未被使用**（死代码），实际归一化靠 `RunningNorm`。
- **目标**：10 帧 × 73 维 = 730。每帧含当前根 yaw 系下的相对位移 3、6D 旋转、相对线/角速度、关节 29、关节速度×0.1。越界时姿态 hold 在末帧，但速度只在记录区间内保留真值。
- **动作**：绝对关节位置 `default_q + 编码器偏置 + α·a`，`α = 0.25·effort/Kp`（BeyondMimic 约定）。**无动作延迟、无执行噪声**，采集时额外加 N(0,0.1²) 再夹 ±20。
- **终止**：硬终止 `|Δz| > 0.25 m` 或 `|ΔR22| > 0.8`；想象用双 sigmoid 软化（温度 0.02/0.04）。
- **评测**：40 条各跑一次，摩擦固定 1.0、无噪声无 DR；失败=触发训练终止；归一化回报 = 回报 / 最大可达值。
- **工程**：`torch.compile` 想象单步，bf16 MLP，检查点每 2000 迭代，单卡。
- **可借鉴**：全部 MDP 头解析化，只依赖一份可微 torch FK（`kinematics.py`），FK 与仿真对齐到 1e-6 量级，因此 sim 与想象天然一致。
- **该警惕**：README 明确"1 小时 AI 辅助速写、未经评审"，死代码已出现；多数后续仓库以它为上游，**继承其缺陷**。

### 3.2 CuboiLeo/SuperMimic-Reproduced（文档最完整的复现）

- **底座**：mjlab + MuJoCo Warp（固定 commit）。
- **观测**：93 维（关节位置 29、速度 29、角速度 3、重力 3、上动作 29），10 帧 **term-major** 展平 = 930。目标单独成组 680 维（anchor_obs 时 689）。注释明确解释了为什么不能把 target 放进同一组历史——否则会被堆成 6800 维。
- **终止**：`bad_root_height`（阈值 0.25）与 `bad_root_tilt`（阈值 0.8，≈78°）两个独立项 + clip 末超时。
- **DR**：足部摩擦 U[0.3,1.2]（每回合重采样、所有足共享）、编码器偏置 ±0.01、COM 仅启动时采样；**无推力**（论文明确去掉）。
- **工程**：`torch.compile(dynamic=False)` 用于 WM 单步损失与想象单步；不用 bf16，matmul 走 TF32；多卡 torchrun 显式 all-reduce，lr×√world。
- **可借鉴**：UTD 用小数累加器避免取整偏差；**预热计数必须相对续训点而不是绝对迭代**（这是很多复现出错的地方）。
- **该警惕**：续训不恢复 replay buffer，需重新预热。

### 3.3 Motphys/Uni-SuperMimic（工程化移植）

- **底座**：UniLab，同时支持 mjwarp / 原生 MuJoCo / Motrix 三个后端。
- **观测/目标**：99 维（多 6 维相对朝向），目标 730 维，**统一表达在当前参考帧的 yaw 系**而非相对机器人。
- **DR**：摩擦、COM、关节偏置全部写进 **33 维 context xi**，因此"策略看到的 xi 与实际物理参数严格一致"。
- **奖励**：单一 `tracking` 项权重 1.0，env 层强制 `scale_rewards_by_dt=False` 并在构造时报错，防止二次乘 dt。
- **工程**：bf16 只包 MLP 前向、几何保持 fp32；bf16 时**关闭** matmul TF32（ROCm 上 rocBLAS 选型开销）；有测试断言该行为。
- **可借鉴**：把整条 MDP 做成无状态、可微、env 与 learner 共用的纯张量实现，软/硬终止只差一个 sigmoid 温度。

### 3.4 Zhengtianjin/go2-supermimic（本项目的主要参考）

- **任务**：Go2 追逐/闪避，50 Hz 自采仿真转移，**没有动作捕捉数据**。
- **WM 状态 70 维**：高度 1 + 重力 3 + 航向系线速 3 + 体坐标角速 3 + 关节 24 + **髋/膝/足相对坐标 36**。平面位姿 (x,y,yaw) 不入状态，模型预测增量保持平移/yaw 不变性。
- **上下文 4 维**：足摩擦 + COM 偏移；**不含**编码器偏置（本项目补到 16 维）。
- **奖励**：18~19 项，含追逐 3 项（1.5/15/0.5）、闪避 4 项（-2/-100/-1.5）、共享项（upright 1.0、tilt_rate -0.05、crouch -30、hit -100 等），想象奖励乘 dt=0.02。
- **终止**：仅超时与倾角 70°；想象里 `sigmoid((tilt-70°)/5°)`，命中改为 `sigmoid(-clearance/0.03)` 的平滑核。
- **循环**：每次迭代采集 1 步 → WM 更新 2 次（**UTD=2**，本项目继承）→ 想象 H=32。前 300 轮只训 WM。
- **训练细节**：OU 探索 25% 环境、稀有事件归档 10 万窗口（保留概率 0.02）、塌缩护栏（连续两次窗口摔倒激增则回滚 actor/critic 并减半 lr）。
- **工程**：fp32 matmul、`torch.compile` 想象单步、单卡、无 AMP、无导出；buffer 存 fp16。
- **该警惕**：世界模型误差会被策略利用，**必须有崩溃回滚与 actor 学习率斜坡**——这两点是本项目直接继承的。

### 3.5 lzyang2000/morphmimic（形态条件化）

- **核心改动**：一个世界模型与策略共享于 G1 / X2 Ultra / T2 三种形态，观测/目标带形态条件。
- **WM 状态**：canonical 状态 + 14 body 高度；`wm_frame` 可选 `root`/`heading` 两种规范化坐标系。
- **终止**：`hazard` / `learned` / `soft` 三选一。默认 hazard 是 4 参数单调头，由两个训练终止的裕度构造。
- **可借鉴**：`reset_norm` 的 `remaining` vs `actual` 之分——论文未定义归一化分母，作者显式指出这一歧义并给了两种实现，这种"标注论文空白"的做法值得学。
- **该警惕**：残差尺度 `residual_scale` 的估计方式影响续训兼容性，切换会破坏检查点。

### 3.6 btx0424/rethinking-supermimic（目标条件化方式消融）

- **观测**：228 维（多出锚点朝向 6、锚点位置 3、14 body 位置 42、朝向 84），10 帧 = 2280 维。**显式加入 body 位置/朝向误差**，注释说明"这是奖励 body 的跟踪误差，不是它们的世界位姿"。
- **策略**：`split`（tanh 编码-解码）/ `trunk`（单 MLP）/ `film` 三种目标注入方式可选。
- **critic**：**集成 2 头**，GAE 与价值损失逐头计算，策略 bootstrap 用均值。
- **其他**：`pi_lr 5e-4`（比其余高一倍）、奖励风格 `gaussian`/`cauchy`/`gaussian_linear` 可选。
- **可借鉴**：这是唯一系统性对比"目标怎么进网络"的实现，对判断 FiLM vs 拼接有直接参考价值。

### 3.7 Baojch/SuperMimic_Dyn（高动态动作）

- **任务**：前/后/侧空翻、侧手翻、爬行、跌倒起身。
- **终止**：**学习终止头 64×64**，想象中 `d = σ(logit)` 逐歩累乘存活率。
- **数据**：BONES-SEED v004 + LAFAN1 的 G1 重定向 CSV，120/30 Hz 统一重采样到 50 Hz，打包成 fp16 紧凑库。
- **评测**：四套（完整验证 3966、val512、train313、hard164），从片段首帧起、每片执行 N−1 次转移。
- **工程**：全网 fp32 + TF32，每卡 4096 环境约 23.17 GiB，双卡显式 all-reduce。
- **该警惕**：DR 只有摩擦/编码器偏置/COM，无推力与延迟，高动态的 sim2real 偏窄。

### 3.8 panhainan2020/SuperMimic-MuJoCo（脱离 Isaac）

- **底座**：原生 MuJoCo CPU 或 MJLab 双后端，可完全不用 GPU 训练。
- **观测**：`legacy99`（含 6D 参考朝向）与 `paper93`（去掉该 6 维）两套可选。
- **奖励**：6 项指数核总权重 5.0，另加动作率 −0.1、软限位 −2；整项乘 dt，单步上限 0.1。
- **评测**：LAFAN 40 条 + **100STYLE 3573 条分布外**，均从首帧跑一次。
- **工程**：TF32；bf16 仅 MLP autocast；`--compile` 编译 5 个热函数；检查点 fsync+replace **原子保存**，`ckpt_*.pt` 只含 pi/wm。
- **该警惕**：续训不恢复回放，需重新预热；作者明确了这一点。

### 3.9 Shimingyi/allmimic（残差动作与 value clip）

- **动作**：默认绝对式，`res` 模式为 `q_ref(t+1) + res_scale·action_scale·a`（**残差叠加在参考上**）。
- **两个真实教训（作者标注 #23/#29）**：
  - **#23 确定性部署塌缩**：世界模型只在有噪动作数据上训练，而想象是确定性的 → 加了 `collect_det_frac`（部分环境不加噪采集）与 `img_action_noise`。
  - **#29 value clip**：长跑 5–7 小时后 critic 过估计导致发散，必须把用作 lambda-return 与 bootstrap 的价值夹到 `±max_step_reward/(1-gamma)`。
- **工程**：`bptt_checkpoint` 梯度重算把峰值从 14.5 GB 降到 7.1 GB；torchrun 多卡 lr×√world。
- **对本项目的意义**：这两条是关于"世界模型被策略利用"的一手经验，比任何调参建议都具体。

### 3.10 Renforce-Dynamics/trackerlab-supermimic（Isaac/PhysX 全链路）

- **底座**：Isaac Lab 2.3.2 / PhysX，覆盖 G1/Go2/A3/T1/Bumi/Pi+ 六种本体。
- **观测**：`relative_ori99` 与 `paper_obs93` 两套。
- **延迟**：**无**观测/动作延迟（代码中不存在）。
- **工程**：tf32 + bf16 仅作用 MLP，`torch.compile` 包五个函数；CSV 记 8 项奖励与显存峰值；**支持 TorchScript 导出**，这是少数有导出链的实现。
- **该警惕（作者自陈）**：93 维观测缺相对朝向会让全局朝向不可观测，**必须显式补 6 维**。

### 3.11 HighTorque-Dev/supermimic-colab（Colab 打包）

- **本质**：vibed 的 Docker/Colab 化封装，附完整里程碑日志与 14k 步 A100 训练记录。
- **价值**：它是唯一给出"逐时刻里程碑 + 失败分析"的仓库（例如 it 6000 时定位到 `fallAndGetUp1_subject5` 是真实跟踪失败而非判据误报）。
- **该警惕**：代码与 vibed 同源，继承同样的问题（`obs_scale` 死代码等）。

### 3.12 project-instinct/InstinctMJ（supermimic 分支：视觉）

- **观测**：1990 维 = 目标 670 + **深度 1×18×32 = 576** + 8 步本体 744。
- **关键设计**：深度编码器输出**潜变量**进世界模型状态，想象里滚动潜变量而非渲染图像，因此不需要渲染器，策略网络也不用改。
- **终止**：比其余多两条——四末端高差 >0.25 m、非法接触力 >500 N。
- **DR**：摩擦 (1.25,2.0)/(1.2,1.8)/(0,0.5)、关节默认角 ±0.01、COM ±0.05、**kp ×0.8–1.2**；推力关闭。
- **工程**：bf16 MLP、`torch.compile`、可选梯度检查点，峰值 15.5 GB。
- **该警惕**：管理器姿态会慢一个子步，解析奖励必须逐项对齐 env。

### 3.13 CumulusAlpha/Dex-Supermimic（灵巧手转笔）

- **观测**：155 维单帧 = motion 89 + 关节位置 20 + 速度 20 + 笔轴端点 6 + 上次动作 20。**没有多帧历史**。
- **动作**：在参考关节角上叠加**残差**，scale 0.5、残差限幅 ±0.6、**EMA α=0.5**，再减编码器偏置。
- **终止**：硬终止含数值发散、笔位置误差 >0.05 m、轴方向误差 ≥60°；想象用两个 sigmoid 的合成。
- **DR**：摩擦对数均匀 0.35–1.4、扭转 0.5–2.0×、编码器偏置 ±0.01、**笔质心 ±0.04 m 与质量 0.4–1.6×**、指尖随机外力 0–4 g。
- **工程**：bf16 autocast + TF32、`torch.compile`、单卡、**ONNX opset 17 导出**、含 RNG 与课程状态的原子保存。
- **该警惕（作者自陈）**：想象终止是软判据、真机是硬阈值 + timeout，两者语义不完全一致。

### 3.14 emoPointer/supermimic_manipulation（机械臂操作）

- **观测**：195 维（39×5），mug_hang 210 维（42×5）。**无独立 goal 轨迹**，目标以 target-in-EE 相对位姿进每帧观测。
- **动作**：7 维（6 相对位姿 + 1 夹爪），raw 限幅 ±12，scale 0.02 m / 0.2 rad，每步范数限 0.075 m 与 0.4 rad。
- **奖励**：核是 `exp(-10d)` 与 `tanh(10d)` 而非 std 形式，总奖励乘 dt。
- **终止**：原生 success 判据保持 0.5 s；想象端用温度 0.01 的 sigmoid 直通软门，**success 梯度缩到 0.02**。
- **工程**：4096 世界约 27 GB 显存；`torch.compile fullgraph` 编译 WM loss 与想象单步。
- **该警惕（作者自陈）**：想象 success 只是几何近似，`surrogate_only` 率高即策略在钻代理空子。

### 3.15 JzKosann/supermimic-isaaclab-h1_2（H1-2 + 双配方）

- **观测**：870 维 = 87×10；目标 640 维（只有根高度，无根 xy 与完整姿态；Anchor 版另加 9 维）。
- **终止**：硬终止两条，超时另算；想象软化温度 0.02/0.04。
- **工程**：TF32 + compile 默认开（策略更新快约 4.5×），**不用 bf16**；FK 按 65536 帧分块省显存；多卡 torchrun 手写 allreduce。
- **该警惕（作者自陈）**：续训不恢复回放；奖励参考锚定与命令更新有**一帧错位**这类静默不一致。

### 3.16 Alchedata/supermimic-locomujoco（ODE 世界模型）

- **核心改动**：世界模型不是离散 delta MLP，而是**连续时间向量场**，预测广义加速度，用 Euler/Midpoint 积分，再用可微 FK 重建 body 特征。
- **观测**：每帧 217 维（含 168 维相对 site 位姿），10 帧 = 2170，**观测无噪声，噪声只加在动作上**。
- **奖励**：原生 LocoMuJoCo `MimicReward` 5 项，逐控制步累加、**不乘 dt**。
- **DR**：**完全没有**——摩擦、偏置、COM、推力、延迟全无。
- **工程**：JAX 实现（`jax.lax.scan` 做窗口前推），matmul 精度 highest，Python 梯度累积。
- **可借鉴**：160 段固定起点（40 动作 × 4 起点）+ provenance 校验，可复现性做得最细。
- **该警惕**：零域随机化，sim2real 泛化没有任何论证。

### 3.17 rzc156/supermimic-rtx5080（单卡适配）

- **与 Reproduced 的关系**：代码结构基本一致，主要差异是单卡 RTX 5080 的显存适配（fp16 紧凑库 + 2048 env）。
- **终止**：学习终止头 `64,64`，想象里 `c_t = ∏(1-d̂)`。
- **该警惕（作者自陈）**：纯本体观测在长 clip 上有根位姿漂移，加锚点后回报从 0.56 升到 0.90——**这是"本体观测够不够"的直接数据点**，对本项目的盲性设计有参考价值。

### 3.18 BarryLi159159/T2_supermimic（新本体 + 延迟）

- **本体**：Booster T2（v3 deploy contract，bare + Sharpa 手），保留 G1 配方。
- **延迟**：**T2 每回合采 0–8 物理步（0–40 ms）执行延迟，G1 无**。这是 19 个仓库里少数把延迟建模进 DR 的。
- **其他**：`loco` 子任务（速度指令），`rsi` 参考状态初始化，`wrist_payload` 腕部负载 0–1.5 kg。
- **工程**：4096 环境 + replay 约 2.9 GB，全量训练约 22–24 GB；`torch.compile` 用于 WM 损失与 ImagStep；ONNX + meta 导出；torchrun 多卡。
- **对本项目的意义**：它是唯一同时具备"新本体 + 延迟 + 速度任务"的实现，其 `LocoCfg`（命令重采样 3–6 s、40% 站立环境、课程 50k–250k）可与本项目命令设计对照。

### 3.19 OMEGA-i/SuperMimicKit（三本体共享 + 最小状态）

- **核心设计**：世界模型只滚动**根 + 关节最小状态**，body 位姿每步由解析 FK 重算，减少状态维。
- **观测**：99 维（含根相对目标 rot6 6 维），目标 730 维。
- **工程**：FP32 存储、可选 TF32/BF16 autocast、`torch.compile + CUDA graphs`、fused Adam、**FP32 ONNX 导出**、单进程单卡。
- **可借鉴**：可微 continuation 做软终止；把 DR 参数作为 context 输入。
- **该警惕（作者自陈）**：论文未说明处自定的细节（40 片段划分、采样器、场景设置）会造成复现偏差。

## 4. 横向专题对比

### 4.1 配置组织方式

| 方式 | 仓库 | 优点 | 代价 |
| --- | --- | --- | --- |
| 纯 argparse | #8、#11、#16、#19 | 零依赖，最直白 | 无类型检查、无嵌套 |
| 单 dataclass + tyro | #2、#5、#7、#13、#17、#18 | 类型安全、CLI 自动生成、可 `asdict` 落盘 | 每个字段一个 flag，组合爆炸 |
| 嵌套 dataclass 树 | #2、#7、#17、#18、**本** | 分组清晰，支持 JSON/CLI 双层覆盖 | 需处理校验与旧快照迁移 |
| Hydra YAML | #3、#6 | defaults 组合与 override 适合消融 | 引入 omegaconf，类型弱于 dataclass |
| argparse + JSON 预设 | #10、#19 | 一份 JSON 一个配方 | 键名易与代码漂移，缺 schema 校验 |

**本项目（事实）**：`SuperMimicConfig`（算法侧）+ `Go2ReferenceCfg`（任务侧）两层 dataclass，`--task-config <json>` 覆盖，`--num-envs/--seed` 优先级最高，运行目录写 `task_config.json` 供续训/播放读取。相比 Hydra 缺少 YAML 组合能力；相比 tyro 不把网络/预算字段逐个暴露为 flag——这是让任务侧配置成为唯一数值来源的有意取舍。

### 4.2 观测设计的分歧

| 设计点 | 给 | 不给 |
| --- | --- | --- |
| 相对参考朝向 6D | #1/#3/#8/#10/#11/#19（99 维） | #2/#7/#9/#17/#18（93 维） |
| 线速度 | #6（228 维含 base_lin_vel） | 多数不含（论文 Table A.1 明确不给） |
| 深度/视觉 | #12（576 维潜变量） | 其余 |
| 全 body 位姿 | #6（126 维）、#16（168 维） | 多数 |
| 多帧历史 | 多数 10 帧 | #13 单帧、**本 5 帧**、#4 5 帧 |

**#17、#18 的作者自陈数据点**：只用 93 维本体观测时根部位姿漂移严重（回报 0.56），补锚点相对观测后到 0.90。这不直接适用于四足速度跟踪（本项目没有"参考根"可跟），但说明**观测里给不给位姿类信息对"能否约束全局漂移"是决定性的**。

### 4.3 终止与生存概率

| 形式 | 仓库 | 特点 |
| --- | --- | --- |
| 解析软化（双 sigmoid） | #1、#2、#3、#6、#8、#9、#10、#11、#15、#19、**#4**、**本** | 无额外参数，但阈值必须手工可微化 |
| 学习终止头（64×64 BCE） | #7、#13、#15、#17、#18 | 能学到非线性失效面，多一组超参 |
| 4 参数 hazard 头 | #5 | 参数少，单调性有保障 |
| 无（仅物理失效） | #16 | 依赖数值发散判据 |

本项目的选择（解析软化，70° 倾倒 + 5° 宽度）与 #4 一致，已有单元测试覆盖。

### 4.4 域随机化覆盖

| 项 | 覆盖仓库 | 未覆盖 |
| --- | --- | --- |
| 足部摩擦 | #1–#11、#13、#15、#17–#19、**本** | #12（另用大范围）、#14（改用质量）、#16（全无） |
| 编码器偏置 | #1–#4、#7–#11、#13、#15、#17–#19、**本** | #5、#12、#14、#16 |
| COM 偏移 | #1–#11、#13、#15、#17–#19、**本** | #12（用 kp 缩放替代）、#14、#16 |
| 动作延迟 | **仅 #18（T2，0–8 步）**与 **#4/本（固定 0–2 步）** | 其余全部 |
| 外力 push | 全部关闭（#13 有指尖 0–4 g） | — |
| 质量/惯量 | **仅 #14（0.7–1.3×）**、#13（笔质量 0.4–1.6×） | 其余 |

**结论（源码事实）**：动作延迟在 19 个实现里几乎无人建模，#18 是唯一做成随机化的；本项目固定 0–2 步且由世界模型隐式近似——这一点上本项目比多数复现更接近真实，但不如 #18 显式。

### 4.5 数据管线

| 数据源 | 仓库 | 帧率处理 |
| --- | --- | --- |
| LAFAN1 G1 重定向（HF `lvhaidong/...`，30 fps） | #1–#4、#8–#11、#15–#19 | 线性插值 + 四元数 slerp → 50 Hz |
| BONES-SEED（120 fps，厘米/度） | #2、#5、#7、#9、#17、#18 | 单位换算 + 重采样 → 50 Hz |
| 100STYLE（OOD 评测） | #8 | 同 LAFAN1 |
| 自采仿真转移 | **#4、#14** | 无数据文件 |
| 程序化 reset bank | #14 | 无帧率概念 |

**没有一个仓库在本仓库内做 SMPL→机器人重定向**，全部依赖上游已重定向数据（#19 例外，它有 BVH→SMPL-X→UMR 的完整链）。本项目更彻底：**完全没有参考动作数据**。

### 4.6 评测协议

| 仓库 | 片段数 | 起点 | 条件 | 成功判据 |
| --- | --- | --- | --- | --- |
| vibed / colab / Reproduced / Uni / MuJoCo / allmimic / trackerlab / rtx5080 / T2 | 全库（40 / 129k / 3573） | 首帧 | 摩擦 1.0、无噪声、无 DR、确定性 | 未触发训练终止 |
| Dyn | 3966 / 512 / 313 / 164 四套 | 片段首帧 | 同上 | 同上 |
| Dex | 64 env × 3 种子 × 2000 步 | 课程固定 1 | 关 auto_reset | goals_reached / cage_drop |
| manipulation | 256（64×4 类） | reset bank | 同上 | 原生 success 判据 |
| locomujoco | 160（40×4 起点） | 固定种子 20261022 | 确定性 | 满 horizon 且无终止 |
| InstinctMJ | 每动作 ≥3 段 | t=0 | 上限 30 s | dataset_exhausted |
| **本** | **4 组固定命令 × 多 env** | **统一种子重置** | **关噪声与 DR** | **跟踪误差/跌倒/存活** |

**本项目的评测与所有外部仓库不可直接比较**：它们是"跑完整条参考动作不摔"，本项目是"按固定速度指令跑 20 s 的跟踪误差"。这一点在引用任何外部回报数字时必须说明。

## 5. 与本项目的关系

### 5.1 可直接对照

**#4 `Zhengtianjin/go2-supermimic`** 是同机器人、同仿真底座、同算法骨架的唯一实现。本项目的训练预算（4096 env、1000 回放、wm_batch 2048×2、imag 8192×32、300 轮预热）、OU 探索、稀有归档、塌缩护栏、actor lr 斜坡都以它为基准。差异见 [Go2 SuperMimic 说明](go2_supermimic.md) 与 [设计记录](design/go2_blind_supermimic.md)。

### 5.2 旁证价值

- **#13、#14**：证明"去掉参考动作、actor 只吃本体历史"在同框架下可训练，但任务是操作/重定向，状态与奖励完全不同，**不能**推断四足盲速度跟踪的收敛性。
- **#17、#18**：给出"本体观测够不够"的实测对照（0.56 vs 0.90），说明全局漂移对观测极其敏感——本项目没有参考轨迹，漂移只能靠速度命令与地形边界约束。
- **#9、#13**：一手记录了"世界模型被策略利用"的两种具体形态（确定性部署塌缩、critic 过估计），比抽象警告有用。

### 5.3 不建议直接移植

| 做法 | 出处 | 不采用的理由 |
| --- | --- | --- |
| Hydra YAML | #3、#6 | 与本地"任务侧单一数值来源"冲突，且 [配置集中化设计](design/go2_config_centralization.md) 已定方案 |
| 参考动作编码器 | 17 个仓库 | 破坏本项目的盲性边界（actor 不含线速度/地形/摩擦），是用户确认的设计 |
| 学习终止头 | #5、#7、#15、#17、#18 | 本项目解析判据已有测试覆盖，替换需论文或参考项目依据 |
| 深度潜变量 | #12 | 需要深度相机与渲染，本项目是盲策略 |
| ODE 世界模型 | #16 | 改动动力学参数化，属于独立研究方向 |

### 5.4 值得单独实验的（推断／待验证假设）

| 假设 | 依据 | 最小验证方法 | 预期代价 |
| --- | --- | --- | --- |
| `2048×3` WM 优于当前 `1024×3` | 18/19 个仓库取 2048 | 固定种子与样本预算，各跑 1000 轮，比较 `wm_err8` 与 `raw track_linear` | 显存约 ×3，速度约 ×0.6 |
| critic 集成（2 头）降低 bootstrap 过估计 | #6 | 仅加第二头，其余不变，看 `v_loss` 与回合长度 | 显存 +1 头，改动局部 |
| 学习终止头优于解析软化 | #7、#17、#18 | 需先补设计文档；解析判据已有测试，替换前须说明依据 | 中 |
| value clip 防止长跑发散 | #9 的 #29 教训 | 在当前 critic 上加 `±max_step_reward/(1-γ)` 夹取，跑 >3000 轮看 `v_loss` | 低 |
| 显式延迟随机化 | #18 | 现为固定 0–2 步，可扩到 0–4 步并测 sim2real | 低，但需先确认参考实现口径 |

## 6. 检索与验证边界

**已执行**：
- GitHub 仓库检索 API（`supermimic`，75 条命中），按 Star 与相关性选 19 个；逐个下载 main 分支 tarball 快照（`InstinctMJ` 取 `supermimic` 分支），实际阅读配置、网络、WM、rollout、训练循环、数据与评测文件。
- 两轮子代理并行阅读 + 人工抽查：`SuperMimic-Reproduced` 的 `PolicyCfg`/`WMTrainerCfg`/`PolicyTrainerCfg`、`Uni-SuperMimic` 的 Hydra `config.yaml`、`T2_supermimic` 的 `PolicyCfg`、`trackerlab` 的 argparse 默认值、`go2-supermimic` 的 `TrainCfg` 与 `RewardWeights`、`Dex` 的 `Actor`/`Critic` 均与表格一致。

**未执行**：
- 未运行任何外部仓库的训练或评测，**表中所有成绩均未采信**。
- `web_search` 工具因搜索端点鉴权失败（HTTP 401）不可用，改用 GitHub codeload 与检索 API。

**口径限制**：
- 数值为代码默认值，作者实跑命令可能不同（#8 CLI 默认 16 环境但发布权重用 4096；#17 实跑 2048 环境 / 2048 想象批量）。
- 本项目"实测 1024"来自运行快照 `task_config.json`，不是默认值（默认 4096）。
- 部分仓库无 `.git` 目录，无法给出提交号。
- `supermimic` 命名之外的"同思路不同名"项目未覆盖；75 条命中中其余 ~56 个未逐一下载。
