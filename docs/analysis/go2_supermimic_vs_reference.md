<!-- Copyright (c) 2026-2027 zh -->

# 本项目 Go2 SuperMimic 与参考仓库逐项对比

> 本文下方逐项表记录2026-10-28本轮实施前的工作树，用于解释已存在的差异。本轮用户已确认改为：MoECTS部署actor、原六项速度奖励加HIM步态／腾空与Go2 PPO姿态、HIM heading命令，其余训练／地形／随机化／续训按参考对齐。最新复现对应关系、来源、接口及验证边界统一维护在[设计中的当前实施基准](../design/go2_blind_supermimic.md#当前实施基准参考训练全对齐him-命令与三项奖励2026-10-28)，运行方式见[任务说明](../go2_supermimic.md)。下文的六项奖励、固定5秒命令、连通地形、负载均衡和默认追加续训不再描述本轮实现。

本文把本工作区的 Go2 SuperMimic 任务与上游参考仓库 [Zhengtianjin/go2-supermimic](https://github.com/Zhengtianjin/go2-supermimic) 做逐项对照，覆盖项目定位、MDP 组装、观测、奖励与终止、状态与世界模型、网络、训练排期、地形与域随机化、外部驱动、入口与产物、测试与文档、许可。

对比基线：

| 项 | 本项目 | 参考项目 |
| --- | --- | --- |
| 仓库路径 | `/home/sunteng/projects/quadruped_lab` | `/home/sunteng/projects/go2-supermimic`（另把上游 tarball 解压到 `/tmp/go2src/go2-supermimic-main`，除本地新增 `SETUP_zh.md`、`logs/` 外与本地副本一致） |
| 提交 | `dfee759a635dcf7a1cb3d4ed357d052017d4601a`，工作区有未提交改动（正在把 `agents/supermimic/single/` 平铺为 `agents/supermimic/`） | `d9eee9b3f248e5300e636c433255db2c1c3e086a`（本机为单提交浅克隆，`git log` 只有这一个；`/tmp/go2src` 的 tarball 与之逐文件一致，仅差 `__pycache__`） |
| Python 规模 | 132 文件 / 16,383 行；核心（`src` + `agents` + `scripts` + `tools`）97 文件 / 10,285 行；测试 35 文件 / 6,098 行 | 27 文件 / 4,859 行 |
| 依赖 | MjLab 1.6.0（MuJoCo Warp）、rsl-rl-lib 5.4.2、Python 3.11、uv 锁定 | mjlab ≥ 1.6.0、Python ≥ 3.10、setuptools |

分工：本文只做这**两个仓库**的逐项对照，不重复索引与横向调研——19 个开源实现的横向表见[SuperMimic 实现设计与配置对比](../supermimic_design_comparison.md)，各仓库索引见 [SuperMimic 相关开源项目调研](../supermimic_projects.md)，本项目内部实现细节见 [Go2 SuperMimic 子系统详细分析](go2_supermimic_analysis.md)。

依据标记沿用 [Go2 SuperMimic 子系统详细分析](go2_supermimic_analysis.md#0-阅读约定与依据分层)：**本项目已验证事实**、**论文依据**、**参考项目依据**、**推断／待验证假设**。

行号说明：形如 `config:181` 的引用指本项目 `src/config/go2/go2_reference_supermimic_config.py`，形如 `env/env_cfg.py:133` 的引用指参考仓库的同名文件；`train.py`、`imagine.py`、`world_model.py`、`state.py`、`driver.py` 等简写指参考仓库 `go2_supermimic/supermimic/` 下的同名模块。参考项目「参考源码」均指本机 `/home/sunteng/projects/go2-supermimic@d9eee9b3` 实际读到的代码；其训练与评估未在本机运行，成绩数字一律视为作者自报。

## 0. 一句话结论

两个仓库共用 SuperMimic 的三阶段外循环（真实采集 → 拟合可微世界模型 → 对想象 rollout 做 BPTT），但解决的是**两个不同的任务**：本项目已迁移为盲（纯本体感知）速度跟踪，参考项目是 oracle 真值感知的追球与闪避。差异集中在任务语义层与工程化层，算法骨架几乎逐参数一致。

## 1. 项目定位与任务目标

| 维度 | 本项目 | 参考项目 |
| --- | --- | --- |
| 任务 | 盲速度跟踪：跟踪 `(vx, vy, ωz)` 命令（**本项目已验证**，`docs/go2_supermimic.md:5-7`） | 追逐 + 闪避：追逃跑球、躲抛来的球与横杆（**参考源码**，`README.md:9-38`） |
| 任务 ID | `Unitree-Go2-SuperMimic-Flat` / `-Rough`（`src/tasks/locomotion/go2_supermimic/config/train_cfg.py:11-14`） | `Go2-Chase-Flat` / `-Obstacles` / `-Terrain`（`go2_supermimic/env/__init__.py:5-13`） |
| 感知 | 只有本体 45 维/帧，无目标、无雷达、无高程图；线速度靠 5 帧历史隐式估计（`mdp/reference_task.py:155-162`） | oracle 真值：两个 14 维目标槽 + 16 线雷达（障碍）+ 11×7 高程扫描（地形）（`README.md:140-170`） |
| 命令来源 | 均匀采样 `((-1.5,2.0),(-1.32,1.32),(-3.35,3.35))`，每 5 s 重采样，10% 站立（`config:177-188`） | 无速度指令；由目标生成器的球速/杆速/抛掷种类课程代替（`env/env_cfg.py:141-164`） |
| 与论文关系 | 借用训练算法与论文表 A.6 速度奖励；未复现论文动作模仿接口与参考仓库成绩（`docs/go2_supermimic.md:5-7`） | 论文方法在 Go2 上的独立重实现（作者 Ziyue Zheng, JHU），任务与论文速度接口也不同（`README.md:1-13`） |
| 可复现产物 | 无预训练权重 | `weights/{flat,obstacles,terrain}.pt`，仅 actor（`README.md:55-64`） |

## 2. 环境与 MDP 组装方式

| 维度 | 本项目 | 参考项目 |
| --- | --- | --- |
| 基类 | 从零构造 `ManagerBasedRlEnvCfg`，刻意不复用任何 PPO 任务实现（`go_supermimic_env_cfg.py:49`） | 在 `make_velocity_env_cfg()` 之上改（`env/env_cfg.py:133`） |
| observations | `observations={}`，原生 manager 全空（`go_supermimic_env_cfg.py:154`） | 原生 `actor` 观测组，含噪声与 `history_length=5`（`env/env_cfg.py:168-197`） |
| commands / rewards / curriculum / metrics | 全部 `{}`；任务语义集中在 `Go2ReferenceTaskModel`，真实采集与想象共用一份实现（`go_supermimic_env_cfg.py:117-122`、`mdp/reference_task.py:28`） | 全部使用 mjlab 原生 manager（`TargetCommandCfg`、奖励项、`terrain_levels`）（`env/env_cfg.py:164-244`） |
| terminations | 原生只有 `time_out`；跌倒/出界在任务模型里算（`go_supermimic_env_cfg.py:115-116`、`reference_task.py:213-222`） | 覆盖为 `time_out` + `fell_over`(70°)，删掉 `out_of_terrain_bounds`、`illegal_contact`（`env/env_cfg.py:121-125`） |
| auto_reset | `False`，adapter 自己管重置与课程（`go_supermimic_env_cfg.py:162`） | mjlab 默认 `True` |
| 回合长度 | 训练 20 s，播放可 `--no-terminations`（`config:29`、`scripts/play.py:50-51`） | 训练 20 s，播放 `1e9` 秒（`env/env_cfg.py:242-243`） |
| 控制频率 | 50 Hz（`config:44,110`，即 `0.005 × 4`） | 50 Hz（mjlab 速度任务默认同为 `0.005 × 4`，本机实测） |

## 3. 观测空间与维度

| 项 | 本项目 | 参考项目 |
| --- | --- | --- |
| actor 单帧 | 45：角速度 3 + 重力 3 + 关节角偏移（含编码器偏置）12 + 关节速度 12 + 上一步动作 12 + 速度命令 3（`mdp/reference_task.py:29,155-159`） | flat 70 / obstacles 86 / terrain 147：前 45 项中把「命令 3」换成「目标 28」，再加雷达 16 或高程 77（`README.md:140-153`） |
| 历史后总维 | 5 帧 → 225（`config:30-33`） | 5 帧 → 350 / 430 / 735（`README.md:58-64,153`） |
| 历史展平布局 | frame-major（旧→新），`[t-4 全项][t-3 全项]…`，与 go2_rl_gym 导出权重一致才能热启动 MoE（`agents/supermimic/policy.py:99-118`、`docs/design/go2_blind_supermimic.md:495`） | term-major，`push_history` 对每项单独左移拼接（`supermimic/state.py:262-270`） |
| 观测噪声（半幅） | 角速度 0.2、重力 0.05、关节角 0.01、关节速度 1.5；只对新帧加噪（`mdp/reference_task.py:54-57`、`config:66-78`） | 同前四项，另加雷达 0.03、高程 0.02（`supermimic/state.py:80-87`） |
| 噪声实现位置 | 任务模型 `observe(..., noisy=True)`，训练与想象共用 | mjlab `ObservationTermCfg(noise=...)`，`enable_corruption=not play`（`env/env_cfg.py:168-195`） |
| critic 输入 | 179 = 70 状态 + 12 上一步动作 + 3 命令 + 1 计时 + 77 扫描 + 16 上下文（`config:34-35`、`mdp/reference_task.py:150-153`） | flat 91 / obstacles 107 / terrain 168（`imagine.py:94,190-211`） |
| 特权内容 | 地形扫描 77 + 摩擦/COM/编码器偏置 16 + 命令与计时 | 两球位姿与 active + 闪避几何 + 逃逸时间 + 雷达/高程 + 4 维 sim 上下文（`imagine.py:190-211`） |

## 4. 奖励与终止

| 项 | 本项目 | 参考项目 |
| --- | --- | --- |
| 项数 | 6 项（论文表 A.6）（`mdp/reference_rewards.py:23-32`） | 基础 18 项，`obstacles` 任务再加 `obstacle_contact` 共 19 项（`env/env_cfg.py:201-235`、`imagine.py:35-56`） |
| 明细 | track_lin 2.0、track_ang 2.0、action_rate −0.3、joint_limits −2、torso_sideways_lean −4、body_ang_vel −0.05 | chase_approach 1.5、chase_tag 15、facing 0.5、evade_danger −2、hit −100、evade_anticipate −1.5、outside_arena −2、obstacle_contact −100、upright 1、tilt_rate −0.05、hard_landing −0.5、crouch −30、joint_deviation −0.1、base_height_low −10、terminated −200（地形 −600）、action_rate −0.05、dof_limits −1、torque −2e-5、acc −2.5e-7 |
| 跟踪形式 | 指数核 `exp(-(v-cmd)²/σ²)`，σ 线性 0.5²、角 0.5（`config:193-194`） | 无跟踪项；追球用速度在指向球方向的投影（`env/mdp.py:57-62`） |
| 侧倾惩罚 | 重力 y 分量平方归一化，纯前倾不受罚（`mdp/reference_rewards.py:19-20`） | `tilt_rate_l2` + `upright` |
| 乘 dt | 最终求和后乘一次 0.02（`mdp/reference_task.py:209-211`） | 同（`imagine.py:166`） |
| 环境侧奖励副本 | 没有，env 奖励 manager 为空 | 有；env 奖励只用于日志，想象另写解析版，需手工保持一致（`env/env_cfg.py:198-199`） |
| 硬终止 | 70° 倾角或超出地形边界（margin 1.1 m）（`mdp/reference_task.py:213-215`） | 70° `bad_orientation` + timeout（`env/env_cfg.py:121-125`） |
| 想象软终止 | `sigmoid((tilt−70°)/5°)`，出界直接 1（`mdp/reference_task.py:217-222`） | `fall_prob` 同形式，不含出界项（`imagine.py:168-173`） |
| 可微处理 | 六项均解析 | 命中用 `sigmoid(-clearance/0.03)` 软化，跌倒用概率（`imagine.py:146`） |

## 5. 状态表示与世界模型

| 项 | 本项目 | 参考项目 |
| --- | --- | --- |
| 物理状态 s | 70 维：高度 1、重力 3、航向系线速度 3、机体系角速度 3、关节角 12、关节速度 12、髋/膝/足 3×4×3（x 主序，腿序 FL,FR,RL,RR）（`mdp/reference_state.py:8-17,19`、`state_spec.py:4`） | 同样 70 维同样分量，腿序 FR,FL,RR,RL（`supermimic/state.py:29-44`） |
| 平面位姿 | 另存 `(x,y,yaw)`，预测旧航向系增量（`mdp/reference_state.py:29-42`） | 同（`pose_delta` / `apply_pose_delta`，`supermimic/state.py`） |
| 网络 | SiLU MLP 1024³，输入 187、输出 73；参数 2,366,537（在本地装配后实算） | SiLU MLP 1024³，`out_dim = 70+3 = 73`；参数 2,354,249（按结构实算，`world_model.py:60-62`） |
| 输入构成 | `70 + a12 + a_prev12 + ctx16 + scan77`，ctx 含编码器偏置 12（`mdp/reference_task.py:140-142`） | `70 + a12 + a_prev12 + xi4 + scan77`，xi 只有摩擦 1 + COM 3，`in_dim = 70+24+4+77 = 175`（`world_model.py:60-69`） |
| 残差形式 | 学归一化增量，经任务侧 `integrate`/`advance` 回物理态（`agents/supermimic/world_model.py:64-69`） | 学 `s'−s` 与 pose delta，`s + y[:, :70]`（`world_model.py:75-82`） |
| 整窗损失 | 有效步掩码 + 输出方差归一，误差平方除有效步数（`agents/supermimic/world_model.py:71-85`） | 同思路（`world_model.py:81-97`） |
| 8 步监督的地形 | 用记录状态的地形 relief，物理状态始终自回归，机身高度用预测值（`mdp/reference_task.py:144-148`） | 直接用记录 relief，无该坐标修正（`train.py:474-495`） |
| 32 步想象的地形 | 随预测位姿推进：`terrain_scan` 由状态里的 `(x,y,yaw)` 与预测机身高度算出（`mdp/reference_task.py:124-129,140-142`、`docs/go2_supermimic.md:46`） | 同思路，`relief`/`scan`/`lidar` 都带 `sensor_grad` 回传（`imagine.py:175-188,213-232`） |
| 归一化 | 自写 `RunningNorm`（Welford 合并，clip 10）（`agents/supermimic/world_model.py:12-41`） | 同语义：输入归一化 + 增量归一化两套（`world_model.py:17-42,63-65`） |

## 6. Actor 与 critic 结构

| 项 | 本项目 | 参考项目 |
| --- | --- | --- |
| actor | MoE 历史编码器：225 维历史 → 共享 ELU 骨干 (512,256) → 8 专家 × 32 维 → softmax 门控 → `L2Norm` → 32 维隐变量；隐变量 ‖ 当前帧（独立归一化）→ ELU MLP (512,256,128) → 12 维无界动作（`agents/supermimic/policy.py:70-131`、`docs/go2_supermimic.md:40`） | 单 MLP：`RunningNorm` → ELU MLP (512,256,128) → 12 维（`train.py:135-142`） |
| 参数 | 可训练 1,293,972；含归一化 buffer 1,294,514（本地实算） | net 权重 flat 345,484 / obstacles 386,444 / terrain 542,604；含 buffer 346,185 / 387,305 / 544,075（读 `weights/*.pt` 实算） |
| 权重兼容 | `load_pretrained()` 可按 go2_rl_gym 导出名热启动，缺张量直接报错（`agents/supermimic/policy.py:141-178`） | 只有 `expand_actor_new_terms`（加观测项时零权重扩展）（`train.py:172-200`） |
| 负载均衡 | 门控使用率对均匀分布的平方误差 ×0.01，按整批统计，在微批循环外反传（`agents/supermimic/learner.py:92-96`） | 无（单 MLP） |
| 动作输出 | 无界线性输出，目标 = 默认角 + 动作×scale | 同 |
| critic | `RunningNorm` → ELU MLP (512,256,128)，输入 179，可训练 256,513（本地实算） | 同结构，输入 91/107/168，参数 211,457（flat）/250,881（terrain）（实算） |

## 7. 训练循环与超参

| 超参 | 本项目 | 参考项目 | 差异 |
| --- | --- | --- | --- |
| envs | 4096（`config:25`） | 4096（`train.py:34`） | 同 |
| 默认轮数 | 10,000（`config/train_cfg.py:27`） | 40,000（`train.py:36`） | 不同 |
| 回放容量 | 1000（`agents/supermimic/config.py:26`） | 1000（`buffer_len`，`train.py:35`） | 同 |
| 每轮采集 | `collect_steps=1`，首轮一次采 16 步（`agents/supermimic/config.py:27-28`） | 主循环前 `while buf.size <= wm_window+1: collect()`，约 10 次单步（`train.py:569-572`） | 预填策略不同 |
| 世界模型 | 8 步窗口、2048 批、每轮 2 次更新、1024³ SiLU（`agents/supermimic/config.py:30-33`） | 同（`train.py:49-51`） | 同 |
| 优先/归档 | `wm_prioritized_frac=.25`、`wm_archive_frac=.25`、archive 100k、keep 0.02（`agents/supermimic/config.py:59-62`） | 同（`train.py:52-59`） | 同 |
| 想象 | 8192 条 × 32 步、γ 0.99、λ 0.95、终点 critic 自举（`agents/supermimic/config.py:34-37`） | 同（`train.py:72-73`、`imagine.py:62-63`） | 同 |
| 预热 | `warmup_iterations=300`，绝对 `policy_start`（`agents/supermimic/config.py:29`） | 同值，但续训会重置倒计时（`train.py:37`） | 本项目有意去掉倒计时（`docs/design/go2_blind_supermimic.md:56`） |
| 学习率 | 2e-4，cosine 到 3%，30k 轮衰减（`agents/supermimic/config.py:44,47-48`） | 同（`train.py:65,70-71`、`lr_at`） | 同 |
| actor 斜率 | `actor_ramp_iterations=1000`，起点 `max(ramp_start, actor_frozen_until, policy_start)`（`agents/supermimic/learner.py:55-67`） | 同（`train.py:576-580`） | 同 |
| critic 训练 | 5 epochs × 4 minibatch，λ-return（`agents/supermimic/config.py:49-50`） | 同（`train.py:76-77`） | 同 |
| 梯度裁剪 | 全局 1.0 + 逐元素 0.1（`agents/supermimic/config.py:51-52`） | 同（`train.py:78-79`、`clip_grads` 见 `:155-160`） | 同 |
| 编译 | `torch.compile(fullgraph=True, backend=inductor)` 包单步（`agents/supermimic/imagine.py:28-29`） | `torch.compile(self._step, dynamic=False)`（`imagine.py:112`） | 同思路 |
| 微批 | 有：`imag_microbatch=256`、`critic_microbatch=4096`，总批量与优化器步数不变（`agents/supermimic/config.py:8-20`、`learner.py:96-155`） | 无 | 本项目新增 |
| 仅训模型 | `--model-only` / `enable_control_updates`（`config/train_cfg.py:44-45`、`config:81`） | 无 | 本项目新增 |
| 续训 | 追加轮数 + 保留绝对计数；`spec_hash` 不匹配直接报错（`docs/design/go2_blind_supermimic.md:57`） | `--continue-run`、`wm_rewarmup_iterations=1500`、`critic_warmup_iterations=500`、`reset_critic`、形状变化时静默重初始化（`train.py:62,109,116,119,305-359`） | 本项目有意差异 |
| 崩溃保护 | 每 `collapse_interval=50` 轮统计绝对跌倒数，>3 连续 2 次则回滚 actor+critic（内存安全副本）、重建优化器、lr 减半（`agents/supermimic/runner.py:223-255,316`） | 同阈值与连击逻辑，回滚源是磁盘 `last_ckpt`（`train.py:620-640`） | 阈值同、来源异 |
| 探索 | OU `θ=2.0, σ=0.3` 作用于前 25% 环境，其余高斯 0.1（`agents/supermimic/runner.py:150-160`、`config.py:55-58`） | 同（`train.py:261,382-388`） | 同 |
| 驱动/专家 | 支持；算法层默认 `driver_frac=0`（`agents/supermimic/config.py:65-66`），任务配置默认 `frac=0.25`、`expert_frac=0`（`config:83-93`） | `driver_frac=0.25`（`train.py:97`），地形配方覆盖为 0.1/0.1（`README.md:79-88`） | 同机制、默认值不同 |
| 任务侧课程计数 | 命令课程按 `control_steps`（`mdp/curriculums.py:44-58`）；地形课程按实际行进距离（`mdp/curriculums.py:95-108`） | 目标课程按 env steps；地形课程按训练进度 `terrain_levels_by_progress(steps=12_000)`（`env/mdp.py:183-197`、`train.py:90,239`） | 语义不同 |

## 8. 地形、课程与域随机化

| 项 | 本项目 | 参考项目 |
| --- | --- | --- |
| 地形生成 | 自建固定连通地图：11 个家族 × 4~5 级 = 91 块 8×8 m 瓦片，7 行 × 13 列 + 1 圈外环，外侧 16 m 平地（`mdp/terrains.py:86-167`，网格尺寸本地实算） | mjlab `ROUGH_TERRAINS` 生成器：5 行 × 7 列 × 12 m，7 种按比例混合的子地形（`env/terrains.py:32-56`） |
| 地形类型 | `random_rough`（0.02→0.06 m）、8 种楼梯（步高 0.05→0.16 m，步宽 0.18/0.21/0.25/0.28，上/下）、上/下坡（0→0.4 梯度）（`config:145-158`） | `seed=7`、`border_width=10`、`curriculum=True`；flat 10%、pyramid_stairs 20%（步高 0.02→0.10、步宽 0.4）、stairs_inv 20%、box_random_grid 15%、random_rough 15%（0.01→0.04，步长 0.01）、hf_pyramid_slope 10%（0→0.3）、slope_inv 10%（`env/terrains.py:32-53`） |
| 课程方式 | 按实际行进距离升降级：`distance > 半块` 升、`distance < 命令位移×0.5` 降，早摔保留剩余预算（`mdp/curriculums.py:95-108`） | 按训练进度开放行：第 k 行在 `k/R` 进度后开放，重置时在已开放行内随机（`env/mdp.py:183-197`） |
| 面类型 | `ConnectedColumnCfg`、`PlaneSurface`、`TerrainSurface` 三种，含解析 `geometry_hash` | 无哈希与固定布局概念 |
| 摩擦 DR | 只随机滑动轴 `(0.3,1.5)`，`shared_random=True`（`config:173`、`go_supermimic_env_cfg.py:99-103`） | 三轴：slide `(0.3,1.5)` uniform、spin `(1e-4,2e-2)` log、roll `(1e-5,5e-3)` log（`env/env_cfg.py:96-112`） |
| COM DR | `x±0.025 / y±0.05 / z±0.05`（论文表 A.3）（`config:175`） | mjlab 默认 `x±0.025 / y±0.025 / z±0.03`，只把 body 改为 `base`（`env/env_cfg.py:114`，默认值本机实测） |
| 编码器偏置 | ±0.01（论文表 A.3），并作为世界模型/critic 输入显式传入（`config:174`、`mdp/reference_task.py:135-138`、`go_supermimic_env_cfg.py:104-106`） | 沿用 mjlab 基类的 `encoder_bias` 事件（参考仓库未删除、也未改写，见 mjlab `tasks/velocity/velocity_env_cfg.py:251`），范围用 mjlab 默认 ±0.015（本机读 `velocity_env_cfg.py:251-258`）；它只进观测、不进世界模型：xi 只有摩擦 1 + COM 3（`supermimic/state.py:31`） |
| 外力 push | 删除：确定性世界模型无法表示速度踢动（`docs/go2_supermimic.md:50`） | 显式 `cfg.events.pop("push_robot")`，理由相同（`env/env_cfg.py:119`） |
| 求解器 | `solver=cg, iterations=10, ls=20, ccd=50, impratio=10, cone=elliptic, njmax=300, nconmax=None, broadphase=nxn`（`config:37-47`、`go_supermimic_env_cfg.py:126-139`） | `impratio=10, cone=elliptic, njmax=300, ccd=50, nconmax=None`，其余用 mjlab 默认（`env/env_cfg.py:64-69`；本机实测 mjlab 默认 `solver=newton, iterations=10, ls=20`） |
| 机器人 | Menagerie Go2 MJCF 本地副本（加中文头注释与 `head/front_camera/imu` site），直流电机、10 Hz 自然频率、阻尼比 2、hip/thigh 23.7 Nm、calf 45.43 Nm、动作比例 `0.25×effort/stiffness`、0~2 物理步延迟、全自碰撞、足 condim 6（`src/assets/robots/go2_reference.py`、`config:95-131`） | 同源 `go2.xml`，同样电机与延迟、同样碰撞设置（`robot/go2.py:20-130`） |

## 9. 外部策略驱动

| 项 | 本项目 | 参考项目 |
| --- | --- | --- |
| 实现 | 三个类：`MoeCtsDriver`（读 go2_rl_gym TorchScript，自建批量历史绕过 batch=1 环形缓冲）、`ActorDriver`（读本项目检查点，作专家）、`ChaseSteer`（把追逐目标变成速度命令）（`rl/driver.py:34-149`） | 三个同名类：`MoeCtsDriver`（`driver.py:42-85`）、`ActorDriver`（`:86-106`）、`ChaseSteer`（`:107-138`），职责一一对应（`supermimic/driver.py:42-138`） |
| 命令缩放 | `(2.0, 2.0, 0.25)`，按 go2_rl_gym `obs_scales`（`rl/driver.py:16`、`docs/design/go2_blind_supermimic.md:40`） | `(0.5, 1.0, 1.0)`（`supermimic/driver.py:30`） |
| 环境分区 | 尾部切片：驱动区最后、专家区紧邻其前、探索区在开头 | 同约定（`train.py:372-380`） |
| 用途 | 让世界模型从一开始见到真实地形数据；课程仍由任务侧推进 | 同（README 强调只塑造数据，部署策略仍是 actor）（`README.md:79-88`） |

## 10. 入口、CLI 与配置

| 项 | 本项目 | 参考项目 |
| --- | --- | --- |
| 训练入口 | `scripts/train.py <TaskId> --flags`，路由到 `src/tasks/locomotion/go2_supermimic/cli.py`（tyro 扁平参数）（`cli.py:1-30`） | `scripts/train.py --task <TaskId> --flags`；`TrainCfg` 每个字段自动成为命令行参数（`train.py:642-659`、`scripts/train.py:1-19`） |
| 参数体系 | 两层：`Go2ReferenceCfg`（任务语义，`src/config/go2/`）+ `SuperMimicTrainConfig`（运行预算）+ 可选 `--task-config JSON` | 单层 `TrainCfg`，全部写入检查点 |
| 播放/评估 | `scripts/play.py <TaskId> --checkpoint-file ... [--agent zero/random/trained] [--viewer auto/native/viser] [--video] [--no-terminations]`（`scripts/play.py:38-51`） | `scripts/eval.py <TaskId> --checkpoint ... [--mode chase/dodge/both/mixed] [--difficulty 0~1] [--terrain ...] [--video]`（`scripts/eval.py:31-44`） |
| 评估协议 | 4 组固定命令（前进/后退/侧移/转向）× N 环境，报每轴实测、RMSE、首回合存活、跌倒率、TorchScript 对拍（`rl/evaluation.py:28-126`） | 多环境统计：tags/min、按种类命中率、跌倒/min、均速，可选出 mp4 与状态卡（`scripts/eval.py:1-160`） |
| 配置校验 | `validate_config()` 在构造仿真前统一检查（`go_supermimic_env_cfg.py:26-43`） | dataclass 类型 + 少量断言，无集中校验 |

## 11. 产物、续训与部署

| 项 | 本项目 | 参考项目 |
| --- | --- | --- |
| 训练目录 | `logs/supermimic/go2_supermimic/<时间戳_标识>/`，只写三类产物：`model_N.pt`、`events.out.tfevents.*`、`task_config.json`（`rl/runner.py:81-91`、`rl/training.py:117-129`） | `logs/supermimic/<run_name>/`，`model_N.pt` + TensorBoard 事件（`train.py:550-566`） |
| 检查点内容 | actor/critic/world_model + 三个优化器 + 计数 + `config` + `spec` + `spec_hash` + RNG 状态（`agents/supermimic/runner.py:322-335`、`rl/runner.py:81-91`） | actor/critic/world_model + 三个优化器 + `cfg` + `iteration` + `obs_dim`（`train.py:550-566`） |
| 发布权重 | 无 | `weights/{flat,obstacles,terrain}.pt`，仅 actor，不可用于续训（`README.md:55-64`） |
| 部署导出 | `tools/export_supermimic_policy.py` → TorchScript `.pt` + ONNX（opset 18）+ 元数据 JSON（关节名、默认角、动作比例、坐标语义、`spec_hash`、参考提交），格式 `supermimic_moe_policy_v1`（`rl/deployment.py:24-52,69-91`） | 无导出工具；驱动器用的 TorchScript 来自 go2_rl_gym 侧 |
| 播放对拍 | 评估时当场导出并在临时目录 `torch.jit.load`，与训练 actor 比对 `atol=1e-4`（`rl/evaluation.py:52-57`） | 无 |

## 12. 测试、文档与工程质量

| 项 | 本项目 | 参考项目 |
| --- | --- | --- |
| 测试规模 | 35 文件 / 236 个 `test_*` 函数 / 6,098 行（本机实算） | 3 文件 / 25 个测试（本机实算） |
| 测试内容 | 参考对拍（`GO2_SUPERMIMIC_REFERENCE`、`GO2_RL_GYM_ROOT` 门控）、显式 CPU/GPU 训练检查（`RUN_REFERENCE_CPU/GPU=1`）、微批梯度等价、日志格式、检查点/CLI/配置、地形与课程、隔离性（`docs/go2_supermimic.md:75-95`） | 纯 CPU、不需要仿真器：目标生成器、碰撞、障碍、高程场、两种热启动扩展（`tests/*`） |
| 文档 | `AGENTS.md` 开发规范、`docs/` + `doc/` 共 28 篇 md（其中 9 篇设计 + 9 篇审查，各含 1 份模板）、两篇分析报告、`docs/supermimic_projects.md`（相关项目索引）、`doc/TENSORBOARD_REMOTE.md` | 单篇长 README（233 行）+ 源码 docstring，无设计/审查记录 |
| 代码风格 | 中文注释与文档、英文运行日志、文件头版权 `Copyright (c) 2026-2027 zh`、中文分块注释 | 英文注释与文档、`##...##` 分块、2 空格缩进 |
| 日志 | TensorBoard 标量 + 80 列中文分块终端摘要（`agents/supermimic/logging.py`） | TensorBoard 标量 + 单行 `it/steps/wm_loss/...` 打印（`train.py:588-617`） |
| 版本策略 | 检查点带 `spec_hash` 与 `model_id`，格式变更拒绝加载，旧 `single` 段自动迁移 | 形状不匹配时自动重初始化/扩展（`expand_*`）（`train.py:172-227,305-359`） |

## 13. 许可与来源声明

| 项 | 本项目 | 参考项目 |
| --- | --- | --- |
| 许可 | MIT，`Copyright (c) 2026 奈奈奈奈奈绪` | MIT，`Copyright (c) 2026 Ziyue Zheng` |
| 第三方声明 | `agents/supermimic/NOTICE.md`、`src/assets/robots/go2/REFERENCE_NOTICE.md`、`LICENSE_menagerie_go2` | README 致谢 + `robot/LICENSE_menagerie_go2` |
| 致谢对象 | SuperMimic 论文、mjlab、Menagerie、go2_rl_gym | 同上，另含 SPIKE Lab 与 Claude Code |

## 14. 已验证表现与已知限制

两者成绩不可互相替代，也不构成对方的验收基线。

| 项 | 本项目 | 参考项目 |
| --- | --- | --- |
| 最佳运行 | MoE 运行 `2026-10-27_19-28-11_197179`（**该 run 用 `--num-envs 1024`，不是默认 4096**），iteration 3043 / real_steps 3 131 392 / 83.7 min，结束时 `model_3000.pt` 是最新检查点；能力约 2400 轮后出现（`docs/analysis/go2_supermimic_analysis.md:21,25`） | 作者自报：flat/obstacles 约 8k 轮（从 flat 热启动），terrain 约 12k + 约 25k 分阶段（**参考项目依据**，`README.md:97-113`） |
| 物理表现 | `model_3000`：前向 0.3132 / 反向 −0.3632 / 侧向 +0.5091 m/s，20 s 内 0 摔倒；偏航 RMSE 从 0.075 退到 0.199（`docs/analysis/go2_supermimic_analysis.md:25,675-701`） | `terrain.pt`：最难行约 0.6 次摔倒/分钟，约 1/10 低杆被击中；易行约 0.3 次/分钟（**参考项目依据**，`README.md:192-200`） |
| 失败记录 | 1560 轮的单 MLP 旧构建控制验收未通过（与当前构建不同代码）（`docs/analysis/go2_supermimic_analysis.md:21-22`） | 从零复跑最终任务：约 7k 轮能追少摔，但仍常被杆击中（`README.md:102-108`） |
| 显式限制 | 训练未收敛，回合长度仍在 442~860 步震荡；无真机验证 | 纯仿真、感知为真值、训练配方未逐条端到端复现、无真机（`README.md:192-202`） |

## 15. 本项目对参考实现的有意偏离

| 偏离点 | 做法与理由 | 依据 |
| --- | --- | --- |
| actor 由 MLP 换成 MoE | 按用户要求 actor 用 MoE、其余对齐 go2_rl_gym `30e74dc`；结构与键名一致才能热启动预训练权重 | 本项目已验证（`docs/go2_supermimic.md:40`） |
| 历史展平改 frame-major | go2_rl_gym 导出权重按帧堆叠，term-major 无法直接载入 | 本项目已验证（`docs/design/go2_blind_supermimic.md:495`） |
| 世界模型/critic 多 12 维编码器偏置 | MjLab 的位置动作会从执行器目标里减掉 encoder bias，它因此同时是动力学与价值条件 | 本项目已验证（`mdp/reference_task.py:135-142`、`docs/go2_supermimic.md:48`） |
| 奖励换成表 A.6 六项 | 任务改成速度跟踪，追逐/闪避的多项事件奖励不再适用 | 论文依据 + 本项目已验证（`docs/go2_supermimic.md:44`） |
| COM 相对未随机化 MJCF 读取 | 避免把已随机化的基准当成名义值 | 本项目已验证（`docs/go2_supermimic.md:48`） |
| 去掉 `push_robot` | 确定性世界模型无法表示速度踢动 | 参考项目同结论（`env/env_cfg.py:119`） |
| 不引入 `wm_rewarmup`/`critic_warmup`/`reset_critic`/`continue_run`/`expand_*` | 用 `spec_hash` 硬校验，形状变化直接报错，这些分支不可达；续训语义是追加轮数并保留绝对计数 | 本项目已验证（`docs/design/go2_blind_supermimic.md:56-58`） |
| 地形课程改为按行进距离 | 速度跟踪的难度与「走了多远」直接相关，按进度开放行无法反映单环境能力 | 推断／待验证假设 |
| 崩溃保护回滚源改为内存安全副本 | 避免依赖最近一次成功保存的磁盘文件，缩短有效回滚点的间隔 | 本项目已验证（`agents/supermimic/runner.py:223-255,316`） |

## 16. 各自适用范围

| 目标 | 选择 | 原因 |
| --- | --- | --- |
| 复现 Go2 追球+闪避，要现成权重与视频 | 参考项目 | 3 份发布权重、`eval.py` 出 mp4、三种场景齐备 |
| 研究 SuperMimic 方法迁移到新任务（改 MDP 结构） | 本项目 | 任务语义集中在一个 `Go2ReferenceTaskModel`，采集与想象共用，改奖励/观测不必同步两份实现 |
| 显存有限（8~16 GB）做完整想象的 BPTT | 本项目 | 独有 `imag_microbatch` / `critic_microbatch`，并有显式 GPU 训练检查 |
| 训盲（仅本体）四足速度跟踪并对齐表 A.6 | 本项目 | 45×5 盲观测、六项奖励、MoE 可吃 go2_rl_gym 预训练 |
| 要立刻可用的有感知基线 | 参考项目 | 本项目盲策略 3000 轮仍未收敛（偏航还退步） |
| 部署到真机（TorchScript/ONNX + 元数据） | 本项目 | 有导出工具与 `supermimic_moe_policy_v1` 元数据 |
| 读一份 4.8k 行、单文件 Trainer 的好读实现 | 参考项目 | `supermimic/train.py` 659 行把三阶段写在一处 |
| 生成式混合地形 + 进度式课程 | 参考项目 | mjlab `ROUGH_TERRAINS` 七种地形按比例混 + 进度开行 |
| 固定可复现的连通地形图与 `geometry_hash` | 本项目 | 自建 `TerrainMap`（91 块、11 族、4~5 级），检查点可校验地形一致性 |

## 17. 证据缺口

- 论文只对照了表 A.6 与算法 A.1 相关段落（本地 `paper/SuperMimic_2610.28480v1.pdf`），未逐条核对其余附录。
- 参考项目的训练曲线与视频未在本地运行；`weights/*.pt` 只做了参数与键名统计，「参数」一列的数量由结构实算。
- 求解器差异（`cg` 与 mjlab 默认 `newton`）的数值影响未做对照实验，属**推断**层面的差异，需实测才能量化。
- 参考项目 README 的成绩均为作者自报，没有在本机复跑；本项目的 3043 轮快照也不代表收敛。
- `docs/supermimic_projects.md` 引用的是上游 `054a13c…`，本文引用 `d9eee9b3`。本机浅克隆只有后者，无法用 `git` 核对两者差异；如需与旧调研对齐，应重新取全历史后逐文件比对。
