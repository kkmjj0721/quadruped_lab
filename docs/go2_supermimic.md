<!-- Copyright (c) 2026-2027 zh -->

# Go2 SuperMimic 盲速度跟踪

文中的本地工作记录位于 `docs/work/`，不随仓库提交。

默认训练对齐指定 [Go2-SuperMimic](https://github.com/Zhengtianjin/go2-supermimic/tree/d9eee9b3f248e5300e636c433255db2c1c3e086a) 的世界模型、想象BPTT和物理随机化，地形采用下述 Ri-5036 HIM 配置，actor使用 [go2_rl_gym](https://github.com/wty-yy/go2_rl_gym/tree/30e74dc507bec7a642a8c98be26081f2c6f0822d) MoECTS学生结构的8专家MoE历史编码器。当前二十一项速度奖励以 [SuperMimic 英文 v1](../paper/SuperMimic_2610.28480v1.pdf) 第16页表 A.6 为起点，按用户要求将平移跟踪改为 HIM 的机体系 xy、偏航指数分母改为0.25并删除新训练侧倾项，增加下述 HIM 接触、角动量、髋姿态和落地冲击约束。命令采用 HIM 的 heading 逻辑；策略读取本体历史、速度命令和步态相位，由历史显式估计当前机身系三维线速度。

这属于用户确认的从零 Go2 速度控制迁移。原论文速度接口与动作模仿联合训练，参考仓库任务是追逐和闪避；本项目没有复现其动作库先验或演示成绩。集中配置见 [Go2ReferenceCfg](../src/config/go2/go2_reference_supermimic_config.py)，算法来源见 [NOTICE](../agents/supermimic/NOTICE.md)，资产与执行器装配见 [Go2 定义](../src/assets/robots/go2/go2_constants.py)。本地工作记录中，与本仓库的逐项差异见[对比报告](work/analysis/go2_supermimic_vs_reference.md)，与相邻 `go2_him` 任务在命令采样、MDP 组装、奖励终止上的对齐评估见[任务层对齐评估](work/analysis/go2_supermimic_vs_go2_him.md)。

## 训练

```bash
.venv/bin/python -u scripts/train.py Unitree-Go2-SuperMimic-Flat \
  --device cuda:0 --max-iterations 40000 --save-interval 1000
```

新训练默认8MoE，无需额外指定策略配置。新格式8MoE检查点按同目录完整 `task_config.json` 恢复；局部配置覆盖其余预算的方式见下文。

日常调参直接修改各机器人的Python配置：[Go2](../src/config/go2/go2_reference_supermimic_config.py)、[RI-4438](../src/config/ri_4438/ri_4438_supermimic_config.py)。`algo`段集中声明网络、学习率、回放、世界模型／想象预算、预热与探索幅度；`runner`段声明训练目标轮数、保存／日志间隔、执行分批和编译／融合开关。共享`agents/supermimic/config.py`保留字段定义、通用默认和校验。两个机器人的配方分别维护，修改其中一个不会改变另一个。

这是显式完整训练命令，不随单元测试启动。粗糙地形任务为 `Unitree-Go2-SuperMimic-Rough`，地形生成配置来自本地 `legged_wbc_mjlab@4eb8fcf71124b20ed4a1154b6c724ba962a424f8` 的 `src/tasks/locomotion/ri_5036_him/ri_5036_him_env_cfg.py:578–615`：8×8米地块、6行、20米外围平地，包含平地、0.8米箱体网格及四种正／四种倒金字塔楼梯。配置保留 `num_cols=20`，但 MjLab 1.6 课程模式按每种子地形一列实际生成6×10；`seed=None`与来源一致，需固定地图时显式设置地形种子。地形布局、高程查询和扫描布局保持不变；难度升降采用同提交 `go2_him/go2_him_env_cfg.py` 转导的 `ri_4438_him/mdp/curriculums.py:terrain_levels_vel`：距地块原点超过半块长度且期望命令距离至少1米时晋级，实际净位移不足期望距离一半时降级，晋级优先；期望距离包含历史世界系命令位移和提前结束时剩余时长的要求。越过最高级时随机回落到现有等级，最低等级为零；仅真实结束回合结算，手动重置不升降。保留零级训练起点，不再按训练进度开放难度，`terrain.curriculum_iterations` 仅保留为旧配置字段。粗糙地形 actor 仍然没有高程输入，不能继承参考有高程策略的效果。

Go2／RI4438新训练默认使用三层2048宽度的世界模型，每轮拟合5批2048个8步窗口（`algo.wm_hidden=(2048,2048,2048)`、`algo.wm_updates=5`）。更新次数采用用户选定的主线配方，论文表A.5为每轮10批。想象批量仍采用参考仓库 `logs/supermimic/2026-10-27_17-09-24_flat/model_1000.pt` 中保存的4096条32步想象。真实采集按用户指定的 [zita-ch/supermimic_vibed@8bfc115](https://github.com/zita-ch/supermimic_vibed/blob/8bfc1158bd968f729fcc65d5e897a05b05ea3d36/supermimic/train.py#L300-L322)：`algo.collect_steps=2.5` 表示每轮平均控制步数，对应UTD=0.4；循环累加后取整，按2、3步交替，世界模型拟合批数另计。

执行方式默认使用工程消融的 `s101`：编译世界模型监督核心，并为actor、世界模型和critic使用fused Adam；护栏重建及新格式检查点恢复也沿用当前 fused 设置。监督核心后端使用 `algo.compile_backend`（默认 `inductor`），保留显式半精度舍入，命令采样仍按原顺序执行；想象编译继续由 `algo.compile_imagination` 控制。首次监督更新含编译开销。使用 `--wm-compile False --fused-adam False` 可关闭这两项以对照原工程基线；这些开关不改变任务参数、训练预算或检查点网络键，实际值写入 `params/agent.yaml` 的 `execution`。

其余预算沿用指定Go2参考源码：4096环境、1000步/环境回放；300轮模型预热；actor动作MLP与critic均为ELU(512,256,128)。世界模型每轮更新次数与论文表A.5不同，其余参考预算按用户选择保留。训练循环外仍独立预填10步；累加器跨分段训练保留余量，重置采集或恢复时重新开始。回放直接保存完整半精度历史。新格式检查点恢复读取同目录`task_config.json`中的采集预算。日志 `collection/control_steps` 为本轮实际步数，吞吐按真实转移差计算。

`--num-envs` 只改变环境数。机器人`runner.imag_microbatch=4096`、`runner.critic_microbatch=32768`默认覆盖整个想象批量和每个critic小批：actor每轮整批反传一次，critic按5轮×4小批反传并更新20次。调小这两个上限时才分段，也可通过同名CLI显式覆盖；每段仍完整反传32步，总批量和优化器更新次数不变。配置不根据显卡自动缩减。整批执行的峰值显存高于分段执行；8 GB显卡应先运行下文显式GPU检查，再按实际容量设定实验规模；短检查不证明默认4096环境的全预算能运行。修改默认值只影响新进程；续训或显式加载JSON时，其中保存的批量仍覆盖Python默认，显式CLI覆盖已加载值。

临时覆盖仍可使用JSON，例如：

```json
{
  "algorithm": "moe_history",
  "env": {"num_envs": 256},
  "algo": {"replay_capacity": 256, "archive_size": 10000, "wm_batch": 256, "imag_batch": 256}
}
```

使用 `--task-config <配置.json>`。这是独立缩减实验，运行快照保留差异；不是论文或参考仓库的完整训练预算。优先级为显式CLI、JSON或续训快照、机器人Python默认。训练轮数、保存／日志间隔和四项执行预算的CLI未指定时，直接使用机器人配置或快照值。运行字段不参与任务指纹；算法超参数仍按原规则进入指纹。动作输出固定为无界线性输出，不提供动作限幅配置。

每轮依次真实采集、世界模型自回归拟合、冻结模型权重并通过确定性想象回报更新完整actor、以固定lambda-return拟合critic。真实采集默认固定25%环境不加额外动作探索噪声，其余环境在移动命令下逐步逐维独立添加高斯噪声，动作缩放前σ=0.3；站立命令下额外动作噪声为零，回放保存实际执行动作。想象继续加入观测噪声，不加动作探索噪声，也不创建可学习幅度或探索critic。固定幅度与确定性想象依据为 [SuperMimic英文v1](../paper/SuperMimic_2610.28480v1.pdf) §3.1／§3.3；论文表A.5默认σ=0.1，当前0.3、固定分组和站立关噪为用户选定的本项目消融配方。来源与任务差异见 [NOTICE](../agents/supermimic/NOTICE.md)。历史固定高斯、确定性想象且关闭OU的默认方案检查点可播放和恢复；已删除的探索方案在读取时明确拒绝。保留接触变化优先采样、稀有事件归档、学习率调度和保存点回滚。固定方案回滚恢复actor、critic并重建其优化器，不恢复物理仿真。

### 固定探索噪声

Go2和RI-4438的Flat／Rough新训练默认 `algo.action_noise=0.3`、`algo.collect_det_frac=0.25`、`runner.standing_zero_noise=True`。前两项为动作缩放前标准差和固定无噪声环境比例。站立判定与奖励一致：当前命令的平移模长加偏航绝对值不超过 `reward.stand_still_command_threshold`（默认0.1）。站立环境与固定无噪声组取并集，切回移动立即恢复原加噪资格；观测噪声及物理随机化沿用原配置。原训练命令直接生效，无需额外JSON；设 `runner.standing_zero_noise=False` 可恢复仅固定组关噪。

`collect_det_frac`按真实地形分层：总名额为环境总数乘比例后就近取整，各地形先下取整，再按小数余量补足；每类地形内部等距选择环境，分组在本次运行保持固定。4096环境时共1024个环境固定关闭动作探索噪声，逐地形整数配额误差小于1个环境。观测噪声和物理随机化仍按原配置启用。设为0并同时设 `runner.standing_zero_noise=False` 可恢复全部环境加动作探索噪声的基线；新格式检查点继续训练不允许改变比例，同结构权重热启动可采用新比例。

Go2按默认混合采集方案从头训练，使用：

```bash
.venv/bin/python -u scripts/train.py Unitree-Go2-SuperMimic-Rough \
  --device cuda:0 --seed 0 --max-iterations 5000 --run-name paper_exploration_01
```

Flat任务将任务标识换成`Unitree-Go2-SuperMimic-Flat`。

RI-4438启动方式为：

```bash
.venv/bin/python -u scripts/train.py RI4438-SuperMimic-Rough \
  --device cuda:0 --run-name paper_exploration_01
```

检查点保存actor、critic、世界模型及各自优化器。当前入口只接受新速度监督格式，旧网络与退休配置字段不再迁移。部署和播放使用确定性动作，ONNX不包含随机采样；恢复不重建物理轨迹或回放。

## 任务与模型

新训练actor单帧47维：机体系角速度3、重力投影3、带编码器偏置的关节角偏移12、关节速度12、上一下发动作12、速度命令3、步态相位sin／cos两维。历史为5帧235维，按旧到新逐帧排列，reset重复首帧。相位在原45列末尾追加，不采样噪声；时钟来自回合控制步，与步态奖励共用周期，命令平移模长加偏航绝对值不超过0.1时两维清零。仍不输入真实线速度、地形、接触或仿真参数；线速度由历史估计。

默认 `algorithm="moe_history"`、`algo.moe_expert_num=8`：235维历史经共享ELU骨干(512,256)展开为8×128，分组1×1卷积每专家输出35维；Softmax融合后拆为32维潜特征与3维当前机身系线速度。仅潜特征做L2归一化，速度保持m/s量纲。按“潜特征32、预测速度3、当前帧47”拼成82维，动作MLP为(512,256,128)，输出12维。训练使用原想象目标加真实回放同一时刻速度MSE，`algo.velocity_loss_coef=1.0`；标签detach，不把真实速度送入actor。日志新增`Loss/velocity_estimation`、`Loss/actor_total`及`Metrics/velocity_estimation/rmse{,_x,_y,_z}`。结构来源与适配差异见[算法说明](../agents/supermimic/NOTICE.md)。

命令采样以 `legged_wbc_mjlab@e6209ec` 的 `go2_him` 为来源；新训练默认vx/vy各为[-1,1]m/s，wz配置范围为[-2,2]rad/s，每4–12秒重采样。此前±4范围在两次训练诊断后收回；当前没有自动扩展到±4的速度课程。heading目标范围±3.14rad，`wz=clip(0.5*wrap(target-yaw),-2,2)`，因此该heading反馈实际产生的wz最多约±1.57rad/s；范围配置不改变反馈增益。heading比例1、站立比例0.1、forward比例0.05。保留来源实际顺序：forward先约束vx≥0.3和vy=0，随后heading会重新计算wz；站立最后把三个命令清零。真实与想象分别用真实／预测yaw更新命令。当前平移奖励评价机体系xy，偏航仍评价world z角速度；部署及评估元数据显式记录坐标系。

当前默认二十二项奖励与权重集中维护在 `Go2ReferenceCfg.reward`，不在此重复参数清单。`track_linear_velocity` 使用body-xy，`track_angular_velocity` 使用world-z，两者指数分母均0.25、权重均2；新训练不计算 `torso_sideways_lean`。所有奖励率仅乘一次0.02秒。`body_ang_vel` 按HIM计算机体系x、y角速度平方和，权重仍为−0.05；`reward.body_ang_vel_frame=body` 标识新默认。Go2终止仍取参考70度倾倒和回合超时，想象用5度宽sigmoid，不增加地图出界终止。`is_terminated` 在真实端使用失败标记，正常超时不罚；想象端使用同一软失败概率并保留梯度。当前默认启用该项。

`lin_vel_z` 使用机身坐标系速度z的平方，倾斜时与世界／heading系z不同；由现有速度与重力解析恢复，不增加世界模型输出。`body_orientation_l2` 为机身重力投影xy平方和。`smoothness` 使用未缩放下发动作的 `sum((a_t-2*a_(t-1)+a_(t-2))**2)`，不除以dt，reset时两级动作历史清零。三项同时用于真实奖励和可微想象更新；权重以用户指定为准，其中smoothness并非来源配置的-0.015。

`stand_still` 为终点默认关节角偏差平方和，`stand_still_joint_vel_l2` 为终点关节速度平方和，两项均在起点命令平移模长加偏航绝对值不超过0.1时启用。速度项直接约束小幅持续摆腿，真实与想象共用公式，复用已有22:34关节速度槽位，也会对站立平衡动作施加代价。零命令下步态／腾空项关闭。`foot_slip` 在命令大小超过0.1时，按逐脚世界xy瞬时速度平方和乘found接触后求和；`joint_acc_l2` 直接使用仿真器qacc平方和。真实端在adapter重置前读取瞬时量；想象端预测48维连续辅助量和4维接触概率，不使用控制周期差分替代瞬时量。接触logit输出统计固定均值0、方差1，概率残差沿用模型MSE；预测量及接触概率仍须单独验证。

`stumble` 使用 `feet_ground_contact` 的世界净接触力：任一脚满足 `norm(force_xy) > stumble_ratio * abs(force_z)` 时为1，否则为0，默认比值5.0；保持来源的严格硬阈值，不增加接触门控或平滑替换。该指标在想象端对接触力没有直接局部梯度。`joint_torques_l2` 为终点实际 `actuator_force` 的平方和，想象端对预测力矩使用相同公式并可反传；新增预测量的准确性及控制效果尚需训练验证。

`angular_momentum` 为根部子树角动量三轴平方和，heading旋转不改变HIM来源的world范数；`hip_pos` 仅在侧移及偏航命令绝对值均不超过0.1时计算四个髋关节默认角偏差平方和，快速直行仍启用。`soft_landing` 在命令大小超过0.1时，对首次触地脚的净接触力模求和；真实端的首次触地为 `0 < current_contact_time < dt + 1e-6`。想象端按用户确认方案，以计时记忆除以接触概率恢复条件时长，再使用同一硬窗口并乘接触概率；这是概率加权代理，力头为无条件量，不能视为严格无偏的冲击期望。触地窗口无直接局部梯度，接触力模及概率幅值可反传，预测准确性和控制效果尚需训练验证。

步态周期0.6秒、支撑比例0.56；按来源FR/FL/RR/RL足序的相位偏移为[0,.5,.5,0]，本项目按足名映射。腾空奖励使用逐脚三角时长得分，目标支撑0.336秒、腾空0.264秒；只有符合该脚目标接触相位时才计分。真实接触和时长读每物理子步更新的接触传感器。新训练想象侧使用世界模型受监督的found接触概率，与滑移项共用接触来源；沿用最大持续耦合连续推进时长，概率恒定时不会反复假设离地／触地。该计时仍是可微近似。配置 `experiment.contact_source=model`、`actor_phase=True` 默认启用。没有足高目标；pose原始评分仍按指定go2_ppo版本以命令速度选择关节标准差，不引入HIM地形容差。

新训练 `reward.pose_tracking_gate=True`：姿态原始评分再乘速度跟踪门控，起点命令评价终点的body-xy／world-z速度。平移模长或偏航绝对值超过0.1的分量使用 `clamp(1-误差平方/目标平方,0,1)`，其余分量沿用对应指数跟踪容差，两分量门控相乘；站立命令门控为1。真实和想象使用同一可微公式，权重与dt仍各乘一次。设该开关False恢复原姿态评分。历史消融保持在冻结工作区，不通过当前入口迁移旧策略。

世界模型使用参考70维状态（含髋、膝、足坐标）及3维平面位姿，直接学习状态增量和旧航向系位姿增量。8步拟合的地形relief与位姿标签取记录状态，物理状态始终自回归；32步想象地形随预测位姿更新。模型和critic使用11×7特权扫描，actor不使用。

仿真上下文为摩擦、COM和编码器偏置；当前模型为239维输入、125维输出：保留原122维槽位（原状态73、qacc12、世界足速xy8、found4、四脚世界接触力12、实际执行器力矩12、最大腿部接触力模1），在122:125追加三维heading系角动量。前73维学习原状态／位姿增量，后52维监督下一时刻绝对量，found经sigmoid输出概率。角动量由已有 `root_angmom` 的世界向量绕yaw逆旋转，平方和不变，避免模型在缺少绝对yaw输入时预测世界向量。指定参考模型输入175维；旧187／73、211／97、235／121、236／122属于历史布局。任务记忆42维、critic262维，新actor仍为47／235维。

`leg_collision` 按指定Go2 HIM规则检测髋、大腿及小腿与地形的接触：任一几何体接触力模严格大于 `reward.leg_collision_force_threshold`（默认1 N），整环境奖励率扣1，不累计碰撞数量，不含足端正常着地和自碰，不新增终止条件。真实端取 `leg_ground_contact` 的单帧历史，想象端预测同一最大力模并使用相同硬阈值；与来源一致，该硬判定没有对力预测的直接局部梯度，控制效果尚未训练验证。

机器人采用参考Menagerie模型和碰撞材料；执行器按用户指定的 `legged_wbc_mjlab` Go2 配置改为 `BuiltinPositionActuatorCfg`，装配定义见[Go2 资产与执行器](../src/assets/robots/go2/go2_constants.py)。动作比例使用当前集中配置中按关节类型显式设置的值。动作输出无界，目标为默认关节角加缩放动作；执行器延迟同步采用指定位置执行器配置，模型只显式接收当前及上一动作，隐藏延迟队列是其近似限制。reset、Newton求解器、三轴摩擦、encoder bias与COM随机化按指定参考及其MjLab1.6基类配置，无外力push。具体数值仅在集中配置维护。

## 保存、续训、播放与评估

产物位于 `logs/supermimic/go2_supermimic/<运行标识>/`，包含 `model_N.pt`（参数、优化器与计数）、`policy.onnx`（最近保存点的部署推理网络）、`events.out.tfevents.*`（TensorBoard逐轮标量与运行来源）和 `task_config.json`（任务快照，播放入口与续训依赖）。每 `--save-interval` 轮及最后一轮同步保存模型并覆盖当前ONNX；保留参考塌缩护栏，存在预警时跳过该保存点，继续保留上次模型与ONNX。ONNX元数据记录轮次和任务指纹。终端仍逐 `--log-interval` 轮打印原80列摘要；曲线操作见本地 [TensorBoard指南](work/TENSORBOARD_REMOTE.md)。

新启动的训练另在 `params/` 集中保存 `env.yaml`（覆盖后的任务、奖励、命令及仿真参数）、`agent.yaml`（算法、实际runner预算、执行分批与驱动配置）、`run.yaml`（命令、原始CLI覆盖、实际种子／环境数、依赖版本、续训来源和代码状态）及 `git.diff`（已跟踪文件相对HEAD的差异）。CLI中的空覆盖值表示采用配置，实际运行预算查看`agent.yaml`的`runner`和`execution`。`task_config.json` 仍用于播放与续训；这些YAML供查看和比较实验配置，不直接作为 `--task-config` 输入。事件文件新增 `Loss/Train/Perf` 常用分组、速度RMSE、回合结束比例、回报分布、回放占用、actor／世界模型／critic学习率；原始tag保留。

终端每轮显示 `Reward_Rate/<name>`：本轮真实采集中所有环境的原始评分均值乘配置权重，不乘控制周期，从首次采集起就有实际数值，不等待回合结束、也不使用占位或旧值。当前两个速度跟踪项的上限均为2；所有分项相加再乘控制周期，等于本轮平均单步奖励。该实时口径是用户指定的显示适配，不能当作HIM已结束回合统计。`Mean reward` 与 `Mean episode length` 仍为最近最多100个已结束回合的均值，无历史回合时省略这两行。TensorBoard逐轮增加同名 `Reward_Rate/*` 标量；原有 `task/reward/*` 单步贡献、`task/reward_raw/*` 原始评分及 `Episode_Reward/*` 回合统计继续保留，后者仍为结束回合的累计加权奖励除以最大回合时长、仅有数据时写入。新格式在下次启动或续训时生效，具体口径见本地[日志设计](work/design/go2_blind_supermimic.md#2026-10-29奖励分项对齐-go2-him)。

新部署策略输入`[B,235]`、输出`[B,12]`，默认图内包含双路归一化、8MoE编码器和动作MLP。heading目标采样、相位生成和历史维护均由调用方在图外完成；元数据记录实际布局、时钟／站立门控、关节顺序、默认姿态、动作比例和命令坐标系。需要带轮次的PT／ONNX／JSON三件套时可显式导出：

```bash
.venv/bin/python -B tools/export_supermimic_policy.py '<实验目录>/model_1000.pt' \
  --output-dir .artifacts/deploy/go2
```

```bash
.venv/bin/python -u scripts/train.py Unitree-Go2-SuperMimic-Flat \
  --device cuda:0 --resume '<实验目录>/model_1000.pt' --continue-run True --max-iterations 1500

.venv/bin/python scripts/play.py Unitree-Go2-SuperMimic-Flat \
  --checkpoint-file '<实验目录>/model_1000.pt' --num-envs 1 --viewer viser --device cpu
```

播放默认关闭回合超时重置，保留跌倒和手动重置；`--no-terminations True` 可进一步关闭跌倒重置。

`--max-iterations`是目标总轮数。`--resume`仅接受新格式检查点，默认热启动权重并从第1轮重新计数；`--continue-run True`用于相同任务接续检查点轮次。恢复会重建回放，不恢复仿真轨迹或随机状态。播放自动读取同目录完整快照；参数见`--help`。

`src.tasks.locomotion.go2_supermimic.rl.evaluation.evaluate_policy` 可按前进、后退、侧移、转向四组固定命令报告真实误差、跌倒、存活及部署对拍；对拍用当场导出的 TorchScript，不依赖训练目录里的策略副本。该四命令检查不是论文附录G.2的672命令步响应协议。

## 验证

本地行为测试：

```bash
OMP_NUM_THREADS=1 .venv/bin/python -B -m unittest discover -s tests -p 'test_moe_velocity*.py' -v
```

显式真实仿真冒烟及训练验收使用`tests/test_training/moe_velocity_training.py`，可选择`--robot go2`或`--robot ri4438`：

```bash
OMP_NUM_THREADS=1 .venv/bin/python -B -m tests.test_training.moe_velocity_training \
  --robot go2 --mode smoke --output .artifacts/moe_velocity_smoke/go2
OMP_NUM_THREADS=1 .venv/bin/python -B -m tests.test_training.moe_velocity_training \
  --robot ri4438 --mode full --output .artifacts/moe_velocity_full/ri4438
```

完整模式每台机器人累计实际训练至少660秒，保留完整actor、8步模型监督与32步想象，缩减环境／批量和世界模型宽度以验证链路；详细预算与断言见模块说明。普通测试不自动启动完整训练。测试、工作记录与产物由Git忽略；验证不能代替收敛或实机验收。

## 配置版本

检查点／部署格式为 `supermimic_moe_velocity_v1`／`supermimic_moe_velocity_policy_v1`。旧32维MoE、4MoE／2MoE历史模型、HIM、单MLP及CTS蒸馏检查点不兼容，当前入口明确拒绝，不做权重填充或旧网络装配；历史日志、检查点和导出文件保留。已删除失效的预训练MoECTS权重映射及旧配置字段迁移，配置只接受当前声明字段。

外部策略驱动通过任务配置 `runner.driver` 提供：`policy` 指向 go2_rl_gym 的 MoE-CTS 导出文件，`expert_policy` 指向本项目同观测接口的MoE检查点，`frac`/`expert_frac` 为环境占比，`fade_iterations` 控制线性衰减。驱动环境位于环境尾部切片，只接管动作、跟随本任务速度命令，只贡献真实数据，不参与策略更新。

算法位于平铺的`agents/supermimic/`，任务配置段为`algo`。同结构新格式actor可以热启动；奖励变化时需显式当前`--task-config`并按需使用`--reset-critic True`。SuperMimic入口只接受当前扁平参数，历史配置及旧网络权重不自动转换。
