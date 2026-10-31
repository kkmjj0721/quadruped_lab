<!-- Copyright (c) 2026-2027 zh -->

# Go2 SuperMimic 盲速度跟踪

文中的本地工作记录位于 `docs/work/`，不随仓库提交。

默认训练对齐指定 [Go2-SuperMimic](https://github.com/Zhengtianjin/go2-supermimic/tree/d9eee9b3f248e5300e636c433255db2c1c3e086a) 的训练结构和物理随机化，地形采用下述 Ri-5036 HIM 配置，actor 改用 [go2_rl_gym](https://github.com/wty-yy/go2_rl_gym/tree/30e74dc507bec7a642a8c98be26081f2c6f0822d) 的 MoECTS 学生部署网络。当前二十一项速度奖励以 [SuperMimic 英文 v1](../paper/SuperMimic_2610.28480v1.pdf) 第16页表 A.6 为起点，按用户要求将平移跟踪改为 HIM 的机体系 xy、偏航指数分母改为0.25并删除新训练侧倾项，增加下述 HIM 接触、角动量、髋姿态和落地冲击约束。命令采用 HIM 的 heading 逻辑；策略读取本体历史、速度命令和步态相位，线速度由历史隐式估计。

这属于用户确认的从零 Go2 速度控制迁移。原论文速度接口与动作模仿联合训练，参考仓库任务是追逐和闪避；本项目没有复现其动作库先验或演示成绩。集中配置见 [Go2ReferenceCfg](../src/config/go2/go2_reference_supermimic_config.py)，算法来源见 [NOTICE](../agents/supermimic/NOTICE.md)，资产来源见 [参考资产说明](../src/assets/robots/go2/REFERENCE_NOTICE.md)。本地工作记录中，与本仓库的逐项差异见[对比报告](work/analysis/go2_supermimic_vs_reference.md)，与相邻 `go2_him` 任务在命令采样、MDP 组装、奖励终止上的对齐评估见[任务层对齐评估](work/analysis/go2_supermimic_vs_go2_him.md)。

## 训练

```bash
.venv/bin/python -u scripts/train.py Unitree-Go2-SuperMimic-Flat \
  --device cuda:0 --max-iterations 40000 --save-interval 1000
```

这是显式完整训练命令，不随单元测试启动。粗糙地形任务为 `Unitree-Go2-SuperMimic-Rough`，地形生成配置来自本地 `legged_wbc_mjlab@4eb8fcf71124b20ed4a1154b6c724ba962a424f8` 的 `src/tasks/locomotion/ri_5036_him/ri_5036_him_env_cfg.py:578–615`：8×8米地块、6行、20米外围平地，包含平地、0.8米箱体网格及四种正／四种倒金字塔楼梯。配置保留 `num_cols=20`，但 MjLab 1.6 课程模式按每种子地形一列实际生成6×10；`seed=None`与来源一致，需固定地图时显式设置地形种子。地形布局、高程查询和扫描布局保持不变；难度升降采用同提交 `go2_him/go2_him_env_cfg.py` 转导的 `ri_4438_him/mdp/curriculums.py:terrain_levels_vel`：距地块原点超过半块长度且期望命令距离至少1米时晋级，实际净位移不足期望距离一半时降级，晋级优先；期望距离包含历史世界系命令位移和提前结束时剩余时长的要求。越过最高级时随机回落到现有等级，最低等级为零；仅真实结束回合结算，手动重置不升降。保留零级训练起点，不再按训练进度开放难度，`terrain.curriculum_iterations` 仅保留为旧配置字段。粗糙地形 actor 仍然没有高程输入，不能继承参考有高程策略的效果。

世界模型拟合预算对齐论文表A.5：每轮10批2048个8步窗口；想象批量仍采用参考仓库 `logs/supermimic/2026-10-27_17-09-24_flat/model_1000.pt` 中保存的4096条32步想象。真实采集按用户指定的 [zita-ch/supermimic_vibed@8bfc115](https://github.com/zita-ch/supermimic_vibed/blob/8bfc1158bd968f729fcc65d5e897a05b05ea3d36/supermimic/train.py#L300-L322)：`algo.collect_steps=2.5` 表示每轮平均控制步数，对应UTD=0.4；循环累加后取整，按2、3步交替，模型10批更新另计。

其余预算沿用指定Go2参考源码：4096环境、1000步/环境回放；世界模型1024³；300轮模型预热；actor动作MLP与critic均为ELU(512,256,128)。世界模型网络等设置仍与论文表A.4/A.5不同，其余参考预算按用户选择保留。训练循环外仍独立预填10步；累加器跨分段训练保留余量，重置采集或恢复时重新开始。回放直接保存完整半精度历史。旧检查点续训默认读取原 `task_config.json` 中的整数采集预算；若要按新节奏继续，显式提供含 `{"algo":{"collect_steps":2.5}}` 的 `--task-config`，并按原入口要求提供其他需要保留的任务参数。日志 `collection/control_steps` 为本轮实际步数，吞吐按真实转移差计算。

`--num-envs` 只改变环境数。默认 `--imag-microbatch 4096 --critic-microbatch 32768` 覆盖整个想象批量和每个critic小批：actor每轮整批反传一次，critic按5轮×4小批反传并更新20次。显式调小这两个上限时才分段，每段仍完整反传32步，总批量和优化器更新次数不变。配置不根据显卡自动缩减。整批执行的峰值显存高于分段执行；8 GB显卡应先运行下文显式GPU检查，再按实际容量设定实验规模；短检查不证明默认4096环境的全预算能运行。修改默认值只影响新进程；续训或显式加载JSON时，其中保存的批量仍覆盖默认值。

其他预算通过JSON显式覆盖，例如：

```json
{
  "algorithm": "moe_history",
  "env": {"num_envs": 256},
  "algo": {"replay_capacity": 256, "archive_size": 10000, "wm_batch": 256, "imag_batch": 256}
}
```

使用 `--task-config <配置.json>`。这是独立缩减实验，运行快照保留差异；不是论文或参考仓库的完整训练预算。命令行环境数/seed覆盖JSON，JSON覆盖默认值。动作输出固定为无界线性输出，不提供动作限幅配置。

每轮依次真实采集、世界模型自回归拟合、冻结模型权重并通过想象回报更新actor、以固定lambda-return拟合critic。保留参考OU探索、接触变化优先采样、稀有事件归档、学习率调度和保存点回滚。回滚只恢复actor/critic并重建其优化器，不恢复物理仿真。

## 任务与模型

新训练actor单帧47维：机体系角速度3、重力投影3、带编码器偏置的关节角偏移12、关节速度12、上一下发动作12、速度命令3、步态相位sin／cos两维。历史为5帧235维，按旧到新逐帧排列，reset重复首帧。相位在原45列末尾追加，不采样噪声；时钟来自回合控制步，与步态奖励共用周期，命令平移模长加偏航绝对值不超过0.1时两维清零。仍不输入真实线速度、地形、接触或仿真参数；线速度由历史隐式估计。

actor为混合专家历史编码器：235维历史经共享ELU骨干(512,256)与按专家分组的1×1卷积得到8个32维专家输出，Softmax门控加权求和后`L2Norm`投影为32维隐变量；隐变量与独立归一化的当前帧47维拼接，经ELU MLP(512,256,128)输出12维未缩放动作。核心网络对齐 go2_rl_gym 提交 `30e74dc`，只用 SuperMimic 想象回报训练，不加入 CTS 的教师、蒸馏或负载均衡损失。旧45维／225维策略按其保存配置装配；两种观测宽度不能直接互换权重。

命令采样以 `legged_wbc_mjlab@e6209ec` 的 `go2_him` 为来源；新训练默认vx/vy各为[-1,1]m/s，wz配置范围为[-2,2]rad/s，每4–12秒重采样。此前±4范围在两次训练诊断后收回；当前没有自动扩展到±4的速度课程。heading目标范围±3.14rad，`wz=clip(0.5*wrap(target-yaw),-2,2)`，因此该heading反馈实际产生的wz最多约±1.57rad/s；范围配置不改变反馈增益。heading比例1、站立比例0.1、forward比例0.05。保留来源实际顺序：forward先约束vx≥0.3和vy=0，随后heading会重新计算wz；站立最后把三个命令清零。真实与想象分别用真实／预测yaw更新命令。当前平移奖励评价机体系xy，偏航仍评价world z角速度；旧完整快照保留heading平移定义，部署及评估元数据显式记录各自坐标系。

当前默认二十一项奖励与权重集中维护在 `Go2ReferenceCfg.reward`，不在此重复参数清单。`track_linear_velocity` 使用body-xy，`track_angular_velocity` 使用world-z，两者指数分母均0.25、权重均2；新训练不计算 `torso_sideways_lean`。所有奖励率仅乘一次0.02秒。`body_ang_vel` 按HIM计算机体系x、y角速度平方和，权重仍为−0.05；`reward.body_ang_vel_frame=body` 标识新默认，旧完整快照缺少该字段时保留原heading定义与指纹。Go2终止仍取参考70度倾倒和回合超时，想象用5度宽sigmoid，不增加地图出界终止。

`lin_vel_z` 使用机身坐标系速度z的平方，倾斜时与世界／heading系z不同；由现有速度与重力解析恢复，不增加世界模型输出。`body_orientation_l2` 为机身重力投影xy平方和。`smoothness` 使用未缩放下发动作的 `sum((a_t-2*a_(t-1)+a_(t-2))**2)`，不除以dt，reset时两级动作历史清零。三项同时用于真实奖励和可微想象更新；权重以用户指定为准，其中smoothness并非来源配置的-0.015。

`stand_still` 为终点默认关节角偏差平方和，`stand_still_joint_vel_l2` 为终点关节速度平方和，两项均在起点命令平移模长加偏航绝对值不超过0.1时启用。速度项直接约束小幅持续摆腿，真实与想象共用公式，复用已有22:34关节速度槽位，也会对站立平衡动作施加代价。零命令下步态／腾空项关闭。`foot_slip` 在命令大小超过0.1时，按逐脚世界xy瞬时速度平方和乘found接触后求和；`joint_acc_l2` 直接使用仿真器qacc平方和。真实端在adapter重置前读取瞬时量；想象端预测48维连续辅助量和4维接触概率，不使用控制周期差分替代瞬时量。接触logit输出统计固定均值0、方差1，概率残差沿用模型MSE；预测量及接触概率仍须单独验证。

`stumble` 使用 `feet_ground_contact` 的世界净接触力：任一脚满足 `norm(force_xy) > stumble_ratio * abs(force_z)` 时为1，否则为0，默认比值5.0；保持来源的严格硬阈值，不增加接触门控或平滑替换。该指标在想象端对接触力没有直接局部梯度。`joint_torques_l2` 为终点实际 `actuator_force` 的平方和，想象端对预测力矩使用相同公式并可反传；新增预测量的准确性及控制效果尚需训练验证。

`angular_momentum` 为根部子树角动量三轴平方和，heading旋转不改变HIM来源的world范数；`hip_pos` 仅在侧移及偏航命令绝对值均不超过0.1时计算四个髋关节默认角偏差平方和，快速直行仍启用。`soft_landing` 在命令大小超过0.1时，对首次触地脚的净接触力模求和；真实端的首次触地为 `0 < current_contact_time < dt + 1e-6`。想象端按用户确认方案，以计时记忆除以接触概率恢复条件时长，再使用同一硬窗口并乘接触概率；这是概率加权代理，力头为无条件量，不能视为严格无偏的冲击期望。触地窗口无直接局部梯度，接触力模及概率幅值可反传，预测准确性和控制效果尚需训练验证。

步态周期0.6秒、支撑比例0.56；按来源FR/FL/RR/RL足序的相位偏移为[0,.5,.5,0]，本项目按足名映射。腾空奖励使用逐脚三角时长得分，目标支撑0.336秒、腾空0.264秒；只有符合该脚目标接触相位时才计分。真实接触和时长读每物理子步更新的接触传感器。新训练想象侧使用世界模型受监督的found接触概率，与滑移项共用接触来源；沿用最大持续耦合连续推进时长，概率恒定时不会反复假设离地／触地。该计时仍是可微近似。配置 `experiment.contact_source=model`、`actor_phase=True` 默认启用；旧完整快照显式恢复原开关。没有足高目标；pose仍采用指定go2_ppo版本按命令速度选择关节标准差，不引入HIM地形容差。

世界模型使用参考70维状态（含髋、膝、足坐标）及3维平面位姿，直接学习状态增量和旧航向系位姿增量。8步拟合的地形relief与位姿标签取记录状态，物理状态始终自回归；32步想象地形随预测位姿更新。模型和critic使用11×7特权扫描，actor不使用。

仿真上下文为摩擦、COM和编码器偏置；当前模型为239维输入、125维输出：保留原122维槽位（原状态73、qacc12、世界足速xy8、found4、四脚世界接触力12、实际执行器力矩12、最大腿部接触力模1），在122:125追加三维heading系角动量。前73维学习原状态／位姿增量，后52维监督下一时刻绝对量，found经sigmoid输出概率。角动量由已有 `root_angmom` 的世界向量绕yaw逆旋转，平方和不变，避免模型在缺少绝对yaw输入时预测世界向量。指定参考模型输入175维；旧187／73、211／97、235／121、236／122属于历史布局。任务记忆42维、critic262维，新actor仍为47／235维。

`leg_collision` 按指定Go2 HIM规则检测髋、大腿及小腿与地形的接触：任一几何体接触力模严格大于 `reward.leg_collision_force_threshold`（默认1 N），整环境奖励率扣1，不累计碰撞数量，不含足端正常着地和自碰，不新增终止条件。真实端取 `leg_ground_contact` 的单帧历史，想象端预测同一最大力模并使用相同硬阈值；与来源一致，该硬判定没有对力预测的直接局部梯度，控制效果尚未训练验证。

机器人采用参考Menagerie模型和碰撞材料；执行器按用户指定的 `legged_wbc_mjlab` Go2 配置改为 `BuiltinPositionActuatorCfg`，来源见[资产说明](../src/assets/robots/go2/REFERENCE_NOTICE.md)。动作比例使用当前集中配置中按关节类型显式设置的值。动作输出无界，目标为默认关节角加缩放动作；执行器延迟同步采用指定位置执行器配置，模型只显式接收当前及上一动作，隐藏延迟队列是其近似限制。reset、Newton求解器、三轴摩擦、encoder bias与COM随机化按指定参考及其MjLab1.6基类配置，无外力push。具体数值仅在集中配置维护。

## 保存、续训、播放与评估

产物位于 `logs/supermimic/go2_supermimic/<运行标识>/`，包含 `model_N.pt`（参数、优化器与计数）、`policy.onnx`（最近保存点的完整actor）、`events.out.tfevents.*`（TensorBoard逐轮标量与运行来源）和 `task_config.json`（任务快照，播放入口与续训依赖）。每 `--save-interval` 轮及最后一轮同步保存模型并覆盖当前ONNX；保留参考塌缩护栏，存在预警时跳过该保存点，继续保留上次模型与ONNX。ONNX元数据记录轮次和任务指纹。终端仍逐 `--log-interval` 轮打印原80列摘要；曲线操作见本地 [TensorBoard指南](work/TENSORBOARD_REMOTE.md)。

新启动的训练另在 `params/` 集中保存 `env.yaml`（覆盖后的任务、奖励、命令及仿真参数）、`agent.yaml`（算法预算、执行分批与驱动配置）、`run.yaml`（命令、实际种子／环境数、依赖版本、续训来源和代码状态）及 `git.diff`（已跟踪文件相对HEAD的差异）。`task_config.json` 仍用于播放与续训；这些YAML供查看和比较实验配置，不直接作为 `--task-config` 输入。事件文件新增 `Loss/Train/Perf` 常用分组、速度RMSE、回合结束比例、回报分布、回放占用和三个优化器的实际学习率；原始tag保留。

终端每轮显示 `Reward_Rate/<name>`：本轮真实采集中所有环境的原始评分均值乘配置权重，不乘控制周期，从首次采集起就有实际数值，不等待回合结束、也不使用占位或旧值。当前两个速度跟踪项的上限均为2；所有分项相加再乘控制周期，等于本轮平均单步奖励。该实时口径是用户指定的显示适配，不能当作HIM已结束回合统计。`Mean reward` 与 `Mean episode length` 仍为最近最多100个已结束回合的均值，无历史回合时省略这两行。TensorBoard逐轮增加同名 `Reward_Rate/*` 标量；原有 `task/reward/*` 单步贡献、`task/reward_raw/*` 原始评分及 `Episode_Reward/*` 回合统计继续保留，后者仍为结束回合的累计加权奖励除以最大回合时长、仅有数据时写入。新格式在下次启动或续训时生效，具体口径见本地[日志设计](work/design/go2_blind_supermimic.md#2026-10-29奖励分项对齐-go2-him)。

新部署策略输入`[B,235]`、输出`[B,12]`，图内包含归一化、MoE编码器和动作MLP；旧45维策略仍导出`[B,225]`输入。heading目标采样、相位生成和历史维护均由调用方在图外完成；元数据记录实际布局、时钟／站立门控、关节顺序、默认姿态、动作比例和命令坐标系。需要带轮次的PT／ONNX／JSON三件套时可显式导出：

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

旧完整45维快照按原相位开关、接触来源和权重播放／续训，不自动升级为47维。新训练使用当前v7的125维物理状态／critic262；旧v4（97／234）、v5（121／258）及v6（122／259）完整快照保留原状态布局、heading平移、原偏航分母和历史侧倾奖励，缺失的新增奖励显式关闭，原语义指纹保持。启用本轮角动量、髋关节及落地奖励须显式使用当前配置。既有无足高47维实验快照可以加载，读取时移除其中零权重足高字段；非零足高实验继续使用原隔离工作区。

`--max-iterations` 是目标总轮数。`--resume` 默认热启动权重并从第1轮重新计数；`--continue-run True` 用于相同任务布局接续检查点轮次。旧73／97／121／122维模型迁入125维任务且actor观测宽度相同时，应显式提供当前 `--task-config`，保持 `--continue-run False` 并用 `--reset-critic True` 重新学习新奖励价值；actor可按原形状加载，模型／critic按现有形状检查重建并重新预热。来源使用绝对预热轮次，高轮次跨形状continue-run可能直接越过预热。恢复会重建回放，不恢复仿真轨迹或随机状态。播放自动读取同目录完整快照；参数见 `--help`。

`src.tasks.locomotion.go2_supermimic.rl.evaluation.evaluate_policy` 可按前进、后退、侧移、转向四组固定命令报告真实误差、跌倒、存活及部署对拍；对拍用当场导出的 TorchScript，不依赖训练目录里的策略副本。该四命令检查不是论文附录G.2的672命令步响应协议。

## 验证

```bash
GO2_SUPERMIMIC_REFERENCE='<固定提交的参考仓库>' \
GO2_RL_GYM_ROOT='<固定提交的go2_rl_gym仓库>' \
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -B -m unittest \
  tests.test_agents.test_reference_supermimic \
  tests.test_tasks.test_reference_supermimic_task \
  tests.test_tasks.test_go2_supermimic_driver \
  tests.test_scripts.test_supermimic_config -v

RUN_REFERENCE_CPU=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m unittest \
  tests.test_training.test_reference_supermimic_training -v

RUN_REFERENCE_GPU=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m unittest \
  tests.test_training.test_reference_supermimic_training -v

RUN_SUPERMIMIC_HIM_LONG=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m unittest \
  tests.test_training.test_supermimic_him_integration -v
```

常规测试不自动启动训练。显式CPU检查用4环境和小网络，flat/rough各4轮；GPU检查用完整参考网络、64环境、64个模型窗口和128条想象，保留8步拟合与32步想象。两者均检查实际参数更新、保存/恢复、播放和部署。算法/任务对拍分别读取 `GO2_RL_GYM_ROOT`（混合专家编码器）与 `GO2_SUPERMIMIC_REFERENCE`（参考任务语义），均指定上述固定提交仓库；缺少参考源码时该对拍项明确跳过。

长训练检查保持完整网络、64环境及缩减的采样批量，初始化和首次编译完成后实际训练660秒，结束后统一检查指标、检查点、ONNX和继续训练。源代码快照、配置与退出状态保留在独立验证产物中。实际执行与独立审查维护在本地 [同主题审查记录](work/review/go2_blind_supermimic.md)；运行检查只能证明对应环节可用，不能代替收敛与真机验收。

## 配置版本

当前只支持 `moe_history`（检查点与导出格式 `supermimic_moe_v1` / `supermimic_moe_policy_v1`）。旧的单 MLP 格式、历史 CTS 与蒸馏路径不再支持；变更前的配置和检查点需要重新训练，不做静默转换。论文、来源记录和历史实验产物保留用于追溯。

十六项奖励沿用v4状态布局：物理97、memory42、critic234，新奖励选择由配置指纹区分。已有十五项v4完整快照缺少站立速度权重时，读取边界显式补0，指纹省略禁用的新字段；原范围、奖励和检查点哈希保留，仍可按原语义播放／恢复。只改代码不会改变已有权重的动作，也不会给旧快照续训自动启用新项。采用新设置热启动时应显式提供当前 `--task-config`、保持 `--continue-run False`，奖励变化时用 `--reset-critic True` 重新学习价值；同形状世界模型和actor可保留。新JSON文件内容为 `{}` 即使用当前默认任务参数，也可按需局部覆盖。

此前十二项的物理73／critic210及九项的memory30／critic198快照不能直接作为新任务播放配置；按当前配置热启动时由既有形状检查重建世界模型／critic，旧actor仍可显式离线导出。此前扩展实验的代码和训练结果保存在冻结归档中，不反向改为当前主线结果；新设置的行走与停步效果需要重新训练验证。

外部策略驱动通过任务配置 `runner.driver` 提供：`policy` 指向 go2_rl_gym 的 MoE-CTS 导出文件，`expert_policy` 指向本项目同观测接口的MoE检查点，`frac`/`expert_frac` 为环境占比，`fade_iterations` 控制线性衰减。驱动环境位于环境尾部切片，只接管动作、跟随本任务速度命令，只贡献真实数据，不参与策略更新。

算法位于平铺的 `agents/supermimic/`，任务配置段为 `algo`。新增奖励、命令及地形改变了任务语义，旧任务快照不能直接作为本轮验收配置；如需旧MoE权重热启动，显式提供当前 `--task-config`，由参考形状检查决定加载或重新初始化相应网络，不能把它称为原实验的无变化继续训练。SuperMimic训练只接受本任务的扁平参数；旧 `--gpu-ids`、`--agent.*` 别名不再翻译。
