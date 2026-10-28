<!-- Copyright (c) 2026-2027 zh -->

# Go2 SuperMimic 盲速度跟踪

默认训练对齐指定 [Go2-SuperMimic](https://github.com/Zhengtianjin/go2-supermimic/tree/d9eee9b3f248e5300e636c433255db2c1c3e086a) 的训练结构、地形和物理随机化，actor 改用 [go2_rl_gym](https://github.com/wty-yy/go2_rl_gym/tree/30e74dc507bec7a642a8c98be26081f2c6f0822d) 的 MoECTS 学生部署网络。速度奖励保留 [SuperMimic 英文 v1](../paper/SuperMimic_2610.28480v1.pdf) 第16页表 A.6 六项，再增加指定 HIM 步态／腾空、垂向速度／姿态／动作二阶差分和 Go2 PPO 姿态奖励；命令采用 HIM 的 heading 逻辑。策略只读取本体历史，不含线速度输入；线速度由历史隐式估计。没有教师、蒸馏或 PPO 更新。

这属于用户确认的从零 Go2 速度控制迁移。原论文速度接口与动作模仿联合训练，参考仓库任务是追逐和闪避；本项目没有复现其动作库先验或演示成绩。集中配置见 [Go2ReferenceCfg](../src/config/go2/go2_reference_supermimic_config.py)，算法来源见 [NOTICE](../agents/supermimic/NOTICE.md)，资产来源见 [参考资产说明](../src/assets/robots/go2/REFERENCE_NOTICE.md)。与本仓库的逐项差异见[对比报告](analysis/go2_supermimic_vs_reference.md)，与相邻 `go2_him` 任务在命令采样、MDP 组装、奖励终止上的对齐评估见[任务层对齐评估](analysis/go2_supermimic_vs_go2_him.md)。

## 训练

```bash
.venv/bin/python -u scripts/train.py Unitree-Go2-SuperMimic-Flat \
  --device cuda:0 --max-iterations 40000 --save-interval 1000
```

这是显式完整训练命令，不随单元测试启动。粗糙地形任务为 `Unitree-Go2-SuperMimic-Rough`，采用参考的12m、5×7混合地形，按12000轮进度开放难度行。粗糙地形 actor 仍然没有高程输入，不能继承参考有高程策略的效果。

默认批量对齐参考仓库 `logs/supermimic/2026-10-27_17-09-24_flat/model_1000.pt` 中保存的实际运行配置：世界模型每轮2批1024个8步窗口，4096条32步想象。其余预算沿用指定参考源码：4096环境、每轮采1步、1000步/环境回放；世界模型1024³；300轮模型预热；actor动作MLP与critic均为ELU(512,256,128)。这些与论文表A.4/A.5的网络和批量不同，按用户选择使用参考配置。先在轮次循环外预填10步，再按每轮采1步训练；回放直接保存完整半精度历史。

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

actor单帧45维：机体系角速度3、重力投影3、带编码器偏置的关节角偏移12、关节速度12、上一下发动作12、速度命令3。历史为5帧225维，按旧到新逐帧排列，reset重复首帧。无相位、真实线速度、地形、接触或仿真参数输入；线速度只能由历史经混合专家编码隐式估计。

actor为混合专家历史编码器：225维历史经共享ELU骨干(512,256)与按专家分组的1×1卷积得到8个32维专家输出，Softmax门控加权求和后`L2Norm`投影为32维隐变量；隐变量与独立归一化的当前帧45维拼接，经ELU MLP(512,256,128)输出12维未缩放动作。核心网络对齐 go2_rl_gym 提交 `30e74dc`，只用 SuperMimic 想象回报训练，不加入 CTS 的教师、蒸馏或负载均衡损失。观测顺序、缩放和历史重置遵循本任务，不能把核心网络一致理解为完整预训练策略接口一致。

命令以 `legged_wbc_mjlab@e6209ec` 的 `go2_him` 为来源：vx/vy 各在[-1,1]m/s内采样，每4–12秒重采样；heading目标范围±3.14rad，`wz=clip(0.5*wrap(target-yaw),-2,2)`，heading比例1、站立比例0.1、forward比例0.05，无速度课程。保留来源实际顺序：forward先约束vx≥0.3和vy=0，随后heading会重新计算wz；站立最后把三个命令清零。真实与想象分别用真实／预测yaw更新命令。原六项奖励的平移仍按heading坐标、角速度按world yaw评价，这是保留的盲速度任务适配。

奖励保留表A.6六项：平移2、yaw2、动作差−0.3、关节限位−2、侧倾−4、roll/pitch角速度−0.05；添加 `foot_gait=0.5`、`feet_air_time=0.2`、`pose=1.0`、`lin_vel_z=-2.0`、`body_orientation_l2=-0.2`、`smoothness=-0.01`，共十二项。平移指数分母为0.25，yaw为0.5；所有奖励率仅乘一次0.02秒。Go2终止取参考70度倾倒和回合超时，想象用5度宽sigmoid，不增加地图出界终止。

`lin_vel_z` 使用机身坐标系速度z的平方，倾斜时与世界／heading系z不同；由现有速度与重力解析恢复，不增加世界模型输出。`body_orientation_l2` 为机身重力投影xy平方和。`smoothness` 使用未缩放下发动作的 `sum((a_t-2*a_(t-1)+a_(t-2))**2)`，不除以dt，reset时两级动作历史清零。三项同时用于真实奖励和可微想象更新；权重以用户指定为准，其中smoothness并非来源配置的-0.015。

步态周期0.6秒、支撑比例0.56；按来源FR/FL/RR/RL足序的相位偏移为[0,.5,.5,0]，本项目按足名映射。腾空奖励使用当前生效的逐脚三角时长得分，目标支撑0.336秒、腾空0.264秒；只有符合该脚目标接触相位时才计分。真实接触和时长读每物理子步更新的接触传感器。想象侧使用用户确认的足高软接触及连续计时，是可微近似，不能视为真实接触预测或已验证的效果改进。pose按命令速度选逐关节容差，采用指定 `go2_ppo` 版本而非HIM地形姿态版本；具体参数集中维护在配置中。

世界模型使用参考70维状态（含髋、膝、足坐标）及3维平面位姿，直接学习状态增量和旧航向系位姿增量。8步拟合的地形relief与位姿标签取记录状态，物理状态始终自回归；32步想象地形随预测位姿更新。模型和critic使用11×7特权扫描，actor不使用。

仿真上下文为摩擦、COM和编码器偏置；保留现有187维模型输入与73维输出，不增加接触预测头。指定参考源码只传摩擦与COM，模型输入为175维；此差异是保留的盲任务接口。任务记忆42维，保存heading目标、采样模式、逐脚接触和时长及前两步动作；critic210维，额外获得这些任务条件及周期相位，actor仍为45/225维。

机器人采用参考Menagerie模型、直流电机、碰撞材料和动作比例 `0.25 × effort_limit / stiffness`。动作输出无界，目标为默认关节角加缩放动作；0～2物理步延迟沿用参考，模型只显式接收当前及上一动作，隐藏延迟队列是其近似限制。reset、Newton求解器、三轴摩擦、encoder bias与COM随机化按指定参考及其MjLab1.6基类配置，无外力push。具体数值仅在集中配置维护。

## 保存、续训、播放与评估

产物位于 `logs/supermimic/go2_supermimic/<运行标识>/`，包含 `model_N.pt`（参数、优化器与计数）、`policy.onnx`（最近保存点的完整actor）、`events.out.tfevents.*`（TensorBoard逐轮标量与运行来源）和 `task_config.json`（任务快照，播放入口与续训依赖）。每 `--save-interval` 轮及最后一轮同步保存模型并覆盖当前ONNX；保留参考塌缩护栏，存在预警时跳过该保存点，继续保留上次模型与ONNX。ONNX元数据记录轮次和任务指纹。终端仍逐 `--log-interval` 轮打印原80列摘要；曲线操作见 [TensorBoard指南](../doc/TENSORBOARD_REMOTE.md)。

部署策略输入 `[B,225]`、输出 `[B,12]`，图内包含归一化、MoE编码器和动作MLP；heading目标采样在图外生成三维速度命令，历史也由调用方维护。关节顺序、默认姿态、动作比例和命令坐标系保存在元数据中。需要带轮次的PT／ONNX／JSON三件套时可显式导出：

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

`--max-iterations` 是目标总轮数。与参考一致，`--resume` 默认热启动权重并从第1轮重新计数；`--continue-run True` 接续检查点轮次，上例继续至1500轮。`--reset-critic True` 重新初始化critic并按参考冻结actor预热；模型形状变化时按参考重新预热。恢复会重建回放，不恢复仿真轨迹或随机状态。播放自动读取同目录完整快照；参数见 `--help`。

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

长训练检查保持完整网络、64环境及缩减的采样批量，初始化和首次编译完成后实际训练660秒，结束后统一检查指标、检查点、ONNX和继续训练。源代码快照、配置与退出状态保留在独立验证产物中。实际执行与独立审查维护在 [同主题审查记录](review/go2_blind_supermimic.md)；运行检查只能证明对应环节可用，不能代替收敛与真机验收。

## 配置版本

当前只支持 `moe_history`（检查点与导出格式 `supermimic_moe_v1` / `supermimic_moe_policy_v1`）。旧的单 MLP 格式、历史 CTS 与蒸馏路径不再支持；变更前的配置和检查点需要重新训练，不做静默转换。论文、来源记录和历史实验产物保留用于追溯。

十二项奖励对应任务语义v3：memory42、critic210。此前九项奖励的memory30／critic198快照不能直接作为新任务的续训或播放配置；若热启动旧MoE权重，应显式提供当前 `--task-config`，actor与世界模型可按原形状加载，critic按既有流程重新初始化。新JSON文件内容为 `{}` 即使用当前默认任务参数，也可按需局部覆盖。旧actor仍可用显式离线导出工具导出。改代码不会热更新已运行进程，它仍使用启动时加载的奖励；新增三项在新进程中生效。此前九项奖励的长训练结果不作为这次新增奖励的训练效果证据。

外部策略驱动通过任务配置 `runner.driver` 提供：`policy` 指向 go2_rl_gym 的 MoE-CTS 导出文件，`expert_policy` 指向本项目同观测接口的MoE检查点，`frac`/`expert_frac` 为环境占比，`fade_iterations` 控制线性衰减。驱动环境位于环境尾部切片，只接管动作、跟随本任务速度命令，只贡献真实数据，不参与策略更新。

算法位于平铺的 `agents/supermimic/`，任务配置段为 `algo`。新增奖励、命令及地形改变了任务语义，旧任务快照不能直接作为本轮验收配置；如需旧MoE权重热启动，显式提供当前 `--task-config`，由参考形状检查决定加载或重新初始化相应网络，不能把它称为原实验的无变化继续训练。SuperMimic训练只接受本任务的扁平参数；旧 `--gpu-ids`、`--agent.*` 别名不再翻译。
