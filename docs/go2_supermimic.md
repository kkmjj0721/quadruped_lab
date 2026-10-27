<!-- Copyright (c) 2026-2027 zh -->

# Go2 SuperMimic 盲速度跟踪

默认训练对齐指定 [Go2-SuperMimic](https://github.com/Zhengtianjin/go2-supermimic/tree/d9eee9b3f248e5300e636c433255db2c1c3e086a) 的训练结构，actor 改用 [go2_rl_gym](https://github.com/wty-yy/go2_rl_gym/tree/30e74dc507bec7a642a8c98be26081f2c6f0822d) 的混合专家历史编码器，速度奖励采用 [SuperMimic 英文 v1](../paper/SuperMimic_2610.28480v1.pdf) 第16页表 A.6。策略只读取本体历史，不含线速度输入；线速度由历史隐式估计。没有教师、蒸馏或 PPO 更新。

这属于用户确认的从零 Go2 速度控制迁移。原论文速度接口与动作模仿联合训练，参考仓库任务是追逐和闪避；本项目没有复现其动作库先验或演示成绩。集中配置见 [Go2ReferenceCfg](../src/config/go2/go2_reference_supermimic_config.py)，算法来源见 [NOTICE](../agents/supermimic/single/NOTICE.md)，资产来源见 [参考资产说明](../src/assets/robots/go2/REFERENCE_NOTICE.md)。

## 训练

```bash
.venv/bin/python -u scripts/train.py Unitree-Go2-SuperMimic-Flat \
  --device cuda:0 --max-iterations 40000 --save-interval 1000
```

这是显式完整训练命令，不随单元测试启动。粗糙地形任务为 `Unitree-Go2-SuperMimic-Rough`；建议先完成平地跟踪验收。两者保留本项目地形资源，粗糙地形 actor 仍然没有高程输入，不能继承参考有高程策略的效果。

默认预算取参考仓库：4096环境、每轮采1步、1000步/环境回放；世界模型1024³，每轮2批2048个8步窗口；8192条32步想象；300轮模型预热；actor/critic均为ELU MLP(512,256,128)。这些与论文表A.4/A.5的网络和批量不同，按用户选择使用参考配置。首次采16步以准备历史和窗口。

`--num-envs` 只改变环境数。`--imag-microbatch 256 --critic-microbatch 4096` 限制同时反传的样本数，不改变总批量或优化器更新次数；各想象分段仍完整反传32步。配置不根据显卡自动缩减。8 GB显卡应先运行下文显式GPU检查，再按实际容量设定实验规模；短检查不证明默认4096环境的全预算能运行。

其他预算通过JSON显式覆盖，例如：

```json
{
  "algorithm": "moe_history",
  "env": {"num_envs": 256},
  "single": {"replay_capacity": 256, "archive_size": 10000, "wm_batch": 256, "imag_batch": 256}
}
```

使用 `--task-config <配置.json>`。这是独立缩减实验，运行快照保留差异；不是论文或参考仓库的完整训练预算。命令行环境数/seed覆盖JSON，JSON覆盖默认值。动作输出固定为无界线性输出，不提供动作限幅配置。

每轮依次真实采集、世界模型自回归拟合、冻结模型权重并通过想象回报更新actor、以固定lambda-return拟合critic。保留参考OU探索、接触变化优先采样、稀有事件归档、学习率调度和保存点回滚。回滚只恢复actor/critic并重建其优化器，不恢复物理仿真。

## 任务与模型

actor单帧45维：机体系角速度3、重力投影3、带编码器偏置的关节角偏移12、关节速度12、上一下发动作12、速度命令3。历史为5帧225维，按旧到新逐帧排列，reset重复首帧。无相位、真实线速度、地形、接触或仿真参数输入；线速度只能由历史经混合专家编码隐式估计。

actor为混合专家历史编码器：225维历史经共享ELU骨干(512,256)与按专家分组的1×1卷积得到8个32维专家输出，Softmax门控加权求和后`L2Norm`投影为32维隐变量；隐变量与独立归一化的当前帧45维拼接，经ELU MLP(512,256,128)输出12维未缩放动作。负载均衡损失（系数0.01）按整批门控使用率与策略目标共同反传。结构、键名与超参对齐 go2_rl_gym 提交 `30e74dc`，可用其预训练权重热启动。

命令为航向坐标系平移速度与世界坐标系yaw角速度；每5秒重采样，范围取论文附录G的均匀命令支路，10%站立。没有额外速度课程或参考动作命令支路。

奖励严格保留表A.6六项：平移2、yaw2、动作差−0.3、关节限位−2、侧倾−4、roll/pitch角速度−0.05。平移指数分母为0.25，yaw为0.5；奖励率仅乘一次0.02秒。侧倾不惩罚纯前倾。Go2终止取参考70度倾倒，想象用5度宽sigmoid；不套用G1的0.35米高度条件，不附加跌倒奖励。超时与地图边界是运行边界。

世界模型使用参考70维状态（含髋、膝、足坐标）及3维平面位姿，直接学习状态增量和旧航向系位姿增量。8步拟合的地形relief与位姿标签取记录状态，物理状态始终自回归；32步想象地形随预测位姿更新。模型和critic使用11×7特权扫描，actor不使用。

仿真上下文为摩擦、COM和编码器偏置。MjLab会把编码器偏置用于执行器目标修正，因此依论文要求向模型与critic补齐这一输入；指定参考源码只传摩擦与COM。COM偏移相对未随机化MJCF读取，避免错误取随机化后的基准。模型输入187维、输出73维；critic179维。

机器人采用参考Menagerie模型、直流电机、碰撞材料和动作比例 `0.25 × effort_limit / stiffness`。动作输出无界，目标为默认关节角加缩放动作；0～2物理步延迟沿用参考，模型只显式接收当前及上一动作，隐藏延迟队列是其近似限制。摩擦范围取Go2参考，编码器/COM范围取论文表A.3；无外力push。具体数值仅在集中配置维护。

## 保存、续训、播放与评估

产物位于 `logs/supermimic/go2_supermimic/<运行标识>/`。`model_N.pt`保存参数、优化器与计数；同时生成 `policy_N.pt/onnx/json` 及最新 `policy.pt/onnx/json`。部署输入 `[B,225]`、输出 `[B,12]`，图内包含历史置换、归一化和MLP；关节顺序、默认姿态、动作比例、命令坐标系保存在元数据中。

```bash
.venv/bin/python -u scripts/train.py Unitree-Go2-SuperMimic-Flat \
  --device cuda:0 --resume '<实验目录>/model_1000.pt' --max-iterations 500

.venv/bin/python scripts/play.py Unitree-Go2-SuperMimic-Flat \
  --checkpoint-file '<实验目录>/model_1000.pt' --num-envs 1 --viewer viser --device cpu
```

续训轮数是追加轮数；重建回放并重新预热，不恢复仿真轨迹。播放自动读取同目录完整快照，关闭观测噪声与域随机化。支持 `--agent zero/random`、`--viewer native` 和录像参数，详细参数见 `--help`。

`src.tasks.locomotion.go2_supermimic.rl.evaluation.evaluate_policy` 可按前进、后退、侧移、转向四组固定命令报告真实误差、跌倒、存活及部署对拍。该四命令检查不是论文附录G.2的672命令步响应协议。

## 验证

```bash
GO2_SUPERMIMIC_REFERENCE='<固定提交的参考仓库>' \
GO2_RL_GYM_ROOT='<固定提交的go2_rl_gym仓库>' \
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -B -m unittest \
  tests.test_agents.test_reference_supermimic \
  tests.test_tasks.test_reference_supermimic_task \
  tests.test_tasks.test_go2_supermimic_driver \
  tests.test_scripts.test_single_supermimic_config -v

RUN_REFERENCE_CPU=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m unittest \
  tests.test_training.test_reference_supermimic_training -v

RUN_REFERENCE_GPU=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m unittest \
  tests.test_training.test_reference_supermimic_training -v
```

常规测试不自动启动训练。显式CPU检查用4环境和小网络，flat/rough各4轮；GPU检查用完整参考网络、64环境、64个模型窗口和128条想象，保留8步拟合与32步想象。两者均检查实际参数更新、保存/恢复、播放和部署。算法/任务对拍分别读取 `GO2_RL_GYM_ROOT`（混合专家编码器）与 `GO2_SUPERMIMIC_REFERENCE`（参考任务语义），均指定上述固定提交仓库；缺少参考源码时该对拍项明确跳过。

训练产物保存配置、seed、源码哈希与归档、JSONL指标、进程退出状态及结果JSON。实际验证与独立审查维护在 [同主题审查记录](review/go2_blind_supermimic.md)。运行检查只能证明对应环节可用；完整训练、跟踪效果和真机部署需要单独验收。

## 配置版本

当前只支持 `moe_history`（检查点与导出格式 `supermimic_moe_v1` / `supermimic_moe_policy_v1`）。旧的单 MLP 格式、历史 CTS 与蒸馏路径不再支持；变更前的配置和检查点需要重新训练，不做静默转换。论文、来源记录和历史实验产物保留用于追溯。

外部策略驱动通过任务配置 `runner.driver` 提供：`policy` 指向 go2_rl_gym 的 MoE-CTS 导出文件，`expert_policy` 指向本项目历史检查点，`frac`/`expert_frac` 为环境占比，`fade_iterations` 控制线性衰减，`speed_min`/`speed_max` 为追逐命令速度范围。驱动环境位于环境尾部切片，只贡献真实数据，不参与策略更新。
