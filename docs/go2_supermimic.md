<!-- Copyright (c) 2026-2027 zh -->

# Go2 SuperMimic 盲走

默认新训练使用**单一盲 MoE actor**：本体历史经过八专家编码器，32维latent与当前观测拼接，经MLP产生12维关节位置动作。MoE与actor直接接收SuperMimic想象回报梯度，没有教师、蒸馏或PPO更新。任务为速度命令跟踪，支持原有平地和粗糙地形。

配置位于 [Go2SingleActorCfg](../src/config/go2/go2_single_supermimic_config.py)，算法预算位于 [SingleConfig](../agents/supermimic/single/config.py)。网络来源是 `go2_rl_gym` 的部署侧MoE结构，训练循环参考 `Zhengtianjin/go2-supermimic`；固定版本、采用内容及适配差异见 [来源说明](../agents/supermimic/single/NOTICE.md) 和 [设计](design/go2_blind_supermimic.md#当前实施参考训练循环与单一盲-moe-actor2026-10-27)。这不是论文G1动作跟踪实验的原样复现，也不继承参考仓库的演示成绩。

## 启动训练

```bash
.venv/bin/python -u scripts/train.py Unitree-Go2-SuperMimic-Flat \
  --device cuda:0 --max-iterations 10000 --save-interval 500
```

粗糙地形任务为 `Unitree-Go2-SuperMimic-Rough`。建议先验收平地四组命令跟踪，再开展粗糙地形训练。使用 `--help` 查看完整参数；原有 `--gpu-ids 0`、`--env.scene.num-envs`、`--agent.seed`、`--agent.max-iterations` 等别名继续可用。

默认固定使用参考预算：4096真实环境，每轮采1步；世界模型1024×3，每轮2批2048个8步窗口；8192条32步想象；每环境1000步回放。网络结构使用用户指定MoE。训练不探测显存来缩减预算，也无需选择档位；实际参数在启动时打印并保存到快照。

单MoE默认使用`--imag-microbatch 256 --critic-microbatch 4096`限制同时反传的样本数。8192条想象分32段，各段仍完整反传32步，按样本数累积后统一裁剪、更新actor一次；critic保持原有5轮×4小批、20次更新。这两个执行参数不会改变总训练批量或检查点任务指纹，续训时可调整。分批可能改变随机观测噪声的抽样顺序，不保证与整批训练逐位相同。

8 GB显卡从1024环境开始验证的命令如下；短跑通过不代表长期训练稳定或步态收敛：

```bash
.venv/bin/python -u scripts/train.py Unitree-Go2-SuperMimic-Flat \
  --device cuda:0 --num-envs 1024 --imag-microbatch 256 --critic-microbatch 4096 \
  --max-iterations 10000 --save-interval 500
```

`Not enough SMs to use max_autotune_gemm mode`表示Inductor不对该GPU使用某类矩阵乘调优模板，本身不会中断训练；应区别于后续`torch.OutOfMemoryError`。仅减环境数不会同步减小8192条想象图；优先通过分批限制其峰值。回放在创建时分配，归档在首次收录时分配完整配置容量。若这些存储或其他进程使设备仍接近容量，需通过JSON显式减少`single.replay_capacity`或`single.archive_size`，并以新配置启动实验，不能把不同任务指纹强行续接。

如需自行减少采集环境数，在命令后添加 `--num-envs 1024`；此参数仅改变环境数量，模型批量、想象批量和回放长度保持默认。其他参数可通过 `--task-config` JSON 显式调整。覆盖优先级为配置默认值、JSON、显式命令行环境数和种子。下面JSON只调整环境数：

```json
{
  "env": {"num_envs": 1024}
}
```

通过 `--task-config <路径>` 使用。新配置字段在 `single` 下；带旧 `policy/model/learner` 字段的完整历史CTS配置恢复旧算法，不能用其覆盖新算法。新旧字段混用会报错。

新训练先收集初始窗口，并按配置执行模型预热；随后每轮采集、拟合模型、想象优化策略与critic。支持OU相关噪声探索、接触变化／飞行／跌倒优先窗口和稀有事件归档。预热、actor学习率渐增、余弦调度以及有保存点后的跌倒回滚均有独立计数与日志。回滚恢复actor/critic参数并重建二者优化器，保留持续拟合的世界模型；不代表完整仿真回滚。命令课程按真实控制步推进，其日志中的课程轮次不等于训练轮次；实际命令范围以课程状态为准。

## 观测、动作与模型

actor单帧45维，按以下顺序排列：机体系角速度3、重力投影3、速度命令3、关节角相对默认姿态12、关节速度12、上一执行动作12。历史为5帧225维，按旧到新逐帧排列；reset重复初始帧。没有相位、真实线速度、地形扫描和接触真值输入。

模型与critic可使用11×7的地形扫描；世界模型只学习物理残差，不学习奖励辅助头或失败头。奖励由速度误差、姿态、动作变化、关节限位及解析物理代理计算，真实与想象共用公式，每步仅乘一次0.02秒控制周期。PD力矩和控制步关节加速度是奖励代理，不是物理子步传感器真值。真实失败使用高度／倾角硬阈值，想象使用对应软概率；地形支持域外存活权重归零。

与参考仓库不同，新默认保留已有 `action_limit=1` 保护，输出为 `L*tanh(raw/L)`，真实探索加噪后同样限制到动作域；可以在新配置中显式设置 `single.action_limit=null` 使用无界输出。保存和导出记录实际动作语义，不能把无界和有界检查点互换解释。

关节目标为 `default_joint_positions + actions * action_scales`。关节顺序、默认姿态、各关节缩放及归一化统计随导出元数据保存；不直接兼容参考仓库权重或其追逐任务观测。

## 保存、续训与播放

产物位于 `logs/supermimic/go2_supermimic/<时间戳>[_run_name]/`。每次保存包含 `model_N.pt`、`policy_N.pt`、`policy_N.onnx`、`policy_N.json`，部署文件同时更新 `policy.pt/onnx/json`。PT为TorchScript，ONNX为动态batch，输入 `obs_history` 为 `[B,225]`，输出 `actions` 为 `[B,12]`。部署图包含MoE、actor及本体归一化，无世界模型、critic或教师。各文件原子替换，不承诺整组文件事务。

```bash
.venv/bin/python -u scripts/train.py Unitree-Go2-SuperMimic-Flat \
  --device cuda:0 --resume '<实验目录>/model_1000.pt' --max-iterations 500

.venv/bin/python scripts/play.py Unitree-Go2-SuperMimic-Flat \
  --checkpoint-file '<实验目录>/model_1000.pt' --num-envs 1 --viewer viser --device cpu
```

`max_iterations`表示本次追加轮数；续训创建新目录，恢复参数、优化器、归一化、课程与累计计数，并重新采集回放和预热，不恢复仿真状态。必须使用匹配的任务ID、配置和环境数量。播放自动读取同目录 `task_config.json`，允许环境数、设备与播放噪声覆盖；新旧检查点严格区分，不能将旧CTS权重载入单actor。

播放支持 `--viewer native`、`--video True --video-length 200`、`--no-terminations True`。`--agent zero/random`可检查新任务而无需检查点。评估接口为 `rl.evaluation.evaluate_policy(..., branch="actor")`，按前进、后退、侧移、转向分别报告误差和存活；单actor检查点不提供teacher分支。旧student调用名仅作为盲策略兼容别名。

## 验证与结论边界

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -B -m unittest \
  tests.test_agents.test_single_supermimic tests.test_tasks.test_single_supermimic_task \
  tests.test_scripts.test_single_supermimic_config tests.test_scripts.test_single_supermimic_checkpoint -v

RUN_SINGLE_SUPERMIMIC_CPU=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m unittest \
  tests.test_training.test_single_supermimic_training -v

RUN_SINGLE_SUPERMIMIC_GPU=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m unittest \
  tests.test_training.test_single_supermimic_training -v

RUN_SINGLE_SUPERMIMIC_GPU_8GB=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m unittest \
  tests.test_training.test_single_supermimic_training -v
```

训练测试默认跳过，只有显式环境变量才启动。CPU使用独立测试JSON中的4环境、小网络与少量轮数验证flat/rough、更新、保存续训和播放导出；GPU使用完整参考网络／环境数／回放／批量与rough地形，仅缩短预热和迭代，检查完整图Inductor编译及物理容量。完整参考预算GPU测试尚未执行；历史缩减预算的短跑证据保留在 [审查记录](review/go2_blind_supermimic.md)，不能替代当前完整预算验收。

`RUN_SINGLE_SUPERMIMIC_GPU_8GB=1`单独启用1024环境的OOM回归，保留完整模型、MoE、2048×8模型批量、8192×32想象和300轮预热，使用上述默认执行分批；共302轮，验证原来失败的第301、302轮，记录参数更新和PyTorch峰值显存。

`metrics.jsonl`逐轮保存真实跟踪误差、奖励项、完成回合、模型／策略／critic损失及阶段耗时；不显示不存在的教师或学生蒸馏指标。首次编译会计入首次控制更新时间，应与后续稳态吞吐分开。`run_config.json`、`task_config.json`、`config.json`、`execution.json`、`source_manifest.json` 和 `sources.zip`记录实际运行条件。短跑尚不能证明长期稳定、相对PPO加速或实机可部署。

## 历史CTS兼容

[Go2SuperMimicCfg](../src/config/go2/go2_supermimic_config.py)保留原47维／235维盲输入、教师、MoE学生及22项HIM奖励语义。显式传入该旧类，或加载完整旧任务快照，继续走历史训练／播放路径；不会为旧检查点添加新字段或修改原spec。旧架构和两次千轮失败的证据保留在设计与审查记录，不作为当前单actor效果。
