<!-- Copyright (c) 2026-2027 zh -->

# Go1 深度 SuperMimic

从项目根目录启动单设备训练，默认4096个环境：

```bash
OMP_NUM_THREADS=1 .venv/bin/python -u scripts/train.py Unitree-Go1-Depth-SuperMimic-Rough \
  --device cuda:0 --num-envs 4096 --experiment-name go1_depth_query
```

平地任务为 `Unitree-Go1-Depth-SuperMimic-Flat`。参数集中在 `src/config/go1_depth/go1_depth_config.py`，命令行覆盖见任务后 `--help`。终端按Go2格式输出英文训练摘要，默认每10轮汇总指标；TensorBoard事件与检查点位于 `logs/supermimic/go1_depth_query/<运行标识>/`。感知损失位于 `perception/`，查询损失为 `perception/read_loss`。远程日志查看遵循工作区TensorBoard指南。

策略输入为五帧47维本体历史和四个实际相机刷新帧；相机64×64、10Hz，控制50Hz。CNN产生四帧共64个16维token，FPM预测四足下一次触地的航向系XY，查询读取视觉后由单路 `323→512→256→128→12` MLP输出动作。没有MoE或GRU。部署网络398020个参数。

物理模型为三层1024宽MLP，八控制步监督；视觉prior八次相机刷新监督。策略保留完整32控制步BPTT、软存活及末端价值。训练默认沿用Go2的Muon＋Adam、MLP BF16、Inductor编译、CUDA Graph、微批与低频指标回读；想象执行批量2048、价值微批16384保持总想象批2048及优化器步数。CNN和查询保持FP32，CUDA推理按256图像固定微批编码，末批补齐后裁掉。像素与标签准备、感知辅助更新位于图外。

回放保存真实转移与终止前的原生final状态，终止当步可监督，之后屏蔽。深度原始新帧以FP16去重存CPU，GPU保存结构化状态、本体帧、触地元数据及图像引用；回放种子由当前CNN重新编码四帧，不保存过期token或整回合前缀。read查询辅助损失按整个想象种子的有效足点数归一化，改变执行微批不会改变该项的权重。CPU像素占用随实际刷新增长，4096环境、1000控制步在正常刷新下约15.7GiB；需要足够主机内存。训练吞吐和峰值显存以实际验收产物为准。

继续训练保留同目录的 `task_config.json`：

```bash
.venv/bin/python -u scripts/train.py Unitree-Go1-Depth-SuperMimic-Rough \
  --device cuda:0 --num-envs 4096 --resume '<运行目录>/model_1000.pt' \
  --continue-run True --max-iterations 20000
```

`max-iterations`表示目标总轮数。默认热启动只加载参数和匹配的优化器，轮数从零开始；`continue-run True`恢复累计计数与课程，重新建立仿真和回放。`--reset-critic True`从恢复轮次起冻结Actor以预热新Critic。Go2或历史RSSM检查点不能直接用于此网络。

每次保存同步生成 `policy.pt`、`policy.onnx` 和 `policy.json`。导出只包含CNN、FPM、查询和Actor；显式输入 `history[B,5,47]`、`depth_frames[B,4,1,64,64]`，输出未缩放动作 `[B,12]`。深度采用光轴z距离，米制裁剪到0～5后除以5；历史按从旧到新排列。重置时重复首个本体帧与深度帧，相机每5控制步追加新帧。导出前向每次重编码输入的四帧，调用端可另行维护新帧编码缓存；导出本身不实现该缓存。动作按配置的关节缩放映射到PD目标。

```bash
.venv/bin/python scripts/play.py Unitree-Go1-Depth-SuperMimic-Rough \
  --checkpoint-file '<运行目录>/model_1000.pt' --num-envs 1
```

训练验收须显式启动，普通单元测试不启动完整训练：

```bash
OMP_NUM_THREADS=1 .venv/bin/python -u tests/test_training/go1_query_training.py \
  --profile full --num-envs 4096 --output logs/go1_query_validation
```

输出目录必须不存在。该命令先完成编译和各图捕获，再进行至少600秒实际训练，最后检查有限更新、保存与部署导出，并独立恢复继续更新所有优化器及重捕获图。`--profile local`缩减为32环境与样本量、保留网络宽度和时域；`--smoke`仅用于短流程排错，不能作为训练验收或越障能力证明。

实现与来源差异见 [算法来源](../agents/depth_supermimic/NOTICE.md)。当前组合需要通过独立地形成功率、速度误差和摔倒率评估控制效果；数值与工程验收不表示视觉越障已经收敛。
