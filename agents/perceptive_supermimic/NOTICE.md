<!-- Copyright (c) 2026-2027 zh -->

# 同步深度视觉 SuperMimic：来源与适配

本目录实现 `perceptive_supermimic_latent_v1`，由本项目维护，通过任务协议接入环境，不依赖参考仓库的运行环境。它是第一阶段同步单帧视觉算法；Go1/MjLab任务、WMP式降频、相机延迟、RSSM和视觉记忆尚未接入，因此没有创建 `memory.py` 占位文件。

## 参考依据

- [SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model](https://arxiv.org/abs/2610.28480v1)，§3.1–3.3、式(1)–(3)：真实采集、多步模型拟合、冻结模型参数的短视界反传、软存活与末端价值。论文实验使用人形参考动作任务，不能当成本目录的四足视觉效果。
- [InstinctMJ@03563ee02665f327425106372264070315673748](https://github.com/project-instinct/InstinctMJ/tree/03563ee02665f327425106372264070315673748/src/instinct_mj/rl/supermimic)，实际核查 `runner.py:239–249,363–395,537–587,707–767`、`networks.py`、`buffer.py`、`config.py`：Latent方法预测物理状态与CNN潜变量；actor视觉输入截断梯度，CNN通过干净深度重建训练。该仓库顶层许可为 CC BY-NC 4.0，README另注明其使用的 `supermimic_vibed` 部分来源为MIT。本目录按本项目协议组织实现，没有直接迁入参考源码文件。
- [World Model-based Perception for Visual Legged Locomotion](https://arxiv.org/html/2409.16784v1#S4)，IV-A–IV-C、式(4)–(7)，及 [WMP@c232c115ada4517453ebded5019078ba055456de](https://github.com/bytedance/WMP/tree/c232c115ada4517453ebded5019078ba055456de) 的世界模型与训练时序：作为后续视觉记忆和动作历史的依据。WMP策略使用真实仿真PPO与AMP；本目录没有实现或宣称复现该配方。

## 本项目适配

- 默认深度为1×64×64，CNN输出32维；本体历史维度由任务声明，历史编码为普通MLP。模型默认三层1024、LayerNorm和SiLU；参考InstinctMJ默认为三层2048、18×32图像及G1任务。本目录不是指定项目的完整参数复刻。
- 重用本项目 `agents.base.modules.mlp.MLP`、`agents.supermimic.world_model.RunningNorm` 和 `agents.supermimic.learner.frozen`，保持既有网络、归一化及参数冻结工具的维护位置。没有改动盲策略实现。
- 图像已经由任务处理为float32的[0,1]深度；处理含义写入 `depth_semantic_id`。原生米制值、无命中处理、相机外参和具体动作映射均由任务负责，导出不会自行猜测。
- 世界模型输入可包含训练专用上下文与几何；actor只接收本体历史和图像。模型学习物理残差、接触logits与视觉残差；任务提供积分、解析奖励及软终止，不包含可学习失败头、失败过采样或G1参考动作库。
- 真实窗口保存reset前目标。终止和超时当步可以监督真实最终转移，随后窗口前缀中断；这与本项目现有盲回放丢弃done当步的历史语义不同。图像以float16独立保存并由版本引用，覆盖后的图像不能继续作为有效标签。
- 物理与潜变量递推保留梯度；监督拟合恢复记录的非物理记忆、命令与上下文。视觉目标由每次模型更新前的同一版CNN重新编码并截断梯度。潜变量误差使用完整批次更新后固定的逐维 `z_std`，区别于参考代码的批次标量方差；这是本项目明确的尺度选择。
- 编码、模型窗口、重建、想象和critic分批均保持完整逻辑批次的目标与优化器步数。模型损失按有效转移加权，策略按起点数量加权，critic使用全部样本而不丢弃除不尽的尾部。
- 三个优化器分别负责模型+CNN+解码器、历史编码+actor、critic。critic的lambda目标从拟合前价值快照计算；更新输入统计后仍使用同一组固定目标。
- 支持带噪动作采集、模型独立预热、2.5控制步的平均采集调度、余量跨调用保留和余弦学习率。没有加入策略回滚、自动重试或旧检查点兼容迁移。
- 检查点包含网络、优化器、归一化、算法计数和Torch随机数状态；不包含仿真或回放。恢复要求任务、深度和算法语义一致，重新采集后继续优化，不能视为逐步精确恢复仿真。微批预算与运行设备可独立选择。
- TorchScript和ONNX只导出CNN、历史编码和actor，元数据声明双输入、本体历史重置、已处理深度及动作限幅；训练用模型、重建解码器和critic不进入部署图。

设计、接口、测试和适用限制见 [工作设计](../../docs/work/design/go1_depth_supermimic.md)。训练验证使用合成动力学适配器，只证明通用算法链路；机器人、地形效果和部署实时性需要后续独立验证。
