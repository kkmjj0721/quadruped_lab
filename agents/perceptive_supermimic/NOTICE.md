<!-- Copyright (c) 2026-2027 zh -->

# 同步深度视觉 SuperMimic：来源与适配

本目录实现 `perceptive_supermimic_moe_velocity_v2`，由本项目维护，通过任务协议接入环境，不依赖参考仓库的运行环境。它是使用MoE本体编码与显式速度估计的同步单帧视觉算法；Go1/MjLab任务、WMP式降频、相机延迟、RSSM和视觉记忆尚未接入，因此没有创建 `memory.py` 占位文件。

## 参考依据

- [SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model](https://arxiv.org/abs/2610.28480v1)，§3.1–3.3、式(1)–(3)：真实采集、多步模型拟合、冻结模型参数的短视界反传、软存活与末端价值。论文实验使用人形参考动作任务，不能当成本目录的四足视觉效果。
- [InstinctMJ@03563ee02665f327425106372264070315673748](https://github.com/project-instinct/InstinctMJ/tree/03563ee02665f327425106372264070315673748/src/instinct_mj/rl/supermimic)，实际核查 `runner.py:239–249,363–395,537–587,707–767`、`networks.py`、`buffer.py`、`config.py`：Latent方法预测物理状态与CNN潜变量；actor视觉输入截断梯度，CNN通过干净深度重建训练。该仓库顶层许可为 CC BY-NC 4.0，README另注明其使用的 `supermimic_vibed` 部分来源为MIT。本目录按本项目协议组织实现，没有直接迁入参考源码文件。
- 本体编码复用本项目盲版的 `StudentMoEEncoder`，其MoE拓扑来源为 [go2_rl_gym@30e74dc](https://github.com/wty-yy/go2_rl_gym/blob/30e74dc507bec7a642a8c98be26081f2c6f0822d/rsl_rl/rsl_rl/modules/utils.py#L96)，本次重新实读该版本。真实机身速度MSE参考 [DreamWaQ v2](https://arxiv.org/html/2301.10602v2#S2.SS3) §II-C式(6)，潜特征与速度的分离归一化交叉核对 [HIMLoco@ef289ac](https://github.com/InternRobotics/HIMLoco/blob/ef289acaa62795009363b7b819c9186690630441/rsl_rl/rsl_rl/modules/him_estimator.py#L64)。本项目保留本体分支的策略梯度，不迁入这些项目的PPO、教师、VAE或对比损失；来源和本项目适配详见 [盲版说明](../supermimic/NOTICE.md)。
- [World Model-based Perception for Visual Legged Locomotion](https://arxiv.org/html/2409.16784v1#S4)，IV-A–IV-C、式(4)–(7)，及 [WMP@c232c115ada4517453ebded5019078ba055456de](https://github.com/bytedance/WMP/tree/c232c115ada4517453ebded5019078ba055456de) 的世界模型与训练时序：作为后续视觉记忆和动作历史的依据。WMP策略使用真实仿真PPO与AMP；本目录没有实现或宣称复现该配方。

## 本项目适配

- 默认深度为1×64×64，CNN输出32维；本体历史维度由任务声明。历史编码使用8专家，隐藏宽度为(512,256,128)，融合后输出32维L2潜特征与3维未归一化机身速度。actor按本体潜特征、估计速度、独立归一化当前帧、截断梯度视觉特征拼接输入。模型仍为三层1024、LayerNorm和SiLU；参考InstinctMJ默认为三层2048、18×32图像及G1任务。本目录不是指定项目的完整参数复刻。
- 重用本项目 `agents.base.modules.mlp.MLP`、`agents.supermimic.policy.StudentMoEEncoder`、`agents.supermimic.world_model.RunningNorm` 和 `agents.supermimic.learner.frozen`，保持既有网络、归一化及参数冻结工具的维护位置。没有改动盲策略实现。
- 图像已经由任务处理为float32的[0,1]深度；处理含义写入 `depth_semantic_id`。原生米制值、无命中处理、相机外参和具体动作映射均由任务负责，导出不会自行猜测。
- 世界模型输入可包含训练专用上下文与几何；actor只接收本体历史和图像。模型学习物理残差、接触logits与视觉残差；任务提供积分、解析奖励及软终止，不包含可学习失败头、失败过采样或G1参考动作库。
- 真实窗口保存reset前目标。终止和超时当步可以监督真实最终转移，随后窗口前缀中断；这与本项目现有盲回放丢弃done当步的历史语义不同。图像以float16独立保存并由版本引用，覆盖后的图像不能继续作为有效标签。
- 物理与潜变量递推保留梯度；监督拟合恢复记录的非物理记忆、命令与上下文。视觉目标由每次模型更新前的同一版CNN重新编码并截断梯度。潜变量误差使用完整批次更新后固定的逐维 `z_std`，区别于参考代码的批次标量方差；这是本项目明确的尺度选择。
- 编码、模型窗口、重建、想象和critic分批均保持完整逻辑批次的目标与优化器步数。模型损失按有效转移加权，策略按起点数量加权，critic使用全部样本而不丢弃除不尽的尾部。
- 三个优化器分别负责模型+CNN+解码器、历史编码+actor、critic。critic的lambda目标从拟合前价值快照计算；更新输入统计后仍使用同一组固定目标。
- 策略在想象负回报上增加真实起点速度MSE，默认`velocity_loss_coef=1.0`；两项按起点数加权后共用一次actor更新。任务新增`body_linear_velocity(state)`，返回与本体历史末帧同一时刻的机身系xyz速度（float32、m/s），标签截断梯度且只用于监督；部署不输入真实速度。日志包含`actor_loss`、`control_loss`、`velocity_loss`及整体和各轴RMSE。
- 支持带噪动作采集、模型独立预热、2.5控制步的平均采集调度、余量跨调用保留和余弦学习率。没有加入策略回滚、自动重试或旧检查点兼容迁移。
- 检查点包含网络、优化器、归一化、算法计数和Torch随机数状态；不包含仿真或回放。恢复要求任务、深度和算法语义一致，重新采集后继续优化，不能视为逐步精确恢复仿真。微批预算与运行设备可独立选择。旧`perceptive_supermimic_latent_v1`检查点明确拒绝；本体编码与动作输入已经改变，不进行隐式权重迁移。
- TorchScript和ONNX只导出CNN、历史编码和actor，格式为`perceptive_supermimic_moe_velocity_policy_v2`，元数据声明双输入、本体历史重置、已处理深度、速度估计坐标与单位及动作限幅；训练用模型、重建解码器和critic不进入部署图。

设计、接口、测试和适用限制见 [工作设计](../../docs/work/design/go1_depth_supermimic.md)。训练验证使用合成动力学适配器，只证明通用算法链路；机器人、地形效果和部署实时性需要后续独立验证。
