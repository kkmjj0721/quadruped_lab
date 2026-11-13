<!-- Copyright (c) 2026-2027 zh -->

# 同步深度视觉 SuperMimic：来源与适配

本目录实现 `perceptive_supermimic_visual_clock_v4`，由本项目维护，通过任务协议接入环境。actor采用WMP式普通MLP本体历史编码、显式速度估计、当前指令与可训练视觉投影；感知可选MGDP式高度重建与跨模态对比监督。Go1/MjLab环境由任务目录负责；视觉输入由任务声明，可以是同步单帧或带采集时钟的双帧CNN特征，世界模型采用SuperMimic可微物理／潜变量递推，不称WMP的RSSM或MGDP视觉GRU复现。

## Go1 原生 MGDP 配方

Go1 默认入口改为两阶段：`--stage 1` 在 mix 上联合训练，`--stage 2 --stage1-run <目录>` 转 parkour 后继续更新感知、动力学与策略。普通 `--resume` 仍只接受同阶段语义。旧本地单帧配方由 `--legacy` 显式选择，下文旧网络默认值仅描述该通用默认配方；旧 v3 权重不能隐式恢复到 v4。

- `perception_model="mgdp"` 使用官方两帧16×16 U-Net：深度32/64/96通道、pool4、64维潜变量；高度16/32/64、pool2、32维潜变量，三层显式skip。InfoNCE只对齐视觉前32维；图像/高度目标保留各自采集时刻，5步保持期间不重算标签。同一图像版本在对比批次中去重。
- 感知 Adam 独立采用1e-4学习率、仅解码器1e-5衰减、梯度范数上限1。其余动力学、actor和critic优化器保留SuperMimic职责。没有迁入MGDP原版GRU或PPO；现有普通MLP历史网络、视觉投影和32步想象保留。
- 相机/模型分别以10Hz/50Hz推进。世界模型在未采图步严格保持潜变量，在采图步预测残差；图像环同时保存两帧输入、干净重建目标和采集时的高度标签。reset前终态与reset后复制首帧明确隔离。
- 阶段间按用户选择：最新策略/critic及优化器，与最佳感知及同轮动力学/统计/优化器组合。最佳指标为保存轮的 `0.6*depth_mse+0.4*height_mse`；按官方阶段零基轮号0/1000/2000…比较，结束不额外选best，因此并非全部轮次的全局最优。来源轮次记入 `stage_transfer.json`；`configuration.stage_iteration_offset`保存阶段时钟，普通resume缺少该字段或best来自未来轮次时明确拒绝。
- Go1任务的真实奖励保持官方硬公式；想象以硬前向、可微代理后向处理接触、绊脚、腾空、边缘及终止。MuJoCo碰撞几何按用户批准采用整数高度场与独立实体，不能视为PhysX接触动力学等价复现。用户明确保留本地标称质量、质心和惯量，仅对齐随机化；本地合并calf/foot刚体接受相同范围的质量比例随机化，不声称与官方独立足刚体的随机参数逐项等价。
- 仿真与PD保持5ms；MuJoCo使用完整implicit。关节速度上限30.1rad/s通过用户批准的迁移实现：先计算原版PD扭矩，再限制积分初速，根速度不直接修改，步末速度可超出。此机制已对照CPU Isaac Gym Preview4单关节行为，不代表多关节接触或GPU求解逐项等价；任务语义记录该行为，不能混载此前未限速的检查点。
- 当前锁定MJWarp3.11.0需在创建环境前启用`src/sim/mjwarp_implicit.py`回补上游PR1663的RNE导数符号修复，原生入口已接入；升级依赖须复核后移除回补。想象的总奖励截零也使用硬前向、identity STE后向，以保留负奖励区的改善方向；真实公式不变。日志单列动作头梯度和有效想象存活步数，不能仅凭总actor梯度判断运控目标有效。

```bash
.venv/bin/python scripts/train_go1_depth.py --stage 1 --num-envs 20 --output logs/supermimic/go1_depth/stage1
.venv/bin/python scripts/train_go1_depth.py --stage 2 --stage1-run logs/supermimic/go1_depth/stage1 --output logs/supermimic/go1_depth/stage2
.venv/bin/python scripts/train_go1_depth.py --resume logs/supermimic/go1_depth/stage2/model_final.pt --output logs/supermimic/go1_depth/stage2_resume
```

这是一套基于MGDP任务与感知配置的SuperMimic适配。通过源码对拍或短训练不代表已复现论文的运动能力；效果需分别检查真实闭环跟踪、摔倒率和各类障碍通过情况。

## 参考依据

- [SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model](https://arxiv.org/abs/2610.28480v1)，§3.1–3.3、式(1)–(3)：真实采集、多步模型拟合、冻结模型参数的短视界反传、软存活与末端价值。论文实验使用人形参考动作任务，不能当成本目录的四足视觉效果。
- [InstinctMJ@03563ee02665f327425106372264070315673748](https://github.com/project-instinct/InstinctMJ/tree/03563ee02665f327425106372264070315673748/src/instinct_mj/rl/supermimic)，实际核查 `runner.py:239–249,363–395,537–587,707–767`、`networks.py`、`buffer.py`、`config.py`：Latent方法预测物理状态与CNN潜变量；actor视觉输入截断梯度，CNN通过干净深度重建训练。该仓库顶层许可为 CC BY-NC 4.0，README另注明其使用的 `supermimic_vibed` 部分来源为MIT。本目录按本项目协议组织实现，没有直接迁入参考源码文件。
- [WMP@c232c115ada4517453ebded5019078ba055456de](https://github.com/bytedance/WMP/blob/c232c115ada4517453ebded5019078ba055456de/rsl_rl/modules/actor_critic_wmp.py#L73) 的历史编码、当前指令与视觉投影融合，以及[实际A1配置](https://github.com/bytedance/WMP/blob/c232c115ada4517453ebded5019078ba055456de/legged_gym/envs/a1/a1_amp_config.py#L295)：210→256→128→35（最后3维速度），视觉h512→64→64→32，融合70→256→128→64→12。代码按本项目现有MLP组件独立组织，未复制参考源码。真实机身速度MSE及先截断输入再训练视觉投影的依据是`rsl_rl/algorithms/amp_ppo.py:124,239`；actor本身不含循环状态。
- [World Model-based Perception for Visual Legged Locomotion](https://arxiv.org/html/2409.16784v1#S4)，IV-A–IV-C、式(4)–(7)：世界模型记忆每5个控制步更新，policy读取stop-gradient的确定性状态。WMP采用真实仿真PPO与AMP；本项目把该actor融合拓扑适配到同步z32和SuperMimic的32步可微想象，保留本地历史RunningNorm、任务单位指令及reset时复制首帧，不宣称等价复现其RSSM、100ms延迟或训练配方。
- [MGDP: Mastering a Generalized Depth Perception Model for Quadruped Locomotion](https://pmc.ncbi.nlm.nih.gov/articles/PMC13285150/)，§4.3式(11)–(14)，以及实际核查的[官方代码@f43139d，random_dog.py:566–608](https://github.com/arclab-hku/MGDP/blob/f43139d1d97849819a0dce609dc0b59faf577b3f/legged_gym/legged_gym/envs/random_dog/random_dog.py#L566)：带噪深度重建干净图、真实高度自重建、同场景高度到深度特征InfoNCE。采用同索引正对、整批其他样本为负对、L2特征归一化和默认0.1温度；代码按本项目网络和任务协议独立编写。原实现使用两帧16×16、64维视觉特征中32维对齐、U-Net跳连、PPO与阶段间迁移；原生Go1配方现已采用其两帧U-Net与32/64维对齐；通用默认配方仍保留单帧32维CNN。两者都使用SuperMimic物理／视觉预测，不称完整MGDP复现。

## 本项目适配

- 默认深度为1×64×64，Go1任务处理为32×32，CNN输出32维。`command_indices`显式声明指令在每帧观测的位置；历史MLP剔除各帧指令后归一化，默认隐藏宽度(256,128)，输出32维潜特征与3维机身速度，不做L2归一化。当前指令单独按任务单位输入。视觉CNN输出先detach，再经可训练(64,64)投影为32维；actor按本体潜特征、速度、当前指令、视觉投影拼接，Go1为70维，隐藏宽度(256,128,64)。世界模型仍为三层1024、LayerNorm和SiLU。
- 重用本项目 `agents.base.modules.mlp.MLP`、`agents.supermimic.world_model.RunningNorm` 和 `agents.supermimic.learner.frozen`，保持既有归一化及参数冻结工具的维护位置。视觉算法不再依赖StudentMoEEncoder，盲版MoE不受影响。
- 图像已经由任务处理为float32的[0,1]深度；处理含义写入 `depth_semantic_id`。原生米制值、无命中处理、相机外参和具体动作映射均由任务负责，导出不会自行猜测。
- 世界模型输入可包含训练专用上下文与几何；actor只接收本体历史和图像。模型学习物理残差、接触logits与视觉残差；任务提供积分、解析奖励及软终止，不包含可学习失败头、失败过采样或G1参考动作库。
- 真实窗口保存reset前目标。终止和超时当步可以监督真实最终转移，随后窗口前缀中断；这与本项目现有盲回放丢弃done当步的历史语义不同。图像以float16独立保存并由版本引用，覆盖后的图像不能继续作为有效标签。
- 物理与潜变量递推保留梯度；监督拟合恢复记录的非物理记忆、命令与上下文。视觉目标由每次模型更新前的同一版CNN重新编码并截断梯度。潜变量误差使用完整批次更新后固定的逐维 `z_std`，区别于参考代码的批次标量方差；这是本项目明确的尺度选择。
- 编码、模型窗口、重建、想象和critic分批均保持完整逻辑批次的目标与优化器步数。模型损失按有效转移加权，策略按起点数量加权，critic使用全部样本而不丢弃除不尽的尾部。
- 可选高度监督由`height_reconstruction_weight`和`contrastive_weight`启用，两者默认0以保留原任务配方；默认深度重建仍为L1，可显式选MSE。任务仅在启用时实现`terrain_height(state)`，返回同图像终态、同设备float32的`[B,1,H,W]`，归一化至[-1,1]；默认H×W为17×11。静态场景按回放终态重算标签，不用当前仿真位置替代，也不向actor泄露高度真值。
- 高度编码器为16／32／64通道步幅卷积、2×2自适应池化和潜变量投影；高度解码从潜变量上采样并以Tanh限制输出。没有BatchNorm、Dropout或U-Net旁路。InfoNCE先计算完整逻辑批次的特征梯度，再在`vision_batch`内确定性重算特征回传，保留全批负样本及梯度、限制中间激活；单样本的交叉熵为零，不伪造负对。
- 三个优化器分别负责模型+CNN+解码器+可选高度网络、历史编码+视觉投影+actor、critic。感知的深度／高度重建和对比损失共用一次模型优化器更新；actor更新不训练CNN或读取高度标签，但训练视觉投影。critic的lambda目标从拟合前价值快照计算；更新输入统计后仍使用同一组固定目标。
- 策略在想象负回报上增加真实起点速度MSE，默认`velocity_loss_coef=1.0`；两项按起点数加权后共用一次actor更新。任务新增`body_linear_velocity(state)`，返回与本体历史末帧同一时刻的机身系xyz速度（float32、m/s），标签截断梯度且只用于监督；部署不输入真实速度。日志包含`actor_loss`、`control_loss`、`velocity_loss`及整体和各轴RMSE。
- 支持带噪动作采集、模型独立预热、2.5控制步的平均采集调度、余量跨调用保留和余弦学习率。没有加入策略回滚、自动重试或旧检查点兼容迁移。
- 检查点包含网络、优化器、归一化、算法计数和Torch随机数状态；不包含仿真或回放。恢复要求任务、深度和算法语义一致，重新采集后继续优化，不能视为逐步精确恢复仿真。微批预算与运行设备可独立选择。旧v1、MoE v2及MLP v3检查点均明确拒绝；历史实验和源码基线保留，不做隐式权重迁移。
- 启用高度监督的检查点必须包含`terrain_perception`完整权重与对应模型优化器参数；缺失或多余模块明确拒绝，不回退到随机初始化。新增配置字段与command索引共同参与严格语义比较。
- TorchScript和ONNX导出CNN、历史编码、视觉投影和actor，格式为`perceptive_supermimic_visual_clock_policy_v4`，元数据声明双输入、指令索引、去指令历史归一化、无L2、速度单位及动作限幅。训练用模型、解码器、高度分支和critic不进入部署图。

设计、接口、测试和适用限制见 [工作设计](../../docs/work/design/go1_depth_supermimic.md)。通用算法单测使用合成动力学适配器，只证明感知梯度、32步想象和持久化链路；真实Go1实验结果由该设计记录，不能由单测推断机器人、地形泛化或部署实时性。
