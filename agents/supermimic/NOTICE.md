<!-- Copyright (c) 2026-2027 zh -->

# SuperMimic 来源与适配

Go2结构化动力学的数学实现以 [BlinkMimic@d264658 `networks.py:113-209`](https://github.com/fuyichen0725/blinkmimic/blob/d264658d6c9ce61fb52f9f0f8985a369f3ccc409/src/blinkmimic/networks.py#L113) 的广义速度增量、位置修正、旋转指数映射及物理状态误差为基准，旋转与树状FK迁自同版本 `rotations.py/kinematics.py`。已交叉核对 [InstantTrack@b9512a9](https://github.com/Huy-Quang-Dao/instantTrack/tree/b9512a96d4e82e6ddea9a6de3b03276888bbcb2c) 和 [XMimic@54860e1 `nets.py/kin.py`](https://github.com/nico-bohlinger/xmimic/tree/54860e100251fa80143627565568d5e6b0ff2c2b/xmimic)。四子步漂移权重0.625与来源一致，子步权重公式来自 [SuperMimic@abc124c `representations.py`](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/representations.py)。派生实现和完整MIT许可位于 [Go2任务MDP](../../src/tasks/locomotion/go2_supermimic/mdp/LICENSE_blinkmimic)，由本项目维护，不依赖参考仓库运行。

这次迁移仅覆盖结构化动力学：Go2使用本地MJCF的髋／膝连杆原点和足端site，足速包含根部twist与关节速度；保留已有瞬时加速度、接触、力矩等独立辅助监督，保留现有网络隐藏层与RunningNorm统计机制。残差输出统计与物理误差增量统计分离，核心误差下限取BlinkMimic `train.py` 的 `ERR_FLOOR`，输出层采用其零初始化。四元数作为内部完整姿态存储，任务观测／奖励接口保持。参考为人形动作跟踪，本项目为Go2速度跟踪；结构对拍和运行检查不能证明控制效果改善。

## 视觉感知与落足接口

`visual/` 是本项目在现有算法上的独立实现，采用已选定的“感知与选点监督、动作策略 BPTT”组合。`VisualActor` 使用 MLP 历史估计器和当前真实速度监督，将米制地图及任务锁存目标接入动作 MLP；`VisualLearner` 复用 `Learner.fit_model/improve` 和 critic 更新；`VisualImagination` 沿用现有折扣奖励、尾部价值及 `Imagination.lambda_returns`。

视觉估计器结构参考 [HIMLoco@ef289ac `him_estimator.py:10–67`](https://github.com/InternRobotics/HIMLoco/blob/ef289acaa62795009363b7b819c9186690630441/rsl_rl/rsl_rl/modules/him_estimator.py#L10-L67)：两层隐藏 MLP（128、64，ELU）、线性输出、速度在前、latent 在后且仅后者 L2 归一化。本项目按用户要求将隐藏层改为 256、128、64，并将官方 latent 16 改为 32，五帧 45 维示例为 `225→256→128→64→35`；这些宽度是用户指定值，不是官方默认值。该前向结构使用已有 `Mlp` 独立装配；归一化及统计操作通过 `HistoryActor` 与盲策略共享。论文《Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response》§3.2–3.3 的宽度叙述与代码不同，此处以固定版本代码为结构依据。训练保留本项目当前速度监督和 SuperMimic BPTT，不迁入 HIM 的 target/prototype、SwAV 或 PPO detach；不宣称完整复现 HIM 或已验证训练收益。视觉模型标识为 `supermimic_visual_foothold_v2`，旧视觉 MoE 权重明确拒绝。

想象时预测位姿渲染和视觉输入停梯度的依据为 [InstinctMJ@03563ee `runner.py:render_depth,act_detached,_imagine`](https://github.com/project-instinct/InstinctMJ/blob/03563ee02665f327425106372264070315673748/src/instinct_mj/rl/supermimic/runner.py)。这里只借鉴接口与梯度边界，未移植其渲染代码；真实相机、噪声及独立射线状态由任务实现。原实现使用 G1 跟踪，本项目改为四足米制地图和锁存目标，不能据外部实现推断 Go1 训练效果。

感知与选点的固定版本和方法差异分别见 [perception/NOTICE.md](../perception/NOTICE.md) 与 [foothold/NOTICE.md](../foothold/NOTICE.md)。算法持有显式视觉种子、缓存 token 和 GRU 状态；感知更新后重放同回合完整前缀，部署副本剔除 GT 分支。候选监督的几何掩码只用于标签，任务奖励和接触状态机仍归 `src/tasks/`。

本接口通过独立 `VisualLearner` 和部署组合使用，未接入原 Go2 runner 的采集、归档或 CUDA Graph。视觉学习器默认五路网络全部采用局部 CUDA BF16，包括 MGDP 卷积／MLP／GRU、训练重建分支和选点注意力矩阵乘法；网络对外输出及记忆、几何、概率归一化、损失归约、物理状态和优化器状态保留 FP32。局部 autocast／输出转 FP32 的边界沿用下述 BlinkMimic 依据；扩展到视觉网络属于本项目工程适配，不代表上游已验证此组合。通用执行配置可显式关闭五路开关，部署副本固定 FP32。算法数值、梯度、有限步更新与持久化检查只证明相应算法接口行为，不代表实际渲染、机器人训练或稀疏地形能力。

## 8MoE主线与SuperMimic机制

可选网络BF16精度采用 [BlinkMimic@d264658 `run_mlp`](https://github.com/fuyichen0725/blinkmimic/blob/d264658d6c9ce61fb52f9f0f8985a369f3ccc409/src/blinkmimic/networks.py#L64) 的局部autocast与输出转FP32边界，并参考 [XMimic@54860e1 `nets.py`](https://github.com/nico-bohlinger/xmimic/blob/54860e100251fa80143627565568d5e6b0ff2c2b/xmimic/nets.py#L43)、`fast_nets.py` 与 `tests/test_fast_equiv.py` 区分矩阵精度、状态精度及梯度差异。三个网络开关独立，Go2主线新训练默认开启BF16；归一化、状态积分、奖励、返回递推、MoE分组卷积与融合均沿用FP32。XMimic使用JAX及G1动作跟踪，本项目为PyTorch与Go2速度跟踪；不移植其跨时间步梯度合并、自定义VJP或LayerNorm，外部加速结果不代表本项目性能。

训练使用 `moe_history`、默认8个专家，检查点格式为 `supermimic_moe_velocity_v1`。本目录作为本项目源码维护，不依赖参考项目的Python运行环境。

各机器人在`src/config/<机器人>/*supermimic_config.py`的`algo`与`runner`段声明实际训练配方；本目录只维护共享字段定义、通用默认、校验与算法实现。CLI显式覆盖已加载配置，新增运行预算不进入任务语义指纹。

CUDA更新执行方式维护在`cuda_graph.py`，由本项目已有独立性能实验迁入。工程依据实读 [BlinkMimic@d264658](https://github.com/fuyichen0725/blinkmimic/blob/d264658d6c9ce61fb52f9f0f8985a369f3ccc409/src/blinkmimic/train.py#L403) 的 `Graphed`、设备学习率及可捕获Adam，以及 [InstantTrack@b9512a9](https://github.com/Huy-Quang-Dao/instantTrack/blob/b9512a96d4e82e6ddea9a6de3b03276888bbcb2c/src/instanttrack/train.py#L455) 的预热与独立捕获流。保留本项目图外采样、归一化校验、全部优化器的非有限梯度跳步、速度监督和护栏，图执行适配本身不迁入参考项目的G1任务、EMA或训练配方，后续Muon接入见下文。默认CUDA开启、CPU使用原执行方式；可用`--cuda-graph False`关闭，配置和限制见[Go2任务说明](../../docs/go2_supermimic.md)。

完整八步监督、critic损失与lambda-return编译由本项目`combined_cuda_graph`消融迁入主线，统一边界位于`compilation.py`，沿用原算法公式。依据[PyTorch2.14 compile API](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html)固定整图与输入形状；Inductor保留半精度舍入和原生随机数消费。捕获时暂停自动循环GC的范围参考[Triton@1735801的cuda_graph_without_gc](https://github.com/triton-lang/triton/blob/173580104f6b1c182472fe0ad3e11f639fbaa177/python/triton/testing.py#L78)，finally恢复原状态，不禁用引用计数析构、不重试或吞异常。远程消融耗时是本项目观测，不能当作论文结论或迁入版本的新运行验证。

[XMimic@8f7693f](https://github.com/nico-bohlinger/xmimic/blob/8f7693f96ca02c49f11dcc98cd14cf03f45df2fa/xmimic/algorithm.py#L465) 使用JAX的`jit/lax.scan`整体迭代编译，提供减少Python调度的工程参照，不作为本项目PyTorch更新图的直接实现。InstantTrack的异步世界模型／critic更新会改变读取网络的时序，本次不采用；参考项目的速度和效果不代表本项目已经复现。

`supermimic_moe_velocity_v1` 按用户确认的 Go2 盲（本体感知）速度跟踪路线实现：算法沿用 SuperMimic，actor 采用 `go2_rl_gym` 的混合专家网络，其余训练机制对齐指定参考实现。

- 训练机制参考 [Zhengtianjin/go2-supermimic](https://github.com/Zhengtianjin/go2-supermimic/tree/d9eee9b3f248e5300e636c433255db2c1c3e086a)，已实读固定提交的 `supermimic/{train,world_model,state,buffer,imagine,driver}.py`。本目录按职责平铺对应模块：`world_model.py`（`RunningNorm`／`WorldModel`）、`policy.py`（混合专家 actor 与 critic）、`buffer.py`、`imagine.py`、`learner.py`、`runner.py`；驱动器装配留在任务侧 `src/tasks/locomotion/go2_supermimic/rl/driver.py`。
- actor 结构参考 [wty-yy/go2_rl_gym](https://github.com/wty-yy/go2_rl_gym/tree/30e74dc507bec7a642a8c98be26081f2c6f0822d)，已实读 `rsl_rl/rsl_rl/modules/{utils,actor_critic_moe_cts}.py`、`algorithms/ac_moe_cts.py`、`legged_gym/utils/exporter.py`。
- `MoeActor` 在参考MoE拓扑上增加显式速度估计：完整五帧47维历史经ELU骨干(512,256)得到8×128，分组1×1卷积每专家输出32+3=35维；Softmax融合后仅对前32维潜特征做L2归一化，后3维估计当前机身系xyz线速度（m/s）。按潜特征、速度、独立归一化当前帧拼接为82维，经ELU MLP(512,256,128)输出12维未缩放动作。
- 历史与当前帧使用两套独立 `RunningNorm`，门控输入为归一化后的完整历史。这是本项目在指定参考的统计归一化上适配 MoE 输入的方案；`go2_rl_gym` 源码使用固定观测缩放，不能将两套运行统计描述为其原有机制。
- 借用MoECTS学生网络的共享骨干与门控结构，不引入教师、蒸馏或门控负载均衡。actor在原想象负回报上增加真实回放同一时刻的速度MSE，`velocity_loss_coef=1.0`；保持潜特征和预测速度的完整策略梯度，标签detach，二者使用同一actor优化器。
- 速度监督参考[DreamWaQ v2](https://arxiv.org/html/2301.10602v2) §II-C公式(6)与[HIMLoco@ef289ac](https://github.com/InternRobotics/HIMLoco/blob/ef289acaa62795009363b7b819c9186690630441/rsl_rl/rsl_rl/modules/him_estimator.py#L64-L108)的潜特征／速度分离归一化。这里仍采用MoE和SuperMimic BPTT，不迁入其VAE／对比目标或PPO，不宣称复现其效果。旧32维MoE权重无法直接载入，已删除失效的`load_pretrained`映射。
- 核心目标仍对应论文 §3、公式(1)～(3)：8 步自回归模型监督，32 步冻结模型权重的 BPTT，软存活与末端价值，固定 lambda-return，无价值目标网络。速度奖励与机器人差异由任务设计单独记录。
- 真实回放以半精度保存完整历史，想象起点均匀抽取全部已保存时刻；模型窗口目标读取下一回放槽，done 当步及其后均无效。终止时下一槽的 reset 后状态仍参与参考的稀有事件判定；仿真接口保留的 reset 前 `final_state` 不作为该窗口目标。
- 真实采集调度按用户指定的 [zita-ch/supermimic_vibed@8bfc115](https://github.com/zita-ch/supermimic_vibed/blob/8bfc1158bd968f729fcc65d5e897a05b05ea3d36/supermimic/train.py#L300-L322)：`collect_steps` 为每轮平均控制步数，默认2.5对应UTD=0.4，累加后取整按2／3交替；余量跨多次 `learn()` 保留，重置采集或恢复时重新初始化。UTD按整轮训练更新计数，模型拟合批数由各机器人的`algo.wm_updates`另行配置，Go2／RI新训练默认5批。这里只迁入其调度，不迁入G1任务或其他参数，也不视为作者官方实现。
- 预填与训练迭代分开：以 `int(collect_steps)` 为整数批次，循环采到回放长度大于 `wm_window+1`，默认8步窗口仍预填10步，不消耗训练累加器。模型归一化统计排除done当步。
- lambda-return 保留参考 `surv[t+1]/clamp(surv[t],1e-6)` 的低存活数值语义；critic 每个 epoch 按 `n//critic_minibatches` 分片，余数按来源丢弃。非有限梯度跳过对应优化器更新；`nonfinite_steps` 依来源统计模型及 actor，critic 另用 `critic_skipped` 记录。非有限日志值省略并报告跳过计数，不改变终端版式。
- 护栏每50轮判定一次，按来源对发生过任何reset的step求失败绝对数量均值，包含只有超时而跌倒数为0的reset step；没有reset的step不进入分母。依据 MjLab1.6 `manager_based_rl_env.py:434,471,614` 和 `termination_manager.py:95`，每step清空`extras["log"]`，仅reset路径输出终止统计。该值以 `guard/falls_per_reset_step` 记录。
- `load(path, continue_run=False, reset_critic=False)` 默认热启动并从第0轮重新计时；继续训练保留迭代号。模型形状变化时保留新构造的模型，策略更新推迟至至少1500轮；critic形状变化或显式重置时，actor冻结至策略更新起点之后500轮。只恢复已载入网络的同形状、同类型优化器，Muon权重衰减也须一致；不恢复回放、仿真、RNG、护栏学习率缩放或旧预热排期。
- MoE检查点和部署格式为 `supermimic_moe_velocity_v1`／`supermimic_moe_velocity_policy_v1`；actor形状不兼容时拒绝热启动。任务语义的继续训练检查由任务装配层负责，普通热启动允许任务语义变化。短流程验证不代表速度跟踪已收敛。

盲性边界：部署只输入五帧本体观测，速度由图内网络显式估计；训练用真实速度仅用于监督，不进入actor前向。地形、摩擦和COM等真值仍只用于世界模型与critic。速度标签从真实回放当前状态转换为body系，不使用世界模型想象速度作为标签。外部成绩不代表本组合已验证的效果。

## 与参考实现的有意差异

- 固定幅度与确定性想象按 [SuperMimic英文v1](../../paper/SuperMimic_2610.28480v1.pdf) §3.1／§3.3、表A.5：通用基础配置的采集噪声`action_noise=0.1`，想象时使用确定性动作；Go2／RI主线采用用户选定的本项目消融配方，动作缩放前标准差为0.3。回放保存执行动作；想象仅重采样观测噪声，不加动作探索噪声，也不创建幅度参数或探索critic。参考 [zita-ch/supermimic_vibed@0525a23](https://github.com/zita-ch/supermimic_vibed/blob/0525a23a2dcfa61f8543f98b84c075fe66d9d286/supermimic/train.py#L267)，交叉核对 [Motphys/Uni-SuperMimic@ac19b73](https://github.com/Motphys/Uni-SuperMimic/blob/ac19b73a4f919fb2ea37dbad860b012315497139/src/unisupermimic/rl/algos/supermimic/learner.py#L380)。两者为G1参考动作跟踪，仍保留本项目现有动作与环境接口，不据此推断四足盲速度任务的训练效果。主线只接受当前固定高斯采集与确定性想象配置；已删除旧策略快照迁移、退休字段吞掉及历史指纹常量，旧格式明确拒绝。
- 用户指定Go2／RI新训练启用`collect_det_frac=0.25`，其余探索环境保留固定高斯。依据为[allmimic@cb334ff6的采集及推荐配方](https://github.com/Shimingyi/allmimic/blob/cb334ff674b2f57248f33f8c21b7fb526fbdf8cc/allmimic/train.py#L399)，只借鉴动作噪声置零机制；本项目按地形类别分层、最大余数分配全局名额并在组内等距选择，替换来源的前段连续选择。分组固定，Go2／RI任务侧默认额外关闭站立命令环境的动作噪声，切回移动恢复原加噪资格；观测噪声和物理随机化沿用原配置；共享基础配置保持0。分组仅作用于真实采集，想象目标不变；尚未验证该分组的四足训练收益。
- 参考 `supermimic/train.py` 是单文件 `Trainer`，把采集、模型拟合与策略改进合在一起；本项目按 `AGENTS.md` 分层，任务 MDP 留在 `src/tasks/`，算法留在本目录，仅对齐模块名与职责。
- 世界模型只学习记录物理转移，不学习奖励头或失败头；奖励与软失败由任务侧解析给出。
- 梯度分批（`imag_microbatch`／`critic_microbatch`）保持总想象批量、完整时间反传与优化器步数；TorchScript／ONNX 导出属于本项目工程适配，不是论文结论。
- 继续训练额外保留真实／想象转移与实际优化器更新的累计日志计数；参考只恢复迭代号。该日志扩展不参与损失、排期或课程数学。
- 任务奖励与机器人配置见项目 [Go2 任务说明](../../docs/go2_supermimic.md)。

## PartyMimic Muon＋Adam

`optim.py`的参数分组、设备学习率更新与门控派生自[PartyMimic@69c53621 `optim.py:64–162`](https://github.com/Roboparty/PartyMimic/blob/69c53621e6394f3c5dd7a923d0a93dcfcd1fb8eb/src/partymimic/optim.py#L64)，默认策略首尾Linear也使用Muon及权重衰减采用同提交[`train.py:82–121,234–247`](https://github.com/Roboparty/PartyMimic/blob/69c53621e6394f3c5dd7a923d0a93dcfcd1fb8eb/src/partymimic/train.py#L82)。该本地适配由本项目维护，不依赖参考仓库运行。

Go2新训练的策略所有Linear权重用Muon，分组卷积及偏置用Adam；世界模型仅中间Linear权重用Muon；critic保留Adam。Muon沿用momentum=.95、Nesterov、5步Newton–Schulz、eps=1e-7、`match_rms_adamw`及默认weight_decay=.1；actor Adam beta=(.9,.95)，WM和critic为(.9,.999)，Adam权重衰减为0。原生更新及图内矩阵内核使用本机PyTorch2.14.1；恢复时按当前fused/capturable执行方式重建设备状态，并保留设备lr地址。

来源为人形参考动作MLP，本项目为含分组卷积与速度监督的Go2 MoE，沿用来源按Linear识别的分组，不将卷积展平为Muon矩阵。来源WM的FlatAdam在本项目使用同数学定义的现有fused Adam；保留本项目WM与actor有限性跳步、梯度裁剪、学习率排期和护栏，不迁入其他训练机制。优化器数值对拍及缩减预算训练不代表上游成绩或Go2控制收益已复现。

以下保留所派生代码的MIT版权与许可原文：

```text
MIT License

Copyright (c) 2026 PartyMimic authors
Copyright (c) 2026 Fuyi Chen (BlinkMimic, from which this software is derived)
Copyright (c) 2026 Huy-Quang-Dao (InstantTrack, from which BlinkMimic is derived)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
