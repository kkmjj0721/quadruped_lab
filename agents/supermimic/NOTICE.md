<!-- Copyright (c) 2026-2027 zh -->

# SuperMimic 来源与适配

## 8MoE主线与SuperMimic机制

训练使用 `moe_history`、默认8个专家，检查点格式为 `supermimic_moe_v1`。本目录作为本项目源码维护，不依赖参考项目的Python运行环境。

各机器人在`src/config/<机器人>/*supermimic_config.py`的`algo`与`runner`段声明实际训练配方；本目录只维护共享字段定义、通用默认、校验与算法实现。CLI显式覆盖已加载配置，新增运行预算不进入任务语义指纹。

`supermimic_moe_v1` 按用户确认的 Go2 盲（本体感知）速度跟踪路线实现：算法沿用 SuperMimic，actor 采用 `go2_rl_gym` 的混合专家网络，其余训练机制对齐指定参考实现。

- 训练机制参考 [Zhengtianjin/go2-supermimic](https://github.com/Zhengtianjin/go2-supermimic/tree/d9eee9b3f248e5300e636c433255db2c1c3e086a)，已实读固定提交的 `supermimic/{train,world_model,state,buffer,imagine,driver}.py`。本目录按职责平铺对应模块：`world_model.py`（`RunningNorm`／`WorldModel`）、`policy.py`（混合专家 actor 与 critic）、`buffer.py`、`imagine.py`、`learner.py`、`runner.py`；驱动器装配留在任务侧 `src/tasks/locomotion/go2_supermimic/rl/driver.py`。
- actor 结构参考 [wty-yy/go2_rl_gym](https://github.com/wty-yy/go2_rl_gym/tree/30e74dc507bec7a642a8c98be26081f2c6f0822d)，已实读 `rsl_rl/rsl_rl/modules/{utils,actor_critic_moe_cts}.py`、`algorithms/ac_moe_cts.py`、`legged_gym/utils/exporter.py`。
- `MoeActor` 对齐 `utils.py:StudentMoEEncoder`的网络结构：完整五帧历史经共享ELU骨干(512,256)得到8×256，按专家分组的1×1卷积输出8×32，Softmax门控加权求和后`L2Norm`得32维隐变量；与独立归一化的当前帧拼接，经ELU MLP(512,256,128)输出12维未缩放动作。观测宽度由任务提供：当前Go2追加相位后为47／235维，旧任务仍为45／225维；输入布局属于本项目适配，不能直接互换旧权重。
- 历史与当前帧使用两套独立 `RunningNorm`，门控输入为归一化后的完整历史。这是本项目在指定参考的统计归一化上适配 MoE 输入的方案；`go2_rl_gym` 源码使用固定观测缩放，不能将两套运行统计描述为其原有机制。
- 只借用 MoECTS 学生部署网络，不引入教师、蒸馏或门控负载均衡目标。策略仅最大化参考的想象回报。
- 状态字典键名与 `go2_rl_gym` 的 `StudentMoEEncoder` 完全一致，同权重前向逐位相等；`MoeActor.load_pretrained` 可按预训练导出名的展平命名热启动，已对 `go2_moe_cts_137k_0.6739.pt` 验证动作差 ≤1e-6。
- 核心目标仍对应论文 §3、公式(1)～(3)：8 步自回归模型监督，32 步冻结模型权重的 BPTT，软存活与末端价值，固定 lambda-return，无价值目标网络。速度奖励与机器人差异由任务设计单独记录。
- 真实回放以半精度保存完整历史，想象起点均匀抽取全部已保存时刻；模型窗口目标读取下一回放槽，done 当步及其后均无效。终止时下一槽的 reset 后状态仍参与参考的稀有事件判定；仿真接口保留的 reset 前 `final_state` 不作为该窗口目标。
- 真实采集调度按用户指定的 [zita-ch/supermimic_vibed@8bfc115](https://github.com/zita-ch/supermimic_vibed/blob/8bfc1158bd968f729fcc65d5e897a05b05ea3d36/supermimic/train.py#L300-L322)：`collect_steps` 为每轮平均控制步数，默认2.5对应UTD=0.4，累加后取整按2／3交替；余量跨多次 `learn()` 保留，重置采集或恢复时重新初始化。UTD按整轮训练更新计数，模型拟合批数由各机器人的`algo.wm_updates`另行配置，Go2／RI新训练默认5批。这里只迁入其调度，不迁入G1任务或其他参数，也不视为作者官方实现。
- 预填与训练迭代分开：以 `int(collect_steps)` 为整数批次，循环采到回放长度大于 `wm_window+1`，默认8步窗口仍预填10步，不消耗训练累加器。旧整数快照保留原预算。模型归一化统计排除done当步。
- lambda-return 保留参考 `surv[t+1]/clamp(surv[t],1e-6)` 的低存活数值语义；critic 每个 epoch 按 `n//critic_minibatches` 分片，余数按来源丢弃。非有限梯度跳过对应优化器更新；`nonfinite_steps` 依来源统计模型及 actor，critic 另用 `critic_skipped` 记录。非有限日志值省略并报告跳过计数，不改变终端版式。
- 护栏每50轮判定一次，按来源对发生过任何reset的step求失败绝对数量均值，包含只有超时而跌倒数为0的reset step；没有reset的step不进入分母。依据 MjLab1.6 `manager_based_rl_env.py:434,471,614` 和 `termination_manager.py:95`，每step清空`extras["log"]`，仅reset路径输出终止统计。该值以 `guard/falls_per_reset_step` 记录。
- `load(path, continue_run=False, reset_critic=False)` 默认热启动并从第0轮重新计时；继续训练保留迭代号。模型形状变化时保留新构造的模型，策略更新推迟至至少1500轮；critic形状变化或显式重置时，actor冻结至策略更新起点之后500轮。只恢复已载入网络的同形状优化器，不恢复回放、仿真、RNG、护栏学习率缩放或旧预热排期。
- MoE检查点和部署格式为 `supermimic_moe_v1`／`supermimic_moe_policy_v1`；actor形状不兼容时拒绝热启动。任务语义的继续训练检查由任务装配层负责，普通热启动允许任务语义变化。短流程验证不代表速度跟踪已收敛。

盲性边界：actor部署时只读取五帧本体观测，不含真实线速度、地形扫描、摩擦或COM；线速度由历史隐式估计。地形与仿真上下文只进入世界模型与critic。外部论文和项目的训练成绩不代表本组合已验证的效果。

## 与参考实现的有意差异

- 固定幅度与确定性想象按 [SuperMimic英文v1](../../paper/SuperMimic_2610.28480v1.pdf) §3.1／§3.3、表A.5：通用基础配置的采集噪声`action_noise=0.1`，想象时使用确定性动作；Go2／RI主线采用用户选定的本项目消融配方，动作缩放前标准差为0.3。回放保存执行动作；想象仅重采样观测噪声，不加动作探索噪声，也不创建幅度参数或探索critic。参考 [zita-ch/supermimic_vibed@0525a23](https://github.com/zita-ch/supermimic_vibed/blob/0525a23a2dcfa61f8543f98b84c075fe66d9d286/supermimic/train.py#L267)，交叉核对 [Motphys/Uni-SuperMimic@ac19b73](https://github.com/Motphys/Uni-SuperMimic/blob/ac19b73a4f919fb2ea37dbad860b012315497139/src/unisupermimic/rl/algos/supermimic/learner.py#L380)。两者为G1参考动作跟踪，仍保留本项目现有动作与环境接口，不据此推断四足盲速度任务的训练效果。主线已移除可学习幅度、随机想象和OU探索；读取历史快照时明确拒绝这些已删除方案。旧固定高斯、确定性想象且关闭OU的默认方案保留原任务指纹，可继续播放和恢复；历史schema中的OU关闭与默认参数仅作为hash常量，不恢复对应运行分支。
- 用户指定Go2／RI新训练启用`collect_det_frac=0.25`，其余探索环境保留固定高斯。依据为[allmimic@cb334ff6的采集及推荐配方](https://github.com/Shimingyi/allmimic/blob/cb334ff674b2f57248f33f8c21b7fb526fbdf8cc/allmimic/train.py#L399)，只借鉴动作噪声置零机制；本项目按地形类别分层、最大余数分配全局名额并在组内等距选择，替换来源的前段连续选择。分组固定，Go2／RI任务侧默认额外关闭站立命令环境的动作噪声，切回移动恢复原加噪资格；观测噪声和物理随机化沿用原配置；共享基础配置保持0，旧完整快照缺失字段恢复0。分组仅作用于真实采集，想象目标不变；尚未验证该分组的四足训练收益。
- 参考 `supermimic/train.py` 是单文件 `Trainer`，把采集、模型拟合与策略改进合在一起；本项目按 `AGENTS.md` 分层，任务 MDP 留在 `src/tasks/`，算法留在本目录，仅对齐模块名与职责。
- 世界模型只学习记录物理转移，不学习奖励头或失败头；奖励与软失败由任务侧解析给出。
- 梯度分批（`imag_microbatch`／`critic_microbatch`）保持总想象批量、完整时间反传与优化器步数；TorchScript／ONNX 导出属于本项目工程适配，不是论文结论。
- 继续训练额外保留真实／想象转移与实际优化器更新的累计日志计数；参考只恢复迭代号。该日志扩展不参与损失、排期或课程数学。
- 任务奖励与机器人配置见项目 [Go2 任务说明](../../docs/go2_supermimic.md)。
