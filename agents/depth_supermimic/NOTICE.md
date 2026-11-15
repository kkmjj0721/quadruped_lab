<!-- Copyright (c) 2026-2027 zh -->

# 深度视觉 SuperMimic 来源与维护边界

训练主干来自 [InstinctMJ@03563ee02665f327425106372264070315673748](https://github.com/project-instinct/InstinctMJ/tree/03563ee02665f327425106372264070315673748)：实际对照 `src/instinct_mj/rl/supermimic/parkour/{runner,config,mdp}.py`、共享 `networks.py` 和 `buffer.py`。四专家策略和MLP另对照锁定的 [instinct_rl@3a2844890387eda6d93a4465cdef9e767aba8546](https://github.com/project-instinct/instinct_rl/tree/3a2844890387eda6d93a4465cdef9e767aba8546) 的 `modules/{moe,moe_actor_critic,mlp}.py`。本目录由本项目独立维护，不依赖这些仓库运行。

InstinctMJ和instinct_rl顶层按CC BY-NC 4.0发布，原始许可全文保存在 [LICENSE-InstinctMJ](LICENSE-InstinctMJ)。InstinctMJ的SuperMimic共享实现注明派生自Chong Zhang／ETH Zurich的MIT参考代码，原始许可全文见 [LICENSE-supermimic_vibed](LICENSE-supermimic_vibed)。这些声明分别保留，不将整个视觉移植包统称为MIT。

当前双帧CNN和单层GRU按用户要求采用 [MGDP@f43139d1d97849819a0dce609dc0b59faf577b3f](https://github.com/arclab-hku/MGDP/tree/f43139d1d97849819a0dce609dc0b59faf577b3f) 的 `legged_gym/legged_gym/envs/random_dog/utils/net_work.py::DoubleConv,ImageEncoder,Memory`。保留32／64／96通道、96瓶颈、八组GroupNorm、SiLU、两次MaxPool/2、4×4平均池化及1536→192→64投影。固定提交的 `legged_gym/setup.py` 声明包许可为BSD-3-Clause；[LICENSE-MGDP](LICENSE-MGDP) 原样保存该包 `legged_gym/legged_gym/utils/__init__.py` 所携带的BSD声明。`net_work.py`自身没有另附许可头，根目录没有独立LICENSE；这里保留已核实的包级声明，不虚构MGDP作者的独立版权文本。

单层GRU用数值等价的GRUCell逐步显式执行，以适配本项目fullgraph编译；权重维度与原单层GRU一致。用户明确要求保留固定源码的有效梯度边界：GRU权重冻结，不进入任何优化器。CNN重建与世界模型监督消费GRU之前的64维token，真实动作和想象动作消费GRU之后的64维状态。此选择对应固定源码 `random_dog.py:203–220,363–369,566–608` 和 `legged_gym/rl/MGDP/algorithms/ppo.py:41`：感知损失不经过GRU，GRU输出detach后送入策略。论文§6.1所述GRU随策略优化与该固定代码不同，本实现按用户选择遵循代码边界。不引入MGDP的高度图自编码器、InfoNCE或PPO，64维不声明已经形成共享32维与私有32维的语义分离。

输入为直接渲染的双帧32×32深度，最旧帧在前，间隔5个控制步。相机保留原16:9角域（HFOV约89.16°、VFOV58°），以32×32分辨率、50Hz采样；闭区间整数延迟0／1／2步对应0／20／40ms，8帧历史环覆盖全部索引。删除预处理缩图与最小池化，直接逐像素归一化和加噪；无效为0，合法最远量程归一化为1。干净双帧仅用于训练重建标签，解码器从32×8×8起点重建2×32×32。32×32是用户指定的输入适配，不是MGDP原16×16基准。旧v2的有效位通道和IMU／帧龄输入已淘汰。

带噪到干净重建沿用本项目masked L1，权重1，只有clean有效像素进入分子与分母；空有效区域贡献可反传零。新帧在米制上先加σ=.03m高斯噪声，再以p=.1独立像素置零，噪声参考MGDP `random_dog.py::get_camera_sensor` 与 `random_dog_config_stage1.py`，两路共享历史和延迟，回放保存reset前clean标签，统一 `noise.add_noise` 控制噪声。不引入MGDP双线性预处理、5m归一化或MSE，也不迁移其10Hz相机时序；这些是本项目已明确保留的任务条件，不声称MGDP整体复现。

保留四专家MoE、五帧本体、8步模型监督、32步解析策略梯度、停梯度CNN潜变量目标及参考归一化。按用户要求参考本项目Go2，在原45维本体末尾追加无噪声的步态相位sin/cos，默认47维／五帧235维；相位复用步态奖励的0.6秒时钟，仅在平移命令模长加偏航命令绝对值大于0.1时启用，站立置零。实现依据为 `src/tasks/locomotion/go2_supermimic/mdp/reference_task.py::observe` 和 `gait_rewards.py::gait_phase`，Go1原45维顺序及缩放保持自身定义。门控和四专家均接收235+64=299维；显式关闭相位的旧45维配置仍按原规格运行，不转换其权重。GRU状态作为显式输入输出，回合结束或外部reset仅清零对应环境。回放保存动作推理前的FP32 GRU状态，CNN token每次由当前网络重新计算，不缓存视觉token。想象由模型预测GRU前token增量，每一步通过同一个固定GRU推进状态。

回放深度使用round量化uint8，默认本体历史采用本项目Go2的FP16单帧压缩，保存实际采集的带噪帧并按回合边界重建，不重新采噪。显式full模式保留按起点年龄和回合长度补齐、同一逻辑帧共享重采噪的基线。终止当步保留reset前标签，其后的跨回合模型转移被屏蔽。4096环境、500容量下三份双帧深度占11.71875GiB，FP32视觉记忆约.488281GiB，尚不含物理状态、采样和网络显存。

近裁剪后继续投射沿用同一InstinctMJ的 `sensors/grouped_ray_caster/grouped_ray_caster.py::_apply_hit_filter_and_continue`，默认六次跳转及1e-4偏移。Go1 FK读取本项目资产，树状递推沿用本地Go2所保留的BlinkMimic MIT实现，原始声明见 `src/tasks/locomotion/go2_supermimic/mdp/LICENSE_blinkmimic`。

用户指定的任务差异包括G1换为Go1、目标点换为速度指令、移除AMP与参考动作依赖，地形和全部22项奖励采用本项目Go2 rough。为这些奖励，世界模型额外预测40维瞬时辅助量，标准化MSE权重为1；它不新增奖励，不替代物理增量或接触BCE。失败头真实和预测两路监督共享该转移实际指令及记录的下一状态接触计时；heading任务共享已记录命令状态；想象计时仍由预测接触推进。

采集和排期按用户指定的Go2配方，并按后续要求将回放缩至每环境500步：4096环境、1024×3世界模型、每轮5批2048×8模型监督，2048条32步想象。无模型预热，基础率4e-4，actor在1000轮内渐增至4倍，30000轮余弦衰减至3%；动作噪声σ=.3，按地形分层固定25%环境关额外噪声，站立命令额外关噪。真实与想象共享前后／侧移／转向和heading反馈。Critic为286维；默认本体帧47维、历史5帧。

执行优化复用 `agents/supermimic/compilation.py` 与 `cuda_graph.py::CapturedUpdate`。默认CUDA启用整函数编译和Graph，保持模型、失败头、actor、critic更新和采样的次数与语义。想象微批上限4096，critic微批32768。默认Muon＋Adam复用 `agents/supermimic/optim.py::MuonAdam`，来源 [PartyMimic@69c53621e6394f3c5dd7a923d0a93dcfcd1fb8eb](https://github.com/Roboparty/PartyMimic/tree/69c53621e6394f3c5dd7a923d0a93dcfcd1fb8eb)，声明见共享 [NOTICE.md](../supermimic/NOTICE.md#partymimic-muonadam)。世界模型只对中间Linear权重用Muon，actor的门控和专家Linear权重用Muon；CNN／解码器及其余参数用Adam，CUDA采用融合Adam。GRU完全排除在优化器外。CUDA矩阵和卷积用BF16，主参数、优化器状态、物理／归一化及返回值保持FP32；部署副本统一FP32并记录训练来源精度。

检查点格式为 `depth_supermimic_mgdp_gru_v3`；任务快照必须显式标记protocol_version=3；旧v1/v2权重和完整快照在加载前拒绝，不转换参数或优化器。恢复核对完整规格、任务语义及原环境数，重新初始化环境、相机、回放与GRU。部署格式为 `depth_supermimic_policy_v3`，三输入为本体历史、双帧深度、visual_hidden，双输出为actions和next_visual_hidden；调用方逐步回传状态，并在结束行清零。

上述结构和数值检查不代表Go1通行、收敛或论文效果已复现。参考项目的机器人、仿真器、相机时序和训练方法与本任务不同，控制效果需要专门同预算评估。
