<!-- Copyright (c) 2026-2027 zh -->

# 混合专家历史策略的来源与适配

## 当前 MoE 基准

`supermimic_moe_v1` 按用户确认的 Go2 盲（本体感知）速度跟踪路线实现：算法沿用 SuperMimic，actor 采用 `go2_rl_gym` 的混合专家网络，其余训练机制对齐指定参考实现。

- 训练机制参考 [Zhengtianjin/go2-supermimic](https://github.com/Zhengtianjin/go2-supermimic/tree/d9eee9b3f248e5300e636c433255db2c1c3e086a)，已实读固定提交的 `supermimic/{train,world_model,state,buffer,imagine,driver}.py`。
- actor 结构参考 [wty-yy/go2_rl_gym](https://github.com/wty-yy/go2_rl_gym/tree/30e74dc507bec7a642a8c98be26081f2c6f0822d)，已实读 `rsl_rl/rsl_rl/modules/{utils,actor_critic_moe_cts}.py`、`algorithms/ac_moe_cts.py`、`legged_gym/utils/exporter.py`。
- `MoeActor` 对齐 `utils.py:StudentMoEEncoder`：完整五帧历史（225 维）经共享 ELU 骨干(512,256)得到 8×256，按专家分组的 1×1 卷积输出 8×32，Softmax 门控加权求和后 `L2Norm` 得 32 维隐变量；隐变量与独立归一化的当前帧 45 维拼接，经 ELU MLP(512,256,128) 输出 12 维未缩放动作。
- 历史与当前帧使用两套独立 `RunningNorm`，对应 `go2_rl_gym` 的 student 编码器输入与单帧归一化分工；门控输入为归一化后的完整历史。
- 负载均衡损失按 `ac_moe_cts.py` 实现（使用率方差、系数 0.01），按整个想象批次统计门控使用率后与策略目标共同反传。
- 状态字典键名与 `go2_rl_gym` 的 `StudentMoEEncoder` 完全一致，同权重前向逐位相等；`MoeActor.load_pretrained` 可按预训练导出名的展平命名热启动，已对 `go2_moe_cts_137k_0.6739.pt` 验证动作差 ≤1e-6。
- 核心目标仍对应论文 §3、公式(1)～(3)：8 步自回归模型监督，32 步冻结模型权重的 BPTT，软存活与末端价值，固定 lambda-return，无价值目标网络。速度奖励与机器人差异由任务设计单独记录。
- 检查点和导出仅支持 `supermimic_moe_v1` 格式。真实回放、终态保存、梯度分批和参数续训保留为本项目工程适配；短流程验证不代表速度跟踪已收敛。

盲性边界：actor 只读取五帧本体观测，不含线速度、地形扫描、摩擦或 COM；线速度必须由历史经专家隐式估计。地形与仿真上下文只进入世界模型与 critic。MoE 结构来自 `go2_rl_gym` 而非论文，因此其收敛性不属于论文结论。
