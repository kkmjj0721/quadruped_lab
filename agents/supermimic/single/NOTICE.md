<!-- Copyright (c) 2026-2027 zh -->

# 单一 MoE actor 的来源与适配

本包依据下列论文公式和参考实现独立编写，不导入外部项目运行时。

- SuperMimic 英文 v1，§3公式1～3、附录D表A.5：记录动作下多步自回归、冻结模型权重但保留动作导数、软存活、末端价值与固定lambda-return。论文为G1动作跟踪，不能据此证明Go2盲走收敛。
- [Zhengtianjin/go2-supermimic](https://github.com/Zhengtianjin/go2-supermimic/tree/d9eee9b3f248e5300e636c433255db2c1c3e086a)，`supermimic/train.py`、`world_model.py`、`buffer.py`、`imagine.py`：预算、Welford归一化、单actor、探索、局部窗口采样、优先及归档、编译想象单步、学习率调度。本包不包含其追逐／闪避规则；任务向模型提供残差坐标、解析奖励与失败。
- 本地 `go2_rl_gym` 提交 `30e74dc507bec7a642a8c98be26081f2c6f0822d`，`rsl_rl/rsl_rl/modules/actor_critic_moe_cts.py:act_inference` 和 `modules/utils.py:StudentMoEEncoder`：复用本项目已对拍的八专家编码结构，保持混合后L2的原epsilon及latent在前拼接。移除CTS教师及蒸馏，整个部署网络直接接受BPTT梯度，是本项目适配。

与参考的明确差异：物理残差及真实终态保存遵循本任务协议；回放历史按真实reset重复首帧，终止转移本身保留监督；actor默认保留已有幅值1的平滑动作保护，可显式设None；首次采集16步保证五帧历史及8步窗口；崩溃保护使用明确单位“每环境每分钟跌倒次数”，只在已保存策略基础上回滚actor/critic权重并按参考重建其Adam优化器，世界模型、累计计数与回放保留。归档由任务提供稀有度，没有任务钩子时仅按真实失败优先。模型、控制效果与吞吐须分别验证，不将参考演示或本包单元测试当作策略成绩。

显存执行方式也是本项目适配：想象按种子分批完成完整时域反传，并按样本数累积梯度；critic在原小批内同样累积。总训练批量、梯度裁剪时机及优化器更新次数保持原定义，避免把整个8192条MoE想象图同时保留。参考项目未提供此方案，其README对16 GB地形训练建议显式减少环境数、想象批量和回放长度。分批改变随机噪声的抽样顺序，不能保证随机训练与整批逐位一致。
