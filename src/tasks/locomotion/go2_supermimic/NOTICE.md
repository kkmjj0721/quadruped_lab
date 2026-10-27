<!-- Copyright (c) 2026-2027 zh -->

# Go2 SuperMimic 任务来源

当前任务由 `mdp/reference_task.py`、`mdp/reference_state.py`、`mdp/reference_rewards.py` 与 `src/config/go2/go2_reference_supermimic_config.py` 定义。历史 CTS、MoE、HIM 奖励及兼容实现已删除。配置与使用见 [任务说明](../../../../docs/go2_supermimic.md)。

| 来源 | 固定版本与实际位置 | 采用内容与差异 |
| --- | --- | --- |
| SuperMimic 英文论文 | [本地 v1](../../../../paper/SuperMimic_2610.28480v1.pdf)，§3、公式1～3、表A.3/A.6和附录G | 世界模型/BPTT方法、六项速度奖励、均匀速度命令支路与部分随机化幅度。无动作模仿先验，不能宣称复现论文速度实验成绩 |
| 指定 Go2 实现 | [Zhengtianjin/go2-supermimic](https://github.com/Zhengtianjin/go2-supermimic/tree/d9eee9b3f248e5300e636c433255db2c1c3e086a)，`supermimic/{train,world_model,buffer,imagine}.py`、`robot/go2.py` | 单MLP、参考状态/位姿增量、训练预算、倾角终止和执行器。原任务追逐/闪避改为论文速度奖励，actor只接收盲本体历史；模型补齐会影响MjLab执行器目标的编码器偏置上下文。算法来源及差异见 [算法说明](../../../../agents/supermimic/single/NOTICE.md) |
| 参考 Go2 资产 | 指定Go2仓库与其 Menagerie 来源 | 模型、关节顺序、碰撞和执行器适配本项目；原始许可和归属见 [资产说明](../../../../src/assets/robots/go2/REFERENCE_NOTICE.md) |
| 指定地形项目 | `/home/sunteng/projects/legged_wbc_mjlab`，`e6209ec3d910b57673d5cbba1d3ca798a7fa3113`；`src/tasks/locomotion/ri_4438_him_lpacrl/mdp/terrains.py`、`src/config/ri_4438/ri_4438_him_lpacrl_config.py:terrain` | 沿用项目连通地形家族与几何参数；rough 是盲地形扩展。独立种子不保证与来源随机地形逐点一致，未使用其奖励和算法 |
| 地形课程 | `/home/sunteng/projects/go2_rl_gym`，`30e74dc507bec7a642a8c98be26081f2c6f0822d`；`legged_gym/envs/base/legged_robot.py:_update_terrain_curriculum` | 可选按移动距离晋降地形难度，默认关闭；命令固定为当前论文范围，没有旧速度课程 |
| MjLab / MuJoCo | 本机 MjLab 1.6.0 的 `manager_based_rl_env.py:step/reset`、`envs/mdp/actions/actions.py:apply_actions`；MuJoCo 3.11.0 | adapter复制终态后重置，记录真实上下文；静态地形查询保留台阶不光滑性，不将它视为完整可微物理仿真器 |

短流程只验证实现与运行接口，不能证明长期行走收敛、论文性能或真机部署效果。当前设计和验收范围见 [设计](../../../../docs/design/go2_blind_supermimic.md) 与 [独立审查记录](../../../../docs/review/go2_blind_supermimic.md)。
