<!-- Copyright (c) 2026-2027 zh -->

# Go2 SuperMimic 任务来源

当前任务由 `mdp/reference_task.py`、`reference_state.py`、`reference_rewards.py`、`commands.py`、`gait_rewards.py` 与集中配置定义。使用MoECTS部署网络、SuperMimic训练和HIM命令／接触奖励，不包含CTS教师或蒸馏。配置与使用见 [任务说明](../../../../docs/go2_supermimic.md)。

| 来源 | 固定版本与实际位置 | 采用内容与差异 |
| --- | --- | --- |
| SuperMimic 英文论文 | [本地 v1](../../../../paper/SuperMimic_2610.28480v1.pdf)，§3、公式1～3、表A.6 | 世界模型/BPTT方法和保留的六项速度奖励。无动作模仿先验，不能宣称复现论文速度实验成绩 |
| 指定 Go2 实现 | [Zhengtianjin/go2-supermimic](https://github.com/Zhengtianjin/go2-supermimic/tree/d9eee9b3f248e5300e636c433255db2c1c3e086a)，`supermimic/{train,world_model,buffer,imagine}.py`、`env/{env_cfg,terrains}.py`、`target/terrain.py`、`robot/go2.py` | 参考状态/位姿增量、训练算法与预算、倾角终止、执行器、地形／课程／随机化；原追逐/闪避改为盲速度任务，actor改MoE，保留187维世界模型输入含编码器偏置。算法来源及差异见 [算法说明](../../../../agents/supermimic/NOTICE.md) |
| 参考 Go2 资产 | 指定Go2仓库与其 Menagerie 来源 | 模型、关节顺序、碰撞和执行器适配本项目；原始许可和归属见 [资产说明](../../../../src/assets/robots/go2/REFERENCE_NOTICE.md) |
| HIM命令及接触奖励 | `/home/sunteng/projects/legged_wbc_mjlab`，`e6209ec3d910b57673d5cbba1d3ca798a7fa3113`；`go2_him/config/env_cfgs.py`、`ri_4438_him/mdp/{rewards,gait,velocity_command}.py` 和 MjLab1.6 `velocity_command.py` | 用户指定的周期／权重、逐脚接触相位与当前时长评分、heading采样逻辑；vx/vy±1、4–12秒、无速度课程。真实端读取原生接触计时；想象端足高软接触属于用户已确认适配，非来源等价实现 |
| 姿态奖励 | 同一仓库 `go2_ppo/mdp/rewards.py:variable_posture`、`go2_ppo/config/env_cfgs.py` | 按命令速度选逐关节std的指数奖励，保留本项目默认姿态，不移入HIM地形容差版本 |
| 垂向速度／倾斜／动作二阶差分 | 同一仓库 `ri_4438_him/mdp/rewards.py:lin_vel_z/body_orientation_l2`，来源安装的MjLab1.6 `envs/mdp/rewards.py:action_acc_l2`、`managers/action_manager.py:process_action/reset` | 使用body-z速度平方、机身重力xy平方和、未缩放动作二阶差分平方和。权重按用户要求-2／-.2／-.01；记忆42与critic210包含前两步动作，actor与世界模型形状不变 |
| MjLab / MuJoCo | 本机 MjLab1.6.0、MuJoCo3.11.0、MuJoCo-Warp3.11.0；参考本地环境MuJoCo3.17.0、MuJoCo-Warp3.13.0 | 参数与算法按参考；来源依赖未固定MuJoCo/Warp版本，不能声称两环境轨迹数值一致。当前Warp无来源的迭代告警位，省略对应纯日志startup事件，不增加兼容回退。终端摘要保留当前格式 |

短流程只验证实现与运行接口，不能证明长期行走收敛、论文性能或真机部署效果。当前设计和验收范围见 [设计](../../../../docs/design/go2_blind_supermimic.md) 与 [独立审查记录](../../../../docs/review/go2_blind_supermimic.md)。
