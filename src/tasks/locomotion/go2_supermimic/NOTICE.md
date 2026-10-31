<!-- Copyright (c) 2026-2027 zh -->

# Go2 SuperMimic 任务来源

当前任务由 `mdp/reference_task.py`、`reference_state.py`、`reference_rewards.py`、`commands.py`、`gait_rewards.py`、`rl/world_model.py` 与集中配置定义。使用MoECTS部署网络、SuperMimic训练和HIM命令／接触奖励，不包含CTS教师或蒸馏。配置与使用见 [任务说明](../../../../docs/go2_supermimic.md)。

| 来源 | 固定版本与实际位置 | 采用内容与差异 |
| --- | --- | --- |
| SuperMimic 英文论文 | [本地 v1](../../../../paper/SuperMimic_2610.28480v1.pdf)，§3、公式1～3、表A.6 | 世界模型/BPTT方法及速度任务奖励的初始依据；当前按用户要求改为body平移、偏航分母0.25并移除新训练侧倾。无动作模仿先验，不能宣称复现论文速度实验成绩 |
| 指定 Go2 实现 | [Zhengtianjin/go2-supermimic](https://github.com/Zhengtianjin/go2-supermimic/tree/d9eee9b3f248e5300e636c433255db2c1c3e086a)，`supermimic/{train,world_model,buffer,imagine}.py`、`env/{env_cfg,terrains}.py`、`target/terrain.py`、`robot/go2.py` | 参考状态/位姿增量、训练算法与预算、倾角终止、地形／课程／随机化；原追逐/闪避改为盲速度任务，actor改MoE。当前模型输入239含编码器偏置及52维瞬时辅助量，输出125；动作比例保留用户当前按关节类型的显式设置。算法来源及差异见 [算法说明](../../../../agents/supermimic/NOTICE.md) |
| 参考 Go2 资产 | 指定Go2仓库与其 Menagerie 来源 | 模型、关节顺序和碰撞适配本项目；原始许可和归属见 [资产说明](../../../../src/assets/robots/go2/REFERENCE_NOTICE.md) |
| Go2内置位置执行器 | `/home/sunteng/projects/legged_wbc_mjlab@4eb8fcf71124b20ed4a1154b6c724ba962a424f8`，`src/assets/robots/go2/go2_constants.py`、`src/config/go2/go2_config.py` | 按用户要求替换为 `BuiltinPositionActuatorCfg`，同步 PD 增益、力矩上限、armature 与延迟参数；仍使用本项目的参考 MJCF 与碰撞配置 |
| 地形难度升降 | 同提交 `go2_him/go2_him_env_cfg.py:497`、转导的 `ri_4438_him/mdp/curriculums.py:terrain_levels_vel`、`mdp/velocity_command.py:HeadingHoldVelocityCommand`及MjLab1.6 `TerrainEntity.update_env_origins` | 仅迁入按行走表现升降、命令距离积分及最高级随机回落；保留当前Ri-5036地形布局和零级起点，不迁入速度课程。真实adapter在命令推进前结算，沿用本项目forward后终态位置读取；不宣称与来源完整物理轨迹数值相同 |
| HIM命令及接触奖励 | `/home/sunteng/projects/legged_wbc_mjlab`，`e6209ec3d910b57673d5cbba1d3ca798a7fa3113`；`go2_him/config/env_cfgs.py`、`ri_4438_him/mdp/{rewards,gait,velocity_command}.py` 和 MjLab1.6 `velocity_command.py` | 用户指定的周期／权重、逐脚接触相位与当前时长评分、heading采样逻辑；4–12秒重采样、无速度课程。新训练vx/vy恢复来源的±1范围，原±4快照按原配置保留；heading反馈增益保留。真实端读取原生接触计时；想象端共用世界模型受监督的found概率并连续计时，属于本项目可微适配，旧快照保留原足高近似 |
| 策略相位 | `/home/sunteng/projects/legged_wbc_mjlab@4eb8fcf71124b20ed4a1154b6c724ba962a424f8`，`ri_4438_him/mdp/observations.py:phase`与`gait.py:gait_phase` | 新训练在原45维末尾追加sin／cos，与现有步态周期和站立门控共用时钟；HIM相位在command后且使用6帧，本项目追加末尾、使用5帧与SuperMimic，不宣称完整HIM复现。该版本实际未启用足高奖励，本项目不增加足高目标 |
| 姿态奖励 | 同一仓库 `go2_ppo/mdp/rewards.py:variable_posture`、`go2_ppo/config/env_cfgs.py` | 按命令速度选逐关节std的指数奖励，保留本项目默认姿态，不移入HIM地形容差版本 |
| 垂向速度／倾斜／动作二阶差分 | 同一仓库 `ri_4438_him/mdp/rewards.py:lin_vel_z/body_orientation_l2`，来源安装的MjLab1.6 `envs/mdp/rewards.py:action_acc_l2`、`managers/action_manager.py:process_action/reset` | 使用body-z速度平方、机身重力xy平方和、未缩放动作二阶差分平方和。权重按用户要求-2／-.2／-.01；记忆42保留前两步动作，critic另含瞬时辅助量 |
| 站立／足端滑移／关节加速度 | 同一仓库 `go2_him/go2_him_env_cfg.py`、`ri_4438_him/mdp/rewards.py:stand_still`；来源MjLab1.6 `tasks/velocity/mdp/rewards.py:feet_slip`、`envs/mdp/rewards.py:joint_acc_l2` | 当前权重-2.0／-.1／-2.5e-7，其中站立权重按用户要求覆盖来源-.5；站立／滑移命令阈值均.1。真实端使用默认角平方偏差、世界瞬时足速与found、qacc；想象端使用扩展连续量与sigmoid接触概率，接触logit归一化固定0／1。当前adapter最后forward后的读取与原生manager前的计算时序不同，不宣称轨迹数值等价；新增头与概率监督属于本项目适配，非论文原有结构 |
| 站立关节速度 | 来源安装的MjLab1.6 `envs/mdp/rewards.py:joint_vel_l2`；本项目 `mdp/reference_rewards.py:stand_still_joint_vel_l2` | 复用关节速度平方和公式，新增与站立角度项相同的起点命令门控；−0.1是本项目待训练验证的初值，不是来源推荐权重。真实／想象复用原有关节速度状态，不增加模型维度；已有十五项v4快照显式禁用本项并保留原指纹 |
| 绊脚／关节力矩 | `/home/sunteng/projects/legged_wbc_mjlab@4eb8fcf71124b20ed4a1154b6c724ba962a424f8`，`go2_him/go2_him_env_cfg.py:407,481`、转导的 `ri_4438_him/mdp/rewards.py:488` 及来源MjLab1.6 `envs/mdp/rewards.py:32` | 按用户指定权重−.1／−2e−5与比值5：任一脚世界净接触力xy模长严格大于5倍竖直力绝对值；实际actuator_force平方和。真实端采集力，想象端预测下一绝对力并共用公式；硬绊脚无直接局部梯度，不擅自软化。新增24个连续辅助量属于本项目模型适配，预测质量与控制效果未验证；旧97／234完整快照保留原v4语义并禁用两项 |
| 腿部碰撞 | 同一提交 `go2_him/go2_him_env_cfg.py:465`、`config/env_cfgs.py:97` 与来源MjLab1.6 `tasks/velocity/mdp/terminations.py:illegal_contact` | 权重−1、力阈值1 N、逐环境any硬判定。传感器沿用none／单槽／单帧历史，匹配本项目小腿几何体时允许collision后的数字；仅腿部对地，不含足端和自碰。真实端将历史力模最大值压为一个N单位辅助量，想象端连续预测该量并共用严格大于阈值的公式，无直接局部梯度，不替换成概率奖励；当前v7物理125／critic262，旧v4／v5／v6完整快照保留原协议及相应奖励，预测与控制效果待验证 |
| 机体系倾转角速度 | 同一HIM提交 `ri_4438_him/mdp/rewards.py:88` 的 `body_angular_velocity_penalty` | 按用户后续要求，新默认 `body_ang_vel` 直接计算body角速度xy平方和，权重−.05不变；旧完整快照恢复heading并保留原指纹，独立paper实验仍使用其原定义；物理／actor／critic宽度不变 |
| 机体系平移／角动量／髋关节／落地 | `legged_wbc_mjlab@4eb8fcf71124b20ed4a1154b6c724ba962a424f8`，`ri_4438_him/mdp/rewards.py:26,79,413`、MjLab1.6 `tasks/velocity/mdp/rewards.py:356`与`ContactSensor.compute_first_contact` | 平移按body-xy，偏航仅按用户要求将分母改0.25、仍world-z；新增三项权重沿用来源−.025／−.5／−.002。角动量取既有root_angmom并旋转到heading，norm²严格不变；髋项保留侧移／偏航门控。真实落地保留硬时间窗口；想象按用户确认用条件时长及接触概率加权，是代理而非严格无偏期望。新增3维连续头后物理125／critic262；新默认21项无侧倾，旧v4／v5／v6完整快照保留原定义及指纹；效果未训练验证 |
| MjLab / MuJoCo | 本机 MjLab1.6.0、MuJoCo3.11.0、MuJoCo-Warp3.11.0；参考本地环境MuJoCo3.17.0、MuJoCo-Warp3.13.0 | 参数与算法按参考；来源依赖未固定MuJoCo/Warp版本，不能声称两环境轨迹数值一致。当前Warp无来源的迭代告警位，省略对应纯日志startup事件，不增加兼容回退。终端摘要保留当前格式 |

短流程只验证实现与运行接口，不能证明长期行走收敛、论文性能或真机部署效果。当前任务和验证范围见 [任务说明](../../../../docs/go2_supermimic.md)。
