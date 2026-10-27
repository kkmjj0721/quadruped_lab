<!-- Copyright (c) 2026-2027 zh -->

# Go2 SuperMimic任务来源

2026-10-27新增的默认单策略任务由 `mdp/single_task.py`、`mdp/single_rewards.py` 和 `src/config/go2/go2_single_supermimic_config.py` 定义，使用45维盲观测及状态／几何解析奖励、硬失败与对应软概率；不启用下述旧CTS的22项HIM奖励预测头。其参考训练循环和MoE来源见 [单策略来源说明](../../../../agents/supermimic/single/NOTICE.md)。地形、命令课程、资产与仿真接口继续使用本表对应来源；下文奖励和CTS说明保留为历史兼容路径的依据。

本目录按本项目Go2资产与通用SuperMimic接口实现，RI奖励公式依用户要求迁入并适配Tensor转移和本体绑定，配置集中在`src/config/go2/go2_supermimic_config.py`。不导入其他具体任务或外部项目目录，也不修改上游依赖。来源核查日期为2026-10-26，Asia/Shanghai。

| 来源 | 固定版本与实际位置 | 采用内容与差异 |
| --- | --- | --- |
| 指定地形项目 | 本地`/home/sunteng/projects/legged_wbc_mjlab`，HEAD `e6209ec3d910b57673d5cbba1d3ca798a7fa3113`，相关文件无未提交差异；`src/tasks/locomotion/ri_4438_him_lpacrl/mdp/terrains.py`的`_base/ConnectedColumnCfg/TerrainMap`，`src/config/ri_4438/ri_4438_him_lpacrl_config.py:terrain` | 仅迁入地形家族、固定几何等级和连通布局：11家族52档、8×8m地块、7×13图、外扩16m平地；未迁入LP任务采样、算法、奖励或控制。独立SHA256语义地址种子保持可重复，但rough随机样本不与原项目逐点相同 |
| Go2课程与奖励 | 本地`/home/sunteng/projects/go2_rl_gym`，HEAD `30e74dc507bec7a642a8c98be26081f2c6f0822d`，相关文件无未提交差异；`legged_gym/envs/base/legged_robot.py:_update_terrain_curriculum/_resample_commands/_reward_tracking_lin_vel/_reward_tracking_ang_vel/_reward_action_rate/_reward_dof_pos_limits`；`envs/go2/go2_config.py:commands/rewards` | 最远移动半块长晋级、命令不足降级、最高级随机回落；真实步预算的速度阶段、零命令概率和地形限速。命令在回合重置冻结上下界，累计命令按当前航向和真实dt积分并考虑早摔剩余预算，均为本项目适配；不声称逐式复现其PhysX统计 |
| RI HIM奖励 | 同上RI仓库固定提交，`src/tasks/locomotion/ri_4438_him/mdp/rewards.py:track_linear_velocity/track_angular_velocity/lin_vel_z/body_orientation_l2/body_angular_velocity_penalty/stand_still` | 依本次用户要求迁入环境实际启用的全部22项、权重、姿态标准差／地形容差和步态门控；实际参数来源还包括`ri_4438_him_env_cfg.py:401–539`和`config/env_cfgs.py`。12项解析，10项真实传感器率加8项计时作为辅助监督；想象使用学习头预测，不能将模型近似视为硬接触公式的精确可微实现。Go2保留自身资产、PD和限矩，启用本任务自碰，未启用的RI `feet_clearance`不加入训练。原始机身高度／正弦净空奖励已移除 |
| 仿真显存预算 | 同上RI仓库固定提交，`src/tasks/locomotion/ri_4438_him/config/env_cfgs.py:ri_4438_rough_env_cfg`第41～48行；本机MjLab1.6.0／MuJoCo Warp3.11.0的`sim.py:_finish_init`、`io.py:put_data/_default_nconmax`与`collision_convex.py:convex_narrowphase` | 本任务显式采用接触48、约束300、CCD50、接触匹配64、nxn与CG，保留4096环境。RI与Go2的尺寸、碰撞过滤和地图不同，容量须独立验证，不将RI实验当作Go2验证。CG与CCD上限改变有限迭代物理结果，配置进入任务指纹；不迁入RI任务、MDP或模型 |
| 原始SuperMimic及CTS-MoE | [算法层来源说明](../../../../agents/supermimic/NOTICE.md)，论文§3公式1～3和算法A.1 | 复用世界模型、闭环想象和四组优化器；本任务的地形、课程、FK净空与上下文协议是迁移实现，原论文不证明这些组合的效果 |
| MjLab与MuJoCo | 实际安装MjLab1.6.0的`manager_based_rl_env.py:step/reset`、`terrains/terrain_generator.py`及MuJoCo3.11.0；[官方课程文档](https://mujocolab.github.io/mjlab/v1.6.0/source/curriculum.html)、[碰撞说明](https://mujoco.readthedocs.io/en/stable/computation/) | raw环境不自动reset，adapter在post-forward后计算业务奖励并复制终态；编译地形提取box／hfield静态表面并按MuJoCo三角插值查询。台阶边界保留不光滑性，不把静态表面查询当成可微物理仿真器 |

本目录没有复制外部训练器、机器人模型或检查点；除指定地形参数外，本次已迁入RI奖励公式并按本地接口适配，来源版本与位置见上表。外部仓库各自原有许可不自动授予本目录其他代码。本轮仅复用项目已有Go2资产，没有新增模型或网格文件；后续实际引入外部源文件或资产时须另行核对并保留原归属。

真实Go2 CPU短跑只能验证数据与优化链路可执行。模型误差、动作导数在真实仿真中的改善方向、学生行走收敛、课程难度效果和GPU成本须另做留出数据与物理实验。使用与验证范围见[任务说明](../../../../docs/go2_supermimic.md)，设计决策统一维护在[迁移设计§13](../../../../docs/design/go2_blind_supermimic.md#13-go2任务指定地形与课程实施契约)。
