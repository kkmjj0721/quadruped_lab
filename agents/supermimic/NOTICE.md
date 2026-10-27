<!-- Copyright (c) 2026-2027 zh -->

# SuperMimic 实现来源

当前默认单一盲 MoE 算法位于 `single/`，来源、固定版本和迁移差异见 [单策略来源说明](single/NOTICE.md)。下文保留历史 CTS 实现及共享组件的依据，不将旧算法限制套用于新默认路径。

本目录是依据论文公式及本项目设计编写的通用PyTorch实现，不是作者官方训练代码，也没有移植外部仓库的训练脚本、机器人资产或检查点。各模块由本项目独立维护；不依赖MjLab、RSL-RL或Go2任务。通用组件测试与Go2物理短跑的实际范围见[设计与审查记录](../../docs/design/go2_blind_supermimic.md)；短跑不证明Go2的模型精度、梯度方向、步态收敛或部署效果。

## 实际参考范围

核查日期：2026-10-26，Asia/Shanghai。

| 来源 | 版本与实际阅读位置 | 采用内容及适用差异 |
| --- | --- | --- |
| *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model* | [arXiv:2610.28480v1](https://arxiv.org/abs/2610.28480v1)；[本地英文PDF](../../paper/SuperMimic_2610.28480v1.pdf)，§3.1～3.3、公式1～3，PDF第4～6页；附录B／D、Algorithm A.1及表A.5，PDF第14～16页 | 持续collect-fit-imagine、多步自回归、模型外观测噪声、软存活、可微末值与固定λ-return。论文为G1动作跟踪；本项目采用CTS-MoE且任务由外部协议定义，不能继承论文成绩 |
| CTS原文 | [arXiv:2405.10830v1](https://arxiv.org/html/2405.10830v1)，§II-B/C和Algorithm 1 | 并发教师学生与共享动作网络的来源；原方法使用PPO且学生为MLP，本文采用模型BPTT与MoE，不声称原文验证过此组合 |
| `wty-yy/go2_rl_gym` | 本地`/home/sunteng/projects/go2_rl_gym`，提交`30e74dc507bec7a642a8c98be26081f2c6f0822d`；`rsl_rl/rsl_rl/modules/{utils,actor_critic_moe_cts}.py`的`Experts/MoE/StudentMoEEncoder`、`act/evaluate`，`algorithms/moe_cts.py:update` | 八专家共享backbone、分组投影、独立gate、混合后L2及学生MSE＋平均负载均衡。其Isaac Gym／PPO、45×5历史和critic detach与本项目不同；本项目为可微历史路径重新实现 |
| `CuboiLeo/SuperMimic-Reproduced` | [world_model.py](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/main/src/supermimic/world_model/world_model.py)：`step/_loss_step/window_loss`；[state_spec.py](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/main/src/supermimic/world_model/state_spec.py)：`canonical/integrate/error_vector`；[objective.py](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/main/src/supermimic/imagination/objective.py)：`policy_objective/gae_targets` | 实际阅读main网页快照，未取得固定提交，因此不能将该链接视为不可变版本。用于交叉核对物理残差／预测状态失败监督／末值导数；其G1状态71、动作29及逐维损失不直接用于本项目，积分和误差交由任务实现 |
| `zita-ch/supermimic_vibed` | 提交`a1508d42c9d1699a52e8233ab16d08c488441ab9`；[train.py](https://github.com/zita-ch/supermimic_vibed/blob/a1508d42c9d1699a52e8233ab16d08c488441ab9/supermimic/train.py)：`_wm_rollout/_imag_step/policy_update/collect`；`networks.py:WorldModel/Policy` | 对照多步历史、冻结模型及critic输入导数。该版本G1使用十帧与目标编码、解析soft termination，critic规范化更新后才计算目标；本项目学习失败头，直接使用同次rollout快照固定目标，不复制其训练调度或动作裁剪 |

官方[项目页](https://supermimic.github.io/)在核查时未提供训练源码入口。以上两份SuperMimic训练实现均为社区复现，外部作者报告的效果没有在本项目复现。详细任务设计与待验证假设统一维护在[Go2迁移设计](../../docs/design/go2_blind_supermimic.md)，不据此宣称本目录能够直接控制Go2。

## 训练日志参考

训练日志在2026-10-26另外精读下列固定版本，只参考统计口径与显示组织，`metrics.py`和`logging.py`独立编写：

- [CuboiLeo训练循环](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/d9be385b3eeb13a638fb33fb200d3bfd9be9839f/src/supermimic/training/loop.py)，`d9be385b3eeb13a638fb33fb200d3bfd9be9839f`，`Trainer.run`：真实完成回合／想象目标分开及sim／WM／policy计时。
- [zita训练循环](https://github.com/zita-ch/supermimic_vibed/blob/ee0608f917d08a591356b9bdc7d94bd949157341/supermimic/train.py)，`ee0608f917d08a591356b9bdc7d94bd949157341`，`collect`及主循环：真实奖励、回合、训练损失和迭代速度。与上表较早算法核查版本分开记录。
- [Go2 SuperMimic训练循环](https://github.com/Zhengtianjin/go2-supermimic/blob/054a13c13285a1c94b2e69dc8ac7331f9a8c71cf/go2_supermimic/supermimic/train.py)，`054a13c13285a1c94b2e69dc8ac7331f9a8c71cf`，`Trainer.collect/improve_policy/run`：真实任务指标与想象指标分开；不迁入其追逐任务规则或保护算法。
- [官方RSL-RL日志](https://github.com/leggedrobotics/rsl_rl/blob/a87ef38fde4e2a554bec1565e14b868a64de5851/rsl_rl/utils/logger.py)，`a87ef38fde4e2a554bec1565e14b868a64de5851`，`Logger.process_env_step/log`；[Legged Gym奖励日志](https://github.com/leggedrobotics/legged_gym/blob/8fa29acc6fd1910c3d9659eef6310bdd301cde0a/legged_gym/utils/logger.py)，`8fa29acc6fd1910c3d9659eef6310bdd301cde0a`，`Logger.log_rewards/print_rewards`：完成回合窗口、真实吞吐、分区显示与奖励单位。不复用其代码或PPO专属指标。

来源版本、实际文件位置、适用差异和本项目指标契约统一见[设计§17](../../docs/design/go2_blind_supermimic.md#17-面向训练诊断的日志设计)。

## 许可边界

- 新增文件保留`Copyright (c) 2026-2027 zh`；未为本项目新增代码单独授予开源许可证，也不将参考项目的许可自动转授给整个目录。
- `CuboiLeo/SuperMimic-Reproduced`公开许可为MIT，Copyright (c) 2026 Yuxi Liu；`zita-ch/supermimic_vibed`为MIT，Copyright (c) 2026 ETH Zurich, Chong Zhang。已阅读其LICENSE；本目录未复制其中的代码段。
- 本地`go2_rl_gym/LICENSE`含MIT（Copyright (c) 2025 Wu Tianyang）与其上游BSD-3-Clause声明。本目录按数学结构独立实现，没有携入其资产、部署文件或代码片段。
- 论文、数据、模型及第三方依赖分别遵守原权利人的条款。若后续实际引入源代码或资产，应在引入前核对、保留对应许可与归属。
