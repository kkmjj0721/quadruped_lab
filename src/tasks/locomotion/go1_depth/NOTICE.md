<!-- Copyright (c) 2026-2027 zh -->

# Go1 纯深度 SuperMimic 任务来源

本目录是独立维护的 Go1 任务实现。配置集中在 [go1_depth_config.py](../../../config/go1_depth/go1_depth_config.py)，算法位于 [agents/depth_supermimic](../../../../agents/depth_supermimic/)。运行时不继承或导入旧 Go2 任务、配置和算法；以下来源仅用于说明本地副本的出处，不构成运行依赖。

| 来源 | 固定位置与采用范围 | 本地适配 |
| --- | --- | --- |
| 本项目已有任务 | 提交 `a50db34197c1bc4a804df9311d0b3c0f6fa865e9` 的 `src/tasks/locomotion/go2_supermimic/mdp/`、`go_supermimic_env_cfg.py`、`rl/transition_adapter.py`、`rl/world_model.py`、`config/loading.py`、`rl/playback.py` | 将实际需要的状态、解析奖励、命令、rough 地形、课程、物理模型尺度、配置读取和播放适配复制到当前目录独立维护。保留既定 47 维本体、262 维特权、239 维物理模型输入和 80 维残差协议；移除旧物理布局与侧倾奖励兼容分支，不复制旧 MoE、驱动策略或训练入口 |
| BlinkMimic 及其上游 InstantTrack | BlinkMimic 提交 `d264658` 的 `src/blinkmimic/rotations.py`、`kinematics.py`、`networks.py:113–209`；已有来源说明随本项目上述版本保留 | `mdp/reference_rotations.py`、`mdp/structured_dynamics.py` 和 `mdp/kinematics.py` 保留来源算法；几何常量读取实际 Go1 MJCF。原作者版权与 MIT 许可完整保留在 [LICENSE_blinkmimic](mdp/LICENSE_blinkmimic) |
| 本项目 Go1 资产及其来源 | [Go1 资产](../../../assets/robots/go1_depth/go1_depth_constants.py)；本地 `wmp_go1@65cf1afd13e4792bce7b9def6f44372ecbb3cdf0` 的 `legged_gym/envs/a1/a1_amp_config.py:152–163,187` | 保留 Go1 默认姿态、关节映射、40/1 的 PD 增益、33.5 N·m 限幅、相机安装及四子步控制；动作缩放按用户要求对齐 Go2 SuperMimic，具体数值在集中配置维护。未采用 Go2 连杆几何或执行器参数 |
| MjLab 与 MuJoCo-Warp | 本次实际环境为 `mjlab 1.6.0`、`mujoco-warp 3.11.0`；上游 `mujoco_warp/_src/render.py` 的 `_make_cast_ray` 与光轴深度定义 | `depth_camera.py` 通过依赖提供的求交函数，仅对相机到期环境发射像素射线，支持异步重置。该上游文件版权为 `Copyright 2026 The Newton Developers`，适用 Apache License 2.0；实现未复制整套上游渲染器，升级依赖后需重跑显式相机一致性检查 |

视觉 CNN、GRU、去噪重建及特征动力学是本次设计组合，不宣称完整复现 Extreme Parkour、MGDP 或 SuperMimic 的原始实验。任务公式沿用既定参考，Go1 机器人、视觉输入、相机时序与资源预算属于明确适配；链路和数值测试不代表训练已收敛、视觉收益已证实或真机控制已通过。
