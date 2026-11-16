<!-- Copyright (c) 2026-2027 zh -->

# 纯深度 SuperMimic 的来源与维护边界

本目录独立维护 CNN–GRU 视觉策略、去噪重建、视觉世界模型和 SuperMimic 学习流程。运行时不导入原 `agents/supermimic/`；状态、统计归一化、MLP、物理标签统计、单帧回放、优化器辅助、指标和日志均位于本目录。原算法与已有训练产物保留原状。

必要通用能力移自本项目提交 `a50db34197c1bc4a804df9311d0b3c0f6fa865e9` 的 SuperMimic 实现，未迁入混合专家、速度估计监督或教师驱动。保留原有逐通道总体方差统计、残差与增量误差尺度分离、固定 λ-return 和微批归一化语义；任务的运动学、奖励、失败和接触尺度由 Go1 任务目录提供，不在通用算法中定义。按用户后续要求，Muon＋Adam 和摔倒回滚现已在本目录独立适配。

- SuperMimic 训练目标依据论文 *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model* §3.1–3.3、式(2)(3)，对应 [本地论文](../../paper/SuperMimic_2610.28480v1.pdf)。采用八步自回归模型监督、三十二步完整 BPTT、冻结模型与价值参数后的输入导数、生存加权收益、末端价值和固定 λ-return。
- 低频深度循环编码参考 [Extreme Parkour 固定提交的深度模块](https://github.com/chengxuxin/extreme-parkour/blob/d2ffe27ba59a3229fad22a9fc94c38010bb1f519/rsl_rl/rsl_rl/modules/depth_backbone.py)。这里只采用时序深度组织方式；当前 CNN 通道16/32/64、GRU128、线性地形表示32及动作 MLP 是本项目设计，没有迁入其方向预测或教师蒸馏。
- 去噪输入与干净深度监督参考 [MGDP 固定版本](https://github.com/arclab-hku/MGDP/tree/f43139d1d97849819a0dce609dc0b59faf577b3f)。本项目 Decoder 只能读取 Actor 使用的同一地形表示，无图像跳接；并非原项目网络的完整复现。
- 层间归一化参考 [SuperMimic@abc124c `models.py:28–40`](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/models.py#L28) 的策略/动力学隐藏层LayerNorm，以及 [MGDP@f43139d `net_work.py:6–23,64–69`](https://github.com/arclab-hku/MGDP/blob/f43139d1d97849819a0dce609dc0b59faf577b3f/legged_gym/legged_gym/envs/random_dog/utils/net_work.py#L6) 的卷积GroupNorm与投影隐藏层LayerNorm。本项目在CNN/Decoder隐藏卷积使用8组GN，在Actor/Critic/WM和编码投影隐藏层使用LN，保留原ELU/SiLU、线性输出和RunningNorm。GRU真实/预测特征共用输入LN，但不改GRU门控或归一化hidden；这属于本项目适配，不等同于论文的LayerNorm-GRU。新参数由Adam更新，旧无归一化检查点由现有完整参数键检查拒绝，需重新训练；已有独立导出策略文件保留。依据为 *Group Normalization* §3.1–3.2和 *Layer Normalization* §3–3.1，未据此宣称Go1训练效果改善。
- MLP 局部 BF16 边界参考 [BlinkMimic `run_mlp`](https://github.com/fuyichen0725/blinkmimic/blob/d264658d6c9ce61fb52f9f0f8985a369f3ccc409/src/blinkmimic/networks.py#L64)：MLP 矩阵计算可用 BF16，输出恢复 FP32，视觉 CNN、GRU、状态积分、损失和返回维持 FP32。编译以完整八步监督、想象单步及固定返回为边界，预热单独计时。
- 显式 CUDA Graph 的静态输入、三次真实预热、可捕获 Adam 和恢复后图失效机制源自本项目上述版本的图执行模块，现独立维护在本目录 `cuda_graph.py`。图外按当前编码器重建完整旧前缀与准备监督；图内重算最新64×64图像的CNN/GRU并执行完整32步策略/重建更新，以及完整8步模型和价值拟合。固定地址、侧流预热和禁止主机同步的约束见 [PyTorch2.14 CUDA Graph说明](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs)。学习率使用设备Tensor，加载优化器后重新捕获，不增加预热更新预算。
- `optim.py`派生自本项目上述版本的Muon实现，原始来源为[PartyMimic@69c53621的optim.py](https://github.com/Roboparty/PartyMimic/blob/69c53621e6394f3c5dd7a923d0a93dcfcd1fb8eb/src/partymimic/optim.py#L64)。Actor动作MLP全部Linear权重使用Muon；视觉CNN、GRU、投影和Decoder使用Adam；WM仅中间Linear权重使用Muon，Critic保持Adam。Muon采用`match_rms_adamw`及权重衰减0.1，感知和重建参数仍由唯一Actor联合优化器持有。来源面向人形MLP，本项目增加视觉循环分支，不将CNN或GRU矩阵直接纳入Muon。
- 摔倒回滚沿用本项目Go2的失败数／重置控制步统计、连续两次超阈值与成功保存后安全快照规则。Go1额外同时恢复Decoder，重建Actor联合优化器和Critic并使对应CUDA Graph失效，保留WM；学习率减半并重新爬升。该机制和默认4096环境配方不构成视觉控制效果或完整规模资源验收。

模型接口、训练闭环和梯度检查只支持对应验证范围，不能据此宣称已收敛、视觉收益成立或达到参考项目的速度。详细设计与验收记录由本项目工作文档维护。

本项目复制部分遵循根目录 [MIT 许可](../../LICENSE)。涉及 BlinkMimic 派生能力时保留下列版权与许可原文；Go1 任务中直接派生的结构化动力学、旋转和 FK 许可由任务目录另行保留。

```text
MIT License

Copyright (c) 2026 PartyMimic authors
Copyright (c) 2026 Fuyi Chen
Copyright (c) 2026 Huy-Quang-Dao (InstantTrack, from which this software is derived)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
