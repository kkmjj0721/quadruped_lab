<!-- Copyright (c) 2026-2027 zh -->

# 深度 SuperMimic 来源与适配

本目录从本项目 Go2 `agents/supermimic` 的采集、物理模型、优化器、完整BPTT、编译与CUDA Graph实现独立派生，由本项目维护。Go1任务的奖励、地形、FK和相机实现位于 `src/tasks/locomotion/go1_depth/`，不从其他task导入。

训练机制参考 [SuperMimic](https://github.com/HybridRobotics/supermimic) 和 [Zhengtianjin/go2-supermimic@d9eee9b3](https://github.com/Zhengtianjin/go2-supermimic/tree/d9eee9b3f248e5300e636c433255db2c1c3e086a)。保留本项目Go2当前配方：五帧本体、物理八步监督、完整32步BPTT、软存活与末端价值、固定lambda-return、2/3交替采集、Muon＋Adam、MLP BF16、固定地址CUDA Graph及指标集中回读。采集归一化排除done行，模型监督使用原生final状态，终止当步有效而后续无效。

物理残差积分、树状FK和尺度下限来自 [BlinkMimic@d264658](https://github.com/fuyichen0725/blinkmimic/tree/d264658d6c9ce61fb52f9f0f8985a369f3ccc409)，许可原文见[Go1本地MDP](../../src/tasks/locomotion/go1_depth/mdp/LICENSE_blinkmimic)。本轮适配Go1 MJCF、固定相机体与四足site顺序。状态、积分、解析奖励和返回递推均为FP32。

感知训练参考 [InstinctMJ@6a483e6](https://github.com/project-instinct/InstinctMJ/tree/6a483e6b62cb8d8fb7cd302ffdf7988597bc9c58) 的SuperMimic视觉训练器：编码器从重建获得梯度，视觉预测目标停止梯度。采用本项目设计的四帧空间token、四足XY查询与单路 `323→512→256→128→12` Actor，不移植MoE、GRU、速度监督头或教师蒸馏。编码器旧token不进入长期回放，种子和采集缓存按当前CNN重编码固定四帧。

[TGI@a64be1f](https://github.com/leggedrobotics/terrain-grounded-imagination/tree/a64be1f185110d21c680ca1baf66e436772801bc) 只提供地形条件查询的参考方向。本任务保留完整32步BPTT，没有引入其事件截断或LR混合目标。严格未来触地标签按原生接触onset建立，未发生事件与回合尾部显式无效；read标签通过历史相机位姿、K与clean optical-z验证可见性，σ=0.5 token、0.08m＋1%深度遮挡容差为本轮工程初值。外部项目与本任务的机器人、框架和监督目标不同，外部成绩不能作为本项目控制效果证据。

`optim.py`派生自 [PartyMimic@69c53621](https://github.com/Roboparty/PartyMimic/blob/69c53621e6394f3c5dd7a923d0a93dcfcd1fb8eb/src/partymimic/optim.py#L64)。策略所有Linear权重用Muon，其余参数用Adam；物理模型仅中间Linear权重用Muon；Critic与感知模块各用独立Adam。感知Adam的参数仅包含CNN、Decoder、视觉prior和FPM；read损失只更新查询与Actor优化器，策略阶段冻结感知与物理权重并保留输入导数。

图捕获前完成编译与预热；原始深度CPU存储、标签准备与编码放在图外，固定形状物理采样、物理/策略/价值更新保持Go2图路径。图内视觉prior每控制步全批计算、仅相机刷新时写入token，不能将该实现描述成只计算10Hz。输入/输出精度、检查点格式与Go2相互独立：`supermimic_depth_query_v2` 使用新的相机协议，不加载旧版检查点。

参考来源、固定提交、差异与验证范围详见本地工作设计；正式用法见[Go1深度训练](../../docs/go1_depth_supermimic.md)。继续训练恢复网络、优化器和累计计数及任务课程；与Go2已有语义一致，真实回放、仿真、RNG与旧预热状态重新开始。

以下保留所派生代码的MIT版权与许可原文：

```text
MIT License

Copyright (c) 2026 PartyMimic authors
Copyright (c) 2026 Fuyi Chen (BlinkMimic, from which this software is derived)
Copyright (c) 2026 Huy-Quang-Dao (InstantTrack, from which BlinkMimic is derived)

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
