<!-- Copyright (c) 2026-2027 zh -->

# 感知模块来源与边界

本目录依据 [MGDP 官方固定提交 f43139d](https://github.com/arclab-hku/MGDP/tree/f43139d1d97849819a0dce609dc0b59faf577b3f) 的公开网络结构和数学定义独立实现，由本项目维护，不依赖该参考仓库运行。核对了该提交的文件树与根 README；未找到覆盖 `legged_gym/legged_gym/envs/random_dog/utils/net_work.py` 的明确源码许可，因此没有直接移植其源码，也不把仓库内 Isaac Gym 或第三方资产的许可套用于此文件。

| 依据位置 | 本目录对应 |
| --- | --- |
| `random_dog_config_stage1.py:Encoder` 与 `random_dog.py` 构造感知网络段 | 深度通道 32、64、96，token 64，池化 4×4；高度通道 16、32、64，token 32，池化 2×2 |
| `utils/net_work.py:DoubleConv,ImageEncoder,MapEncoder` | 双 3×3 卷积、GroupNorm、SiLU，两次最大池化、bottleneck、全局池化及 LayerNorm 投影；跳接作为调用返回值，无隐式缓存 |
| `utils/net_work.py:ImageDecoder,MapDecoder` | token 投影、三级跳接融合、双线性插值 `align_corners=True`；深度 Sigmoid、高度自重建 Tanh |
| `random_dog.py:compute_world_model_outputs` | 双帧深度 MSE、高度自重建 MSE、单向高度→深度 InfoNCE，温度 0.1；共有前 32 维，batch 少于 2 时对比项为零 |
| `random_dog_config_stage1.py:camera` | 基准损失为 `1.2 L_depth + L_height + 0.3 L_InfoNCE`，边缘和潜变量 MSE 权重为零 |

`DepthPerception` 的 GRUCell（hidden=128）及米制地图头 `128→256→128→187` 为本项目新增，隐藏层采用 SiLU，输出层线性；与源码中的控制器记忆网络不是同一结构。输入为当前双帧深度 token 与本体历史，地图为 17×11。该扩展不会使用 GT 高度编码器或其跳接。

调用方提供按固定量程归一化的深度、归一化高度自重建标签、独立的米制地图标签。传感器噪声、相机更新相位、帧保持、坐标变换、接触事件和回合重置由任务与训练装配侧负责。`step` 显式传入／返回隐藏状态，模块不保存历史；`inference_module()` 返回只包含深度编码器、GRU 和米制地图头的独立副本。

这里实现的是感知结构、监督损失与本项目地图扩展。数值或梯度检查不代表 MGDP 的控制器、训练过程、机器人性能或 Go1 地形能力已复现。
