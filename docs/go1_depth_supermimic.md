<!-- Copyright (c) 2026-2027 zh -->

# Go1 深度 SuperMimic

从项目根目录启动单设备训练，默认4096个环境：

```bash
.venv/bin/python -u scripts/train.py Unitree-Go1-Depth-SuperMimic-Rough \
  --device cuda:0 --num-envs 4096 --experiment-name go1_depth_query
```

平地任务为 `Unitree-Go1-Depth-SuperMimic-Flat`。参数集中在 `src/config/go1_depth/go1_depth_config.py`，命令行覆盖见任务后 `--help`。终端按Go2格式输出英文训练摘要，默认每10轮汇总指标；TensorBoard事件与检查点位于 `logs/supermimic/go1_depth_query/<运行标识>/`。感知损失位于 `perception/`，查询损失为 `perception/read_loss`。远程日志查看遵循工作区TensorBoard指南。

粗糙任务默认地图有 6 档难度和 9 类地形。垫脚石位于最后一列，机器人从中央平台出生。课程逐档缩小石面，增大柱间间隙与石面高度变化。单个地块为 8×8 米，核心区外四周各有 20 米平地。

入口默认将PyTorch CPU张量算子线程数设为1，避免CPU帧池分块索引的线程开销；可用 `--cpu-threads 2` 显式覆盖。启动终端打印实际 `intraop`/`interop`，`params/run.yaml`记录实际值与后续子进程线程环境。当前已初始化的OpenBLAS线程池不由该环境记录保证改变；`interop`保持原值。线程覆盖属于进程执行参数，不改变任务hash或检查点语义。

策略输入为五帧47维本体历史和四个实际相机刷新帧；相机64×64、10Hz，控制50Hz。CNN产生四帧共64个16维token，FPM预测四足下一次触地的航向系XY，查询读取视觉后由单路 `323→512→256→128→12` MLP输出动作。没有MoE或GRU。部署网络398020个参数。

物理模型为三层1024宽MLP，八控制步监督；视觉prior八次相机刷新监督。策略保留完整32控制步BPTT、软存活及末端价值。训练默认沿用Go2的Muon＋Adam、MLP BF16、Inductor编译、CUDA Graph、微批与低频指标回读；想象执行批量2048、价值微批16384保持总想象批2048及优化器步数。CNN和查询保持FP32，CUDA推理按256图像固定微批编码，末批补齐后裁掉。像素与标签准备、感知辅助更新位于图外。

回放保存真实转移与终止前的原生final状态，终止当步可监督，之后屏蔽。深度原始新帧以FP16去重存CPU，GPU保存结构化状态、本体帧、触地元数据及图像引用；回放种子由当前CNN重新编码四帧，不保存过期token或整回合前缀。read查询辅助损失按整个想象种子的有效足点数归一化，改变执行微批不会改变该项的权重。CPU像素占用随实际刷新增长，4096环境、1000控制步在正常刷新下约15.7GiB；需要足够主机内存。训练吞吐和峰值显存以实际验收产物为准。

继续训练保留同目录的 `task_config.json`：

```bash
.venv/bin/python -u scripts/train.py Unitree-Go1-Depth-SuperMimic-Rough \
  --device cuda:0 --num-envs 4096 --resume '<运行目录>/model_1000.pt' \
  --continue-run True --max-iterations 20000
```

`max-iterations`表示目标总轮数。默认热启动只加载参数和匹配的优化器，轮数从零开始；`continue-run True`恢复累计计数与课程，重新建立仿真和回放。`--reset-critic True`从恢复轮次起冻结Actor以预热新Critic。Go2或历史RSSM检查点不能直接用于此网络。新版相机改变了策略输入分布，不支持加载旧 5 m 检查点（包括热启动）；请使用新版默认配置重新训练。

从新增垫脚石前的粗糙地图检查点迁移时，使用热启动。`continue-run True` 要求地图几何匹配。

每次保存同步生成 `policy.pt`、`policy.onnx` 和 `policy.json`。导出只包含CNN、FPM、查询和Actor；显式输入 `history[B,5,47]`、`depth_frames[B,4,1,64,64]`，输出未缩放动作 `[B,12]`。历史按从旧到新排列。重置时重复首个本体帧与深度帧，相机每5控制步追加新帧。导出前向每次重编码输入的四帧，调用端可另行维护新帧编码缓存；导出本身不实现该缓存。动作按配置的关节缩放映射到PD目标。

`policy.json` 与 ONNX 元数据的 `format` 为 `supermimic_depth_query_policy_v2`。调用端须自行把米制光轴 Z 编码成策略输入：有限且 `Z > 0` 时用 `clip(Z, 0, 2) / 2 - 0.5`，非有限值、`Z ≤ 0` 和无回波填 `+0.5`；2 m 及更远同样为 `+0.5`。训练随机掉点也填 `+0.5`，部署不额外添加该噪声。旧版 `v1` 是有限且 `0 < Z ≤ 5 m` 时 `Z / 5`，其余置零；虽张量形状不变，旧部署产物仍须沿用原预处理，不能套用 `v2`。

训练相机使用 `64×64` 方图，名义水平和垂直视场均为 `58°`，图像整数像素中心从 0 编号，主点为 `(32, 32)`。深度射线只看第 `0` 组地形，不看机器人。每个环境创建时在机身系对安装位置三轴独立均匀采样 `±1 cm`，下俯角均匀采样 `15–25°`、偏航和横滚各 `±2°`，回合重置不重采样。

```bash
.venv/bin/python scripts/play.py Unitree-Go1-Depth-SuperMimic-Rough \
  --checkpoint-file '<运行目录>/model_1000.pt' --num-envs 1 --viewer viser
```

Viser 右侧顶部的「Go1 深度相机」和 `--viewer native` 窗口左下角显示当前环境的最新策略深度帧，灰度固定为 0～2 m，近黑远白，无回波或掉点为白。3D 场景用红、青、黄、紫球分别显示左前、右前、左后、右后的下一次预测触地位置。策略输出航向坐标系中的 XY，标记球底与对应世界 XY 的地形表面相切。落足点随查看器的调试可视化开关显示，`--agent zero` 和 `--agent random` 可查看深度图。

独立预览可用 `.venv/bin/python tools/preview_go1_depth.py --headless` 查看名义安装位姿下的地形深度，不随机外参、不加噪。离屏渲染关闭多重采样，按像素中心取深度。黄色视锥按资产内参绘制到 2 m；灰度默认也显示 0～2 m，`--depth-max` 只改变灰度范围。`depth_m.npy` 保留离屏渲染得到的原始米制深度（背景保留渲染器远平面值），不是模型输入。

训练验收须显式启动，普通单元测试不启动完整训练：

```bash
OMP_NUM_THREADS=1 .venv/bin/python -u tests/test_training/go1_query_training.py \
  --profile full --num-envs 4096 --output logs/go1_query_validation
```

输出目录必须不存在。该命令先完成编译和各图捕获，再进行至少600秒实际训练，最后检查有限更新、保存与部署导出，并独立恢复继续更新所有优化器及重捕获图。`--profile local`缩减为32环境与样本量、保留网络宽度和时域；`--smoke`仅用于短流程排错，不能作为训练验收或越障能力证明。

实现与来源差异见 [算法来源](../agents/depth_supermimic/NOTICE.md)。当前组合需要通过独立地形成功率、速度误差和摔倒率评估控制效果；数值与工程验收不表示视觉越障已经收敛。
