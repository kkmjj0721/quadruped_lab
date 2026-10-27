# SuperMimic 相关开源项目调研索引

方法讲解见 [SuperMimic 阅读笔记：训练如何工作，PPO 在哪里被替换](../paper/SuperMimic阅读笔记.md)，包含通俗示例、论文公式和当前 Go2 训练调用链；本页维护相关项目与资源索引。

> 调研时间：2026-10-26（Asia/Shanghai）。检索范围：GitHub 仓库检索 API（`supermimic` / `super mimic` / `supermini` / `super+mimic`、名称＋简介＋README 覆盖，共 94 条命中）、GitHub 仓库元数据 API、Hugging Face 模型／数据集 API、arXiv 与官方项目页。本文件只做索引与事实记录，**不代表本项目已复现或验证任何外部实现**，表中成绩均为各仓库作者自报。未检索到官方训练代码发布（官方项目页与论文均只给出论文、视频与浏览器 demo）。

补充核查：2026-10-27（Asia/Shanghai），定向补充 [Go2-SuperMimic 专项说明](#31-go2-supermimic)；表中 Star 保留原调研快照。

## 1. 原始工作（论文与官方资源）

| 项目 | 说明 | 链接 |
| --- | --- | --- |
| 论文 | SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model；Qiayuan Liao, Yiyang Shao, Junfeng Long, Kevin Zakka, Roberto Horowitz, Pieter Abbeel, Koushil Sreenath；UC Berkeley；CoRL 2026 Oral | [arXiv:2610.28480](https://arxiv.org/abs/2610.28480)（本地：`paper/SuperMimic_2610.28480v1.pdf`，中文版 `paper/SuperMimic_2610.28480v1_zh.pdf`） |
| 官方项目页 | 方法示意、训练曲线、与 SONIC 的对比、真机视频；**未提供代码仓库链接** | <https://supermimic.github.io/> |
| 官方浏览器 demo | 直接在浏览器里试跑官方策略 | <https://supermimic.github.io/demo/> |
| 社区索引（英文） | `awesome-supermimic`，按“独立复现／方法扩展／框架与部署／灵感实验”四类维护 29 个条目，并附代码快照 commit | <https://github.com/DiaoOrg/awesome-supermimic> |
| 社区索引（中文） | `SuperMimicCarnival`，每 3 小时增量检索，附“完整训练实现／暂不收录”两级核查口径 | <https://github.com/QinnyuLan/SuperMimicCarnival> |

论文核心结论（论文摘要原文，非本项目验证）：用一份可微世界模型分摊多动作学习，世界模型与跟踪策略在同一循环中训练，对短时想象 rollout 做 BPTT 得到解析梯度；单卡 3 小时超过基线 128 卡 7 天的回报，GPU-hours 降低 >1000×；129,514 段基准上 3 小时失败 25 段、双卡 9.8 小时零失败；策略迁移到 Unitree G1 真机。

## 2. 独立复现（G1 全身动作跟踪主线）

| 仓库 | 底座 | 说明与作者自报 | Star |
| --- | --- | --- | --- |
| [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) | Isaac Lab 3 kit-less / MuJoCo Warp | 最早的独立复现之一，README 明确“1 小时 AI 辅助速写、未经评审”；40 条 LAFAN1，单卡 RTX PRO 6000 1.47 h，归一化回报 0.829、0/40 失败。多数后续仓库以它为上游 | 564 |
| [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) | mjlab / MuJoCo Warp | 从零实现数据管线、可微 FK、世界模型、想象训练与分布式；含 LAFAN1 与 BONES-SEED（129.9k 段）流程；RTX 5090 上 1.66 h 达 0.824、5/40 失败 | 236 |
| [Motphys/Uni-SuperMimic](https://github.com/Motphys/Uni-SuperMimic) | UniLab（mjwarp / MuJoCo / Motrix 三后端） | 工程化移植，含 40 条动作资产、确定性全量评测与视频导出；12k 迭代回报 0.827227、0/40 失败 | 144 |
| [lzyang2000/morphmimic](https://github.com/lzyang2000/morphmimic) | 自研（mjlab） | 形态条件化的世界模型／策略共享，G1 / X2 Ultra / T2 共用，支持五点稀疏目标；另有论文与在线 demo | 100 |
| [btx0424/rethinking-supermimic](https://github.com/btx0424/rethinking-supermimic) | mjlab | 研究目标条件化方式（拼接 vs FiLM），附与论文、`supermimic_vibed` 的实现对照表 | 23 |
| [Baojch/SuperMimic_Dyn](https://github.com/Baojch/SuperMimic_Dyn) | mjlab（基于 CuboiLeo） | 面向高动态与低姿态：前/后/侧空翻、侧手翻、爬行、跌倒起身 | 16 |
| [panhainan2020/SuperMimic-MuJoCo](https://github.com/panhainan2020/SuperMimic-MuJoCo) | 原生 MuJoCo / MJLab | 脱离 Isaac Lab，含 100STYLE 分布外评测（作者报 3,569/3,573 段完成、0.8045） | 13 |
| [Shimingyi/allmimic](https://github.com/Shimingyi/allmimic) | mjlab（基于 supermimic_vibe） | 残差上肢动作＋绝对下肢动作、世界模型损失加权、动作采样平衡；含 held-out 划分实验 | 10 |
| [Renforce-Dynamics/trackerlab-supermimic](https://github.com/Renforce-Dynamics/trackerlab-supermimic) | Isaac Lab / PhysX（TrackerLab） | 重定向→MotionLib→训练→全段评测→导出全链路，覆盖 G1/Go2/A3/T1/Bumi/Pi+ | 9 |
| [YanYuan98/Supermimic_mod](https://github.com/YanYuan98/Supermimic_mod) | mjlab | 完整训练评测，含与 RGMT 的效果对比 | 8 |
| [tianshuwu/supermimic-retargeted](https://github.com/tianshuwu/supermimic-retargeted) | mjlab（基于 CuboiLeo） | 多本体：G1 / A3 / X2 Ultra，SOMA 重定向、坐标修正、足接触检查 | 7 |
| [RaraAlu/Supermimic](https://github.com/RaraAlu/Supermimic) | mjlab（移植 zita-ch） | 支持 CSV / 原生 HDF5、多卡分片、数据指纹与全量评测；目标 129,514 段基准 | 4 |
| [humble02/supermimic_mjlab_vibed](https://github.com/humble02/supermimic_mjlab_vibed) | mjlab | `supermimic_vibed` 的后端迁移，含离线服务器流程与续训 | 4 |
| [kangzejian1896/supermimic-g1-reproduction](https://github.com/kangzejian1896/supermimic-g1-reproduction) | MuJoCo Warp（基于 CuboiLeo） | 同协议对照与 BONES-SEED 验证动作演示 | 4 |
| [Breezewrf/supermimic_mjlab](https://github.com/Breezewrf/supermimic_mjlab) | mjlab | 在线 WM／策略训练＋离线 WM 训练两条路径，含回放与 W&B | 2 |
| [fan-ziqi/supermimic-agy](https://github.com/fan-ziqi/supermimic-agy) | mjlab / PyTorch | 以“无人工干预的端到端论文复现能力”为实验目的，附全过程提示词与模型切换记录 | 2 |
| [samindaa/supermimic_mjlab](https://github.com/samindaa/supermimic_mjlab) | mjlab | 即 `supermimic_vibed` 作者从 Isaac Lab 迁到 mjlab 的版本（同作者 Chong Zhang） | 1 |
| [HighTorque-Dev/supermimic-colab](https://github.com/HighTorque-Dev/supermimic-colab) | Isaac Lab 3 / MuJoCo Warp | 面向 Colab／A100 的复现与 14k 迭代训练日志 | 1 |
| [KunYing-Lee/SuperMimic](https://github.com/KunYing-Lee/SuperMimic) | mjlab | 固定动作库评测与论文对齐审计 | 1 |
| [dhmbb2/supermimic_vibe](https://github.com/dhmbb2/supermimic_vibe) | mjlab | PPO-Sim 基线、视界×critic 消融、双卡训练；A6000 上 1.34 h 达 0.833 | 1 |
| [WXyuany/supermimic-m5](https://github.com/WXyuany/supermimic-m5) | mjlab（MPS 学习 + CPU 物理） | Apple M5 上从零训练，40 条 LAFAN1，附权重与视频 | 1 |
| [rzc156/supermimic-rtx5080](https://github.com/rzc156/supermimic-rtx5080) | mjlab | 单卡 RTX 5080 16 GB，3.3 h、0.874 回报 | 0 |
| [somidaa/supermimic-lafan1-reproduction](https://github.com/somidaa/supermimic-lafan1-reproduction) | mjlab | 单卡 TITAN RTX 全量 40 条 LAFAN1，独立复评 0.8849、1/40 失败 | 0 |
| [Go-Youngster/supermimic-reproduce](https://github.com/Go-Youngster/supermimic-reproduce) | 从零（mjlab） | 含 PPO-Sim / PPO-WM 基线与消融、A100 训练记录 | 0 |
| [LaoYang1994/SuperMimic-Rep](https://github.com/LaoYang1994/SuperMimic-Rep) | mjlab（基于 CuboiLeo） | 强调“同迭代数对齐论文协议”，BONES-SEED 全量评测、2× 训练提速 | 0 |
| [tianny0078/supermimic-repro](https://github.com/tianny0078/supermimic-repro) | mjlab（基于 CuboiLeo） | Slurm 入口与调研记录，README 声明尚无收敛结论 | 0 |
| [enffft/supermimic](https://github.com/enffft/supermimic) | mjlab（基于 CuboiLeo） | 增加 Q1 训练文档 | 0 |
| [sxh-kk/SuperMimic_Recurrence](https://github.com/sxh-kk/SuperMimic_Recurrence) | mjlab（基于 CuboiLeo） | RTX 4090／CUDA 12 适配，探索 Ollie／滑板路线 | 0 |
| [CesarLiu/smimic](https://github.com/CesarLiu/smimic) | UniLab | mjwarp／MuJoCo／Motrix 三后端，Hydra 配置与 Hugging Face 数据管线 | 0 |
| [yuntian137/supermimic_lab](https://github.com/yuntian137/supermimic_lab) | Isaac Lab 3（基于 zita-ch） | 独立安装、配置、续训、全量评测与参考虚影录像 | 0 |
| [Alchedata/supermimic-locomujoco](https://github.com/Alchedata/supermimic-locomujoco) | JAX / LocoMuJoCo | 移植到 Unitree H1，含残差专家、动作敏感度与 trust-aware 世界模型变体 | 1 |
| [PrimebotTech/SuperMimic](https://github.com/PrimebotTech/SuperMimic) | PyTorch / MuJoCo Warp | Q1 V2 独立训练栈（单/八卡、续训、筛选、录像） | 1 |
| [Mark054/SuperMimic_TienKung](https://github.com/Mark054/SuperMimic_TienKung) | mjlab | 天工 3.0 独立适配，附模型、数据与转换工具 | 0 |
| [joreb/supermimic-yai](https://github.com/joreb/supermimic-yai) | 自研（WIP） | 论文笔记＋进行中实现 | 0 |
| [yagami-light7/SuperMimic-codex](https://github.com/yagami-light7/SuperMimic-codex) | Isaac Lab 3 / Newton | 仅到 M0 环境冒烟，**无训练结果** | 0 |

## 3. 新本体、新任务与感知扩展

| 仓库 | 方向 | 说明 | Star |
| --- | --- | --- | --- |
| [HorizonRobotics/HoloMotion](https://github.com/HorizonRobotics/HoloMotion) | 框架集成 | 地平线 HoloMotion 内加入 SuperMimic 算法、配置、训练与评测（MuJoCo Warp 与 Isaac PhysX 双后端）；单卡 RTX 5090 30 分钟完成 32/40 条 | 736 |
| [project-instinct/InstinctMJ](https://github.com/project-instinct/InstinctMJ/tree/supermimic) | 深度视觉地形 | 世界模型同时预测机器人状态与深度编码潜变量，训练原深度策略（无 PPO）；跑酷、跨越障碍、爬台阶。**注意用 `supermimic` 分支** | 272 |
| [Zhengtianjin/go2-supermimic](https://github.com/Zhengtianjin/go2-supermimic) | 四足（与本项目最相关） | Go2 追逐＋闪避，平地／障碍／地形三档任务，LiDAR 或高程扫描输入；附预训练 actor 与评测录制工具，见[专项说明](#31-go2-supermimic) | 42 |
| [swaggyliu/FlatWorld](https://github.com/swaggyliu/FlatWorld) | 2D 教学复现 | 自研 2D 刚体/软体引擎上的 SuperMimic 完整闭环，十杆人形（8 驱动关节）＋程序化动作库 | 38 |
| [JackMin7/supermimic_whole_body_tracking](https://github.com/JackMin7/supermimic_whole_body_tracking) | 对照实验 | 同一静态站立任务上 MBRL（SuperMimic 思路）vs PPO（BeyondMimic），各约 1 小时 | 19 |
| [CumulusAlpha/Dex-Supermimic](https://github.com/CumulusAlpha/Dex-Supermimic) | 灵巧手 | 迁移到无界灵巧手 Wuji Hand 2 的手中立方体重定向，附预训练权重与 Viser 查看器 | 9 |
| [shuayyy/humanoid-amp-ppo](https://github.com/shuayyy/humanoid-amp-ppo) | 任务对照 | 在 G1 行走／下蹲／搬箱任务上对比 SuperMimic 与 PPO-Sim | 1 |
| [wenconggan/SuperMimic_e1](https://github.com/wenconggan/SuperMimic_e1) | 新本体 | 21 DoF DroidUp E1，训练后可在原生 MuJoCo 做 sim2sim | 4 |
| [OMEGA-i/SuperMimicKit](https://github.com/OMEGA-i/SuperMimicKit) | 多本体工具包 | G1/H2/A3，含重定向、ONNX 导出与 WebGPU 浏览器预览 | 3 |
| [mode1223345/SuperMimic-perceptive](https://github.com/mode1223345/SuperMimic-perceptive) | 高程图感知 | 冻结共享高程图自编码器供 actor/critic/世界模型共用，含台阶地形实验 | 4 |
| [NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real](https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real) | 真机部署 | 自定义舞蹈动作、ONNX 导出与 vanila MuJoCo / Unitree SDK 的 G1 真机部署 | 2 |
| [KyrieYoung926/supermimic_deploy](https://github.com/KyrieYoung926/supermimic_deploy) | 部署运行时 | ONNX Runtime + MuJoCo + Unitree SDK2，仅部署不含新训练基准 | 5 |
| [liudi159/supermimic-backflip-5090](https://github.com/liudi159/supermimic-backflip-5090) | 后空翻诊断 | 单卡 5090 的后空翻专项，跟踪接触力、穿模与落地稳定性 | 1 |
| [liuskywalkerjskd/supermimic_oli](https://github.com/liuskywalkerjskd/supermimic_oli) | 新本体 | 31 DoF LimX Oli，含重定向与整段 rollout 导出 | 1 |
| [chu-fa/isaaclab-supermimic](https://github.com/chu-fa/isaaclab-supermimic) | 可复用任务接口 | 把 SuperMimic learner/runner 打包给自定义 Isaac Lab 任务，附 Agibot A3 示例 | 0 |
| [shangchaohu718/supermimic-pera](https://github.com/shangchaohu718/supermimic-pera) | 交互任务 | 冻结跟踪器＋独立滑板策略；搬箱与楼梯仍在研发 | 0 |
| [luoshaqi/g1-moonwalk-supermimic](https://github.com/luoshaqi/g1-moonwalk-supermimic) | 微调实验 | 基于 SuperMimic_Dyn 的太空步微调，含前后对比视频 | 0 |
| [jackzedgar/SuperDex](https://github.com/jackzedgar/SuperDex) | 灵巧操作 | 目前只有概念 README，未见完整训练代码 | 2 |
| [hitsz-oyx/Ref2Dex-agent](https://github.com/hitsz-oyx/Ref2Dex-agent) | 接触式 MBRL | 把“可微动力学＋短视界 BPTT”思路迁到灵巧手接触任务（非人形跟踪） | 0 |
| [azizkhadraoui/g05](https://github.com/azizkhadraoui/g05) | VLA 借鉴 | 用世界模型／critic 梯度训练冻结 flow sampler 上游的噪声策略（LIBERO 转向任务） | 0 |

### 3.1 Go2-SuperMimic

**参考项目依据**：已阅读 [Zhengtianjin/go2-supermimic 的 README](https://github.com/Zhengtianjin/go2-supermimic/blob/054a13c13285a1c94b2e69dc8ac7331f9a8c71cf/README.md)，固定提交 `054a13c13285a1c94b2e69dc8ac7331f9a8c71cf`，核查日期为 2026-10-27。以下摘述作者说明，未运行其代码或复测权重。

该项目在 Unitree Go2 上独立实现 SuperMimic 的世界模型与想象轨迹反向传播，使用 mjlab ≥ 1.6／MuJoCo Warp，需要 CUDA GPU；任务是追逐逃逸球并闪避投来的球或横杆。

| 任务 | 场景与额外输入 | 发布的 actor 权重 |
| --- | --- | --- |
| `Go2-Chase-Flat` | 平地；本体与目标观测 | `weights/flat.pt` |
| `Go2-Chase-Obstacles` | 静态障碍；增加 16 射线平面 LiDAR | `weights/obstacles.pt` |
| `Go2-Chase-Terrain` | 台阶、方块、粗糙地面与坡面；增加 11 × 7 高程扫描 | `weights/terrain.pt` |

- **训练与产物**：README 的 `Train`、`Pretrained policies` 节说明，障碍与地形策略从平地策略续训；地形训练另用地形行走策略和平地策略采集世界模型数据。发布权重仅含 actor，可用于评估和录像；续训还需世界模型与 critic。
- **验证边界**：README 的 `Limitations`、`How the released policies were trained` 节说明，仅完成仿真，目标观测来自仿真真值，训练配方尚未逐一从零端到端复现；困难地形仍有跌倒与撞杆。
- **与本项目的差异**：本项目的 [Go2 SuperMimic 任务](go2_supermimic.md)面向速度跟踪与本体历史输入。该仓库可作为同机器人、同仿真底座的参考，其感知输入和训练历史需单独考虑，不能作为本项目盲走效果的验证。

## 4. 权重、数据与相关基线

| 资源 | 说明 |
| --- | --- |
| [RayZhao/supermimic_getup](https://huggingface.co/RayZhao/supermimic_getup) | Hugging Face 上的 G1 起身／跌倒恢复权重（标签含 `arxiv:2610.28480`） |
| [unpackableorange/SuperMimic_Dyn](https://huggingface.co/unpackableorange/SuperMimic_Dyn) | `SuperMimic_Dyn` 的权重镜像 |
| [lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) | 多数复现使用的 G1 重定向 LAFAN1 数据（40 条，2.45 h，50 Hz） |
| [bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed) | BONES-SEED，论文大规模基准的数据来源（需在 HF 接受条款） |
| [BeyondMimic](https://beyondmimic.github.io/) · [SONIC / GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) | 论文的主要对照基线，也是 SuperMimic 观测／执行器约定的来源 |
| [mujocolab/mjlab](https://github.com/mujocolab/mjlab) | 大量复现的公共底座（MuJoCo Warp）；本项目当前也基于 MjLab |

## 5. 与本项目的关系与可用性判断

- **直接相关**：[Zhengtianjin/go2-supermimic](https://github.com/Zhengtianjin/go2-supermimic) 是将 SuperMimic（世界模型＋想象 BPTT）用到四足 Unitree Go2 上的公开参考实现；任务、权重、训练限制及与本项目的差异统一见[专项说明](#31-go2-supermimic)。
- **底座一致**：`CuboiLeo/SuperMimic-Reproduced`、`humble02/supermimic_mjlab_vibed`、`panhainan2020/SuperMimic-MuJoCo`、`rzc156/supermimic-rtx5080` 等都跑在 mjlab / MuJoCo Warp 上，与本项目技术栈重合度最高。
- **迁移成本提示**：多数复现依赖 LAFAN1 重定向数据、G1 29 DoF 模型与 Isaac Lab 3；Isaac Lab 系仓库（zita-ch、HighTorque-Dev、wenconggan、liuskywalkerjskd、mode1223345）需要完整 Isaac Sim 环境，与本项目 MjLab 环境不通用。
- **可信度分层**：上述作者自报成绩（回报、失败段数、耗时）均未经本项目复跑；`zita-ch/supermimic_vibed` 自述“1 小时 AI 速写、未评审”，`fan-ziqi/supermimic-agy` 是智能体复现实验，`yagami-light7/SuperMimic-codex` 仅到 M0 冒烟，均应区别对待。
- **不建议直接使用**：`Tianao-Zhang/supermimic_dualhand`、`xinruilee04/supermimic-test`、`yhx1203/supermimic_mjlab`、`yangkunpeng-coder/supermimic_reproduce`、`jboyxs/Supermimic_for_transcube` 为空仓库或仅含 LICENSE；`BlackOtters/supermimicA3`、`zzhanYoung/supermimic-lite` 已 404；`Green-Fr0g/supermimic` 只有标题 README。

## 6. 检索方法与缺口

- **已执行**：GitHub 仓库检索 API 多组查询（`supermimic`、`super mimic`、`supermini`、`supermini+robot`、`supermimic+fork:true` 等）；对候选仓库逐个抓取 README（默认分支 main/master 回退）并检查文件列表；Hugging Face 模型／数据集／Space 检索；arXiv API 与官方项目页；Gitee API 与网页检索（无结果）；Bing 精确短语检索（仅中文博客、知乎、B 站等二手解读，未发现新仓库）。
- **无法执行**：本机 `web_search` 工具因搜索端点鉴权失败不可用，改用 GitHub/HF/arXiv 官方 API 与直接抓取；DuckDuckGo/SearX 实例网络不可达，百度返回跳转；小红书、B 站、知乎的社区帖未逐条核查（`SuperMimicCarnival` 中有其采集记录，属于他人转述）。
- **字段口径**：Star、推送时间取 2026-10-26 的 GitHub API 快照；`eess` 类论文与自动驾驶世界模型列表已剔除，只保留与 SuperMimic 直接相关的条目；GitHub 代码搜索需鉴权，故未做全站代码级检索，`supermimic` 命名之外的“同思路不同名”项目可能存在遗漏。
- **后续可做**：若要把世界模型路线接进本项目，建议先精读论文第 3 节与附录 A.1 的算法、`CuboiLeo/SuperMimic-Reproduced` 与 `Zhengtianjin/go2-supermimic` 的实现，再在 `docs/design/` 下补一份方案与验收标准，小规模验证（单卡、少量动作、固定种子）后再决定是否引入。

## 7. 本轮针对盲四足的补充核查（2026-10-27）

重新执行网页检索 `SuperMimic quadruped locomotion`、`SuperMimic go2 blind` 等组合，并核对GitHub默认分支和相关分支提交。检索未发现可直接作为本项目“从零训练盲Go2速度控制已复现”证据的另一份完整实现；这仅是本轮已检查范围的结论，不代表不存在其他项目。

- **同机器人训练源码**：`Zhengtianjin/go2-supermimic` 的main仍为 `054a13c13285a1c94b2e69dc8ac7331f9a8c71cf`。本轮额外阅读 [train.py](https://github.com/Zhengtianjin/go2-supermimic/blob/054a13c13285a1c94b2e69dc8ac7331f9a8c71cf/go2_supermimic/supermimic/train.py) 的 `TrainCfg`、`fit_world_model`、`improve_policy` 和 `run`，以及 [imagine.py](https://github.com/Zhengtianjin/go2-supermimic/blob/054a13c13285a1c94b2e69dc8ac7331f9a8c71cf/go2_supermimic/supermimic/imagine.py) 的 `_rewards`。它使用记录动作多步模型拟合和短视界BPTT；奖励主要解析计算，力矩由PD近似、加速度由关节速度差分得到。本项目为保留既有HIM传感器奖励而学习18维辅助状态，属于不同建模选择，不能直接用它的训练曲线推断本项目精度。其默认想象批量8192、模型窗口批量2048、1024宽模型和训练调度也与本机小批量不同；是否需要扩大批量或加入稀有接触采样，需本项目预测／控制评估后判断。
- **同仿真器的感知扩展**：实际读取 [InstinctMJ supermimic分支README](https://github.com/project-instinct/InstinctMJ/blob/7ca851e002980011f24a1664011305d0ba94aefc/README.md)，固定提交 `7ca851e002980011f24a1664011305d0ba94aefc`。该分支明确为G1深度视觉shadowing／parkour，不是盲四足locomotion；本轮未阅读其完整训练源码，不把其自报成绩作为本项目依据。
- **方法边界**：再次读取 [官方项目页](https://supermimic.github.io/) 的训练循环及问答，并对照本地英文原文§3.2、§3.3、§4.5。可复用的是持续拟合可微世界模型和短视界解析策略梯度；原文人形动作跟踪先验不随机器人改成Go2而自动存在。部署学生的CTS与八专家结构来自`go2_rl_gym`，不归因于SuperMimic原文。

源码／README快照和提交元数据保存在 `.artifacts/supermimic-reproduction/references/` 及既有 `.artifacts/quadruped-supermimic-survey/`。这里记录的是实际阅读与检索边界；本项目的实现修复、预测检查和物理实验统一见[迁移设计§18](design/go2_blind_supermimic.md#18-复现问题核查部署导出与千轮实验2026-10-27)及对应审查记录。
