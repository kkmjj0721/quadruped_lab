# MGDP 地形来源与适配边界

本目录以 [arclab-hku/MGDP](https://github.com/arclab-hku/MGDP/tree/f43139d1d97849819a0dce609dc0b59faf577b3f) 的 `f43139d1d97849819a0dce609dc0b59faf577b3f` 为基准。本项目新增代码版权为 `Copyright (c) 2026-2027 zh`；不改变上游代码所属权利。

| 本地文件 | 官方来源及范围 | 迁移方式 |
| --- | --- | --- |
| `_parkour.py` | `legged_gym/legged_gym/utils/new_terrains/add_extreme_gap_terrain.py:217–1118` 中原版默认实际可达地形所需函数 | 纯几何语句等价移植；每个函数注明原始行号；仅把全局 NumPy/Python 随机流改为实例局部流；保留量化、常数、坐标与随机调用顺序 |
| `_primitives.py` | `isaacgym/python/isaacgym/terrain_utils.py:17–50,77–151,211–244` 的噪声、坡、障碍、阶梯数学规则 | 独立实现相同整数运算及抽样顺序，不导入或分发 IsaacGym；原工具文件版权属于 NVIDIA CORPORATION |
| `generator.py` | `utils/terrain.py:14–167,199–205,236–327`；`new_terrains/add_mix_terrain.py:4–42,333–397`；`new_terrains/add_extreme_gap_terrain.py:10–213` | 保留原版默认有效列分配、`i/rows`、随机流、origin、边缘掩码及悬空对象标签补写顺序 |
| `generator.py` 两套配置 | `envs/random_dog/random_dog_config_stage1.py`、`random_dog_config_stage2.py` 和其基类 | 保留未归一化累计比例；默认 mix 只生成 0/1/2/3/4，parkour 只生成 3/4/5/7/8/13/14/15/16/17 |
| `query.py` | `envs/baseline/legged_robot_terrains.py:181–218` 的 min 三点高程查询 | 以分离家族后的世界偏移转换回原索引；不加入观测噪声 |
| `config.py` | 本项目 MjLab/MuJoCo 适配 | 原高程以 hfield 承载，悬空 mesh 的最终八顶点还原独立盒体碰撞 |

训练按用户最新要求采用原版二阶段：第一阶段只生成 mix，第二阶段只生成 parkour；单阶段地图保持官方世界原点，各有十列，不归一化原版权重、不补造原版未实际采样到的类别。每次生成使用给定种子的独立 NumPy/Python 随机流，便于与官方单独运行同种子对拍，同时避免污染训练的全局随机流。

显式诊断选项 `family="combined"` 可同时生成两套地图，默认训练不采用。该选项将 parkour 沿世界 x 平移 210 米以避免重叠，统一 origins 的 parkour 后十行仅为数组占位；使用者必须服从各列 `level_counts`。单家族模式不需要占位或整体平移。

原版先从整数高程构建地面 mesh 和 edge mask，再补写悬空梁顶面／顶棚底面到监督高程。本目录分别保存 `collision_height_raw` 与 `label_height_raw`，不会把顶棚标签当作地板。悬空实体的最终三位小数顶点、原始整数标签截断、附加对象使用 `i/(rows-1)` 的难度都保留。原版 `add_roughness` 调用未传难度，因此 parkour 噪声幅值始终从 `[0.01,0.04]` 采样。

SciPy 新版本移除了 `interp2d`。基础噪声采用规则网格线性 `RectBivariateSpline` 等价表达，保持原 x/y 轴与 `rint` 整数量化。独立测试从官方 Git 对象读源码作为 oracle；若本机未提供该仓库，oracle 测试明确跳过，常规几何和 MuJoCo 对齐测试仍运行。

MuJoCo 适配经用户明确批准：保留原始整数格点、格距和全部附加盒体尺寸，不复现 PhysX 的陡边顶点挪移及 `pyfqmr` 保留百分之五三角面的简化。hfield 在相邻格点之间形成三角面，陡边附近与原始陡边 mesh 的接触位置存在一个格距量级的局部横向差异，具体接触动力学不能声称等价。验证分别检查格点高度量化误差、世界坐标、代表陡边射线及悬空碰撞和独立标签。代表例：mix 第十行上楼梯的 0.14 米首级，在距阶边 0.05 米的位置，MuJoCo hfield 射线高度比官方未简化陡边网格高 0.07 米；格点高度误差小于 `5e-7` 米。未比较最终 `pyfqmr` 简化网格或 PhysX 接触动力学，不能把这一代表样本当作所有碰撞差异的上界。
