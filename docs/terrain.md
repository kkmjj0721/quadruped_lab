# 盲走基础地形

实现位于 `src/terrain/blind/`，使用 MjLab 1.6.0 的 `SubTerrainCfg`，可直接交给地形生成器或场景配置。机器人前进参考为局部 `+X`；正/倒金字塔楼梯及正/倒金字塔坡的出生参考点位于地块中心平台，其他地形的出生参考点位于起始平台；机器人机身高度需在任务重置时另加。

| 名称 | 配置类型 | 默认参数与含义 |
| --- | --- | --- |
| `flat` | `FlatTerrainCfg` | 零高度有限平地 |
| `rough` | `RoughTerrainCfg` | 双线性插值随机高度图，最大起伏范围从 ±0 到 ±0.10 米；采样间距不大于 0.05 米，随机粗网格间距不大于 0.25 米 |
| `pyramid_stairs` | `PyramidStairsTerrainCfg` | 正金字塔楼梯，中心平台升高，级高从 0 到 0.18 米，踏面宽 0.35 米 |
| `pyramid_stairs_inv` | `InvertedPyramidStairsTerrainCfg` | 倒金字塔楼梯，中心平台下沉，级高从 0 到 0.18 米，踏面宽 0.35 米 |
| `slope_up` | `SlopeTerrainCfg` | MjLab 风格正金字塔坡，中心方平台升高，坡度从 0 到 1.0 |
| `slope_down` | `SlopeTerrainCfg(descending=True)` | MjLab 风格倒金字塔坡，中心方平台下沉，坡度从 0 到 1.0 |

默认地块长、宽均为 8 米，金字塔楼梯中心平台宽 1.5 米、外围边框宽 0.5 米；金字塔坡中心平台宽 2 米、外围边框宽 0.25 米；底座厚 0.15 米。楼梯沿四边生成同心方环，坡使用 MjLab 的两个三角剖面相乘形成四面连续坡，中心平台由平台宽度截平。高度图使用均匀网格；地块尺寸不能整除采样间距时，外围边框向内量化到第一个内部采样点，并以实际采样区间计算坡高。默认难度零全部退化成平地；修改各范围下限后，难度零对应修改后的下限。

## 生成截图

在仓库根目录执行：

```bash
uv run --no-sync python -B tools/preview_blind_terrain.py --output-dir outputs/terrain/blind
```

输出六张 `960×640` 单图、`overview.png` 总览、六个 `.mjb` 模型及 `metadata.json`。绿色点为起点，箭头为 `+X`；标记不参与碰撞。截图使用实际 MuJoCo 碰撞几何、统一相机和真实比例，不拉高粗糙地形的视觉幅值。

默认难度 `0.75`、种子 `42`；此时台阶级高 0.135 米，正/倒金字塔均按中心平台生成，金字塔坡坡度 0.75（约 36.9°），正坡中心平台高度约 1.5125 米，粗糙度上限 ±0.075 米。模型记录包含地形参数、实际高度范围、相机、软件版本、Git 状态及源文件 SHA256。

单独预览、调整难度和种子：

```bash
uv run --no-sync python -B tools/preview_blind_terrain.py \
  --terrain rough --difficulty 1 --seed 7 --output-dir outputs/terrain/rough_seed7
```

相同输出目录中的同名文件会被覆盖；保留多个结果时使用不同 `--output-dir`。Linux 默认使用 EGL，尊重已有 `MUJOCO_GL`；需要可用的离屏 OpenGL 环境。产物目录已由项目现有忽略规则排除。

生成整套课程配置接入训练场景的预览：

```bash
uv run --no-sync python -B tools/preview_blind_training_cfg.py \
  --output-dir outputs/terrain/training_cfg
```

该命令实际装配 `SceneCfg -> TerrainEntityCfg -> TerrainGeneratorCfg`，渲染 5 行课程、6 列地形和 30 个环境出生点，并保存 `blind_training_scene.png/.mjb/.json`。当前仓库没有完整机器人任务或 PPO 训练入口，截图是场景装配预览，不代表已经运行机器人训练。

## 接入场景

```python
from mjlab.scene import Scene, SceneCfg
from mjlab.terrains import TerrainEntityCfg
from src.terrain import make_blind_terrain_cfg

terrain_cfg = make_blind_terrain_cfg(seed=42, num_rows=5)
scene_cfg = SceneCfg(
    num_envs=6,
    terrain=TerrainEntityCfg(terrain_type="generator", terrain_generator=terrain_cfg),
)
scene = Scene(scene_cfg, device="cpu")
model = scene.compile()
```

上述示例仅装配地形，不包含机器人或训练。课程模式每种地形一列，难度沿行从 `difficulty_range[0]` 增至 `[1]`；只有一行时使用下限。每次工厂调用返回独立配置，可按需修改。整套训练场景装配预览使用 `tools/preview_blind_training_cfg.py`。

```python
terrain_cfg = make_blind_terrain_cfg("pyramid_stairs", num_rows=3)
terrain_cfg.sub_terrains["pyramid_stairs"].step_height_range = (0.03, 0.12)
terrain_cfg.sub_terrains["pyramid_stairs"].step_width = 0.4

random_cfg = make_blind_terrain_cfg(curriculum=False, num_rows=3, num_cols=6)
random_cfg.sub_terrains["rough"].proportion = 2.0
```

`size=(长度, 宽度)` 在工厂统一设置，由生成器传给各地块。起点地面参考高度通过生成器的 `terrain_origins` 或场景的 `env_origins` 获取；它不是机器人机身的 Z 坐标。默认固定起点平台，不支持 `flat_patch_sampling`，显式设置会报错。

这些是独立训练地块，相邻地块的边缘高度不保证连续。后续任务需按本地块边界重置，并使用与 `+X` 一致的初始朝向与前进命令；本次不实现边界终止逻辑。

## 验证与限制

```bash
uv run --no-sync python -B -m unittest discover -s tests/test_terrain -v
```

本项目已验证事实：测试覆盖六类地形的编译和平台原点、零难度退化、正/倒金字塔同心台阶、正/倒金字塔坡中心平台与边框、随机种子及起伏范围、非整除平台边界、课程/随机地块组合，以及球体落地接触。实际渲染结果见 `outputs/terrain/blind/`；整套训练场景装配截图见 `outputs/terrain/training_cfg/`。具体结果与独立审查见 [设计记录](design/custom_terrain.md) 和 [审查记录](review/custom_terrain.md)。

本地 MuJoCo 3.11.0 的 `mj_ray` 在部分恰落高度图网格行的射线上可能返回底面；测试轮廓使用非网格行坐标，并另测原点和实体接触。该观察不等于已证实物理碰撞失效，也不影响本次真实渲染结果；后续若使用高度射线观测，应单独验证所用传感器后端。

默认地形参数是初始配置，不是已经验证的机器人通行范围。本次没有机器人行走训练或 GPU 批量接触实验。接口与高度图编码依据及与参考项目的差异，记录在设计文档中。
