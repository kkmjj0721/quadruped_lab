# quadruped_lab

基于 MjLab 的四足机器人项目，提供基础 PPO 和 Go2 SuperMimic＋CTS-MoE 任务。

## 安装

环境安装与检查见 [安装说明](docs/setup.md)。

## 算法

项目算法统一位于 `agents/`，基础 PPO 放在 `agents/base/`。目录、环境接口与用法见 [算法说明](agents/README.md)。

## Go2 SuperMimic 训练

通过统一训练入口启动粗糙地形任务：

```bash
uv run --no-sync python scripts/train.py Unitree-Go2-SuperMimic-Rough \
  --device cuda:0 --num-envs 64 --max-iterations 1000
```

平地使用 `Unitree-Go2-SuperMimic-Flat`，CPU 使用 `--device cpu`，环境数量至少为2。
默认执行模型与策略更新；配置覆盖、仅模型训练、检查点续训及验证范围见
[SuperMimic任务说明](docs/go2_supermimic.md)。

## Go2 PPO 任务

Go2 PPO 任务位于 `src/tasks/locomotion/go2_ppo/`，参考任务注册为 `Unitree-Go2-Flat` 和 `Unitree-Go2-Rough`。

本项目已按需求移除 `pose` 姿态奖励及对应的站立／行走／跑动标准差参数，其余奖励保留。

训练和播放入口参考相邻 `legged_wbc_mjlab/scripts/train.py`、`play.py`，
通过任务 ID 读取注册的配置和 MjLab PPO runner。训练示例：

```bash
uv run --no-sync python scripts/train.py Unitree-Go2-Flat \
  --env.scene.num-envs 4096 --agent.max-iterations 10001
```

粗糙地形使用 `Unitree-Go2-Rough`。默认使用首张可见 GPU；CPU 轻量检查可传
`--gpu-ids None --env.scene.num-envs 1 --agent.max-iterations 1`。
环境和训练参数分别通过 `--env.*`、`--agent.*` 覆盖，例如 `--agent.seed 7`；
布尔参数显式传入 `True` 或 `False`。完整参数见 `python scripts/train.py Unitree-Go2-Flat --help`。

训练产物保存在 `logs/rsl_rl/<experiment_name>/<时间戳>[_run_name]/`，
包含检查点、`policy.onnx` 和 `params/env.yaml`、`params/agent.yaml`；可通过
`--log-root`、`--agent.experiment-name`、`--agent.run-name` 设置位置和名称。
续训时指定已有实验目录名称，任务和网络配置应与检查点匹配：

```bash
uv run --no-sync python scripts/train.py Unitree-Go2-Flat \
  --agent.resume True --agent.load-run '<已有实验目录名>' \
  --agent.max-iterations 1000
```

播放本地检查点：

```bash
uv run --no-sync python scripts/play.py Unitree-Go2-Flat \
  --checkpoint-file '<实验目录>/model_1000.pt' --num-envs 1
```

`--viewer auto` 根据显示环境选择 MuJoCo 窗口或 Viser 浏览器查看器，也可显式设置
`native`／`viser`。没有检查点时，可用 `--agent zero` 或 `--agent random` 检查环境。
播放数量默认沿用任务配置；日常查看建议显式传入 `--num-envs 1`。
`--video True` 启用录像；`python scripts/play.py Unitree-Go2-Flat --help` 可查看播放选项。
两个入口也支持 `python -m scripts.train` 和 `python -m scripts.play`。
旧入口 `scripts/train_go2.py` 已由统一训练入口替代。

环境和训练参数统一在 [Go2 配置](src/config/go2/go2_config.py) 中维护：`Go2Cfg`
定义控制、命令、观测噪声、随机化、奖励、扫描、地形和播放参数，`Go2CfgPPO`
定义网络、PPO 和 runner 参数。任务工厂读取配置并装配 MjLab 对象，也可传入独立实例：

```python
from src.config.go2.go2_config import Go2Cfg, Go2CfgPPO
from src.tasks.locomotion.go2_ppo.config import (
    unitree_go2_ppo_runner_cfg,
    unitree_go2_rough_env_cfg,
)

go2_cfg = Go2Cfg()
go2_cfg.env.num_envs = 64
go2_cfg.commands.ranges.lin_vel_x = (-0.5, 1.0)
go2_cfg.commands.curriculum = False  # 固定命令范围，不被速度课程覆盖。
go2_cfg.terrain.generator["num_rows"] = 5
env_cfg = unitree_go2_rough_env_cfg(go2_cfg=go2_cfg)

ppo_cfg = Go2CfgPPO()
ppo_cfg.algorithm.learning_rate = 5e-4
runner_cfg = unitree_go2_ppo_runner_cfg(ppo_cfg=ppo_cfg)
```

`sim.flat` 和 `play` 分别管理平地与播放覆盖；地形生成器的 `curriculum`
控制地形分配和地形课程。改变扫描范围、历史长度或网络尺寸后，应重新核对检查点兼容性；
配置中的观测维度常量描述默认设置。参数来源、验证范围和独立审查见
[配置集中设计](docs/design/go2_config_centralization.md)及其
[审查记录](docs/review/go2_config_centralization.md)。
