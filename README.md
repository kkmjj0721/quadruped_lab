# quadruped_lab

基于 MjLab 的四足机器人项目，提供基础 PPO 和 Go2 SuperMimic 盲速度跟踪任务。SuperMimic 默认采用混合专家历史编码策略与可微世界模型。

## 安装

环境安装与检查见 [安装说明](docs/setup.md)。

## 算法

项目算法统一位于 `agents/`，基础 PPO 放在 `agents/base/`。目录、环境接口与用法见 [算法说明](agents/README.md)。

## Go2 SuperMimic 训练

通过统一训练入口启动平地速度跟踪：

```bash
uv run --no-sync python scripts/train.py Unitree-Go2-SuperMimic-Flat \
  --device cuda:0 --num-envs 64 --max-iterations 1000
```

粗糙地形使用 `Unitree-Go2-SuperMimic-Rough`，CPU 使用 `--device cpu`，环境数量至少为2。
默认执行模型与策略更新；配置覆盖、仅模型训练、检查点续训及验证范围见
[SuperMimic任务说明](docs/go2_supermimic.md)。

## Go2 PPO 任务

Go2 PPO 任务位于 `src/tasks/locomotion/go2_ppo/`，参考任务注册为 `Unitree-Go2-Flat` 和 `Unitree-Go2-Rough`。

任务实现完整采用相邻 `legged_wbc_mjlab/src/tasks/locomotion/go2_ppo`，包含参考姿态奖励、阶段速度课程和原始域随机化。

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

环境、奖励和课程参数见 [基础任务配置](src/tasks/locomotion/go2_ppo/go_ppo_env_cfg.py)，
机器人与播放覆盖见 [环境工厂](src/tasks/locomotion/go2_ppo/config/env_cfgs.py)，
网络、PPO 和 runner 默认值见 [训练配置](src/tasks/locomotion/go2_ppo/config/rl_cfg.py)。
[Go2 配置](src/config/go2/go2_config.py) 提供机器人控制、初始状态和基础重置参数。
任务默认使用 1 个环境，可通过 CLI 或返回的配置对象覆盖：

```python
from src.tasks.locomotion.go2_ppo.config import (
    unitree_go2_ppo_runner_cfg,
    unitree_go2_rough_env_cfg,
)

env_cfg = unitree_go2_rough_env_cfg()
env_cfg.scene.num_envs = 64
env_cfg.commands["twist"].ranges.lin_vel_x = (-0.5, 1.0)
env_cfg.curriculum.pop("command_vel")  # 固定命令范围。

runner_cfg = unitree_go2_ppo_runner_cfg()
runner_cfg.algorithm.learning_rate = 5e-4
```

复制范围、依赖差异与验证边界见 [Go2 任务设计](docs/design/go2_local_ppo.md)及其
[审查记录](docs/review/go2_local_ppo.md)。
