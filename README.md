# 🐾 Quadruped Lab

基于 **MjLab** 的四足机器人运动控制项目，面向 **Unitree Go2**，支持平地与粗糙地形的速度跟踪、策略训练、可视化播放和策略导出。

提供 PPO 速度跟踪任务与 SuperMimic 盲速度跟踪任务。SuperMimic 使用混合专家历史策略与可微世界模型；Go2 PPO 任务使用 MjLab / RSL-RL 运行器。

| 算法 | 平地任务 | 粗糙地形任务 |
| --- | --- | --- |
| PPO | `Unitree-Go2-Flat` | `Unitree-Go2-Rough` |
| SuperMimic | `Unitree-Go2-SuperMimic-Flat` | `Unitree-Go2-SuperMimic-Rough` |

## 🚀 快速开始

环境要求：Linux x86_64、Python 3.11；GPU 训练使用兼容 CUDA 13.0 的 NVIDIA 驱动。以下命令均在仓库根目录执行。

```bash
bash setup.sh --torch-backend cu130
export PATH="$HOME/.local/bin:$PATH"
uv run --no-sync python -B tools/check_install.py --device cuda
```

安装脚本会准备 uv、Python 和项目虚拟环境。CPU 安装与更多选项见 [安装指南](docs/setup.md)；日常运行保留 `--no-sync`，沿用已安装的后端。

## 🧠 训练

```bash
# PPO：平地速度跟踪
uv run --no-sync python scripts/train.py Unitree-Go2-Flat \
  --env.scene.num-envs 4096 --agent.max-iterations 10001

# SuperMimic：平地盲速度跟踪
uv run --no-sync python scripts/train.py Unitree-Go2-SuperMimic-Flat \
  --device cuda:0 --num-envs 64 --max-iterations 1000
```

粗糙地形使用表中对应的 `Rough` 任务。环境数与训练轮数按资源和实验目标调整；SuperMimic 至少使用 2 个并行环境，其他训练预算见 [任务指南](docs/go2_supermimic.md)。

默认产物目录：

- **PPO**：`logs/rsl_rl/go2_ppo/<运行标识>/`
- **SuperMimic**：`logs/supermimic/go2_supermimic/<运行标识>/`

## 🎮 策略播放

将检查点路径替换为实际文件，并选择与训练一致的任务：

```bash
# 播放 PPO 策略
uv run --no-sync python scripts/play.py Unitree-Go2-Flat \
  --checkpoint-file '<实验目录>/model_1000.pt' --num-envs 1

# 播放 SuperMimic 策略
uv run --no-sync python scripts/play.py Unitree-Go2-SuperMimic-Flat \
  --checkpoint-file '<实验目录>/model_1000.pt' --num-envs 1
```

查看器默认自动选择 MuJoCo 窗口或 Viser 浏览器界面，也可通过 `--viewer native` / `--viewer viser` 指定。添加 `--video True` 可将录像保存到实验目录的 `videos/play/`。

没有检查点时，可用 `--agent zero --num-envs 1` 查看环境。SuperMimic 播放需要保留检查点同目录的 `task_config.json`。完整参数可在训练或播放命令后添加 `--help` 查看。

## 项目结构

```text
src/tasks/   # 任务环境、奖励与课程
src/config/  # 机器人与任务配置
src/assets/  # 机器人模型与场景资源
agents/      # 基础 PPO 组件与 SuperMimic 算法
scripts/     # 训练与播放入口
tools/       # 安装检查与策略导出
docs/        # 安装与任务用法
```

## 文档与许可

- [SuperMimic 任务指南](docs/go2_supermimic.md)：配置、续训、评估与导出。
- [算法说明](agents/README.md)：算法组件、接口与来源。
- [Go2 配置](src/config/go2/)：机器人、控制与任务参数。

项目采用 [MIT 许可](LICENSE)；引入的算法与机器人资产遵循各自目录中的许可及来源说明。
