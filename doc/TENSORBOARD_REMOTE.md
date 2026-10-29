# TensorBoard 使用与远程查看指南

本文是 `AGENTS.md` 与 [开发工作流](../docs/DEVELOPMENT_WORKFLOW.md#6-tensorboard-前置要求) 引用的 TensorBoard 操作规范。涉及训练日志可视化前先读本文，并按本文的命令与路径执行；不要在缺少依据时自行假设远程地址、端口或日志目录。

## 1. 本项目的日志位置

| 任务 | 运行目录 | 日志文件 |
| --- | --- | --- |
| Go2 SuperMimic（`Unitree-Go2-SuperMimic-*`） | `logs/supermimic/<experiment_name>/<运行标识>/` | `events.out.tfevents.*` |
| Go2 PPO（`Unitree-Go2-*`） | `logs/rsl_rl/<experiment_name>/<运行标识>/` | `events.out.tfevents.*` |

SuperMimic 一次运行包含 `model_<N>.pt` 检查点、当前 `policy.onnx`、一个 TensorBoard 事件文件、一份 `task_config.json` 任务快照（播放与续训依赖），以及 `params/` 下的环境、算法、运行来源 YAML 与代码差异。每次保存检查点同步覆盖ONNX，元数据标识其轮次；指标口径、tag 分组与终端摘要的关系见 [日志设计](../docs/design/go2_blind_supermimic.md)。

常用曲线集中在 `Loss/`（世界模型、价值与控制损失和各自学习率）、`Train/`（回报与回合长度，`/time` 以会话秒数为横轴）、`Perf/`（吞吐及采集／学习耗时）。`Metrics/tracking/*_rmse` 与终端速度误差口径一致；`episode/` 还含完成数、失败／超时占比和回报分布，`replay/` 含占用情况。失败／超时占比只在本轮有回合结束时写入。原有标量tag保留，新记录在下次启动或续训后生效。

`logs/` 已被 `.gitignore` 忽略，不纳入版本控制。

## 2. 本地查看

```bash
# 单次运行
.venv/bin/tensorboard --logdir logs/supermimic/go2_supermimic/<运行标识> --port 6006

# 整个实验（按运行列出多条曲线）
.venv/bin/tensorboard --logdir logs/supermimic/go2_supermimic --port 6006
```

`tensorboard` 可执行文件由依赖 `tensorboard`（经 `mjlab` 传递引入）提供；若提示 `TensorFlow installation not found`，属于只装 `tensorboard` 包时的正常提示，不影响曲线查看。

默认只监听 `127.0.0.1`。同机浏览器直接访问 `http://127.0.0.1:6006`。需要局域网访问时才加 `--host 0.0.0.0`，并确认主机防火墙策略；不要把端口暴露到公网。

## 3. 远程机器训练

训练在远程主机、浏览器在本地时，优先用 SSH 端口转发，不额外开放服务端口。

```bash
# 在本地机器执行：把远程 6006 端口转发到本地 6006
ssh -N -L 6006:127.0.0.1:6006 <用户名>@<远程主机>
```

随后在远程主机上用默认监听地址启动：

```bash
.venv/bin/tensorboard --logdir logs/supermimic/go2_supermimic --port 6006
```

本地浏览器访问 `http://127.0.0.1:6006`。训练与 TensorBoard 分属两个 SSH 会话即可，TensorBoard 读取事件文件不需要与训练同进程；训练中断后事件文件仍可查看。

远程主机地址、跳板机、用户名与端口由使用者按实际环境填写，本项目不内置任何默认远程主机。

## 4. 常见问题

- **看不到曲线**：确认 `--logdir` 指向的运行目录内确有 `events.out.tfevents.*`；SuperMimic 不再写 `metrics.jsonl`，用旧路径找 JSONL 会找不到文件。
- **曲线只到训练中断处**：`SummaryWriter` 每10秒刷新待写入事件，正常关闭时完成刷新；异常退出时最后若干轮可能未写入，以检查点 `model_<N>.pt` 的轮次为准。
- **多个事件文件**：续训或重复启动会新建运行目录，同一目录出现多个 `events.out.tfevents.*` 时 TensorBoard 会并列显示，属预期行为。
- **端口被占用**：换 `--port`，并同步修改 SSH 转发的本地端口。

## 5. 不要做的事

- 不要为查看日志在共享主机上长期占用默认端口；用完即停。
- 不要把 TensorBoard 直连公网或设置 `--bind_all` 后不加访问控制。
- 不要删除训练目录中的事件文件来“清理”，它与检查点同为实验记录的一部分。
