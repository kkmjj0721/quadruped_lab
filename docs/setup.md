# 安装配置

文中的本地工作记录位于 `docs/work/`，不随仓库提交。

本项目参考相邻 `legged_wbc_mjlab` 的 uv 安装方式，单独维护依赖声明、锁文件和 `.venv`。当前已提供环境安装与检查入口，基础 PPO 已迁入 `agents/base/`，机器人训练入口、评估和任务逻辑仍待开发。

## 环境与版本

本次支持 Linux x86_64，推荐 Ubuntu 22.04。CUDA 模式需要兼容 CUDA 13.0 的 NVIDIA 驱动；CPU 模式用于导入和配置开发，不代表支持 MjLab 的 GPU 训练。

| 组件 | 版本与来源 |
| --- | --- |
| Python | `.python-version` 固定为 3.11.16，由 uv 准备 |
| uv | 至少 0.8.0；本机验证使用 0.12.10 |
| MjLab | `mjlab==1.6.0`，来自 PyPI |
| 基础 PPO | 本地 `agents/base/`，从参考仓库的 RSL-RL 5.4.2 精简迁移 |
| RSL-RL 发行依赖 | `rsl-rl-lib==5.4.2`，保留供 MjLab 使用；项目算法使用本地 `agents` |
| TensorDict | `tensordict>=0.7.0`，算法直接使用；实际版本记录在 `uv.lock` |
| PyTorch | `torch==2.14.1`，按后端使用官方 CPU 或 cu130 wheel |
| torchvision | `torchvision==0.29.1`，与 PyTorch 使用相同后端 |
| 其他依赖 | 完整版本与下载校验信息记录于 `uv.lock` |

截至 2026-10-25，本机为 Ubuntu 22.04.5、RTX 4060 8 GiB、NVIDIA 驱动 580.159.03、CUDA Toolkit 13.0，安装入口会自动选择 `cu130`。

项目算法统一从本地 `agents` 导入，目前仅包含前馈基础 PPO；参考仓库的 HIM 等扩展未迁移。保留上游 RSL-RL 发行包以满足 MjLab 的依赖，不建立同名遮蔽包或相邻仓库路径依赖。用法及维护边界见 [算法说明](../agents/README.md)，来源见 [迁移记录](../agents/base/NOTICE.md)。

## 安装与检查

在仓库根目录运行：

```bash
bash setup.sh
export PATH="$HOME/.local/bin:$PATH"
uv run --no-sync python -B tools/check_install.py --device cuda
```

安装入口定位自身所在的仓库，复用已有 uv，缺少 uv 时通过官方安装器准备；随后按 `.python-version` 选择 Python，并对实际下载文件测速、选择镜像与失败回退，创建或更新本仓库 `.venv`。首次安装需要联网，后续复用 uv 缓存。脚本不会安装系统级 NVIDIA 驱动或 CUDA Toolkit。

上面的 `export PATH` 让当前终端找到默认安装到 `~/.local/bin` 的 uv；子进程安装工具后不能修改父终端的 PATH。后续终端可在自己的 Shell 配置中保留该目录，或按 uv 官方安装器的提示加载环境。

检查工具验证核心依赖、RSL-RL 子模块及本地 `agents.base` 导入路径、MuJoCo 简单模型加载与步进、PyTorch 张量运算和 torchvision NMS。CUDA 模式还验证 Warp 的 CUDA 设备分配与数据传输；任一失败返回非零退出码。此检查不包含训练、策略效果或图形窗口验证。

查看参数：

```bash
bash setup.sh --help
uv run --no-sync python -B tools/check_install.py --help
```

## 后端选择

| 安装命令 | 行为 |
| --- | --- |
| `bash setup.sh` | 自动读取 PATH 中 `nvcc --version`；Toolkit 至少为 13.0 时选 `cu130`，其余情况选 `cpu` 并显示原因 |
| `bash setup.sh --torch-backend cu130` | 显式安装 CUDA 13.0 wheel，适用于已有兼容驱动但没有 nvcc 的机器 |
| `bash setup.sh --torch-backend cpu` | 显式安装 CPU wheel |

自动选择沿用参考项目的 Toolkit 检测依据，不使用 `nvidia-smi` 显示的驱动 CUDA 支持上限。预编译 PyTorch CUDA wheel 本身不要求本机安装相同版本 Toolkit；只有驱动的机器可显式指定 `cu130`，再用 CUDA 检查确认实际可用。自动选择也不代替运行时的驱动验证。

切换到 CPU 并检查：

```bash
bash setup.sh --torch-backend cpu
uv run --no-sync python -B tools/check_install.py --device cpu
```

切回本机 CUDA 环境：

```bash
bash setup.sh --torch-backend cu130
uv run --no-sync python -B tools/check_install.py --device cuda
```

`cpu` 与 `cu130` 是互斥的可选依赖组，不同时安装。日常命令使用 `uv run --no-sync`，避免未指定后端的自动同步改变环境。手动同步时始终带上后端，例如 `uv sync --locked --extra cu130`。

## 镜像、代理与离线模式

默认的 `bash setup.sh` 按以下阶段准备下载通道：

| 阶段 | 测速样本 | 默认候选 |
| --- | --- | --- |
| Python | uv 返回的固定版本真实 standalone archive | Astral、GitHub、南大、USTC |
| 普通 PyPI | 锁定的 numpy Linux x86_64 / CPython 3.11 wheel | 官方、清华、阿里、华为 |
| Torch | 锁定后端的 torch 与 torchvision 两个实际 wheel | 官方 R2、官方主域名、阿里、南大 |

每个文件默认读取最多 512 KiB，检查实际 ZIP/压缩包前缀，并按实际读取字节与耗时排序；HTML 验证页、空响应和错误 Range 不会算作可用通道。前缀速度受连接建立和网络波动影响，不保证整个大文件保持相同速度。

Python 已存在时直接复用。主环境已经与锁文件一致时跳过测速；需要主动查看通道速度时执行：

```bash
bash setup.sh --probe-only
```

只测速模式要求 uv 与固定版本的 uv Python 已存在，不下载 Python、不安装项目依赖。普通安装优先用系统 Python 3.10+ 准备下载助手，再切换到项目的 Python 3.11；系统没有 Python 3.10+ 时，先由 uv 的默认通道或显式给定的 Python 镜像准备解释器，该初次 Python 下载无法预先测速。

安装前在原项目检查锁文件，随后仅在项目内的临时副本替换 artifact 下载 URL，使用 `uv sync --frozen` 下载。registry、版本、文件名、依赖、markers 与 SHA256 保持原样，uv 校验完整文件；成功或失败后清理临时副本，并核对原 `uv.lock` 未变。普通依赖和后端的声明来源仍是 PyPI/PyTorch 官方索引，后端配置依据 [uv 官方 PyTorch 配置](https://docs.astral.sh/uv/guides/integration/pytorch/)。

某次安装失败或达到时限时，先终止并等待安装进程退出，再尝试下一组测速源；默认最多尝试 5 组测速结果与最后一次原官方通道。自定义任一类源时只使用该类给出的列表，不额外尝试该类的公网源。Python 下载阶段同样按测速顺序回退，不实现独立分段下载器。

调整测速与安装上限：

```bash
bash setup.sh --probe-timeout 10 --probe-bytes 1048576 --download-timeout 1800
```

`--probe-timeout` 是每个实际文件的前缀读取上限；`--download-timeout` 是每次 uv 同步的时限。已有系统 Python 时，助手准备目标 Python 每次最多使用其中的 600 秒；没有系统 Python 的初次引导下载使用 uv 的网络超时和重试配置。可重复指定每类的下载根，例如：

```bash
bash setup.sh \
  --python-mirror https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone \
  --pypi-mirror https://repo.huaweicloud.com/repository/pypi/packages \
  --torch-mirror https://mirrors.aliyun.com/pytorch-wheels \
  --torch-mirror https://mirror.nju.edu.cn/pytorch/whl
```

`--pypi-mirror` 包含 `/packages`，保留锁文件中该目录后的哈希路径；`--torch-mirror` 后追加 `/cpu/<文件名>` 或 `/cu130/<文件名>`。自定义根需要提供相同版本、相同字节内容的文件，不以镜像自己声明的 hash 替换项目原 hash。

直接使用原锁定命令可加 `--skip-probes`：

```bash
bash setup.sh --skip-probes
```

`--offline`、`--check`、`--dry-run` 跳过测速；相对链接、解析索引覆盖参数，或未来引入本地 path/editable/workspace 依赖时，也回到原项目的锁定同步，保留原路径语义。

需要维护普通依赖的解析索引时，先显式更新锁文件来源，再使用同一索引安装，例如：

```bash
uv lock --default-index https://pypi.tuna.tsinghua.edu.cn/simple
bash setup.sh -- --default-index https://pypi.tuna.tsinghua.edu.cn/simple
```

此操作可能更新 `uv.lock` 中的源地址和解析结果，完成后审阅差异。该 `--default-index` 参数走直接锁定同步，PyTorch 后端仍由项目的专用索引决定。

切回项目默认源：

```bash
uv lock
bash setup.sh
```

uv、官方安装器和测速助手读取标准 `HTTP_PROXY`、`HTTPS_PROXY` 及对应小写变量；`UV_PROXY` 可统一补齐尚未设置的标准变量，保留已有设置。Python 下载也支持 `UV_PYTHON_INSTALL_MIRROR`，显式 `--python-mirror` 优先；使用方式见 [uv Python 安装说明](https://docs.astral.sh/uv/guides/install-python/)。

网络较慢时可调整 uv 的超时和并发：

```bash
UV_HTTP_TIMEOUT=120 UV_HTTP_RETRIES=5 UV_CONCURRENT_DOWNLOADS=4 bash setup.sh
```

已有 uv、匹配的 Python 和完整依赖缓存时，可以离线同步，不会发出测速请求：

```bash
bash setup.sh --torch-backend cu130 -- --offline
# 也可以使用 UV_OFFLINE=1 bash setup.sh
```

`--` 后透传 uv sync 参数；后端、目标项目、Python 与虚拟环境由安装入口统一管理，相关覆盖参数会被拒绝。离线模式不负责准备尚未缓存的工具、Python 或 wheel。

## 维护与验证范围

依赖声明变更后先运行 `uv lock` 更新锁文件，再执行安装与设备检查。直接安装使用 `--locked`，加速安装先验证原锁再使用只改 URL 的临时 `--frozen` 副本，两条路径都不静默重写原锁文件。Python 版本声明、锁文件及本说明应同步维护。

本项目暂作为 uv 管理的应用仓库，`package = false`，不构建发行包。开发时在仓库根目录运行，IDE 解释器选择本项目 `.venv/bin/python`。虚拟环境、缓存和验证产物已加入 `.gitignore`。

可核对当前 CUDA 环境与锁文件是否一致：

```bash
uv lock --check
uv sync --locked --extra cu130 --check
uv pip check --python .venv/bin/python
```

本地环境验收与独立审查记录见 [安装环境设计](work/design/installation.md) 和 [安装配置审查](work/review/installation.md)。

本地下载加速设计与验收见 [下载测速与回退](work/design/download_acceleration.md) 及 [下载加速审查](work/review/download_acceleration.md)。
