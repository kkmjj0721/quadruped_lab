# Copyright (c) 2026-2027 zh
"""内容：配置训练入口的已知依赖弃用提示和实际 CUDA 编译库路径。"""

import os
from pathlib import Path
import warnings


def configure_dependency_warnings() -> None:
    """保留部署所需 TorchScript，仅处理当前依赖版本的确切弃用提示。"""
    warnings.filterwarnings(
        "ignore",
        message=r"^`torch\.jit\.script` is deprecated\. Please switch to `torch\.compile` or `torch\.export`\.$",
        category=FutureWarning,
        module=r"^torch\.jit\._script$",
    )


def configure_cuda_libdevice() -> None:
    """优先尊重显式路径；仅在实际 CUDA libdevice 存在时补齐专用环境变量。"""
    if "TRITON_LIBDEVICE_PATH" in os.environ:
        return
    root = Path(os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda")
    library = root / "nvvm/libdevice/libdevice.10.bc"
    if library.is_file():
        os.environ["TRITON_LIBDEVICE_PATH"] = str(library.resolve())
