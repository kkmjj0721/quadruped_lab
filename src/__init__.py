# Copyright (c) 2026-2027 zh
"""
内容：
    定义项目源码根路径，供静态资源配置使用。
"""

from pathlib import Path


SRC_PATH: Path = Path(__file__).parent

__all__ = ["SRC_PATH"]
