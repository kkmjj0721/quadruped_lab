# Copyright (c) 2026-2027 zh
"""内容：导出 Go1 Depth 基础配置与相机配置工厂。"""

from .go1_depth_config import Go1DepthCfg, make_go1_depth_camera_cfg

__all__ = ["Go1DepthCfg", "make_go1_depth_camera_cfg"]
