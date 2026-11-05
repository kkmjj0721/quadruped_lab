# Copyright (c) 2026-2027 zh
"""
内容：
    导出 Go1 Depth 机器人与深度相机的资产配置工厂。
"""

from .go1_depth_constants import get_go1_depth_camera_cfg, get_go1_depth_robot_cfg

__all__ = ["get_go1_depth_camera_cfg", "get_go1_depth_robot_cfg"]
