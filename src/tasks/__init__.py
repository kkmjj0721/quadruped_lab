# Copyright (c) 2026-2027 zh
"""
内容：
    提供显式任务注册入口，单独导入某个任务时不加载其他任务。
"""

from importlib import import_module


def register_tasks() -> None:
    """注册本项目使用MjLab注册表的任务；重复调用不会重复注册。"""
    # SuperMimic使用独立工厂及转移协议，不参与PPO注册表装配。
    import_module(".locomotion.go2_ppo.config", __name__)


__all__ = ["register_tasks"]
