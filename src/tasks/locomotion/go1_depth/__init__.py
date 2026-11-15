# Copyright (c) 2026-2027 zh
"""内容：提供独立的 Go1 深度视觉 SuperMimic 任务工厂。"""


def make_go1_depth_env(*args, **kwargs):
    """延迟加载真实环境，避免导入任务时创建仿真资源。"""
    from .rl.sim_adapter import Go1DepthAdapter

    return Go1DepthAdapter(*args, **kwargs)


__all__ = ["make_go1_depth_env"]
