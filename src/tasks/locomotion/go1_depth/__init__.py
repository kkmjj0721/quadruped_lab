# Copyright (c) 2026-2027 zh
"""内容：公开 Go1 纯深度 SuperMimic 的真实采集环境工厂。"""


def make_go1_depth_env(*args, **kwargs):
    from .environment import make_go1_depth_env as factory
    return factory(*args, **kwargs)


__all__ = ["make_go1_depth_env"]
