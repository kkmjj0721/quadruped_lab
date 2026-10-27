# Copyright (c) 2026-2027 zh
"""
内容：
    提供独立Go2 SuperMimic任务工厂，不向PPO专用注册入口伪装算法类型。
"""


def make_go2_supermimic_env(*args, **kwargs):
    from .rl import make_go2_supermimic_env as factory
    return factory(*args, **kwargs)


def make_go2_supermimic_runner(*args, **kwargs):
    from .rl import make_go2_supermimic_runner as factory
    return factory(*args, **kwargs)


__all__ = ["make_go2_supermimic_env", "make_go2_supermimic_runner"]
