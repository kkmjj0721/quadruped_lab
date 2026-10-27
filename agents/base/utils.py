# Copyright (c) 2026-2027 zh
"""
内容：
    解析基础算法类与 actor、critic 使用的观测组。
"""

from __future__ import annotations

from copy import deepcopy
from importlib import import_module

from tensordict import TensorDict


def resolve_class(config: dict, default_class: type) -> tuple[type, dict]:
    """复制配置并解析短类名或本地模块的冒号路径。"""
    options = deepcopy(config)
    name = options.pop("class_name", default_class.__name__)
    if name == default_class.__name__:
        return default_class, options
    if not isinstance(name, str) or ":" not in name:
        raise ValueError(f"Expected {default_class.__name__} or a local module:class path; got {name!r}.")
    module_name, class_name = name.split(":", 1)
    if not module_name.startswith("agents."):
        raise ValueError(f"Algorithm components must be imported from agents; got {module_name!r}.")
    component = getattr(import_module(module_name), class_name)
    if not isinstance(component, type) or not issubclass(component, default_class):
        raise TypeError(f"{name!r} must be a subclass of {default_class.__name__}.")
    return component, options


def resolve_obs_groups(obs: TensorDict, obs_groups: dict | None = None) -> dict[str, list[str]]:
    """解析显式分组；缺省时优先同名组，再使用 policy 或 actor 组。"""
    groups = deepcopy(obs_groups or {})
    for name in ("actor", "critic"):
        if name not in groups:
            if name in obs:
                groups[name] = [name]
            elif "policy" in obs:
                groups[name] = ["policy"]
            elif name == "critic":
                groups[name] = list(groups["actor"])
            else:
                raise ValueError("Actor observations are missing; specify obs_groups explicitly.")
        if not isinstance(groups[name], (tuple, list)) or not groups[name]:
            raise ValueError(f"Observation set {name!r} must contain at least one group.")
        for key in groups[name]:
            if key not in obs:
                raise ValueError(f"Observation group {key!r} for {name!r} is missing.")
        groups[name] = list(groups[name])
    return groups
