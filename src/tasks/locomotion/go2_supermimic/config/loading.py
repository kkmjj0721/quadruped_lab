# Copyright (c) 2026-2027 zh
"""
内容：
    统一读取训练／播放配置快照、覆盖声明字段并验证声明字段。
"""

from copy import deepcopy
import json
import math
from pathlib import Path


def _merge_mapping(defaults: dict, values: dict) -> dict:
    result = deepcopy(defaults)
    for name, value in values.items():
        if isinstance(result.get(name), dict) and isinstance(value, dict):
            result[name] = _merge_mapping(result[name], value)
        else:
            result[name] = deepcopy(value)
    return result


def apply_settings(target, values: dict, prefix: str = "", *, replace_mappings: bool = False) -> None:
    """只覆盖已声明配置字段；字典参数允许局部覆盖和自定义键。"""
    if not isinstance(values, dict):
        raise ValueError(f"Configuration section {prefix or 'root'} must be an object")
    for name, value in values.items():
        path = f"{prefix}.{name}" if prefix else name
        if name.startswith("_") or not hasattr(target, name):
            raise ValueError(f"Unknown task configuration field: {path}")
        current = getattr(target, name)
        # 兼容既有BaseConfig快照中的空方法元数据，不覆盖实际实例方法。
        if callable(current):
            if not prefix and name in ("init_member_classes", "to_dict") and value == {}:
                continue
            raise ValueError(f"Task configuration field is not writable: {path}")
        if hasattr(current, "__dict__"):
            apply_settings(current, value, path, replace_mappings=replace_mappings)
            continue
        if isinstance(current, dict):
            if not isinstance(value, dict):
                raise ValueError(f"Task configuration field {path} must be an object")
            value = deepcopy(value) if replace_mappings else _merge_mapping(current, value)
        elif isinstance(current, bool):
            if not isinstance(value, bool):
                raise ValueError(f"Task configuration field {path} must be a boolean")
        elif isinstance(current, int):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Task configuration field {path} must be an integer")
        elif isinstance(current, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Task configuration field {path} must be finite numeric data")
        elif isinstance(current, str):
            if not isinstance(value, str):
                raise ValueError(f"Task configuration field {path} must be a string")
        elif isinstance(current, (tuple, list)):
            if not isinstance(value, (tuple, list)):
                raise ValueError(f"Task configuration field {path} must be an array")
            value = type(current)(value)
        setattr(target, name, deepcopy(value))


def read_settings(path: Path) -> tuple[dict, bool]:
    if not path.is_file():
        raise FileNotFoundError(f"Task configuration file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        # JSON解析器默认接受NaN／Infinity；配置快照只接受有限JSON数值。
        json.dumps(payload, allow_nan=False)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"Invalid task configuration JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("Task configuration JSON must be an object")
    settings = payload.get("config", payload)
    if not isinstance(settings, dict):
        raise ValueError("Task configuration snapshot must contain an object in config")
    return settings, "config" in payload


def configuration_for_settings(settings: dict | None = None):
    """只接受当前单 MLP 配置；未声明算法时使用同一默认值。"""
    from src.config.go2.go2_reference_supermimic_config import Go2ReferenceCfg

    settings = settings or {}
    algorithm = settings.get("algorithm", "single_mlp")
    if algorithm != "single_mlp":
        raise ValueError(f"Unsupported SuperMimic algorithm: {algorithm}")
    return Go2ReferenceCfg()
