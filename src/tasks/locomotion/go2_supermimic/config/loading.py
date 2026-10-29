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
    settings = _migrate_sections(settings)
    wrapped = "config" in payload
    complete_snapshot = wrapped and {"spec_hash", "geometry_hash"}.issubset(payload)
    if complete_snapshot:
        # 缺少该段的历史快照属于45维足高代理任务；局部新JSON沿用新训练默认。
        settings.setdefault("experiment", {"contact_source": "height", "actor_phase": False})
    reward = settings.get("reward")
    if isinstance(reward, dict) and isinstance(reward.get("scales"), dict):
        scales = reward["scales"]
        if "foot_clearance" in scales:
            weight = scales["foot_clearance"]
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight != 0.0:
                raise ValueError("Foot-clearance rewards are not supported by the mainline task; use the frozen experiment workspace")
            if complete_snapshot:
                # 已训练无足高实验的零项不参与数学定义，清理后仍保留原任务指纹。
                del scales["foot_clearance"]
                reward.pop("clearance", None)
        if complete_snapshot:
            # 完整旧快照没有站立速度惩罚时，必须显式保留关闭状态。
            scales.setdefault("stand_still_joint_vel_l2", 0.0)
    return settings, wrapped


def _migrate_sections(settings: dict) -> dict:
    """把重构前的``single``算法段并回``algo``，使旧快照仍可续训与播放。

    两段同时存在时属于手工合并的异常快照，以当前代码写出的``algo``为准，
    旧段只作为缺省值，避免同一份快照被解释成两套预算。
    """
    if "single" not in settings:
        return settings
    migrated = deepcopy(settings)
    legacy = migrated.pop("single")
    if not isinstance(legacy, dict):
        raise ValueError("Task configuration section single must be an object")
    current = migrated.get("algo")
    if current is not None and not isinstance(current, dict):
        raise ValueError("Task configuration section algo must be an object")
    migrated["algo"] = _merge_mapping(legacy, current or {})
    return migrated


def configuration_for_settings(settings: dict | None = None):
    """只接受当前混合专家历史策略配置；未声明算法时使用同一默认值。"""
    from src.config.go2.go2_reference_supermimic_config import Go2ReferenceCfg

    settings = settings or {}
    algorithm = settings.get("algorithm", "moe_history")
    if algorithm != "moe_history":
        raise ValueError(f"Unsupported SuperMimic algorithm: {algorithm}")
    return Go2ReferenceCfg()
