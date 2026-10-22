# -*- coding: utf-8 -*-
"""人格内核配置加载与初始状态构建。

配置是版本化工程配置（configs/personality_core.json）：所有数值均为实验初值，
按 docs/personality_core/参数校准指南.md 校准后锁定 config revision。
本模块不包含运行算法，只负责读取、校验和把配置构造成初始 PersonalityState。
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from gwen_agent.utils import PERSONALITY_CORE_CONFIG_PATH, PERSONALITY_CORE_DB_PATH
from .state import (
    CAPS_CATEGORIES,
    CAPS_SCOPES,
    REGULATION_STRATEGIES,
    SELF_REPRESENTATION_MODES,
    AffectState,
    AttachmentState,
    CapsState,
    CapsUnit,
    IfThenSignature,
    PersonalityState,
    SelfRepresentation,
    TemperamentEAS,
    assert_valid_state,
)

REQUIRED_SECTIONS = (
    "config_version",
    "engine",
    "affect",
    "temperament_eas",
    "attachment",
    "self_representations",
    "situation_features",
    "caps",
    "expression",
)


def validate_config(config: dict) -> list:
    """返回配置结构违规列表；空列表表示合法。"""
    issues = []
    for section in REQUIRED_SECTIONS:
        if section not in config:
            issues.append(f"缺少配置段: {section}")
    if "config_version" in config and not config["config_version"]:
        issues.append("config_version 不能为空")
    affect = config.get("affect", {})
    for axis in ("valence", "arousal"):
        bounds = affect.get("bounds", {}).get(axis)
        if bounds is not None and not (len(bounds) == 2 and bounds[0] < bounds[1]):
            issues.append(f"affect.bounds.{axis} 非法: {bounds}")
    for feature_name, feature in config.get("situation_features", {}).items():
        f_range = feature.get("range")
        if f_range is not None and not (len(f_range) == 2 and f_range[0] <= f_range[1]):
            issues.append(f"situation_features.{feature_name}.range 非法")
        if f_range is not None:
            default = feature.get("default", 0.0)
            if not f_range[0] <= default <= f_range[1]:
                issues.append(f"situation_features.{feature_name}.default 超出 range")
        lexicon = feature.get("lexicon", {})
        if not isinstance(lexicon, dict) or any(
            not isinstance(words, list) for words in lexicon.values()
        ):
            issues.append(f"situation_features.{feature_name}.lexicon 必须是词表对象")

    feature_ids = set(config.get("situation_features", {}))
    caps = config.get("caps", {})
    units = caps.get("units", [])
    unit_ids = [unit.get("unit_id") for unit in units]
    _append_duplicate_issues(issues, "CAPS unit_id", unit_ids)
    for unit in units:
        unit_id = unit.get("unit_id", "<missing>")
        if unit.get("category") not in CAPS_CATEGORIES - {"if_then_signatures"}:
            issues.append(f"CAPS {unit_id} category 非法: {unit.get('category')}")
        feature_ref = unit.get("feature_ref")
        if feature_ref and feature_ref not in feature_ids:
            issues.append(f"CAPS {unit_id} 引用了未知特征: {feature_ref}")
        if unit.get("scope", "global") not in CAPS_SCOPES:
            issues.append(f"CAPS {unit_id} scope 非法: {unit.get('scope')}")
        if unit.get("subject_ref") is not None:
            issues.append(f"CAPS {unit_id} 配置模板不应指定 subject_ref")

    intents = caps.get("intents", [])
    intent_ids = [intent.get("intent_id") for intent in intents]
    _append_duplicate_issues(issues, "intent_id", intent_ids)
    intent_id_set = set(intent_ids)
    unit_id_set = set(unit_ids)
    for intent in intents:
        intent_id = intent.get("intent_id", "<missing>")
        for unit_id in intent.get("state_weights", {}):
            if unit_id not in unit_id_set:
                issues.append(f"意图 {intent_id} 引用了未知 CAPS 单元: {unit_id}")
        _validate_feature_refs(
            issues,
            f"意图 {intent_id}",
            intent.get("situation_weights", {}),
            feature_ids,
        )

    for gate in caps.get("hard_gates", []):
        if gate.get("intent") not in intent_id_set:
            issues.append(f"硬门控引用了未知意图: {gate.get('intent')}")
        _validate_condition(issues, "硬门控", gate.get("when", ""), feature_ids)

    strategies = caps.get("regulation_strategies", [])
    strategy_ids = [strategy.get("strategy_id") for strategy in strategies]
    _append_duplicate_issues(issues, "strategy_id", strategy_ids)
    for strategy in strategies:
        strategy_id = strategy.get("strategy_id", "<missing>")
        if strategy_id not in REGULATION_STRATEGIES:
            issues.append(f"调节策略 ID 非法: {strategy_id}")
        _validate_condition(
            issues,
            f"调节策略 {strategy_id}",
            strategy.get("eligible_when", ""),
            feature_ids,
        )
        _validate_feature_refs(
            issues,
            f"调节策略 {strategy_id}",
            strategy.get("situation_weights", {}),
            feature_ids,
        )
        for unit_id in strategy.get("state_weights", {}):
            if unit_id not in unit_id_set:
                issues.append(f"调节策略 {strategy_id} 引用了未知 CAPS 单元: {unit_id}")

    representation_ids = [
        item.get("representation_id") for item in config.get("self_representations", [])
    ]
    _append_duplicate_issues(issues, "representation_id", representation_ids)
    for item in config.get("self_representations", []):
        representation_id = item.get("representation_id", "<missing>")
        if item.get("mode") not in SELF_REPRESENTATION_MODES:
            issues.append(f"自我表征 {representation_id} mode 非法")
        for condition in item.get("activate_when", []):
            feature_ref = condition.get("feature")
            if feature_ref not in feature_ids:
                issues.append(
                    f"自我表征 {representation_id} 引用了未知特征: {feature_ref}"
                )

    signatures = caps.get("if_then_signatures", [])
    signature_ids = [item.get("signature_id") for item in signatures]
    _append_duplicate_issues(issues, "signature_id", signature_ids)
    for signature in signatures:
        signature_id = signature.get("signature_id", "<missing>")
        if signature.get("expected_intent") not in intent_id_set:
            issues.append(f"行为签名 {signature_id} 引用了未知意图")
        for feature_ref in signature.get("feature_refs", []):
            if feature_ref not in feature_ids:
                issues.append(f"行为签名 {signature_id} 引用了未知特征: {feature_ref}")
        for condition in signature.get("conditions", []):
            if condition.get("feature") not in feature_ids:
                issues.append(f"行为签名 {signature_id} 条件引用了未知特征")

    emotion_map = config.get("expression", {}).get("emotion_map", [])
    for index, rule in enumerate(emotion_map[:-1]):
        if set(rule) == {"emotion"}:
            issues.append(f"expression.emotion_map[{index}] 提前出现兜底规则")
    return issues


def _append_duplicate_issues(issues: list, label: str, values: list) -> None:
    seen = set()
    for value in values:
        if not value:
            issues.append(f"{label} 不能为空")
        elif value in seen:
            issues.append(f"{label} 重复: {value}")
        seen.add(value)


def _validate_feature_refs(
    issues: list, label: str, weights: dict, feature_ids: set
) -> None:
    for feature_ref in weights:
        if feature_ref not in feature_ids:
            issues.append(f"{label} 引用了未知特征: {feature_ref}")


def _validate_condition(
    issues: list, label: str, condition: str, feature_ids: set
) -> None:
    for feature_ref in re.findall(r"feature:([A-Za-z0-9_]+)", condition or ""):
        if feature_ref not in feature_ids:
            issues.append(f"{label} 条件引用了未知特征: {feature_ref}")


def load_config(path: Optional[str] = None) -> dict:
    """加载人格内核配置；path 缺省时使用项目标准路径。"""
    config_path = path or PERSONALITY_CORE_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    issues = validate_config(config)
    if issues:
        raise ValueError("personality_core.json 配置非法: " + "; ".join(issues))
    return config


def _build_caps(config: dict) -> CapsState:
    caps_cfg = config.get("caps", {})
    by_category = {category: [] for category in CAPS_CATEGORIES}
    for unit in caps_cfg.get("units", []):
        category = unit.get("category")
        if category not in by_category:
            raise ValueError(f"未知 CAPS 单元类别: {category}")
        by_category[category].append(CapsUnit(**unit))
    signatures = [
        IfThenSignature(**sig) for sig in caps_cfg.get("if_then_signatures", [])
    ]
    return CapsState(
        encodings=by_category["encodings"],
        expectancies_and_beliefs=by_category["expectancies_and_beliefs"],
        goals_and_values=by_category["goals_and_values"],
        competencies_and_regulatory_plans=by_category[
            "competencies_and_regulatory_plans"
        ],
        if_then_signatures=signatures,
    )


def build_genesis_state(config: dict, updated_at: float = 0.0) -> PersonalityState:
    """从配置构建初始 PersonalityState（只读项来自版本化角色设定）。"""
    affect_cfg = config.get("affect", {})
    baseline = affect_cfg.get("baseline", {"valence": 0.0, "arousal": 0.0})
    eas_cfg = config.get("temperament_eas", {})
    attachment_cfg = config.get("attachment", {})
    state = PersonalityState(
        affect=AffectState(
            valence=baseline.get("valence", 0.0), arousal=baseline.get("arousal", 0.0)
        ),
        temperament_eas=TemperamentEAS(
            emotionality=eas_cfg.get("emotionality", 0.0),
            activity=eas_cfg.get("activity", 0.0),
            sociability=eas_cfg.get("sociability", 0.0),
        ),
        attachment=AttachmentState(
            anxiety=attachment_cfg.get("anxiety", 0.0),
            avoidance=attachment_cfg.get("avoidance", 0.0),
        ),
        self_representations=[
            SelfRepresentation(**rep) for rep in config.get("self_representations", [])
        ],
        caps=_build_caps(config),
        updated_at=updated_at,
        config_version=config.get("config_version"),
    )
    assert_valid_state(state)
    return state


def config_db_path(config: dict, override: Optional[str] = None) -> str:
    """解析人格内核 SQLite 路径：优先显式 override，其次配置，缺省项目 data 目录。"""
    if override:
        db_path = override
    else:
        configured = (config.get("engine") or {}).get("db_path")
        db_path = configured or PERSONALITY_CORE_DB_PATH
    parent = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(parent, exist_ok=True)
    return db_path
