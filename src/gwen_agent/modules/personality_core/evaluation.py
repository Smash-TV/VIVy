# -*- coding: utf-8 -*-
"""人格内核 bootstrap 合同检查。

48 个手写情境只能检查当前规则是否满足显式合同，不能证明人格稳定性、泛化能力，
也不能替代 prompt 消融、留出集或真正的配对反事实评测。
"""

from __future__ import annotations

import json
import math
import pathlib

from .config import build_genesis_state
from .perception import perceive
from .runtime import reduce
from .state import InteractionEvent, PerceptState, new_uuid_urn

FEATURE_INTENT_SENSITIVITY = {
    "user_tone_valence": ("comfort", -1.0),
    "user_engagement": ("engage_social", 1.0),
    "user_intimacy": ("engage_social", 1.0),
    "social_pressure": ("assert_boundary", 1.0),
    "threat_or_harm": ("care", 1.0),
    "goal_relevance": ("assist", 1.0),
}

DIRECTION_THRESHOLD = 0.05


def _direction(value: float, threshold: float = DIRECTION_THRESHOLD) -> str:
    if value > threshold:
        return "positive"
    if value < -threshold:
        return "negative"
    return "neutral"


def _recovery_error(config: dict) -> float:
    """无刺激时恢复曲线与分析解的最大误差。"""
    state = build_genesis_state(config)
    state.affect.valence = 0.8
    state.affect.arousal = 0.6
    baseline = config["affect"]["baseline"]
    half_life = config["affect"]["half_life_seconds"]["valence"]
    percept = PerceptState(timestamp="t", situation="", items=[])
    max_error = 0.0
    for elapsed_seconds in (60, 300, 900, 1800, 3600, 7200):
        next_state, _ = reduce(state, percept, elapsed_seconds, config)
        analytic = baseline["valence"] + math.exp(
            -math.log(2) * elapsed_seconds / half_life
        ) * (0.8 - baseline["valence"])
        max_error = max(max_error, abs(next_state.affect.valence - analytic))
    return max_error


def _feature_sensitivity(config: dict) -> dict:
    """用合成特征检查 reducer 方向；这是实现合同，不是文本泛化证据。"""
    results = {}
    feature_config = config.get("situation_features", {})
    for feature, (intent, expected_sign) in FEATURE_INTENT_SENSITIVITY.items():
        low, high = feature_config[feature].get("range", [0.0, 1.0])
        low_percept = PerceptState(
            timestamp="t",
            situation="",
            items=[],
            situation_features={feature: low},
        )
        high_percept = PerceptState(
            timestamp="t",
            situation="",
            items=[],
            situation_features={feature: high},
        )
        genesis = build_genesis_state(config)
        _, low_trace = reduce(genesis, low_percept, 0.0, config)
        _, high_trace = reduce(genesis, high_percept, 0.0, config)
        delta = high_trace.candidate_scores.get(
            intent, 0.0
        ) - low_trace.candidate_scores.get(intent, 0.0)
        results[feature] = {
            "intent": intent,
            "score_delta_high_minus_low": delta,
            "expected_sign": expected_sign,
            "passed": delta * expected_sign > 0,
        }
    return results


def _signature_results(config: dict, rows: list) -> dict:
    def matches(signature, features):
        for condition in signature.get("conditions", []):
            value = features.get(condition["feature"], 0.0)
            if "min" in condition and value < condition["min"]:
                return False
            if "max" in condition and value > condition["max"]:
                return False
        return True

    results = {}
    for signature in config["caps"].get("if_then_signatures", []):
        matching_rows = [row for row in rows if matches(signature, row["features"])]
        hits = sum(
            row["intent"] == signature["expected_intent"] for row in matching_rows
        )
        results[signature["signature_id"]] = {
            "matched_scenarios": len(matching_rows),
            "hits": hits,
            "accuracy": hits / len(matching_rows) if matching_rows else None,
        }
    return results


def evaluate(config: dict, fixture_dir) -> dict:
    """执行 bootstrap 合同检查，并显式报告尚未完成的行为门禁。"""
    scenarios = sorted(
        path
        for path in pathlib.Path(fixture_dir).glob("*.json")
        if path.name != "manifest.json"
    )
    genesis = build_genesis_state(config)
    rows = []
    deterministic_hits = 0
    for path in scenarios:
        scenario = json.loads(path.read_text(encoding="utf-8"))
        event = InteractionEvent(
            event_id=new_uuid_urn(),
            source_id="eval",
            stream_id="eval",
            occurred_at="2026-08-21T10:00:00+08:00",
            recorded_at="2026-08-21T10:00:00+08:00",
            channel="system",
            kind="message",
            payload={"text": scenario["text"]},
        )
        percept = perceive(event, config)
        _, first_trace = reduce(genesis, percept, 0.0, config, event)
        _, second_trace = reduce(genesis, percept, 0.0, config, event)
        if (
            first_trace.candidate_scores == second_trace.candidate_scores
            and first_trace.selected_intent == second_trace.selected_intent
            and first_trace.selected_regulation == second_trace.selected_regulation
            and first_trace.decision.va_delta == second_trace.decision.va_delta
        ):
            deterministic_hits += 1
        rows.append(
            {
                "scenario": scenario,
                "features": percept.situation_features,
                "intent": first_trace.selected_intent,
                "regulation": first_trace.selected_regulation,
                "va": first_trace.decision.va_delta,
            }
        )

    contract_hits = {"valence": 0, "arousal": 0, "intent": 0, "regulation": 0}
    for row in rows:
        expected = row["scenario"]["expected"]
        contract_hits["valence"] += (
            _direction(row["va"]["valence"]) in expected["valence_direction"]
        )
        contract_hits["arousal"] += (
            _direction(row["va"]["arousal"]) in expected["arousal_direction"]
        )
        contract_hits["intent"] += row["intent"] == expected["dominant_intent"]
        contract_hits["regulation"] += row["regulation"] == expected["regulation"]

    groups = {}
    for row in rows:
        scenario = row["scenario"]
        groups.setdefault((scenario["feature"], scenario["condition"]), []).append(row)
    consistent_groups = sum(
        len({(row["intent"], row["regulation"]) for row in group}) == 1
        for group in groups.values()
    )
    total = len(rows)
    return {
        "scope": "bootstrap_contract_checks_only",
        "total_scenarios": total,
        "deterministic_consistency": {
            "hits": deterministic_hits,
            "total": total,
            "rate": deterministic_hits / total if total else None,
        },
        "scenario_contract_accuracy": {
            name: round(hits / total, 4) if total else None
            for name, hits in contract_hits.items()
        },
        "paraphrase_consistency": (
            round(consistent_groups / len(groups), 4) if groups else None
        ),
        "feature_sensitivity_contract": _feature_sensitivity(config),
        "state_recovery_error": _recovery_error(config),
        "if_then_signature_match": _signature_results(config, rows),
        "stage_gate": {
            "passed": False,
            "missing": [
                "prompt/persona 基线消融",
                "独立留出集",
                "共享 pair_id 的最小配对反事实",
                "人工盲评与跨模型复测",
            ],
        },
    }
