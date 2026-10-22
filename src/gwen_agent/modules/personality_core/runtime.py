# -*- coding: utf-8 -*-
"""纯 reducer：PersonalityState + PerceptState + elapsed_seconds → 新状态 + DecisionTrace。

这是唯一人格写入口。确定性承诺：
  state_before + PerceptState + elapsed_time + config_version → 相同核心转移与候选分数。
LLM、记忆、知识图谱、外部评估与 TTS 标签都不能绕过本模块写人格状态。

v0 范围：Russell 效价-唤醒在线更新；CAPS 激活/意图/调节用无循环权重的可解释评分；
EAS、依恋、自我表征只读（阶段 4 前不更新）。动态贝叶斯/Kalman/LoRA/persona vector 均未引入。
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

from .perception import _eval_condition
from .state import (
    INTENT_DEFAULT,
    AffectState,
    CapsState,
    CapsUnit,
    Decision,
    DecisionTrace,
    IfThenSignature,
    InteractionEvent,
    PerceptState,
    PersonalityState,
    assert_valid_state,
    clip,
    new_uuid_urn,
    percept_to_dict,
)

# 派生输出用的实验初值权重（empathy 为本轮派生量，不持久化、不参与参数校准）
_EMPATHY_INTIMACY_WEIGHT = 0.4
_EMPATHY_THREAT_WEIGHT = 0.6
_EMPATHY_DISTRESS_WEIGHT = 0.5


def _half_life_decay(elapsed_seconds: float, half_life: float) -> float:
    if half_life <= 0:
        return 0.0
    return math.exp(-math.log(2) * max(elapsed_seconds, 0.0) / half_life)


def _axis_update(
    previous: float,
    baseline: float,
    drive: float,
    elapsed_seconds: float,
    half_life: float,
    affect_intensity: float,
    positive_sensitivity: float,
    negative_sensitivity: float,
    event_fusion_max: float,
    bounds,
) -> float:
    decay = _half_life_decay(elapsed_seconds, half_life)
    signed_sensitivity = positive_sensitivity if drive >= 0 else negative_sensitivity
    signal = math.tanh(affect_intensity * signed_sensitivity * drive)
    recovered = baseline + decay * (previous - baseline)
    lower_bound, upper_bound = bounds
    if signal >= 0:
        target = baseline + signal * (upper_bound - baseline)
    else:
        target = baseline + (-signal) * (lower_bound - baseline)
    fusion = min(clip(event_fusion_max, 0.0, 1.0), abs(signal))
    value = recovered + fusion * (target - recovered)
    return clip(value, lower_bound, upper_bound)


def _reduce_affect(
    state: PersonalityState, features: dict, elapsed_seconds: float, config: dict
) -> AffectState:
    affect_cfg = config.get("affect", {})
    baseline = affect_cfg.get("baseline", {"valence": 0.0, "arousal": 0.0})
    bounds = affect_cfg.get("bounds", {"valence": [-1.0, 1.0], "arousal": [-1.0, 1.0]})
    half_life = affect_cfg.get(
        "half_life_seconds", {"valence": 3600.0, "arousal": 1800.0}
    )
    intensity = affect_cfg.get("affect_intensity", 1.0)
    sensitivity = affect_cfg.get("sensitivity", {"positive": 1.0, "negative": 1.0})

    valence_drive = sum(
        feature.get("valence_weight", 0.0) * features.get(name, 0.0)
        for name, feature in config.get("situation_features", {}).items()
    )
    arousal_drive = sum(
        feature.get("arousal_weight", 0.0) * features.get(name, 0.0)
        for name, feature in config.get("situation_features", {}).items()
    )
    valence = _axis_update(
        state.affect.valence,
        baseline.get("valence", 0.0),
        valence_drive,
        elapsed_seconds,
        half_life.get("valence", 3600.0),
        intensity,
        sensitivity.get("positive", 1.0),
        sensitivity.get("negative", 1.0),
        affect_cfg.get("event_fusion_max", 0.75),
        bounds.get("valence", [-1.0, 1.0]),
    )
    arousal = _axis_update(
        state.affect.arousal,
        baseline.get("arousal", 0.0),
        arousal_drive,
        elapsed_seconds,
        half_life.get("arousal", 1800.0),
        intensity,
        sensitivity.get("positive", 1.0),
        sensitivity.get("negative", 1.0),
        affect_cfg.get("event_fusion_max", 0.75),
        bounds.get("arousal", [-1.0, 1.0]),
    )
    return AffectState(valence=valence, arousal=arousal)


def _updated_caps_unit(
    unit_cfg: dict,
    previous: Optional[CapsUnit],
    features: dict,
    elapsed_seconds: float,
    half_life: float,
    subject_ref: Optional[str] = None,
    updated_at: Optional[float] = None,
) -> CapsUnit:
    """更新一个 CAPS 单元；关系副本可使用自己的 elapsed 与更新时间。"""
    baseline = unit_cfg.get("baseline", 0.0)
    previous_activation = (
        previous.activation
        if previous is not None
        else unit_cfg.get("activation", baseline)
    )
    decay = _half_life_decay(elapsed_seconds, half_life)
    recovered = baseline + decay * (previous_activation - baseline)
    feature_ref = unit_cfg.get("feature_ref")
    input_weight = unit_cfg.get("input_weight", 0.0)
    activation = recovered
    if feature_ref:
        activation = clip(
            recovered + input_weight * features.get(feature_ref, 0.0),
            -1.0,
            1.0,
        )
    return CapsUnit(
        unit_id=unit_cfg["unit_id"],
        category=unit_cfg["category"],
        content=unit_cfg.get("content", ""),
        activation=activation,
        baseline=baseline,
        input_weight=input_weight,
        feature_ref=feature_ref,
        scope=unit_cfg.get("scope", "global"),
        subject_ref=subject_ref,
        updated_at=updated_at,
    )


def _reduce_caps(
    state: PersonalityState,
    features: dict,
    elapsed_seconds: float,
    config: dict,
    subject_ref: Optional[str] = None,
) -> CapsState:
    """更新全局 CAPS，并只更新当前关系副本；其他关系的时钟保持不动。"""
    caps_cfg = config.get("caps", {})
    half_life = caps_cfg.get("decay_half_life_seconds", 3600.0)
    current_timestamp = state.updated_at + max(elapsed_seconds, 0.0)
    subject_key = None if subject_ref is None else str(subject_ref)
    existing_units = {
        (unit.unit_id, unit.subject_ref): unit for unit in state.caps.all_units()
    }
    new_units = []

    for unit_cfg in caps_cfg.get("units", []):
        unit_id = unit_cfg["unit_id"]
        scope = unit_cfg.get("scope", "global")
        template = existing_units.get((unit_id, None))
        if scope != "relationship":
            new_units.append(
                _updated_caps_unit(
                    unit_cfg,
                    template,
                    features,
                    elapsed_seconds,
                    half_life,
                    updated_at=current_timestamp,
                )
            )
            continue

        new_units.append(
            _updated_caps_unit(
                unit_cfg,
                None,
                {},
                0.0,
                half_life,
                updated_at=None,
            )
        )
        clones = {
            str(unit.subject_ref): replace(
                unit,
                category=unit_cfg["category"],
                content=unit_cfg.get("content", ""),
                baseline=unit_cfg.get("baseline", 0.0),
                input_weight=unit_cfg.get("input_weight", 0.0),
                feature_ref=unit_cfg.get("feature_ref"),
                scope="relationship",
            )
            for unit in state.caps.all_units()
            if unit.unit_id == unit_id and unit.subject_ref is not None
        }
        if subject_key is not None:
            previous = clones.get(subject_key)
            last_updated = (
                previous.updated_at
                if previous is not None and previous.updated_at is not None
                else state.updated_at
            )
            relationship_elapsed = max(current_timestamp - last_updated, 0.0)
            clones[subject_key] = _updated_caps_unit(
                unit_cfg,
                previous,
                features,
                relationship_elapsed,
                half_life,
                subject_ref=subject_key,
                updated_at=current_timestamp,
            )
        new_units.extend(clones[key] for key in sorted(clones))

    buckets = {category: [] for category in CapsState.__dataclass_fields__}
    for unit in new_units:
        buckets[unit.category].append(unit)
    return CapsState(
        encodings=buckets["encodings"],
        expectancies_and_beliefs=buckets["expectancies_and_beliefs"],
        goals_and_values=buckets["goals_and_values"],
        competencies_and_regulatory_plans=buckets["competencies_and_regulatory_plans"],
        if_then_signatures=[
            IfThenSignature(**item) for item in caps_cfg.get("if_then_signatures", [])
        ],
    )


def _score_intents(
    state: PersonalityState,
    features: dict,
    config: dict,
    event: Optional[InteractionEvent],
):
    """行动意图统一评分：bias + state_weight*激活单元 + situation_weight*情境特征。

    选择顺序（10 §4.3）：硬门控排除 → CAPS 价值性评分 → 生产选最高分并记录全部候选分。
    """
    caps_cfg = config.get("caps", {})
    unit_map = state.caps.unit_map(
        event.relationship_ref if event is not None else None
    )
    scores = {}
    gates = []
    blocked = set()
    for gate in caps_cfg.get("hard_gates", []):
        if _eval_condition(gate.get("when", ""), features, event):
            blocked.add(gate.get("intent"))
            gates.append(
                {
                    "intent": gate.get("intent"),
                    "blocked": True,
                    "reason": gate.get("reason", ""),
                }
            )
    for intent in caps_cfg.get("intents", []):
        intent_id = intent["intent_id"]
        score = intent.get("bias", 0.0)
        for unit_id, weight in intent.get("state_weights", {}).items():
            unit = unit_map.get(unit_id)
            if unit is not None:
                score += weight * unit.activation
        for feature_name, weight in intent.get("situation_weights", {}).items():
            score += weight * features.get(feature_name, 0.0)
        scores[intent_id] = score
    allowed = [
        (intent_id, score)
        for intent_id, score in scores.items()
        if intent_id not in blocked
    ]
    selected = max(allowed, key=lambda item: item[1])[0] if allowed else INTENT_DEFAULT
    return scores, gates, selected


def _select_regulation(
    state: PersonalityState,
    features: dict,
    config: dict,
    event: Optional[InteractionEvent],
) -> Optional[str]:
    """四种调节策略：先做资格门控，再由 CAPS 做价值性评分（10 §3/§4.5）。"""
    caps_cfg = config.get("caps", {})
    unit_map = state.caps.unit_map(
        event.relationship_ref if event is not None else None
    )
    eligible = []
    for strategy in caps_cfg.get("regulation_strategies", []):
        if not _eval_condition(strategy.get("eligible_when", ""), features, event):
            continue
        score = strategy.get("bias", 0.0)
        for unit_id, weight in strategy.get("state_weights", {}).items():
            unit = unit_map.get(unit_id)
            if unit is not None:
                score += weight * unit.activation
        for feature_name, weight in strategy.get("situation_weights", {}).items():
            score += weight * features.get(feature_name, 0.0)
        eligible.append((strategy["strategy_id"], score))
    if not eligible:
        return None
    return max(eligible, key=lambda item: item[1])[0]


def _derive_working_self(state: PersonalityState, features: dict) -> list:
    """工作自我概念：本轮被激活的当前自我图式子集（只派生，不持久化）。

    Markus 边界：hoped/feared 是可能自我，不冒充当前图式；没有情境条件
    activate_when 的稳定图式也不自动进入本轮工作自我。
    """
    activated = []
    for rep in state.self_representations:
        if rep.mode != "current" or not rep.activate_when:
            continue
        matched = True
        for cond in rep.activate_when:
            value = features.get(cond.get("feature"), 0.0)
            if "min" in cond and value < cond["min"]:
                matched = False
                break
            if "max" in cond and value > cond["max"]:
                matched = False
                break
        if matched:
            activated.append(rep.representation_id)
    return activated


def _derive_empathy(features: dict) -> float:
    if features.get("assistant_directed_attack", 0.0) >= 0.5:
        distress = 0.0
    else:
        distress = max(0.0, -features.get("user_tone_valence", 0.0))
    value = (
        _EMPATHY_INTIMACY_WEIGHT * features.get("user_intimacy", 0.0)
        + _EMPATHY_THREAT_WEIGHT * features.get("threat_or_harm", 0.0)
        + _EMPATHY_DISTRESS_WEIGHT * distress
    )
    return clip(value, 0.0, 1.0)


def _derive_attention_hint(features: dict) -> Optional[str]:
    if features.get("threat_or_harm", 0.0) >= 0.5:
        return "safety"
    if features.get("relationship_confirmation_request", 0.0) >= 0.5:
        return "relationship"
    if features.get("user_intimacy", 0.0) >= 0.5:
        return "relationship"
    if features.get("goal_relevance", 0.0) >= 0.6:
        return "knowledge"
    return None


def _derive_emotion(
    affect: AffectState, features: dict, selected_intent: str, config: dict
) -> str:
    """离散情绪标签只从连续内核与情境/意图派生（输出映射，禁止反向写 VA）。

    规则顺序：先按 VA+威胁匹配，再按本轮意图补足（如 playful→happy/surprised、
    comfort→sad、care→fearful、assert_boundary→angry），最后 neutral。
    """
    emotion_map = config.get("expression", {}).get(
        "emotion_map", [{"emotion": "neutral"}]
    )
    for rule in emotion_map:
        if any(
            features.get(name, 0.0) < minimum
            for name, minimum in rule.get("feature_mins", {}).items()
        ):
            continue
        if any(
            features.get(name, 0.0) > maximum
            for name, maximum in rule.get("feature_maxs", {}).items()
        ):
            continue
        if "valence_min" in rule and affect.valence < rule["valence_min"]:
            continue
        if "valence_max" in rule and affect.valence > rule["valence_max"]:
            continue
        if "arousal_min" in rule and affect.arousal < rule["arousal_min"]:
            continue
        if "arousal_max" in rule and affect.arousal > rule["arousal_max"]:
            continue
        if (
            "threat_min" in rule
            and features.get("threat_or_harm", 0.0) < rule["threat_min"]
        ):
            continue
        if (
            "threat_max" in rule
            and features.get("threat_or_harm", 0.0) > rule["threat_max"]
        ):
            continue
        if "intent" in rule and rule["intent"] != selected_intent:
            continue
        return rule.get("emotion", "neutral")
    return "neutral"


def _derive_constraints(
    previous_state: PersonalityState,
    current_state: PersonalityState,
    features: dict,
    subject_ref: Optional[str],
    selected_intent: str,
    selected_regulation: Optional[str],
    config: dict,
) -> list:
    """把当前关系余波与本轮社交行为转成短小、可审计的表达边界。"""
    expression = config.get("expression", {})
    constraints = list(expression.get("always_constraints", []))
    dynamic_constraints = expression.get("dynamic_constraints", {})
    thresholds = expression.get("dynamic_thresholds", {})

    expectation = current_state.caps.unit_map(subject_ref).get("u_expect_user_kind")
    relationship_hurt = (
        expectation is not None
        and expectation.activation
        <= thresholds.get("repair_pending_relationship_expectation_max", 0.3)
    )
    negative_residue = min(
        previous_state.affect.valence, current_state.affect.valence
    ) <= thresholds.get("repair_pending_valence_max", 0.0)
    if (
        features.get("repair_attempt", 0.0) >= 0.5
        and relationship_hurt
        and negative_residue
    ):
        rule = dynamic_constraints.get("repair_pending")
        if rule:
            constraints.append(rule)

    if features.get("relationship_confirmation_request", 0.0) >= 0.5:
        rule = dynamic_constraints.get("relationship_confirmation")
        if rule:
            constraints.append(rule)

    if selected_regulation:
        rule = expression.get("regulation_constraints", {}).get(selected_regulation)
        if rule:
            constraints.append(rule)
    intent_rule = expression.get("intent_constraints", {}).get(selected_intent)
    if intent_rule:
        constraints.append(intent_rule)
    return constraints


def _intent_label(selected_intent: str, config: dict) -> str:
    for intent in config.get("caps", {}).get("intents", []):
        if intent["intent_id"] == selected_intent:
            return intent.get("label", selected_intent)
    return selected_intent


def reduce(
    state: PersonalityState,
    percept: PerceptState,
    elapsed_seconds: float,
    config: dict,
    event: Optional[InteractionEvent] = None,
) -> tuple:
    """纯函数：返回 (新 PersonalityState, DecisionTrace)。不写库、不调 LLM。"""
    features = percept.situation_features or {}
    subject_ref = event.relationship_ref if event is not None else None
    affect = _reduce_affect(state, features, elapsed_seconds, config)
    caps = _reduce_caps(state, features, elapsed_seconds, config, subject_ref)
    new_state = PersonalityState(
        affect=affect,
        temperament_eas=state.temperament_eas,
        attachment=state.attachment,
        self_representations=state.self_representations,
        caps=caps,
        updated_at=state.updated_at + max(elapsed_seconds, 0.0),
        config_version=config.get("config_version"),
    )
    candidate_scores, gates, selected_intent = _score_intents(
        new_state, features, config, event
    )
    selected_regulation = _select_regulation(new_state, features, config, event)
    working_self_refs = _derive_working_self(new_state, features)
    empathy_intensity = _derive_empathy(features)
    attention_hint = _derive_attention_hint(features)
    expressivity = config.get("affect", {}).get("expressivity", 0.5)
    expression_intensity = clip(
        expressivity
        * (0.4 + 0.3 * abs(affect.valence) + 0.3 * max(affect.arousal, 0.0)),
        0.0,
        1.0,
    )
    decision = Decision(
        working_self_refs=working_self_refs,
        empathy_intensity=empathy_intensity,
        attention_hint=attention_hint,
        va_delta={
            "valence": affect.valence - state.affect.valence,
            "arousal": affect.arousal - state.affect.arousal,
        },
        expression_intensity=expression_intensity,
        expression_constraints=_derive_constraints(
            state,
            new_state,
            features,
            subject_ref,
            selected_intent,
            selected_regulation,
            config,
        ),
        emotion_label=_derive_emotion(affect, features, selected_intent, config),
    )
    assert_valid_state(new_state)
    trace = DecisionTrace(
        trace_id=new_uuid_urn(),
        event_id=event.event_id if event else None,
        state_revision_before=None,
        state_revision_after=None,
        percept_state=percept_to_dict(percept),
        candidate_scores=candidate_scores,
        gates=gates,
        selected_intent=selected_intent,
        selected_regulation=selected_regulation,
        decision=decision,
        config_version=config.get("config_version"),
    )
    return new_state, trace


def decision_text(trace: DecisionTrace, config: dict) -> str:
    """把本轮决策压缩成给 LLM 的短小表达约束（不塞整份状态）。"""
    intensity = trace.decision.expression_intensity
    if intensity < 0.35:
        intensity_label = "轻"
    elif intensity < 0.65:
        intensity_label = "中等"
    else:
        intensity_label = "强"
    parts = [f"本轮情绪底色：{trace.decision.emotion_label}（{intensity_label}强度）"]
    if trace.selected_intent != INTENT_DEFAULT:
        parts.append(f"本轮表达倾向：{_intent_label(trace.selected_intent, config)}")
    parts.extend(trace.decision.expression_constraints)
    return "；".join(parts) if parts else ""
