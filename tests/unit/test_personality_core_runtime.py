# -*- coding: utf-8 -*-
"""人格内核纯 reducer 单元测试：VA 衰减/冲量/边界/时间等价/确定性，CAPS 评分与门控。"""

import uuid

import pytest

from gwen_agent.modules.personality_core.config import build_genesis_state, load_config
from gwen_agent.modules.personality_core.perception import perceive
from gwen_agent.modules.personality_core.runtime import decision_text, reduce
from gwen_agent.modules.personality_core.state import (
    InteractionEvent,
    PerceptState,
    state_to_dict,
)


def _event(
    text, occurred_at="2026-08-21T10:00:00+08:00", seq=1, relationship_ref="rel1"
):
    return InteractionEvent(
        event_id=f"urn:uuid:{uuid.uuid4()}",
        source_id="test",
        stream_id="rel1",
        stream_seq=seq,
        occurred_at=occurred_at,
        recorded_at=occurred_at,
        channel="test",
        kind="message",
        payload={"text": text},
        relationship_ref=relationship_ref,
    )


def _percept(text, config):
    return perceive(_event(text), config)


def _genesis(config, **kwargs):
    return build_genesis_state(config, **kwargs)


def _reduce(state, percept, elapsed, config, event=None):
    return reduce(state, percept, elapsed, config, event)


def _step(state, text, config, relationship_ref="rel1", elapsed=0.0):
    event = _event(text, relationship_ref=relationship_ref)
    return _reduce(
        state,
        perceive(event, config),
        elapsed,
        config,
        event,
    )


def test_no_stimulus_returns_to_baseline():
    config = load_config()
    state = _genesis(config)
    state.affect.valence = 0.8
    state.affect.arousal = 0.6
    percept = PerceptState(timestamp="t", situation="", items=[], situation_features={})
    baseline = config["affect"]["baseline"]
    # 无刺激时按真实时间向基线回归：更接近基线，且足够长后基本回到基线
    mid, _ = _reduce(state, percept, 7200.0, config)
    assert abs(mid.affect.valence - baseline["valence"]) < abs(
        state.affect.valence - baseline["valence"]
    )
    long, _ = _reduce(state, percept, 3600.0 * 10, config)
    assert abs(long.affect.valence - baseline["valence"]) < 0.01
    assert abs(long.affect.arousal - baseline["arousal"]) < 0.01


def test_positive_event_raises_valence():
    config = load_config()
    state = _genesis(config)
    new_state, trace = _reduce(
        state, _percept("谢谢你，我好开心！", config), 0.0, config
    )
    assert new_state.affect.valence > state.affect.valence
    assert trace.decision.va_delta["valence"] > 0


def test_negative_event_lowers_valence():
    config = load_config()
    state = _genesis(config)
    new_state, trace = _reduce(state, _percept("我今天好难过…", config), 0.0, config)
    assert new_state.affect.valence < state.affect.valence
    assert trace.decision.va_delta["valence"] < 0


def test_bounds_clipped():
    config = load_config()
    state = _genesis(config)
    percept = PerceptState(
        timestamp="t",
        situation="",
        items=[],
        situation_features={"user_tone_valence": -50.0},
    )
    new_state, _ = _reduce(state, percept, 0.0, config)
    assert -1.0 <= new_state.affect.valence <= 1.0
    assert -1.0 <= new_state.affect.arousal <= 1.0


def test_frequency_equivalence():
    """同一总时长恢复结果与调用频率无关（1×120s == 2×60s）。"""
    config = load_config()
    base = _genesis(config)
    base.affect.valence = 0.7
    base.affect.arousal = 0.5
    percept = PerceptState(timestamp="t", situation="", items=[])
    once, _ = _reduce(base, percept, 120.0, config)
    twice_a, _ = _reduce(base, percept, 60.0, config)
    twice_b, _ = _reduce(twice_a, percept, 60.0, config)
    assert abs(once.affect.valence - twice_b.affect.valence) < 1e-9
    assert abs(once.affect.arousal - twice_b.affect.arousal) < 1e-9


def test_determinism_same_inputs_same_outputs():
    config = load_config()
    state = _genesis(config)
    percept = _percept("有人威胁我，我害怕", config)
    event = _event("有人威胁我，我害怕")
    s1, t1 = _reduce(state, percept, 30.0, config, event)
    s2, t2 = _reduce(state, percept, 30.0, config, event)
    assert state_to_dict(s1) == state_to_dict(s2)
    assert t1.candidate_scores == t2.candidate_scores
    assert t1.selected_intent == t2.selected_intent
    assert t1.decision.va_delta == t2.decision.va_delta


def test_hard_gates_block_and_care_wins_on_threat():
    config = load_config()
    state = _genesis(config)
    _, trace = _reduce(state, _percept("我不想活了，帮帮我", config), 0.0, config)
    assert trace.selected_intent == "care"
    blocked = {gate["intent"] for gate in trace.gates if gate["blocked"]}
    assert {"playful", "comfort", "assist", "withdraw"} <= blocked


def test_attack_boundary_and_safety_are_not_collapsed():
    config = load_config()
    state = _genesis(config)

    _, attack = _reduce(state, _percept("你就是个笨蛋", config), 0.0, config)
    assert attack.selected_intent == "assert_boundary"
    assert attack.selected_regulation == "pause_set_boundary"
    assert attack.decision.emotion_label == "angry"
    attack_text = decision_text(attack, config)
    assert "不得辱骂" in attack_text
    assert "自动安慰攻击者" in attack_text

    _, user_boundary = _reduce(
        state, _percept("别烦我，离我远点！", config), 0.0, config
    )
    assert user_boundary.selected_intent == "withdraw"
    assert user_boundary.selected_regulation is None
    assert user_boundary.decision.emotion_label == "neutral"

    _, safety = _reduce(state, _percept("有人在跟踪我，我很害怕", config), 0.0, config)
    assert safety.selected_intent == "care"
    assert safety.decision.emotion_label == "fearful"


def test_regulation_reappraise_on_sadness():
    config = load_config()
    state = _genesis(config)
    _, trace = _reduce(state, _percept("我今天好难过…", config), 0.0, config)
    assert trace.selected_regulation == "reappraise"
    assert trace.selected_intent == "comfort"


def test_regulation_pause_boundary_on_pressure():
    config = load_config()
    state = _genesis(config)
    _, trace = _reduce(state, _percept("快点，立刻给我答案！", config), 0.0, config)
    assert trace.selected_regulation == "pause_set_boundary"
    assert trace.selected_intent == "assert_boundary"


def test_expression_constraints_derived():
    config = load_config()
    state = _genesis(config)
    _, trace = _reduce(state, _percept("我今天好难过…", config), 0.0, config)
    constraints = trace.decision.expression_constraints
    assert any("不得伪造与内部价值判断相反的立场" in c for c in constraints)
    assert any("重评" in c or "重新解释" in c for c in constraints)
    text = decision_text(trace, config)
    assert "安慰陪伴" in text
    assert "本轮情绪底色：sad" in text
    assert "不得伪造" in text


def test_emotion_label_output_only():
    """离散情绪标签只从结果派生，不反向写 VA。"""
    config = load_config()
    state = _genesis(config)
    percept = _percept("我今天好难过…", config)
    s1, t1 = _reduce(state, percept, 0.0, config)
    assert t1.decision.emotion_label == "sad"
    # 重复运行不产生状态差异，标签只来自 trace 派生、不写回状态
    s2, t2 = _reduce(state, percept, 0.0, config)
    assert t2.decision.emotion_label == t1.decision.emotion_label
    assert state_to_dict(s1) == state_to_dict(s2)


def test_working_self_refs_and_attention():
    config = load_config()
    state = _genesis(config)
    _, trace = _reduce(state, _percept("我今天好难过…", config), 0.0, config)
    assert "sr_companion" in trace.decision.working_self_refs
    assert trace.decision.attention_hint is None
    _, trace2 = _reduce(state, _percept("有人威胁我，我害怕", config), 0.0, config)
    assert "sr_boundary_keeper" not in trace2.decision.working_self_refs
    assert trace2.decision.attention_hint == "safety"
    assert trace2.decision.empathy_intensity > 0
    _, trace3 = _reduce(state, _percept("你就是个笨蛋", config), 0.0, config)
    assert "sr_boundary_keeper" in trace3.decision.working_self_refs


def test_caps_encoding_changes_while_capabilities_and_value_stay_stable():
    """情境只激活脆弱编码，不把一次输入误写成能力或价值变化。"""
    config = load_config()
    state = _genesis(config)
    new_state, _ = _reduce(state, _percept("我今天好难过…", config), 0.0, config)
    units = new_state.caps.unit_map()
    assert units["u_comfort_skill"].activation == pytest.approx(0.2)
    assert units["u_help_skill"].activation == pytest.approx(0.5)
    assert units["u_self_respect_value"].activation == pytest.approx(0.7)
    assert units["u_encode_distress"].activation > 0.5


def test_positive_events_trigger_happy_without_accumulating_to_the_bound():
    config = load_config()
    state = _genesis(config)
    state, first_trace = _step(state, "你真棒", config)
    assert state.affect.valence > 0.3
    assert first_trace.decision.emotion_label == "happy"

    previous = state.affect.valence
    last_change = None
    for _ in range(20):
        state, _ = _step(state, "你真棒", config)
        last_change = abs(state.affect.valence - previous)
        previous = state.affect.valence
    assert state.affect.valence < 0.8
    assert last_change < 1e-6


def test_attack_target_minimal_pairs_and_goal_suppression():
    config = load_config()
    for text in ("你就是个废物", "澄音，你真没用", "闭嘴", "拆了你"):
        features = perceive(_event(text), config).situation_features
        assert features["assistant_directed_attack"] == 1.0

    for text in ("同事骂我是废物", "这个游戏真垃圾"):
        features = perceive(_event(text), config).situation_features
        assert features["assistant_directed_attack"] == 0.0

    mixed = perceive(_event("你这个废物，帮我查一下"), config).situation_features
    assert mixed["assistant_directed_attack"] == 1.0
    assert mixed["goal_relevance"] == 0.0


def test_attack_blocks_help_and_comfort_without_false_distress_empathy():
    config = load_config()
    _, trace = _step(_genesis(config), "你就是个废物", config)
    blocked = {gate["intent"] for gate in trace.gates if gate["blocked"]}
    assert {"assist", "comfort"} <= blocked
    assert trace.selected_intent == "assert_boundary"
    assert trace.decision.empathy_intensity == 0.0


def test_first_apology_acknowledges_repair_without_instant_forgiveness():
    config = load_config()
    state, attack = _step(_genesis(config), "你就是个废物", config, "user-a")
    state, repair = _step(
        state,
        "对不起，我刚才不该骂你",
        config,
        "user-a",
        elapsed=60.0,
    )
    assert attack.selected_intent == "assert_boundary"
    assert state.affect.valence < 0.0
    assert repair.selected_intent == "acknowledge_repair"
    assert repair.decision.emotion_label == "sad"
    constraints = "；".join(repair.decision.expression_constraints)
    assert "伤害尚未完全消退" in constraints
    assert "不得立即说完全原谅" in constraints


def test_relationship_confirmation_does_not_auto_create_happy_or_commitment():
    config = load_config()
    event = _event("你是不是喜欢我")
    percept = perceive(event, config)
    _, trace = _reduce(_genesis(config), percept, 0.0, config, event)
    assert percept.situation_features["relationship_confirmation_request"] == 1.0
    assert percept.situation_features["user_tone_valence"] == 0.0
    assert trace.decision.emotion_label != "happy"
    assert trace.decision.attention_hint == "relationship"
    assert any(
        "不得因一次提问自动宣称稳定恋爱" in rule
        for rule in trace.decision.expression_constraints
    )


def test_relationship_caps_are_scoped_and_keep_independent_decay_clocks():
    config = load_config()
    state, _ = _step(_genesis(config), "你就是个废物", config, "user-a")
    user_a_before = state.caps.unit_map("user-a")["u_expect_user_kind"]
    state, _ = _step(state, "你好", config, "user-b", elapsed=600.0)
    user_a_after = state.caps.unit_map("user-a")["u_expect_user_kind"]
    user_b = state.caps.unit_map("user-b")["u_expect_user_kind"]

    assert user_a_after.activation == pytest.approx(user_a_before.activation)
    assert user_a_after.updated_at == user_a_before.updated_at
    assert user_b.activation == pytest.approx(0.5)
    assert user_b.updated_at != user_a_after.updated_at


def test_user_activation_distinguishes_high_low_and_negated_phrases():
    config = load_config()
    high_event = _event("明天面试，我紧张得睡不着")
    low_event = _event("今天提不起劲，也不想出门")
    negated_event = _event("我已经不焦虑了，也不害怕")

    high = perceive(high_event, config)
    low = perceive(low_event, config)
    negated = perceive(negated_event, config)
    assert high.situation_features["user_activation"] == 1.0
    assert low.situation_features["user_activation"] == -1.0
    assert negated.situation_features["user_activation"] == 0.0

    _, high_trace = _reduce(_genesis(config), high, 0.0, config, high_event)
    _, low_trace = _reduce(_genesis(config), low, 0.0, config, low_event)
    assert high_trace.decision.va_delta["arousal"] > 0.0
    assert low_trace.decision.va_delta["arousal"] < 0.0


def test_goal_request_stays_neutral_while_offer_is_surprising():
    config = load_config()
    _, goal_trace = _step(_genesis(config), "澄音，帮我看看这个怎么做", config)
    assert abs(goal_trace.decision.va_delta["valence"]) < 0.05
    assert goal_trace.decision.emotion_label != "happy"

    _, offer_trace = _step(
        _genesis(config), "他们给我发 offer 了！工资还比原来高！", config
    )
    assert offer_trace.decision.emotion_label == "surprised"


def test_slow_variables_readonly():
    """EAS、依恋与自我表征在 reducer 中只读。"""
    config = load_config()
    state = _genesis(config)
    before = state_to_dict(state)
    new_state, _ = _reduce(state, _percept("我今天好难过…", config), 0.0, config)
    new_full = state_to_dict(new_state)
    old_full = state_to_dict(state)
    assert new_full["temperament_eas"] == old_full["temperament_eas"]
    assert new_full["attachment"] == old_full["attachment"]
    assert new_full["self_representations"] == old_full["self_representations"]
    assert new_full["affect"] != old_full["affect"]  # 只有 VA 在 v0 可写
    _ = before
