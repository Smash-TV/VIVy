# -*- coding: utf-8 -*-
"""人格内核数据契约单元测试：结构、默认值、校验与序列化。"""

import copy

import pytest

from gwen_agent.modules.personality_core.config import (
    build_genesis_state,
    load_config,
    validate_config,
)
from gwen_agent.modules.personality_core.state import (
    REGULATION_STRATEGIES,
    SELF_REPRESENTATION_MODES,
    AttachmentState,
    CapsState,
    CapsUnit,
    Decision,
    DecisionTrace,
    InteractionEvent,
    PerceptItem,
    PerceptState,
    PersonalityState,
    is_valid_urn_uuid,
    new_uuid_urn,
    percept_from_dict,
    percept_to_dict,
    state_from_dict,
    state_to_dict,
    validate_state,
)


def _config():
    return load_config()


def test_config_loads_and_validates():
    config = _config()
    assert validate_config(config) == []
    assert config["config_version"]
    assert config["engine"]["enabled"] is True


def test_genesis_state_bounds_and_readonly_values():
    config = _config()
    state = build_genesis_state(config)
    assert validate_state(state) == []
    assert state.affect.valence == config["affect"]["baseline"]["valence"]
    assert state.affect.arousal == config["affect"]["baseline"]["arousal"]
    assert (
        state.temperament_eas.emotionality == config["temperament_eas"]["emotionality"]
    )
    assert state.attachment.anxiety == config["attachment"]["anxiety"]
    modes = {rep.mode for rep in state.self_representations}
    assert modes <= SELF_REPRESENTATION_MODES
    # CAPS 五类单元至少有一个
    assert state.caps.all_units()


def test_uuid_helpers():
    urn = new_uuid_urn()
    assert is_valid_urn_uuid(urn)
    assert not is_valid_urn_uuid("not-a-uuid")
    assert not is_valid_urn_uuid(None)
    assert not is_valid_urn_uuid("urn:uuid:garbage")


def test_event_roundtrip_and_text_property():
    event = InteractionEvent(
        event_id=new_uuid_urn(),
        source_id="s1",
        stream_id="rel1",
        stream_seq=1,
        occurred_at="2026-08-21T10:00:00+08:00",
        recorded_at="2026-08-21T10:00:01+08:00",
        channel="desktop",
        kind="message",
        payload={"text": "你好呀"},
    )
    assert event.text == "你好呀"
    restored = InteractionEvent.from_dict(event.to_dict())
    assert restored.to_dict() == event.to_dict()


def test_event_text_falls_back_to_content():
    event = InteractionEvent(
        event_id=new_uuid_urn(),
        source_id="s1",
        stream_id="rel1",
        stream_seq=1,
        occurred_at="2026-08-21T10:00:00+08:00",
        recorded_at="2026-08-21T10:00:01+08:00",
        channel="qq",
        kind="message",
        payload={"content": "正文"},
    )
    assert event.text == "正文"
    event.payload = {"text": "优先", "content": "兜底"}
    assert event.text == "优先"


def test_state_roundtrip():
    config = _config()
    state = build_genesis_state(config, updated_at=100.0)
    restored = state_from_dict(state_to_dict(state))
    assert state_to_dict(restored) == state_to_dict(state)


def test_attachment_type_derivation():
    assert (
        AttachmentState(anxiety=0.2, avoidance=0.2).attachment_type(0.5, 0.5)
        == "secure"
    )
    assert (
        AttachmentState(anxiety=0.8, avoidance=0.2).attachment_type(0.5, 0.5)
        == "preoccupied"
    )
    assert (
        AttachmentState(anxiety=0.2, avoidance=0.8).attachment_type(0.5, 0.5)
        == "dismissive"
    )
    assert (
        AttachmentState(anxiety=0.8, avoidance=0.8).attachment_type(0.5, 0.5)
        == "fearful"
    )
    with pytest.raises(TypeError):
        AttachmentState().attachment_type()


def test_percept_roundtrip_preserves_inferred_items():
    percept = PerceptState(
        timestamp="t",
        situation="s",
        items=[
            PerceptItem(
                kind="inferred", content="feature:user_boundary=1", source_ref="e"
            )
        ],
        situation_features={"user_boundary": 1.0},
        parser_version="rules-v1",
    )
    restored = percept_from_dict(percept_to_dict(percept))
    assert percept_to_dict(restored) == percept_to_dict(percept)


def test_config_validation_rejects_duplicate_and_unknown_refs():
    config = copy.deepcopy(_config())
    config["caps"]["units"].append(copy.deepcopy(config["caps"]["units"][0]))
    config["caps"]["intents"][0]["situation_weights"]["missing_feature"] = 1.0
    issues = validate_config(config)
    assert any("unit_id 重复" in issue for issue in issues)
    assert any("未知特征" in issue for issue in issues)


def test_validate_state_detects_violations():
    config = _config()
    state = build_genesis_state(config)
    state.affect.valence = 1.5
    assert any("valence" in issue for issue in validate_state(state))
    state = build_genesis_state(config)
    state.caps.encodings.append(
        CapsUnit(unit_id="bad", category="encodings", content="x", activation=2.0)
    )
    assert any("activation" in issue for issue in validate_state(state))


def test_decision_trace_to_dict():
    trace = DecisionTrace(
        trace_id=new_uuid_urn(),
        event_id=new_uuid_urn(),
        state_revision_before=None,
        state_revision_after="rev-1",
        selected_intent="comfort",
        selected_regulation="reappraise",
        decision=Decision(va_delta={"valence": -0.5, "arousal": 0.0}),
    )
    data = trace.to_dict()
    assert data["selected_intent"] == "comfort"
    assert data["decision"]["va_delta"]["valence"] == -0.5


def test_regulation_strategy_enum_frozen():
    assert REGULATION_STRATEGIES == {
        "reappraise",
        "situation_modification",
        "attention_deployment",
        "pause_set_boundary",
    }


def test_percept_item_kinds():
    assert {
        PerceptItem(kind="observed", content="x", source_ref="s").kind,
        PerceptItem(kind="inferred", content="x", source_ref="s", confidence=0.6).kind,
    } <= {"observed", "inferred"}


def test_caps_state_unit_map():
    caps = CapsState(
        encodings=[CapsUnit(unit_id="u1", category="encodings", content="c")],
        goals_and_values=[
            CapsUnit(unit_id="u2", category="goals_and_values", content="g")
        ],
    )
    assert set(caps.unit_map()) == {"u1", "u2"}
    assert {u.unit_id for u in caps.all_units()} == {"u1", "u2"}


def test_caps_relationship_units_overlay_only_the_requested_subject():
    caps = CapsState(
        expectancies_and_beliefs=[
            CapsUnit(
                unit_id="u_expect",
                category="expectancies_and_beliefs",
                content="template",
                activation=0.5,
                scope="relationship",
            ),
            CapsUnit(
                unit_id="u_expect",
                category="expectancies_and_beliefs",
                content="a",
                activation=0.1,
                scope="relationship",
                subject_ref="user-a",
                updated_at=10.0,
            ),
            CapsUnit(
                unit_id="u_expect",
                category="expectancies_and_beliefs",
                content="b",
                activation=0.4,
                scope="relationship",
                subject_ref="user-b",
                updated_at=20.0,
            ),
        ]
    )
    assert caps.unit_map()["u_expect"].activation == pytest.approx(0.5)
    assert caps.unit_map("user-a")["u_expect"].activation == pytest.approx(0.1)
    assert caps.unit_map("user-b")["u_expect"].activation == pytest.approx(0.4)
    assert validate_state(PersonalityState(caps=caps)) == []


def test_caps_duplicate_validation_uses_unit_and_subject_together():
    duplicate = CapsUnit(
        unit_id="u_expect",
        category="expectancies_and_beliefs",
        content="duplicate",
        scope="relationship",
        subject_ref="user-a",
    )
    state = PersonalityState(
        caps=CapsState(expectancies_and_beliefs=[duplicate, copy.deepcopy(duplicate)])
    )
    assert any("subject_ref 重复" in issue for issue in validate_state(state))


def test_legacy_caps_rows_without_scope_metadata_still_restore():
    state = build_genesis_state(_config(), updated_at=100.0)
    payload = state_to_dict(state)
    for category in (
        "encodings",
        "expectancies_and_beliefs",
        "goals_and_values",
        "competencies_and_regulatory_plans",
    ):
        for unit in payload["caps"][category]:
            unit.pop("scope", None)
            unit.pop("subject_ref", None)
            unit.pop("updated_at", None)

    restored = state_from_dict(payload)
    assert validate_state(restored) == []
    assert all(unit.scope == "global" for unit in restored.caps.all_units())
    assert all(unit.subject_ref is None for unit in restored.caps.all_units())


def test_only_relationship_expectation_is_scoped_in_v4_config():
    config = _config()
    scoped = {
        unit["unit_id"]
        for unit in config["caps"]["units"]
        if unit.get("scope") == "relationship"
    }
    assert scoped == {"u_expect_user_kind"}
