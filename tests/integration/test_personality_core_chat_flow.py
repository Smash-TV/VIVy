# -*- coding: utf-8 -*-
"""人格内核集成测试：完整聊天流、回放一致性、幂等、CAS 与 48 情境方向验证。"""

import json
import os
import pathlib
import shutil
import sqlite3
import tempfile
import uuid

import pytest

from gwen_agent.modules.personality_core import (
    ConcurrencyError,
    PersonalityCore,
    build_genesis_state,
)
from gwen_agent.modules.personality_core.state import state_to_dict
from gwen_agent.utils import TMP_DIR

FIXTURE_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "fixtures"
    / "personality_core_scenarios"
)


@pytest.fixture()
def core():
    """在项目 tmp 目录创建临时 DB（本机系统 temp 目录被沙箱限制）。"""
    workdir = tempfile.mkdtemp(prefix="personality_core_test_", dir=TMP_DIR)
    try:
        yield PersonalityCore(db_path=os.path.join(workdir, "personality_core.db"))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _event(core, text, occurred_at=None, seq=None, channel="test", kind="message"):
    return core.build_event(
        text,
        channel=channel,
        actor_ref="u1",
        relationship_ref="rel1",
        session_ref="s1",
        stream_id="rel1",
        kind=kind,
        occurred_at=occurred_at,
    )


def test_full_flow_commit_and_trace(core):
    event = _event(core, "我今天好难过…")
    percept = core.perceive(event)
    result = core.commit(
        event, percept, outcome={"reply": "我在这里陪着你", "emotion": "sad"}
    )
    assert result.newly_applied is True
    assert result.applied_sequence == 1
    assert core.store.count_revisions() == 1
    assert result.state_after.affect.valence < 0
    trace = core.get_trace(event.event_id)
    assert trace["selected_intent"] == "comfort"
    assert trace["selected_regulation"] == "reappraise"
    assert trace["decision"]["emotion_label"] == "sad"
    event_row = core.store.load_events()[0]
    assert event_row["actor_ref"] == "u1"
    assert event_row["relationship_ref"] == "rel1"
    assert event_row["session_ref"] == "s1"


def test_idempotent_commit_does_not_change_state(core):
    event = _event(core, "我今天好难过…")
    percept = core.perceive(event)
    first = core.commit(event, percept)
    state_before = state_to_dict(first.state_after)
    second = core.commit(event, percept, outcome={"reply": "dup"})
    assert second.newly_applied is False
    assert core.store.count_revisions() == 1
    latest, _ = core.latest_state()
    assert state_to_dict(latest) == state_before
    assert json.loads(core.store.load_events()[0]["outcome"]) == {"reply": "dup"}


def test_cas_conflict_raises(core):
    event1 = _event(core, "你好呀")
    core.commit(event1, core.perceive(event1))
    latest, revision_id = core.latest_state()
    assert revision_id is not None
    event2 = _event(core, "在吗？")
    with pytest.raises(ConcurrencyError):
        core.commit(
            event2, core.perceive(event2), expected_revision_id="stale-revision"
        )
    ok = core.commit(event2, core.perceive(event2))
    assert ok.newly_applied is True


def test_replay_consistency(core):
    texts = ["你好呀", "我今天好难过…", "谢谢你陪我", "帮我看看这个怎么做"]
    for i, text in enumerate(texts):
        occurred_at = f"2026-08-21T10:0{i}:00+08:00"
        event = _event(core, text, occurred_at=occurred_at)
        core.commit(event, core.perceive(event))
    final_state, consistent, mismatches = core.replay()
    assert consistent is True
    assert mismatches == 0
    latest, _ = core.latest_state()
    assert state_to_dict(final_state) == state_to_dict(latest)


def test_stale_preview_is_recomputed_without_lost_update(core):
    first_event = _event(core, "谢谢你，我很开心")
    second_event = _event(core, "我今天好难过")
    first_preview = core.preview(first_event, core.perceive(first_event))
    second_preview = core.preview(second_event, core.perceive(second_event))
    assert first_preview[1].state_revision_before is None
    assert second_preview[1].state_revision_before is None

    first = core.commit(
        first_event, core.perceive(first_event), precomputed=first_preview
    )
    second = core.commit(
        second_event, core.perceive(second_event), precomputed=second_preview
    )
    assert core.store.count_revisions() == 2
    assert state_to_dict(second.state_before) == state_to_dict(first.state_after)
    assert any(gate["intent"] == "__preview__" for gate in second.trace.gates)
    final_state, consistent, mismatches = core.replay()
    assert consistent is True
    assert mismatches == 0
    assert state_to_dict(final_state) == state_to_dict(second.state_after)


def test_replay_uses_frozen_percept_and_config(core):
    event = _event(core, "我今天好难过")
    core.commit(event, core.perceive(event))
    core.config["situation_features"]["user_tone_valence"]["lexicon"]["negative"] = []
    final_state, consistent, mismatches = core.replay()
    latest, _ = core.latest_state()
    assert consistent is True
    assert mismatches == 0
    assert state_to_dict(final_state) == state_to_dict(latest)


def test_replay_follows_applied_sequence_across_streams(core):
    events = [
        core.build_event("谢谢你", channel="system", stream_id="z-stream"),
        core.build_event("我很难过", channel="system", stream_id="a-stream"),
        core.build_event("今天真开心", channel="system", stream_id="z-stream"),
    ]
    for event in events:
        core.commit(event, core.perceive(event))
    final_state, consistent, mismatches = core.replay()
    latest, _ = core.latest_state()
    assert consistent is True
    assert mismatches == 0
    assert state_to_dict(final_state) == state_to_dict(latest)


def test_uncommitted_event_is_not_replayed(core):
    committed = _event(core, "你好呀")
    core.commit(committed, core.perceive(committed))
    orphan = _event(core, "我今天好难过")
    core.store.append_event(orphan)
    final_state, consistent, mismatches = core.replay()
    latest, _ = core.latest_state()
    assert consistent is True
    assert mismatches == 0
    assert state_to_dict(final_state) == state_to_dict(latest)


def test_event_retry_restores_sequence_and_rejects_conflicting_payload(core):
    orphan = _event(core, "原始内容")
    assert core.store.append_event(orphan) is True
    assert orphan.stream_seq == 1

    retry = type(orphan).from_dict(orphan.to_dict())
    retry.stream_seq = None
    assert core.store.append_event(retry) is False
    assert retry.stream_seq == 1

    conflict = type(orphan).from_dict(orphan.to_dict())
    conflict.stream_seq = None
    conflict.payload = {"text": "冲突内容"}
    with pytest.raises(ValueError, match="事件语义不一致"):
        core.commit(conflict, core.perceive(conflict))
    assert core.store.count_revisions() == 0
    assert json.loads(core.store.load_events()[0]["payload"])["text"] == "原始内容"

    stream_conflict = type(orphan).from_dict(orphan.to_dict())
    stream_conflict.stream_seq = None
    stream_conflict.stream_id = "other-stream"
    with pytest.raises(ValueError, match="事件语义不一致"):
        core.commit(stream_conflict, core.perceive(stream_conflict))

    next_event = _event(core, "下一条")
    assert core.store.append_event(next_event) is True
    assert next_event.stream_seq == 2
    assert len(core.store.load_events()) == 2


def test_orphan_event_can_resume_once_and_replay(core):
    orphan = _event(core, "谢谢你")
    assert core.store.append_event(orphan) is True
    retry = type(orphan).from_dict(orphan.to_dict())
    retry.stream_seq = None

    first = core.commit(retry, core.perceive(retry))
    second = core.commit(retry, core.perceive(retry))
    assert retry.stream_seq == 1
    assert first.newly_applied is True
    assert second.newly_applied is False
    assert core.store.count_revisions() == 1
    _, consistent, mismatches = core.replay()
    assert consistent is True
    assert mismatches == 0


def test_reduce_and_commit_requires_matching_ledger_event(core):
    event = _event(core, "原始内容")
    with pytest.raises(ValueError, match="尚未写入原始事件账本"):
        core.store.reduce_and_commit(event, core.perceive(event))
    assert core.store.count_revisions() == 0

    assert core.store.append_event(event) is True
    conflict = type(event).from_dict(event.to_dict())
    conflict.payload = {"text": "冲突内容"}
    with pytest.raises(ValueError, match="事件语义不一致"):
        core.store.reduce_and_commit(conflict, core.perceive(conflict))
    assert core.store.count_revisions() == 0

    core.commit(event, core.perceive(event))
    with pytest.raises(ValueError, match="事件语义不一致"):
        core.store.reduce_and_commit(conflict, core.perceive(conflict))
    assert core.store.count_revisions() == 1


def test_reduce_and_commit_rejects_percept_from_another_event(core):
    event = _event(core, "你好")
    other_event = _event(core, "我今天好难过")
    core.store.append_event(event)
    with pytest.raises(ValueError, match="PerceptState 与权威事件不一致"):
        core.store.reduce_and_commit(event, core.perceive(other_event))
    assert core.store.count_revisions() == 0


def test_idempotent_retry_keeps_frozen_percept_across_config_change(core):
    event = _event(core, "我今天好难过")
    percept = core.perceive(event)
    first = core.commit(event, percept)
    core.config["situation_features"]["user_tone_valence"]["lexicon"]["negative"] = []

    second = core.commit(event, percept)
    assert first.newly_applied is True
    assert second.newly_applied is False
    assert core.store.count_revisions() == 1


def test_first_non_null_outcome_cannot_be_overwritten(core):
    event = _event(core, "你好")
    core.commit(event, core.perceive(event), outcome={"reply": "第一次"})
    retry = type(event).from_dict(event.to_dict())
    retry.stream_seq = None
    with pytest.raises(ValueError, match="不能覆盖"):
        core.commit(retry, core.perceive(retry), outcome={"reply": "第二次"})
    assert json.loads(core.store.load_events()[0]["outcome"]) == {"reply": "第一次"}
    assert core.store.count_revisions() == 1


def test_outcome_uses_json_semantics_and_preserves_empty_dict(core):
    tuple_event = _event(core, "结构化结果")
    first = core.commit(
        tuple_event,
        core.perceive(tuple_event),
        outcome={"parts": ("a", "b")},
    )
    tuple_retry = type(tuple_event).from_dict(tuple_event.to_dict())
    tuple_retry.stream_seq = None
    second = core.commit(
        tuple_retry,
        core.perceive(tuple_retry),
        outcome={"parts": ["a", "b"]},
    )
    assert first.newly_applied is True
    assert second.newly_applied is False

    empty_event = _event(core, "空结果")
    core.commit(empty_event, core.perceive(empty_event), outcome={})
    empty_row = next(
        row
        for row in core.store.load_events()
        if row["event_id"] == empty_event.event_id
    )
    assert json.loads(empty_row["outcome"]) == {}
    with pytest.raises(ValueError, match="不能覆盖"):
        core.commit(
            empty_event,
            core.perceive(empty_event),
            outcome={"reply": "不能补写"},
        )


def test_replay_reports_orphan_revision_in_legacy_database(core):
    state_before = state_to_dict(build_genesis_state(core.config))
    state_after = json.loads(json.dumps(state_before))
    state_after["affect"]["valence"] = 0.77
    conn = sqlite3.connect(core.db_path)
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(
            """
            INSERT INTO state_revisions (
                revision_id, event_id, prev_revision_id, applied_at,
                config_version, state_before, state_after
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"urn:uuid:{uuid.uuid4()}",
                f"urn:uuid:{uuid.uuid4()}",
                None,
                "2026-08-22T00:00:00+08:00",
                core.config_version,
                json.dumps(state_before, ensure_ascii=False),
                json.dumps(state_after, ensure_ascii=False),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    replayed, consistent, mismatches = core.replay()
    assert consistent is False
    assert mismatches >= 1
    assert replayed.affect.valence == pytest.approx(0.77)


def test_late_event_does_not_rewind(core):
    event1 = _event(core, "你好呀", occurred_at="2026-08-21T12:00:00+08:00")
    core.commit(event1, core.perceive(event1))
    latest, _ = core.latest_state()
    updated_at = latest.updated_at
    event2 = _event(core, "在吗？", occurred_at="2026-08-21T11:00:00+08:00")
    core.commit(event2, core.perceive(event2))
    latest2, _ = core.latest_state()
    assert latest2.updated_at >= updated_at


def test_attachment_and_eas_stay_readonly_across_commits(core):
    genesis = build_genesis_state(core.config)
    for text in ["你好呀", "我今天好难过…", "谢谢你"]:
        event = _event(core, text)
        core.commit(event, core.perceive(event))
    latest, _ = core.latest_state()
    new_full = state_to_dict(latest)
    old_full = state_to_dict(genesis)
    assert new_full["temperament_eas"] == old_full["temperament_eas"]
    assert new_full["attachment"] == old_full["attachment"]


def _direction(value):
    if value > 0.05:
        return "positive"
    if value < -0.05:
        return "negative"
    return "neutral"


def test_48_scenarios_bootstrap_contracts(core):
    """48 个 bootstrap 情境：方向、主导意图与调节策略符合契约。"""
    scenario_files = sorted(
        p for p in FIXTURE_DIR.glob("*.json") if p.name != "manifest.json"
    )
    assert len(scenario_files) == 48, "应为 48 个情境"
    mismatches = []
    for path in scenario_files:
        scenario = json.loads(path.read_text(encoding="utf-8"))
        event = _event(core, scenario["text"])
        percept = core.perceive(event)
        _, trace = core.preview(event, percept)
        va = trace.decision.va_delta
        expected = scenario["expected"]
        problems = []
        if _direction(va["valence"]) not in expected["valence_direction"]:
            problems.append(f"valence={_direction(va['valence'])}")
        if _direction(va["arousal"]) not in expected["arousal_direction"]:
            problems.append(f"arousal={_direction(va['arousal'])}")
        if trace.selected_intent != expected["dominant_intent"]:
            problems.append(f"intent={trace.selected_intent}")
        if trace.selected_regulation != expected["regulation"]:
            problems.append(f"regulation={trace.selected_regulation}")
        if problems:
            mismatches.append((scenario["scenario_id"], "; ".join(problems)))
    assert not mismatches, "\n".join(f"{sid}: {msg}" for sid, msg in mismatches)


def test_stream_seq_unique_per_stream(core):
    e1 = _event(core, "a")
    assert e1.stream_seq is None  # 构建时不分配，提交时分配
    core.commit(e1, core.perceive(e1))
    assert e1.stream_seq == 1
    e2 = _event(core, "b")
    core.commit(e2, core.perceive(e2))
    assert e2.stream_seq == 2
    assert (e1.stream_id, e1.stream_seq) != (e2.stream_id, e2.stream_seq)


def test_uuid_refs_are_stable_urns(core):
    event = _event(core, "你好")
    assert event.event_id.startswith("urn:uuid:")
    core.commit(event, core.perceive(event))
    trace = core.get_trace(event.event_id)
    assert trace["trace_id"].startswith("urn:uuid:")


def test_relationship_scope_and_first_apology_survive_commit_and_replay(core):
    attack = core.build_event(
        "你就是个废物",
        channel="qq",
        actor_ref="user-a",
        relationship_ref="user-a",
        session_ref="user-a",
        stream_id="user-a",
        occurred_at="2026-08-21T10:00:00+08:00",
    )
    first = core.commit(attack, core.perceive(attack))
    user_a_before = first.state_after.caps.unit_map("user-a")["u_expect_user_kind"]
    assert first.trace.selected_intent == "assert_boundary"

    neutral = core.build_event(
        "你好",
        channel="qq",
        actor_ref="user-b",
        relationship_ref="user-b",
        session_ref="user-b",
        stream_id="user-b",
        occurred_at="2026-08-21T10:10:00+08:00",
    )
    second = core.commit(neutral, core.perceive(neutral))
    user_a_after = second.state_after.caps.unit_map("user-a")["u_expect_user_kind"]
    user_b = second.state_after.caps.unit_map("user-b")["u_expect_user_kind"]
    assert user_a_after.activation == pytest.approx(user_a_before.activation)
    assert user_a_after.updated_at == user_a_before.updated_at
    assert user_b.activation == pytest.approx(0.5)

    apology = core.build_event(
        "对不起，我刚才不该骂你",
        channel="qq",
        actor_ref="user-a",
        relationship_ref="user-a",
        session_ref="user-a",
        stream_id="user-a",
        occurred_at="2026-08-21T10:11:00+08:00",
    )
    third = core.commit(apology, core.perceive(apology))
    assert third.state_after.affect.valence < 0.0
    assert third.trace.selected_intent == "acknowledge_repair"
    assert any(
        "伤害尚未完全消退" in rule
        for rule in third.trace.decision.expression_constraints
    )

    final_state, consistent, mismatches = core.replay()
    assert consistent is True
    assert mismatches == 0
    assert state_to_dict(final_state) == state_to_dict(third.state_after)
