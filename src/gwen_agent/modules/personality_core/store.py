# -*- coding: utf-8 -*-
"""SQLite 快照/修订/事件/trace 存储：幂等提交 + 乐观版本（CAS）。

v0 使用 SQLite 默认 rollback journal + BEGIN IMMEDIATE 短事务（见 10 §7：
本机 SQLite 3.50.4 处于 WAL-reset bug 影响范围，暂不启用 WAL）。
原始事件是权威，人格状态是幂等投影：同一 event_id 只产生一个 revision；
并发冲突按流内顺序读取最新状态并重放，不能后写覆盖先写。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from . import runtime
from .config import build_genesis_state
from .perception import perceive as derive_percept
from .state import (
    DecisionTrace,
    InteractionEvent,
    PerceptState,
    PersonalityState,
    percept_from_dict,
    percept_to_dict,
    state_from_dict,
    state_to_dict,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    stream_id TEXT NOT NULL,
    stream_seq INTEGER NOT NULL,
    occurred_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    channel TEXT NOT NULL,
    kind TEXT NOT NULL,
    actor_ref TEXT,
    relationship_ref TEXT,
    session_ref TEXT,
    payload TEXT NOT NULL,
    source_refs TEXT NOT NULL,
    causation_ref TEXT,
    supersedes_ref TEXT,
    status TEXT NOT NULL,
    outcome TEXT,
    schema_version INTEGER NOT NULL,
    UNIQUE(stream_id, stream_seq)
);
CREATE TABLE IF NOT EXISTS state_revisions (
    applied_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    revision_id TEXT NOT NULL UNIQUE,
    event_id TEXT NOT NULL UNIQUE REFERENCES events(event_id),
    prev_revision_id TEXT,
    applied_at TEXT NOT NULL,
    config_version TEXT NOT NULL,
    state_before TEXT NOT NULL,
    state_after TEXT NOT NULL,
    percept_state TEXT,
    elapsed_seconds REAL,
    config_snapshot TEXT
);
CREATE TABLE IF NOT EXISTS decision_traces (
    trace_id TEXT NOT NULL UNIQUE,
    event_id TEXT NOT NULL UNIQUE REFERENCES events(event_id),
    revision_id TEXT NOT NULL REFERENCES state_revisions(revision_id),
    trace TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_stream ON events(stream_id, stream_seq);
CREATE INDEX IF NOT EXISTS idx_revisions_sequence ON state_revisions(applied_sequence);
"""


class ConcurrencyError(Exception):
    """乐观版本检查失败：提交基准 revision 已过期。"""


@dataclass
class CommitResult:
    newly_applied: bool
    revision_id: Optional[str]
    applied_sequence: Optional[int]
    state_before: Optional[PersonalityState]
    state_after: Optional[PersonalityState]
    trace: Optional[DecisionTrace]


def _iso_to_epoch(value: Optional[str]) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value).timestamp()
    except (ValueError, TypeError):
        return 0.0


def _precomputed_matches(precomputed, computed) -> bool:
    """校验预览结果，阻止调用方把任意状态伪装成 reducer 输出。"""
    preview_state, preview_trace = precomputed
    computed_state, computed_trace = computed
    if state_to_dict(preview_state) != state_to_dict(computed_state):
        return False
    preview_data = preview_trace.to_dict()
    computed_data = computed_trace.to_dict()
    for data in (preview_data, computed_data):
        data.pop("trace_id", None)
        data.pop("state_revision_before", None)
        data.pop("state_revision_after", None)
    return preview_data == computed_data


def _normalize_json(value):
    """按 SQLite 中 JSON 文本实际可还原的形式规范化值。"""
    return json.loads(json.dumps(value, ensure_ascii=False))


def _event_identity(event: InteractionEvent) -> dict:
    """提取事件不可变语义；recorded_at/status/outcome 属于记录与处理元数据。"""
    return {
        "source_id": str(event.source_id),
        "stream_id": str(event.stream_id),
        "occurred_at": str(event.occurred_at),
        "channel": str(event.channel),
        "kind": str(event.kind),
        "actor_ref": None if event.actor_ref is None else str(event.actor_ref),
        "relationship_ref": (
            None if event.relationship_ref is None else str(event.relationship_ref)
        ),
        "session_ref": None if event.session_ref is None else str(event.session_ref),
        "payload": _normalize_json(event.payload),
        "source_refs": _normalize_json(event.source_refs),
        "causation_ref": (
            None if event.causation_ref is None else str(event.causation_ref)
        ),
        "supersedes_ref": (
            None if event.supersedes_ref is None else str(event.supersedes_ref)
        ),
        "schema_version": int(event.schema_version),
    }


def _percept_identity(percept: PerceptState) -> dict:
    """比较感知的决定性内容；timestamp 只表示本次解析时刻。"""
    data = percept_to_dict(percept)
    data.pop("timestamp", None)
    return data


def _event_from_row(row: sqlite3.Row) -> InteractionEvent:
    """从账本行还原权威事件，供幂等校验与 reducer 使用。"""
    return InteractionEvent(
        event_id=row["event_id"],
        source_id=row["source_id"],
        stream_id=row["stream_id"],
        stream_seq=int(row["stream_seq"]),
        occurred_at=row["occurred_at"],
        recorded_at=row["recorded_at"],
        channel=row["channel"],
        kind=row["kind"],
        actor_ref=row["actor_ref"],
        relationship_ref=row["relationship_ref"],
        session_ref=row["session_ref"],
        payload=json.loads(row["payload"]),
        source_refs=json.loads(row["source_refs"]),
        causation_ref=row["causation_ref"],
        supersedes_ref=row["supersedes_ref"],
        status=row["status"],
        outcome=json.loads(row["outcome"]) if row["outcome"] is not None else None,
        schema_version=int(row["schema_version"]),
    )


def _stored_event_identity(row: sqlite3.Row) -> dict:
    """把账本行还原为可与 InteractionEvent 比较的不可变语义。"""
    return _event_identity(_event_from_row(row))


class PersonalityStore:
    """人格内核持久化：事件账本 + 状态修订 + 决策 trace，幂等与 CAS。"""

    def __init__(self, db_path: str, config: dict):
        self.db_path = db_path
        self.config = config
        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # 连接与建表
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        # 明确保持 rollback journal，不启用 WAL（见模块文档）
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(SCHEMA)
                self._migrate_schema(conn)
                conn.execute("PRAGMA journal_mode=DELETE")
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _migrate_schema(conn: sqlite3.Connection) -> None:
        """给已有 v0 数据库补列；旧修订因缺少冻结快照会在回放中明确标记。"""
        migrations = {
            "events": {
                "actor_ref": "TEXT",
                "relationship_ref": "TEXT",
                "session_ref": "TEXT",
            },
            "state_revisions": {
                "percept_state": "TEXT",
                "elapsed_seconds": "REAL",
                "config_snapshot": "TEXT",
            },
        }
        for table, columns in migrations.items():
            existing = {
                row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for column, declaration in columns.items():
                if column not in existing:
                    conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
                    )

    # ------------------------------------------------------------------
    # 事件
    # ------------------------------------------------------------------
    def append_event(self, event: InteractionEvent) -> bool:
        """记录原始事件；完全相同的 event_id 重试返回 False。

        stream_seq 为 None 时在 BEGIN IMMEDIATE 事务内按流分配单调序号，
        避免两个未提交事件拿到相同序号后触发 UNIQUE(stream_id, stream_seq) 冲突。
        相同 event_id 若携带不同语义则拒绝，不能把冲突误当成幂等重试。
        """
        with self._lock:
            conn = self._connect()
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("BEGIN IMMEDIATE")
                existing = conn.execute(
                    "SELECT * FROM events WHERE event_id=?", (event.event_id,)
                ).fetchone()
                if existing:
                    sequence_conflict = (
                        event.stream_seq is not None
                        and event.stream_seq != existing["stream_seq"]
                    )
                    identity_conflict = _event_identity(
                        event
                    ) != _stored_event_identity(existing)
                    if sequence_conflict or identity_conflict:
                        conn.rollback()
                        raise ValueError(
                            f"event_id {event.event_id} 已存在，但事件语义不一致"
                        )
                    event.stream_seq = int(existing["stream_seq"])
                    conn.rollback()
                    return False
                if event.stream_seq is None:
                    row = conn.execute(
                        "SELECT COALESCE(MAX(stream_seq), 0) FROM events WHERE stream_id=?",
                        (event.stream_id,),
                    ).fetchone()
                    event.stream_seq = int(row[0]) + 1
                try:
                    conn.execute(
                        """
                        INSERT INTO events (
                            event_id, source_id, stream_id, stream_seq,
                            occurred_at, recorded_at, channel, kind,
                            actor_ref, relationship_ref, session_ref, payload,
                            source_refs, causation_ref, supersedes_ref, status,
                            outcome, schema_version
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event.event_id,
                            event.source_id,
                            event.stream_id,
                            event.stream_seq,
                            event.occurred_at,
                            event.recorded_at,
                            event.channel,
                            event.kind,
                            event.actor_ref,
                            event.relationship_ref,
                            event.session_ref,
                            json.dumps(event.payload, ensure_ascii=False),
                            json.dumps(event.source_refs, ensure_ascii=False),
                            event.causation_ref,
                            event.supersedes_ref,
                            event.status,
                            (
                                json.dumps(
                                    _normalize_json(event.outcome), ensure_ascii=False
                                )
                                if event.outcome is not None
                                else None
                            ),
                            event.schema_version,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    conn.rollback()
                    occupied = conn.execute(
                        "SELECT event_id FROM events WHERE stream_id=? AND stream_seq=?",
                        (event.stream_id, event.stream_seq),
                    ).fetchone()
                    if occupied:
                        raise ConcurrencyError(
                            f"流 {event.stream_id} 的序号 {event.stream_seq} "
                            "已被其他事件占用"
                        ) from exc
                    raise
                conn.commit()
                return True
            finally:
                conn.close()

    def load_events(self, limit: Optional[int] = None) -> list:
        """按 (stream_id, stream_seq) 读取事件，仅供账本检查；回放按 applied_sequence。"""
        sql = "SELECT * FROM events ORDER BY stream_id, stream_seq"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql).fetchall()
                columns = [d[0] for d in conn.execute(sql).description]
                return [dict(zip(columns, row)) for row in rows]
            finally:
                conn.close()

    def next_stream_seq(self, stream_id: str) -> int:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT COALESCE(MAX(stream_seq), 0) FROM events WHERE stream_id=?",
                    (stream_id,),
                ).fetchone()
                return int(row[0]) + 1
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    def _latest_revision_row(self, conn: sqlite3.Connection):
        return conn.execute(
            "SELECT revision_id, state_after FROM state_revisions "
            "ORDER BY applied_sequence DESC LIMIT 1"
        ).fetchone()

    def latest_state(self) -> tuple:
        """返回 (最新 PersonalityState | None, revision_id | None)。"""
        with self._lock:
            conn = self._connect()
            try:
                row = self._latest_revision_row(conn)
                if not row:
                    return None, None
                return state_from_dict(json.loads(row[1])), row[0]
            finally:
                conn.close()

    def get_trace(self, event_id: str) -> Optional[dict]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT trace FROM decision_traces WHERE event_id=?", (event_id,)
                ).fetchone()
                return json.loads(row[0]) if row else None
            finally:
                conn.close()

    def count_revisions(self) -> int:
        with self._lock:
            conn = self._connect()
            try:
                return int(
                    conn.execute("SELECT COUNT(*) FROM state_revisions").fetchone()[0]
                )
            finally:
                conn.close()

    def list_recent_traces(self, limit: int = 20) -> list:
        """按提交顺序返回最近 N 条决策 trace（调试面板用，只读）。"""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT event_id, revision_id, trace FROM decision_traces
                    ORDER BY rowid DESC LIMIT ?
                    """,
                    (int(limit),),
                ).fetchall()
                return [
                    {
                        "event_id": row[0],
                        "revision_id": row[1],
                        "trace": json.loads(row[2]),
                    }
                    for row in rows
                ]
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # 提交（幂等 + CAS）
    # ------------------------------------------------------------------
    def reduce_and_commit(
        self,
        event: InteractionEvent,
        percept: PerceptState,
        expected_revision_id: Optional[str] = None,
        precomputed=None,
    ) -> CommitResult:
        """在同一事务里提交状态修订与 trace。

        - event 必须已存在于原始事件账本，且不可变语义一致；
        - 同一 event_id 已有修订 → 幂等返回既有结果，不重复改变状态；
        - outcome 只允许从空值补成第一个结果，不能用重试覆盖已有结果；
        - expected_revision_id 与当前最新 revision 不一致 → 抛 ConcurrencyError；
        - precomputed=(state_after, trace) 可复用 preview 结果，避免二次计算。
        """
        with self._lock:
            conn = self._connect()
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("BEGIN IMMEDIATE")
                event_row = conn.execute(
                    "SELECT * FROM events WHERE event_id=?", (event.event_id,)
                ).fetchone()
                if event_row is None:
                    conn.rollback()
                    raise ValueError(f"event_id {event.event_id} 尚未写入原始事件账本")
                sequence_conflict = (
                    event.stream_seq is not None
                    and event.stream_seq != event_row["stream_seq"]
                )
                identity_conflict = _event_identity(event) != _stored_event_identity(
                    event_row
                )
                if sequence_conflict or identity_conflict:
                    conn.rollback()
                    raise ValueError(
                        f"event_id {event.event_id} 已存在，但事件语义不一致"
                    )
                stored_event = _event_from_row(event_row)
                event.stream_seq = stored_event.stream_seq
                incoming_outcome = (
                    None if event.outcome is None else _normalize_json(event.outcome)
                )
                if (
                    stored_event.outcome is not None
                    and incoming_outcome is not None
                    and stored_event.outcome != incoming_outcome
                ):
                    conn.rollback()
                    raise ValueError(
                        f"event_id {event.event_id} 已有不同 outcome，不能覆盖"
                    )
                outcome_added = (
                    stored_event.outcome is None and incoming_outcome is not None
                )
                existing = conn.execute(
                    "SELECT revision_id, state_after FROM state_revisions WHERE event_id=?",
                    (event.event_id,),
                ).fetchone()
                if existing:
                    if outcome_added:
                        conn.execute(
                            "UPDATE events SET outcome=? WHERE event_id=?",
                            (
                                json.dumps(incoming_outcome, ensure_ascii=False),
                                event.event_id,
                            ),
                        )
                        conn.commit()
                    else:
                        conn.rollback()
                    return CommitResult(
                        newly_applied=False,
                        revision_id=existing[0],
                        applied_sequence=None,
                        state_before=None,
                        state_after=state_from_dict(json.loads(existing[1])),
                        trace=None,
                    )
                derived_percept = derive_percept(stored_event, self.config)
                if _percept_identity(percept) != _percept_identity(derived_percept):
                    conn.rollback()
                    raise ValueError(
                        f"event_id {event.event_id} 的 PerceptState 与权威事件不一致"
                    )
                latest = self._latest_revision_row(conn)
                latest_revision_id = latest[0] if latest else None
                if (
                    expected_revision_id is not None
                    and expected_revision_id != latest_revision_id
                ):
                    conn.execute("ROLLBACK")
                    raise ConcurrencyError(
                        f"乐观版本检查失败：期望基于 {expected_revision_id}，"
                        f"当前最新为 {latest_revision_id}"
                    )
                state_before = (
                    state_from_dict(json.loads(latest[1]))
                    if latest
                    else build_genesis_state(self.config)
                )
                elapsed = max(
                    _iso_to_epoch(stored_event.occurred_at) - state_before.updated_at,
                    0.0,
                )
                computed = runtime.reduce(
                    state_before, percept, elapsed, self.config, stored_event
                )
                if (
                    precomputed is not None
                    and precomputed[1].state_revision_before == latest_revision_id
                    and _precomputed_matches(precomputed, computed)
                ):
                    state_after, trace = precomputed
                else:
                    state_after, trace = computed
                    if precomputed is not None:
                        trace.gates.append(
                            {
                                "intent": "__preview__",
                                "blocked": False,
                                "reason": "预览基准已过期或内容校验失败，提交时按最新状态重算",
                            }
                        )
                applied_at = datetime.now().astimezone().isoformat()
                cur = conn.execute(
                    """
                    INSERT INTO state_revisions (
                        revision_id, event_id, prev_revision_id, applied_at,
                        config_version, state_before, state_after,
                        percept_state, elapsed_seconds, config_snapshot
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trace.trace_id,
                        event.event_id,
                        latest_revision_id,
                        applied_at,
                        self.config.get("config_version", ""),
                        json.dumps(state_to_dict(state_before), ensure_ascii=False),
                        json.dumps(state_to_dict(state_after), ensure_ascii=False),
                        json.dumps(percept_to_dict(percept), ensure_ascii=False),
                        elapsed,
                        json.dumps(self.config, ensure_ascii=False, sort_keys=True),
                    ),
                )
                revision_id = trace.trace_id
                applied_sequence = cur.lastrowid
                trace.state_revision_before = latest_revision_id
                trace.state_revision_after = revision_id
                conn.execute(
                    """
                    INSERT INTO decision_traces (trace_id, event_id, revision_id, trace)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        trace.trace_id,
                        event.event_id,
                        revision_id,
                        json.dumps(trace.to_dict(), ensure_ascii=False),
                    ),
                )
                if outcome_added:
                    conn.execute(
                        "UPDATE events SET outcome=? WHERE event_id=?",
                        (
                            json.dumps(incoming_outcome, ensure_ascii=False),
                            event.event_id,
                        ),
                    )
                conn.commit()
                return CommitResult(
                    newly_applied=True,
                    revision_id=revision_id,
                    applied_sequence=applied_sequence,
                    state_before=state_before,
                    state_after=state_after,
                    trace=trace,
                )
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # 回放
    # ------------------------------------------------------------------
    def replay(self) -> tuple:
        """按实际提交顺序使用冻结感知/配置重放，返回状态与逐修订审计结果。"""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT
                        r.applied_sequence,
                        r.state_before,
                        r.state_after,
                        r.percept_state,
                        r.elapsed_seconds,
                        r.config_snapshot,
                        e.event_id,
                        e.source_id,
                        e.stream_id,
                        e.stream_seq,
                        e.occurred_at,
                        e.recorded_at,
                        e.channel,
                        e.kind,
                        e.actor_ref,
                        e.relationship_ref,
                        e.session_ref,
                        e.payload,
                        e.source_refs,
                        e.causation_ref,
                        e.supersedes_ref,
                        e.status,
                        e.outcome,
                        e.schema_version
                    FROM state_revisions AS r
                    LEFT JOIN events AS e ON e.event_id = r.event_id
                    ORDER BY r.applied_sequence
                    """
                ).fetchall()
                missing_trace_count = int(
                    conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM state_revisions AS r
                        LEFT JOIN decision_traces AS d
                          ON d.revision_id = r.revision_id
                        WHERE d.trace_id IS NULL OR d.event_id <> r.event_id
                        """
                    ).fetchone()[0]
                )
                extra_trace_count = int(
                    conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM decision_traces AS d
                        LEFT JOIN state_revisions AS r
                          ON r.revision_id = d.revision_id
                        LEFT JOIN events AS e ON e.event_id = d.event_id
                        WHERE r.revision_id IS NULL OR e.event_id IS NULL
                        """
                    ).fetchone()[0]
                )
            finally:
                conn.close()

        state = None
        mismatches = missing_trace_count + extra_trace_count
        for row in rows:
            (
                _,
                state_before_json,
                state_after_json,
                percept_json,
                elapsed_seconds,
                config_json,
                event_id,
                source_id,
                stream_id,
                stream_seq,
                occurred_at,
                recorded_at,
                channel,
                kind,
                actor_ref,
                relationship_ref,
                session_ref,
                payload,
                source_refs,
                causation_ref,
                supersedes_ref,
                status,
                outcome,
                schema_version,
            ) = row
            recorded_before = state_from_dict(json.loads(state_before_json))
            recorded_after = state_from_dict(json.loads(state_after_json))
            if event_id is None:
                # 旧库可能没有外键；孤儿修订不能重算，但也不能被 JOIN 静默隐藏。
                if state is not None and state_to_dict(state) != state_to_dict(
                    recorded_before
                ):
                    mismatches += 1
                mismatches += 1
                state = recorded_after
                continue
            if not percept_json or elapsed_seconds is None or not config_json:
                # v0 旧记录没有足够证据，不能假装能用当前解析器重建过去。
                mismatches += 1
                state = recorded_after
                continue
            frozen_config = json.loads(config_json)
            if state is None:
                state = build_genesis_state(frozen_config)
            if state_to_dict(state) != state_to_dict(recorded_before):
                mismatches += 1
                state = recorded_before
            event = InteractionEvent(
                event_id=event_id,
                source_id=source_id,
                stream_id=stream_id,
                stream_seq=stream_seq,
                occurred_at=occurred_at,
                recorded_at=recorded_at,
                channel=channel,
                kind=kind,
                actor_ref=actor_ref,
                relationship_ref=relationship_ref,
                session_ref=session_ref,
                payload=json.loads(payload),
                source_refs=json.loads(source_refs),
                causation_ref=causation_ref,
                supersedes_ref=supersedes_ref,
                status=status,
                outcome=json.loads(outcome) if outcome is not None else None,
                schema_version=schema_version,
            )
            state, _ = runtime.reduce(
                state,
                percept_from_dict(json.loads(percept_json)),
                float(elapsed_seconds),
                frozen_config,
                event,
            )
            if state_to_dict(state) != state_to_dict(recorded_after):
                mismatches += 1
        if state is None:
            state = build_genesis_state(self.config)
        return state, mismatches == 0, mismatches

    def close(self) -> None:
        pass
