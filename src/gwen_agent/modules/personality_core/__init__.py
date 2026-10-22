# -*- coding: utf-8 -*-
"""人格内核 v0：事件 → 感知 → 纯 reducer → CAPS/VA → 意图/调节 → 表达约束 → trace。

面向既有聊天链的唯一入口是 PersonalityCore（薄门面）：
  - build_event / perceive / preview / commit 形成一条垂直切片；
  - 状态只通过 runtime.reduce 修改，LLM/记忆/KG/外部评估/TTS 标签都没有写权限；
  - 阶段 4（提案式慢学习）到来前不创建 learning.py。

文件落点按 10 号文档 §10：state.py（契约）/ perception.py（感知）/ runtime.py（reducer）/
store.py（SQLite 幂等与 CAS）/ config.py（配置加载）。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from gwen_agent.utils import PERSONALITY_CORE_CONFIG_PATH

from .config import build_genesis_state, config_db_path, load_config
from .perception import perceive
from .runtime import decision_text, reduce
from .state import InteractionEvent, PerceptState, new_uuid_urn
from .store import CommitResult, ConcurrencyError, PersonalityStore

__all__ = [
    "PersonalityCore",
    "ConcurrencyError",
    "load_config",
    "build_genesis_state",
    "perceive",
    "reduce",
    "decision_text",
    "PERSONALITY_CORE_CONFIG_PATH",
]


class PersonalityCore:
    """人格内核薄门面：桌面/QQ 聊天链通过它调用同一套人格决策。"""

    def __init__(
        self, config_path: Optional[str] = None, db_path: Optional[str] = None
    ):
        self.config = load_config(config_path or PERSONALITY_CORE_CONFIG_PATH)
        self.db_path = config_db_path(self.config, db_path)
        self.store = PersonalityStore(self.db_path, self.config)

    @property
    def enabled(self) -> bool:
        return bool((self.config.get("engine") or {}).get("enabled", True))

    @property
    def config_version(self) -> Optional[str]:
        return self.config.get("config_version")

    @property
    def emotion_source(self) -> str:
        """情绪来源：kernel=决策后派生；llm=沿用 LLM 解析（内测 A/B 用）。"""
        return (self.config.get("engine") or {}).get("emotion_source", "kernel")

    # ------------------------------------------------------------------
    def build_event(
        self,
        text: str,
        channel: str,
        actor_ref: Optional[str] = None,
        relationship_ref: Optional[str] = None,
        session_ref: Optional[str] = None,
        source_id: str = "installation-default",
        stream_id: str = "global",
        kind: str = "message",
        occurred_at: Optional[str] = None,
    ) -> InteractionEvent:
        """构造带稳定 UUID 与流内序号的 InteractionEvent。"""
        now_iso = datetime.now(timezone.utc).isoformat()
        return InteractionEvent(
            event_id=new_uuid_urn(),
            source_id=source_id,
            stream_id=stream_id,
            stream_seq=None,  # 提交时在事务内原子分配
            occurred_at=occurred_at or now_iso,
            recorded_at=now_iso,
            channel=channel,
            actor_ref=None if actor_ref is None else str(actor_ref),
            relationship_ref=(
                None if relationship_ref is None else str(relationship_ref)
            ),
            session_ref=None if session_ref is None else str(session_ref),
            kind=kind,
            payload={"text": text} if text else {},
        )

    def perceive(self, event: InteractionEvent) -> PerceptState:
        return perceive(event, self.config)

    def latest_state(self):
        return self.store.latest_state()

    def preview(self, event: InteractionEvent, percept: PerceptState):
        """从最新状态计算本轮决策（不写库），返回 (state_after, trace)。"""
        state, revision_id = self.store.latest_state()
        if state is None:
            state = build_genesis_state(self.config)
        from .store import _iso_to_epoch

        elapsed = _iso_to_epoch(event.occurred_at) - state.updated_at
        state_after, trace = reduce(
            state, percept, max(elapsed, 0.0), self.config, event
        )
        trace.state_revision_before = revision_id
        return state_after, trace

    def commit(
        self,
        event: InteractionEvent,
        percept: PerceptState,
        outcome: Optional[dict] = None,
        expected_revision_id: Optional[str] = None,
        precomputed=None,
    ) -> CommitResult:
        """提交事件与状态修订（幂等 + CAS）；event 尚未入库时先落事件账本。"""
        if outcome is not None:
            event.outcome = outcome
        self.store.append_event(event)
        return self.store.reduce_and_commit(
            event,
            percept,
            expected_revision_id=expected_revision_id,
            precomputed=precomputed,
        )

    def get_trace(self, event_id: str) -> Optional[dict]:
        return self.store.get_trace(event_id)

    def replay(self):
        return self.store.replay()

    def next_stream_seq(self, stream_id: str) -> int:
        return self.store.next_stream_seq(stream_id)
