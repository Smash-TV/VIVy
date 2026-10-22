# -*- coding: utf-8 -*-
"""并发压力测试 worker：向同一 SQLite 人格库提交 N 个事件后退出。

供 tests/integration/test_personality_core_concurrency.py 以子进程方式调用，
模拟桌面/QQ 双进程同时写一个人格状态库（阶段 3 的 SQLite 并发验收）。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from gwen_agent.modules.personality_core import PersonalityCore


def main() -> None:
    db_path, stream_id, count, tag = sys.argv[1:5]
    core = PersonalityCore(db_path=db_path)
    for i in range(int(count)):
        event = core.build_event(
            f"stress-{tag}-{i} 我今天有点累",
            channel="stress",
            actor_ref=tag,
            relationship_ref=tag,
            session_ref=tag,
            stream_id=stream_id,
        )
        result = core.commit(event, core.perceive(event))
        if not result.newly_applied:
            raise RuntimeError(f"{tag} 事件 {i} 未作为新修订提交")
    print(f"worker-{tag} committed {count} events")


if __name__ == "__main__":
    main()
