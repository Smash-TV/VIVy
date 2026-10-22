# -*- coding: utf-8 -*-
"""人格内核 v0 离线演示：固定情境 → PerceptState → VA 衰减/冲量 → 意图/调节 → 表达约束 → DecisionTrace。

用法：
    python scripts/personality_core_demo.py            # 跑 4 条内置示例
    python scripts/personality_core_demo.py "我好难过"  # 跑任意一句

只读演示：不写入正式 data/personality_core.db，使用 tmp/ 下的临时库。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from gwen_agent.modules.personality_core import PersonalityCore, decision_text
from gwen_agent.utils import TMP_DIR

DEFAULT_SAMPLES = [
    "谢谢你，我今天真的好开心！",
    "我今天好难过，压力好大…",
    "有人威胁我，我害怕",
    "澄音，帮我看看这个怎么做",
]


def run_one(core: PersonalityCore, text: str) -> None:
    event = core.build_event(
        text,
        channel="demo",
        actor_ref="demo-user",
        relationship_ref="demo",
        session_ref="demo",
        stream_id="demo",
    )
    percept = core.perceive(event)
    state_after, trace = core.preview(event, percept)
    decision = trace.decision
    print("-" * 72)
    print(f"输入: {text!r}")
    print(f"特征: {percept.situation_features}")
    print(
        f"VA 变化: {decision.va_delta} -> 新坐标 ({state_after.affect.valence:.3f}, {state_after.affect.arousal:.3f})"
    )
    print(
        f"意图: {trace.selected_intent} | 调节: {trace.selected_regulation} | 情绪标签: {decision.emotion_label}"
    )
    print(
        f"工作自我: {decision.working_self_refs} | 共情: {decision.empathy_intensity:.2f} | 注意: {decision.attention_hint}"
    )
    print(f"表达约束: {decision_text(trace, core.config)}")


def main() -> None:
    os.makedirs(TMP_DIR, exist_ok=True)
    core = PersonalityCore(db_path=os.path.join(TMP_DIR, "personality_core_demo.db"))
    samples = sys.argv[1:] or DEFAULT_SAMPLES
    print(f"人格内核 v0 演示（config_version={core.config_version}）\n")
    for text in samples:
        run_one(core, text)


if __name__ == "__main__":
    main()
