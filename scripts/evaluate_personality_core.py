# -*- coding: utf-8 -*-
"""人格内核 bootstrap 合同检查 CLI；不会把 48 个手写样例冒充阶段 2 验收。

指标逻辑在 src/gwen_agent/modules/personality_core/evaluation.py。

用法：
    python scripts/evaluate_personality_core.py            # 控制台报告
    python scripts/evaluate_personality_core.py --json     # 同时写 results/personality_core_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from gwen_agent.modules.personality_core.config import load_config
from gwen_agent.modules.personality_core.evaluation import evaluate
from gwen_agent.utils import RESULTS_DIR

FIXTURE_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "personality_core_scenarios"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="写 results/personality_core_eval.json"
    )
    args = parser.parse_args()
    report = evaluate(load_config(), FIXTURE_DIR)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = os.path.join(RESULTS_DIR, "personality_core_eval.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
