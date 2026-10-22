# -*- coding: utf-8 -*-
"""VIVY `_0` 独立运行入口：预览、提交、查看状态和冻结回放。"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sqlite3
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gwen_agent.modules.personality_core import (  # noqa: E402
    PersonalityCore,
    decision_text,
)
from gwen_agent.modules.personality_core.state import state_to_dict  # noqa: E402

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "personality_core.db"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "personality_core.json"


def _build_core(args: argparse.Namespace) -> PersonalityCore:
    db_path = pathlib.Path(args.db).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return PersonalityCore(
        config_path=str(pathlib.Path(args.config).resolve()),
        db_path=str(db_path),
    )


def _decision_payload(core: PersonalityCore, text: str, commit: bool) -> dict:
    event = core.build_event(
        text,
        channel="standalone",
        actor_ref="standalone-user",
        relationship_ref="standalone-relationship",
        session_ref="standalone-session",
        source_id="vivy-fb-0",
        stream_id="standalone-relationship",
    )
    percept = core.perceive(event)
    state_after, trace = core.preview(event, percept)
    payload = {
        "mode": "commit" if commit else "preview",
        "config_version": core.config_version,
        "event_id": event.event_id,
        "situation_features": percept.situation_features,
        "state_after": state_to_dict(state_after),
        "trace": trace.to_dict(),
        "expression_guidance": decision_text(trace, core.config),
    }
    if commit:
        result = core.commit(
            event,
            percept,
            outcome={"mode": "standalone-kernel-only", "reply": None},
            precomputed=(state_after, trace),
        )
        payload["commit"] = {
            "newly_applied": result.newly_applied,
            "revision_id": result.revision_id,
            "applied_sequence": result.applied_sequence,
        }
    return payload


def _inspect(core: PersonalityCore) -> dict:
    state, revision_id = core.latest_state()
    replay_state, replay_ok, mismatch_count = core.replay()
    with sqlite3.connect(core.db_path) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        event_count = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        revision_count = connection.execute(
            "SELECT COUNT(*) FROM state_revisions"
        ).fetchone()[0]
        trace_count = connection.execute(
            "SELECT COUNT(*) FROM decision_traces"
        ).fetchone()[0]
        versions = dict(
            connection.execute(
                "SELECT config_version, COUNT(*) FROM state_revisions "
                "GROUP BY config_version ORDER BY config_version"
            ).fetchall()
        )
    return {
        "database": os.path.abspath(core.db_path),
        "integrity": integrity,
        "event_count": event_count,
        "revision_count": revision_count,
        "trace_count": trace_count,
        "config_versions": versions,
        "latest_revision_id": revision_id,
        "latest_state": state_to_dict(state) if state is not None else None,
        "replay_ok": replay_ok,
        "replay_mismatch_count": mismatch_count,
        "replay_state": state_to_dict(replay_state),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="工作数据库路径")
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH), help="人格内核配置路径"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("preview", "只计算，不写数据库"),
        ("run", "计算并提交到工作数据库"),
    ):
        command_parser = subparsers.add_parser(command, help=help_text)
        command_parser.add_argument("text", help="要送入人格内核的文本")
    subparsers.add_parser("inspect", help="检查工作库状态与冻结回放")
    return parser


def main() -> None:
    args = _parser().parse_args()
    core = _build_core(args)
    if args.command == "inspect":
        payload = _inspect(core)
    else:
        payload = _decision_payload(core, args.text, commit=args.command == "run")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
