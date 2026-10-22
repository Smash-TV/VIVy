# -*- coding: utf-8 -*-
"""人格内核阶段 3 并发验收：多线程/双进程提交无丢失更新、幂等、回放一致、序号单调。"""

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading

from gwen_agent.modules.personality_core import PersonalityCore
from gwen_agent.modules.personality_core.state import state_to_dict
from gwen_agent.utils import PROJECT_ROOT, TMP_DIR

STRESS_WORKER = (
    pathlib.Path(PROJECT_ROOT) / "scripts" / "personality_core_stress_worker.py"
)


def _workdir():
    return tempfile.mkdtemp(prefix="personality_core_conc_", dir=TMP_DIR)


def test_threaded_commits_no_lost_updates():
    """4 个线程（各自独立实例）并发提交，总数不丢、序号单调、回放一致。"""
    workdir = _workdir()
    db = os.path.join(workdir, "pc.db")
    errors = []

    def worker(tag):
        try:
            core = PersonalityCore(db_path=db)
            for i in range(20):
                event = core.build_event(
                    f"t{tag}-{i} 我今天有点累",
                    channel="thread",
                    actor_ref=tag,
                    relationship_ref=tag,
                    session_ref=tag,
                    stream_id="stress",
                )
                result = core.commit(event, core.perceive(event))
                assert result.newly_applied
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors

    core = PersonalityCore(db_path=db)
    assert core.store.count_revisions() == 80
    final_state, consistent, mismatches = core.replay()
    assert consistent is True
    assert mismatches == 0
    latest, _ = core.latest_state()
    assert state_to_dict(final_state) == state_to_dict(latest)
    shutil.rmtree(workdir, ignore_errors=True)


def test_dual_process_stress():
    """两个子进程同时写同一 SQLite 人格库：40 个修订、回放一致。"""
    workdir = _workdir()
    db = os.path.join(workdir, "pc.db")
    procs = []
    for tag in ("A", "B"):
        procs.append(
            subprocess.Popen(
                [sys.executable, str(STRESS_WORKER), db, "stress", "20", tag],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    for proc in procs:
        out, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, f"worker 失败: {out} {err}"

    core = PersonalityCore(db_path=db)
    assert core.store.count_revisions() == 40
    final_state, consistent, mismatches = core.replay()
    assert consistent is True
    assert mismatches == 0
    latest, _ = core.latest_state()
    assert state_to_dict(final_state) == state_to_dict(latest)
    shutil.rmtree(workdir, ignore_errors=True)


def test_idempotent_across_instances():
    """同一事件经两个实例重复提交，只产生一个修订。"""
    workdir = _workdir()
    db = os.path.join(workdir, "pc.db")
    core_a = PersonalityCore(db_path=db)
    event = core_a.build_event(
        "你好呀",
        channel="thread",
        actor_ref="u1",
        relationship_ref="rel1",
        session_ref="s1",
        stream_id="rel1",
    )
    first = core_a.commit(event, core_a.perceive(event))
    assert first.newly_applied is True

    core_b = PersonalityCore(db_path=db)
    second = core_b.commit(event, core_b.perceive(event))
    assert second.newly_applied is False
    assert core_b.store.count_revisions() == 1
    shutil.rmtree(workdir, ignore_errors=True)


def test_dual_process_stream_seq_monotonic():
    """跨进程的 stream_seq 严格单调递增且不重复。"""
    workdir = _workdir()
    db = os.path.join(workdir, "pc.db")
    procs = [
        subprocess.Popen(
            [sys.executable, str(STRESS_WORKER), db, "stress", "10", tag],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for tag in ("X", "Y")
    ]
    for proc in procs:
        out, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, f"worker 失败: {out} {err}"

    core = PersonalityCore(db_path=db)
    events = core.store.load_events()
    seqs = [row["stream_seq"] for row in events if row["stream_id"] == "stress"]
    assert len(seqs) == 20
    assert len(set(seqs)) == 20
    assert seqs == sorted(seqs)
    shutil.rmtree(workdir, ignore_errors=True)
