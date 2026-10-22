# -*- coding: utf-8 -*-
"""人格内核 bootstrap 合同检查：不把手写样例冒充阶段 2 行为验收。"""

import pathlib

from gwen_agent.modules.personality_core.config import load_config
from gwen_agent.modules.personality_core.evaluation import evaluate

FIXTURE_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "fixtures"
    / "personality_core_scenarios"
)


def _report():
    return evaluate(load_config(), FIXTURE_DIR)


def test_total_scenarios_is_48():
    assert _report()["total_scenarios"] == 48


def test_deterministic_consistency_equals_one():
    consistency = _report()["deterministic_consistency"]
    assert consistency["rate"] == 1.0


def test_scenario_contract_accuracy_all_one():
    accuracy = _report()["scenario_contract_accuracy"]
    for name, value in accuracy.items():
        assert value >= 0.99, f"{name} bootstrap 合同命中率过低: {value}"


def test_paraphrase_consistency_equals_one():
    assert _report()["paraphrase_consistency"] >= 0.99


def test_feature_sensitivity_contracts():
    sensitivity = _report()["feature_sensitivity_contract"]
    assert all(item["passed"] for item in sensitivity.values())


def test_state_recovery_error_tiny():
    assert _report()["state_recovery_error"] < 1e-6


def test_signature_contracts_all_match():
    sig = _report()["if_then_signature_match"]
    assert len(sig) == 3
    for sig_id, item in sig.items():
        assert item["accuracy"] == 1.0, f"{sig_id} 契约未全部命中: {item}"


def test_stage_gate_remains_open_until_real_behavior_evaluation():
    gate = _report()["stage_gate"]
    assert gate["passed"] is False
    assert "prompt/persona 基线消融" in gate["missing"]
