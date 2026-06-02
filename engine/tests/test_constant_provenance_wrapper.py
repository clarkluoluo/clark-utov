"""Wrapper integration for constant_provenance."""

from __future__ import annotations

import pytest

from engine.constant_provenance import ConstantProvenanceConfig
from engine.discipline_wrapper import DisciplineWrapper
from engine.methodology import MethodologyConfig


def _wrapper(
    *,
    cfg: ConstantProvenanceConfig | None = None,
) -> DisciplineWrapper:
    return DisciplineWrapper(
        config=MethodologyConfig.from_env(),
        constant_provenance_config=cfg,
    )


def test_envelope_surfaces_constant_provenance_for_session_value():
    w = _wrapper()
    params = {
        "report": {
            "values": [
                {"value_name": "session_key",
                 "rerun_observations": [
                     {"dimension": "same_session", "value_hex": "aa"},
                     {"dimension": "same_session", "value_hex": "aa"},
                     {"dimension": "new_session",  "value_hex": "bb"},
                 ]},
            ],
        },
    }
    _, env = w.step("submit_report", params, lambda m, p: {"ok": True})
    assert env.constant_provenance, "expected constant_provenance sibling"
    item = env.constant_provenance[0]
    assert item["value_name"] == "session_key"
    assert item["category"] == "session_level_derived"
    assert item["evidence_class_ceiling"] == "B"
    assert item["scope"] == "per_session"
    assert item["recommended_action"] == "escalate_usage_decision"
    # Alert line emitted too.
    assert any("CONST-PROV" in a for a in env.alerts)


def test_envelope_skips_values_without_probes():
    w = _wrapper()
    params = {
        "report": {
            "values": [
                {"value_name": "no_probes", "source": "hook"},
            ],
        },
    }
    _, env = w.step("submit_report", params, lambda m, p: {"ok": True})
    assert env.constant_provenance == []


def test_dataflow_override_session_entropy_visible_on_envelope():
    w = _wrapper()
    params = {
        "report": {
            "values": [
                {"value_name": "template",
                 "rerun_observations": [
                     {"dimension": "same_session", "value_hex": "aa"},
                     {"dimension": "same_session", "value_hex": "aa"},
                     {"dimension": "new_session",  "value_hex": "aa"},
                     {"dimension": "new_appkey",   "value_hex": "aa"},
                     {"dimension": "new_per_input", "value_hex": "aa"},
                 ],
                 "producer_dataflow": {
                     "producer_reads": ["static", "session_token"],
                 }},
            ],
        },
    }
    _, env = w.step("submit_report", params, lambda m, p: {"ok": True})
    assert env.constant_provenance
    item = env.constant_provenance[0]
    # Despite all-stable reruns, dataflow flipped this to session.
    assert item["category"] == "session_level_derived"


def test_disabled_toggle_skips_classification():
    w = _wrapper(cfg=ConstantProvenanceConfig(enabled=False))
    params = {
        "report": {
            "values": [
                {"value_name": "k",
                 "rerun_observations": [
                     {"dimension": "same_session", "value_hex": "aa"},
                     {"dimension": "same_session", "value_hex": "aa"},
                 ]},
            ],
        },
    }
    _, env = w.step("submit_report", params, lambda m, p: {"ok": True})
    assert env.constant_provenance == []


def test_envelope_to_dict_includes_constant_provenance():
    w = _wrapper()
    params = {
        "values": [
            {"value_name": "k",
             "rerun_observations": [
                 {"dimension": "same_session", "value_hex": "aa"},
                 {"dimension": "same_session", "value_hex": "aa"},
                 {"dimension": "new_session",  "value_hex": "bb"},
             ]},
        ],
    }
    _, env = w.step("submit_report", params, lambda m, p: {"ok": True})
    d = env.to_dict()
    assert "constant_provenance" in d
