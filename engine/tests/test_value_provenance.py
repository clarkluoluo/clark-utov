"""Value-provenance state machine — acceptance tests.

Locks in the reference target Round 5 distinction: hook/dump-sourced bytes
do NOT amount to closed-form recovery, even when observation_parity
holds. The framework caps the evidence_class accordingly so the
agent cannot forget.
"""

from __future__ import annotations

from engine.discipline_wrapper import DisciplineWrapper
from engine.methodology import MethodologyConfig
from engine.value_provenance import (
    ValueProvenanceConfig,
    tag_value,
    tag_values_in_params,
    PROVENANCE_CLOSED_FORM,
    PROVENANCE_HYBRID,
    PROVENANCE_OBSERVED,
    PROVENANCE_UNKNOWN,
)


# ---------------------------------------------------------------------------
# Module-level
# ---------------------------------------------------------------------------


def test_hook_source_tagged_observed_and_capped_at_b():
    r = tag_value(
        {"value_name": "vmp_key", "source": "hook",
         "recompute_fn_present": False,
         "recompute_matches_measured": False,
         "evidence_class": "A"},
        cfg=ValueProvenanceConfig(),
    )
    assert r.provenance == PROVENANCE_OBSERVED
    assert r.ceiling == "B"
    assert r.final_class == "B"
    assert r.downgraded is True


def test_closed_form_with_verified_recompute_keeps_a():
    r = tag_value(
        {"value_name": "sigma0_pred", "source": "formula",
         "recompute_fn_present": True,
         "recompute_matches_measured": True,
         "evidence_class": "A"},
        cfg=ValueProvenanceConfig(),
    )
    assert r.provenance == PROVENANCE_CLOSED_FORM
    assert r.ceiling == "A"
    assert r.final_class == "A"
    assert r.downgraded is False


def test_closed_form_declared_but_unverified_falls_to_observed():
    r = tag_value(
        {"value_name": "candidate_xor", "source": "formula",
         "recompute_fn_present": True,
         "recompute_matches_measured": False,
         "evidence_class": "A"},
        cfg=ValueProvenanceConfig(),
    )
    assert r.provenance == PROVENANCE_OBSERVED
    assert r.ceiling == "B"
    assert r.downgraded is True


def test_observation_parity_alone_does_not_imply_closed_form():
    """The retro lesson: two hook points producing equal bytes is
    parity, not recovery. Must stay at B."""
    r = tag_value(
        {"value_name": "captured_const", "source": "hook",
         "observation_parity": True,
         "recompute_fn_present": False,
         "evidence_class": "A"},
        cfg=ValueProvenanceConfig(),
    )
    assert r.provenance == PROVENANCE_OBSERVED
    assert r.ceiling == "B"
    assert r.parity_disclaimer is not None
    assert "BYTES match" in r.parity_disclaimer or "BYTES" in r.parity_disclaimer


def test_hybrid_record_capped_at_b():
    r = tag_value(
        {"value_name": "struct_with_constants", "source": "formula",
         "hybrid": True,
         "evidence_class": "A"},
        cfg=ValueProvenanceConfig(),
    )
    assert r.provenance == PROVENANCE_HYBRID
    assert r.ceiling == "B"


def test_unknown_source_falls_to_c():
    r = tag_value(
        {"value_name": "mystery", "source": "guess",
         "evidence_class": "B"},
        cfg=ValueProvenanceConfig(),
    )
    assert r.provenance == PROVENANCE_UNKNOWN
    assert r.ceiling == "C"
    assert r.final_class == "C"


def test_env_toggle_off_passes_through():
    cfg = ValueProvenanceConfig.from_env({"UTOV_VALUE_PROVENANCE": "off"})
    assert cfg.enabled is False
    r = tag_value(
        {"value_name": "x", "source": "hook",
         "evidence_class": "A"}, cfg=cfg,
    )
    assert r.final_class == "A"
    assert r.downgraded is False


def test_tag_values_in_params_mutates_records_in_place():
    params = {
        "report": {
            "values": [
                {"value_name": "k1", "source": "hook", "evidence_class": "A"},
                {"value_name": "k2", "source": "formula",
                 "recompute_fn_present": True,
                 "recompute_matches_measured": True,
                 "evidence_class": "A"},
            ],
        },
    }
    out = tag_values_in_params(params, cfg=ValueProvenanceConfig())
    names = {r.value_name for r in out}
    assert names == {"k1", "k2"}
    # k1 must be downgraded; k2 stays.
    k1 = params["report"]["values"][0]
    k2 = params["report"]["values"][1]
    assert k1["evidence_class"] == "B"
    assert k1["provenance"] == PROVENANCE_OBSERVED
    assert k2["evidence_class"] == "A"
    assert k2["provenance"] == PROVENANCE_CLOSED_FORM


# ---------------------------------------------------------------------------
# Wrapper integration
# ---------------------------------------------------------------------------


def test_wrapper_caps_observed_value_before_dispatch():
    wrapper = DisciplineWrapper(config=MethodologyConfig())
    captured = {}

    def dispatch(method, params):
        captured["ec"] = params["values"][0]["evidence_class"]
        captured["pv"] = params["values"][0]["provenance"]
        return {"ok": True}

    params = {
        "values": [
            {"value_name": "k_vmp", "source": "hook",
             "evidence_class": "A"},
        ],
    }
    result, env = wrapper.step("submit_finding", params, dispatch)
    assert result == {"ok": True}
    # Dispatch sees the capped class, not the requested A.
    assert captured["ec"] == "B"
    assert captured["pv"] == PROVENANCE_OBSERVED
    assert env.value_provenance
    assert any("VALUE-PROVENANCE" in a for a in env.alerts)


def test_wrapper_silent_when_no_downgrade_needed():
    wrapper = DisciplineWrapper(config=MethodologyConfig())
    params = {
        "values": [
            {"value_name": "ok", "source": "formula",
             "recompute_fn_present": True,
             "recompute_matches_measured": True,
             "evidence_class": "A"},
        ],
    }
    _, env = wrapper.step("submit_finding", params, lambda m, p: None)
    # closed_form with verified recompute → no alert.
    assert all("VALUE-PROVENANCE" not in a for a in env.alerts)
