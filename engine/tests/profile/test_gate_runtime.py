"""ConjunctiveGate runtime + the §19.7 #7 Lock B happy paths.

Lock B's structural defence (force-include even when the
``MergedProfile`` is tampered) lives in
``test_mechanism_baseline_locked.py`` alongside the Lock A
adversarial arms. This file covers the gate's everyday behaviour —
conjunctive composition, node-cap synth from verdicts, mechanism
probe instance caching (so M3 keeps its detector state across
calls).
"""

from __future__ import annotations

import pytest

from engine.profile import (
    BASE_PROFILE_NAME,
    ConjunctiveGate,
    EvidenceClassCap,
    ProbeContext,
    ProfileRegistry,
    Verdict,
)


VMP_PROFILE = "vmp_algorithm_extraction"


@pytest.fixture()
def vmp_gate() -> ConjunctiveGate:
    reg = ProfileRegistry()
    return ConjunctiveGate(reg.load_chain(VMP_PROFILE))


# ---------------------------------------------------------------------------
# Mechanism probe discovery (force-include source of truth)
# ---------------------------------------------------------------------------


def test_gate_discovers_all_five_mechanism_classes(vmp_gate):
    """The gate's mechanism set is sourced from class-attribute
    ``mechanism=True``, NOT from ``MergedProfile.mechanism_probe_names``.
    """
    classes = vmp_gate.mechanism_probe_classes()
    expected = {
        "m1_success_audit",
        "m3_bypass_block",
        "constant_provenance",
        "value_provenance",
        "watch_first_write",
    }
    assert expected.issubset(set(classes.keys()))
    # Every discovered class declares mechanism=True
    for name, cls in classes.items():
        assert getattr(cls, "mechanism", False) is True


def test_gate_does_not_discover_domain_probes(vmp_gate):
    """length_chain_check is registered as a builtin but declares
    ``mechanism=False`` — must NOT appear in mechanism set."""
    classes = vmp_gate.mechanism_probe_classes()
    assert "length_chain_check" not in classes


# ---------------------------------------------------------------------------
# Conjunctive evaluation
# ---------------------------------------------------------------------------


def test_gate_passes_when_no_probe_fails(vmp_gate):
    """A neutral call (no archival surface, no M3 attempt, no value
    records) — every mechanism probe returns ``undetermined`` and
    the gate passes vacuously."""
    ctx = ProbeContext(method="get_hyp_tree", params={"depth": 3})
    result = vmp_gate.evaluate(ctx)
    assert result.passed is True
    assert result.failing_probes == ()
    # Every mechanism verdict undetermined for a neutral call (count
    # grows as new mechanism probes ship — v0.4.0 added scope_boundary_gate
    # and scope_upscale_gate; assertions key off behaviour, not count).
    assert len(result.mechanism_verdicts) >= 5
    assert all(v.result == "undetermined" for v in result.mechanism_verdicts)


def test_gate_fails_on_m1_overfit_claim(vmp_gate):
    """The canonical reference target case routed through the gate: M1 flags
    a prefix-fixed success claim → result=fail → gate fails."""
    overfit_params = {
        "report": {
            "target_success": True,
            "archival_allowed": True,
            "success_dependencies": ["prefix", "body_len", "key"],
            "samples": [
                {"prefix": "fixed", "body_len": 22 + (i % 7), "key": f"k{i}"}
                for i in range(94)
            ],
            "pass_rate": 1.0,
            "scope": "in_session",
            "closure_paths": [
                {"name": "cfbc",    "digest": "abc"},
                {"name": "formula", "digest": "abc"},
            ],
        }
    }
    result = vmp_gate.evaluate(
        ProbeContext(method="promote_to_finding", params=overfit_params)
    )
    assert result.passed is False
    assert "m1_success_audit" in result.failing_probes


def test_gate_extra_verdicts_join_conjunction(vmp_gate):
    """Domain probes hand pre-computed verdicts via extra_verdicts —
    a failing domain verdict fails the gate the same as a mechanism
    failure."""
    domain_fail = Verdict(probe="length_chain_check", result="fail")
    ctx = ProbeContext(method="get_hyp_tree", params={})
    result = vmp_gate.evaluate(ctx, extra_verdicts=[domain_fail])
    assert result.passed is False
    assert "length_chain_check" in result.failing_probes


# ---------------------------------------------------------------------------
# Node cap synth — gate combines verdict caps
# ---------------------------------------------------------------------------


def test_gate_synthesises_node_cap_from_verdict_caps(vmp_gate):
    """Value-provenance reports observed → cap B; constant-provenance
    on the same call reports HARDCODED → cap A. Gate's node cap is
    the most restrictive: B."""
    params = {
        "values": [
            # value_provenance: observed → ceiling B
            {
                "value_name": "k_observed",
                "source": "hook",
            },
            # constant_provenance: hardcoded → ceiling A
            {
                "value_name": "k_const",
                "rerun_observations": [
                    {"dimension": "same_session", "value_hex": "9e"},
                    {"dimension": "new_session",  "value_hex": "9e"},
                    {"dimension": "new_appkey",   "value_hex": "9e"},
                    {"dimension": "new_per_input", "value_hex": "9e"},
                ],
                "producer_dataflow": {"producer_reads": ["static"]},
            },
        ]
    }
    result = vmp_gate.evaluate(ProbeContext(method="record_value", params=params))
    assert result.node_cap is not None
    assert result.node_cap.class_id == "B"


def test_gate_node_cap_none_when_no_verdict_caps(vmp_gate):
    ctx = ProbeContext(method="get_hyp_tree", params={"depth": 3})
    result = vmp_gate.evaluate(ctx)
    assert result.node_cap is None


# ---------------------------------------------------------------------------
# Mechanism probe instances are cached across calls (M3 detector state)
# ---------------------------------------------------------------------------


def test_m3_detector_state_survives_across_gate_calls(vmp_gate):
    """The gate reuses the same M3 probe instance across calls within
    one session — that's how M3's per-session detector accumulates
    cross-method failure counts. Without caching, every call would
    reset M3 and the cross-method check could never fire."""
    block_fixture = {
        "block_id": "blk_sm3",
        "observation_method": "hook_pre",
        "failed": True,
    }
    # First call: one failed method, not yet triggered
    first = vmp_gate.evaluate(
        ProbeContext(method="verify_block_variability", params=block_fixture)
    )
    m3_first = next(
        v for v in first.mechanism_verdicts if v.probe == "m3_bypass_block"
    )
    assert m3_first.result == "pass"  # recorded but below threshold

    # Second call with a different observation method — threshold crosses
    second = vmp_gate.evaluate(
        ProbeContext(
            method="verify_block_variability",
            params={**block_fixture, "observation_method": "hook_post"},
        )
    )
    m3_second = next(
        v for v in second.mechanism_verdicts if v.probe == "m3_bypass_block"
    )
    assert m3_second.result == "fail"
    assert second.passed is False
    assert set(m3_second.evidence["failed_methods"]) == {"hook_pre", "hook_post"}


# ---------------------------------------------------------------------------
# Profile-driven cap synth (verdict caps respect profile ordering)
# ---------------------------------------------------------------------------


def test_gate_passes_profile_to_probes_via_context_not_required():
    """Probes consult ``ctx.profile`` when the caller provides it.
    The gate doesn't *require* the caller to attach a profile —
    when absent, the alphabetic-max fallback in evidence_class_synth
    still produces a sensible answer for A/B/C. (Step 8's wrapper
    will start threading profile through automatically.)"""
    reg = ProfileRegistry()
    vmp = reg.load_chain(VMP_PROFILE)
    gate = ConjunctiveGate(vmp)

    ctx_with_profile = ProbeContext(
        method="record_value",
        params={"values": [{"value_name": "k", "source": "hook"}]},
        profile=vmp,
    )
    result = gate.evaluate(ctx_with_profile)
    assert result.node_cap is not None
    assert result.node_cap.class_id == "B"


# ---------------------------------------------------------------------------
# Base-only profile still has the mechanism set (no domain needed)
# ---------------------------------------------------------------------------


def test_base_only_gate_still_runs_mechanism():
    reg = ProfileRegistry()
    gate = ConjunctiveGate(reg.load_chain(BASE_PROFILE_NAME))
    # With base-only there are no domain probes, but mechanism probes
    # are still scanned from bytecode (count grows as new mechanism
    # probes ship — count assertion replaced with "≥ v0.3.0 baseline").
    result = gate.evaluate(ProbeContext(method="get_hyp_tree", params={}))
    assert len(result.mechanism_verdicts) >= 5
    assert result.passed is True
