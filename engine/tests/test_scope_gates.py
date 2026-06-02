"""v0.4.0 B1 + B2 — scope gates.

ScopeBoundaryGate (§19.9 base #3): scope claim ≤ observed boundary.
ScopeUpscaleGate (§19.9 base #1): pinned-at-observation → cross_env
needs dataflow proof.

Both ship as base mechanism probes. The vmp profile's
``scope_order = [task_bound, env_bound, single_identity_bound, cross_env]``
drives all the ordering decisions.
"""

from __future__ import annotations

import pytest

from engine.profile import (
    BASE_PROFILE_NAME,
    ConjunctiveGate,
    ProbeContext,
    ProfileRegistry,
)
from engine.profile.probes.scope_boundary_gate import ScopeBoundaryGateProbe
from engine.profile.probes.scope_upscale_gate import ScopeUpscaleGateProbe
from engine.scope_boundary_gate import (
    ScopeBoundaryConfig,
    check_scope_boundary,
)
from engine.scope_upscale_gate import (
    ScopeUpscaleConfig,
    check_scope_upscale,
)


VMP_PROFILE = "vmp_algorithm_extraction"


@pytest.fixture()
def vmp_profile():
    return ProfileRegistry().load_chain(VMP_PROFILE)


# ---------------------------------------------------------------------------
# B1 — ScopeBoundaryGate
# ---------------------------------------------------------------------------


def test_boundary_gate_passes_when_claim_equals_observed(vmp_profile):
    params = {"scope_claim": "env_bound", "scope_observed": "env_bound"}
    v = check_scope_boundary(params, scope_rank=vmp_profile.scope_rank)
    assert v.result == "pass"


def test_boundary_gate_passes_when_claim_narrower_than_observed(vmp_profile):
    params = {"scope_claim": "task_bound", "scope_observed": "env_bound"}
    v = check_scope_boundary(params, scope_rank=vmp_profile.scope_rank)
    assert v.result == "pass"


def test_boundary_gate_fails_when_claim_wider_than_observed(vmp_profile):
    """The canonical tc3 finding: observed only inside one runner+task
    context but claim is cross_env — refuse the extrapolation."""
    params = {"scope_claim": "cross_env", "scope_observed": "task_bound"}
    v = check_scope_boundary(params, scope_rank=vmp_profile.scope_rank)
    assert v.result == "fail"
    assert "extrapolate" in v.reason


def test_boundary_gate_undetermined_without_observed(vmp_profile):
    params = {"scope_claim": "env_bound"}
    v = check_scope_boundary(params, scope_rank=vmp_profile.scope_rank)
    assert v.result == "undetermined"


def test_boundary_gate_undetermined_without_profile_ordering():
    params = {"scope_claim": "cross_env", "scope_observed": "task_bound"}
    # No scope_rank supplied — gate can't compare.
    v = check_scope_boundary(params, scope_rank=None)
    assert v.result == "undetermined"
    assert "ordering" in v.reason


def test_boundary_gate_disabled_returns_none():
    cfg = ScopeBoundaryConfig(enabled=False)
    assert check_scope_boundary({"scope_claim": "cross_env"}, cfg=cfg) is None


# ---------------------------------------------------------------------------
# B1 — wired into ConjunctiveGate
# ---------------------------------------------------------------------------


def test_boundary_probe_registered_as_mechanism():
    probe = ScopeBoundaryGateProbe()
    assert probe.mechanism is True


def test_conjunctive_gate_fails_on_scope_overreach(vmp_profile):
    """End-to-end through the conjunctive gate: a claim that overreaches
    the observed boundary fails the gate."""
    gate = ConjunctiveGate(vmp_profile)
    ctx = ProbeContext(
        method="promote_to_finding",
        params={
            "scope_claim": "cross_env",
            "scope_observed": "task_bound",
        },
        profile=vmp_profile,
    )
    result = gate.evaluate(ctx)
    assert "scope_boundary_gate" in result.failing_probes
    assert result.passed is False


def test_conjunctive_gate_passes_when_claim_matches_observation(vmp_profile):
    gate = ConjunctiveGate(vmp_profile)
    ctx = ProbeContext(
        method="promote_to_finding",
        params={
            "scope_claim": "task_bound",
            "scope_observed": "task_bound",
        },
        profile=vmp_profile,
    )
    result = gate.evaluate(ctx)
    sb = next(
        v for v in result.mechanism_verdicts
        if v.probe == "scope_boundary_gate"
    )
    assert sb.result == "pass"


# ---------------------------------------------------------------------------
# B2 — ScopeUpscaleGate
# ---------------------------------------------------------------------------


def test_upscale_gate_passes_with_closed_form_attestation(vmp_profile):
    params = {"scope_claim": "cross_env", "value_class": "closed_form"}
    v = check_scope_upscale(params, scope_rank=vmp_profile.scope_rank)
    assert v.result == "pass"
    assert "closed-form" in v.reason


def test_upscale_gate_passes_with_explicit_dataflow_proof(vmp_profile):
    params = {
        "scope_claim": "cross_env",
        "value_class": "observed",
        "producer_dataflow": {"proof_of_invariance": True},
    }
    v = check_scope_upscale(params, scope_rank=vmp_profile.scope_rank)
    assert v.result == "pass"
    assert v.proof_of_invariance is True


def test_upscale_gate_passes_with_hardcoded_fixed_cp_category(vmp_profile):
    params = {
        "scope_claim": "cross_env",
        "constant_provenance_category": "hardcoded_fixed",
    }
    v = check_scope_upscale(params, scope_rank=vmp_profile.scope_rank)
    assert v.result == "pass"


def test_upscale_gate_fails_on_observed_only_cross_env_claim(vmp_profile):
    """The canonical tc3 finding: pinned at observation → claim cross_env
    without dataflow proof → fail."""
    params = {
        "scope_claim": "cross_env",
        "value_class": "observed",
    }
    v = check_scope_upscale(params, scope_rank=vmp_profile.scope_rank)
    assert v.result == "fail"
    assert "runtime-locked artefact" in v.reason


def test_upscale_gate_inactive_for_narrow_claims(vmp_profile):
    """Below the wide threshold, the gate doesn't fire — task_bound /
    env_bound claims are handled by ScopeBoundaryGate."""
    params = {
        "scope_claim": "task_bound",
        "value_class": "observed",
    }
    v = check_scope_upscale(params, scope_rank=vmp_profile.scope_rank)
    assert v.result == "pass"
    assert "not engaged" in v.reason


def test_upscale_gate_passes_with_appkey_fixed_function_category(vmp_profile):
    params = {
        "scope_claim": "cross_env",
        "constant_provenance_category": "appkey_fixed_function",
    }
    v = check_scope_upscale(params, scope_rank=vmp_profile.scope_rank)
    assert v.result == "pass"


def test_upscale_gate_fails_on_session_level_cross_env_claim(vmp_profile):
    params = {
        "scope_claim": "cross_env",
        "constant_provenance_category": "session_level_derived",
    }
    v = check_scope_upscale(params, scope_rank=vmp_profile.scope_rank)
    assert v.result == "fail"


def test_upscale_gate_disabled_returns_none():
    cfg = ScopeUpscaleConfig(enabled=False)
    assert check_scope_upscale({"scope_claim": "cross_env"}, cfg=cfg) is None


# ---------------------------------------------------------------------------
# B2 — wired into ConjunctiveGate
# ---------------------------------------------------------------------------


def test_upscale_probe_registered_as_mechanism():
    probe = ScopeUpscaleGateProbe()
    assert probe.mechanism is True


def test_conjunctive_gate_fails_on_observed_cross_env_claim(vmp_profile):
    gate = ConjunctiveGate(vmp_profile)
    ctx = ProbeContext(
        method="promote_to_finding",
        params={
            "scope_claim": "cross_env",
            "scope_observed": "cross_env",   # observed wide enough to pass B1
            "value_class": "observed",       # but B2 still demands dataflow proof
        },
        profile=vmp_profile,
    )
    result = gate.evaluate(ctx)
    assert "scope_upscale_gate" in result.failing_probes


def test_conjunctive_gate_passes_with_dataflow_proof(vmp_profile):
    gate = ConjunctiveGate(vmp_profile)
    ctx = ProbeContext(
        method="promote_to_finding",
        params={
            "scope_claim": "cross_env",
            "scope_observed": "cross_env",
            "value_class": "observed",
            "producer_dataflow": {"proof_of_invariance": True},
        },
        profile=vmp_profile,
    )
    result = gate.evaluate(ctx)
    upscale = next(
        v for v in result.mechanism_verdicts
        if v.probe == "scope_upscale_gate"
    )
    assert upscale.result == "pass"


# ---------------------------------------------------------------------------
# Lock A — scope gates are mechanism-locked (cannot be disabled / overridden)
# ---------------------------------------------------------------------------


def test_lock_a_scope_boundary_disable_rejected(tmp_path):
    """Mechanism-locked: subprofile cannot ``disable: scope_boundary_gate``."""
    import json
    from engine.profile.registry import ProfileMergeError

    pdir = tmp_path / "profiles"
    pdir.mkdir()
    # Mirror shipped base with the v0.4.0 mechanism set.
    (pdir / "base.json").write_text(json.dumps({
        "profile": "base",
        "probes": [
            {"name": "m1_success_audit",    "mechanism": True},
            {"name": "m3_bypass_block",     "mechanism": True},
            {"name": "constant_provenance", "mechanism": True},
            {"name": "value_provenance",    "mechanism": True},
            {"name": "watch_first_write",   "mechanism": True},
            {"name": "scope_boundary_gate", "mechanism": True},
            {"name": "scope_upscale_gate",  "mechanism": True},
        ],
    }))
    (pdir / "hostile.json").write_text(json.dumps({
        "profile": "hostile",
        "inherits": "base",
        "evidence_classes": [{"id": "A"}],
        "node_states": [{"name": "closed_form", "roles": ["closure_state"]}],
        "disable": ["scope_boundary_gate"],
    }))
    reg = ProfileRegistry(pdir)
    with pytest.raises(ProfileMergeError):
        reg.load_chain("hostile")


def test_lock_a_scope_upscale_override_rejected(tmp_path):
    import json
    from engine.profile.registry import ProfileMergeError

    pdir = tmp_path / "profiles"
    pdir.mkdir()
    (pdir / "base.json").write_text(json.dumps({
        "profile": "base",
        "probes": [
            {"name": "m1_success_audit",    "mechanism": True},
            {"name": "m3_bypass_block",     "mechanism": True},
            {"name": "constant_provenance", "mechanism": True},
            {"name": "value_provenance",    "mechanism": True},
            {"name": "watch_first_write",   "mechanism": True},
            {"name": "scope_boundary_gate", "mechanism": True},
            {"name": "scope_upscale_gate",  "mechanism": True},
        ],
    }))
    (pdir / "hostile.json").write_text(json.dumps({
        "profile": "hostile",
        "inherits": "base",
        "evidence_classes": [{"id": "A"}],
        "node_states": [{"name": "closed_form", "roles": ["closure_state"]}],
        "probes": [{"name": "scope_upscale_gate", "module": "x.y"}],
    }))
    reg = ProfileRegistry(pdir)
    with pytest.raises(ProfileMergeError):
        reg.load_chain("hostile")
