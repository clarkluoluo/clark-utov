"""v0.4.0 V4 — use-case fork rule (§19.9 vmp #5).

Domain probe ``use_case_fork`` (vmp_algorithm_extraction). Encodes
the current-vs-cross-context decision point as a declarable rule:

  * ``use_case = current_context_reproduction`` → scope claim must
    stay ≤ env_bound.
  * ``use_case = cross_context_claim`` → scope claim must be cross_env
    AND backed by dataflow proof / closed-form attestation / cross-env-safe
    constant_provenance category.

Probe is non-mechanism (domain semantics — open layer, freely
overridable / disable-able).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.profile import (
    BASE_PROFILE_NAME,
    ProbeContext,
    ProfileRegistry,
)
from engine.profile.probe_runtime import get_builtin_probe_class
from engine.profile.probes.use_case_fork import UseCaseForkProbe


VMP_PROFILE = "vmp_algorithm_extraction"


@pytest.fixture()
def vmp_profile():
    return ProfileRegistry().load_chain(VMP_PROFILE)


# ---------------------------------------------------------------------------
# Probe registered as domain (non-mechanism) under vmp
# ---------------------------------------------------------------------------


def test_use_case_fork_registered_as_builtin():
    assert get_builtin_probe_class("use_case_fork") is UseCaseForkProbe


def test_use_case_fork_is_non_mechanism(vmp_profile):
    probe = next(p for p in vmp_profile.probes if p.name == "use_case_fork")
    assert probe.mechanism is False
    assert "use_case_fork" not in vmp_profile.mechanism_probe_names


# ---------------------------------------------------------------------------
# current_context_reproduction branch
# ---------------------------------------------------------------------------


def test_current_context_with_task_bound_claim_passes(vmp_profile):
    ctx = ProbeContext(
        method="finalize_verdict",
        params={
            "use_case": "current_context_reproduction",
            "scope_claim": "task_bound",
        },
        profile=vmp_profile,
    )
    verdict = UseCaseForkProbe().run(ctx)
    assert verdict.result == "pass"


def test_current_context_with_env_bound_claim_passes(vmp_profile):
    ctx = ProbeContext(
        method="finalize_verdict",
        params={
            "use_case": "current_context_reproduction",
            "scope_claim": "env_bound",
        },
        profile=vmp_profile,
    )
    verdict = UseCaseForkProbe().run(ctx)
    assert verdict.result == "pass"


def test_current_context_with_cross_env_claim_fails(vmp_profile):
    """The "本可下放却上抛" fix: a current-context use case asking for
    cross_env is over-extension."""
    ctx = ProbeContext(
        method="finalize_verdict",
        params={
            "use_case": "current_context_reproduction",
            "scope_claim": "cross_env",
        },
        profile=vmp_profile,
    )
    verdict = UseCaseForkProbe().run(ctx)
    assert verdict.result == "fail"
    assert "exceeds env_bound" in verdict.evidence["reason"]


# ---------------------------------------------------------------------------
# cross_context_claim branch
# ---------------------------------------------------------------------------


def test_cross_context_with_closed_form_passes(vmp_profile):
    ctx = ProbeContext(
        method="finalize_verdict",
        params={
            "use_case": "cross_context_claim",
            "scope_claim": "cross_env",
            "value_class": "closed_form",
        },
        profile=vmp_profile,
    )
    verdict = UseCaseForkProbe().run(ctx)
    assert verdict.result == "pass"


def test_cross_context_with_dataflow_proof_passes(vmp_profile):
    ctx = ProbeContext(
        method="finalize_verdict",
        params={
            "use_case": "cross_context_claim",
            "scope_claim": "cross_env",
            "producer_dataflow": {"proof_of_invariance": True},
        },
        profile=vmp_profile,
    )
    verdict = UseCaseForkProbe().run(ctx)
    assert verdict.result == "pass"


def test_cross_context_with_hardcoded_fixed_cp_category_passes(vmp_profile):
    ctx = ProbeContext(
        method="finalize_verdict",
        params={
            "use_case": "cross_context_claim",
            "scope_claim": "cross_env",
            "constant_provenance_category": "hardcoded_fixed",
        },
        profile=vmp_profile,
    )
    verdict = UseCaseForkProbe().run(ctx)
    assert verdict.result == "pass"


def test_cross_context_without_proof_fails(vmp_profile):
    """The canonical tc3 finding: observed-only cross-context claim
    needs dataflow proof to land."""
    ctx = ProbeContext(
        method="finalize_verdict",
        params={
            "use_case": "cross_context_claim",
            "scope_claim": "cross_env",
            "value_class": "observed",
        },
        profile=vmp_profile,
    )
    verdict = UseCaseForkProbe().run(ctx)
    assert verdict.result == "fail"


def test_cross_context_with_narrower_claim_fails(vmp_profile):
    """A cross-context use case can't advertise task_bound — the use
    case demands the widest scope plus dataflow proof."""
    ctx = ProbeContext(
        method="finalize_verdict",
        params={
            "use_case": "cross_context_claim",
            "scope_claim": "task_bound",
            "producer_dataflow": {"proof_of_invariance": True},
        },
        profile=vmp_profile,
    )
    verdict = UseCaseForkProbe().run(ctx)
    assert verdict.result == "fail"
    assert "narrower than cross_env" in verdict.evidence["reason"]


# ---------------------------------------------------------------------------
# Missing / unknown use_case → undetermined
# ---------------------------------------------------------------------------


def test_no_use_case_returns_undetermined(vmp_profile):
    ctx = ProbeContext(
        method="finalize_verdict",
        params={"scope_claim": "env_bound"},
        profile=vmp_profile,
    )
    verdict = UseCaseForkProbe().run(ctx)
    assert verdict.result == "undetermined"


def test_unknown_use_case_returns_undetermined(vmp_profile):
    ctx = ProbeContext(
        method="finalize_verdict",
        params={"use_case": "esoteric_target_specific_thing"},
        profile=vmp_profile,
    )
    verdict = UseCaseForkProbe().run(ctx)
    assert verdict.result == "undetermined"


# ---------------------------------------------------------------------------
# Open-layer: domain probe can be disabled by a subprofile
# ---------------------------------------------------------------------------


def test_subprofile_can_disable_use_case_fork(tmp_path):
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
    (pdir / "vmp_algorithm_extraction.json").write_text(json.dumps({
        "profile": "vmp_algorithm_extraction",
        "inherits": "base",
        "evidence_classes": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
        "node_states": [{"name": "closed_form", "roles": ["closure_state"]}],
        "probes": [{"name": "use_case_fork"}],
    }))
    (pdir / "vmp_no_fork.json").write_text(json.dumps({
        "profile": "vmp_no_fork",
        "inherits": "vmp_algorithm_extraction",
        "disable": ["use_case_fork"],
    }))
    reg = ProfileRegistry(pdir)
    merged = reg.load_chain("vmp_no_fork")
    assert "use_case_fork" not in {p.name for p in merged.probes}
