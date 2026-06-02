"""v0.4.0 B3 + B4 — M1 audit triage + env_limit_rows carve-out.

B3 (§19.9 base #9) — separate M1's deterministic checks (dimension
coverage / overfit / scope / closure / sample count) from the
non-deterministic ones (target intent / budget / "is this success?").
A success-audit result now carries a ``triage`` field:
``agent_self_resolved`` when every weak axis is deterministic, or
``user_decision_required`` when the caller has flagged a subjective
input.  Configurable subjective-key vocabulary so other domains can
extend the list.

B4 (§19.9 base #7) — env-limit row carve-out. Pass-rate / dimension
coverage math currently treats runner-imposed nulls (a runner that
can't synthesise a signature for a sample) the same as algorithm
failures. Opt-in: top-level ``env_limit_rows`` count OR per-sample
``env_limit: true`` flag strips those rows before grading so an
algorithm-correct claim isn't penalised for the runner's environment
ceiling.
"""

from __future__ import annotations

import pytest

from engine.m1_success_audit import (
    DEFAULT_SUBJECTIVE_INPUT_KEYS,
    M1AuditConfig,
    SuccessAuditResult,
    audit_success_claim,
)


# ---------------------------------------------------------------------------
# B3 — triage default = agent_self_resolved
# ---------------------------------------------------------------------------


def _clean_archive_params(**overrides) -> dict:
    """An A-grade archival claim — no subjective inputs, every
    deterministic axis green."""
    base = {
        "target_success": True,
        "success_dependencies": ["input_len", "prefix"],
        "samples": [
            {"input_len": 8,  "prefix": "AA", "passed": True},
            {"input_len": 9,  "prefix": "BB", "passed": True},
            {"input_len": 10, "prefix": "CC", "passed": True},
            {"input_len": 11, "prefix": "DD", "passed": True},
            {"input_len": 12, "prefix": "EE", "passed": True},
        ],
        "pass_rate": 1.0,
        "scope": "cross_session",
        "closure_paths": [
            {"name": "cfbc",    "digest": "deadbeef"},
            {"name": "formula", "digest": "deadbeef"},
        ],
    }
    base.update(overrides)
    return base


def test_default_triage_is_agent_self_resolved():
    result = audit_success_claim("promote_to_finding", _clean_archive_params())
    assert result is not None
    assert result.action == "allow"
    assert result.triage == "agent_self_resolved"
    assert result.subjective_inputs == ()


def test_subjective_flag_routes_to_user():
    params = _clean_archive_params(
        user_target_intent_review_required=True,
    )
    result = audit_success_claim("promote_to_finding", params)
    assert result is not None
    assert result.triage == "user_decision_required"
    assert "user_target_intent_review_required" in result.subjective_inputs


def test_budget_decision_flag_routes_to_user():
    params = _clean_archive_params(budget_decision_required=True)
    result = audit_success_claim("promote_to_finding", params)
    assert result is not None
    assert result.triage == "user_decision_required"
    assert "budget_decision_required" in result.subjective_inputs


def test_multiple_subjective_keys_all_listed():
    params = _clean_archive_params(
        user_target_intent_review_required=True,
        subjective_success_review=True,
    )
    result = audit_success_claim("promote_to_finding", params)
    assert result is not None
    assert set(result.subjective_inputs) == {
        "user_target_intent_review_required",
        "subjective_success_review",
    }
    assert result.triage == "user_decision_required"


def test_subjective_flag_does_not_change_grade():
    """Triage and grade are orthogonal — the audit grade is still A
    when every deterministic check passes; triage only says *who*
    decides what to do with that grade."""
    params = _clean_archive_params(user_target_intent_review_required=True)
    result = audit_success_claim("promote_to_finding", params)
    assert result is not None
    assert result.evidence_class == "A"
    assert result.action == "allow"
    assert result.triage == "user_decision_required"


def test_custom_subjective_keys_via_config():
    cfg = M1AuditConfig(subjective_keys=("my_custom_subjective_flag",))
    params = _clean_archive_params(my_custom_subjective_flag=True)
    result = audit_success_claim("promote_to_finding", params, cfg=cfg)
    assert result is not None
    assert result.triage == "user_decision_required"
    assert "my_custom_subjective_flag" in result.subjective_inputs


def test_existing_archival_tests_emit_triage_field():
    """Backward-compat sanity: the new field appears in to_dict()."""
    result = audit_success_claim("promote_to_finding", _clean_archive_params())
    payload = result.to_dict()
    assert payload["triage"] == "agent_self_resolved"
    assert payload["subjective_inputs"] == []


def test_default_subjective_keys_export():
    assert "user_target_intent_review_required" in DEFAULT_SUBJECTIVE_INPUT_KEYS
    assert "budget_decision_required" in DEFAULT_SUBJECTIVE_INPUT_KEYS


# ---------------------------------------------------------------------------
# B4 — env_limit_rows carve-out
# ---------------------------------------------------------------------------


def _archival_params_with_env_limit_count(env_limit: int) -> dict:
    """4 rows pass, 4 rows are env-limited nulls. Raw pass_rate = 0.5;
    adjusted pass_rate = 1.0."""
    return {
        "target_success": True,
        "success_dependencies": ["input_len"],
        "samples": [
            {"input_len": 8,  "passed": True},
            {"input_len": 9,  "passed": True},
            {"input_len": 10, "passed": True},
            {"input_len": 11, "passed": True},
            {"input_len": 12, "passed": False},  # env-limit row
            {"input_len": 13, "passed": False},  # env-limit row
            {"input_len": 14, "passed": False},  # env-limit row
            {"input_len": 15, "passed": False},  # env-limit row
        ],
        "checked": 8,
        "passed": 4,
        "env_limit_rows": env_limit,
        "scope": "cross_session",
        "closure_paths": [
            {"name": "cfbc", "digest": "abc"},
            {"name": "formula", "digest": "abc"},
        ],
    }


def test_env_limit_top_level_count_strips_nulls_from_pass_rate():
    params = _archival_params_with_env_limit_count(env_limit=4)
    result = audit_success_claim("promote_to_finding", params)
    assert result is not None
    assert result.env_limit_rows_excluded == 4
    assert result.adjusted_pass_rate == 1.0
    # Raw pass_rate is preserved for transparency.
    assert pytest.approx(result.pass_rate, rel=1e-3) == 0.5


def test_env_limit_carve_out_promotes_above_reject_floor():
    """Without the carve-out, pass_rate=0.5 ≤ min_pass_rate (0.95) →
    reject. With it, adjusted_pass_rate=1.0 → allow/downgrade."""
    params = _archival_params_with_env_limit_count(env_limit=4)
    result = audit_success_claim("promote_to_finding", params)
    assert result is not None
    assert result.action in ("allow", "downgrade")


def test_env_limit_zero_keeps_legacy_behaviour():
    params = _archival_params_with_env_limit_count(env_limit=0)
    result = audit_success_claim("promote_to_finding", params)
    assert result is not None
    assert result.env_limit_rows_excluded == 0
    assert result.adjusted_pass_rate is None
    # Raw 0.5 fails the min_pass_rate floor → reject.
    assert result.action == "reject"


def test_env_limit_per_sample_flag_counts_automatically():
    """Per-sample ``env_limit: true`` flag triggers the carve-out
    without an explicit top-level count."""
    params = {
        "target_success": True,
        "success_dependencies": ["input_len"],
        "samples": [
            {"input_len": 8,  "passed": True},
            {"input_len": 9,  "passed": True},
            {"input_len": 10, "passed": True},
            {"input_len": 11, "passed": True},
            {"input_len": 12, "passed": False, "env_limit": True},
            {"input_len": 13, "passed": False, "env_limit": True},
        ],
        "checked": 6,
        "passed": 4,
        "scope": "cross_session",
        "closure_paths": [
            {"name": "cfbc", "digest": "abc"},
            {"name": "formula", "digest": "abc"},
        ],
    }
    result = audit_success_claim("promote_to_finding", params)
    assert result is not None
    assert result.env_limit_rows_excluded == 2
    assert result.adjusted_pass_rate == 1.0
    # env_limit rows must not be counted toward sample_count either —
    # the runner-imposed nulls aren't graded coverage.
    assert result.sample_count == 4


def test_env_limit_note_in_audit_output():
    """When the carve-out fires, the result carries an explanatory
    note so the envelope makes the adjustment visible."""
    params = _archival_params_with_env_limit_count(env_limit=4)
    result = audit_success_claim("promote_to_finding", params)
    assert result is not None
    assert any("env-limit carve-out" in note for note in result.notes)
