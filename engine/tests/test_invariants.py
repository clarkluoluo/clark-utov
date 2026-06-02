"""capability_request.md §P1-2 / M8 — parity-invariant guard tests."""

from __future__ import annotations

from engine.invariants import annotate_report, check_invariants


def test_clean_report_no_failures():
    report = {
        "hook_src_valid":       100,
        "hook_digest_eq_sign":  1.0,
        "vectors_total":        50,
        "sm3_input_lens_seen":  [22, 38, 70],
        "hook_digest_unique_count": 50,
        "numeric_claims":       [{"metric": "x"}],
        "archival_allowed":     True,
    }
    assert check_invariants(report) == []


def test_hook_valid_but_no_match_flags_invariant():
    """The canonical reference target trap: 626 valids + 0% match."""
    report = {"hook_src_valid": 626, "hook_digest_eq_sign": 0.0}
    failures = check_invariants(report)
    names = {f["name"] for f in failures}
    assert "hook_valid_but_no_match" in names


def test_constant_input_len_across_many_vectors_flags():
    report = {
        "sm3_input_lens_seen": [68, 68, 68, 68, 68],
        "vectors_total":       5,
    }
    failures = check_invariants(report)
    names = {f["name"] for f in failures}
    assert "input_len_constant_but_many_vectors" in names


def test_pass_rate_without_numeric_claims_flags():
    report = {"hook_eq_sign_rate": 0.9873}
    failures = check_invariants(report)
    names = {f["name"] for f in failures}
    assert "pass_rate_without_evidence_class" in names


def test_target_success_without_archival_allowed_flags():
    report = {"target_success": True}
    failures = check_invariants(report)
    names = {f["name"] for f in failures}
    assert "target_success_without_archival_allowed" in names


def test_constant_buffer_unique_count_threshold():
    """vectors=10, hook_digest_unique_count=1 → fail."""
    report = {"vectors_total": 10, "hook_digest_unique_count": 1}
    failures = check_invariants(report)
    names = {f["name"] for f in failures}
    assert "hook_digest_unique_count_below_threshold" in names


def test_annotate_report_writes_failures_field():
    report = {"hook_src_valid": 1, "hook_digest_eq_sign": 0.0}
    out = annotate_report(report)
    assert out["invariants_failed"], "must record at least one failure"
    assert all("message" in f for f in out["invariants_failed"])


def test_metrics_under_nested_key_also_checked():
    """Predicates tolerate both flat and `metrics` nested layouts."""
    report = {"metrics": {"hook_src_valid": 1, "hook_digest_eq_sign": 0.0}}
    failures = check_invariants(report)
    names = {f["name"] for f in failures}
    assert "hook_valid_but_no_match" in names
