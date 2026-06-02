"""capability_request.md §P1-1 / M1 — numeric-claim guard tests."""

from __future__ import annotations

import pytest

from engine.evidence import (
    ClaimStatus,
    EvidenceClass,
    GateClaims,
    NumericClaim,
    NumericClaimError,
    confirm_claim,
    invalidate_claim,
    looks_like_known_negative,
    record_numeric_claim,
)


def test_record_numeric_claim_defaults_pending():
    c = record_numeric_claim(
        "hook_eq_sign_rate", 0.0,
        layer="binary_sm3_body",
        evidence_class=EvidenceClass.IO_ORACLE,
    )
    assert c.status == ClaimStatus.PENDING_REVIEW
    assert c.evidence_class == EvidenceClass.IO_ORACLE


def test_record_numeric_claim_accepts_string_evidence_class():
    c = record_numeric_claim("foo", 1, layer="x", evidence_class="experiment")
    assert c.evidence_class == EvidenceClass.EXPERIMENT


def test_record_numeric_claim_rejects_unknown_class():
    with pytest.raises(NumericClaimError):
        record_numeric_claim("foo", 1, layer="x", evidence_class="vibes")


def test_record_numeric_claim_rejects_empty_metric_or_layer():
    with pytest.raises(NumericClaimError):
        record_numeric_claim("", 1, layer="x", evidence_class=EvidenceClass.IO_ORACLE)
    with pytest.raises(NumericClaimError):
        record_numeric_claim("m", 1, layer="", evidence_class=EvidenceClass.IO_ORACLE)


def test_confirm_requires_explicit_true():
    """confirm_claim(verifier_ack=1) must NOT promote — only True does."""
    c = record_numeric_claim("m", 1, layer="x", evidence_class=EvidenceClass.IO_ORACLE)
    # truthy-but-not-True does nothing
    same = confirm_claim(c, verifier_ack=1)  # type: ignore[arg-type]
    assert same.status == ClaimStatus.PENDING_REVIEW
    promoted = confirm_claim(c, verifier_ack=True)
    assert promoted.status == ClaimStatus.CONFIRMED


def test_invalidate_appends_reason_to_note():
    c = record_numeric_claim("m", 1, layer="x", evidence_class=EvidenceClass.IO_ORACLE,
                             note="initial")
    inv = invalidate_claim(c, "constant buffer")
    assert inv.status == ClaimStatus.INVALIDATED
    assert "initial" in (inv.note or "")
    assert "constant buffer" in (inv.note or "")


def test_known_negative_pattern_matches_reference_target_blob():
    """The e9a86ab9 prefix is the canonical reference target 32B digest-export
    constant. The guard must catch it."""
    hit = looks_like_known_negative("e9a86ab9deadbeefcafef00d" + "00" * 20)
    assert hit == "e9a86ab9"


def test_known_negative_pattern_clean_blob_returns_none():
    assert looks_like_known_negative("00" * 32) is None
    assert looks_like_known_negative("") is None


def test_gate_claims_has_unconfirmed_until_all_promoted():
    c1 = record_numeric_claim("a", 1, layer="x", evidence_class=EvidenceClass.IO_ORACLE)
    c2 = record_numeric_claim("b", 2, layer="x", evidence_class=EvidenceClass.STATIC_ELF)
    bundle = GateClaims(claims=(c1, c2))
    assert bundle.has_unconfirmed()
    c1p = confirm_claim(c1, verifier_ack=True)
    c2p = confirm_claim(c2, verifier_ack=True)
    bundle2 = GateClaims(claims=(c1p, c2p))
    assert not bundle2.has_unconfirmed()


def test_numeric_claim_serialises_to_dict():
    c = record_numeric_claim("hook_eq", 0, layer="L", evidence_class="io_oracle",
                             artifact_path="/tmp/x.json")
    d = c.to_dict()
    assert d["metric"] == "hook_eq"
    assert d["evidence_class"] == "io_oracle"
    assert d["status"] == "pending_review"
    assert d["artifact_path"] == "/tmp/x.json"
