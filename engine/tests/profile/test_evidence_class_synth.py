"""Evidence-class cap synthesis (PLAN §19.3 / §19.7 #5).

The acceptance §19.7 #5 says: if the profile declares
``[A, B, C]`` (strongest first), the cap synth picks B over A as
the more restrictive when both are present. If the profile is
reordered to ``[S, A, B, C]`` — adding S as a new strongest tier —
the kernel doesn't change a line; synth follows the new declaration.
"""

from __future__ import annotations

import pytest

from engine.profile import (
    EvidenceClassCap,
    EvidenceClassSpec,
    ProfileRegistry,
    Verdict,
    most_restrictive_class_id,
    synth_node_cap,
)


# ---------------------------------------------------------------------------
# most_restrictive_class_id
# ---------------------------------------------------------------------------


def _ec(*ids: str) -> tuple[EvidenceClassSpec, ...]:
    return tuple(EvidenceClassSpec(id=i) for i in ids)


def test_picks_weakest_under_a_b_c_profile():
    """A is strongest, C is weakest. Among {A, B}, B wins (more restrictive)."""
    assert most_restrictive_class_id(["A", "B"], _ec("A", "B", "C")) == "B"


def test_picks_weakest_with_three_inputs():
    assert most_restrictive_class_id(["A", "B", "C"], _ec("A", "B", "C")) == "C"


def test_single_input_is_returned():
    assert most_restrictive_class_id(["A"], _ec("A", "B", "C")) == "A"


def test_empty_input_is_none():
    assert most_restrictive_class_id([], _ec("A", "B", "C")) is None


def test_falsy_entries_filtered_out():
    """Empty-string ceilings come from underlying modules that report
    "no cap"; synth ignores them and picks among the real ones."""
    assert most_restrictive_class_id(["", "A", "B"], _ec("A", "B", "C")) == "B"


def test_reordered_profile_changes_the_answer():
    """§19.7 #5: a kernel-untouched reordering of evidence_classes
    rotates the synth's notion of "most restrictive" with no code
    change. Compare two profiles that disagree on whether A or B is
    the stronger class."""
    # Profile [A, B, C]: A is strongest. Of {A, B}, B is weaker → wins.
    assert most_restrictive_class_id(["A", "B"], _ec("A", "B", "C")) == "B"

    # Reordered profile [B, A, C]: now B is strongest, A is weaker.
    # Same input, OPPOSITE answer — synth followed the declaration.
    assert most_restrictive_class_id(["A", "B"], _ec("B", "A", "C")) == "A"

    # And adding a stronger tier at the front: [S, A, B, C].
    # Of {S, A}, A is weaker → wins. Zero kernel change.
    assert most_restrictive_class_id(["S", "A"], _ec("S", "A", "B", "C")) == "A"


def test_known_class_wins_over_unknown():
    """When inputs mix known (in profile) and unknown classes, the
    known one wins — synth can reason about its position, so it trusts
    that data over a label it cannot place. Also a small defence: an
    attacker can't poison the synth by emitting a verdict with a
    fabricated class label, because the real (known) caps still
    dominate."""
    # 'Z' is alphabetically greater than 'A' but unknown to profile —
    # synth picks A as the only one whose position it knows.
    assert most_restrictive_class_id(["A", "Z"], _ec("A", "B", "C")) == "A"


def test_alphabetic_fallback_when_all_inputs_unknown_to_profile():
    """Edge case: every input is unknown to the profile. With nothing
    to order, synth falls back to alphabetic-max so the caller still
    gets a deterministic answer (rather than None or arbitrary)."""
    assert most_restrictive_class_id(["X", "Y", "Z"], _ec("A", "B", "C")) == "Z"


def test_empty_profile_uses_alphabetic_fallback():
    """No profile context → step-4-compatible alphabetic-max."""
    assert most_restrictive_class_id(["A", "B"]) == "B"
    assert most_restrictive_class_id(["B", "A"]) == "B"


# ---------------------------------------------------------------------------
# synth_node_cap — combines Verdict caps into one node-level cap
# ---------------------------------------------------------------------------


def _verdict_capping(class_id: str, *, probe: str = "p") -> Verdict:
    return Verdict(
        probe=probe,
        result="pass",
        affects_evidence_class=EvidenceClassCap(class_id=class_id),
    )


def test_synth_no_caps_returns_none():
    assert synth_node_cap([], _ec("A", "B", "C")) is None
    no_cap_verdict = Verdict(probe="p", result="pass")
    assert synth_node_cap([no_cap_verdict], _ec("A", "B", "C")) is None


def test_synth_picks_weakest_cap():
    verdicts = [_verdict_capping("A"), _verdict_capping("B")]
    cap = synth_node_cap(verdicts, _ec("A", "B", "C"))
    assert cap is not None
    assert cap.class_id == "B"


def test_synth_reason_records_contributor_count():
    verdicts = [
        _verdict_capping("A", probe="p1"),
        _verdict_capping("B", probe="p2"),
        _verdict_capping("B", probe="p3"),
    ]
    cap = synth_node_cap(verdicts, _ec("A", "B", "C"))
    assert cap is not None
    assert "3 verdict cap(s)" in cap.reason
    # Two probes contributed to the winning class B
    assert "2 verdict(s)" in cap.reason


def test_synth_reordered_profile_changes_winner():
    """Same verdict set, different profile ordering — different cap.
    Demonstrates §19.7 #5 at the synth level. Uses {A, B} so both
    inputs are known to both candidate orderings."""
    verdicts = [_verdict_capping("A"), _verdict_capping("B")]

    # Profile [A, B, C] — B is weaker than A → B wins.
    cap_normal = synth_node_cap(verdicts, _ec("A", "B", "C"))
    assert cap_normal.class_id == "B"

    # Reordered [B, A, C] — now A is weaker than B → A wins.
    cap_swapped = synth_node_cap(verdicts, _ec("B", "A", "C"))
    assert cap_swapped.class_id == "A"


# ---------------------------------------------------------------------------
# Integration with the shipped VMP profile
# ---------------------------------------------------------------------------


def test_synth_against_shipped_vmp_profile():
    """End-to-end: load the real vmp_algorithm_extraction profile,
    feed in synthetic verdicts, get the right cap."""
    reg = ProfileRegistry()
    vmp = reg.load_chain("vmp_algorithm_extraction")

    verdicts = [_verdict_capping("A"), _verdict_capping("B"), _verdict_capping("C")]
    cap = synth_node_cap(verdicts, vmp.evidence_classes)
    assert cap.class_id == "C"
