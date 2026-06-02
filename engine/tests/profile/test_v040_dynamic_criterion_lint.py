"""v0.4.0 B6 — dynamic-criterion lint (§19.9 base #6).

A review-gate criterion that pins a concrete byte offset (e.g.
``"value at offset 0x18"``) becomes a false seam when the implementation
drifts. The lint flags such criteria so profile authors are forced to
frame the rule as a scan-style descriptor instead
(``"canonical slot located by scan over <pattern>"``).

The check fires on :class:`Profile` JSON before merge — both base and
domain are checked. Domain authors who really do mean a specific offset
must mention a scan-style keyword in the same rule to opt out (the
offset then reads as illustrative, not load-bearing).
"""

from __future__ import annotations

import pytest

from engine.profile.lint import lint_dynamic_criteria
from engine.profile.types import (
    GateSpec,
    Profile,
    ProbeSpec,
    ScopeRule,
)


def _profile(*, gates: tuple[GateSpec, ...] = (), scope_rules: tuple[ScopeRule, ...] = ()) -> Profile:
    return Profile(
        name="under_test",
        inherits="base",
        evidence_classes=(),
        node_states=(),
        probes=(),
        gates=gates,
        scope_semantics=scope_rules,
        is_base=False,
    )


# ---------------------------------------------------------------------------
# Pinned-offset criteria → violation
# ---------------------------------------------------------------------------


def test_gate_with_hex_offset_in_rule_violates():
    profile = _profile(gates=(GateSpec(
        id="canonical_slot",
        rule="canonical slot is at +0x18 from base",
    ),))
    violations = lint_dynamic_criteria(profile)
    assert len(violations) == 1
    assert "+0x18" in violations[0]
    assert "canonical_slot" in violations[0]


def test_gate_with_decimal_offset_phrase_violates():
    profile = _profile(gates=(GateSpec(
        id="g1",
        rule="value is at offset 24 from struct head",
    ),))
    violations = lint_dynamic_criteria(profile)
    assert len(violations) == 1
    assert "offset 24" in violations[0].lower()


def test_multiple_gates_each_violation_listed_independently():
    profile = _profile(gates=(
        GateSpec(id="a", rule="at offset 0x10"),
        GateSpec(id="b", rule="ratio between adjacent lengths is 4/3"),
        GateSpec(id="c", rule="byte at +0x44 must equal sentinel"),
    ),)
    violations = lint_dynamic_criteria(profile)
    assert len(violations) == 2
    ids_flagged = {v.split("gate '")[1].split("'")[0] for v in violations}
    assert ids_flagged == {"a", "c"}


# ---------------------------------------------------------------------------
# Dynamic-language opt-out — offset literal + scan keyword passes
# ---------------------------------------------------------------------------


def test_offset_with_scan_keyword_passes():
    """Offset appears as an illustrative addendum to a scan-style rule."""
    profile = _profile(gates=(GateSpec(
        id="canonical_slot",
        rule="scan for signature SIGCANON; the canonical slot then sits at +0x18 of the match",
    ),))
    assert lint_dynamic_criteria(profile) == []


def test_offset_with_pattern_keyword_passes():
    profile = _profile(gates=(GateSpec(
        id="g",
        rule="match pattern P, derive slot from +0x4 of match",
    ),))
    assert lint_dynamic_criteria(profile) == []


def test_clean_dynamic_rule_passes():
    profile = _profile(gates=(GateSpec(
        id="g",
        rule="length chain adjacent ratio explainable by base64/zlib/identity",
    ),))
    assert lint_dynamic_criteria(profile) == []


# ---------------------------------------------------------------------------
# Empty / no-criteria profile
# ---------------------------------------------------------------------------


def test_empty_profile_no_violations():
    profile = _profile()
    assert lint_dynamic_criteria(profile) == []


def test_gate_with_empty_rule_no_violation():
    profile = _profile(gates=(GateSpec(id="g", rule=""),))
    assert lint_dynamic_criteria(profile) == []
