"""spec A — verdict -> next_actions (declarative remedy mapping on failure results).

Fixtures map 1:1 to the spec "Fixtures" section:
  (a) UNCLOSABLE-distinct-collapse (99 distinct < 100) -> collect_real_gold
      next-action (with why + minimal call);
  (b) a verdict with no mapping -> empty next_actions;
  (c) a second mapping added via the registry fires;
  (d) regression: verdict values / reasons byte-for-byte unchanged (next_actions is
      purely additive — absent from to_dict when empty).
"""

from __future__ import annotations

import pytest

from engine.next_actions import (
    NextActionRegistry,
    NextActionRule,
    PARITY_VECTORS_KIND,
    REGISTRY,
    reason_contains,
    register_next_action,
    suggest_next_actions,
)
from engine.setup_symex import ParityVector, check_parity_vectors


_WINDOW = (0x2000, 0x2100)


def _unclosable_99_lt_100() -> "object":
    """Build a real UNCLOSABLE report: 99 DISTINCT independent observed outputs,
    floor min_vectors=100 -> independent side collapses below the distinct floor.
    Every vector matches its own observed (predicted == observed), proving the
    verdict is judged on COHORT diversity, not the match floor."""
    vectors = [
        ParityVector(
            input_key=f"in{i}",
            observed=f"out{i}",
            predicted=f"out{i}",
            exec_id=f"e{i}",
        )
        for i in range(99)
    ]
    return check_parity_vectors(vectors, window=_WINDOW, min_vectors=100)


# --------------------------------------------------------------------------- (a)
def test_unclosable_distinct_collapse_surfaces_collect_real_gold():
    report = _unclosable_99_lt_100()
    assert report.verdict == "UNCLOSABLE"
    assert report.observed_distinct == 99

    # The proof-point: a collect_real_gold next-action is attached.
    assert len(report.next_actions) == 1
    action = report.next_actions[0]
    assert action["helper"] == "real_gold.collect_real_gold"
    # why explains the COHORT (not F) nature.
    assert "cohort" in action["why"].lower()
    assert "distinct" in action["why"].lower()
    # minimal call references the actual helper + key kwargs.
    assert "collect_real_gold(" in action["example"]
    assert "distinct_output_floor" in action["example"]

    # surfaced in to_dict.
    d = report.to_dict()
    assert d["next_actions"] == [dict(action)]
    assert d["verdict"] == "UNCLOSABLE"


# --------------------------------------------------------------------------- (b)
def test_verdict_with_no_mapping_has_empty_next_actions():
    # An EXACT verdict has no registered mapping -> empty next_actions, and the key
    # is ABSENT from to_dict (additive).
    vectors = [
        ParityVector(input_key=f"in{i}", observed=f"out{i}",
                     predicted=f"out{i}", exec_id=f"e{i}")
        for i in range(3)
    ]
    report = check_parity_vectors(vectors, window=_WINDOW, min_vectors=3)
    assert report.verdict == "EXACT"
    assert report.next_actions == ()

    d = report.to_dict()
    assert "next_actions" not in d  # byte-for-byte: no key when empty


# --------------------------------------------------------------------------- (c)
def test_second_mapping_added_via_registry_fires(monkeypatch):
    # A BLOCK verdict has no seed mapping. Register one at runtime -> it fires with
    # NO change to any verdict-construction call site (extensible registry).
    block_vectors = [
        # only 1/3 independent matches -> BLOCK (not UNCLOSABLE: 3 distinct observed)
        ParityVector(input_key="a", observed="o1", predicted="o1", exec_id="e1"),
        ParityVector(input_key="b", observed="o2", predicted="WRONG", exec_id="e2"),
        ParityVector(input_key="c", observed="o3", predicted="WRONG", exec_id="e3"),
    ]
    before = check_parity_vectors(block_vectors, window=_WINDOW, min_vectors=3)
    assert before.verdict == "BLOCK"
    assert before.next_actions == ()

    rule = register_next_action(
        report_kind=PARITY_VECTORS_KIND,
        verdict="BLOCK",
        helper="setup_symex.check_emit_self_consistency",
        why="BLOCK with mismatches: re-check the emitted F reproduces its own trace.",
        example="check_emit_self_consistency(intent, seeds)",
        predicate=reason_contains("mismatch"),
    )
    # Clean up so the runtime-added rule does not leak into other tests.
    monkeypatch.setattr(REGISTRY, "_rules", list(REGISTRY._rules))
    try:
        after = check_parity_vectors(block_vectors, window=_WINDOW, min_vectors=3)
        assert after.verdict == "BLOCK"  # verdict UNCHANGED
        assert len(after.next_actions) == 1
        assert after.next_actions[0]["helper"] == \
            "setup_symex.check_emit_self_consistency"
    finally:
        # remove the rule we appended (monkeypatch restored _rules to a copy, but be
        # explicit in case ordering changes)
        if rule in REGISTRY._rules:
            REGISTRY._rules.remove(rule)


# --------------------------------------------------------------------------- (d)
def test_regression_verdict_and_reasons_byte_for_byte_unchanged():
    # next_actions must not perturb verdict / reasons / any prior to_dict key.
    report = _unclosable_99_lt_100()
    d = report.to_dict()

    # The reason text is exactly what the gate produced — unchanged by the mapping.
    assert report.reasons == (
        "independent-side observed collapses to 99 < 100 distinct — no F can "
        "EXACT-close; fix the cohort (output-diverse seeds), not F",
        "only 99 independent cross-run vector(s) matched; need >= 100 (a 1/1 ≈ "
        "verifying the transform with the trace it was derived from — tautological)",
    )
    assert d["reasons"] == list(report.reasons)
    assert d["verdict"] == "UNCLOSABLE"

    # Every prior key still present + unchanged; only ``next_actions`` is added.
    prior_keys = {
        "window_pcs", "min_vectors", "independent_pass", "counted", "total",
        "mismatches", "determinism_ok", "determinism_seen", "verdict", "reasons",
        "observed_distinct", "independent_observed_distinct", "vectors", "kind",
    }
    assert prior_keys.issubset(d.keys())
    assert set(d.keys()) - prior_keys == {"next_actions"}


# --------------------------------------------------------------------------- unit
def test_registry_isolated_no_match_empty_and_dedup():
    reg = NextActionRegistry()
    assert reg.suggest("k", "V", ()) == ()
    r = NextActionRule(
        report_kind="k", verdict="V", helper="h", why="w", example="e")
    reg.register(r)
    reg.register(r)  # duplicate-equivalent rule
    out = reg.suggest("k", "V", ("any",))
    assert out == ({"helper": "h", "why": "w", "example": "e"},)  # de-duped


def test_misbehaving_predicate_never_breaks_path():
    reg = NextActionRegistry()

    def boom(_reasons, _report):
        raise RuntimeError("predicate blew up")

    reg.register(NextActionRule(
        report_kind="k", verdict="V", helper="h", why="w", example="e",
        predicate=boom))
    # Advisory-only: a broken predicate simply does not contribute.
    assert reg.suggest("k", "V", ("x",)) == ()
