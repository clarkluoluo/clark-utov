"""Goal-aware advisor (spec #6, thin slice) — read-only evidence-state view +
NON-blocking rule advisor.

Pins the spec §Fixtures contract:
  (a) ``evidence_state`` composes sink/boundary/parity/recovery/pacing from the
      REAL source modules (closure_classification / progress / authority_projection
      / a ledger-entry shape) with a ``sources`` provenance map;
  (b) the over-investment rule fires the NON-blocking suggestion (TC2 proof-point:
      phase_E long, confirmed=0);
  (c) an unreadable signal → ``None`` + a ``sources`` gap, never fabricated;
  (d) a NEW rule added via the registry fires with no other change;
  and the invariant that EVERY advisory is ``blocking is False``.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.advisor import (
    RULES,
    Advisory,
    EvidenceState,
    advise,
    evidence_state,
    register_rule,
    rule_subline_overinvested_far_from_acceptance,
)
from engine.closure_classification import classify_closure
from engine.authority_projection import project_authority

# A generic goal_spec (the TaskSpec concept — A8①). NOT case-specific.
GOAL_SPEC = {
    "goal": "recover the transform",
    "acceptance_criteria": ["output_sink_confirmed", "provenance_closed", "parity_exact"],
    "suggested_phase_order": ["io_observe", "provenance", "candidate", "parity"],
}


# A minimal LedgerEntry-like stand-in (the shape cvd_ledger.LedgerEntry exposes:
# an ``is_closed`` property). Used as a fake ledger pull.
@dataclass(frozen=True)
class FakeLedgerEntry:
    is_closed: bool


# --------------------------------------------------------------------------- #
# (a) evidence_state composes from the REAL source modules + sources map.
# --------------------------------------------------------------------------- #


def test_evidence_state_composes_real_sources_with_provenance_map():
    # Real closure verdict from the real classifier (no re-derivation in advisor).
    closure = classify_closure(
        structural_closed=True,
        output_sink_confirmed=True,
        provenance_closed=True,
        parity_exact=False,        # on-chain candidate, not yet对拍
    )
    # Real authority projection (a current claim).
    claims = project_authority("CaseX", [{"claim": "a"}])
    # Fake ledger pull (LedgerEntry-shaped) + a progress-snapshot-shaped dict.
    ledger = [FakeLedgerEntry(is_closed=True), FakeLedgerEntry(is_closed=False)]
    progress = {"pacing": "ok", "closure_rate_per_min": 1.5}

    ev = evidence_state(ledger=ledger, claims=claims, progress=progress, closure=closure)

    assert ev.sink_confirmed is True
    assert ev.boundary_explicit is True
    assert ev.parity_candidate_exists is False
    assert ev.recovery_confirmed_count == 1          # one closed entry
    assert ev.pacing == {"pacing": "ok", "closure_rate_per_min": 1.5}

    # sources provenance map names where each signal came from (no gaps here).
    assert ev.sources["sink"] == "closure_classification.output_sink_confirmed"
    assert ev.sources["boundary"] == "closure_classification.provenance_closed"
    assert ev.sources["parity"] == "closure_classification.parity_exact"
    assert ev.sources["recovery"].startswith("cvd_ledger")
    assert ev.sources["pacing"] == "progress.ProgressTracker.snapshot"
    assert ev.sources["authority"] == "authority_projection.project_authority"
    # to_dict round-trips the view.
    assert ev.to_dict()["sink_confirmed"] is True


def test_evidence_state_accepts_real_progress_snapshot_object():
    from engine.cost import CostMeter
    from engine.progress import Tracker

    snap = Tracker(CostMeter()).snapshot()       # a real ProgressSnapshot
    ev = evidence_state(progress=snap)
    assert ev.pacing is not None
    assert "pacing" in ev.pacing
    assert ev.sources["pacing"] == "progress.ProgressTracker.snapshot"


# --------------------------------------------------------------------------- #
# (b) the over-investment rule fires the NON-blocking suggestion (TC2 proof-point).
# --------------------------------------------------------------------------- #


def test_overinvestment_rule_fires_nonblocking_suggestion():
    # phase_E long with confirmed=0: pacing stalled, no recovery, parity not exact.
    closure = classify_closure(
        structural_closed=True,
        output_sink_confirmed=False,
        provenance_closed=False,
        parity_exact=False,
    )
    ev = evidence_state(
        closure=closure,
        ledger=0,                                  # confirmed=0
        progress={"pacing": "stalled", "closure_rate_per_min": 0.0},
    )
    out = advise(GOAL_SPEC, ev)
    assert len(out) == 1
    adv = out[0]
    assert adv.level == "SUGGEST"
    assert adv.trigger == "subline_overinvested_far_from_acceptance"
    assert adv.rebalance_to == "boundary_explicit_candidate"
    assert "held-out parity" in adv.message
    assert adv.blocking is False                   # NON-blocking by construction


def test_rule_silent_when_not_overinvested_or_already_close():
    # Healthy pacing → no advisory (no false "all good", just an empty list).
    ev_ok = evidence_state(ledger=0, progress={"pacing": "ok"})
    assert advise(GOAL_SPEC, ev_ok) == []

    # Over-invested BUT already at acceptance (parity exact) → not far → silent.
    closure_done = classify_closure(
        structural_closed=True,
        output_sink_confirmed=True,
        provenance_closed=True,
        parity_exact=True,
    )
    ev_done = evidence_state(
        closure=closure_done, ledger=0,
        progress={"pacing": "stalled"},
    )
    assert advise(GOAL_SPEC, ev_done) == []

    # Over-invested + a confirmed recovery already exists → not far → silent.
    ev_recovered = evidence_state(
        ledger=2, progress={"pacing": "stalled"},
    )
    assert advise(GOAL_SPEC, ev_recovered) == []


def test_rule_function_directly_returns_advisory_or_none():
    far = EvidenceState(parity_candidate_exists=False, recovery_confirmed_count=0,
                        pacing={"pacing": "stalled"})
    assert rule_subline_overinvested_far_from_acceptance(GOAL_SPEC, far) is not None
    near = EvidenceState(parity_candidate_exists=True, pacing={"pacing": "stalled"})
    assert rule_subline_overinvested_far_from_acceptance(GOAL_SPEC, near) is None


# --------------------------------------------------------------------------- #
# (c) an unreadable signal → None + sources gap, NOT fabricated.
# --------------------------------------------------------------------------- #


def test_unreadable_signal_is_null_plus_gap_not_fabricated():
    ev = evidence_state()                          # nothing supplied
    assert ev.sink_confirmed is None
    assert ev.boundary_explicit is None
    assert ev.parity_candidate_exists is None
    assert ev.pacing is None
    # recovery is a count: an unread ledger is 0 BUT the gap is explicit (not a
    # fabricated "0 confirmed" verdict).
    assert ev.recovery_confirmed_count == 0
    for key in ("sink", "boundary", "parity", "recovery", "pacing", "phase", "authority"):
        assert ev.sources[key].startswith("gap:"), key

    # A null pacing → the over-investment rule does NOT fire (no fabricated verdict).
    assert advise(GOAL_SPEC, ev) == []


def test_partial_source_only_fills_what_is_readable():
    # Only progress supplied: pacing readable, closure signals stay null+gap.
    ev = evidence_state(progress={"pacing": "stalled"})
    assert ev.pacing == {"pacing": "stalled", "closure_rate_per_min": None}
    assert ev.sink_confirmed is None
    assert ev.sources["sink"].startswith("gap:")
    assert ev.sources["pacing"] == "progress.ProgressTracker.snapshot"


# --------------------------------------------------------------------------- #
# (d) a NEW rule via the registry fires with no other change.
# --------------------------------------------------------------------------- #


def test_new_rule_via_registry_fires():
    def rule_sink_unconfirmed(goal_spec, evidence):
        if evidence.sink_confirmed is False:
            return Advisory(
                level="SUGGEST",
                trigger="sink_unconfirmed",
                message="output sink not confirmed → confirm the writer first",
                rebalance_to="output_sink_confirm",
                blocking=False,
            )
        return None

    ev = EvidenceState(sink_confirmed=False)
    # Passed via the `rules=` override — proves adding a rule is one entry, no
    # change to the rest.
    out = advise(GOAL_SPEC, ev, rules=[rule_sink_unconfirmed])
    assert [a.trigger for a in out] == ["sink_unconfirmed"]
    assert out[0].blocking is False


def test_register_rule_appends_to_module_registry():
    before = len(RULES)

    def noop_rule(goal_spec, evidence):
        return None

    try:
        register_rule(noop_rule)
        assert len(RULES) == before + 1
        assert RULES[-1] is noop_rule
    finally:
        RULES.remove(noop_rule)                    # keep the module registry clean
    assert len(RULES) == before


# --------------------------------------------------------------------------- #
# Invariant: every advisory the advisor can emit is NON-blocking.
# --------------------------------------------------------------------------- #


def test_advisor_never_blocks():
    # A rule that (wrongly) tries to block is rejected by the advise() invariant.
    def bad_rule(goal_spec, evidence):
        return Advisory(level="SUGGEST", trigger="bad", message="x", blocking=True)

    import pytest
    with pytest.raises(AssertionError):
        advise(GOAL_SPEC, EvidenceState(), rules=[bad_rule])
