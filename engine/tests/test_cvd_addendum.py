"""CVD Addendum: PLACE contract (§A), credibility EVALUATED by us (§B), and the
credibility law — orders verification, never replaces it (§C).
"""

from __future__ import annotations

import pytest

from engine.cvd import (
    AgentSubmission,
    CvdOutcome,
    CvdState,
    PlacementError,
    evaluate_credibility,
    place,
    run_cvd,
)
from engine.types import Instruction, MemOp


def _ins(idx, mnem, mem=()):
    return Instruction(idx=idx, pc=0x70000 + idx * 4, bytes_=b"\x00\x00\x00\x00",
                       mnemonic=mnem, regs_read={}, regs_write={}, mem=mem)


def _le(b):
    return int.from_bytes(b, "little")


EXPECTED = bytes([0x34, 0x15, 0x5f, 0xe9])
SCRATCH = bytes([0x00, 0x11, 0x22, 0x33])
SCR = 0x70f80     # the benchmark's scratch "sink" (no oracle match)
REAL = 0x2000     # the real output buffer == expected


# --- §A: PLACE refuses without the mandatory preconditions -------------------

def test_place_refuses_without_runner():
    with pytest.raises(PlacementError):
        place([_ins(0, "nop")], EXPECTED, has_runner=False)


def test_place_refuses_without_oracle():
    with pytest.raises(PlacementError):
        place([_ins(0, "nop")], b"")          # no expected/oracle


def test_place_refuses_without_trace():
    with pytest.raises(PlacementError):
        place([], EXPECTED)                   # no trace


def test_place_ingests_submissions_as_candidates():
    state, ingested = place(
        [_ins(0, "nop")], EXPECTED,
        submissions=[AgentSubmission(locus=SCR, provenance="agent_hypothesis",
                                     evidence=[{"note": "looked like a store"}])])
    assert len(ingested) == 1
    assert ingested[0].locus == SCR
    assert ingested[0].provenance == "agent_hypothesis"


# --- §B: credibility is evaluated from evidence, not supplied ----------------

def test_credibility_bare_assumption_is_lowest():
    state = CvdState([_ins(0, "nop")], EXPECTED)
    assert evaluate_credibility(SCR, [], state) == 0.0   # no evidence, no oracle fit


def test_credibility_rises_with_independent_evidence():
    state = CvdState([_ins(0, "nop")], EXPECTED)
    few = evaluate_credibility(SCR, [{"a": 1}], state)
    many = evaluate_credibility(SCR, [{"a": 1}, {"b": 2}, {"c": 3}], state)
    assert many > few > 0.0


def test_credibility_rises_with_fit_to_our_oracle():
    # a locus our oracle-reconstruct corroborates (writes == expected) scores higher
    trace = [_ins(0, "str x8, [x9]", mem=(MemOp("w", REAL, _le(EXPECTED), 4),))]
    state = CvdState(trace, EXPECTED)
    fits = evaluate_credibility(REAL, [{"a": 1}], state)
    no_fit = evaluate_credibility(SCR, [{"a": 1}], state)
    assert fits > no_fit                                 # oracle agreement boosts fit


# --- §C: credibility orders verification but never grants trust --------------

def test_low_credibility_assumption_is_verified_then_eliminated_not_trusted():
    # the benchmark root cause as the LEAD: "0x70f80 is the sink" with NO evidence.
    # CVD does not treat it as ground truth — it is verified (oracle mismatch) and
    # ELIMINATED in round 1, never propagated. (No real sink here -> terminal.)
    trace = [_ins(0, "str x8, [x10]", mem=(MemOp("w", SCR, _le(SCRATCH), 4),))]
    res = run_cvd(trace, EXPECTED,
                  submissions=[AgentSubmission(locus=SCR, provenance="agent_hypothesis",
                                               evidence=[])])
    assert any(e.get("event") == "ELIMINATED"
               and e.get("candidate", {}).get("provenance") == "agent_hypothesis"
               for e in res.log)                          # verified, then eliminated
    assert res.sink_base != SCR                           # never trusted as the sink


def test_credibility_orders_but_real_sink_still_wins_over_assumption():
    # a bare assumption at scratch + the real buffer present: the real sink wins;
    # the wrong assumption is never built upon (no ~10-round cascade).
    trace = [
        _ins(0, "str x8, [x10]", mem=(MemOp("w", SCR, _le(SCRATCH), 4),)),   # scratch
        _ins(1, "str x8, [x9]",  mem=(MemOp("w", REAL, _le(EXPECTED), 4),)),  # real
    ]
    res = run_cvd(trace, EXPECTED,
                  submissions=[AgentSubmission(locus=SCR, provenance="agent_hypothesis",
                                               evidence=[])])
    assert res.outcome is CvdOutcome.SUCCESS
    assert res.sink_base == REAL


def test_high_credibility_overridden_when_oracle_disagrees():
    # even a heavily-"evidenced" submission at a scratch locus is overridden the
    # instant oracle-reconstruct disagrees — credibility sets order, not verdict.
    trace = [
        _ins(0, "str x8, [x10]", mem=(MemOp("w", SCR, _le(SCRATCH), 4),)),
        _ins(1, "str x8, [x9]",  mem=(MemOp("w", REAL, _le(EXPECTED), 4),)),
    ]
    res = run_cvd(trace, EXPECTED,
                  submissions=[AgentSubmission(
                      locus=SCR, provenance="agent_finding",
                      evidence=[{"x": i} for i in range(5)])])  # lots of evidence
    assert res.outcome is CvdOutcome.SUCCESS
    assert res.sink_base == REAL                          # oracle wins over credibility
