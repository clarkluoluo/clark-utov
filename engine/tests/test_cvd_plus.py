"""CVD-Plus: the extensibility + registry-dispatch properties (CVD_PLUS_DESIGN
§2-§9). The driver dispatches only over the Registry, never hardcodes a tool;
unknown -> EXTENSION_REQUEST (not a stall); a registered tool of a NEW kind drives
to a result with no driver change; a buggy tool eliminates only its candidate; the
run state serializes and resumes.
"""

from __future__ import annotations

import json

from engine.cvd import (
    Candidate,
    CandidateGenerator,
    CvdBudget,
    CvdOutcome,
    ProvenanceVerifier,
    Registry,
    SinkGenerator,
    SinkValidatorVerifier,
    Verdict,
    Verifier,
    VStatus,
    resume,
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


class _MysteryGen(CandidateGenerator):
    name = "mystery_gen"; version = "1"; kind = "mystery"

    def generate(self, state):
        # high base_value so it is popped first (lets the governance test exercise
        # the tool_error path before any other candidate resolves).
        return [Candidate("mystery", 0x1, "weird", "an unknown candidate kind",
                          base_value=10.0)]


def _trivial_trace():
    return [_ins(0, "nop")]


# --- §4: unknown candidate kind -> EXTENSION_REQUEST(verifier), not a stall --

def test_no_verifier_applies_emits_extension_request():
    reg = Registry().register(_MysteryGen())   # generator but NO verifier for it
    res = run_cvd(_trivial_trace(), EXPECTED, registry=reg)
    assert res.outcome is CvdOutcome.EXTENSION_REQUEST
    assert res.extension_request["missing_kind"] == "verifier"
    assert "mystery" in res.extension_request["why"]


# --- §3/§5: a registered tool of a NEW kind drives to a result, driver unchanged

class _MysteryVerifier(Verifier):
    name = "mystery_v"; version = "1"

    def applies(self, c, state):
        return c.kind == "mystery"

    def verify(self, c, state):
        return Verdict(VStatus.TERMINAL, terminal_kind="CONTINUOUS_BUFFER",
                       success=True, evidence={"chain": []}, located_base=c.locus)


def test_registered_tool_of_new_kind_is_dispatched():
    reg = Registry().register(_MysteryGen()).register(_MysteryVerifier())
    res = run_cvd(_trivial_trace(), EXPECTED, registry=reg)
    assert res.outcome is CvdOutcome.SUCCESS
    assert res.verdict == "CONTINUOUS_BUFFER"
    assert res.sink_base == 0x1


# --- §7 governance: a buggy tool eliminates only its candidate, train continues

class _RaisingVerifier(Verifier):
    name = "raiser"; version = "1"

    def applies(self, c, state):
        return c.kind == "mystery"

    def verify(self, c, state):
        raise RuntimeError("boom")


def test_buggy_tool_eliminated_as_tool_error_and_train_continues():
    trace = [
        _ins(0, "str x8, [x9]", mem=(MemOp("w", 0x2000, _le(EXPECTED), 4),)),  # real sink
    ]
    reg = (Registry()
           .register(SinkGenerator()).register(_MysteryGen())
           .register(SinkValidatorVerifier()).register(ProvenanceVerifier())
           .register(_RaisingVerifier()))
    res = run_cvd(trace, EXPECTED, registry=reg)
    # the raising tool's candidate was eliminated as tool_error; the sink still won
    assert res.outcome is CvdOutcome.SUCCESS
    assert any(e.get("event") == "ELIMINATED" and "tool_error" in e.get("reason", "")
               for e in res.log)


# --- §7 observability: a tool_error carries the exception MESSAGE (key name) +
#     a traceback (line number) so a black-box "tool_error:KeyError" can be routed

_RAISED_KEY = "the_missing_key"


class _KeyErrorVerifier(Verifier):
    name = "key_raiser"; version = "1"

    def applies(self, c, state):
        return c.kind == "mystery"

    def verify(self, c, state):
        d: dict = {}
        return Verdict(VStatus.CONFIRMED, evidence={"x": d[_RAISED_KEY]})  # KeyError


def test_tool_error_surfaces_message_and_traceback():
    reg = (Registry().register(_MysteryGen()).register(_KeyErrorVerifier()))
    res = run_cvd(_trivial_trace(), EXPECTED, registry=reg)
    elim = [e for e in res.log
            if e.get("event") == "ELIMINATED" and "tool_error" in e.get("reason", "")]
    assert elim, "the KeyError-raising tool must still ELIMINATE its candidate (§7)"
    reason = elim[0]["reason"]
    # governance §7 unchanged: still ELIMINATED, still tool_error:<type> …
    assert reason.startswith("tool_error:KeyError")
    # … but now it carries the message (the missing key name) so it can be routed.
    assert _RAISED_KEY in reason


def test_tool_error_evidence_carries_traceback_with_line_number():
    # The ELIMINATED record must carry the traceback (top frames + the raising
    # line) so a black-box tool_error can be routed to the exact file:line.
    reg = (Registry().register(_MysteryGen()).register(_KeyErrorVerifier()))
    res = run_cvd(_trivial_trace(), EXPECTED, registry=reg)
    elim = [e for e in res.log
            if e.get("event") == "ELIMINATED" and "tool_error" in e.get("reason", "")]
    assert elim, "the tool_error must still ELIMINATE its candidate"
    detail = elim[0].get("error_detail")
    assert detail, "the ELIMINATED record must carry the traceback (error_detail)"
    assert "Traceback (most recent call last)" in detail
    assert "KeyError" in detail
    assert _RAISED_KEY in detail           # the missing key is in the traceback
    # a traceback locates a file:line — at least one frame line is present.
    assert "line " in detail and ".py" in detail


def test_success_path_has_no_error_detail():
    # invariant 7: a CONFIRMED (success) verdict must carry NO error_detail field
    # value — the except-branch additions never touch the success path.
    reg = Registry().register(_MysteryGen()).register(_MysteryVerifier())
    res = run_cvd(_trivial_trace(), EXPECTED, registry=reg)
    assert res.outcome is CvdOutcome.SUCCESS
    # no ELIMINATED tool_error in a clean run, and no traceback leaked anywhere.
    assert not any("tool_error" in str(e.get("reason", "")) for e in res.log)
    assert not any("error_detail" in e for e in res.log)


# --- §4: dead end with no TerminalClassifier -> EXTENSION_REQUEST(terminal) --

def test_dead_end_without_terminal_classifier_emits_extension_request():
    trace = [_ins(0, "str x8, [x9]", mem=(MemOp("w", 0x1000, _le(SCRATCH), 4),))]
    reg = (Registry().register(SinkGenerator())
           .register(SinkValidatorVerifier()).register(ProvenanceVerifier()))
    # no TerminalClassifier registered
    res = run_cvd(trace, EXPECTED, registry=reg)
    assert res.outcome is CvdOutcome.EXTENSION_REQUEST
    assert res.extension_request["missing_kind"] == "terminal"


# --- §6/§9.4: run state serializes (json) and resume continues --------------

def test_checkpoint_is_json_serializable_and_resume_continues():
    trace = [
        _ins(0, "str x8, [x9]", mem=(MemOp("w", 0x1000, _le(SCRATCH), 4),)),
        _ins(1, "str x8, [x10]", mem=(MemOp("w", 0x3000, _le(SCRATCH), 4),)),
    ]
    paused = run_cvd(trace, EXPECTED, budget=CvdBudget(max_candidates=1, max_widen=0))
    assert paused.outcome is CvdOutcome.BUDGET_EXHAUSTED
    cp = paused.checkpoint
    json.dumps(cp)                              # must be fully serializable (no objects)
    assert cp["frontier"]                       # remaining work carried in the checkpoint
    # resume with a larger budget -> the train continues from the checkpoint
    res = resume(cp, trace, budget=CvdBudget(max_candidates=10, max_widen=0))
    assert res.outcome in (CvdOutcome.TERMINAL, CvdOutcome.SUCCESS,
                           CvdOutcome.EXTENSION_REQUEST)
