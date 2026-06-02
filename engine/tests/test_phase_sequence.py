"""Pin the generic phase-sequence skeleton: forced order + gated escalation.

The skeleton (engine.phase_sequence) is content-agnostic; these tests exercise
the three guarantees with a tiny synthetic sequence, independent of VMP.
"""

from __future__ import annotations

import pytest

from engine.phase_sequence import (
    EscalationConfirmation,
    EscalationProof,
    PhaseDef,
    PhaseGateError,
    PhaseOutcome,
    PhaseRun,
    PhaseSequence,
    PhaseStatus,
)


def _seq() -> PhaseSequence:
    return PhaseSequence(steps=(
        PhaseDef("a", order=1),
        PhaseDef("b", order=2, requires=("a",)),
        PhaseDef("c", order=3, requires=("b",)),
        PhaseDef("heavy", order=9, requires=("a", "b", "c"), is_escalation=True),
    ))


# --- sequence validation ----------------------------------------------------

def test_sequence_rejects_duplicate_name():
    with pytest.raises(ValueError):
        PhaseSequence(steps=(PhaseDef("a", 1), PhaseDef("a", 2)))


def test_sequence_rejects_duplicate_order():
    with pytest.raises(ValueError):
        PhaseSequence(steps=(PhaseDef("a", 1), PhaseDef("b", 1)))


def test_sequence_rejects_unknown_require():
    with pytest.raises(ValueError):
        PhaseSequence(steps=(PhaseDef("a", 1, requires=("ghost",)),))


def test_sequence_rejects_forward_dependency():
    # a non-escalation phase may not require a later-ordered phase
    with pytest.raises(ValueError):
        PhaseSequence(steps=(
            PhaseDef("a", 1, requires=("b",)),
            PhaseDef("b", 2),
        ))


# --- forced order -----------------------------------------------------------

def test_in_order_entry_is_allowed():
    run = PhaseRun(_seq())
    run.enter("a")
    run.record(PhaseOutcome("a", PhaseStatus.RAN))
    run.enter("b")
    run.record(PhaseOutcome("b", PhaseStatus.RAN))
    run.enter("c")
    assert run.entered("c")


def test_out_of_order_entry_is_refused_and_names_missing():
    run = PhaseRun(_seq())
    with pytest.raises(PhaseGateError) as ei:
        run.enter("c")  # b (and a) never ran
    assert "requires" in str(ei.value)
    assert "b" in str(ei.value)


def test_record_requires_entry_first():
    run = PhaseRun(_seq())
    with pytest.raises(PhaseGateError):
        run.record(PhaseOutcome("a", PhaseStatus.RAN))


def test_could_not_close_requires_a_reason():
    with pytest.raises(ValueError):
        PhaseOutcome("a", PhaseStatus.COULD_NOT_CLOSE)  # no reason
    # with a reason it is fine
    PhaseOutcome("a", PhaseStatus.COULD_NOT_CLOSE, could_not_close_reason="wall")


# --- gated escalation -------------------------------------------------------

def _run_through_c(*, c_status: PhaseStatus, reason: str = "") -> PhaseRun:
    run = PhaseRun(_seq())
    for name in ("a", "b"):
        run.enter(name)
        run.record(PhaseOutcome(name, PhaseStatus.RAN))
    run.enter("c")
    run.record(PhaseOutcome("c", c_status, could_not_close_reason=reason))
    return run


def test_escalation_without_proof_is_refused():
    run = _run_through_c(c_status=PhaseStatus.COULD_NOT_CLOSE, reason="wall")
    with pytest.raises(PhaseGateError) as ei:
        run.enter("heavy")  # no proof
    assert "EscalationProof" in str(ei.value)


def test_escalation_with_empty_reason_is_refused():
    run = _run_through_c(c_status=PhaseStatus.COULD_NOT_CLOSE, reason="wall")
    with pytest.raises(PhaseGateError):
        run.request_escalation("heavy", EscalationProof(cites=("c",), reason="  "))


def test_escalation_citing_unrun_phase_is_refused():
    run = _run_through_c(c_status=PhaseStatus.COULD_NOT_CLOSE, reason="wall")
    with pytest.raises(PhaseGateError) as ei:
        run.request_escalation("heavy", EscalationProof(cites=("ghost",), reason="x"))
    assert "ghost" in str(ei.value)


def test_escalation_citing_a_phase_that_did_not_wall_is_refused():
    # c merely RAN — it did not hit a wall, so it cannot justify escalation
    run = _run_through_c(c_status=PhaseStatus.RAN)
    with pytest.raises(PhaseGateError) as ei:
        run.request_escalation("heavy", EscalationProof(cites=("c",), reason="x"))
    assert "could_not_close" in str(ei.value)


def test_escalation_with_valid_proof_is_allowed():
    run = _run_through_c(
        c_status=PhaseStatus.COULD_NOT_CLOSE,
        reason="producer PC outside observable range",
    )
    run.request_escalation(
        "heavy",
        EscalationProof(cites=("c",), reason="phase c hit a true boundary"),
    )
    assert run.entered("heavy")


def test_non_escalation_phase_rejects_a_stray_proof():
    run = PhaseRun(_seq())
    with pytest.raises(PhaseGateError):
        run.enter("a", proof=EscalationProof(cites=(), reason="x"))


# --- escalation via human confirmation (the prompt path) --------------------

def test_escalation_via_confirmation_unlocks_without_a_machine_proof():
    run = _run_through_c(c_status=PhaseStatus.COULD_NOT_CLOSE, reason="wall")
    run.confirm_escalation("heavy", EscalationConfirmation(who="user", note="go"))
    assert run.entered("heavy")
    assert run.confirmation("heavy").who == "user"


def test_confirmation_overrides_even_when_light_phases_not_walled():
    # human deliberately escalates without a wall — allowed, but recorded
    run = PhaseRun(_seq())
    run.enter("a"); run.record(PhaseOutcome("a", PhaseStatus.RAN))
    run.enter("b"); run.record(PhaseOutcome("b", PhaseStatus.RAN))
    run.enter("c"); run.record(PhaseOutcome("c", PhaseStatus.RAN))  # not walled
    run.confirm_escalation("heavy", EscalationConfirmation(who="user"))
    assert run.entered("heavy")


def test_prompt_is_context_aware():
    # nothing tried yet → prompt warns about untried predecessors
    run = PhaseRun(_seq())
    p = run.escalation_prompt("heavy")
    assert set(p.untried_required) == {"a", "b", "c"}
    assert "尚未尝试" in p.question
    # after a/b/c walled → prompt cites the walls instead
    run2 = _run_through_c(c_status=PhaseStatus.COULD_NOT_CLOSE, reason="wall")
    p2 = run2.escalation_prompt("heavy")
    assert p2.walled == ("c",)
    assert not p2.untried_required


def test_neither_proof_nor_confirmation_raises_with_prompt_hint():
    run = _run_through_c(c_status=PhaseStatus.COULD_NOT_CLOSE, reason="wall")
    with pytest.raises(PhaseGateError) as ei:
        run.enter("heavy")
    assert "escalation_prompt" in str(ei.value)


def test_proof_and_confirmation_together_rejected():
    run = _run_through_c(c_status=PhaseStatus.COULD_NOT_CLOSE, reason="wall")
    with pytest.raises(PhaseGateError):
        run.enter("heavy",
                  proof=EscalationProof(cites=("c",), reason="wall"),
                  confirmation=EscalationConfirmation(who="user"))


def test_escalation_unlock_is_first_time_only():
    run = _run_through_c(c_status=PhaseStatus.COULD_NOT_CLOSE, reason="wall")
    run.confirm_escalation("heavy", EscalationConfirmation(who="user"))
    # already unlocked — re-entering needs neither again
    run.enter("heavy")
    assert run.entered("heavy")


# --- queries ----------------------------------------------------------------

def test_is_closed_and_trail():
    run = PhaseRun(_seq())
    run.enter("a")
    run.record(PhaseOutcome("a", PhaseStatus.RAN, summary="io"))
    run.enter("b")
    run.record(PhaseOutcome("b", PhaseStatus.CLOSED, summary="done"))
    assert run.is_closed()
    assert [o.phase for o in run.trail()] == ["a", "b"]
