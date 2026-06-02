"""Phase sequence — generic forced-order methodology skeleton.

Encodes the "light-to-heavy + escalation-needs-proof" discipline as an
*interface*, so the path is enforced by call order rather than by agent
self-discipline (roadmap §9.4 "由轻到重 + 升级要证明"; §8.12/§8.13 record the
field failures this prevents — full vmtrace as a first move, and re-inventing
the "throw-candidates" detour).

Three properties the shape guarantees:

  1. **Forced order.** A phase cannot be entered until every phase it
     ``requires`` has already run. The state machine refuses out-of-order
     entry and names the missing predecessor.

  2. **Gated escalation.** A phase flagged ``is_escalation`` (an expensive /
     heavy method) can only be entered with an :class:`EscalationProof` that
     CITES prior phases which reported they *could not close*
     (:attr:`PhaseStatus.COULD_NOT_CLOSE`). No proof, an empty reason, or a
     proof citing a phase that actually closed → refused with a message naming
     the offending citation. This is "升级要闭不住证明" made physical.

  3. **No guess verb.** What the shape OMITS matters as much as what it has:
     there is no "try-candidates / guess" entry point. The only edge out of an
     analysis phase is the next ordered phase. An agent that wants to guess an
     algorithm finds no interface for it (roadmap §8.13).

This module is deliberately CONTENT-AGNOSTIC — it knows nothing about VMP,
traces, or crypto. It is the candidate to lift into clark-Hypotask as the
generic "phase-sequence + upgrade-needs-reason" framework; the VMP-specific
phase content lives in :mod:`engine.vmp_phase_api`. (roadmap note: "phase API
是 utov 的能力；'轻→重 + 升级要证明' 这个模式是通用的，Hypotask 该提供通用框架".)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional


class PhaseStatus(enum.Enum):
    """The verdict a phase records when it finishes.

    ``RAN`` means the phase executed and produced output but neither closed
    the node nor hit a hard "this method cannot close it" wall — the normal
    hand-off to the next ordered phase.  ``CLOSED`` means the phase produced
    closure-grade evidence (the chain can stop).  ``COULD_NOT_CLOSE`` is the
    explicit "this light method is exhausted" verdict — the ONLY thing an
    :class:`EscalationProof` may cite to unlock a heavier phase.
    """

    RAN             = "ran"
    CLOSED          = "closed"
    COULD_NOT_CLOSE = "could_not_close"


# ---------------------------------------------------------------------------
# Static shape — phase definitions + the ordered sequence.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PhaseDef:
    """One phase in a sequence.

    ``order``       strictly-increasing position in the linear chain;
                    escalation phases share the order space but are reached
                    only through :meth:`PhaseRun.request_escalation`.
    ``requires``    names of phases that must have run before this one.
    ``is_judgment`` this phase is the agent's call (e.g. formula induction),
                    not a mechanical step — informational, for renderers.
    ``is_escalation`` this phase is gated behind an :class:`EscalationProof`.
    """

    name:          str
    order:         int
    requires:      tuple[str, ...] = ()
    is_judgment:   bool = False
    is_escalation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name":          self.name,
            "order":         self.order,
            "requires":      list(self.requires),
            "is_judgment":   self.is_judgment,
            "is_escalation": self.is_escalation,
        }


@dataclass(frozen=True, slots=True)
class PhaseSequence:
    """An ordered, validated set of phases.

    Validation (at construction): names unique, orders unique, every
    ``requires`` references a declared phase, and a non-escalation phase may
    not require a later-ordered phase (no forward dependency in the linear
    chain).
    """

    steps: tuple[PhaseDef, ...]

    def __post_init__(self) -> None:
        names = [s.name for s in self.steps]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate phase name in sequence: {names}")
        orders = [s.order for s in self.steps]
        if len(orders) != len(set(orders)):
            raise ValueError(f"duplicate phase order in sequence: {orders}")
        by_name = {s.name: s for s in self.steps}
        for s in self.steps:
            for req in s.requires:
                if req not in by_name:
                    raise ValueError(
                        f"phase {s.name!r} requires unknown phase {req!r}"
                    )
                if not s.is_escalation and by_name[req].order >= s.order:
                    raise ValueError(
                        f"phase {s.name!r} (order {s.order}) requires "
                        f"{req!r} (order {by_name[req].order}) which is not "
                        f"earlier — forward dependency forbidden"
                    )

    def get(self, name: str) -> PhaseDef:
        for s in self.steps:
            if s.name == name:
                return s
        raise KeyError(f"no such phase {name!r}; known={[s.name for s in self.steps]}")

    @property
    def ordered(self) -> tuple[PhaseDef, ...]:
        """Linear phases (non-escalation) in ``order``."""
        return tuple(sorted(
            (s for s in self.steps if not s.is_escalation), key=lambda s: s.order
        ))

    @property
    def escalations(self) -> tuple[PhaseDef, ...]:
        return tuple(s for s in self.steps if s.is_escalation)


# ---------------------------------------------------------------------------
# Runtime — outcome record + escalation proof.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PhaseOutcome:
    """The record a phase leaves after it finishes."""

    phase:               str
    status:              PhaseStatus
    summary:             str = ""
    could_not_close_reason: str = ""
    payload:             dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status is PhaseStatus.COULD_NOT_CLOSE and not self.could_not_close_reason:
            raise ValueError(
                f"phase {self.phase!r} recorded COULD_NOT_CLOSE without a "
                f"reason — the reason is what an escalation proof must cite"
            )

    @property
    def closed(self) -> bool:
        return self.status is PhaseStatus.CLOSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase":   self.phase,
            "status":  self.status.value,
            "summary": self.summary,
            "could_not_close_reason": self.could_not_close_reason,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True, slots=True)
class EscalationProof:
    """Proof that the light phases are exhausted, unlocking a heavy phase.

    ``cites`` names the prior phases whose recorded outcome must be
    :attr:`PhaseStatus.COULD_NOT_CLOSE`.  ``reason`` is the free-text
    justification (it should reference the concrete failure, e.g. "phase_3
    provenance reached a true_boundary: producer PC outside observable range").
    """

    cites:  tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"cites": list(self.cites), "reason": self.reason}


@dataclass(frozen=True, slots=True)
class EscalationConfirmation:
    """A human/driver "yes" to an escalation — the interactive alternative to a
    machine :class:`EscalationProof`.

    The escalation gate can be satisfied two ways: autonomously by a machine
    proof (cites phases that COULD_NOT_CLOSE), or interactively by a human who
    answers the confirmation prompt. This is the recorded form of that yes —
    kept verbatim for audit, so "the user green-lit vmtrace at this point,
    having (not) tried the light phases" is on the record.
    """

    who:  str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"who": self.who, "note": self.note}


@dataclass(frozen=True, slots=True)
class EscalationPrompt:
    """The context a driver renders as the confirmation question.

    Built by :meth:`PhaseRun.escalation_prompt`. The engine does not pop a UI;
    it hands back this payload and the driver turns it into a yes/no — a human
    prompt when there's a human in the loop, or an autonomous policy otherwise.
    ``question`` is pre-rendered and context-aware (it warns when the light
    phases were skipped, which is the case most worth stopping).
    """

    phase:            str
    question:         str
    phases_run:       tuple[str, ...]
    walled:           tuple[str, ...]   # required phases recorded COULD_NOT_CLOSE
    untried_required: tuple[str, ...]   # required phases not yet run

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase":            self.phase,
            "question":         self.question,
            "phases_run":       list(self.phases_run),
            "walled":           list(self.walled),
            "untried_required": list(self.untried_required),
        }


class PhaseGateError(RuntimeError):
    """Raised when a phase is entered out of order or an escalation is
    attempted without a valid :class:`EscalationProof`."""


# ---------------------------------------------------------------------------
# State machine — the ledger that enforces order + gates escalation.
# ---------------------------------------------------------------------------


class PhaseRun:
    """Tracks which phases have run and enforces the discipline.

    Usage::

        run = PhaseRun(VMP_SEQUENCE)
        run.enter("phase_1_io_observe")          # ok — no predecessors
        run.record(PhaseOutcome("phase_1_io_observe", PhaseStatus.RAN))
        run.enter("phase_3_provenance")          # PhaseGateError: phase_2 not run
    """

    def __init__(self, sequence: PhaseSequence) -> None:
        self._seq = sequence
        self._outcomes: dict[str, PhaseOutcome] = {}
        self._entered: set[str] = set()
        # Escalation phases unlocked this run (first-time-only — once a proof or
        # a human confirmation clears the gate, re-entry does not re-prompt).
        self._escalation_unlocked: set[str] = set()
        self._confirmations: dict[str, EscalationConfirmation] = {}

    # -- queries ----------------------------------------------------------

    @property
    def sequence(self) -> PhaseSequence:
        return self._seq

    def ran(self, name: str) -> bool:
        return name in self._outcomes

    def entered(self, name: str) -> bool:
        """Whether the phase has been entered (an outcome may not be recorded
        yet)."""
        return name in self._entered

    def outcome(self, name: str) -> Optional[PhaseOutcome]:
        return self._outcomes.get(name)

    def is_closed(self) -> bool:
        """True once any phase has recorded closure-grade evidence."""
        return any(o.closed for o in self._outcomes.values())

    def trail(self) -> list[PhaseOutcome]:
        """Outcomes in the order their phases are defined."""
        order = {s.name: s.order for s in self._seq.steps}
        return sorted(self._outcomes.values(), key=lambda o: order.get(o.phase, 1 << 30))

    def can_enter(self, name: str) -> tuple[bool, str]:
        """Whether ``name`` may be entered now, and why not if not.

        For escalation phases this only checks predecessors-ran; the proof
        requirement is enforced by :meth:`enter` / :meth:`request_escalation`.
        """
        step = self._seq.get(name)
        missing = [r for r in step.requires if not self.ran(r)]
        if missing:
            return False, (
                f"phase {name!r} requires {missing} to have run first"
            )
        return True, ""

    def confirmation(self, name: str) -> Optional[EscalationConfirmation]:
        """The recorded human/driver confirmation for an escalation, if any."""
        return self._confirmations.get(name)

    def escalation_prompt(self, name: str) -> EscalationPrompt:
        """Build the context-aware confirmation question for an escalation.

        The driver renders this as a yes/no to the user (or feeds it to an
        autonomous policy). The question warns when the light phases were
        skipped — the case most worth stopping ("尚未尝试 … 就要上 vmtrace").
        """
        step = self._seq.get(name)
        if not step.is_escalation:
            raise PhaseGateError(f"phase {name!r} is not an escalation phase")
        run = [r for r in step.requires if self.ran(r)]
        walled = []
        for r in step.requires:
            o = self._outcomes.get(r)
            if o is not None and o.status is PhaseStatus.COULD_NOT_CLOSE:
                walled.append(r)
        untried = [r for r in step.requires if not self.ran(r)]
        if untried:
            question = (
                f"尚未尝试 {untried} 就要升级到 {name!r}（如全量 vmtrace）——"
                f"轻量手段还没走完，确定直接上吗？"
            )
        elif walled:
            question = (
                f"已尝试 {walled} 均未闭合，确认升级到 {name!r}（如全量 vmtrace）吗？"
            )
        else:
            question = (
                f"前序阶段已跑（无一记为 could_not_close），仍要升级到 {name!r} 吗？"
            )
        return EscalationPrompt(
            phase=name, question=question,
            phases_run=tuple(run), walled=tuple(walled),
            untried_required=tuple(untried),
        )

    # -- transitions ------------------------------------------------------

    def enter(
        self,
        name: str,
        *,
        proof: Optional[EscalationProof] = None,
        confirmation: Optional[EscalationConfirmation] = None,
    ) -> None:
        """Mark a phase as entered. Raises :class:`PhaseGateError` on a
        violation.

        An escalation phase is unlocked by EITHER a machine ``proof`` (cites
        phases that COULD_NOT_CLOSE — the autonomous path) OR a human
        ``confirmation`` (the interactive yes to :meth:`escalation_prompt` — a
        deliberate override that does NOT require the light phases to have
        walled, since the prompt already surfaced that). First unlock sticks:
        re-entry needs neither again (first-time-only). Supplying both, or a
        proof/confirmation on a non-escalation phase, is rejected.
        """
        step = self._seq.get(name)
        if step.is_escalation:
            if proof is not None and confirmation is not None:
                raise PhaseGateError(
                    f"escalation to {name!r}: pass a proof OR a confirmation, "
                    f"not both"
                )
            if name in self._escalation_unlocked:
                pass  # already cleared this run — do not re-prompt
            elif confirmation is not None:
                # human/driver override — recorded for audit; the prompt already
                # surfaced whether the light phases were tried.
                self._confirmations[name] = confirmation
                self._escalation_unlocked.add(name)
            elif proof is not None:
                self._validate_escalation(step, proof)
                self._escalation_unlocked.add(name)
            else:
                raise PhaseGateError(
                    f"phase {name!r} is an escalation — it needs a machine "
                    f"EscalationProof or a human EscalationConfirmation. Call "
                    f"escalation_prompt({name!r}) to get the question to put to "
                    f"the user/driver."
                )
        else:
            if proof is not None or confirmation is not None:
                raise PhaseGateError(
                    f"phase {name!r} is not an escalation; a proof/confirmation "
                    f"is meaningless here"
                )
            ok, why = self.can_enter(name)
            if not ok:
                raise PhaseGateError(why)
        self._entered.add(name)

    def request_escalation(self, name: str, proof: EscalationProof) -> None:
        """Convenience alias for :meth:`enter` with a machine proof."""
        step = self._seq.get(name)
        if not step.is_escalation:
            raise PhaseGateError(f"phase {name!r} is not an escalation phase")
        self.enter(name, proof=proof)

    def confirm_escalation(
        self, name: str, confirmation: EscalationConfirmation
    ) -> None:
        """Convenience alias for :meth:`enter` with a human confirmation — the
        'user answered yes to the prompt' path."""
        step = self._seq.get(name)
        if not step.is_escalation:
            raise PhaseGateError(f"phase {name!r} is not an escalation phase")
        self.enter(name, confirmation=confirmation)

    def record(self, outcome: PhaseOutcome) -> None:
        """Record a phase's outcome. The phase must have been entered."""
        self._seq.get(outcome.phase)  # validates the name
        if outcome.phase not in self._entered:
            raise PhaseGateError(
                f"phase {outcome.phase!r} recorded an outcome without being "
                f"entered — call enter() first"
            )
        self._outcomes[outcome.phase] = outcome

    # -- internal ---------------------------------------------------------

    def _validate_escalation(self, step: PhaseDef, proof: EscalationProof) -> None:
        # predecessors must have run
        ok, why = self.can_enter(step.name)
        if not ok:
            raise PhaseGateError(why)
        if not proof.reason.strip():
            raise PhaseGateError(
                f"escalation to {step.name!r} needs a non-empty reason"
            )
        if not proof.cites:
            raise PhaseGateError(
                f"escalation to {step.name!r} must cite at least one prior "
                f"phase that COULD_NOT_CLOSE"
            )
        for cited in proof.cites:
            o = self._outcomes.get(cited)
            if o is None:
                raise PhaseGateError(
                    f"escalation to {step.name!r} cites {cited!r} which has "
                    f"not run"
                )
            if o.status is not PhaseStatus.COULD_NOT_CLOSE:
                raise PhaseGateError(
                    f"escalation to {step.name!r} cites {cited!r} but its "
                    f"status is {o.status.value!r}, not could_not_close — a "
                    f"phase that did not hit a wall cannot justify escalation"
                )


__all__ = [
    "PhaseStatus",
    "PhaseDef",
    "PhaseSequence",
    "PhaseOutcome",
    "EscalationProof",
    "EscalationConfirmation",
    "EscalationPrompt",
    "PhaseGateError",
    "PhaseRun",
]
