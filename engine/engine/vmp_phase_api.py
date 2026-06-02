"""VMP phase API — the sign5 light-to-heavy methodology, encoded as a sequence.

This is the VMP-domain *content* that fills the generic skeleton in
:mod:`engine.phase_sequence`.  It freezes the path that worked on the reference case's
sign5 (calltrace → hook I/O → watch materialization → trace data-flow
provenance → induce formula → parity) into a forced-order interface so the
next VMP target does not re-invent the detours the reference case hit (roadmap
§8.12 full-vmtrace-first, §8.13 throw-candidates-to-guess).

The canonical chain (each wires an existing utov primitive):

  phase_1_io_observe        calltrace + I/O hook — returns I/O shape only.
                            Physically cannot see VMP internals → forces the
                            light start.  (engine.phase_instrument, func entry,
                            coarse granularity)
  phase_2_materialization_trace  hook the output write sequence (the strb's)
                            — the source of the sign5 prefix formula.
                            (engine.phase_instrument over the output region)
  phase_3_provenance        trace the data flow: watch first write + 5-way
                            constant-provenance classify → producer chain.
                            (engine.watch_first_write + engine.constant_provenance)
                            >>> There is NO "guess the algorithm" entry here.
                                The only move is to follow the data flow. <<<
  phase_4_formula_induction the agent's judgment: induce ONE formula from the
                            observed materialization + provenance (not a sprayed
                            candidate set).  is_judgment=True.
  phase_5_parity            full-chain bytewise parity (the oracle).

  phase_heavy_vmtrace       the escalation.  GATED: only reachable with an
                            EscalationProof citing phase_1-3 outcomes that
                            recorded COULD_NOT_CLOSE.  Builds a GRAN_FULL
                            instrument (the expensive full vmtrace).

`utov judges`: every phase method enters the phase (order enforced) and hands
back the primitive spec / analysis; the *verdict* (RAN / CLOSED /
COULD_NOT_CLOSE) is recorded by the caller via :meth:`VmpPhaseApi.record`,
because judgement is utov's role, not the interface's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .constant_provenance import (
    ConstantProvenanceResult,
    DataflowSummary,
    RerunObservation,
    classify_value,
)
from .phase import Anchor, ANCHOR_FUNC_ENTRY, ANCHOR_MEMREGION_FIRST_ACCESS
from .phase_instrument import (
    GRAN_FULL,
    GRAN_PC_BAND,
    GRAN_REG_DELTA,
    PhaseInstrumentSpec,
    request_phase_instrument,
)
from .phase_sequence import (
    EscalationConfirmation,
    EscalationProof,
    EscalationPrompt,
    PhaseDef,
    PhaseGateError,
    PhaseOutcome,
    PhaseRun,
    PhaseSequence,
    PhaseStatus,
)
from .watch_first_write import WatchFirstWriteSpec, request_watch_first_write


# Phase names — wire-stable; also used as the only legal entry points.
PHASE_IO_OBSERVE          = "phase_1_io_observe"
PHASE_MATERIALIZATION     = "phase_2_materialization_trace"
PHASE_PROVENANCE          = "phase_3_provenance"
PHASE_FORMULA_INDUCTION   = "phase_4_formula_induction"
PHASE_PARITY              = "phase_5_parity"
PHASE_HEAVY_VMTRACE       = "phase_heavy_vmtrace"


VMP_PHASE_SEQUENCE = PhaseSequence(steps=(
    PhaseDef(PHASE_IO_OBSERVE,        order=1),
    PhaseDef(PHASE_MATERIALIZATION,   order=2, requires=(PHASE_IO_OBSERVE,)),
    PhaseDef(PHASE_PROVENANCE,        order=3, requires=(PHASE_MATERIALIZATION,)),
    PhaseDef(PHASE_FORMULA_INDUCTION, order=4, requires=(PHASE_PROVENANCE,),
             is_judgment=True),
    PhaseDef(PHASE_PARITY,            order=5, requires=(PHASE_FORMULA_INDUCTION,)),
    # The escalation lives in the same order-space but is reached only through
    # the gate. It requires the three light analysis phases to have run.
    PhaseDef(PHASE_HEAVY_VMTRACE,     order=99,
             requires=(PHASE_IO_OBSERVE, PHASE_MATERIALIZATION, PHASE_PROVENANCE),
             is_escalation=True),
))


# ---------------------------------------------------------------------------
# Phase-4 / phase-5 carriers (judgment + parity intents).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FormulaInduction:
    """One formula induced from observed evidence — NOT a candidate set.

    ``derived_from`` should reference the concrete phase outputs the formula
    was read off (materialization sequence, provenance chain). Spraying many
    guesses has no representation here on purpose."""

    expression:   str
    derived_from: tuple[str, ...] = ()
    note:         str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression":   self.expression,
            "derived_from": list(self.derived_from),
            "note":         self.note,
        }


@dataclass(frozen=True, slots=True)
class VmtraceBudget:
    """The resource budget the agent must commit before a full vmtrace.

    A full instruction-level trace is the expensive escalation; the agent
    states up front what it expects to spend so the cost is a conscious,
    recorded decision (and so the confirmation prompt can show it). Orthogonal
    to *why* it escalates (proof / confirmation) — this is *what it costs*.
    """

    runtime_s: float   # estimated wall-clock seconds
    disk_mb:   float    # estimated trace disk footprint, MB
    note:      str = ""

    def __post_init__(self) -> None:
        if not (self.runtime_s > 0):
            raise ValueError("VmtraceBudget.runtime_s must be > 0 — estimate the cost")
        if not (self.disk_mb > 0):
            raise ValueError("VmtraceBudget.disk_mb must be > 0 — estimate the cost")

    def human(self) -> str:
        return f"~{self.runtime_s:g}s runtime / ~{self.disk_mb:g}MB disk"

    def to_dict(self) -> dict[str, Any]:
        return {"runtime_s": self.runtime_s, "disk_mb": self.disk_mb, "note": self.note}


@dataclass(frozen=True, slots=True)
class ParityIntent:
    """A full-chain bytewise parity request the runner/oracle fulfils."""

    formula:    FormulaInduction
    inputs_min: int = 1
    note:       str = "full-chain bytewise parity against the live oracle"

    def to_dict(self) -> dict[str, Any]:
        return {
            "formula":    self.formula.to_dict(),
            "inputs_min": self.inputs_min,
            "note":       self.note,
        }


# ---------------------------------------------------------------------------
# The API — order-enforced, escalation-gated.
# ---------------------------------------------------------------------------


class VmpPhaseApi:
    """Stateful driver over :data:`VMP_PHASE_SEQUENCE`.

    Each ``phase_*`` method enters the phase (order is enforced by the
    underlying :class:`PhaseRun`) and returns the primitive artifact for that
    step. The caller records the verdict with :meth:`record`.
    """

    def __init__(self, run: Optional[PhaseRun] = None) -> None:
        self.run = run or PhaseRun(VMP_PHASE_SEQUENCE)
        self.heavy_budget: Optional[VmtraceBudget] = None  # committed at escalation

    # -- verdict recording (utov judges) ----------------------------------

    def record(
        self,
        phase: str,
        status: PhaseStatus,
        *,
        summary: str = "",
        could_not_close_reason: str = "",
        payload: Optional[dict[str, Any]] = None,
    ) -> PhaseOutcome:
        outcome = PhaseOutcome(
            phase=phase,
            status=status,
            summary=summary,
            could_not_close_reason=could_not_close_reason,
            payload=payload or {},
        )
        self.run.record(outcome)
        return outcome

    # -- phase 1: I/O observe (light, forced first) -----------------------

    def phase_1_io_observe(
        self,
        *,
        entry_pc: int,
        max_steps: int = 20_000,
        label: str = "",
    ) -> PhaseInstrumentSpec:
        """Calltrace + hook the crypto entry; capture I/O shape only.

        Coarse granularity (pc_band) — we want the I/O boundary, not the VMP
        internals. Returns the runner-fulfillable instrument spec."""
        self.run.enter(PHASE_IO_OBSERVE)
        anchor = Anchor(
            anchor_type=ANCHOR_FUNC_ENTRY,
            params={"pc": int(entry_pc)},
            label=label or "crypto entry",
        )
        return request_phase_instrument(
            phase_name=PHASE_IO_OBSERVE,
            anchor=anchor,
            granularity=GRAN_PC_BAND,
            max_steps=max_steps,
            label="phase_1 I/O observe",
        )

    # -- phase 2: materialization trace -----------------------------------

    def phase_2_materialization_trace(
        self,
        *,
        output_base: int,
        output_len: int,
        max_steps: int = 50_000,
    ) -> PhaseInstrumentSpec:
        """Hook the output write sequence (the strb's) — the prefix-formula
        source. Anchored at first write into the output region, capturing every
        write to it."""
        self.run.enter(PHASE_MATERIALIZATION)
        anchor = Anchor(
            anchor_type=ANCHOR_MEMREGION_FIRST_ACCESS,
            params={"base": int(output_base), "length": int(output_len), "access": "w"},
            label="output materialization",
        )
        return request_phase_instrument(
            phase_name=PHASE_MATERIALIZATION,
            anchor=anchor,
            granularity=GRAN_REG_DELTA,
            regions=((int(output_base), int(output_len)),),
            max_steps=max_steps,
            label="phase_2 materialization trace",
        )

    # -- phase 3: provenance (follow data flow — NO guessing) -------------

    def phase_3_watch_producer(
        self,
        *,
        addr: int,
        value_name: str,
        reason: str = "phase_3: trace producer of observed value",
    ) -> WatchFirstWriteSpec:
        """Build the watchpoint that finds who produced an observed value."""
        self.run.enter(PHASE_PROVENANCE)
        return request_watch_first_write(addr, value_name, reason=reason)

    def phase_3_classify(
        self,
        value_name: str,
        *,
        rerun_observations: Iterable[RerunObservation] = (),
        dataflow: DataflowSummary | None = None,
    ) -> ConstantProvenanceResult:
        """Run the 5-way constant-provenance classify on the producer chain.

        May be called after :meth:`phase_3_watch_producer` (which enters the
        phase). If the phase has not been entered yet, enter it now — both are
        legitimate phase-3 moves and neither is a guess."""
        if not self.run.entered(PHASE_PROVENANCE):
            self.run.enter(PHASE_PROVENANCE)
        return classify_value(
            value_name,
            rerun_observations=rerun_observations,
            dataflow=dataflow,
        )

    # -- phase 4: formula induction (judgment) ----------------------------

    def phase_4_formula_induction(self, formula: FormulaInduction) -> ParityIntent:
        """Take ONE induced formula and stage a parity check. The agent's call;
        the interface offers no way to submit a candidate *set*."""
        self.run.enter(PHASE_FORMULA_INDUCTION)
        return ParityIntent(formula=formula)

    # -- phase 5: parity (the oracle) -------------------------------------

    def phase_5_parity(self, intent: ParityIntent) -> ParityIntent:
        """Stage the full-chain bytewise parity. Returns the intent the runner/
        oracle fulfils; the verdict is recorded by the caller."""
        self.run.enter(PHASE_PARITY)
        return intent

    # -- escalation: heavy vmtrace (gated) --------------------------------

    def heavy_vmtrace_prompt(
        self, budget: VmtraceBudget | None = None
    ) -> EscalationPrompt:
        """The context-aware confirmation question to put to the user/driver
        before unlocking vmtrace (e.g. "你还没试 phase_1-3 就要上 vmtrace?").
        When the agent's ``budget`` is supplied it is appended so the human
        confirms *with the cost in view*. The driver renders the question; a
        'yes' becomes an :class:`EscalationConfirmation` passed to
        :meth:`phase_heavy_vmtrace`."""
        prompt = self.run.escalation_prompt(PHASE_HEAVY_VMTRACE)
        if budget is None:
            return prompt
        return EscalationPrompt(
            phase=prompt.phase,
            question=f"{prompt.question}（预算 {budget.human()}）",
            phases_run=prompt.phases_run,
            walled=prompt.walled,
            untried_required=prompt.untried_required,
        )

    def phase_heavy_vmtrace(
        self,
        *,
        anchor: Anchor,
        budget: VmtraceBudget,
        proof: EscalationProof | None = None,
        confirmation: EscalationConfirmation | None = None,
        max_steps: int = 500_000,
    ) -> PhaseInstrumentSpec:
        """Escalate to a full instruction-level vmtrace.

        ``budget`` is REQUIRED — the agent must state the estimated runtime +
        disk cost up front (a conscious, recorded resource commitment). It is
        orthogonal to the gate justification.

        GATED two ways (supply exactly one):
          * ``proof`` — autonomous path: must cite phase_1-3 outcomes recorded
            as COULD_NOT_CLOSE.
          * ``confirmation`` — interactive path: the user/driver answered yes to
            :meth:`heavy_vmtrace_prompt`. A deliberate, audited override.

        Without either, raises :class:`engine.phase_sequence.PhaseGateError`
        (carrying the hint to call the prompt) and no vmtrace spec is produced.
        First unlock sticks — re-entry does not re-prompt."""
        if not isinstance(budget, VmtraceBudget):
            raise PhaseGateError(
                "phase_heavy_vmtrace requires a VmtraceBudget — the agent must "
                "estimate runtime_s + disk_mb before a full trace"
            )
        self.run.enter(
            PHASE_HEAVY_VMTRACE, proof=proof, confirmation=confirmation
        )
        self.heavy_budget = budget
        return request_phase_instrument(
            phase_name=PHASE_HEAVY_VMTRACE,
            anchor=anchor,
            granularity=GRAN_FULL,
            max_steps=max_steps,
            label=f"phase_heavy full vmtrace (escalated; budget {budget.human()})",
        )


__all__ = [
    "PHASE_IO_OBSERVE",
    "PHASE_MATERIALIZATION",
    "PHASE_PROVENANCE",
    "PHASE_FORMULA_INDUCTION",
    "PHASE_PARITY",
    "PHASE_HEAVY_VMTRACE",
    "VMP_PHASE_SEQUENCE",
    "FormulaInduction",
    "ParityIntent",
    "VmtraceBudget",
    "VmpPhaseApi",
    "EscalationConfirmation",
    "EscalationProof",
    "EscalationPrompt",
]
