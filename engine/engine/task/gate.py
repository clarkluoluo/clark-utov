"""TaskGate — the second M1 layer (PLAN §20.1.2 task ↔ gate).

The conjunctive gate already shipped in v0.4.0 is the **node**-level
M1: per-call, per-archival-surface mechanism enforcement.  The
**task**-level M1 is a different question: when the agent declares
"task done", does the spec's ``done_criterion`` actually hold?  The
two MUST be walked separately — "all nodes closed" is *not* "task
done", which is exactly the reference-case pothole.

:class:`TaskGate` answers the task-level question:

  * input: a :class:`TaskSpec` + a runtime context (closed nodes,
    done child tasks, present artefacts).
  * output: :class:`TaskGateResult` — passed / failed, the gap list
    from the criterion evaluator, and a human-readable refusal
    message ready for envelope display.

When the agent declares "task done" via clark, clark calls
:meth:`TaskGate.evaluate_task_done`.  If the result reports
``passed=False``, clark MUST refuse the declaration and return the
refusal message; never silently let through.

Mechanism floor penetration (PLAN §20.1.3 invariant #1) — a separate
hook in T7 wires the task gate to also walk the v0.4.0 conjunctive
gate against any params the agent supplies with the "task done"
declaration, so the M1 / scope gates / use_case_fork all fire on the
task-level call surface too.  This module exposes the seam
(``conjunctive_gate`` field on :class:`TaskGate`) but does not require
it — running TaskGate without the v0.4.0 hook still enforces
done_criterion, which is the headline guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from engine.task.done_criterion import (
    CriterionEvalContext,
    CriterionEvalResult,
    evaluate_done_criterion,
)
from engine.task.types import NodeState, TaskSpec


class TaskDoneRefusal(Exception):
    """Raised by :meth:`TaskGate.assert_task_done` when the criterion
    isn't satisfied.  The exception message is the same human-readable
    refusal text :class:`TaskGateResult` carries — useful for callers
    that prefer exception flow over conditional check."""


@dataclass(frozen=True)
class TaskGateResult:
    """Outcome of one task-gate evaluation."""

    passed: bool
    task_id: str
    criterion_result: CriterionEvalResult
    mechanism_floor_passed: bool = True
    mechanism_failing_probes: tuple[str, ...] = ()
    refusal_message: str = ""

    @property
    def gaps(self) -> tuple[str, ...]:
        return self.criterion_result.gaps


@dataclass
class TaskGate:
    """Per-task gate.

    Construct one per task instance (cheap — just keeps a reference
    to the spec).  Optional :attr:`conjunctive_gate` is the v0.4.0
    :class:`engine.profile.ConjunctiveGate`; when supplied the task
    gate also runs every base mechanism probe on the task-level
    declaration call, threading the mechanism floor through the new
    surface (PLAN §20.1.3 invariant #1).
    """

    spec: TaskSpec
    conjunctive_gate: Any = None  # engine.profile.ConjunctiveGate | None

    def evaluate_task_done(
        self,
        *,
        ctx: Optional[CriterionEvalContext] = None,
        probe_ctx: Any = None,
    ) -> TaskGateResult:
        """Evaluate the task done question against ``ctx``.

        ``ctx`` is the runtime view of the world (which nodes have
        closed, which children have signed done, which artefacts the
        ledger carries).  If ``ctx`` is omitted, the gate derives a
        context from the spec's own node states — useful for unit
        tests and for callers that pre-stamp closure on the spec.

        ``probe_ctx`` is the :class:`engine.profile.ProbeContext`
        passed through to the conjunctive gate when present.  Without
        a conjunctive gate the parameter is ignored.
        """
        if ctx is None:
            ctx = self._derive_ctx_from_spec()

        criterion_result = evaluate_done_criterion(self.spec.done_criterion, ctx)

        mech_passed = True
        mech_failing: tuple[str, ...] = ()
        if self.conjunctive_gate is not None and probe_ctx is not None:
            mech_result = self.conjunctive_gate.evaluate(probe_ctx)
            mech_passed = mech_result.passed
            mech_failing = mech_result.failing_probes

        passed = criterion_result.satisfied and mech_passed
        refusal_message = (
            "" if passed
            else _format_refusal(self.spec.id, criterion_result, mech_failing)
        )
        return TaskGateResult(
            passed=passed,
            task_id=self.spec.id,
            criterion_result=criterion_result,
            mechanism_floor_passed=mech_passed,
            mechanism_failing_probes=mech_failing,
            refusal_message=refusal_message,
        )

    def assert_task_done(
        self,
        *,
        ctx: Optional[CriterionEvalContext] = None,
        probe_ctx: Any = None,
    ) -> TaskGateResult:
        """Same as :meth:`evaluate_task_done` but raises
        :class:`TaskDoneRefusal` when the gate fails."""
        result = self.evaluate_task_done(ctx=ctx, probe_ctx=probe_ctx)
        if not result.passed:
            raise TaskDoneRefusal(result.refusal_message)
        return result

    def _derive_ctx_from_spec(self) -> CriterionEvalContext:
        closed = frozenset(
            n.id for n in self.spec.nodes if n.state is NodeState.CLOSED
        )
        # Child / artefact derivation requires external state — without
        # an explicit ctx the gate assumes "no children done yet, no
        # artefacts present yet", which is the conservative answer.
        return CriterionEvalContext(closed_nodes=closed)


def _format_refusal(
    task_id: str,
    criterion_result: CriterionEvalResult,
    mech_failing: tuple[str, ...],
) -> str:
    parts: list[str] = [
        f"[TASK-GATE/REFUSE] task '{task_id}' done declaration refused:"
    ]
    if not criterion_result.satisfied:
        parts.append("  done_criterion unsatisfied — gaps:")
        for g in criterion_result.gaps:
            parts.append(f"    - {g}")
    if mech_failing:
        parts.append(
            f"  mechanism floor failed: {sorted(mech_failing)}"
        )
    parts.append(
        "  resolve every gap and re-declare; the task gate will re-evaluate."
    )
    return "\n".join(parts)
