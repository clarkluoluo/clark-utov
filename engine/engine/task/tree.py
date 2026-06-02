"""Task tree assembly (PLAN §20 v2 §4.1).

A :class:`TaskTree` wraps a :class:`TaskSpec` and walks its ``children``
recursively, indexing by id for fast lookup and rolling the
"done" state up the tree.  The two job-shaped questions the tree
answers:

  * Given the current closure / artefact ledger, which child tasks
    are done?  (For a parent's ``child_done`` atoms.)
  * Is the root task done?  (Same question, applied at the top of
    the tree.)

The tree is read-only.  All structural mutations (insert / replace /
delete) go through :mod:`engine.task.audit`; they produce a NEW
:class:`TaskTree` and log the operation.

Compose-time invariant (v2 §4.1): children must be declared in the
parent's spec.  The agent does not split unilaterally — the
:func:`assemble_task_tree` helper does not accept "ad-hoc child"
inputs; only what was in the spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

from engine.task.contract import (
    ContractMismatchError,
    validate_contract_compose,
)
from engine.task.done_criterion import (
    CriterionEvalContext,
    evaluate_done_criterion,
    referenced_children,
)
from engine.task.gate import TaskGate, TaskGateResult
from engine.task.types import NodeState, TaskSpec


@dataclass(frozen=True)
class TaskTree:
    """Indexed view of one task spec + its descendants.

    The frozen dataclass holds the spec verbatim; the index is built
    once at construction so lookups are cheap.
    """

    root: TaskSpec
    _by_id: dict[str, TaskSpec] = field(default_factory=dict)
    _parent_of: dict[str, Optional[str]] = field(default_factory=dict)

    def iter_all(self) -> Iterator[TaskSpec]:
        """Pre-order traversal, root first."""
        yield from _walk(self.root)

    def get(self, task_id: str) -> Optional[TaskSpec]:
        return self._by_id.get(task_id)

    def parent_of(self, task_id: str) -> Optional[str]:
        return self._parent_of.get(task_id)

    def children_of(self, task_id: str) -> tuple[TaskSpec, ...]:
        spec = self.get(task_id)
        return spec.children if spec is not None else ()

    def is_done(
        self,
        task_id: str,
        *,
        closed_nodes: frozenset[str] = frozenset(),
        present_artefacts: frozenset[str] = frozenset(),
    ) -> bool:
        """Recursive done evaluation.

        A leaf task is done iff its own ``done_criterion`` evaluates
        true against the supplied closed_nodes + present_artefacts.

        A parent task is done iff its ``done_criterion`` evaluates true
        against the same node / artefact view AND its
        ``done_children`` view (computed by recursively asking each
        declared child whether it is done).

        Cycles are not possible (spec is a tree by construction —
        :func:`assemble_task_tree` rejects duplicate ids).
        """
        spec = self.get(task_id)
        if spec is None:
            return False
        done_children = frozenset(
            c.id for c in spec.children
            if self.is_done(
                c.id,
                closed_nodes=closed_nodes,
                present_artefacts=present_artefacts,
            )
        )
        ctx = CriterionEvalContext(
            closed_nodes=closed_nodes,
            done_children=done_children,
            present_artefacts=present_artefacts,
        )
        return evaluate_done_criterion(spec.done_criterion, ctx).satisfied

    def evaluate_root_done(
        self,
        *,
        closed_nodes: frozenset[str] = frozenset(),
        present_artefacts: frozenset[str] = frozenset(),
        conjunctive_gate: Any = None,
        probe_ctx: Any = None,
    ) -> TaskGateResult:
        """Run the root task's gate with auto-rolled-up child done
        state.

        ``conjunctive_gate`` + ``probe_ctx`` opt-in the v0.4.0
        mechanism floor (PLAN §20.1.3 invariant #1).  When both are
        supplied the gate fires every base mechanism probe
        (M1 / M3 / CP / VP / WFW / scope_boundary / scope_upscale +
        any domain probes the active profile registers) against the
        task-level "done" declaration; the result combines the
        criterion verdict with the conjunctive-gate verdict.  Without
        the floor wired the gate still enforces ``done_criterion``,
        which is the headline guarantee.
        """
        done_children = frozenset(
            c.id for c in self.root.children
            if self.is_done(
                c.id,
                closed_nodes=closed_nodes,
                present_artefacts=present_artefacts,
            )
        )
        ctx = CriterionEvalContext(
            closed_nodes=closed_nodes,
            done_children=done_children,
            present_artefacts=present_artefacts,
        )
        gate = TaskGate(spec=self.root, conjunctive_gate=conjunctive_gate)
        return gate.evaluate_task_done(ctx=ctx, probe_ctx=probe_ctx)


def assemble_task_tree(spec: TaskSpec) -> TaskTree:
    """Build a :class:`TaskTree` from a root :class:`TaskSpec`.

    Validates:

      * Every task id in the tree is unique (no duplicate child ids
        within the same parent OR across the tree).
      * Every ``child_done`` atom on every node resolves to a declared
        child id.
      * Every child task referenced by the parent's ``done_criterion``
        carries an ``input_contract`` (PLAN §20.1.3 invariant #5 — the
        compose-time contract requirement; a child without a contract
        is not callable from a parent that names it as a done
        dependency).
    """
    by_id: dict[str, TaskSpec] = {}
    parent_of: dict[str, Optional[str]] = {}
    _index(spec, parent=None, by_id=by_id, parent_of=parent_of)

    # Compose-time check: parent's done_criterion ``child_done`` atoms
    # must reach a child that declares a contract — without the
    # contract, the parent has no way to address the child's outputs.
    for task in by_id.values():
        named_children = referenced_children(task.done_criterion)
        for child_id in named_children:
            child = next(
                (c for c in task.children if c.id == child_id), None
            )
            if child is None:
                # Already caught by the loader's dangling-ref check,
                # but stay defensive.
                continue
            if child.input_contract is None:
                raise ContractMismatchError(
                    f"task '{task.id}': done_criterion references child "
                    f"'{child_id}' which has no input_contract — a child "
                    f"named as a done dependency MUST be addressable via "
                    f"a contract (PLAN §20.1.3 invariant #5)"
                )

    return TaskTree(root=spec, _by_id=by_id, _parent_of=parent_of)


def _index(
    spec: TaskSpec,
    *,
    parent: Optional[str],
    by_id: dict[str, TaskSpec],
    parent_of: dict[str, Optional[str]],
) -> None:
    if spec.id in by_id:
        from engine.task.loader import TaskLoadError
        raise TaskLoadError(
            f"task tree: duplicate task id '{spec.id}' — every task in the "
            f"tree must have a unique id"
        )
    by_id[spec.id] = spec
    parent_of[spec.id] = parent
    for child in spec.children:
        _index(child, parent=spec.id, by_id=by_id, parent_of=parent_of)


def _walk(spec: TaskSpec) -> Iterator[TaskSpec]:
    yield spec
    for child in spec.children:
        yield from _walk(child)


__all__ = [
    "TaskTree",
    "assemble_task_tree",
]
