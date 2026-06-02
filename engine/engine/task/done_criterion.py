"""done_criterion AST + evaluator (PLAN §20).

A task's termination check is composed from five atoms:

  * ``node_closed(node_id)``      — satisfied iff the node id appears
                                    in :attr:`CriterionEvalContext.closed_nodes`.
  * ``child_done(child_id)``      — satisfied iff the child task id
                                    appears in
                                    :attr:`CriterionEvalContext.done_children`.
  * ``named_artefact(name)``      — satisfied iff the artefact name
                                    appears in
                                    :attr:`CriterionEvalContext.present_artefacts`.
  * ``all_of(items)``             — satisfied iff every item satisfied.
  * ``any_of(items)``             — satisfied iff at least one item satisfied.

The evaluator returns both the boolean verdict AND a list of
human-readable gaps so the task gate's refusal message can name
exactly what's missing. The agent does not have to grep the spec to
find the open node — the gate hands the answer.

The grammar is intentionally small: every real-world task termination
the v0.4.0 audit surfaced fits into this set. Adding atoms later is
non-breaking (the loader rejects unknown kinds).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


VALID_KINDS: frozenset[str] = frozenset({
    "node_closed",
    "child_done",
    "named_artefact",
    "all_of",
    "any_of",
})


@dataclass(frozen=True)
class CriterionItem:
    """One node of the done_criterion AST.

    The shape is a tagged union — only the fields appropriate to
    ``kind`` carry meaningful values. The loader validates that the
    right fields appear for each kind and rejects ill-formed entries
    at parse time.
    """

    kind: str
    # node_closed
    node: str = ""
    # child_done
    child: str = ""
    # named_artefact
    name: str = ""
    # all_of / any_of
    items: tuple["CriterionItem", ...] = ()


@dataclass(frozen=True)
class CriterionEvalContext:
    """The world the evaluator consults.

    Three frozensets so the evaluator stays pure — callers compose
    the context from whatever runtime state they have (task tree,
    ledger, node closure status). The evaluator never reaches outside
    these sets.
    """

    closed_nodes: frozenset[str] = frozenset()
    done_children: frozenset[str] = frozenset()
    present_artefacts: frozenset[str] = frozenset()


@dataclass(frozen=True)
class CriterionEvalResult:
    """Outcome of one evaluation pass.

    ``satisfied`` is the headline verdict. ``gaps`` carries the
    human-readable description of every leaf that did NOT satisfy —
    even when ``satisfied=True`` an ``any_of`` branch may have unmet
    siblings; only ``unmet_branch=False`` reflects "fully satisfied
    every path".  Use ``gaps`` in refusal messages; do not use it as
    a secondary truthiness signal.
    """

    satisfied: bool
    gaps: tuple[str, ...] = ()

    def with_extra_gaps(self, extra: Iterable[str]) -> "CriterionEvalResult":
        return CriterionEvalResult(
            satisfied=self.satisfied,
            gaps=self.gaps + tuple(extra),
        )


def evaluate_done_criterion(
    criterion: CriterionItem,
    ctx: CriterionEvalContext,
) -> CriterionEvalResult:
    """Walk the AST against ``ctx``; return verdict + gap list."""
    kind = criterion.kind
    if kind == "node_closed":
        ok = criterion.node in ctx.closed_nodes
        gaps = () if ok else (f"node '{criterion.node}' not closed",)
        return CriterionEvalResult(satisfied=ok, gaps=gaps)

    if kind == "child_done":
        ok = criterion.child in ctx.done_children
        gaps = () if ok else (f"child task '{criterion.child}' not done",)
        return CriterionEvalResult(satisfied=ok, gaps=gaps)

    if kind == "named_artefact":
        ok = criterion.name in ctx.present_artefacts
        gaps = () if ok else (f"artefact '{criterion.name}' missing",)
        return CriterionEvalResult(satisfied=ok, gaps=gaps)

    if kind == "all_of":
        if not criterion.items:
            # Empty all_of is vacuously satisfied; the loader rejects
            # this shape at parse time, but the evaluator stays
            # defensive.
            return CriterionEvalResult(satisfied=True)
        gaps: list[str] = []
        ok = True
        for item in criterion.items:
            r = evaluate_done_criterion(item, ctx)
            if not r.satisfied:
                ok = False
            gaps.extend(r.gaps)
        return CriterionEvalResult(satisfied=ok, gaps=tuple(gaps))

    if kind == "any_of":
        if not criterion.items:
            # Empty any_of is unsatisfiable — caller should never write
            # this; treated as fail with a synthetic gap.
            return CriterionEvalResult(
                satisfied=False,
                gaps=("any_of with no items — criterion is unsatisfiable",),
            )
        results = [evaluate_done_criterion(item, ctx) for item in criterion.items]
        if any(r.satisfied for r in results):
            return CriterionEvalResult(satisfied=True)
        # None satisfied — surface every branch's gap so the user sees
        # the full picture (any one of which would have satisfied).
        gaps = []
        for i, r in enumerate(results):
            for g in r.gaps:
                gaps.append(f"any_of[{i}]: {g}")
        return CriterionEvalResult(satisfied=False, gaps=tuple(gaps))

    # Unknown kind — defensive; loader catches this at parse time.
    return CriterionEvalResult(
        satisfied=False,
        gaps=(f"unknown criterion kind '{kind}'",),
    )


def referenced_nodes(criterion: CriterionItem) -> frozenset[str]:
    """All node ids the criterion mentions via ``node_closed`` atoms.

    Used by the loader to verify every reference resolves to a
    declared node, and by the task tree to detect dangling refs.
    """
    out: set[str] = set()
    _collect_nodes(criterion, out)
    return frozenset(out)


def referenced_children(criterion: CriterionItem) -> frozenset[str]:
    """All child task ids the criterion mentions via ``child_done``."""
    out: set[str] = set()
    _collect_children(criterion, out)
    return frozenset(out)


def referenced_artefacts(criterion: CriterionItem) -> frozenset[str]:
    """All artefact names the criterion mentions via ``named_artefact``."""
    out: set[str] = set()
    _collect_artefacts(criterion, out)
    return frozenset(out)


def _collect_nodes(criterion: CriterionItem, out: set[str]) -> None:
    if criterion.kind == "node_closed" and criterion.node:
        out.add(criterion.node)
    for item in criterion.items:
        _collect_nodes(item, out)


def _collect_children(criterion: CriterionItem, out: set[str]) -> None:
    if criterion.kind == "child_done" and criterion.child:
        out.add(criterion.child)
    for item in criterion.items:
        _collect_children(item, out)


def _collect_artefacts(criterion: CriterionItem, out: set[str]) -> None:
    if criterion.kind == "named_artefact" and criterion.name:
        out.add(criterion.name)
    for item in criterion.items:
        _collect_artefacts(item, out)
