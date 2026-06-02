"""spec #4 — provenance-driven observation planner (general heuristic layer).

When a producer backtrace STALLS at ``NEEDS_OBSERVATION`` / a boundary, the
existing :class:`engine.oracle_provenance.ProvenanceResult` already carries a
``next_watch`` list of address-gaps (addr/pc/reason). That list is generated from
*address gaps only*. This module adds the missing **instruction / edge-shape
heuristic** layer: an extensible registry of small (matcher → proposal-builder)
rules that read the *code shape* around a gap and propose the next batch of
concrete observe points, each tagged with a ``reason`` + ``heuristic``.

GENERAL-FIRST (A8②): the rules key off generic aarch64 instruction shapes and the
target-agnostic :mod:`engine.import_map` summary table — never a baked address.
TC2 sink→src32 is a proof-point, not the design target. Adding a heuristic = one
:class:`Rule` entry (prove it with a test). The rule registry + its three seed
rules + the proposal/context types live in the sibling :mod:`engine._rules` (kept
under the 500-line/file ceiling); this module is the public facade.

Reuse, don't rebuild (A8①):
  * :class:`engine.runner_client.ObservePoint` / :class:`RegRelWatch` — the wire
    types; :func:`suggest_observations` builds these via the existing
    :func:`engine.watch_first_write.request_point_watch` builder.
  * :func:`engine.runner_client.mem_snapshots_from_rerun` + ``adapter.rerun`` —
    :func:`run_plan` is a THIN standalone wrapper over the exact two calls
    :func:`engine.recapture_loop.run_recapture_loop` already makes per round,
    surfaced so a caller mid-analysis can run one plan WITHOUT driving the whole
    convergence loop. Same-execution G1 invariant inherited from that
    rerun/snapshot path.
  * :class:`engine.import_map.ImportMap` / :class:`ExternSummary` — the
    ``extern_call`` rule reads ABI arg roles from the per-symbol summary.

Preserve (A8③): purely additive. ``next_watch`` and ``run_recapture_loop`` are
untouched; ``observation_plan`` sits ALONGSIDE ``next_watch``.

Degenerate → surfaced (A8④): a gap that no rule matches is NEVER silently dropped
— it still appears in ``next_watch`` (that list is generated independently of this
module). A silently-empty plan that hides an unobserved gap is the one thing this
module must never produce; the ``observation_plan`` is an ADDITION on top of
``next_watch``, never a replacement.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from ._rules import (
    DEFAULT_RULES,
    ObserveProposal,
    Rule,
    RuleContext,
    _parse_addr,
    rule_boundary_copy,
    rule_extern_call,
    rule_write_chain,
)
from .import_map import ImportMap
from .oracle_provenance import ProvenanceResult
from .runner_client import ObservePoint, RerunResult, mem_snapshots_from_rerun
from .types import Instruction

__all__ = [
    "ObserveProposal",
    "RuleContext",
    "Rule",
    "DEFAULT_RULES",
    "rule_write_chain",
    "rule_extern_call",
    "rule_boundary_copy",
    "suggest_proposals",
    "suggest_observations",
    "plan_for_result",
    "run_plan",
]


# --------------------------------------------------------------------------- #
# Public API.
# --------------------------------------------------------------------------- #


def suggest_proposals(
    prov: ProvenanceResult,
    items: Iterable[Instruction],
    *,
    rules: Sequence[Rule] = DEFAULT_RULES,
    import_map: ImportMap | None = None,
) -> list[ObserveProposal]:
    """Run the heuristic registry over a provenance result's gaps and return the
    proposed :class:`ObserveProposal`s (the ``observation_plan`` entries). Pure.

    For every ``next_watch`` gap each rule's matcher is consulted; matching rules
    build proposals. Result-level rules (e.g. ``boundary_copy``, which keys off
    ``prov.anchored_edge`` rather than a gap PC) are also run once with no gap, so
    they fire even when ``next_watch`` is empty. Duplicate proposals (same pc/when/
    capture/regs/mem/heuristic) are de-duplicated, preserving first-seen order. A
    gap no rule matches is left to ``next_watch`` (NOT dropped — A8④)."""
    items = tuple(items)
    by_pc: dict[int, list[Instruction]] = {}
    for ins in items:
        by_pc.setdefault(ins.pc, []).append(ins)

    proposals: list[ObserveProposal] = []
    seen: set[tuple] = set()

    def _emit(props: list[ObserveProposal]) -> None:
        for p in props:
            key = (p.pc, p.when, p.capture, p.regs, p.mem, p.heuristic)
            if key not in seen:
                seen.add(key)
                proposals.append(p)

    # Per-gap rules (write_chain / extern_call key off the gap PC's instruction).
    for gap in prov.next_watch:
        gap_pc = _parse_addr(gap.get("pc"))
        gap_addr = _parse_addr(gap.get("addr"))
        ctx = RuleContext(prov=prov, items=items, by_pc=by_pc,
                          gap_addr=gap_addr, gap_pc=gap_pc, import_map=import_map)
        for rule in rules:
            if rule.matches(ctx):
                _emit(rule.build(ctx))

    # Result-level rules (run once with no gap so they fire even with empty gaps).
    ctx0 = RuleContext(prov=prov, items=items, by_pc=by_pc, import_map=import_map)
    for rule in rules:
        if rule.matches(ctx0):
            _emit(rule.build(ctx0))

    return proposals


def suggest_observations(
    prov: ProvenanceResult,
    items: Iterable[Instruction],
    *,
    rules: Sequence[Rule] = DEFAULT_RULES,
    import_map: ImportMap | None = None,
) -> list[ObservePoint]:
    """Pure planner: heuristic-propose the next batch of runner-fulfillable
    :class:`ObservePoint`s for a stalled provenance result. Thin lowering of
    :func:`suggest_proposals` (which carries the reasons/heuristic tags)."""
    return [p.to_observe_point()
            for p in suggest_proposals(prov, items, rules=rules, import_map=import_map)]


def plan_for_result(
    prov: ProvenanceResult,
    items: Iterable[Instruction],
    *,
    rules: Sequence[Rule] = DEFAULT_RULES,
    import_map: ImportMap | None = None,
) -> list[dict[str, Any]]:
    """Build the ``observation_plan`` field value (list of proposal dicts) for a
    provenance result. This is what :mod:`engine.oracle_provenance` attaches
    ALONGSIDE ``next_watch``. Gaps no rule covers are NOT added here — they stay
    surfaced in ``next_watch`` (A8④: never a silently-empty plan hiding a gap)."""
    return [p.to_dict()
            for p in suggest_proposals(prov, items, rules=rules, import_map=import_map)]


def run_plan(
    adapter: Any,
    input_bytes: bytes,
    plan: Iterable[ObservePoint],
) -> RerunResult:
    """Accept a plan & rerun: the standalone, one-shot version of the inner step
    :func:`engine.recapture_loop.run_recapture_loop` runs per round.

    A THIN wrapper (A8①): it makes the EXACT two calls the loop makes —
    ``adapter.rerun(input_bytes, observe_points)`` then
    :func:`engine.runner_client.mem_snapshots_from_rerun` to fold the captured mem
    into canonical snapshots — and returns the :class:`RerunResult` (snapshots are
    re-derivable from it via the same helper). It is NOT a new loop and adds no
    convergence logic; it lets a caller mid-analysis run ONE plan and feed the
    snapshots back into ``trace_provenance`` by hand. Same-execution G1 invariant
    is inherited from this rerun/snapshot path (one rerun → one execution).

    Raises ``TypeError`` if the adapter returns a non-:class:`RerunResult`, mirroring
    the loop's guard (a misbehaving adapter fails loud, never silently)."""
    observe_points = list(plan)
    result = adapter.rerun(input_bytes, observe_points)
    if not isinstance(result, RerunResult):
        raise TypeError(
            "adapter.rerun must return a RerunResult; got "
            f"{type(result).__name__} (run_plan folds RerunResult.mem captures into "
            "snapshots via mem_snapshots_from_rerun — the same call run_recapture_loop "
            "makes)")
    # Fold mem captures → canonical snapshots (the loop's next step). Run it eagerly
    # so any truncation WARN surfaces here too; the snapshots ride on the result the
    # caller can re-derive with mem_snapshots_from_rerun.
    mem_snapshots_from_rerun(result)
    return result
