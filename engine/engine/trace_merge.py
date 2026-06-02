"""Multi-source trace merge (item ① of the trace-sufficiency work package).

The analyses (build_dfg / cohort_diff / opaque_staging / oracle_provenance) all
read the main :class:`Instruction` stream. Real traces, however, often carry the
memory dimension in a SEPARATE sidecar (``_mem.jsonl``) and register-pointed
dumps in a hook sidecar — so ``Instruction.mem`` is empty and the memory
dimension is lost. This module merges already-parsed CANONICAL sidecar data back
into the main stream.

Architecture boundary (hard — see :class:`engine.types.MemSnapshot` docstring):
the engine NEVER parses a runner format. ``obs_readers`` / the trace readers turn
sidecar files into canonical shapes (:class:`MemEvent` / :class:`MemSnapshot`);
THIS module only consumes those canonical shapes. A reader "supporting a sidecar"
means it reads the sidecar file into canonical events, then hands them to
:func:`merge_trace_sources` — the parse stays in the adapter/reader layer.

Honesty / no-fabrication (hard boundary): merge only folds in data that ALREADY
EXISTS in canonical form. Snapshots are attached as region OBSERVATIONS, never
forged into executed ``MemOp`` steps (a snapshot is a view of memory, not an
executed read/write — keeping the ``MemOp`` vs ``MemSnapshot`` semantic split of
``obs_readers``). Sidecar events that align to no main instruction are surfaced in
``unaligned`` — never silently dropped, never force-fit.

No sidecar / nothing to merge → the output is the input verbatim, instruction by
instruction (invariant 7: the green baseline is byte-for-byte unchanged).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from .types import Instruction, MemOp, MemSnapshot

__all__ = [
    "MemEvent",
    "MergeReport",
    "MergedTrace",
    "merge_trace_sources",
]


@dataclass(frozen=True, slots=True)
class MemEvent:
    """A canonical memory read/write event from a separate execution sidecar
    (e.g. ``_mem.jsonl``), carrying a locating key so it can be aligned back onto
    the main instruction stream.

    ``idx`` is the preferred alignment key (it pins the exact trace step); ``pc``
    is the fallback when the sidecar only records the program counter. At least
    one must be non-None — an event with neither cannot be aligned and is reported
    ``unaligned``. ``op`` is the canonical :class:`MemOp` to fold into the matched
    instruction's ``mem`` tuple."""

    op:  MemOp
    idx: int | None = None
    pc:  int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
            "pc":  None if self.pc is None else f"0x{self.pc:x}",
            "op":  {"rw": self.op.rw, "addr": f"0x{self.op.addr:x}",
                    "val": self.op.val, "size": self.op.size},
        }


@dataclass(frozen=True, slots=True)
class MergeReport:
    """What the merge folded in and what it could not align — never silent.

    ``n_items`` — main stream length (unchanged by merge). ``mem_events_merged``
    — sidecar memory events folded into an instruction's ``mem``. ``reg_overlay_*``
    — overlay register values filled (``filled`` = a missing key supplied;
    ``conflicts`` = an overlay value that disagreed with an already-present trace
    value, NOT applied, surfaced here). ``unaligned`` — sidecar events whose
    idx/pc matched no main instruction. ``snapshots_attached`` — region
    observations carried alongside (NOT turned into MemOps)."""

    n_items:            int
    mem_events_merged:  int = 0
    reg_overlay_filled: int = 0
    reg_overlay_conflicts: tuple[dict, ...] = ()
    unaligned:          tuple[dict, ...] = ()
    snapshots_attached: int = 0
    alignment_key:      str = "idx"   # the key actually used ("idx" | "pc")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind":                  "trace_merge_report",
            "n_items":               self.n_items,
            "mem_events_merged":     self.mem_events_merged,
            "reg_overlay_filled":    self.reg_overlay_filled,
            "reg_overlay_conflicts": list(self.reg_overlay_conflicts),
            "unaligned":             list(self.unaligned),
            "snapshots_attached":    self.snapshots_attached,
            "alignment_key":         self.alignment_key,
        }


@dataclass(frozen=True, slots=True)
class MergedTrace:
    """The merged instruction stream plus the carried region observations + report.

    ``items`` is the new main stream (frozen :class:`Instruction` objects — new
    objects where a merge changed a step, the SAME object where nothing changed).
    ``snapshots`` are the region observations carried for downstream sink-captured
    / provenance checks (③/④), never folded into ``mem``."""

    items:     tuple[Instruction, ...]
    snapshots: tuple[MemSnapshot, ...] = ()
    report:    MergeReport | None = None


def merge_trace_sources(
    main: Sequence[Instruction],
    *,
    mem_events: Sequence[MemEvent] = (),
    snapshots: Sequence[MemSnapshot] = (),
    reg_overlay: Mapping[int, Mapping[str, int]] | None = None,
    align_by: str = "idx",
) -> MergedTrace:
    """Fold canonical sidecar data into the ``main`` instruction stream.

    Deterministic. ``main`` is the skeleton; its order is preserved verbatim.

    Algorithm:
      1. Memory events are aligned to a main instruction by ``idx`` (preferred) or
         ``pc`` (fallback when the event has no idx). Multiple events for the same
         step are appended to ``mem`` in supplied order. ``Instruction`` is frozen,
         so a touched step becomes a NEW object (``dataclasses.replace``); an
         untouched step is left as the SAME object (invariant 7).
      2. ``reg_overlay`` (idx → {reg: value}) fills MISSING register keys only —
         a key already present in ``regs_read``/``regs_write`` is never overwritten;
         a disagreeing overlay value is recorded in ``reg_overlay_conflicts``
         (surfaced, not silently applied). Overlay fills ``regs_read`` (the overlay
         supplies observed live-in values); it never invents ``regs_write``.
      3. Snapshots are carried as region observations on the result, NOT folded
         into ``mem`` (a :class:`MemSnapshot` is an observation, not an executed
         step).
      4. Any mem event whose idx/pc matches no main instruction is collected into
         ``report.unaligned`` (never dropped, never force-fit).

    With no ``mem_events`` / ``reg_overlay`` (the no-sidecar case), every output
    instruction is the SAME object as its input — byte-for-byte unchanged."""
    if align_by not in ("idx", "pc"):
        raise ValueError(f"align_by must be 'idx' or 'pc', got {align_by!r}")
    items = list(main)
    reg_overlay = reg_overlay or {}

    # Index the main stream by both keys so a per-event preference (idx, else pc)
    # can be honoured without re-scanning. idx is unique per step; a pc may recur
    # (a loop) — for the pc fallback we align to the FIRST instruction at that pc.
    by_idx: dict[int, int] = {}        # idx -> position in items
    by_pc_first: dict[int, int] = {}   # pc  -> first position in items
    for pos, ins in enumerate(items):
        by_idx.setdefault(ins.idx, pos)
        by_pc_first.setdefault(ins.pc, pos)

    # Accumulate per-position folded mem ops + reg fills before rebuilding, so a
    # step touched by several events/overlay is rebuilt exactly once.
    add_mem: dict[int, list[MemOp]] = {}
    add_reads: dict[int, dict[str, int]] = {}
    unaligned: list[dict] = []
    conflicts: list[dict] = []

    for ev in mem_events:
        pos: int | None = None
        if ev.idx is not None and ev.idx in by_idx:
            pos = by_idx[ev.idx]
        elif ev.pc is not None and ev.pc in by_pc_first:
            pos = by_pc_first[ev.pc]
        if pos is None:
            unaligned.append(ev.to_dict())
            continue
        add_mem.setdefault(pos, []).append(ev.op)

    reg_overlay_filled = 0
    for idx, regs in reg_overlay.items():
        pos = by_idx.get(idx)
        if pos is None:
            for r, v in regs.items():
                unaligned.append({
                    "idx": idx, "pc": None,
                    "reg_overlay": {r: v},
                    "reason": "reg_overlay idx matches no main instruction",
                })
            continue
        ins = items[pos]
        for r, v in regs.items():
            existing = ins.regs_read.get(r, ins.regs_write.get(r))
            if existing is not None:
                if existing != v:
                    conflicts.append({
                        "idx": idx, "reg": r,
                        "trace_value": existing, "overlay_value": v,
                        "reason": "overlay disagrees with present trace value "
                                  "— overlay NOT applied",
                    })
                # present and equal (or conflicting) → never overwrite the trace.
                continue
            add_reads.setdefault(pos, {})[r] = v
            reg_overlay_filled += 1

    # Rebuild only the touched positions; untouched stay the SAME object.
    touched = set(add_mem) | set(add_reads)
    mem_events_merged = 0
    for pos in touched:
        ins = items[pos]
        new_mem = ins.mem
        if pos in add_mem:
            new_mem = ins.mem + tuple(add_mem[pos])
            mem_events_merged += len(add_mem[pos])
        new_reads = ins.regs_read
        if pos in add_reads:
            new_reads = {**ins.regs_read, **add_reads[pos]}
        items[pos] = replace(ins, regs_read=new_reads, mem=new_mem)

    report = MergeReport(
        n_items=len(items),
        mem_events_merged=mem_events_merged,
        reg_overlay_filled=reg_overlay_filled,
        reg_overlay_conflicts=tuple(conflicts),
        unaligned=tuple(unaligned),
        snapshots_attached=len(snapshots),
        alignment_key=align_by,
    )
    return MergedTrace(items=tuple(items), snapshots=tuple(snapshots), report=report)
