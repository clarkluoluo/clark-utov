"""Instruction-level / register-level VMP observation API
(capability_request.md §P0-1, §P0-2).

The reference target hindsight: ``ObservePoint`` (one PC, single dump) was not
enough to follow ``x22 / x24 / x27`` inside the VMP compress_leg —
those registers flow through tens of handler instructions before the
real SM3 block input materialises in memory. The agent needed *step-
through* register history within a PC band; the current PLT/BL-level
hook gives only single-point snapshots.

This module specifies the new capability, surfaces a fallback that
synthesises the same data from an in-memory trace (so the engine can
already answer agent queries against File-mode runs while the Live-mode
``code_hook_range`` is being implemented), and gives the runner a
contract to grow into.

API summary (consumed by stages / agent RPC):

  - ``CodeHookRange(start_pc, end_pc, regs, step="every"|"on_change")``
    declares "trace these registers at every (or on-change) instruction
    in [start_pc, end_pc)". Step-callback semantics, not single dump.
  - ``register_trace_from_instructions(items, hooks)`` derives the same
    RegisterTrace shape from a static Instruction list — used by both
    the File-mode default and the live fallback.
  - ``RunnerAdapter.code_hook_range(input_bytes, hooks)`` is the
    runtime version (Java side). Default `NotImplementedError` — older
    runners route through ``get_trace`` + the synthesis helper.

The synthesis helper is the engine-side win — it unlocks the
``compress_leg`` watcher on existing File-mode samples without any
runner change.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from .types import Instruction


StepMode = Literal["every", "on_change"]


@dataclass(frozen=True, slots=True)
class CodeHookRange:
    """A PC-band step-callback observation request.

    ``start_pc`` inclusive, ``end_pc`` exclusive. ``regs`` lists the
    registers whose value should be recorded; an empty tuple means
    "all read/written registers at each step".

    ``step``:
      - ``"every"``     — emit one ``RegisterTraceEntry`` per instruction
                          in band (fires often; use a narrow band).
      - ``"on_change"`` — emit only when any of ``regs`` changes value.
                          Right default for compress_leg / x22 tracking.
    """
    start_pc: int
    end_pc: int
    regs: tuple[str, ...] = ()
    step: StepMode = "on_change"
    label: str | None = None     # optional human name, e.g. "compress_leg"


@dataclass(frozen=True, slots=True)
class RegisterTraceEntry:
    """One step within a CodeHookRange. ``idx`` is the global trace
    index; ``pc`` the instruction PC; ``regs`` the values that were
    requested (whether they changed at this step or not)."""
    idx: int
    pc: int
    regs: dict[str, int] = field(default_factory=dict)
    # mem ops at this step, when the underlying source carries them
    mem_reads:  tuple[tuple[int, int], ...] = ()    # (addr, size)
    mem_writes: tuple[tuple[int, int, int], ...] = ()  # (addr, val, size)


@dataclass(frozen=True, slots=True)
class RegisterTrace:
    """Result of one CodeHookRange: ordered list of entries within band."""
    hook: CodeHookRange
    entries: tuple[RegisterTraceEntry, ...] = ()

    def unique_register_values(self, reg: str) -> tuple[int, ...]:
        """Return the distinct values seen for ``reg`` in order of first
        appearance. Used by M3 hook sanity (≥3 distinct across inputs)."""
        seen: list[int] = []
        sset: set[int] = set()
        for e in self.entries:
            v = e.regs.get(reg)
            if v is None or v in sset:
                continue
            seen.append(v)
            sset.add(v)
        return tuple(seen)

    def first_change_idx(self, reg: str) -> int | None:
        """Trace idx where ``reg`` first changes from its initial value.
        Useful for locating the compress_leg w24 transition (0x54→0x22
        →0x2a)."""
        baseline: int | None = None
        for e in self.entries:
            v = e.regs.get(reg)
            if v is None:
                continue
            if baseline is None:
                baseline = v
                continue
            if v != baseline:
                return e.idx
        return None


# ---------------------------------------------------------------------------
# Engine-side synthesis: derive a RegisterTrace from an in-memory
# Instruction list. Works for File-mode runs (replays a static trace)
# and for Live-mode runners that haven't yet implemented native code-
# hook ranges — caller falls back to ``get_trace`` + this helper.
# ---------------------------------------------------------------------------


def register_trace_from_instructions(
    items: Iterable[Instruction],
    hook: CodeHookRange,
) -> RegisterTrace:
    """Walk ``items`` and emit one RegisterTraceEntry per qualifying
    instruction. ``items`` must be in trace order.

    Semantics:
      - keep only instructions with ``hook.start_pc <= pc < hook.end_pc``
      - ``regs == ()`` → record every read/written register seen at the
        instruction; otherwise record only the requested subset.
      - on ``step="every"`` emit each in-band instruction; on
        ``step="on_change"`` emit only when at least one tracked reg's
        post-instruction value differs from the previous emitted entry.
    """
    requested = hook.regs
    entries: list[RegisterTraceEntry] = []
    last_emitted_regs: dict[str, int] = {}

    for ins in items:
        if not (hook.start_pc <= ins.pc < hook.end_pc):
            continue

        snapshot: dict[str, int] = {}
        # Build the post-state from both regs_read (the source values
        # we know going in) and regs_write (the result of this step).
        # When requested is empty we take the union of both maps.
        source = {**ins.regs_read, **ins.regs_write}
        if requested:
            for r in requested:
                if r in source:
                    snapshot[r] = source[r]
        else:
            snapshot = dict(source)

        if hook.step == "on_change":
            # Only emit if something tracked changed since last emit.
            if requested:
                changed = any(snapshot.get(r) != last_emitted_regs.get(r) for r in requested)
            else:
                changed = snapshot != last_emitted_regs
            if not changed:
                continue

        mr = tuple((m.addr, m.size) for m in ins.mem if m.rw == "r")
        mw = tuple((m.addr, m.val, m.size) for m in ins.mem if m.rw == "w")
        entries.append(RegisterTraceEntry(
            idx=ins.idx, pc=ins.pc, regs=snapshot,
            mem_reads=mr, mem_writes=mw,
        ))
        last_emitted_regs = dict(snapshot)

    return RegisterTrace(hook=hook, entries=tuple(entries))


# ---------------------------------------------------------------------------
# M3 sanity helper — uniqueness check across N inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HookSanity:
    """Result of running a CodeHookRange against ≥3 inputs.

    ``per_input`` carries one RegisterTrace per probe. ``unique_count``
    counts distinct value tuples seen for the *full* tracked register
    snapshot at hook end — when this is 1 across inputs the hook is on
    a constant buffer (M3 ``INVALID_CONSTANT_BUFFER``).
    """
    hook: CodeHookRange
    per_input: tuple[RegisterTrace, ...]
    unique_count: int
    threshold: int
    invalid_constant: bool


def evaluate_hook_sanity(
    traces: list[RegisterTrace],
    *,
    min_inputs: int = 3,
) -> HookSanity:
    """Given the same CodeHookRange's RegisterTrace from ≥``min_inputs``
    distinct inputs, classify whether the hook captures variable or
    constant data.

    The decision rule mirrors `mechanism_improvements.md §M3`:
        unique_count < min(3, vectors/2)   →   INVALID_CONSTANT_BUFFER
    """
    assert traces, "evaluate_hook_sanity needs at least one trace"
    hook = traces[0].hook
    # Use the LAST entry's register snapshot as the per-input fingerprint
    # — that's the post-band value, which is what the surrounding gate
    # is conceptually asking about.
    fingerprints: set[tuple[tuple[str, int], ...]] = set()
    for t in traces:
        last = t.entries[-1] if t.entries else None
        fp = tuple(sorted(last.regs.items())) if last else ()
        fingerprints.add(fp)
    vectors = len(traces)
    # M3 strict: with vectors >= min_inputs, require ≥ min(3, vectors)
    # distinct fingerprints. With exactly 3 inputs all three must vary;
    # with 10 inputs we accept 3+ unique (a noisy-but-real hook).
    threshold = min(3, vectors) if vectors >= min_inputs else 1
    unique = len(fingerprints)
    invalid = vectors >= min_inputs and unique < threshold
    return HookSanity(
        hook=hook,
        per_input=tuple(traces),
        unique_count=unique,
        threshold=threshold,
        invalid_constant=invalid,
    )


# ---------------------------------------------------------------------------
# Trace-window override (P0-2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TraceWindow:
    """A PC band the engine asks the runner to capture beyond the
    default ``[algo_entry_pc, algo_exit_pc]``. Multiple windows let
    main-VMP bands (e.g. ``0x32302c..0x325708``) be added without
    moving the primary anchors."""
    start_pc: int
    end_pc: int
    label: str | None = None


def merge_windows(
    primary: tuple[int, int],
    extra:   Iterable[TraceWindow],
) -> tuple[tuple[int, int], ...]:
    """Return the sorted, merged set of (start, end) bands. Overlapping
    bands collapse into one; the primary always anchors band #0.

    The result feeds runner ``get_trace`` (which can issue one call per
    band) and S4 ``producer_backward`` (which traverses across all
    bands when reverse-slicing).
    """
    items: list[tuple[int, int]] = [primary] + [(w.start_pc, w.end_pc) for w in extra]
    items.sort()
    merged: list[tuple[int, int]] = []
    for s, e in items:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    return tuple(merged)


def in_any_window(pc: int, windows: tuple[tuple[int, int], ...]) -> bool:
    return any(s <= pc < e for s, e in windows)
