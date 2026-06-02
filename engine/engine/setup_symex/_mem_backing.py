"""setup_symex.mem_backing section (split from the monolithic module)."""
from __future__ import annotations


import enum
import os
import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..dataflow import classify_semop
from ..types import Instruction, MemSnapshot
from ..watch_first_write import (
    WatchFirstWriteConfig,
    WatchFirstWriteSpec,
    request_watch_first_write,
)
from ._hybrid import _MEM_SEMOPS


@dataclass(frozen=True, slots=True)
class MemBackingReport:
    """Whether the materialization / staging window carries mem[] operands.

    The backward alias spine walks the trace's mem[]. If memory-class
    instructions in the staging window carry no mem[] (the runner did not emit
    read/write operands), the memory leg is BLIND and the backtrace degrades to
    a cross-run-diff guess. This report surfaces that BEFORE the agent spends a
    pass on a placeholder result (the case lost 57/65 bytes exactly this way)."""

    window_pcs:           tuple[int, int]  # (lo, hi) pc band examined
    mem_class_steps:      int              # memory-class instructions in window
    backed_steps:         int              # backed by a trace mem[] operand OR snapshot
    blind_pcs:            tuple[int, ...]  # PCs of steps still blind (no operand, no backing)
    sufficient:           bool
    snapshot_backed_steps: int = 0         # of backed_steps, how many via snapshot/hook closure

    @property
    def backing_rate(self) -> float:
        if self.mem_class_steps == 0:
            return 1.0
        return self.backed_steps / self.mem_class_steps

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_pcs":            [f"0x{self.window_pcs[0]:x}", f"0x{self.window_pcs[1]:x}"],
            "mem_class_steps":       self.mem_class_steps,
            "backed_steps":          self.backed_steps,
            "snapshot_backed_steps": self.snapshot_backed_steps,
            "backing_rate":          round(self.backing_rate, 4),
            "blind_pcs":             [f"0x{p:x}" for p in self.blind_pcs],
            "sufficient":            self.sufficient,
            "kind":                  "setup_symex_mem_backing",
        }

    @property
    def advisory(self) -> str:
        if self.sufficient:
            via = (f" ({self.snapshot_backed_steps} via same-execution "
                   f"snapshot/hook backing)" if self.snapshot_backed_steps else "")
            return (
                f"mem[] backing OK ({self.backed_steps}/{self.mem_class_steps} "
                f"memory steps backed{via}) — the alias spine has its substrate"
            )
        return (
            f"mem[] BLIND: {len(self.blind_pcs)} memory-class step(s) in the "
            f"staging window carry no mem[] — the memory leg is blind and the "
            f"backtrace will degrade to a cross-run-diff guess. Re-capture the "
            f"trace with read/write operands over this window before backtracing "
            f"(observation decides recoverability)."
        )


def check_mem_backing(
    items: Iterable[Instruction],
    *,
    window: tuple[int, int] | None = None,
    window_is_idx: bool = False,
    backing: "ConcreteBacking | None" = None,
    backed_regs: Iterable[str] = (),
    backed_addrs: Iterable[int] = (),
    trace_exec_id: str | None = None,
    max_depth: int = 64,
) -> MemBackingReport:
    """Audit whether memory-class steps in ``window`` are backed.

    ``window`` is an inclusive band — a ``(pc_lo, pc_hi)`` address band by default,
    or a ``(idx_lo, idx_hi)`` trace-execution-order band when ``window_is_idx`` is
    set (a VMP handler segment). It is the materialization /
    staging region. A memory-class step is BACKED when it either carries a
    trace ``mem[]`` operand OR — for runners that emit reg-traces without store
    EAs — its effective-address closure resolves against same-execution
    snapshot/hook backing (``backing`` / ``backed_regs`` / ``backed_addrs``).
    This is the unified backing criterion: a step that :func:`audit_address_closure`
    judges backed is no longer counted blind here, so the two C4 checks can no
    longer give opposite answers on the same "is it backed?" question.

    The determinism guard: when ``trace_exec_id`` is asserted, a ``backing``
    whose ``exec_id`` differs is NOT counted (a stale / cross-run snapshot must
    not mask a real blind leg → no false-pass). Without backing this reduces to
    the original mem[]-presence audit."""
    items = list(items)
    lo, hi = _window_bounds(items, window, by_idx=window_is_idx)
    breg, baddr = _effective_backing(backing, backed_regs, backed_addrs, trace_exec_id)
    win = [ins for ins in items
           if lo <= (ins.idx if window_is_idx else ins.pc) <= hi]
    resolve = _build_resolver(win, breg, baddr, max_depth) if (breg or baddr) else None

    mem_class = 0
    backed = 0
    snapshot_backed = 0
    blind: list[int] = []
    for ins in win:
        if classify_semop(ins.mnemonic) not in _MEM_SEMOPS:
            continue
        mem_class += 1
        if ins.mem:
            backed += 1
            continue
        # Empty mem[]: still backed if the EA closure resolves against the
        # same-execution snapshot/hook (agree with audit_address_closure).
        if resolve is not None and _addr_regs(ins.mnemonic) \
                and not _step_unbacked(resolve, ins):
            backed += 1
            snapshot_backed += 1
            continue
        # §5′ VALUE dimension (dynamic / concolic backing): a mem-class step with
        # no mem[] operand is still NOT blind if the value it moves is observable
        # on the register side — for a load, the loaded value lands in the dest
        # register (visible in ``regs_write``; recoverable via the ② regs_read/
        # regs_write fallback); for a store, the stored value is read from a source
        # register (``regs_read``). A truly blind leg = the EA (op.addr) is absent
        # AND the value is absent on both the mem[] and the register side.
        if _step_value_on_regs(ins):
            backed += 1
            continue
        if ins.pc not in blind:
            blind.append(ins.pc)
    sufficient = (mem_class == 0) or (not blind)
    return MemBackingReport(
        window_pcs=(lo, hi),
        mem_class_steps=mem_class,
        backed_steps=backed,
        blind_pcs=tuple(blind),
        sufficient=sufficient,
        snapshot_backed_steps=snapshot_backed,
    )


# ---------------------------------------------------------------------------
# Contract 4 (backing arm) — address-computation CLOSURE audit.
#
# check_mem_backing answers "do the load/store steps in the window carry mem[]
# operands?" — necessary but not sufficient. The cipher-body hash loop passed
# that audit 4/4 yet still emitted SymVar_0 (parity 0/3): every load DID carry
# an operand, but the BASE registers those loads computed their EA from
# (x20/x24/x25) had no concrete value and pointed at memory that was never
# backed. The indirect access had no real EA to resolve, so the symbol never
# reached the mixing and the input passed straight through.
#
# This audit closes that gap. From each load/store's effective address it walks
# the address computation BACKWARD — base/index register → whoever wrote it
# (an ALU op, or a chained load) → recursively to the live-in roots — and asks
# whether the WHOLE closure has concrete backing. Any un-backed address leg is
# flagged BEFORE emit (re-capture / inject backing, never emit a stub).
# ---------------------------------------------------------------------------


# Loads that bring a value IN from memory (a base register written by one of
# these was itself read from memory — its pointed bytes are part of the closure).
_MEM_LOAD_SEMOPS = frozenset({"memory_load", "stack_restore"})

# Pull the address-operand register tokens out of a mnemonic: the registers
# inside the ``[...]`` are the base/index that compute the effective address.
_ADDR_OPERAND_RE = re.compile(r"\[([^\]]*)\]")
_ADDR_REG_RE = re.compile(r"\b(x\d+|w\d+|sp)\b")


def _addr_regs(mnemonic: str) -> tuple[str, ...]:
    """Registers used to compute the effective address (inside ``[...]``)."""
    m = _ADDR_OPERAND_RE.search(mnemonic)
    if not m:
        return ()
    return tuple(dict.fromkeys(_ADDR_REG_RE.findall(m.group(1))))


def _step_value_on_regs(ins: Instruction) -> bool:
    """Whether the value a mem-class step moves is observable on the register side.

    §5′ dynamic-backing VALUE dimension (structural, no addresses/case names): a
    load brings a value IN to its destination register — observable when the step
    has any ``regs_write`` (the dest value, recoverable via the ② regs_read/
    regs_write fallback). A store sends a value OUT from a source register —
    observable when the step has any ``regs_read``. Either presence means the
    moved value is in the trace's register file, so the leg is NOT value-blind
    even though no ``mem[]`` operand carried the address+value."""
    semop = classify_semop(ins.mnemonic)
    if semop in _MEM_LOAD_SEMOPS:
        return bool(ins.regs_write)
    # stores / stack_save: the value flows out of a source register.
    return bool(ins.regs_read)


def _window_bounds(
    items: list[Instruction], window: tuple[int, int] | None, *, by_idx: bool = False,
) -> tuple[int, int]:
    """Resolve an inclusive ``(lo, hi)`` band (whole trace when ``window`` is None).

    ``by_idx`` selects the basis for the whole-trace fallback: trace ``idx`` order
    vs ``pc`` address. When ``window`` is given the bounds are taken verbatim (they
    are already in the caller's basis)."""
    if window is None:
        vals = [(ins.idx if by_idx else ins.pc) for ins in items] or [0]
        return min(vals), max(vals)
    lo, hi = int(window[0]), int(window[1])
    return (lo, hi) if lo <= hi else (hi, lo)


def _effective_backing(
    backing: "ConcreteBacking | None",
    backed_regs: Iterable[str],
    backed_addrs: Iterable[int],
    trace_exec_id: str | None,
) -> tuple[set[str], set[int]]:
    """The backing sets to count, after the determinism guard.

    A :class:`ConcreteBacking` is counted only when it is same-execution: if the
    caller asserts ``trace_exec_id`` and the backing's ``exec_id`` differs, it is
    a cross-run / stale snapshot and is dropped (so it cannot mask a real blind
    leg). Raw ``backed_regs`` / ``backed_addrs`` are the unscoped escape hatch."""
    if backing is not None:
        if trace_exec_id is not None and backing.exec_id != trace_exec_id:
            return set(), set()
        return set(backing.backed_regs), set(backing.backed_addrs)
    return set(backed_regs), set(backed_addrs)


def _build_resolver(
    win: list[Instruction], breg: set[str], baddr: set[int], max_depth: int = 64,
):
    """Return ``resolve(reg, before_idx, depth, seen)`` over the window's def-use.

    Walks an address register backward to its live-in roots: an ALU writer
    recurses into its sources; a chained load recurses into the load's pointer
    regs and needs the load's pointed bytes captured; an un-written register is a
    live-in root that must be in ``breg``. Returns the set of un-backed roots
    (empty = the closure is fully backed)."""
    def latest_writer(reg: str, before_idx: int) -> Instruction | None:
        writer = None
        for ins in win:
            if ins.idx < before_idx and reg in ins.regs_write:
                if writer is None or ins.idx > writer.idx:
                    writer = ins
        return writer

    def resolve(reg: str, before_idx: int, depth: int,
                seen: set[tuple[str, int]]) -> set[str]:
        if reg in breg or reg == "sp":   # backed value, or the stack pointer base
            return set()
        key = (reg, before_idx)
        if key in seen or depth > max_depth:
            return set()
        seen.add(key)
        writer = latest_writer(reg, before_idx)
        if writer is None:
            return {reg}                 # live-in register, no concrete backing
        semop = classify_semop(writer.mnemonic)
        unbacked: set[str] = set()
        if semop in _MEM_LOAD_SEMOPS:
            # value came FROM memory: its pointed bytes must be captured, and the
            # pointer (the load's own address regs) must itself resolve.
            ea_observed = bool(writer.mem) or any(
                op.addr in baddr for op in writer.mem
            )
            if not ea_observed:
                unbacked.add(f"mem@0x{writer.pc:x}")
            for areg in _addr_regs(writer.mnemonic):
                unbacked |= resolve(areg, writer.idx, depth + 1, seen)
            return unbacked
        # ALU / move / addr_calc: recurse into the source registers.
        for src in writer.regs_read:
            unbacked |= resolve(src, writer.idx, depth + 1, seen)
        return unbacked

    return resolve


def _step_unbacked(resolve, ins: Instruction) -> set[str]:
    """Un-backed closure roots for one memory step's effective address."""
    unb: set[str] = set()
    for reg in _addr_regs(ins.mnemonic):
        unb |= resolve(reg, ins.idx, 0, set())
    return unb


@dataclass(frozen=True, slots=True)
class AddressLeg:
    """One memory op's effective-address dependency, resolved to its roots.

    ``addr_regs`` are the base/index registers in the op's ``[...]`` operand;
    ``unbacked`` are the closure roots (a live-in register with no concrete
    value, or a chained load whose pointed bytes were never captured) that have
    NO backing — each is a blind address leg."""

    pc:        int
    addr_regs: tuple[str, ...]
    unbacked:  tuple[str, ...]

    @property
    def backed(self) -> bool:
        return not self.unbacked


@dataclass(frozen=True, slots=True)
class AddressClosureReport:
    """Whether every address leg in the window has a backed closure.

    ``sufficient`` is True only when no address leg has an un-backed root. A
    False report means forward symex would resolve those loads against an
    unknown EA and emit a passthrough stub — inject the missing backing
    (:class:`ConcreteBacking`) or re-capture, then re-audit before emitting."""

    window_pcs:     tuple[int, int]
    legs:           tuple[AddressLeg, ...]
    unbacked_roots: tuple[str, ...]   # union of un-backed roots across all legs
    sufficient:     bool

    @property
    def blind_pcs(self) -> tuple[int, ...]:
        return tuple(leg.pc for leg in self.legs if not leg.backed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_pcs":     [f"0x{self.window_pcs[0]:x}", f"0x{self.window_pcs[1]:x}"],
            "legs": [
                {"pc": f"0x{leg.pc:x}", "addr_regs": list(leg.addr_regs),
                 "unbacked": list(leg.unbacked), "backed": leg.backed}
                for leg in self.legs
            ],
            "unbacked_roots": list(self.unbacked_roots),
            "blind_pcs":      [f"0x{p:x}" for p in self.blind_pcs],
            "sufficient":     self.sufficient,
            "kind":           "setup_symex_address_closure",
        }

    @property
    def advisory(self) -> str:
        if self.sufficient:
            return (
                f"address closure backed ({len(self.legs)} leg(s)) — every "
                f"load/store EA resolves to concrete roots; symex can compute "
                f"real effective addresses"
            )
        return (
            f"address closure BLIND: roots {sorted(self.unbacked_roots)} have no "
            f"concrete backing across {len(self.blind_pcs)} address leg(s). The "
            f"loads compute their EA from un-backed base register(s) / un-captured "
            f"pointed memory, so the indirect access degrades and forward symex "
            f"emits an input-passthrough stub. Inject ConcreteBacking for these "
            f"roots (or re-capture) BEFORE emitting — do not emit a stub."
        )


def audit_address_closure(
    items: Iterable[Instruction],
    *,
    window: tuple[int, int] | None = None,
    window_is_idx: bool = False,
    backing: ConcreteBacking | None = None,
    backed_regs: Iterable[str] = (),
    backed_addrs: Iterable[int] = (),
    trace_exec_id: str | None = None,
    max_depth: int = 64,
) -> AddressClosureReport:
    """Audit whether every load/store address closure in ``window`` is backed.

    For each memory-class step, the base/index registers in its ``[...]`` operand
    are resolved BACKWARD through the window's def-use chain: a register written
    by an ALU op recurses into that op's source registers; a register written by
    a chained load recurses into the load's pointer regs and requires the load's
    pointed bytes to be backed; a register never written in the window is a
    live-in root that must have concrete backing. Any root with no backing is a
    blind address leg.

    Backing is supplied either as a :class:`ConcreteBacking` (``backing=``, the
    contract-2 arm) or as raw ``backed_regs`` / ``backed_addrs`` sets, and is
    subject to the same-execution determinism guard (``trace_exec_id``) as
    :func:`check_mem_backing` — the two share one backing criterion, so a window
    cannot read backed by one C4 check and blind by the other.

    ``window`` is a ``(pc_lo, pc_hi)`` address band by default, or a trace
    ``(idx_lo, idx_hi)`` execution-order band when ``window_is_idx`` is set — the
    same basis :func:`check_mem_backing` uses, so both C4 arms select the same
    steps."""
    items = list(items)
    lo, hi = _window_bounds(items, window, by_idx=window_is_idx)
    breg, baddr = _effective_backing(backing, backed_regs, backed_addrs, trace_exec_id)
    win = [ins for ins in items
           if lo <= (ins.idx if window_is_idx else ins.pc) <= hi]
    resolve = _build_resolver(win, breg, baddr, max_depth)

    legs: list[AddressLeg] = []
    all_unbacked: set[str] = set()
    for ins in win:
        if classify_semop(ins.mnemonic) not in _MEM_SEMOPS:
            continue
        aregs = _addr_regs(ins.mnemonic)
        if not aregs:
            continue
        unb = _step_unbacked(resolve, ins)
        legs.append(AddressLeg(pc=ins.pc, addr_regs=aregs,
                               unbacked=tuple(sorted(unb))))
        all_unbacked |= unb

    return AddressClosureReport(
        window_pcs=(lo, hi),
        legs=tuple(legs),
        unbacked_roots=tuple(sorted(all_unbacked)),
        sufficient=not all_unbacked,
    )


# ---------------------------------------------------------------------------
# Dual-mode switch — forward symbolic vs backward alias.
# ---------------------------------------------------------------------------


