"""setup_symex.entry_state section (split from the monolithic module)."""
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
from ._config import SetupSymexConfig, _require_enabled


@dataclass(frozen=True, slots=True)
class ConcreteBacking:
    """Concrete base-register values + the memory they point at.

    The contract-2 *backing arm*. Symbolizing the input is not enough when the
    window's loads/stores compute their effective address from base registers
    (x20/x24/x25 in the cipher-body case) whose concrete values — and the memory
    those values point at — were never injected. The indirect access then has no
    EA to resolve, degrades to a blind leg, and forward symex emits an
    input-passthrough stub (parity 0/N). This carries the backing the address
    *closure* needs (see :func:`audit_address_closure`), injected from the SAME
    execution as the trace (determinism — never merge cross-run captures).

    ``reg_values`` are concrete (name, value) pairs from the hook snapshot.
    ``mem`` are the pointed regions as canonical :class:`MemSnapshot`s (read-only
    observations; the engine never parses runner formats — an adapter fills
    these). The runner-specific JSON parsing stays in case config.

    ``exec_id`` tags WHICH execution the snapshot was captured in (any stable
    token — typically the run's ExecIdentity ref). The backing audits only count
    this backing when the caller's ``trace_exec_id`` matches it (the determinism
    guard): a snapshot from a different run must NOT mask a real blind leg. Left
    empty it is "unscoped" — counted whenever no ``trace_exec_id`` is asserted."""

    reg_values: tuple[tuple[str, int], ...] = ()
    mem:        tuple[MemSnapshot, ...] = ()
    exec_id:    str = ""

    @property
    def backed_regs(self) -> frozenset[str]:
        return frozenset(name for name, _ in self.reg_values)

    @property
    def backed_addrs(self) -> frozenset[int]:
        addrs: set[int] = set()
        for snap in self.mem:
            for off in range(len(snap.data)):
                addrs.add(snap.addr + off)
        return frozenset(addrs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reg_values": {name: f"0x{val:x}" for name, val in self.reg_values},
            "mem": [
                {"addr": s.addr, "addr_hex": f"0x{s.addr:x}",
                 "length": len(s.data), "label": s.label}
                for s in self.mem
            ],
            "exec_id": self.exec_id,
            "kind": "setup_symex_concrete_backing",
        }


def build_concrete_backing(
    *,
    reg_values: Mapping[str, int] | None = None,
    mem: Iterable[MemSnapshot] = (),
    exec_id: str = "",
) -> ConcreteBacking:
    """Build a :class:`ConcreteBacking` from a hook snapshot's concrete values.

    ``reg_values`` is the concrete register map (base/index registers the window's
    address computations depend on); ``mem`` are the regions those registers point
    at. The values must come from the same execution as the trace (determinism);
    pass ``exec_id`` (the run's identity token) so the backing audits can enforce
    that — a backing whose ``exec_id`` differs from the trace's is not counted."""
    pairs = tuple(sorted((str(k), int(v)) for k, v in (reg_values or {}).items()))
    return ConcreteBacking(reg_values=pairs, mem=tuple(mem), exec_id=str(exec_id))


class ConcreteBackingConflict(ValueError):
    """A register is asked to be BOTH symbolized and concretely pinned."""


@dataclass(frozen=True, slots=True)
class EntryStateSpec:
    """The seed-state the runner must symbolize at the entry anchor.

    Completeness is the contract: the input is frequently already sitting in a
    register when control reaches the window, so symbolizing one assumed buffer
    address misses it. Symbolize the WHOLE reg_file plus every buffer those
    registers point at, and keep the symbols alive along the kept chain (no
    concrete overwrite — see contract 3)."""

    entry_pc:         int
    symbolic_regs:    tuple[str, ...]       # registers to symbolize at entry
    pointed_buffers:  tuple[tuple[int, int], ...]  # (base, length) buffers to symbolize
    note:             str = ""
    # The contract-2 backing arm (see ConcreteBacking): concrete base-register
    # values + the memory they point at, injected from a same-execution hook
    # snapshot so the window's indirect loads/stores resolve a real EA instead of
    # degrading to a blind leg. ``None`` = symbolize-only (the original behaviour).
    concrete_backing: "ConcreteBacking | None" = None
    # The memory arm of the input: external memory regions to symbolize (an input
    # that enters through ``ldr``, not in a register). Each is (addr, size,
    # concrete_shadow) — symbolized so the chain starts, with the trace's observed
    # value as the variable's shadow so it still evaluates along the path.
    symbolic_mem:     tuple[tuple[int, int, int], ...] = ()
    # P2(i) forwarding feed: the (addr, size) staging interval(s) a symbolic store
    # lands in, so the runner's ``_symbolic_staging`` set forwards the symbol to a
    # later load that reads the same bytes. EMPTY by default → the runner's
    # ``_symbolic_staging`` is unchanged (invariant 7: byte-for-byte). drive injects
    # these from ``diagnose_opaque_staging`` (verdict == symbolic_address) backbone
    # intervals (+ optional pointer-chain resolution) in the clo_deferred branch,
    # BEFORE the symex call, so the symbolic-address window forwards on the FIRST run
    # instead of collapsing to an opaque frontier. The field name aligns with the
    # key the runner's ``reset`` already reads (``entry.get("symbolic_staging")``).
    symbolic_staging: tuple[tuple[int, int], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_pc":        self.entry_pc,
            "entry_pc_hex":    f"0x{self.entry_pc:x}",
            "symbolic_regs":   list(self.symbolic_regs),
            "pointed_buffers": [
                {"base": b, "base_hex": f"0x{b:x}", "length": n}
                for (b, n) in self.pointed_buffers
            ],
            "symbolic_mem": [
                {"addr": a, "addr_hex": f"0x{a:x}", "size": s, "value": v}
                for (a, s, v) in self.symbolic_mem
            ],
            "concrete_backing": (
                self.concrete_backing.to_dict() if self.concrete_backing else None
            ),
            # Backing flow internalized (no agent-side decoder/hook injection): the
            # concrete base-register values AND the bytes the pointed regions hold,
            # handed to the runner so it upfront-seeds Triton. ``concrete_regs`` are
            # the pinned pointer bases (stay concrete, not symbolized); ``concrete_mem``
            # carries the actual region bytes (``data_hex``) the runner injects via
            # setConcreteMemoryAreaValue — the memory half of the concrete shadow.
            "concrete_regs": (
                {name: val for name, val in self.concrete_backing.reg_values}
                if self.concrete_backing else {}
            ),
            "concrete_mem": (
                [{"addr": s.addr, "addr_hex": f"0x{s.addr:x}",
                  "size": len(s.data), "data_hex": s.data.hex()}
                 for s in self.concrete_backing.mem]
                if self.concrete_backing else []
            ),
            # P2(i): the staging intervals the runner seeds into _symbolic_staging.
            # Empty by default (invariant 7). Key name aligns with reset()'s reader.
            "symbolic_staging": [[a, s] for (a, s) in self.symbolic_staging],
            "note":            self.note,
            "kind":            "setup_symex_entry_state",
        }


class IncompleteEntryState(ValueError):
    """The proposed seed state covers only an assumed address, not the reg_file."""


def seed_entry_state(
    *,
    entry_pc: int,
    reg_file: Sequence[str],
    pointed_buffers: Iterable[tuple[int, int]] = (),
    symbolic_regs: Sequence[str] | None = None,
    concrete_backing: ConcreteBacking | None = None,
    symbolic_mem: Iterable[tuple[int, int, int]] = (),
    cfg: SetupSymexConfig | None = None,
) -> EntryStateSpec:
    """Build a COMPLETE entry-state seed spec.

    ``reg_file`` is the full register set available at the entry anchor. By
    default every register is symbolized (the safe, complete default). Passing
    ``symbolic_regs`` to symbolize a subset is allowed but the subset must be a
    real subset of ``reg_file`` — symbolizing a register the entry state does
    not even have is the "assumed address" error wearing a register hat, and is
    rejected. ``pointed_buffers`` are the (base,length) regions those registers
    point at, which must ALSO be symbolized (the input may live behind a
    pointer).

    ``concrete_backing`` is the *backing arm* (contract 2): concrete values for
    the base/index registers — and the memory they point at — that the window's
    indirect loads/stores need so their effective address resolves instead of
    going blind. A register cannot be both symbolized AND concretely pinned
    (that is contradictory — you symbolize the input, you pin the pointers), so
    the two sets must be disjoint. Pair with :func:`audit_address_closure`, which
    consumes this backing to confirm every address leg in the window is covered."""
    cfg = cfg or SetupSymexConfig.from_env()
    _require_enabled(cfg)
    reg_set = tuple(dict.fromkeys(reg_file))  # de-dup, preserve order
    if not reg_set:
        raise IncompleteEntryState(
            "entry state needs the full reg_file — an empty reg_file means the "
            "seed is being bound to an assumed address, the contract-2 failure"
        )
    if symbolic_regs is None:
        chosen = reg_set
    else:
        chosen = tuple(dict.fromkeys(symbolic_regs))
        unknown = [r for r in chosen if r not in reg_set]
        if unknown:
            raise IncompleteEntryState(
                f"symbolic_regs {unknown} are not in the entry reg_file "
                f"{list(reg_set)} — symbolize registers the entry state actually "
                f"has, not assumed ones"
            )
    if concrete_backing is not None:
        clash = sorted(set(chosen) & concrete_backing.backed_regs)
        if clash:
            raise ConcreteBackingConflict(
                f"registers {clash} are both symbolized and concretely pinned — "
                f"a register is either the symbolic input or a pinned pointer, "
                f"not both. Symbolize the input regs; back the address/base regs."
            )
    note = (
        "symbolize the full reg_file + pointed buffers; keep symbols alive "
        "along the kept chain (no concrete overwrite — contract 3)"
    )
    if concrete_backing is not None:
        note += (
            f"; concrete backing injected for {sorted(concrete_backing.backed_regs)} "
            f"+ {len(concrete_backing.mem)} pointed region(s) so the address "
            f"closure resolves (audit with audit_address_closure)"
        )
    return EntryStateSpec(
        entry_pc=entry_pc,
        symbolic_regs=chosen,
        pointed_buffers=tuple((int(b), int(n)) for (b, n) in pointed_buffers),
        note=note,
        concrete_backing=concrete_backing,
        symbolic_mem=tuple((int(a), int(s), int(v)) for (a, s, v) in symbolic_mem),
    )


def derive_window_symbolic_regs(
    items: Sequence[Instruction],
    *,
    window: tuple[int, int],
    reg_file: Sequence[str] | None = None,
    window_is_idx: bool = False,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Auto-derive a window's symbolic input registers = its *live-in* set.

    A handler/window's symbolic inputs are exactly the registers it READS with
    no producer INSIDE the window — the seed plus the threaded state carrier from
    the previous handler. That is a mechanical data-flow query (the regs whose
    DFG producer is ``None`` / external), so utov derives it instead of asking
    the agent to hand-config ``symbolic_regs`` per handler. The hand-config path
    is the run-once-look-once trap: a handler the agent forgets to fill silently
    seeds nothing → ``sym_regs_n=0`` and the symbolic input never propagates
    (the VMP-cipher F0 handler11 failure).

    Returns ``(live_in, info)``. ``info`` carries provenance for surfacing: the
    raw live-in, any regs DROPPED because they are not in ``reg_file`` (a real
    config gap — surfaced, never silently swallowed), and whether the set is
    empty (a degenerate window the caller must NOT treat as "seeded"). Pure
    dataflow, target-agnostic; the parity gate still backstops a wrong derive."""
    from ..stages.s3_triton import build_dfg as _build_dfg  # local: heavy import

    lo, hi = int(window[0]), int(window[1])
    if window_is_idx:
        win = [ins for ins in items if lo <= ins.idx <= hi]
    else:
        win = [ins for ins in items if lo <= ins.pc <= hi]
    live_in: list[str] = []
    seen: set[str] = set()
    nodes = _build_dfg(win)
    n_inferred = 0
    for n in nodes:
        for r, producer in n.reg_deps.items():
            if producer is None and r not in seen:
                seen.add(r)
                live_in.append(r)
        # ② count inferred-interval edges: a reg-dep recovered from regs_read value
        # changes when regs_write recorded no writer (low-confidence, coarse).
        n_inferred += len(getattr(n, "reg_deps_inferred", {}) or {})
    dropped: list[str] = []
    if reg_file is not None:
        rf = set(reg_file)
        dropped = [r for r in live_in if r not in rf]
        live_in = [r for r in live_in if r in rf]
    # ③/④ trust-gate: assess this window's regs_write coverage via the unified
    # profile. The live-in derivation keys off regs_write producers (None ==
    # external); when regs_write is largely empty the "external" set is a
    # MEASUREMENT artefact, not ground truth — and any dependency the chain rests
    # on is item-② inferred (low confidence). Surface this in info so the caller's
    # downstream conclusion can be marked inconclusive / low-confidence rather than
    # trusting a possibly-blind live-in set. Additive (does not change live_in).
    from ..trace_observability import assess_trace_observability as _assess
    _obs = _assess(win, window=None)
    regs_write_sufficient = _obs.overall_sufficient_for("regs_write") if win else False
    info: dict[str, Any] = {
        "kind":                    "setup_symex_auto_seed",
        "window":                  [lo, hi],
        "window_basis":            "idx" if window_is_idx else "pc",
        "live_in":                 list(live_in),
        "dropped_not_in_reg_file": dropped,
        "empty":                   not live_in,
        "n_window_items":          len(win),
        # ④ readiness: regs_write coverage of this window + whether the DFG rests
        # on inferred (low-confidence) edges. A consumer drawing a symbolic-reg
        # conclusion should mark it inconclusive when not regs_write_sufficient.
        "regs_write_rate":         round(_obs.regs_write_rate, 4),
        "regs_write_sufficient":   regs_write_sufficient,
        "n_inferred_edges":        n_inferred,
        "readiness_note": (
            "" if regs_write_sufficient else
            f"regs_write coverage low ({_obs.regs_write_rate:.2%}); the live-in set "
            f"keys off regs_write producers that are largely invisible here and the "
            f"DFG carries {n_inferred} inferred (low-confidence) edge(s) — treat any "
            f"symbolic-reg conclusion as INCONCLUSIVE / low-confidence (merge "
            f"regs_write-populated data or diff regs_read)"),
    }
    return tuple(live_in), info


@dataclass(frozen=True, slots=True)
class MemLiveIn:
    """An external MEMORY input of a window: a load whose bytes have no writer
    INSIDE the window (the carrier byte / table entry the previous handler or the
    seed put there). Register live-in only pins register inputs; a value that
    enters through ``ldr`` is invisible to it — left un-symbolized, the symbolic
    chain never starts and the window's exit collapses to a concrete 0 (the F0
    handler11 ``symbolic=0`` failure). This is the memory arm of the live-in."""

    addr:      int                  # effective address (from the trace)
    size:      int                  # bytes loaded externally at this address
    src_idx:   int                  # the loading instruction's trace idx
    base_regs: tuple[str, ...] = ()  # the load's address registers (re-pin context)

    def to_dict(self) -> dict[str, Any]:
        return {"addr": f"0x{self.addr:x}", "size": self.size,
                "src_idx": self.src_idx, "base_regs": list(self.base_regs)}


def derive_window_mem_live_in(
    items: Sequence[Instruction],
    *,
    window: tuple[int, int],
    window_is_idx: bool = False,
) -> tuple[tuple[MemLiveIn, ...], dict[str, Any]]:
    """Auto-derive a window's external MEMORY inputs from byte-granular mem deps.

    A loaded byte with no in-window writer is an external input (its producer is
    the seed / a previous handler, outside the window). Pure dataflow, target-
    agnostic; mirrors :func:`derive_window_symbolic_regs` for the memory leg.
    Each entry is marked **symbolize-or-back**: a real input is symbolized, a
    pointer/table base is pinned via ``concrete_backing`` — and *which* is the
    agent's judgment (a checkpoint), surfaced by the caller, never auto-guessed."""
    lo, hi = int(window[0]), int(window[1])
    if window_is_idx:
        win = [ins for ins in items if lo <= ins.idx <= hi]
    else:
        win = [ins for ins in items if lo <= ins.pc <= hi]
    last_mem_writer: dict[int, int] = {}
    mem_live_in: list[MemLiveIn] = []
    for ins in win:
        base_regs = tuple(sorted(ins.regs_read.keys()))   # the load's address regs
        for op in ins.mem:
            if op.rw == "r":
                # Collect the contiguous byte runs with no in-window writer.
                run_start: int | None = None
                for b in range(op.addr, op.addr + op.size):
                    external = b not in last_mem_writer
                    if external and run_start is None:
                        run_start = b
                    elif not external and run_start is not None:
                        mem_live_in.append(
                            MemLiveIn(run_start, b - run_start, ins.idx, base_regs))
                        run_start = None
                if run_start is not None:
                    mem_live_in.append(MemLiveIn(
                        run_start, op.addr + op.size - run_start, ins.idx, base_regs))
        for op in ins.mem:
            if op.rw == "w":
                for b in range(op.addr, op.addr + op.size):
                    last_mem_writer[b] = ins.idx
    info: dict[str, Any] = {
        "kind":          "setup_symex_mem_live_in",
        "window":        [lo, hi],
        "window_basis":  "idx" if window_is_idx else "pc",
        "mem_live_in":   [m.to_dict() for m in mem_live_in],
        "empty":         not mem_live_in,
        "n_window_items": len(win),
    }
    return tuple(mem_live_in), info


# ---------------------------------------------------------------------------
# Contract 3 — symbol-preserving hybrid execution.
# ---------------------------------------------------------------------------


