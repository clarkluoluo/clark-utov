"""The Level-2 runner, decoder seam, and Triton bulk decoder."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Protocol, runtime_checkable

from ..types import Instruction
from ._audit import SYMBOLIC_FORWARD_SITE_CAP
from ._base import (opcode_hex, is_control_flow, triton_available,
                    triton_unavailable_reason)
from ._semantics import (InstructionSemantics, SemanticsTable, UnmodeledInstruction,
                         SemanticsParseError, SemanticsApplyError, parse_sexpr,
                         validate_sexpr, SEMANTICS_BINOPS, SEMANTICS_UNOPS)


@runtime_checkable
class StepDecoder(Protocol):
    """A per-instruction symbolic stepper. The bulk default is Triton; tests
    inject a deterministic fake so the runner framework is covered without Triton.

    Contract: ``step`` returns True iff the instruction was modeled symbolically.
    It MUST NOT force-concretize a symbolic value to "get past" an opcode it can't
    model — it returns False and lets the runner raise the escape-hatch checkpoint.
    """

    def reset(self, entry: Mapping[str, Any]) -> None:
        """Symbolize the entry state (reg_file + pointed buffers) for a fresh run."""

    def step(self, ins: Instruction) -> bool:
        """Symbolically execute one instruction. True=modeled, False=unmodeled."""

    def apply_semantics(self, ins: Instruction, sem: InstructionSemantics) -> None:
        """Apply a hand-filled semantics entry for an instruction ``step`` rejected."""

    def expression(self) -> str:
        """The recovered symbolic transform (sink expression) after the window."""
@dataclass(frozen=True, slots=True)
class RunnerResult:
    """What :func:`run_window` produced: either an expression or a block."""

    expr_source: str
    unmodeled:   UnmodeledInstruction | None
    steps:       int
    modeled:     int
    escape_hatch_hits: int           # steps modeled via the semantics table
    branches_skipped:  int = 0       # control-flow steps taken from the trace, not Triton

    @property
    def blocked(self) -> bool:
        return self.unmodeled is not None
def run_window(
    decoder: StepDecoder,
    table: SemanticsTable,
    items: Iterable[Instruction],
    *,
    window: tuple[int, int],
    entry: Mapping[str, Any],
    window_kind: str = "pc",
) -> RunnerResult:
    """Drive the decoder over the trace segment, applying the escape hatch.

    Trace-guided: iterate ``items`` in recorded (executed) order, processing the
    steps inside the segment — this follows the concrete control flow the trace
    recorded. ``window_kind`` selects how the segment is bounded:

    - ``"pc"`` (default): an inclusive ``(pc_lo, pc_hi)`` band;
    - ``"idx"``: an inclusive trace-index range ``(idx_lo, idx_hi)`` — the
      execution-order segment. Prefer this when a pc recurs across handler
      invocations / branch sides, so a pc-band can't pull in the wrong occurrence
      (the b.hi side-path trap).

    Control-flow instructions (branches/calls/returns) are NOT handed to Triton —
    the recorded order is the taken path, so letting Triton evaluate a possibly-
    symbolic branch would diverge. Each non-branch step is then (a) modeled by the
    bulk decoder, (b) modeled via a hand-filled semantics-table entry, or (c)
    **un-modeled** → return immediately with an :class:`UnmodeledInstruction` and
    an empty expression. It NEVER force-concretizes or skips an un-modeled step."""
    lo, hi = int(window[0]), int(window[1])
    by_idx = window_kind == "idx"
    decoder.reset(entry)
    steps = modeled = hatch = branches = 0
    for ins in items:
        key = ins.idx if by_idx else ins.pc
        if not (lo <= key <= hi):
            continue
        if is_control_flow(ins.mnemonic):
            # Trace-guided: control flow comes from the recorded order, not Triton.
            branches += 1
            continue
        steps += 1
        if decoder.step(ins):
            modeled += 1
            continue
        sem = table.lookup(opcode_hex(ins), ins.mnemonic)
        if sem is not None:
            decoder.apply_semantics(ins, sem)
            modeled += 1
            hatch += 1
            continue
        # The whole point: an un-modeled step is a checkpoint, not a guess.
        return RunnerResult(
            expr_source="",
            unmodeled=UnmodeledInstruction(
                opcode_hex=opcode_hex(ins), mnemonic=ins.mnemonic,
                idx=ins.idx, pc=ins.pc),
            steps=steps, modeled=modeled, escape_hatch_hits=hatch,
            branches_skipped=branches)
    return RunnerResult(expr_source=decoder.expression(), unmodeled=None,
                        steps=steps, modeled=modeled, escape_hatch_hits=hatch,
                        branches_skipped=branches)
# ---------------------------------------------------------------------------
# The runner — conforms to the drive `triton_runner` protocol
# ---------------------------------------------------------------------------

# A gold callable computes parity for a recovered expression against the live
# oracle. Target-specific (needs the runner cmd / gold corpus), so the agent
# supplies it; the framework never fabricates a parity number. Returns the keys
# drive reads: {"gold_parity": "m/n", "parity_vectors": [...]}.
GoldFn = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]


def build_level2_runner(
    *,
    table: SemanticsTable | None = None,
    decoder: StepDecoder | None = None,
    gold: GoldFn | None = None,
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Build a Level-2 ``triton_runner`` for :func:`setup_symex.drive`.

    ``decoder`` defaults to Triton (:class:`TritonStepDecoder`) — raising at call
    time if Triton is unavailable (honest, not a silent concretize). ``table`` is
    the persistent escape-hatch cache (a fresh in-memory one by default). ``gold``
    computes parity from the recovered expression; without it the runner returns
    the expression with ``propagated`` set but no parity (drive then won't close —
    parity is a hard gate)."""
    tbl = table if table is not None else SemanticsTable()

    def _runner(ctx: Mapping[str, Any]) -> dict[str, Any]:
        dec = decoder if decoder is not None else TritonStepDecoder()
        # Issue 7 — per-window EXPLICIT mem-sink descriptor. drive / the recovery
        # verifier forwards the sink descriptor on the ctx; apply it to a decoder
        # that supports the explicit mem-sink mode so expression() reads the sink
        # bytes (not x8). Absent → the decoder's own output_mem stands (None on the
        # register path — the regression guard); no ctx key never touches reg flow.
        if ctx.get("output_mem") is not None and hasattr(dec, "output_mem"):
            om = ctx["output_mem"]
            try:
                dec.output_mem = {
                    "sink_addr": int(om["sink_addr"]),
                    "sink_size": int(om["sink_size"]),
                    "sink_idx": (int(om["sink_idx"])
                                 if om.get("sink_idx") is not None else None)}
            except (KeyError, TypeError, ValueError):
                pass                              # malformed → leave decoder as-is
        try:
            res = run_window(
                dec, tbl, list(ctx.get("items", [])),
                window=tuple(ctx["window"]), entry=ctx.get("entry", {}),
                window_kind=str(ctx.get("window_kind", "pc")))
        except (SemanticsApplyError, SemanticsParseError) as e:
            # A hand-filled semantics entry was malformed / un-injectable — surface
            # it precisely (the agent fixes that one fill), do not emit.
            return {"propagated": False, "expr_source": "",
                    "semantics_error": str(e)}
        # Seed visibility: an incomplete seed (only some symbolic_regs took) is the
        # "exit all concrete / x8 == 0" trap — surface it, never silent.
        seed_info = {
            "seeded_regs": list(getattr(dec, "seeded_regs", ()) or ()),
            "unseeded_regs": list(getattr(dec, "unseeded_regs", ()) or ()),
            "seeded_mem": [[a, s] for (a, s) in getattr(dec, "seeded_mem", ()) or ()],
            "shadowed_mem_reads": int(getattr(dec, "shadowed_mem_reads", 0) or 0),
            "shadowed_reg_writes": int(getattr(dec, "shadowed_reg_writes", 0) or 0),
            "unshadowed_steps": int(getattr(dec, "unshadowed_steps", 0) or 0),
            "symbolic_forwards": int(getattr(dec, "symbolic_forwards", 0) or 0),
            "symbolic_forward_sites": [
                [int(pc), int(a), int(s)]
                for (pc, a, s) in (getattr(dec, "symbolic_forward_sites", ()) or ())
            ],
            "branches_skipped": res.branches_skipped,
        }
        # Issue 7 — mem-sink readability: when the decoder ran in EXPLICIT mem-sink
        # mode and could not read the sink bytes back symbolically (EA never
        # symbolic / read failed / input-invariant store), surface the structured
        # reason so the recovery layer raises MEM_SINK_UNPLACEABLE / routes the
        # input-invariant store to the seed-independence exclusion — never a silent
        # register/constant fallback. None on the register path (regression guard).
        mem_unreadable = getattr(dec, "mem_sink_unreadable", None)
        if mem_unreadable:
            seed_info["mem_sink_unreadable"] = str(mem_unreadable)
        if res.blocked:
            # Escape hatch: surface the precise checkpoint; emit nothing.
            return {
                "propagated": False,
                "expr_source": "",
                "unmodeled": res.unmodeled.to_dict(),
                "symex_steps": res.steps,
                "symex_modeled": res.modeled,
                **seed_info,
            }
        out: dict[str, Any] = {
            "propagated": True,
            "expr_source": res.expr_source,
            "symex_steps": res.steps,
            "symex_modeled": res.modeled,
            "escape_hatch_hits": res.escape_hatch_hits,
            **seed_info,
        }
        if gold is not None:
            out.update(dict(gold(res.expr_source, ctx)))
        return out

    return _runner


# ---------------------------------------------------------------------------
# The default bulk decoder — Triton (behind availability)
# ---------------------------------------------------------------------------


class TritonStepDecoder:
    """Default bulk decoder: Triton AArch64, symbolic memory kept symbolic.

    The hand-rolled L1 ``_sym_emulate_mem`` force-concretized symbolic loads (the
    chain-snapping root flaw). Here Triton owns *all* instruction + memory
    semantics with ``ALIGNED_MEMORY`` symbolic memory, so a load off a symbolic
    address stays symbolic. An opcode Triton can't process makes :meth:`step`
    return False (→ escape hatch), never a concretized guess.

    ``output_reg`` is the sink register whose symbolic value is the recovered
    transform; if unset, :meth:`expression` falls back to the last register the
    window wrote. Triton is required — constructing this without it raises.

    ``output_mem`` is the EXPLICIT mem-sink mode (Issue 7): a descriptor
    ``{"sink_addr": int, "sink_size": int}`` (optionally ``"sink_idx"``) naming
    the memory interval whose bytes are the recovered output. When set,
    :meth:`expression` reads the SYMBOLIC value of ``[sink_addr, sink_addr+
    sink_size)`` after the window (a Triton symbolic-memory read over those bytes)
    and emits a byte-list expression — NOT a register value, and NOT the implicit
    ``_last_write`` guess. ``output_reg`` stays the default; mem is purely
    additive. A descriptor that cannot be read (the EA's bytes never became
    symbolic / Triton could not read them) → :meth:`expression` returns the empty
    string, and :attr:`mem_sink_unreadable` records why so the recovery layer can
    surface the structured MEM_SINK_UNPLACEABLE terminal rather than silently fall
    back to a register / constant."""

    def __init__(
        self, *, output_reg: str | None = None,
        output_mem: "Mapping[str, Any] | None" = None,
    ) -> None:
        if not triton_available():
            raise RuntimeError(
                f"TritonStepDecoder needs Triton: {triton_unavailable_reason()}. "
                f"Install triton, or pass a different StepDecoder. The engine does "
                f"NOT fall back to a hand-rolled symbolic emulator (that was the L1 "
                f"force-concretize flaw this module replaces).")
        from triton import ARCH, MODE, TritonContext  # type: ignore
        self._ARCH, self._MODE, self._TritonContext = ARCH, MODE, TritonContext
        self.output_reg = output_reg
        # Issue 7 — EXPLICIT mem-sink mode. None → register-only behaviour, byte-
        # for-byte the pre-Issue-7 path (the regression guard). A descriptor pins
        # the memory interval expression() reads after the window.
        self.output_mem: dict[str, Any] | None = (
            {"sink_addr": int(output_mem["sink_addr"]),
             "sink_size": int(output_mem["sink_size"]),
             "sink_idx": (int(output_mem["sink_idx"])
                          if output_mem.get("sink_idx") is not None else None)}
            if output_mem is not None else None)
        # Set by expression() when a mem-sink descriptor's bytes could NOT be read
        # back symbolically (EA never symbolic / Triton read failed) — a structured
        # reason string the recovery layer turns into MEM_SINK_UNPLACEABLE. None
        # while unread / on the register path (invariant: reg path is untouched).
        self.mem_sink_unreadable: str | None = None
        self._ctx = None
        self._last_write: str | None = None
        self.seeded_regs: tuple[str, ...] = ()
        self.unseeded_regs: tuple[str, ...] = ()
        self.seeded_mem: tuple[tuple[int, int], ...] = ()
        # Concolic memory shadow: how many non-symbolized memory reads got their
        # trace ground-truth value injected before processing (see :meth:`step`).
        self.shadowed_mem_reads: int = 0
        # Register reconciliation: how many non-symbolized written registers got
        # their trace ground-truth value injected AFTER processing — the register-
        # trace path that does not depend on a populated mem[] (see :meth:`step`).
        self.shadowed_reg_writes: int = 0
        # Observability: load steps for which the trace gave NO ground-truth source
        # at all (neither a mem[] read value NOR a regs_write entry). Such a load ran
        # on Triton's uninitialised 0 with nothing to correct it — surfaced, never
        # silent (the F0 false-green was exactly a blind load that looked fine).
        self.unshadowed_steps: int = 0
        # Symbolic staging intervals (opaque-staging Phase 1/2(i)): byte ranges a
        # SYMBOLIC store landed in, so a later load whose EA hits the interval is
        # left symbolic (forwarded) instead of overwritten with the trace's concrete
        # value. Seeded in reset() from the symbolic_mem leg + the pointer-chain-
        # resolved landings (entry["symbolic_staging"]); GROWN in step() each time a
        # symbolic store is processed. EMPTY → _shadow_concrete_reads is byte-for-byte
        # the original behaviour (invariant 7: the green baseline does not move).
        self._symbolic_staging: set[tuple[int, int]] = set()
        # Opaque-staging Phase 2(i) "+ record-a-line": forwarding is the third
        # destination a non-symbolized load can take (the other two are
        # shadowed_mem_reads = concrete value injected, and unshadowed_steps = ran
        # blind). When a read hits a SYMBOLIC staging interval _shadow_concrete_reads
        # leaves it symbolic (forwards the staged symbol) — count it here and sample
        # the site so "injected N intervals, forwarded M loads" is directly visible.
        # EMPTY _symbolic_staging → _intersects_symbolic_staging is always False →
        # this never increments (invariant 7: stays 0/empty, byte-for-byte baseline).
        # Purely observational: never read by any close/parity/G4/seed gate.
        self.symbolic_forwards: int = 0
        self.symbolic_forward_sites: list[tuple[int, int, int]] = []

    def reset(self, entry: Mapping[str, Any]) -> None:
        ctx = self._TritonContext()
        ctx.setArchitecture(self._ARCH.AARCH64)
        ctx.setMode(self._MODE.ALIGNED_MEMORY, True)
        # Symbolize the WHOLE requested register set so the transform is a function
        # of every input (seeding only one reg → the exit is concrete, x8 == 0 — the
        # C2 incompleteness trap). Track what failed instead of silently dropping it.
        requested = [str(r) for r in
                     (entry.get("symbolic_regs") or entry.get("reg_file") or ())]
        concrete = {str(k): int(v) for k, v in (entry.get("concrete_regs") or {}).items()}
        seeded: list[str] = []
        unseeded: list[str] = []
        symvars: dict[str, Any] = {}
        for name in requested:
            try:
                symvars[name] = ctx.symbolizeRegister(ctx.getRegister(name))
                seeded.append(name)
            except Exception:
                unseeded.append(name)               # surfaced, not swallowed
        # Concrete shadow goes on the VARIABLE (the register stays symbolic): the
        # transform remains a function of the inputs AND evaluates along the trace's
        # recorded concrete path (trace-guided concolic).
        for name, val in concrete.items():
            try:
                reg = ctx.getRegister(name)
                masked = val & ((1 << reg.getBitSize()) - 1)
                if name in symvars:
                    ctx.setConcreteVariableValue(symvars[name], masked)
                else:
                    ctx.setConcreteRegisterValue(reg, masked)
            except Exception:
                continue
        # Upfront concrete memory backing (contract-2 backing arm, internalized):
        # the bytes a pointed region holds, for memory a base register addresses
        # but that no trace MemOp carries a value for (e.g. a reg-trace runner that
        # emits empty mem[]). Seed it concrete so processing reads real bytes, not
        # Triton's uninitialised 0. Done BEFORE the symbolic_mem arm so an external
        # input symbolized there wins on any overlap (symbolic over concrete).
        for cm in entry.get("concrete_mem", ()) or ():
            try:
                addr = int(cm["addr"])
                data = bytes.fromhex(str(cm["data_hex"]))
                if data:
                    ctx.setConcreteMemoryAreaValue(addr, data)
            except Exception:
                continue
        # Memory arm: symbolize external memory inputs (a value that enters through
        # ldr — register seed can't reach it). Without this the symbolic chain never
        # starts at that load and the exit is concrete 0 (the handler11 trap). The
        # variable carries the trace's observed value as its shadow.
        self.seeded_mem: tuple[tuple[int, int], ...] = ()
        seeded_mem: list[tuple[int, int]] = []
        for entry_mem in entry.get("symbolic_mem", ()) or ():
            try:
                addr = int(entry_mem["addr"])
                size = int(entry_mem["size"])
                val = int(entry_mem.get("value", 0))
                from triton import MemoryAccess  # type: ignore
                sv = ctx.symbolizeMemory(MemoryAccess(addr, size))
                ctx.setConcreteVariableValue(sv, val & ((1 << (8 * size)) - 1))
                seeded_mem.append((addr, size))
            except Exception:
                continue
        self.seeded_mem = tuple(seeded_mem)
        # Symbolic staging intervals (opaque-staging Phase 1/2(i)). Initial set:
        #  - every symbolized memory leg above (a symbolic store target / external
        #    input region) is itself a symbolic interval its later loads forward;
        #  - the pointer-chain-resolved staging landings the caller passes in via
        #    entry["symbolic_staging"] (a list of (addr,size) the chain stores into).
        # The set GROWS at runtime in step() as symbolic stores land. Empty when the
        # window does not stage a symbol → _shadow_concrete_reads is unchanged.
        staging: set[tuple[int, int]] = set(seeded_mem)
        for iv in entry.get("symbolic_staging", ()) or ():
            try:
                addr, size = int(iv[0]), int(iv[1])
                if size > 0:
                    staging.add((addr, size))
            except (TypeError, ValueError, IndexError):
                continue
        self._symbolic_staging = staging
        self._ctx = ctx
        self._last_write = None
        self.mem_sink_unreadable = None
        self.seeded_regs = tuple(seeded)
        self.unseeded_regs = tuple(unseeded)
        self.shadowed_mem_reads = 0
        self.shadowed_reg_writes = 0
        self.unshadowed_steps = 0
        self.symbolic_forwards = 0
        self.symbolic_forward_sites = []

    def _shadow_concrete_reads(self, ins: Instruction) -> None:
        """Concolic memory shadow: give each NON-symbolized memory read its trace
        ground-truth value before the instruction is processed.

        Register state has a concrete shadow on its symbolic variables; memory did
        not — so a load whose bytes were never seeded (intra-handler state, a
        constant table, upstream thread state) read Triton's uninitialised 0, and a
        downstream ``mul``/``eor`` collapsed the whole transform to 0 (the F0
        type10 ``emitted_F="0"`` with parity 0/N). Here every memory READ that is
        NOT symbolized gets the trace's recorded value (``MemOp.val``); input-
        derived bytes (a window store of a symbolic value) test as symbolized and
        are left untouched, so the recovered F stays a function of the input.

        This also auto-answers the mem class-2/3 question per step: symbolized
        (input-tainted) → stays symbolic; everything else → trace concrete. Only a
        genuine "external input vs thread state" ambiguity needs a checkpoint (the
        A-arm symbolic_mem path), not a per-slot hand judgment."""
        if not ins.mem:
            return
        from triton import MemoryAccess  # type: ignore
        ctx = self._ctx
        for op in ins.mem:
            if op.rw != "r" or op.size <= 0:
                continue
            try:
                ma = MemoryAccess(op.addr, op.size)
                if ctx.isMemorySymbolized(ma):
                    continue                    # input-derived — keep symbolic
                # Opaque-staging Phase 1/2(i): a read that hits a SYMBOLIC staging
                # interval must NOT be overwritten with the trace's concrete value —
                # that is exactly the forwarding break (the symbolic store's bytes
                # would be clobbered, taint dies). Skip the injection so Triton's
                # ALIGNED_MEMORY forwards the symbolic store to this load. Keyed on
                # op.addr (the trace's concrete EA) so a symbolic-addressed load
                # (Phase 2(i): address concolic, value symbolic) forwards the same.
                # Empty _symbolic_staging → this never fires (invariant 7).
                if self._intersects_symbolic_staging(op.addr, op.size):
                    # "+ record-a-line": the forward actually happened HERE — count
                    # it (op granularity, same as shadowed_mem_reads) and sample the
                    # site before continuing. Empty staging → never reached → 0/empty.
                    self.symbolic_forwards += 1
                    if len(self.symbolic_forward_sites) < SYMBOLIC_FORWARD_SITE_CAP:
                        self.symbolic_forward_sites.append(
                            (ins.pc, op.addr, op.size))
                    continue
                ctx.setConcreteMemoryValue(ma, op.val & ((1 << (8 * op.size)) - 1))
                self.shadowed_mem_reads += 1
            except Exception:
                continue                        # a bad MemOp must not abort the step

    def _intersects_symbolic_staging(self, addr: int, size: int) -> bool:
        """Does ``[addr, addr+size)`` overlap any recorded symbolic staging interval?

        Interval (range) intersection, not the single-point ``isMemorySymbolized``
        — a symbolic store of one width forwards to a load of a different/overlapping
        width at the same staging address. Empty set → always False (the baseline)."""
        if not self._symbolic_staging:
            return False
        lo, hi = addr, addr + size            # [lo, hi)
        for (s_addr, s_size) in self._symbolic_staging:
            if lo < s_addr + s_size and s_addr < hi:
                return True
        return False

    def _reconcile_concrete_regs(self, ins: Instruction) -> None:
        """Register reconciliation: after processing, give each NON-symbolized
        WRITTEN register its trace ground-truth value (``regs_write[reg]``).

        The mem-shadow only has a data source when ``ins.mem`` is populated. A
        register-trace (the F0 type10 reality) has a SPARSE ``mem[]`` — the window's
        ``ldr`` steps carry no ``MemOp`` — so ``_shadow_concrete_reads`` runs empty
        (``shadowed_mem_reads==0``) and a load off un-seeded memory still produces
        Triton's 0, collapsing the downstream ``mul`` to 0 (``emitted_F="0"``). But
        the loaded value IS in the trace: ``ldr w9,[…]`` records ``regs_write["w9"]``.
        Reconcile from there — it does NOT depend on ``mem[]``.

        Concolic invariant: the symbolic skeleton is the input-dependent computation;
        everything else follows the trace's concrete values. ``regs_write`` is exactly
        the "everything else" ground-truth. A register Triton kept SYMBOLIC is input-
        tainted (a real input-dependent result) and is LEFT untouched — overwriting it
        would drop the input path. ``isRegisterSymbolized`` guards that; the multi-
        vector parity gate backstops any over-concretization."""
        if not ins.regs_write:
            return
        ctx = self._ctx
        for name, val in ins.regs_write.items():
            try:
                reg = ctx.getRegister(name)
            except Exception:
                continue                        # unknown reg name — skip, don't abort
            try:
                if ctx.isRegisterSymbolized(reg):
                    continue                    # input-derived — keep symbolic
                masked = int(val) & ((1 << reg.getBitSize()) - 1)
                ctx.setConcreteRegisterValue(reg, masked)
                self.shadowed_reg_writes += 1
            except Exception:
                continue                        # a bad reg/value must not abort the step

    def _record_symbolic_stores(self, ins: Instruction) -> None:
        """Grow ``_symbolic_staging`` with any store whose landing bytes are now
        symbolic (opaque-staging Phase 1/2(i)).

        Called AFTER ``processing`` so Triton has applied the store. A write MemOp
        whose ``[addr, addr+size)`` Triton reports symbolized carries an input-
        derived value into memory — that landing is a staging interval a later load
        forwards from. Recording it (vs relying only on single-point
        ``isMemorySymbolized`` at the later load) keeps the forwarding robust to a
        load of a different/overlapping width. No write MemOp (reg-trace store) →
        nothing recorded here; the reset()-seeded intervals still cover it."""
        if not ins.mem:
            return
        from triton import MemoryAccess  # type: ignore
        ctx = self._ctx
        for op in ins.mem:
            if op.rw != "w" or op.size <= 0:
                continue
            try:
                if ctx.isMemorySymbolized(MemoryAccess(op.addr, op.size)):
                    self._symbolic_staging.add((op.addr, op.size))
            except Exception:
                continue                        # a bad MemOp must not abort the step

    def step(self, ins: Instruction) -> bool:
        from triton import Instruction as TritonInstr  # type: ignore
        # Concolic memory shadow BEFORE processing: an un-symbolized load must read
        # the trace's real value, not Triton's uninitialised 0 (chain-collapse fix).
        self._shadow_concrete_reads(ins)
        try:
            t = TritonInstr()
            t.setAddress(ins.pc)
            t.setOpcode(bytes(ins.bytes_))
            self._ctx.processing(t)             # raises on an opcode it can't decode
        except Exception:
            return False                        # → escape hatch (never concretize)
        # Opaque-staging Phase 1/2(i): GROW the symbolic staging set. A store whose
        # written bytes Triton now reports symbolized (the source value derived from
        # a symbolic register) is a SYMBOLIC store — its landing is a staging interval
        # a later load must forward from, so record it. Done before reconciliation so
        # a symbolic store landing is never mistaken for a concretizable slot. Only
        # touches state when the window actually stores a symbol (invariant 7).
        self._record_symbolic_stores(ins)
        # Register reconciliation AFTER processing: a register-trace carries the
        # ground-truth in regs_write even when mem[] is empty, so a load off
        # un-seeded memory gets its real value here (not Triton's 0). Symbolic
        # (input-tainted) writes are preserved — F stays a function of the input.
        self._reconcile_concrete_regs(ins)
        # Observability (no silent blind load): a load is the one step whose value
        # ENTERS from memory; if the trace gives neither a mem[] read value nor a
        # regs_write entry, NEITHER shadow path can correct it and it ran on 0.
        # Count it (zero case-specific knowledge — purely "load with no trace
        # source") so "shadow had no data" is visible, not a silent emit "0".
        if ins.mnemonic.strip().lower().startswith("ld"):
            has_mem_truth = any(op.rw == "r" for op in ins.mem)
            if not has_mem_truth and not ins.regs_write:
                self.unshadowed_steps += 1
        if ins.regs_write:
            self._last_write = sorted(ins.regs_write.keys())[-1]
        return True

    def apply_semantics(self, ins: Instruction, sem: InstructionSemantics) -> None:
        # Hand-filled escape hatch: compile each written register's DSL S-expression
        # into a Triton AST and ASSIGN it to that register's symbolic value — a real
        # injection into the symbolic state, not a concretize. A malformed fill or a
        # width/register error is a precise SemanticsApply/ParseError (caught + the
        # parity gate is still the backstop), never a silent pass.
        ctx = self._ctx
        actx = ctx.getAstContext()
        for dst, text in sem.effects:
            try:
                node = parse_sexpr(text)
                validate_sexpr(node)
            except SemanticsParseError as e:
                raise SemanticsApplyError(
                    f"semantics for {sem.opcode_hex} dst {dst}: {e}") from e
            try:
                built = self._compile(node, ctx, actx)
                se = ctx.newSymbolicExpression(
                    built, f"escape-hatch {sem.opcode_hex} -> {dst}")
                ctx.assignSymbolicExpressionToRegister(se, ctx.getRegister(dst))
            except SemanticsApplyError:
                raise
            except Exception as e:                       # Triton width / reg errors
                raise SemanticsApplyError(
                    f"inject {sem.opcode_hex} -> {dst}: {type(e).__name__}: {e}") from e
            self._last_write = dst

    def _compile(self, node: "int | str | list", ctx: Any, actx: Any) -> Any:
        """Compile a parsed DSL expression into a Triton AST node."""
        if isinstance(node, int):
            # validate_sexpr already rejects bare immediates; defensive.
            raise SemanticsApplyError("bare immediate — wrap as (bv <value> <size>)")
        if isinstance(node, str):
            try:
                return ctx.getRegisterAst(ctx.getRegister(node))
            except Exception as e:
                raise SemanticsApplyError(f"unknown register {node!r}: {e}") from e
        op, args = node[0], node[1:]
        if op == "bv":
            return actx.bv(args[0], args[1])
        if op == "extract":
            return actx.extract(args[0], args[1], self._compile(args[2], ctx, actx))
        if op == "zx":
            return actx.zx(args[0], self._compile(args[1], ctx, actx))
        if op == "sx":
            return actx.sx(args[0], self._compile(args[1], ctx, actx))
        if op == "concat":
            return actx.concat([self._compile(a, ctx, actx) for a in args])
        if op in SEMANTICS_UNOPS:
            a = self._compile(args[0], ctx, actx)
            return actx.bvnot(a) if op == "bvnot" else actx.bvneg(a)
        if op in SEMANTICS_BINOPS:
            a = self._compile(args[0], ctx, actx)
            b = self._compile(args[1], ctx, actx)
            return getattr(actx, op)(a, b)
        raise SemanticsApplyError(f"unsupported op {op!r}")

    def expression(self) -> str:
        if self._ctx is None:
            return ""
        # Issue 7 — EXPLICIT mem-sink mode. After the window, read the SYMBOLIC
        # value of [sink_addr, sink_addr+sink_size) byte by byte (a Triton
        # symbolic-memory read over those bytes) and emit a byte-list expression.
        # The register path below is untouched when output_mem is None (the
        # regression guard). A degenerate read (no byte of the interval ever became
        # symbolic — the store at sink_idx wrote a seed/driver-independent constant,
        # or the EA was never resolved) is NOT silently emitted as a constant: it
        # records mem_sink_unreadable and returns "" so the recovery layer surfaces
        # MEM_SINK_UNPLACEABLE / the seed-independence exclusion instead.
        if self.output_mem is not None:
            return self._mem_sink_expression()
        target = self.output_reg or self._last_write
        if not target:
            return ""
        try:
            reg_obj = self._ctx.getRegister(target)
            return str(self._ctx.getSymbolicRegisterValue(reg_obj))
        except Exception:
            return ""

    def _mem_sink_expression(self) -> str:
        """Emit the recovered output as the symbolic-memory bytes at the sink.

        Reads ``[sink_addr, sink_addr+sink_size)`` one byte at a time off Triton's
        symbolic memory (``getSymbolicMemoryValue`` over each byte). Emits a
        ``bytes([...])`` literal — the recovered F's output in its REAL shape (a
        store's bytes), never coerced to a single register value. Two honest
        degrade exits (never a silent constant — Issue 7 A8④):

          * the descriptor is malformed / Triton cannot read the interval →
            ``mem_sink_unreadable`` records the read failure;
          * NO byte of the interval ever became symbolic (the sink store wrote a
            seed/driver-independent constant, or the EA never resolved to where the
            symbolic value landed) → ``mem_sink_unreadable`` records the input-
            invariant collapse, so the recovery layer routes it to the seed-
            independence exclusion / MEM_SINK_UNPLACEABLE rather than emitting the
            constant via a register fallback.
        """
        from triton import MemoryAccess  # type: ignore
        desc = self.output_mem or {}
        addr = int(desc.get("sink_addr", 0))
        size = int(desc.get("sink_size", 0))
        if size <= 0:
            self.mem_sink_unreadable = (
                f"mem-sink descriptor has non-positive sink_size={size} — cannot "
                f"read the output interval")
            return ""
        try:
            values: list[int] = []
            any_symbolic = False
            for i in range(size):
                mb = MemoryAccess(addr + i, 1)
                values.append(int(self._ctx.getSymbolicMemoryValue(mb)) & 0xFF)
                if self._ctx.isMemorySymbolized(mb):
                    any_symbolic = True
        except Exception as e:
            self.mem_sink_unreadable = (
                f"cannot read symbolic memory bytes at "
                f"[0x{addr:x}, 0x{addr + size:x}): {type(e).__name__}: {e}")
            return ""
        if not any_symbolic:
            # No byte of the sink interval is a function of the input — the store
            # is input-invariant over this window (a constant / a seed/driver-
            # independent store). NOT a recovery target; surfaced, never emitted.
            self.mem_sink_unreadable = (
                f"the sink interval [0x{addr:x}, 0x{addr + size:x}) is input-"
                f"invariant (no byte became symbolic after the window) — the store "
                f"is seed/driver-independent, not a recovery target")
            return ""
        return "bytes([" + ", ".join(str(v) for v in values) + "])"
