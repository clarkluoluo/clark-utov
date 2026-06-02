"""Opaque-staging diagnosis + symbolic staging-interval logic (Phase 0/1/2(i)).

Origin: the VMP cipher-body recovery (`dev-symbolic-input-through-opaque-staging.md`).
Two representative recovery windows converged on the SAME hard frontier — the
symbolic input enters the window through a *pointer-indirect staging buffer*
(``str x8,[x10]`` → staging → later ``ldr`` reads it back into the computation),
and the symbol does not propagate through it: ``emitted_F`` collapses to a
constant. That is not "one more known-address symbolize" — when the staging
store/load address is *computed* (input-derived / resolved through a pointer
chain), the store-to-load forwarding never happens and taint dies.

This module is the Phase 0 **diagnosis primitive** that splits an opaque window
into the two sub-cases the design enumerates, plus the Phase 1/2(i) symbolic
staging-interval helpers the runner uses to forward a symbolic store to its
later load:

  * Phase 0 — :func:`diagnose_opaque_staging`: deterministic, zero-LLM. Uses
    :func:`setup_symex.audit_address_closure` to find the blind load(s), then
    :func:`stages.s3_triton.build_dfg` to backtrace each blind load's EA register
    and decide whether the EA is symbolic / input-derived. With >= 2 cohort traces
    it byte-level-corroborates (does the EA vary across vectors? which staging
    bytes vary, stored at which idx, loaded at which idx). Verdict:
    ``known_addr`` (→ Phase 1) | ``symbolic_address`` (→ Phase 2) | ``inconclusive``.

  * Phase 1/2(i) — :class:`PointerChainSpec` + :func:`resolve_staging_address`:
    the pointer-chain SHAPE (register names / store→staging→load structure) lives
    in config; the concrete landing addresses come from the trace. The runner's
    ``_symbolic_staging`` interval set (in ``setup_symex_runner``) is seeded from
    this so a symbolic store forwards to a later load whose EA hits the interval —
    by ``op.addr`` (trace concolic), so a symbolic EA forwards just the same.

Zero case-specific knowledge (utov-arch-index invariant 2/6): no concrete
address / offset / idx / hook-table slot / handler-id / case name appears in this
module. The pointer-chain SHAPE is a :class:`PointerChainSpec` the caller fills
from fixture/config; every concrete coordinate arrives via the trace or that spec.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

from .setup_symex import _addr_regs as _ea_address_regs
from .setup_symex import audit_address_closure
from .types import Instruction


# Output discipline (utov-arch-index invariant 4): a list longer than this is
# digested to {count, sha1, sample} so a long trace's byte map is never inlined.
_MAX_INLINE_LIST = 16

# Memory-load semops whose value enters FROM memory — a load through one of these
# is the "read the staging byte back" step. Kept dependency-light (a substring
# check on the mnemonic head) so the primitive does not pull the whole semop
# classifier for one query; it mirrors the runner's own ``ld`` head test.
def _is_load(mnemonic: str) -> bool:
    return mnemonic.strip().lower().startswith("ld")


def _is_store(mnemonic: str) -> bool:
    return mnemonic.strip().lower().startswith("st")


# ---------------------------------------------------------------------------
# Pointer-chain SHAPE — config, never concrete coordinates.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PointerChainSpec:
    """The SHAPE of the staging pointer chain — structure only, zero coordinates.

    The opaque-staging path is ``<base pointer> → store a value → staging buffer →
    later load reads it back``. What is target-specific is only the SHAPE of that
    chain: which register holds the staging base pointer at the store, and which
    register holds it at the load. The CONCRETE landing addresses are resolved
    from the trace (:func:`resolve_staging_address`), never hand-typed here — that
    is the invariant-2 boundary (the spec keeps "register names / store→staging→
    load structural relation" in config, "concrete addr/offset/idx/slot" in
    fixture).

    ``store_base_regs`` — the register(s) the staging STORE computes its EA from.
    ``load_base_regs``  — the register(s) the staging LOAD computes its EA from
                          (often the same pointer; separate so a multi-hop chain
                          can name a different carrier).
    ``store_size`` / ``load_size`` — optional width hint (bytes); ``None`` = take
                          the trace op's own size.
    """

    store_base_regs: tuple[str, ...] = ()
    load_base_regs:  tuple[str, ...] = ()
    store_size:      int | None = None
    load_size:       int | None = None
    note:            str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "store_base_regs": list(self.store_base_regs),
            "load_base_regs":  list(self.load_base_regs),
            "store_size":      self.store_size,
            "load_size":       self.load_size,
            "note":            self.note,
            "kind":            "opaque_staging_pointer_chain",
        }


# ---------------------------------------------------------------------------
# Phase 0 — diagnosis result types.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StagingByte:
    """One staging-buffer byte address: where it is stored, where it is read back,
    and whether its value varies across the cohort (the localize side, sub-case 4).

    ``varies_cohort`` is ``None`` when no cohort was supplied (single-trace
    diagnosis cannot tell) — never silently False."""

    addr:          int
    store_idx:     int | None = None
    load_idx:      int | None = None
    varies_cohort: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "addr":          self.addr,
            "addr_hex":      f"0x{self.addr:x}",
            "store_idx":     self.store_idx,
            "load_idx":      self.load_idx,
            "varies_cohort": self.varies_cohort,
        }


@dataclass(frozen=True, slots=True)
class BlindLoad:
    """A blind load (un-backed address closure) whose EA we diagnosed.

    ``ea_symbolic`` — the EA register backtraces to a symbolic / input-derived
    root, or through a chained memory load (a pointer chain). ``ea_varies_cohort``
    — corroborated across the cohort (``None`` = no cohort / could not align)."""

    idx:              int
    pc:               int
    ea_regs:          tuple[str, ...]
    ea_symbolic:      bool
    ea_varies_cohort: bool | None = None
    reason:           str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "idx":              self.idx,
            "pc":               f"0x{self.pc:x}",
            "ea_regs":          list(self.ea_regs),
            "ea_symbolic":      self.ea_symbolic,
            "ea_varies_cohort": self.ea_varies_cohort,
            "reason":           self.reason,
        }


# Verdicts (the Phase routing the design enumerates).
VERDICT_KNOWN_ADDR       = "known_addr"        # EA constant/non-input-derived → Phase 1
VERDICT_SYMBOLIC_ADDRESS = "symbolic_address"  # EA symbolic / varies → Phase 2
VERDICT_INCONCLUSIVE     = "inconclusive"      # no blind leg / insufficient evidence


@dataclass(frozen=True, slots=True)
class StagingDiagnosis:
    """The Phase 0 verdict: is this opaque window a known-address forwarding miss
    (→ Phase 1) or a real symbolic-addressing frontier (→ Phase 2)?

    ``to_dict`` is what the gap map ingests; the byte list is digested when it
    exceeds the inline cap (invariant 4)."""

    window:        tuple[int, int]
    window_is_idx: bool
    verdict:       str
    blind_loads:   tuple[BlindLoad, ...] = ()
    staging_bytes: tuple[StagingByte, ...] = ()
    reasons:       tuple[str, ...] = ()

    @property
    def routes_to_phase(self) -> int | None:
        if self.verdict == VERDICT_KNOWN_ADDR:
            return 1
        if self.verdict == VERDICT_SYMBOLIC_ADDRESS:
            return 2
        return None

    def to_dict(self) -> dict[str, Any]:
        bytes_list = [b.to_dict() for b in self.staging_bytes]
        if len(bytes_list) > _MAX_INLINE_LIST:
            bytes_field: Any = {
                "_trimmed_list": True,
                "count": len(bytes_list),
                "sha1": hashlib.sha1(
                    json.dumps(bytes_list, default=str, sort_keys=True).encode()
                ).hexdigest(),
                "sample": bytes_list[:8],
            }
        else:
            bytes_field = bytes_list
        return {
            "kind":          "opaque_staging_diagnosis",
            "window":        list(self.window),
            "window_basis":  "idx" if self.window_is_idx else "pc",
            "verdict":       self.verdict,
            "routes_to_phase": self.routes_to_phase,
            "blind_loads":   [b.to_dict() for b in self.blind_loads],
            "staging_bytes": bytes_field,
            "reasons":       list(self.reasons),
        }


# ---------------------------------------------------------------------------
# Phase 0 — the diagnosis algorithm (deterministic, zero LLM).
# ---------------------------------------------------------------------------


def _window_items(items: Sequence[Instruction], window: tuple[int, int],
                  window_is_idx: bool) -> list[Instruction]:
    lo, hi = int(window[0]), int(window[1])
    if lo > hi:
        lo, hi = hi, lo
    key = (lambda ins: ins.idx) if window_is_idx else (lambda ins: ins.pc)
    return [ins for ins in items if lo <= key(ins) <= hi]


@dataclass(slots=True)
class _DfgIndex:
    """Side tables over a window's DFG nodes so the EA backtrace finds a register's
    latest writer (and that writer's node) without an O(n^2) rescan per hop."""

    nodes:      list
    idx_map:    dict           # node idx → DfgNode
    writer_map: dict           # reg name → sorted [writer node idxs]

    def latest_writer(self, reg: str, before_idx: int):
        cand = None
        for widx in self.writer_map.get(reg, ()):   # ascending
            if widx < before_idx:
                cand = widx
            else:
                break
        return self.idx_map.get(cand) if cand is not None else None


def _build_dfg_index(win_items: Sequence[Instruction], nodes) -> _DfgIndex:
    idx_map: dict[int, Any] = {}
    writer_map: dict[str, list[int]] = {}
    for ins, node in zip(win_items, nodes):
        idx_map[node.idx] = node
        for wreg in ins.regs_write:
            writer_map.setdefault(wreg, []).append(node.idx)
    for v in writer_map.values():
        v.sort()
    return _DfgIndex(nodes=list(nodes), idx_map=idx_map, writer_map=writer_map)


def _ea_is_symbolic(
    dfg: _DfgIndex, node, ea_regs: Sequence[str],
    symbolic_inputs: set[str], max_depth: int = 64,
) -> tuple[bool, str]:
    """Backtrace a load's EA register(s) through the DFG and decide symbolic/input.

    Returns ``(ea_symbolic, reason)``. The EA is symbolic when, walking the
    register's producer chain to its roots: (a) a root is itself a symbolic input,
    or (b) the chain passes through a memory load (the address came out of memory —
    a pointer chain). A chain that bottoms out on concrete base + constant offset
    producers only is NOT symbolic (``known_addr``)."""
    seen: set[tuple[str, int]] = set()

    def walk(reg: str, before_idx: int, depth: int) -> tuple[bool, str]:
        if reg in symbolic_inputs:
            return True, f"EA reg {reg} is a symbolic input"
        if depth > max_depth:
            return False, ""
        producer = dfg.latest_writer(reg, before_idx)
        if producer is None:
            # external / live-in register with no in-window producer and not a
            # known symbolic input → a concrete base (known address).
            return False, ""
        key = (reg, producer.idx)
        if key in seen:
            return False, ""
        seen.add(key)
        # A chained memory load feeding the EA = pointer chain → symbolic address.
        if _is_load(producer.mnemonic):
            return True, (f"EA reg {reg} produced by a memory load at idx "
                          f"{producer.idx} (pointer chain)")
        # ALU / move producer: recurse into its source registers.
        for src in producer.reg_deps:
            sym, why = walk(src, producer.idx, depth + 1)
            if sym:
                return True, why
        return False, ""

    for r in ea_regs:
        sym, why = walk(r, node.idx, 0)
        if sym:
            return True, why
    return False, ""


def _cohort_ea_values(
    cohort_traces: Sequence[Sequence[Instruction]], pc: int,
) -> list[int]:
    """The actual EA (MemOp.addr) each cohort trace's load AT ``pc`` reads — the
    byte-level corroboration. Aligned by PC (a window load recurs at one PC)."""
    vals: list[int] = []
    for trace in cohort_traces:
        addr = None
        for ins in trace:
            if ins.pc == pc and _is_load(ins.mnemonic):
                for op in ins.mem:
                    if op.rw == "r":
                        addr = op.addr
                        break
                if addr is not None:
                    break
        if addr is not None:
            vals.append(addr)
    return vals


def _cohort_pc_ea_values(
    cohort_traces: Sequence[Sequence[Instruction]],
    *, region: tuple[int, int] | None, window_is_idx: bool,
) -> dict[tuple[int, str], list[int]]:
    """Per-PC, per-vector EA collection for BOTH store and load mem accesses.

    Generalises :func:`_cohort_ea_values` (which is load-only and per single PC):
    for every ``(pc, rw)`` that has a memory access in ``region``, collect the
    actual ``MemOp.addr`` each cohort vector accesses at that PC. The key is
    ``(pc, "r"|"w")`` so a store PC and a load PC at the same address are separate
    sites; the value is one EA per vector (first access of that rw kind at that PC
    in the vector's region). Aligned by PC (cohort_diff alignment thesis): a
    window's store/load recurs at one PC across vectors.

    ``region`` (inclusive) is in idx or pc basis per ``window_is_idx``; ``None`` =
    the whole trace. Pure / deterministic — no DFG, no symbolic_inputs."""
    out: dict[tuple[int, str], list[int]] = {}
    for trace in cohort_traces:
        pool = (list(trace) if region is None
                else _window_items(list(trace), region, window_is_idx))
        # first EA of each (pc, rw) kind in this vector's region.
        seen_here: dict[tuple[int, str], int] = {}
        for ins in pool:
            for op in ins.mem:
                if op.rw not in ("r", "w"):
                    continue
                key = (ins.pc, op.rw)
                if key not in seen_here:
                    seen_here[key] = op.addr
        for key, addr in seen_here.items():
            out.setdefault(key, []).append(addr)
    return out


@dataclass(frozen=True, slots=True)
class EaVaryingSite:
    """One ``(pc, rw)`` staging site whose effective address VARIES across the
    cohort — the non-redundant opaque signal (the store/load value diff is empty
    under opaque; the EA itself differs because each vector touches a different
    address that never enters the common-address intersection a value diff sees).

    ``idx`` is the reference (first) vector's trace idx for this pc; ``rw`` is
    ``"w"`` (store site) or ``"r"`` (load site). ``sample_eas`` is capped for the
    output discipline (invariant 4)."""

    pc:            int
    idx:           int
    rw:            str
    n_distinct_ea: int
    sample_eas:    tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "pc":            f"0x{self.pc:x}",
            "idx":           self.idx,
            "rw":            self.rw,
            "n_distinct_ea": self.n_distinct_ea,
            "sample_eas":    [f"0x{a:x}" for a in self.sample_eas],
        }


@dataclass(frozen=True, slots=True)
class CohortStagingAdvisory:
    """Localize-side opaque advisory (sub-case 4): the store/load PCs whose EA
    varies across the cohort, so a narrow agent / symex knows WHERE the staging
    is and WHERE to start, without re-deriving it (externalize long-horizon
    state). Cohort-only: zero DFG, zero symbolic_inputs.

    Empty ``ea_varying_sites`` + a note when there is no cohort (< 2 vectors) or
    no EA-varying site (genuinely invisible) — honest, never a false locator."""

    ea_varying_sites: tuple[EaVaryingSite, ...] = ()
    n_cohort:         int = 0
    aligned_region:   tuple[int, int] = (0, 0)
    note:             str = ""

    def to_dict(self) -> dict[str, Any]:
        sites = [s.to_dict() for s in self.ea_varying_sites]
        if len(sites) > _MAX_INLINE_LIST:
            sites_field: Any = {
                "_trimmed_list": True,
                "count": len(sites),
                "sha1": hashlib.sha1(
                    json.dumps(sites, default=str, sort_keys=True).encode()
                ).hexdigest(),
                "sample": sites[:8],
            }
        else:
            sites_field = sites
        return {
            "kind":             "cohort_staging_advisory",
            "ea_varying_sites": sites_field,
            "n_cohort":         self.n_cohort,
            "aligned_region":   list(self.aligned_region),
            "note":             self.note,
        }


def cohort_staging_advisory(
    cohort_traces: Sequence[Sequence[Instruction]],
    *,
    region: tuple[int, int] | None = None,
    window_is_idx: bool = True,
    ignore_addrs: Sequence[int] = (),
) -> CohortStagingAdvisory:
    """Localize-side (Phase 3) opaque advisory — cohort-only, zero DFG / zero
    symbolic_inputs.

    The non-redundant opaque signal is **per-PC EA (address) variance**: each
    vector accesses a DIFFERENT ``MemOp.addr`` at the same store/load PC, so the
    addresses never enter the common-address intersection cohort_diff's VALUE diff
    looks at — the window looks opaque even though the staging location is right
    there. This surfaces those PCs (with idx + rw + sampled EAs) so symex gets a
    head-start. It does NOT use ``_cohort_staging_bytes`` (a fixed-address value
    diff, identically empty under opaque).

    For each ``(pc, rw)`` with a memory access in ``region``, collect each
    vector's EA, drop EAs entirely inside ``ignore_addrs`` (a coupling axis), and
    record an :class:`EaVaryingSite` when >= 2 distinct EAs remain. No cohort
    (< 2) / no EA-varying site → empty advisory + a precise note (never silent,
    never a false locator)."""
    traces = [list(t) for t in cohort_traces]
    n = len(traces)
    region_out = (int(region[0]), int(region[1])) if region is not None else (0, 0)
    if n < 2:
        return CohortStagingAdvisory(
            ea_varying_sites=(), n_cohort=n, aligned_region=region_out,
            note=(f"no cohort staging localization: need >= 2 cohort traces, got "
                  f"{n}; symex must pierce the staging on its own"))

    ig = set(ignore_addrs)
    per_site = _cohort_pc_ea_values(traces, region=region, window_is_idx=window_is_idx)

    # reference vector's first idx for each (pc, rw) — a stable coordinate for the
    # gap map (PC recurs; first occurrence in the reference vector's region).
    ref_pool = (traces[0] if region is None
                else _window_items(traces[0], region, window_is_idx))
    ref_idx: dict[tuple[int, str], int] = {}
    for ins in ref_pool:
        for op in ins.mem:
            if op.rw in ("r", "w"):
                ref_idx.setdefault((ins.pc, op.rw), ins.idx)

    sites: list[EaVaryingSite] = []
    for (pc, rw), eas in per_site.items():
        # only EAs not in the ignored coupling axis count toward the variance.
        kept = [a for a in eas if a not in ig]
        distinct = sorted(set(kept))
        if len(distinct) > 1:
            sites.append(EaVaryingSite(
                pc=pc, idx=ref_idx.get((pc, rw), -1), rw=rw,
                n_distinct_ea=len(distinct),
                sample_eas=tuple(distinct[:_MAX_INLINE_LIST])))
    sites.sort(key=lambda s: (s.idx, s.pc, s.rw))

    if not sites:
        return CohortStagingAdvisory(
            ea_varying_sites=(), n_cohort=n, aligned_region=region_out,
            note=("no cohort-visible staging localization: no store/load PC has an "
                  "effective address that varies across the cohort — the staging is "
                  "genuinely invisible to an address-level diff; symex must pierce "
                  "it on its own (still opaque, just no head-start)"))
    return CohortStagingAdvisory(
        ea_varying_sites=tuple(sites), n_cohort=n, aligned_region=region_out,
        note=(f"{len(sites)} store/load PC(s) access an EA that varies across the "
              f"{n}-vector cohort — symbolize / pierce the staging at these PCs"))


def _cohort_staging_bytes(
    cohort_traces: Sequence[Sequence[Instruction]],
    window: tuple[int, int], window_is_idx: bool,
    pointer_chain: PointerChainSpec | None,
) -> dict[int, StagingByte]:
    """Byte-level: for each staging address written in the window, does its stored
    value vary across the cohort, and at which store/load idx (the localize side).

    Uses the reference (first) trace's window for the addr→(store_idx, load_idx)
    map, then diffs the stored value of that address across all cohort traces."""
    if not cohort_traces:
        return {}
    ref = list(cohort_traces[0])
    win = _window_items(ref, window, window_is_idx)
    store_idx: dict[int, int] = {}
    load_idx: dict[int, int] = {}
    for ins in win:
        for op in ins.mem:
            if op.rw == "w":
                for b in range(op.addr, op.addr + op.size):
                    store_idx.setdefault(b, ins.idx)
            elif op.rw == "r":
                for b in range(op.addr, op.addr + op.size):
                    load_idx.setdefault(b, ins.idx)
    # value of each stored address per cohort vector (byte-granular).
    out: dict[int, StagingByte] = {}
    for addr in sorted(store_idx):
        vals: set[int] = set()
        for trace in cohort_traces:
            for ins in _window_items(list(trace), window, window_is_idx):
                hit = None
                for op in ins.mem:
                    if op.rw == "w" and op.addr <= addr < op.addr + op.size:
                        shift = (addr - op.addr) * 8
                        hit = (op.val >> shift) & 0xFF
                if hit is not None:
                    vals.add(hit)
        varies = (len(vals) > 1) if vals else None
        out[addr] = StagingByte(addr=addr, store_idx=store_idx.get(addr),
                                load_idx=load_idx.get(addr), varies_cohort=varies)
    return out


def diagnose_opaque_staging(
    items: Sequence[Instruction],
    *,
    window: tuple[int, int],
    window_is_idx: bool = True,
    pointer_chain: PointerChainSpec | None = None,
    cohort_traces: Sequence[Sequence[Instruction]] = (),
    symbolic_inputs: Sequence[str] = (),
    min_regs_write_coverage: float = 0.05,
) -> StagingDiagnosis:
    """Phase 0 — split an opaque window: ``known_addr`` (→P1) vs ``symbolic_address``
    (→P2) vs ``inconclusive``. Deterministic, zero LLM.

    Algorithm (Phase 0 + Phase 0b — the load-source gate is the union of two
    sources so the diagnosis fires in BOTH the pre-P1 blind window AND the
    verifier's ``opaque`` branch, where ``backing_ok=True`` makes the un-backed
    legs empty):
      1a (Phase 0).  Un-backed address legs from
         :func:`setup_symex.audit_address_closure` over the window — the loads
         whose EA closure is blind. Their ``addr_regs`` are the EA registers.
      1b (Phase 0b). DFG-derived staging loads: scan the window's
         :func:`stages.s3_triton.build_dfg`, take every memory LOAD, backtrace its
         EA register(s) and keep the ones whose EA is input- / pointer-derived.
         These are BACKED (the trace has a concrete value) yet should carry the
         symbol — exactly the loads the opaque branch fires on. The EA judgment of
         step 2 is moved FORWARD here as the candidate gate. The final diagnostic
         target set is ``(1a) ∪ (1b)``, de-duplicated by load idx.
      2. For each target load, backtrace its EA register(s) through the DFG:
         symbolic if a root is a ``symbolic_inputs`` member OR the chain passes
         through a memory load (pointer chain); else ``known_addr`` (concrete base
         + constant offset).
      3. With >= 2 ``cohort_traces``, corroborate: does the load's actual EA vary
         across vectors (``ea_varies_cohort``), and which staging bytes vary
         (``StagingByte``, the localize side).
      4. Verdict: any target load ``ea_symbolic`` OR ``ea_varies_cohort`` →
         ``symbolic_address``; all EA constant/non-input-derived → ``known_addr``;
         no candidate / insufficient evidence → ``inconclusive`` (+ a note, never
         silent).

    ``pointer_chain`` (when supplied) only NARROWS / prioritises the Phase 0b
    candidates by its load-base register shape; the gate's backbone is the
    structural DFG scan (always present in the window, case-agnostic), so the
    routing never hinges on a pointer-chain spec being supplied.
    """
    from .stages.s3_triton import build_dfg as _build_dfg  # heavy import, local

    sym_inputs = set(symbolic_inputs)
    win_items = _window_items(items, window, window_is_idx)
    reasons: list[str] = []

    # regs_write coverage self-check (the dataflow regs_write-hard-dependency
    # cross-cut): build_dfg / the EA backtrace key off regs_write, so a window
    # whose instructions barely populate regs_write (values only on the read side)
    # cannot be trusted to say "EA backtraces to a concrete base" — that would be a
    # FALSE known_addr (the producer is simply invisible, not absent). When the
    # fraction of window instructions with a non-empty regs_write is below
    # ``min_regs_write_coverage`` the DFG-derived EA classification is not
    # trustworthy; the known_addr verdict is downgraded to inconclusive below.
    # Mirrors cohort_diff's opaque trust-gate, on the Phase 0 / DFG side.
    # ③/④: the regs_write readiness is decided by the SINGLE unified profile
    # (engine.trace_observability), not a second inline count. The threshold is
    # passed through so the verdict stays byte-for-byte; the profile is the source
    # of both the number and the precise downgrade reason.
    from .trace_observability import assess_trace_observability as _assess
    _obs = _assess(win_items, window=None,
                   thresholds={"regs_write": min_regs_write_coverage})
    n_win = _obs.n_items
    regs_write_coverage = _obs.regs_write_rate
    n_regs_write = round(regs_write_coverage * n_win) if n_win else 0
    low_regs_write_coverage = (n_win > 0
                               and not _obs.overall_sufficient_for("regs_write"))

    nodes = _build_dfg(win_items)
    dfg = _build_dfg_index(win_items, nodes)
    idx_map = dfg.idx_map

    # Window loads keyed by idx (the diagnostic-target unit) and by pc (legs
    # report a pc). A pc may recur; the leg/load alignment uses the first.
    loads_by_idx = {ins.idx: ins for ins in win_items if _is_load(ins.mnemonic)}
    pc_to_load: dict[int, Instruction] = {}
    for ins in win_items:
        if _is_load(ins.mnemonic):
            pc_to_load.setdefault(ins.pc, ins)

    # 1a — blind loads from the address-closure audit (the pre-P1 / blind path).
    closure = audit_address_closure(
        items, window=window, window_is_idx=window_is_idx)
    blind_legs = [leg for leg in closure.legs if not leg.backed]

    # Build the merged target set: idx → (load_ins, ea_regs, origin). Origin is
    # bookkeeping only (it does not change the verdict — both paths run the same
    # step-2 EA judgment), but it surfaces in the BlindLoad.reason for the gap map.
    targets: dict[int, tuple[Instruction, tuple[str, ...], str]] = {}
    for leg in blind_legs:
        load_ins = pc_to_load.get(leg.pc)
        if load_ins is None:
            # a blind STORE leg (no load) — not a staging read; skip for EA judgment
            continue
        targets[load_ins.idx] = (load_ins, tuple(leg.addr_regs), "unbacked_leg")

    # 1b (Phase 0b) — DFG-derived backed staging loads. Every window load whose EA
    # backtraces to a symbolic / input- / pointer-derived root is a candidate, even
    # when its address closure is fully backed (the opaque-branch case). This is the
    # gate that catches the staging load the un-backed-legs gate misses.
    chain_load_regs = (set(pointer_chain.load_base_regs)
                       if pointer_chain is not None else set())
    for idx, load_ins in loads_by_idx.items():
        if idx in targets:
            continue   # already a blind leg — keep that origin, do not double-add
        node = idx_map.get(idx)
        if node is None:
            continue
        ea_regs = _ea_address_regs(load_ins.mnemonic)
        if not ea_regs:
            continue
        ea_sym, _why = _ea_is_symbolic(dfg, node, ea_regs, sym_inputs)
        if not ea_sym:
            continue   # backed, concrete-base EA — not a staging-symbol candidate
        # Optional pointer-chain narrowing: when a chain shape is supplied, prefer
        # loads whose EA register matches the chain's named load base. A non-match
        # is still a candidate (the DFG already proved it input-derived) — the
        # chain only re-orders / annotates, the structural scan stays the backbone.
        matches_chain = bool(chain_load_regs & set(ea_regs)) if chain_load_regs else False
        origin = "dfg_staging_chain" if matches_chain else "dfg_staging"
        targets[idx] = (load_ins, tuple(ea_regs), origin)

    blind_loads: list[BlindLoad] = []
    any_symbolic = False
    any_varies = False

    for idx in sorted(targets):
        load_ins, ea_regs, origin = targets[idx]
        node = idx_map.get(idx)
        if node is None:
            continue
        ea_sym, why = _ea_is_symbolic(dfg, node, ea_regs, sym_inputs)
        # cohort corroboration of the EA value.
        ea_varies: bool | None = None
        if len(cohort_traces) >= 2:
            ea_vals = _cohort_ea_values(cohort_traces, load_ins.pc)
            if len(ea_vals) >= 2:
                ea_varies = len(set(ea_vals)) > 1
        if ea_sym:
            any_symbolic = True
        if ea_varies:
            any_varies = True
        base_reason = why or "EA backtraces to concrete base + constant offset"
        blind_loads.append(BlindLoad(
            idx=load_ins.idx, pc=load_ins.pc, ea_regs=tuple(ea_regs),
            ea_symbolic=ea_sym, ea_varies_cohort=ea_varies,
            reason=f"[{origin}] {base_reason}"))

    # 3 — byte-level staging map (the localize side, sub-case 4).
    staging_map = _cohort_staging_bytes(
        cohort_traces, window, window_is_idx, pointer_chain)
    # Also surface any window store landings even without a cohort, so the gap map
    # always shows WHERE staging is (single-trace: varies_cohort stays None).
    if not staging_map:
        store_idx: dict[int, int] = {}
        load_idx: dict[int, int] = {}
        for ins in win_items:
            for op in ins.mem:
                if op.rw == "w":
                    for b in range(op.addr, op.addr + op.size):
                        store_idx.setdefault(b, ins.idx)
                elif op.rw == "r":
                    for b in range(op.addr, op.addr + op.size):
                        load_idx.setdefault(b, ins.idx)
        for addr in sorted(set(store_idx) | set(load_idx)):
            staging_map[addr] = StagingByte(
                addr=addr, store_idx=store_idx.get(addr),
                load_idx=load_idx.get(addr), varies_cohort=None)
    staging_bytes = tuple(staging_map[a] for a in sorted(staging_map))

    # 4 — verdict.
    if not blind_loads:
        if closure.sufficient:
            reasons.append(
                "no candidate load: every address closure is backed AND no window "
                "load's EA backtraces to a symbolic / input- / pointer-derived "
                "root. This window does not stage the symbol behind an input- "
                "derived address (Phase 0b found no DFG candidate). Inconclusive "
                "for opaque-staging routing (not silent).")
        else:
            reasons.append(
                "address closure has un-backed leg(s) but none is a LOAD whose EA "
                "we could backtrace, and no window load's EA is input-derived "
                "(no Phase 0b candidate either) — insufficient evidence to route. "
                "Inconclusive (not silent).")
        verdict = VERDICT_INCONCLUSIVE
    elif any_symbolic or any_varies:
        verdict = VERDICT_SYMBOLIC_ADDRESS
        if any_symbolic:
            reasons.append(
                "at least one blind load's EA backtraces to a symbolic / input-"
                "derived root (or through a chained memory load = pointer chain) "
                "→ symbolic addressing (Phase 2).")
        if any_varies:
            reasons.append(
                "at least one blind load's effective address VARIES across the "
                "cohort → the address is input-derived (Phase 2).")
    elif low_regs_write_coverage:
        # The EA backtraces "clean" (concrete base) — but regs_write coverage in
        # this window is below threshold, so the producers the backtrace would
        # have followed are largely invisible. We must NOT emit a false
        # known_addr: downgrade to inconclusive with a precise note (honest, never
        # silent). regs_read fallback: the read-side value is observed but its
        # provenance is unknown, so "concrete base" cannot be trusted here.
        verdict = VERDICT_INCONCLUSIVE
        reasons.append(
            f"regs_write coverage self-check: only {n_regs_write}/{n_win} window "
            f"instruction(s) have a non-empty regs_write "
            f"(coverage={regs_write_coverage:.2%} < {min_regs_write_coverage:.2%}) "
            f"— the DFG-derived EA classification keys off regs_write producers "
            f"that are largely invisible here, so 'EA backtraces to a concrete "
            f"base' is untrustworthy. The EA register value(s) may be a "
            f"provenance-unknown read-side live-in, NOT a known constant address. "
            f"Inconclusive (not a false known_addr) — feed a regs_write-populated "
            f"trace, or merge regs_read-observed producers.")
    else:
        verdict = VERDICT_KNOWN_ADDR
        reasons.append(
            "every blind load's EA backtraces to a concrete base + constant "
            "offset and is constant across the cohort → the address is known; "
            "the symbol does not forward because the store→load forwarding did "
            "not happen, not because the address is symbolic (Phase 1).")

    return StagingDiagnosis(
        window=(int(window[0]), int(window[1])),
        window_is_idx=window_is_idx,
        verdict=verdict,
        blind_loads=tuple(blind_loads),
        staging_bytes=staging_bytes,
        reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# Phase 1/2(i) — resolve the staging landing address(es) from the pointer chain.
# ---------------------------------------------------------------------------


def resolve_staging_address(
    items: Sequence[Instruction],
    pointer_chain: PointerChainSpec | None,
    *,
    window: tuple[int, int] | None = None,
    window_is_idx: bool = True,
) -> list[tuple[int, int]]:
    """Resolve the concrete staging landing ``(addr, size)`` interval(s) the
    pointer chain stores into, from the TRACE (never hand-typed).

    The chain SHAPE (:class:`PointerChainSpec`) names which register(s) the
    staging store computes its EA from; the trace's store ``MemOp`` at a matching
    instruction gives the concrete landing address + size. Returned intervals seed
    the runner's ``_symbolic_staging`` set so a symbolic store forwards to its
    later load. With no chain spec / no matching store, returns ``[]`` (the runner
    then falls back to the symbolic_mem leg / runtime store growth — never a
    silent guess of an address).
    """
    if pointer_chain is None:
        return []
    base_regs = set(pointer_chain.store_base_regs)
    pool = (items if window is None
            else _window_items(items, window, window_is_idx))
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for ins in pool:
        if not _is_store(ins.mnemonic):
            continue
        # the store's EA registers must include the chain's named base register(s)
        if base_regs and not (base_regs & set(ins.regs_read.keys())):
            continue
        for op in ins.mem:
            if op.rw != "w":
                continue
            size = pointer_chain.store_size or op.size
            iv = (op.addr, int(size))
            if iv not in seen:
                seen.add(iv)
                out.append(iv)
    return out


def derive_pointer_chain(
    diag: "StagingDiagnosis",
    items: Sequence[Instruction],
    *,
    window: tuple[int, int],
    window_is_idx: bool = True,
) -> PointerChainSpec | None:
    """Self-produce the pointer-chain SHAPE from a finished diagnosis + the trace,
    so the opaque fallback narrows the store side WITHOUT the caller hand-typing a
    :class:`PointerChainSpec` (the caller cannot fill a case-specific shape — it is
    derived structurally here, never a caller obligation).

    The shape sources are already on the diagnosis / window — no extra analysis:

      * ``load_base_regs`` — the union of every blind-load's ``ea_regs`` (the
        staging LOAD's EA base register(s) the Phase 0 DFG backtrace already named),
        de-duplicated.
      * ``store_base_regs`` — the EA base register(s) of the staging STOREs. A
        staging store is a window STORE whose landing byte(s) the diagnosis already
        mapped as a staging address (``StagingByte.store_idx``); its EA registers
        come from :func:`setup_symex._addr_regs` over the store mnemonic. This ties
        the store side to the SAME staging bytes the diagnosis found (not every
        window store) and is self-contained — it needs no cohort, so it produces a
        shape in drive's single-trace fallback where ``cohort_traces=()``.

    Returns ``None`` when no staging store / no store base register can be derived
    (invariant 8 — never fabricate a shape). The store side is the load-bearing one
    for forwarding (``resolve_staging_address`` keys off ``store_base_regs``); with
    no store base reg the narrow would be empty anyway, so ``None`` is honest and
    keeps the fallback backbone-only (invariant 7)."""
    # The staging store idxs the diagnosis already mapped (byte → store_idx). This
    # restricts the store-base extraction to the staging bytes, not every window
    # store, so the derived shape stays tied to the diagnosed staging structure.
    staging_store_idxs = {b.store_idx for b in diag.staging_bytes
                          if b.store_idx is not None}
    if not staging_store_idxs:
        return None
    pool = _window_items(items, window, window_is_idx)
    store_base: list[str] = []
    seen_store: set[str] = set()
    for ins in pool:
        if ins.idx not in staging_store_idxs:
            continue
        if not _is_store(ins.mnemonic):
            continue
        for reg in _ea_address_regs(ins.mnemonic):
            if reg not in seen_store:
                seen_store.add(reg)
                store_base.append(reg)
    if not store_base:
        return None
    load_base: list[str] = []
    seen_load: set[str] = set()
    for bl in diag.blind_loads:
        for reg in bl.ea_regs:
            if reg not in seen_load:
                seen_load.add(reg)
                load_base.append(reg)
    return PointerChainSpec(
        store_base_regs=tuple(store_base),
        load_base_regs=tuple(load_base),
        note=("auto-derived from the opaque-staging diagnosis: store base from the "
              "diagnosed staging store(s)' EA register(s), load base from the blind "
              "load(s)' EA register(s) — no caller-supplied shape"))


__all__ = [
    "PointerChainSpec",
    "StagingByte",
    "BlindLoad",
    "StagingDiagnosis",
    "EaVaryingSite",
    "CohortStagingAdvisory",
    "cohort_staging_advisory",
    "VERDICT_KNOWN_ADDR",
    "VERDICT_SYMBOLIC_ADDRESS",
    "VERDICT_INCONCLUSIVE",
    "diagnose_opaque_staging",
    "resolve_staging_address",
    "derive_pointer_chain",
]
