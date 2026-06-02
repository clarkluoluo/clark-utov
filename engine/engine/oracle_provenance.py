"""Oracle-anchored provenance mode (#3) — backtrace from a located sink.

Builds on the #1 sink-validator: given a sink that has already been LOCATED and
confirmed observable (its base address, typically from
:func:`engine.oracle_sink.validate_sink`), walk the producer chain backward and
classify how the output is produced — with an explicit, terminating verdict.

The verdict is the deliverable (it tells the consumer what to do next):

  CONTINUOUS_BUFFER   the output sits in a contiguous WRITE buffer matching the
                      expected bytes — reports base + the producing PCs.
  STREAMING           the producer chain is observable, but the output is written
                      byte/chunk-wise from transient values and never lives in a
                      single contiguous buffer — reports each chunk's producer PC.
                      (production visible, just no buffer.)
  NEEDS_OBSERVATION   the chain breaks at a read of an address with NO captured
                      write/snapshot — production is not visible at all — reports
                      a precise next-watch list (addr + reading PC).

Distinction: STREAMING = production visible, no buffer; NEEDS_OBSERVATION =
production not visible. Generic — no target address or runner format is baked in;
reuses Instruction.mem + canonical MemSnapshot observations + the S3 reg graph.

Non-goals: this mode does NOT collect new observations (that is the consumer's
job — it acts on next_watch) and does NOT decode/transcode (it only backtraces).
"""

from __future__ import annotations

import enum
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .types import Instruction


class ProvenanceVerdict(str, enum.Enum):
    CONTINUOUS_BUFFER = "CONTINUOUS_BUFFER"
    STREAMING = "STREAMING"
    OPAQUE_CALLEE = "OPAQUE_CALLEE"          # aka BRIDGE_BOUNDARY: produced by an
                                            # un-traced callee/bridge across a call
    BOUNDARY_EDGE = "BOUNDARY_EDGE"         # anchored via a case-DECLARED boundary
                                            # edge (call-return/decode/framing): the
                                            # logical/transform output has no native
                                            # writer, but a declared edge places it
                                            # at the boundary and surfaces the
                                            # pre-transform source_ptr as the next
                                            # backtrace anchor. NOT a traced writer,
                                            # NOT closure (see BoundaryEdge / inv 7).
    NEEDS_OBSERVATION = "NEEDS_OBSERVATION"


@dataclass(frozen=True)
class BoundaryEdge:
    """A case-DECLARED boundary / provenance edge (additive case input).

    The logical/transform output (post call-return / base64 / framing / jni_string
    decode) has a VALUE but NO native writer in the trace, so the producer backtrace
    dead-ends UNPLACEABLE. A case that KNOWS the boundary can declare this edge to
    anchor the output at the boundary and surface the pre-transform ``source_ptr``
    (which DOES have a native producer) as the next backtrace anchor — pushing the
    frontier one notch forward instead of stalling.

    Generic — utov NEVER hard-codes a boundary; this is caller-supplied DATA. The
    shape is target-agnostic: any (boundary_pc_from → boundary_pc_to, source_ptr,
    transform). ``transform`` reuses the ``expected_repr`` vocabulary
    (raw|base64|framing|jni_string|…; see :class:`engine.recapture.RecaptureSpec`).
    ``decode_meta`` is an opaque case-carried annotation (marker/raw_len/body_offset
    …) — utov does NOT decode/transcode (module contract); it is a hint + audit
    record only.

    SAFETY (invariant 7, A2): declaring an edge anchors but does NOT close. The
    resulting verdict is :attr:`ProvenanceVerdict.BOUNDARY_EDGE`, explicitly marked
    "anchored via DECLARED boundary edge (case-asserted, not a traced writer)", and
    it never feeds a close/parity gate. If ``source_ptr`` also has no producer the
    result still surfaces it as the next watch (the wall moves one notch; it is not
    falsely re-anchored forever).
    """
    sink_surface: int                # the post-transform sink addr being anchored
    boundary_pc_from: int            # boundary start pc (e.g. the call site)
    boundary_pc_to: int              # boundary end pc — output appears here
    source_ptr: int                  # pre-transform native source pointer
    transform: str = "raw"           # raw|base64|framing|jni_string|… (expected_repr)
    decode_meta: dict[str, Any] = field(default_factory=dict)  # opaque case hint

    def to_dict(self) -> dict[str, Any]:
        return {
            "sink_surface": f"0x{self.sink_surface:x}",
            "boundary_pc_from": f"0x{self.boundary_pc_from:x}",
            "boundary_pc_to": f"0x{self.boundary_pc_to:x}",
            "source_ptr": f"0x{self.source_ptr:x}",
            "transform": self.transform,
            "decode_meta": dict(self.decode_meta),
        }

    @staticmethod
    def from_wire(d: dict[str, Any]) -> "BoundaryEdge":
        """Build from a case-declared dict (hex strings or ints accepted)."""
        def _addr(v: Any) -> int:
            return int(v, 16) if isinstance(v, str) else int(v)
        return BoundaryEdge(
            sink_surface=_addr(d["sink_surface"]),
            boundary_pc_from=_addr(d["boundary_pc_from"]),
            boundary_pc_to=_addr(d["boundary_pc_to"]),
            source_ptr=_addr(d["source_ptr"]),
            transform=str(d.get("transform", "raw")),
            decode_meta=dict(d.get("decode_meta", {})),
        )


@dataclass(frozen=True)
class ProvenanceResult:
    verdict: ProvenanceVerdict
    base: int | None = None                  # CONTINUOUS_BUFFER: the buffer base
    producer_pcs: tuple[int, ...] = ()       # producing PCs (buffer / stream chunks)
    boundary_pcs: tuple[int, ...] = ()       # OPAQUE_CALLEE: the call boundary PC(s)
    callee_targets: tuple[int, ...] = ()     # OPAQUE_CALLEE: resolved callee addr(s)
    transient: bool = False                  # CONTINUOUS_BUFFER existed then scrubbed
    streaming: str = ""                      # ""|"confirmed"|"unprovable" (honest mark)
    chain: list[dict] = field(default_factory=list)        # backtrace steps
    next_watch: list[dict] = field(default_factory=list)   # NEEDS_OBSERVATION gaps
    expected: bytes = b""
    detail: str = ""
    # ④ trust-gate: the unified ③ output-observation self-check. None when not
    # assessed; True/False = the sink region was / was not observed in the trace
    # (a captured write or snapshot covers it). False → "output not observed, needs
    # re-capture" (utov self-check; re-capture is the harness's job). Additive
    # (default None) so existing serializations stay byte-for-byte unchanged.
    sink_captured: bool | None = None
    # Generation/backtrace budget (dev-recovery-generation-budget-spec): the producer
    # backtrace is bounded in depth (steps) AND breadth (BFS frontier). When either
    # ceiling truncated the walk, this carries WHAT was cut (mode + counts) so the
    # truncation is never silent (A8④ / No silent caps). None ⇒ the walk completed
    # within budget → byte-for-byte unchanged serialization (invariant 7).
    backtrace_truncated: dict[str, Any] | None = None
    # boundary-edge anchoring (dev-boundary-edge-provenance-spec): when a case
    # DECLARES a boundary edge that covers an otherwise-UNPLACEABLE logical/transform
    # sink, this carries the edge that was used to anchor (case-asserted, NOT a
    # traced writer). None ⇒ no edge consumed → byte-for-byte unchanged serialization
    # (invariant 7). Verdict is BOUNDARY_EDGE; never an auto-CLOSED/CONFIRMED signal.
    anchored_edge: BoundaryEdge | None = None
    # observation planner (spec_provenance_observation_planner / #4): the
    # instruction/edge-SHAPE heuristic layer's proposed next observe points,
    # generated ALONGSIDE next_watch. Each entry is a proposal dict
    # {pc, when, capture, regs?, mem?, mem_regrel?, reason, heuristic} — what to
    # capture next + WHY + which rule proposed it. This is ADDITIVE on top of (never
    # a replacement for) next_watch: a gap no heuristic matched stays surfaced in
    # next_watch, so the plan can never silently hide an unobserved gap (A8④).
    # Default empty ⇒ the key is omitted from to_dict (byte-for-byte unchanged
    # serialization for every result with no proposals; invariant 7). Built by
    # engine.observation_planner.plan_for_result (kept out of this module to avoid an
    # import cycle; trace_provenance attaches it in _finish).
    observation_plan: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "verdict": self.verdict.value,
            "base": None if self.base is None else f"0x{self.base:x}",
            "producer_pcs": [f"0x{p:x}" for p in self.producer_pcs],
            "boundary_pcs": [f"0x{p:x}" for p in self.boundary_pcs],
            "callee_targets": [f"0x{t:x}" for t in self.callee_targets],
            "transient": self.transient,
            "streaming": self.streaming,
            "chain": self.chain,
            "next_watch": self.next_watch,
            "expected": self.expected.hex(),
            "detail": self.detail,
        }
        # Additive: only emit the key when actually assessed, so every existing
        # result's serialization is byte-for-byte unchanged (default None path).
        if self.sink_captured is not None:
            out["sink_captured"] = self.sink_captured
        # Additive: only emit when the backtrace actually truncated (default None →
        # unchanged serialization for every budget-internal walk; invariant 7).
        if self.backtrace_truncated is not None:
            out["backtrace_truncated"] = self.backtrace_truncated
        # Additive: only emit when a declared boundary edge was actually consumed
        # (default None → unchanged serialization; invariant 7). WARN-loud audit:
        # the edge is case-asserted, never a traced writer.
        if self.anchored_edge is not None:
            out["anchored_edge"] = self.anchored_edge.to_dict()
        # Additive: only emit when the heuristic planner proposed observe points
        # (default empty → unchanged serialization; invariant 7). The plan rides
        # ALONGSIDE next_watch — never replaces it (A8④).
        if self.observation_plan:
            out["observation_plan"] = self.observation_plan
        return out


def _recon(write_map, base, length):
    """Last-write-wins reconstruction over WRITE bytes; (None, ()) on a hole."""
    out = bytearray(length)
    pcs: list[int] = []
    for off in range(length):
        e = write_map.get(base + off)
        if e is None:
            return None, ()
        out[off] = e[0]
        if e[1] not in pcs:
            pcs.append(e[1])
    return bytes(out), tuple(pcs)


def _temporal_buffer_hit(items, sink_base: int, expected: bytes):
    """Replay write HISTORY (not just last-write-wins). Return (idx, producer_pcs)
    of the FIRST point at which [sink_base, +len) == expected — recovering a buffer
    that existed transiently and was later overwritten/scrubbed. (None, ()) if it
    never held."""
    L = len(expected)
    running: dict[int, tuple[int, int]] = {}   # addr -> (byte, pc)
    for ins in items:
        touched = False
        for op in ins.mem:
            if op.rw == "w" and op.size > 0:
                for k in range(op.size):
                    running[op.addr + k] = ((op.val >> (8 * k)) & 0xFF, ins.pc)
                    touched = True
        if not touched:
            continue
        pcs: list[int] = []
        ok = True
        for off in range(L):
            e = running.get(sink_base + off)
            if e is None or e[0] != expected[off]:
                ok = False
                break
            if e[1] not in pcs:
                pcs.append(e[1])
        if ok:
            return ins.idx, tuple(pcs)
    return None, ()


_CALL_OPS = ("bl", "blr", "blx", "call")


def _is_call(mnemonic: str) -> bool:
    op = mnemonic.strip().split(None, 1)[0].lower() if mnemonic.strip() else ""
    return op in _CALL_OPS


def _resolve_call_target(ins) -> int | None:
    """Resolve a call's concrete target from the trace (which is concrete).

    Direct call ('bl #0x72ecc') -> the literal. Indirect call ('blr x8' / 'br x9')
    -> the register's concrete value READ at this instruction (regs_read). Returns
    None if it cannot be resolved from the available data.
    """
    parts = ins.mnemonic.strip().split(None, 1)
    if len(parts) < 2:
        return None
    first = parts[1].split(",")[0].strip().lstrip("#")
    if first and first[0] in "xXwW" and first[1:].isdigit():
        reg = ("x" + first[1:]) if first[0] in "wW" else first.lower()
        return ins.regs_read.get(reg)
    m = re.search(r"0x[0-9a-fA-F]+", first)
    return int(m.group(0), 16) if m else None


def _first_seen_ins(items, sink_base: int, expected: bytes):
    """The instruction at which the sink region first READS back as expected
    (accumulated read observations). None if never observed via reads."""
    L = len(expected)
    read_view: dict[int, int] = {}
    for ins in items:
        for op in ins.mem:
            if op.rw == "r" and op.size > 0:
                for k in range(op.size):
                    read_view[op.addr + k] = (op.val >> (8 * k)) & 0xFF
        if all(read_view.get(sink_base + off) == expected[off] for off in range(L)):
            return ins
    return None


def _step(ins, note: str = "", **extra) -> dict:
    s = {"idx": ins.idx, "pc": f"0x{ins.pc:x}", "mnemonic": ins.mnemonic,
         "reads": sorted(ins.regs_read.keys())}
    if note:
        s["note"] = note
    s.update(extra)
    return s


def _opaque_callee(items, sink_base: int, expected: bytes, write_byte_producer):
    """Detect output produced by an un-traced callee/bridge: NO traced write
    touches the sink, but the data is OBSERVED present (via reads) only AFTER a
    call returns (absent before). Returns (boundary_ins, first_seen_ins) or None.
    """
    L = len(expected)
    if any((sink_base + off) in write_byte_producer for off in range(L)):
        return None  # a traced write touches the sink -> NEEDS_OBSERVATION path
    first_seen = _first_seen_ins(items, sink_base, expected)
    if first_seen is None:
        return None  # never observed via reads -> cannot prove a call boundary
    before_calls = [ins for ins in items
                    if ins.idx < first_seen.idx and _is_call(ins.mnemonic)]
    if not before_calls:
        return None
    return before_calls[-1], first_seen


def _streaming_match(items, expected: bytes):
    """If the ordered stream of WRITTEN bytes contains expected as a contiguous
    run, return the producing PCs (ordered, unique); else ()."""
    stream: list[tuple[int, int]] = []   # (byte, pc) in trace order, little-endian
    for ins in items:
        for op in ins.mem:
            if op.rw == "w" and op.size > 0:
                for k in range(op.size):
                    stream.append(((op.val >> (8 * k)) & 0xFF, ins.pc))
    L = len(expected)
    for i in range(len(stream) - L + 1):
        if bytes(b for b, _ in stream[i:i + L]) == expected:
            pcs: list[int] = []
            for _, pc in stream[i:i + L]:
                if pc not in pcs:
                    pcs.append(pc)
            return tuple(pcs)
    return ()


def _walk_back(start_idxs, by_idx, dfg, write_byte_producer, snap_addrs,
               max_steps, max_breadth=None):
    """BFS backward over reg-deps + per-byte memory producers. Records the chain
    and, at any memory read of an un-captured byte, a next-watch entry.

    Bounded in DEPTH (``max_steps`` — how many producers are popped) AND BREADTH
    (``max_breadth`` — how many pending producers may sit on the stack at once).
    Both are budget ceilings (dev-recovery-generation-budget-spec): a long trace or
    a wide producer fan-out can otherwise make this O(huge). On either ceiling the
    walk STOPS and returns a ``truncated`` report (mode + counts) so the cut is
    never silent (A8④). ``max_breadth=None`` → no breadth cap (today's behaviour,
    invariant 7)."""
    chain: list[dict] = []
    next_watch: list[dict] = []
    seen: set[int] = set()
    seen_watch: set[tuple[int, int]] = set()
    stack = list(start_idxs)
    steps = 0
    truncated: dict[str, Any] | None = None
    breadth_dropped = 0
    while stack and steps < max_steps:
        idx = stack.pop()
        steps += 1
        if idx in seen:
            continue
        seen.add(idx)
        ins = by_idx.get(idx)
        if ins is None:
            continue
        reads_info: list[str] = []
        for op in ins.mem:
            if op.rw == "r" and op.size > 0:
                reads_info.append(f"mem[0x{op.addr:x}:{op.size}]")
                for k in range(op.size):
                    a = op.addr + k
                    if a in write_byte_producer:
                        pidx = write_byte_producer[a][0]
                        if pidx not in seen:
                            stack.append(pidx)
                    elif a in snap_addrs:
                        pass  # observed via snapshot — a terminus, not a gap
                    else:
                        key = (a, ins.pc)
                        if key not in seen_watch:
                            seen_watch.add(key)
                            next_watch.append({
                                "addr": f"0x{a:x}", "pc": f"0x{ins.pc:x}",
                                "reason": "read of address with no captured write/snapshot",
                            })
        node = dfg.get(idx)
        if node is not None:
            for r, pidx in node.reg_deps.items():
                reads_info.append(r)
                if pidx is not None and pidx not in seen:
                    stack.append(pidx)
        chain.append({"idx": idx, "pc": f"0x{ins.pc:x}",
                      "mnemonic": ins.mnemonic, "reads": reads_info})
        # Breadth ceiling: the pending-producer frontier (branch fan-out) is capped.
        # Drop the OLDEST pending entries (FIFO tail of the LIFO stack) so the walk
        # keeps following the most-recently-discovered producers; report how many
        # branches were pruned. (Depth is handled by the while-condition above.)
        if max_breadth is not None and len(stack) > max_breadth:
            breadth_dropped += len(stack) - max_breadth
            stack = stack[-max_breadth:]
    if steps >= max_steps and stack:
        truncated = {"mode": "depth", "max_steps": max_steps, "steps": steps,
                     "pending_producers": len(stack)}
    if breadth_dropped:
        t = {"mode": "breadth", "max_breadth": max_breadth,
             "branches_dropped": breadth_dropped}
        # depth + breadth can both fire; keep both (depth is the harder stop).
        truncated = {**t, **truncated} if truncated else t
    chain.sort(key=lambda s: s["idx"])
    return chain, next_watch, truncated


def trace_provenance(
    items: Iterable[Instruction],
    expected_output: bytes,
    *,
    sink_base: int,
    snapshots: Iterable | None = None,
    max_steps: int = 100_000,
    max_breadth: int | None = None,
    assess_observability: bool = False,
    boundary_edge: BoundaryEdge | dict | None = None,
    plan_observations: bool = False,
    import_map: Any = None,
) -> ProvenanceResult:
    """Backtrace from the located ``sink_base`` and classify the production.

    ``boundary_edge`` (dev-boundary-edge-provenance-spec, additive / opt-in): an
    OPTIONAL case-DECLARED boundary edge (:class:`BoundaryEdge` or its wire dict).
    When the sink is a logical/transform output that has NO native writer — so the
    producer backtrace would otherwise dead-end UNPLACEABLE — and a declared edge
    COVERS the sink, the verdict becomes :attr:`ProvenanceVerdict.BOUNDARY_EDGE`:
    anchored at ``boundary_pc_to`` (recorded in ``boundary_pcs``) with the
    pre-transform ``source_ptr`` surfaced as the next backtrace anchor (next_watch).
    SAFETY: anchoring is NOT closure — the verdict is explicitly marked
    "anchored via DECLARED boundary edge (case-asserted, not a traced writer)" and
    never feeds a close/parity gate (invariant 7). With NO edge declared the
    behaviour is byte-for-byte today (honest UNPLACEABLE BLOCK). utov NEVER decodes
    the transform; ``transform``/``decode_meta`` are annotation + anchoring hints
    only. The edge is only consumed when the sink truly has no native writer — a
    real traced CONTINUOUS_BUFFER / STREAMING production always wins (an edge never
    overrides observed production).

    ``assess_observability`` (④ trust-gate, opt-in so existing results stay
    byte-for-byte): when set, the unified ③ self-check
    (:func:`engine.trace_observability.assess_trace_observability`) is run over the
    sink region and its ``sink_captured`` verdict is attached to every result —
    so the consumer can see "output not observed, needs re-capture" explicitly,
    distinct from a producer-chain gap. Re-capture is the harness's job; this only
    self-checks and reports.

    ``max_steps`` / ``max_breadth`` bound the producer backtrace in DEPTH (steps
    popped) and BREADTH (pending-producer frontier). When either truncates the walk,
    every result carries a ``backtrace_truncated`` report (mode + counts) — the cut
    is explicit, never silent (dev-recovery-generation-budget-spec / A8④).
    ``max_breadth=None`` (default) → no breadth cap, byte-for-byte today (inv 7).

    ``plan_observations`` (spec #4, opt-in so existing results stay byte-for-byte):
    when set, the instruction/edge-SHAPE heuristic layer
    (:mod:`engine.observation_planner`) is run over this result and its proposed
    next observe points are attached as ``observation_plan`` — ALONGSIDE (never a
    replacement for) ``next_watch``. ``import_map`` (optional) is the per-binary
    :class:`engine.import_map.ImportMap` the ``extern_call`` rule consults to resolve
    a ``bl/blr`` target to a symbol; with no map that rule simply does not fire (the
    gap stays in ``next_watch``). Default off → no plan generated, serialization
    unchanged."""
    items = list(items)
    snapshots = list(snapshots or [])
    expected = bytes(expected_output)
    if not expected:
        raise ValueError("expected_output must be non-empty")
    L = len(expected)

    # Normalise an optional case-DECLARED boundary edge (additive). Accept either a
    # BoundaryEdge or its wire dict. An edge is only RELEVANT when it covers this
    # sink_base (sink_surface within [sink_base, +L)); a mismatched edge is ignored
    # (no silent mis-anchoring of a different surface).
    edge: BoundaryEdge | None = None
    if boundary_edge is not None:
        edge = (boundary_edge if isinstance(boundary_edge, BoundaryEdge)
                else BoundaryEdge.from_wire(boundary_edge))
        if not (sink_base <= edge.sink_surface < sink_base + L):
            edge = None

    # Backtrace-budget truncation report (set by _walk_back below); attached to
    # every result by _finish so the cut rides every verdict, never silently.
    walk_truncated: dict[str, Any] | None = None

    def _finish(result: ProvenanceResult) -> ProvenanceResult:
        import dataclasses as _dc
        # Attach the backtrace-budget truncation report (None ⇒ unchanged; inv 7).
        if walk_truncated is not None:
            result = _dc.replace(result, backtrace_truncated=walk_truncated)
        # ④: attach the unified ③ sink-captured self-check when requested. Uses the
        # same module the other analyses gate on (single source). Re-uses the
        # already-built snapshots; sink_window covers the whole expected region.
        if assess_observability:
            from .trace_observability import assess_trace_observability
            obs = assess_trace_observability(
                items, sink_window=(sink_base, sink_base + L - 1),
                snapshots=snapshots)
            result = _dc.replace(result, sink_captured=obs.sink_captured)
        # spec #4: attach the heuristic observation plan ALONGSIDE next_watch when
        # requested. Lazy import (planner imports this module — break the cycle).
        # Default off ⇒ serialization byte-for-byte unchanged (invariant 7 / A8③).
        if plan_observations:
            from .observation_planner import plan_for_result
            plan = plan_for_result(result, items, import_map=import_map)
            if plan:
                result = _dc.replace(result, observation_plan=plan)
        return result

    by_idx = {ins.idx: ins for ins in items}
    write_map: dict[int, tuple[int, int]] = {}            # addr -> (byte, pc)
    write_byte_producer: dict[int, tuple[int, int]] = {}  # addr -> (idx, pc)
    for ins in items:
        for op in ins.mem:
            if op.rw == "w" and op.size > 0:
                for k in range(op.size):
                    a = op.addr + k
                    write_map[a] = ((op.val >> (8 * k)) & 0xFF, ins.pc)
                    write_byte_producer[a] = (ins.idx, ins.pc)
    snap_addrs: set[int] = set()
    for s in snapshots:
        for k in range(len(s.data)):
            snap_addrs.add(s.addr + k)

    from .stages.s3_triton import build_dfg
    dfg = {n.idx: n for n in build_dfg(items)}

    sink_writer_idxs = sorted({
        write_byte_producer[a][0]
        for a in range(sink_base, sink_base + L) if a in write_byte_producer
    })
    chain, next_watch, walk_truncated = _walk_back(
        sink_writer_idxs, by_idx, dfg, write_byte_producer, snap_addrs,
        max_steps, max_breadth)
    # Snapshot-located sink (no traced writer): _walk_back is empty. Seed the chain
    # with the first-seen observation step so it is never empty when the data was
    # observed in the trace (round-3 fix: no more chain_len=0).
    first_seen = _first_seen_ins(items, sink_base, expected)
    if not chain and first_seen is not None:
        chain = [_step(first_seen, note=f"first observation of sink data at 0x{sink_base:x}")]

    # 1. CONTINUOUS_BUFFER (final) — contiguous WRITE buffer == expected, still live.
    rec, region_pcs = _recon(write_map, sink_base, L)
    if rec is not None and rec == expected:
        return _finish(ProvenanceResult(
            ProvenanceVerdict.CONTINUOUS_BUFFER, base=sink_base,
            producer_pcs=region_pcs, chain=chain, expected=expected,
            detail=f"output sits in a contiguous write buffer at 0x{sink_base:x}",
        ))

    # 1b. CONTINUOUS_BUFFER (transient) — the buffer existed at some point then was
    #     overwritten/scrubbed; the temporal write-history scan recovers it instead
    #     of mislabelling the scrubbed final state as NEEDS_OBSERVATION.
    t_idx, t_pcs = _temporal_buffer_hit(items, sink_base, expected)
    if t_idx is not None:
        return _finish(ProvenanceResult(
            ProvenanceVerdict.CONTINUOUS_BUFFER, base=sink_base, producer_pcs=t_pcs,
            transient=True, chain=chain, expected=expected,
            detail=(f"output formed a contiguous write buffer at 0x{sink_base:x} "
                    f"(idx {t_idx}) then was overwritten/scrubbed — recovered by the "
                    f"temporal write-history scan, not last-write-wins"),
        ))

    # 2. STREAMING (confirmed) — every output byte is visibly written, never a buffer.
    streaming_pcs = _streaming_match(items, expected)
    if streaming_pcs:
        return _finish(ProvenanceResult(
            ProvenanceVerdict.STREAMING, producer_pcs=streaming_pcs, streaming="confirmed",
            chain=chain, expected=expected,
            detail=("producer chain observable, but the output is written "
                    "byte/chunk-wise from transient values — no contiguous buffer "
                    "holds the full output (streaming confirmed)"),
        ))

    # 3. A traced write DOES touch the sink, but produced no buffer/stream → the
    #    producer chain reads an un-captured native address. NEEDS_OBSERVATION.
    has_traced_sink_write = any(
        (sink_base + off) in write_byte_producer for off in range(L))
    if has_traced_sink_write:
        if not next_watch:
            for off in range(L):
                a = sink_base + off
                if a not in write_byte_producer and a not in snap_addrs:
                    next_watch.append({
                        "addr": f"0x{a:x}", "pc": None,
                        "reason": "sink byte has no captured write/snapshot producer",
                    })
        return _finish(ProvenanceResult(
            ProvenanceVerdict.NEEDS_OBSERVATION, chain=chain, next_watch=next_watch,
            expected=expected,
            detail=("a traced instruction produces part of the sink, but its chain "
                    "reads an address with no captured write/snapshot — watch the "
                    "listed native addresses and re-validate."),
        ))

    # 3b. BOUNDARY_EDGE — the sink is a logical/transform output with NO native
    #     writer (the producer backtrace would dead-end UNPLACEABLE), but the case
    #     DECLARED a boundary edge covering it. Anchor at the edge's boundary_pc_to
    #     and surface the pre-transform source_ptr as the NEXT backtrace anchor —
    #     pushing the frontier one notch forward. SAFETY (invariant 7 / A2): this is
    #     anchoring, NOT closure; the verdict is explicitly case-asserted (not a
    #     traced writer) and never feeds a close/parity gate. utov does NOT decode
    #     the transform; the edge is a placement + hint only. Reached only when no
    #     traced write touches the sink (has_traced_sink_write is False above), so a
    #     real traced production always wins — an edge never overrides observed data.
    if edge is not None:
        # Does the declared pre-transform source_ptr itself have a traced native
        # producer? If so, surface it as the next anchor WITH its reading/producing
        # PC (the wall moves to a real, walkable surface). If not, surface it as a
        # plain watch (pc:null) — the wall moves one notch forward, NOT re-anchored
        # forever (honest UNPLACEABLE at the new frontier).
        src_producer = write_byte_producer.get(edge.source_ptr)
        src_pc = f"0x{src_producer[1]:x}" if src_producer is not None else None
        if src_producer is not None:
            src_reason = ("pre-transform source has a traced producer — "
                          "backtrace continues from here")
        else:
            src_reason = ("pre-transform source has no captured write/snapshot "
                          "producer — wall moved one notch forward, still "
                          "unplaceable here (anchor did NOT close it)")
        edge_next_watch = [{
            "addr": f"0x{edge.source_ptr:x}", "pc": src_pc, "reason": src_reason,
            "role": "pre_transform_source_ptr",
        }]
        edge_chain = sorted(
            chain + [{
                "idx": -1, "pc": f"0x{edge.boundary_pc_to:x}",
                "mnemonic": f"<declared boundary edge: {edge.transform}>",
                "reads": [f"source_ptr=0x{edge.source_ptr:x}"],
                "note": ("anchored via DECLARED boundary edge "
                         f"0x{edge.boundary_pc_from:x}→0x{edge.boundary_pc_to:x} "
                         f"(transform={edge.transform}; case-asserted, NOT a traced "
                         "writer)"),
                "source_ptr": f"0x{edge.source_ptr:x}",
                "transform": edge.transform,
            }],
            key=lambda s: s["idx"],
        )
        return _finish(ProvenanceResult(
            ProvenanceVerdict.BOUNDARY_EDGE,
            boundary_pcs=(edge.boundary_pc_to,),
            chain=edge_chain, next_watch=edge_next_watch, expected=expected,
            anchored_edge=edge,
            detail=(
                "anchored via DECLARED boundary edge "
                f"0x{edge.boundary_pc_from:x}→0x{edge.boundary_pc_to:x} "
                f"(transform={edge.transform}; case-asserted, NOT a traced writer): "
                f"the logical/transform output at 0x{edge.sink_surface:x} has no "
                f"native writer; the declared edge places it at the boundary and "
                f"surfaces the pre-transform source 0x{edge.source_ptr:x} as the next "
                "backtrace anchor. This is PROGRESS (frontier pushed forward), NOT "
                "closure — the source's producer chain must still self-prove; this "
                "verdict never feeds a close/parity gate (invariant 7). utov does NOT "
                "decode the transform (decode is the case/runner's job)."),
        ))

    # 4. OPAQUE_CALLEE / BRIDGE_BOUNDARY — no traced write produces the sink, and
    #    the data appears only AFTER a call returns (absent before). Produced by an
    #    un-traced callee/bridge; report the call boundary.
    opaque = _opaque_callee(items, sink_base, expected, write_byte_producer)
    if opaque is not None:
        boundary, first_seen_ins = opaque
        target = _resolve_call_target(boundary)
        callee_targets = (target,) if target is not None else ()
        arrow = f" → 0x{target:x}" if target is not None else " → <unresolved>"
        # Non-empty chain: the call boundary step + the first-seen observation step
        # (round-3 fix: OPAQUE_CALLEE no longer carries chain_len=0).
        opaque_chain = sorted(
            [
                _step(boundary, note="call boundary — sink produced across this call",
                      callee=(f"0x{target:x}" if target is not None else None)),
                _step(first_seen_ins,
                      note=f"first observation of sink data at 0x{sink_base:x}"),
            ],
            key=lambda s: s["idx"],
        )
        return _finish(ProvenanceResult(
            ProvenanceVerdict.OPAQUE_CALLEE, boundary_pcs=(boundary.pc,),
            callee_targets=callee_targets, chain=opaque_chain, expected=expected,
            detail=(f"no traced instruction produces the sink; the output is first "
                    f"observed (idx {first_seen_ins.idx}) only AFTER the call at "
                    f"0x{boundary.pc:x}{arrow} returns (absent before) — produced by "
                    f"an un-traced callee/bridge. Boundary idx {boundary.idx}; trace "
                    f"into the resolved callee to see the production."),
        ))

    # 5. NEEDS_OBSERVATION (streaming unprovable) — production not visible, no
    #    traced producer, and no observed call-return appearance. Be honest: we can
    #    neither confirm nor refute streaming with the captured observations.
    if not next_watch:
        for off in range(L):
            next_watch.append({
                "addr": f"0x{sink_base + off:x}", "pc": None,
                "reason": "no captured write/snapshot producer for this sink byte",
            })
    return _finish(ProvenanceResult(
        ProvenanceVerdict.NEEDS_OBSERVATION, streaming="unprovable", chain=chain,
        next_watch=next_watch, expected=expected,
        detail=("production is not visible: no traced producer, and the output was "
                "not observed appearing across a call return. Streaming could be "
                "neither confirmed nor refuted with the captured observations — "
                "capture the sink's write history and the surrounding call returns, "
                "then re-validate. This mode does not collect observations itself."),
    ))
