"""#7 — final-materialization gate (route before whole-chain composite).

Universal principle (spec_tc2_final_materialization_gate): when the runner-visible
output is produced by a SMALL final construction — a fixed header concatenated with
a bulk copy from a live buffer (``output = HDR ‖ memcpy(src)`` / a ``strb`` run off
one contiguous region) — the SHORTEST explanation of the oracle is that construction
+ the source buffer's value, NOT the upstream composite chain (SHA/table/XOR) that
may also be present in the trace. Chasing the composite chain first is what pulled
the utov-heavy path off the oracle.

This is a GENERALIZATION of oracle-anchoring (anchor sink + window, then trust
downstream): before escalating to a composite symbolic recovery, check whether the
sink is materialized by a final copy/header and, if so, ROUTE to recover the source
buffer's provenance first.

Inventory (A8①, don't rebuild): consume :func:`oracle_sink.validate_sink` (the sink
is already located/confirmed), :func:`oracle_provenance.trace_provenance` (the
producer chain), and :func:`closure_classification.classify_closure` (the verdict
layer). This module ADDS a classifier on the materialization pattern + a routing
decision, nothing more.

THE GATE ROUTES, IT DOES NOT CLOSE (cross-cutting decision #1). A detected final
copy is NEVER promoted to :attr:`ClosureLevel.ORACLE` by itself — the recovered F
(header ‖ source) still has to clear ``closure_classification`` (provenance on the
source's producer chain + multi-input live parity). A constant-``F`` window with an
unconfirmed sink (the TC2 ``F=7`` pseudo-closure) is caught by the existing
``PSEUDO_CLOSURE_TRAP``, not re-emitted as an answer.

Degenerate (A8④, always a verdict): if the materialization SOURCE buffer is itself
not observable / not seed-derivable, emit ``SOURCE_UNOBSERVABLE`` (route to a
same-execution watch on the source buffer, or BLOCK with that precise reason) —
never silently emit a constant or fall into the composite chain unannounced.

Generic — no TC2 address / runner format is baked in; the detector works over the
already-captured trace/snapshots. No runner change.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .types import Instruction

__all__ = [
    "MaterializationVerdict",
    "NextMove",
    "FinalMaterialization",
    "detect_final_materialization",
]


class MaterializationVerdict(str, enum.Enum):
    """How the located sink's bytes were materialized (A8②, enumerated shapes)."""

    FINAL_COPY = "FINAL_COPY"                  # (a) bulk memcpy(dst,src,n) into sink
    FINAL_HEADER_COPY = "FINAL_HEADER_COPY"    # (c) fixed header ‖ copy from live buf
    FINAL_BYTEMAP = "FINAL_BYTEMAP"            # (d) per-byte buf[i] = f(src[i]) map
    NO_FINAL_MATERIALIZATION = "NO_FINAL_MATERIALIZATION"  # (b-none) computed in place


class NextMove(str, enum.Enum):
    """The routing decision — the gate's deliverable (a next-move, never a closure)."""

    RECOVER_SOURCE_PROVENANCE = "recover_source_provenance"  # source observable
    WATCH_SOURCE_BUFFER = "watch_source_buffer"              # source unobservable
    FALL_THROUGH_COMPOSITE = "fall_through_composite"        # no final materialization


# The degenerate marker (A8④): a final materialization was detected but its source
# buffer is not observable in the captured trace/snapshots.
SOURCE_UNOBSERVABLE = "SOURCE_UNOBSERVABLE"


@dataclass(frozen=True)
class FinalMaterialization:
    """The materialization classification + routing decision (#7).

    NOT a closure verdict — ``next_move`` tells the consumer what to recover NEXT.
    ``source_region`` is the live buffer copied into the sink (the thing to recover);
    ``header_bytes`` the fixed prefix when the output is ``HDR ‖ copy``; ``copy_call``
    the memcpy boundary or the strb-run window. ``source_observable`` is the A8④
    self-check: False ⇒ ``next_move = watch_source_buffer`` + ``SOURCE_UNOBSERVABLE``."""

    verdict: MaterializationVerdict
    next_move: NextMove
    header_bytes: bytes | None = None
    source_region: dict[str, Any] | None = None     # {"base": int, "len": int}
    copy_call: dict[str, Any] | None = None          # {"pc": int, "kind": str}
    source_observable: bool | None = None
    source_unobservable_reason: str = ""             # SOURCE_UNOBSERVABLE detail
    watch_spec: dict[str, Any] | None = None         # same-execution watch on source
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": "final_materialization",
            "verdict": self.verdict.value,
            "next_move": self.next_move.value,
            "detail": self.detail,
        }
        if self.header_bytes is not None:
            out["header_bytes"] = self.header_bytes.hex()
        if self.source_region is not None:
            sr = self.source_region
            out["source_region"] = {
                "base": f"0x{sr['base']:x}", "len": sr["len"]}
        if self.copy_call is not None:
            cc = self.copy_call
            out["copy_call"] = {
                "pc": f"0x{cc['pc']:x}" if cc.get("pc") is not None else None,
                "kind": cc.get("kind", ""),
            }
        if self.source_observable is not None:
            out["source_observable"] = self.source_observable
        if self.source_unobservable_reason:
            out["source_unobservable_reason"] = self.source_unobservable_reason
            out["degenerate"] = SOURCE_UNOBSERVABLE
        if self.watch_spec is not None:
            out["watch_spec"] = self.watch_spec
        return out


def _sink_writers(items: list[Instruction], sink_base: int, sink_len: int):
    """The instructions whose WRITES land in the sink region, in trace order, and
    a per-byte producer map. Used to recognise the final write SEQUENCE."""
    writers: list[Instruction] = []
    seen_idx: set[int] = set()
    byte_producer: dict[int, int] = {}    # sink addr -> writing idx
    sink_hi = sink_base + sink_len
    for ins in items:
        touches = False
        for op in ins.mem:
            if op.rw == "w" and op.size > 0:
                for k in range(op.size):
                    a = op.addr + k
                    if sink_base <= a < sink_hi:
                        byte_producer[a] = ins.idx
                        touches = True
        if touches and ins.idx not in seen_idx:
            seen_idx.add(ins.idx)
            writers.append(ins)
    return writers, byte_producer


def _strb_run_source(writers: list[Instruction], sink_base: int, sink_len: int):
    """If the sink is filled by a run of single/few-byte stores each reading from a
    CONTIGUOUS source region (``buf[i] = src[i]`` / ``buf[i] = f(src[i])``), recover
    that source region. Returns ({"base", "len"}, is_bytemap) or (None, False).

    We pair each sink-writing instruction with a source READ in the SAME
    instruction (the classic load-byte / store-byte couple). When the read offsets
    track the write offsets contiguously, the reads name a source buffer."""
    pairs: list[tuple[int, int]] = []    # (sink_off, src_addr)
    transformed = False
    for ins in writers:
        w_addr = None
        for op in ins.mem:
            if op.rw == "w" and op.size > 0 and sink_base <= op.addr < sink_base + sink_len:
                w_addr = op.addr
                w_val = op.val & ((1 << (8 * op.size)) - 1)
                break
        if w_addr is None:
            continue
        r_addr = None
        r_val = None
        for op in ins.mem:
            if op.rw == "r" and op.size > 0:
                r_addr = op.addr
                r_val = op.val & ((1 << (8 * op.size)) - 1)
                break
        if r_addr is None:
            # A store with no same-instruction source read = a header / immediate
            # byte (the fixed prefix of a HDR ‖ copy). It is NOT part of the copy
            # run — skip it (the copy portion is the writers that DO read a source).
            # Only if NO writer reads a source at all does this collapse to "no copy
            # run" (caught by the empty-pairs guard below).
            continue
        pairs.append((w_addr - sink_base, r_addr))
        if r_val is not None and w_val != r_val:
            transformed = True
    if not pairs:
        return None, False
    pairs.sort()
    src_addrs = [a for _, a in pairs]
    base = min(src_addrs)
    span = max(src_addrs) - base + 1
    # Require the source reads to be (near-)contiguous: a real source buffer, not a
    # scatter of unrelated loads (which would be a composite chain, not a copy).
    if span > 2 * len(src_addrs) + 1:
        return None, False
    return {"base": base, "len": span}, transformed


def _memcpy_source(annotated_calls, sink_base, sink_len, items):
    """If a known ``memcpy``/``memmove`` produced the sink (via the #4 annotated
    call + #5 edge), recover the source region from the call's ABI. Lazily imports
    #5 to keep the gate usable without an import map. Returns {"base","len"} or
    None."""
    if not annotated_calls:
        return None
    # Find a memcpy/memmove annotation whose call wrote the sink (its dst covers it).
    from .import_map import extern_summary
    from .libc_boundary import _find_call_ins, _read_reg_at
    for ann in annotated_calls:
        sym = (ann.get("symbol") or "").split("@", 1)[0]
        summ = extern_summary(sym)
        if summ is None or summ.role_reg("src") is None:
            continue
        call_ins = _find_call_ins(items, ann["idx"])
        if call_ins is None:
            continue
        dst = _read_reg_at(call_ins, summ.role_reg("dst"))
        src = _read_reg_at(call_ins, summ.role_reg("src"))
        n = _read_reg_at(call_ins, summ.role_reg("n"))
        if dst is None or src is None or n is None:
            continue
        if dst <= sink_base and sink_base + sink_len <= dst + n:
            offset = sink_base - dst
            return {"base": src + offset, "len": sink_len, "via": sym,
                    "copy_pc": call_ins.pc}
    return None


def _region_observable(items: list[Instruction], snapshots, base: int, length: int) -> bool:
    """A8④ self-check: is the source region observable in the captured data? True
    iff every byte has a captured write/read/snapshot — otherwise the source value
    is not visible (SOURCE_UNOBSERVABLE)."""
    covered: set[int] = set()
    for ins in items:
        for op in ins.mem:
            if op.size > 0:
                for k in range(op.size):
                    a = op.addr + k
                    if base <= a < base + length:
                        covered.add(a)
    for s in snapshots:
        for k in range(len(s.data)):
            a = s.addr + k
            if base <= a < base + length:
                covered.add(a)
    return all((base + off) in covered for off in range(length))


def _watch_spec_for(base: int, length: int, reading_pc: int | None) -> dict[str, Any]:
    """A same-execution watch spec on the source buffer (reuses the watch-first-write
    / observe-point shape). Capturing it same-execution is #3 (runner) territory if
    the source isn't already in a snapshot; this names WHAT to watch."""
    return {
        "watch": "source_buffer",
        "addr": f"0x{base:x}",
        "len": length,
        "reading_pc": None if reading_pc is None else f"0x{reading_pc:x}",
        "reason": ("the final-materialization source buffer is not observable in the "
                   "captured trace/snapshots — watch this region same-execution "
                   "(#3) so its value can be recovered, then re-route"),
    }


def detect_final_materialization(
    items: Iterable[Instruction],
    *,
    sink_base: int,
    output: bytes,
    snapshots: Iterable | None = None,
    annotated_calls: Iterable[dict[str, Any]] | None = None,
    header_len_hint: int | None = None,
) -> FinalMaterialization:
    """Classify the located sink's materialization shape + ROUTE (#7).

    Args:
      items: the instruction stream (already captured).
      sink_base / output: the located, confirmed sink — its base and the observed
        output bytes (from :func:`oracle_sink.validate_sink`).
      snapshots: captured :class:`MemSnapshot` observations (source-observability).
      annotated_calls: #4's :func:`import_map.annotate_calls` output (so a
        ``memcpy`` final copy is recognised from the ABI, not just from strb runs).
      header_len_hint: optional fixed-prefix length; when omitted the header is
        inferred as the leading sink bytes NOT covered by the recovered source copy.

    Returns a :class:`FinalMaterialization`. ROUTING (the gate):
      * ``FINAL_*`` with an OBSERVABLE source ⇒ ``recover_source_provenance``
        (point ``trace_provenance`` / a recovery CVD at the SOURCE buffer).
      * ``FINAL_*`` with an UNOBSERVABLE source ⇒ ``watch_source_buffer`` +
        ``SOURCE_UNOBSERVABLE`` (emit a same-execution watch spec) — A8④.
      * ``NO_FINAL_MATERIALIZATION`` ⇒ ``fall_through_composite`` (unchanged path).

    NEVER promotes to ORACLE — the recovered (header ‖ source) F must still clear
    ``closure_classification`` (decision #1)."""
    items = list(items)
    snapshots = list(snapshots or [])
    annotated_calls = list(annotated_calls or [])
    output = bytes(output)
    out_len = len(output)
    if out_len == 0:
        raise ValueError("output must be non-empty")

    writers, byte_producer = _sink_writers(items, sink_base, out_len)

    # No traced write touches the sink at all → there is no final WRITE construction
    # we can see; fall through to the composite path (it is computed elsewhere /
    # in place from the gate's point of view). NO_FINAL_MATERIALIZATION.
    sink_written = any((sink_base + off) in byte_producer for off in range(out_len))

    # 1. memcpy/memmove final copy (recognised from #4 annotations).
    mc = _memcpy_source(annotated_calls, sink_base, out_len, items)
    if mc is not None:
        return _route(
            items, snapshots,
            verdict=MaterializationVerdict.FINAL_COPY,
            source_region={"base": mc["base"], "len": mc["len"]},
            header_bytes=None,
            copy_call={"pc": mc.get("copy_pc"), "kind": f"memcpy:{mc.get('via')}"},
            detail=(f"sink at 0x{sink_base:x} is a bulk {mc.get('via')} copy from "
                    f"0x{mc['base']:x} — recover the SOURCE buffer's provenance, not "
                    "the upstream composite chain"),
        )

    if not sink_written:
        return FinalMaterialization(
            verdict=MaterializationVerdict.NO_FINAL_MATERIALIZATION,
            next_move=NextMove.FALL_THROUGH_COMPOSITE,
            detail=("no traced write produces the sink and no known copy call targets "
                    "it — no final materialization visible; the composite/recovery "
                    "path runs unchanged (byte-for-byte)"),
        )

    # 2. strb/str run reading a contiguous source region (header ‖ copy / bytemap).
    src_region, is_bytemap = _strb_run_source(writers, sink_base, out_len)
    if src_region is not None:
        # Infer the header: leading sink bytes whose writers had NO source read are
        # the fixed prefix. The explicit hint wins; else infer from the leading run
        # of sink bytes with no paired source read.
        header_len = header_len_hint
        if header_len is None:
            # Header = the leading run of sink bytes that have NO paired source read.
            header_len = _infer_header_len(writers, sink_base, out_len)
        header_bytes = output[:header_len] if header_len and header_len > 0 else None
        verdict = (MaterializationVerdict.FINAL_HEADER_COPY if header_bytes
                   else (MaterializationVerdict.FINAL_BYTEMAP if is_bytemap
                         else MaterializationVerdict.FINAL_COPY))
        copy_pc = writers[-1].pc if writers else None
        kind = "bytemap" if is_bytemap else "strb_run"
        return _route(
            items, snapshots,
            verdict=verdict,
            source_region=src_region,
            header_bytes=header_bytes,
            copy_call={"pc": copy_pc, "kind": kind},
            detail=(f"sink at 0x{sink_base:x} is materialized by a {kind} from a "
                    f"contiguous source region 0x{src_region['base']:x} "
                    f"(len {src_region['len']})"
                    + (f" with a {header_len}-byte fixed header" if header_bytes
                       else "")
                    + " — recover the SOURCE buffer's provenance first"),
        )

    # 3. The sink is written but not by a recognisable copy/strb-run from a single
    #    source region → it is computed in place (or via a scattered composite). Fall
    #    through to the composite path unchanged.
    return FinalMaterialization(
        verdict=MaterializationVerdict.NO_FINAL_MATERIALIZATION,
        next_move=NextMove.FALL_THROUGH_COMPOSITE,
        detail=("the sink is written in place / from a scattered (non-contiguous) "
                "set of sources — no single source buffer to recover; the composite "
                "path runs unchanged (byte-for-byte)"),
    )


def _infer_header_len(writers, sink_base: int, out_len: int) -> int:
    """The leading run of sink bytes whose writing instruction had NO source read
    (a fixed/immediate header), measured from sink_base. Stops at the first sink
    byte produced by a copy (a write paired with a source read)."""
    # Map sink offset -> whether its writer read a source.
    off_has_source: dict[int, bool] = {}
    for ins in writers:
        has_read = any(op.rw == "r" and op.size > 0 for op in ins.mem)
        for op in ins.mem:
            if op.rw == "w" and op.size > 0:
                for k in range(op.size):
                    a = op.addr + k
                    if sink_base <= a < sink_base + out_len:
                        off = a - sink_base
                        # last writer wins, mirroring last-write reconstruction
                        off_has_source[off] = has_read
    header = 0
    for off in range(out_len):
        if off_has_source.get(off, False):
            break
        if off in off_has_source:
            header += 1
        else:
            break
    return header


def _route(
    items, snapshots, *,
    verdict: MaterializationVerdict,
    source_region: dict[str, Any],
    header_bytes: bytes | None,
    copy_call: dict[str, Any],
    detail: str,
) -> FinalMaterialization:
    """Apply the routing rule + the A8④ source-observability self-check."""
    base, length = source_region["base"], source_region["len"]
    observable = _region_observable(items, snapshots, base, length)
    if observable:
        return FinalMaterialization(
            verdict=verdict,
            next_move=NextMove.RECOVER_SOURCE_PROVENANCE,
            header_bytes=header_bytes,
            source_region={"base": base, "len": length},
            copy_call=copy_call,
            source_observable=True,
            detail=detail,
        )
    # A8④ degenerate: source not observable → watch it same-execution, do NOT fall
    # silently into the composite chain.
    reading_pc = copy_call.get("pc")
    return FinalMaterialization(
        verdict=verdict,
        next_move=NextMove.WATCH_SOURCE_BUFFER,
        header_bytes=header_bytes,
        source_region={"base": base, "len": length},
        copy_call=copy_call,
        source_observable=False,
        source_unobservable_reason=(
            f"the {verdict.value} source region [0x{base:x},+{length}) is not "
            "observable in the captured trace/snapshots — its value cannot be "
            "recovered without a same-execution watch (#3); routing to "
            "watch_source_buffer rather than falling into the composite chain"),
        watch_spec=_watch_spec_for(base, length, reading_pc),
        detail=detail + " — but the source buffer is NOT observable (SOURCE_UNOBSERVABLE)",
    )
