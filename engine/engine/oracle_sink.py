"""Oracle sink-validator — verify/locate the real output sink before slicing.

Motivation: in the field, the sink a heuristic phase picks (some store PC) was a
SCRATCH region; s4 backward-slice + s6 taint + diffusion all ran on it for a
long time before a manual oracle byte-compare revealed the error. This gate
short-circuits that: given the EXPECTED output bytes, reconstruct memory by
last-write-wins and confirm/locate where those bytes actually land, or classify
the case as an observation-capability gap (the output was never captured) rather
than letting every downstream pass inherit a wrong sink.

Generic and fully parameterised — `expected_output` and any candidate sink come
from the caller; no target address is hardcoded. Reuses `Instruction.mem`
(rw/addr/val/size) only; no new dependency.

Verdicts:
  - SINK_CONFIRMED        the candidate (or, with no candidate, an auto-located
                          region) reconstructs to expected_output.
  - WRONG_SINK            a candidate was given and is wrong, but the real sink
                          IS in the trace (located elsewhere).
  - OUTPUT_NOT_OBSERVABLE no region anywhere (writes, or read/snapshot-observed
                          memory) reconstructs expected_output → the final output
                          was not captured. An observation-capability gap, NOT an
                          analysis problem. Reports the longest partial match to
                          guide the next step (widen the trace window / snapshot
                          the output region).

Non-goals (left for a later provenance mode): searching for an ENCODED form of
expected_output (base64 etc.), and reverse producer-chain tracing from the
output. OUTPUT_NOT_OBSERVABLE's diagnosis is written to guide those next.
"""

from __future__ import annotations

import enum
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import Instruction


class SinkVerdict(str, enum.Enum):
    SINK_CONFIRMED = "SINK_CONFIRMED"
    WRONG_SINK = "WRONG_SINK"
    OUTPUT_NOT_OBSERVABLE = "OUTPUT_NOT_OBSERVABLE"


@dataclass(frozen=True)
class SinkValidation:
    verdict: SinkVerdict
    base: int | None = None                 # base addr of the confirmed/located region
    writer_pcs: tuple[int, ...] = ()         # PCs that wrote/established the region bytes
    located_via: str = ""                    # "write" | "read" | "snapshot" | ""
    scanned_sources: tuple[str, ...] = ()    # which sources this scan covered
    candidate_base: int | None = None        # the candidate that was checked, if any
    first_diff_offset: int = -1              # candidate mismatch: first differing byte
    reconstructed: bytes | None = None       # candidate region reconstruction (mismatch)
    expected: bytes = b""
    longest_partial: dict = field(default_factory=dict)  # not-observable: best partial
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "base": None if self.base is None else f"0x{self.base:x}",
            "writer_pcs": [f"0x{p:x}" for p in self.writer_pcs],
            "located_via": self.located_via,
            "scanned_sources": list(self.scanned_sources),
            "candidate_base": None if self.candidate_base is None else f"0x{self.candidate_base:x}",
            "first_diff_offset": self.first_diff_offset,
            "reconstructed": None if self.reconstructed is None else self.reconstructed.hex(),
            "expected": self.expected.hex(),
            "longest_partial": dict(self.longest_partial),
            "detail": self.detail,
        }


def _trace_maps(items: list[Instruction]):
    """Last-op-wins byte maps from the instruction stream, in idx order.

    write_map: addr -> (byte, pc) from WRITE ops only (the sink IS a write).
    read_map:  addr -> (byte, pc) from READ ops (reads reveal memory contents).
    """
    write_map: dict[int, tuple[int, int]] = {}
    read_map: dict[int, tuple[int, int]] = {}
    for ins in items:
        for op in ins.mem:
            if op.size <= 0:
                continue
            for k in range(op.size):
                addr = op.addr + k
                bval = (op.val >> (8 * k)) & 0xFF
                if op.rw == "w":
                    write_map[addr] = (bval, ins.pc)
                elif op.rw == "r":
                    read_map[addr] = (bval, ins.pc)
    return write_map, read_map


def _snapshot_map(snapshots):
    """Byte map from canonical MemSnapshot observations. pc is None (a snapshot
    has no executing instruction); reconstruction uses 0 as a placeholder pc so
    writer_pcs stays a tuple of ints — callers key on located_via='snapshot'."""
    snap_map: dict[int, tuple[int, int]] = {}
    for snap in snapshots:
        data = bytes(snap.data)
        for k, b in enumerate(data):
            snap_map[snap.addr + k] = (b, 0)
    return snap_map


def _search_sources(sources, expected: bytes, exclude=None):
    """Return (base, writer_pcs, source_name) of the first source whose region
    reconstructs to expected. Sources are tried in the given priority order.

    ``exclude`` is the (source_name, base) pair already tried and rejected in
    Mode 1 (the candidate WRITE) — only that exact source+base is skipped, so a
    DIFFERENT observation source (snapshot/read) that reconstructs the SAME base
    is still allowed to confirm the sink. (The earlier "exclude_base" semantics
    banned every source sharing that base, wrongly suppressing a snapshot/read
    that fully reconstructed the output at the candidate base.)"""
    for name, bmap in sources:
        for base, pcs in _locate(bmap, expected):
            if exclude is None or (name, base) != exclude:
                return base, pcs, name
    return None


def _reconstruct(byte_map: dict[int, tuple[int, int]], base: int, length: int):
    """Last-write-wins reconstruction of [base, base+length). Returns
    (bytes, writer_pcs) or (None, ()) if any byte is uncovered."""
    out = bytearray(length)
    pcs: list[int] = []
    for off in range(length):
        e = byte_map.get(base + off)
        if e is None:
            return None, ()
        out[off] = e[0]
        if e[1] not in pcs:
            pcs.append(e[1])
    return bytes(out), tuple(pcs)


def _first_diff(a: bytes, b: bytes) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


def _locate(byte_map, expected: bytes):
    """All bases where [base, base+len) fully reconstructs to expected."""
    L = len(expected)
    hits = []
    for base in sorted(set(byte_map)):
        rec, pcs = _reconstruct(byte_map, base, L)
        if rec is not None and rec == expected:
            hits.append((base, pcs))
    return hits


def _best_partial(byte_map, expected: bytes) -> dict:
    """Region (base) with the most byte-positions matching expected."""
    L = len(expected)
    best = {"base": None, "match_count": 0, "length": L}
    for base in set(byte_map):
        m = sum(
            1 for off in range(L)
            if byte_map.get(base + off) is not None and byte_map[base + off][0] == expected[off]
        )
        if m > best["match_count"]:
            best = {"base": f"0x{base:x}", "match_count": m, "length": L}
    return best


def validate_sink(
    items: Iterable[Instruction],
    expected_output: bytes,
    *,
    candidate_idxs: Iterable[int] | None = None,
    candidate_base: int | None = None,
    snapshots: Iterable | None = None,
) -> SinkValidation:
    """Validate / locate the real output sink against expected_output.

    Scans three sources, in priority order: WRITES (the sink is a write), then
    SNAPSHOTS (canonical MemSnapshot observations — the output may live only
    here), then READS. ``scanned_sources`` on the result records which sources
    were actually covered, so an OUTPUT_NOT_OBSERVABLE verdict can be read
    correctly (it means "not in the scanned sources", and explicitly flags when
    no snapshots were provided).
    """
    items = list(items)
    snapshots = list(snapshots or [])
    expected = bytes(expected_output)
    if not expected:
        raise ValueError("expected_output must be non-empty")
    L = len(expected)
    write_map, read_map = _trace_maps(items)
    snap_map = _snapshot_map(snapshots)
    snapshot_present = bool(snapshots)
    scanned = ("writes", "reads") + (("snapshots",) if snapshot_present else ())

    # Search priority after a candidate write fails: real write sink, then the
    # canonical snapshot, then an incidental read. Empty maps are skipped.
    fallback = [("write", write_map)]
    if snap_map:
        fallback.append(("snapshot", snap_map))
    fallback.append(("read", read_map))

    cand_base = candidate_base
    if cand_base is None and candidate_idxs is not None:
        cand_set = {int(i) for i in candidate_idxs}
        cand_addrs = [
            op.addr for ins in items if ins.idx in cand_set
            for op in ins.mem if op.rw == "w" and op.size > 0
        ]
        cand_base = min(cand_addrs) if cand_addrs else None

    # --- Mode 1: validate the candidate write ---
    if cand_base is not None:
        rec, pcs = _reconstruct(write_map, cand_base, L)
        if rec is not None and rec == expected:
            return SinkValidation(
                SinkVerdict.SINK_CONFIRMED, base=cand_base, writer_pcs=pcs,
                located_via="write", scanned_sources=scanned,
                candidate_base=cand_base, expected=expected,
                detail="candidate sink reconstructs to expected output",
            )
        fd = _first_diff(rec, expected) if rec is not None else 0
        # Only the candidate WRITE at cand_base was just tried and rejected —
        # exclude exactly ("write", cand_base), NOT every source sharing that
        # base. A snapshot/read that fully reconstructs expected AT cand_base is
        # an independent observation confirming the sink (the traced write was
        # scratch/partial; the output materialized where the snapshot saw it).
        hit = _search_sources(fallback, expected, exclude=("write", cand_base))
        if hit:
            base, hpcs, via = hit
            if base == cand_base:
                # A different observation source (snapshot/read) fully
                # reconstructs expected AT the candidate base: the output DID
                # materialize there; the traced write just wasn't the final
                # write. This is the sink, confirmed via that source.
                return SinkValidation(
                    SinkVerdict.SINK_CONFIRMED, base=base, writer_pcs=hpcs,
                    located_via=via, scanned_sources=scanned,
                    candidate_base=cand_base, expected=expected,
                    detail=(f"candidate base 0x{base:x} confirmed via {via} "
                            f"(OBSERVED memory) — the output materialized at the "
                            f"candidate base; the traced write was not the final "
                            f"write but the snapshot/read observed expected output "
                            f"there"),
                )
            where = ("real sink" if via == "write"
                     else f"OBSERVED memory ({via}, not a write) — materialized "
                          f"outside traced writes; treat it as the sink or widen "
                          f"extra_trace_windows")
            return SinkValidation(
                SinkVerdict.WRONG_SINK, base=base, writer_pcs=hpcs,
                located_via=via, scanned_sources=scanned, candidate_base=cand_base,
                first_diff_offset=fd, reconstructed=rec, expected=expected,
                detail=(f"candidate 0x{cand_base:x} is not the sink (byte {fd} "
                        f"differs); expected output found at 0x{base:x} via {where}"),
            )
        return _not_observable(write_map, read_map, snap_map, expected,
                               snapshot_present, scanned, cand_base, rec, fd)

    # --- Mode 2: no candidate — auto-locate ---
    hit = _search_sources(fallback, expected)
    if hit:
        base, hpcs, via = hit
        if via == "write":
            detail = f"auto-located sink: writes at 0x{base:x} reconstruct expected output"
        else:
            detail = (f"expected output found in OBSERVED memory at 0x{base:x} "
                      f"({via}, not a write) — materialized outside traced writes")
        return SinkValidation(
            SinkVerdict.SINK_CONFIRMED, base=base, writer_pcs=hpcs,
            located_via=via, scanned_sources=scanned, expected=expected, detail=detail,
        )
    return _not_observable(write_map, read_map, snap_map, expected,
                           snapshot_present, scanned, None, None, -1)


class SinkGateError(RuntimeError):
    """Raised by :func:`apply_sink_gate` when the verdict is
    OUTPUT_NOT_OBSERVABLE — reverse-slicing an uncaptured sink is meaningless,
    so the gate blocks rather than letting downstream passes run on nothing.
    Carries the :class:`SinkValidation` for the caller / audit."""

    def __init__(self, validation: "SinkValidation", recapture_directive: dict | None = None):
        self.validation = validation
        # When :func:`apply_sink_gate` was given a ``value_name`` it auto-derives a
        # register-relative recapture directive for the unobserved buffer and
        # attaches it here, so the blocking caller / terminal evidence sees
        # "re-collect THIS" without hand-searching. None when not derived.
        self.recapture_directive = recapture_directive
        super().__init__(validation.detail)


def apply_sink_gate(
    items: Iterable[Instruction],
    sinks: list[int],
    expected_output: bytes,
    *,
    candidate_base: int | None = None,
    snapshots: Iterable | None = None,
    record_to: Path | None = None,
    value_name: str | None = None,
) -> tuple[list[int], SinkValidation, dict | None]:
    """The verdict -> action policy, in one auditable place.

    Runs :func:`validate_sink`, records the verdict (to ``record_to`` if given,
    BEFORE acting, so the verdict is on record even when the gate blocks), then:

      - SINK_CONFIRMED         -> sinks unchanged.
      - WRONG_SINK             -> sinks REDIRECTED to the instructions that wrote
                                  the validator-located real sink; the redirect
                                  is returned for the caller to record. Not a
                                  warning — the input to s4 is corrected.
      - OUTPUT_NOT_OBSERVABLE  -> auto-derive a register-relative recapture
                                  directive for the unobserved buffer (when
                                  ``value_name`` is given) so the gap-map / the
                                  raised error ships "re-collect THIS"; then
                                  raise :class:`SinkGateError` (block; do not
                                  slice).

    ``value_name`` is OPTIONAL and purely additive: when None the verdict,
    recorded JSON, and raised error are byte-for-byte what they were before
    (invariant 7). When given, the unobserved-output path enriches both with a
    ``recapture_directive`` derived from real dataflow (invariant 8 — degrades
    explicitly, never fabricates an address).

    Returns ``(effective_sinks, validation, redirect_or_None)``. The caller
    opted into validation by supplying expected_output, so the verdict is
    authoritative — this never degrades to "just a warning".
    """
    items = list(items)
    sv = validate_sink(items, bytes(expected_output),
                       candidate_idxs=sinks, candidate_base=candidate_base,
                       snapshots=snapshots)

    directive_dict: dict | None = None
    directive_obj = None
    if sv.verdict is SinkVerdict.OUTPUT_NOT_OBSERVABLE and value_name is not None:
        # Auto-orchestrate: NEEDS_OBSERVATION on the target -> reg-relative
        # recapture directive. Local import avoids an import cycle
        # (recapture_target imports oracle_sink).
        from .recapture_target import derive_recapture_directive
        directive_obj = derive_recapture_directive(
            items, bytes(expected_output), value_name,
            sink_validation=sv, snapshots=list(snapshots) if snapshots else None)
        directive_dict = directive_obj.to_dict()

    if record_to is not None:
        record_to.parent.mkdir(parents=True, exist_ok=True)
        payload = sv.to_dict()
        if directive_dict is not None:
            payload["recapture_directive"] = directive_dict
        record_to.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    if sv.verdict is SinkVerdict.OUTPUT_NOT_OBSERVABLE:
        raise SinkGateError(sv, recapture_directive=directive_dict)

    redirect: dict | None = None
    effective = list(sinks)
    if sv.verdict is SinkVerdict.WRONG_SINK and sv.writer_pcs:
        pc_set = set(sv.writer_pcs)
        real_idxs = sorted({ins.idx for ins in items if ins.pc in pc_set})
        if real_idxs:
            redirect = {
                "from": list(sinks),
                "to": real_idxs,
                "located_base": None if sv.base is None else f"0x{sv.base:x}",
                "reason": sv.detail,
            }
            effective = real_idxs
    return effective, sv, redirect


def _not_observable(write_map, read_map, snap_map, expected, snapshot_present,
                    scanned, cand_base, rec, fd) -> SinkValidation:
    # widest partial across every scanned source guides the next step
    union: dict[int, tuple[int, int]] = {}
    for m in (read_map, snap_map, write_map):
        union.update(m)
    partial = _best_partial(union, expected)
    snap_note = (
        "" if snapshot_present else
        " NOTE: no snapshot observations were provided to this scan — this verdict "
        "means 'not present in the scanned sources (writes/reads)', NOT 'unobservable "
        "by any means'. If the output materialises in a region the runner only "
        "snapshots, have the adapter emit that snapshot in the canonical MemSnapshot "
        "shape (contracts/runner_interface.md §3.7) and re-validate."
    )
    return SinkValidation(
        SinkVerdict.OUTPUT_NOT_OBSERVABLE,
        scanned_sources=scanned, candidate_base=cand_base, first_diff_offset=fd,
        reconstructed=rec, expected=expected, longest_partial=partial,
        detail=("no region in the scanned sources reconstructs the expected output "
                "— OBSERVATION CAPABILITY GAP: the final output was not captured in "
                "the sources scanned. This is not an analysis problem; the output "
                "likely materializes outside the traced window. Next: widen "
                "extra_trace_windows, or have the adapter emit a snapshot of the "
                f"output buffer. Longest partial match: {partial}." + snap_note),
    )
