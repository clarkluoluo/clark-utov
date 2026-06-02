"""S4: backward data-flow slice (PLAN §3 step S4).

Pick sink instructions (default: every memory-store + every register-write
near the function exit). BFS backward in the S3 data-flow graph; mark every
ancestor as kept. Everything not marked is "noise" and can be discarded.

For our trace format the default sinks are:
  - The final SET of registers written by the last block.
  - Memory writes ('str' family) — when our trace records mem_ops these
    feed too; current UnidbgTextTraceReader doesn't record mem_ops so we
    work from registers only.

Caller can override sinks via ctx['sinks'] = list of instruction indices.

Output: stage_outputs/s4_slice.jsonl
  - one row per surviving instruction idx, with which sink pulled it in.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from ..types import Instruction
from .s3_triton import DfgNode, read_dfg

CODE_VERSION = "s4-v2"

# AArch64 ABI: x0..x7 carry args & return values. A function's OUTPUT is the
# last write to these regs before ret. We treat the LAST write to each of
# them as a sink — that anchors the slice in "what the function returned /
# computed for the caller".
_OUTPUT_REGS = ("x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7")


def default_sinks(items: Iterable[Instruction]) -> list[int]:
    """Pick sinks intelligently: last write to each ABI return register +
    any instruction that wrote to memory before return.

    Better than "just the last instruction" because:
      - ARM64 returns values through x0..x7
      - A digest write happens via str/stp into the caller-supplied buffer
        and the sink should capture those memory writes too
    """
    items_list = list(items) if not isinstance(items, list) else items
    if not items_list:
        return []
    last_writer: dict[str, int] = {}
    for ins in items_list:
        for r in ins.regs_write:
            last_writer[r] = ins.idx
    sinks: list[int] = [last_writer[r] for r in _OUTPUT_REGS if r in last_writer]
    # Add the final instruction's writes too (any reg) — covers epilog
    last = items_list[-1]
    if last.regs_write and last.idx not in sinks:
        sinks.append(last.idx)
    return sinks or [items_list[-1].idx]


def _parse_window_endpoint(v) -> int | None:
    """Window endpoints in session.json may be strings like ``"0x32302c"``
    or already-decoded ints. Tolerate both; reject anything else."""
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        try:
            return int(v, 16) if v.lower().startswith("0x") else int(v, 16)
        except ValueError:
            return None
    return None


def _idxs_in_windows(
    items: Iterable[Instruction],
    windows: Iterable,
) -> list[int]:
    """Return trace indices of every instruction whose PC falls in any
    of ``windows``. Each window is ``[start, end]`` (end exclusive)."""
    bands: list[tuple[int, int]] = []
    for w in windows:
        if not isinstance(w, (list, tuple)) or len(w) < 2:
            continue
        s = _parse_window_endpoint(w[0])
        e = _parse_window_endpoint(w[1])
        if s is None or e is None:
            continue
        bands.append((s, e))
    if not bands:
        return []
    out: list[int] = []
    for ins in items:
        if any(s <= ins.pc < e for s, e in bands):
            out.append(ins.idx)
    return out


def outparam_sinks(items: Iterable[Instruction], *, min_span_bytes: int = 2) -> list[int]:
    """Detect the output buffer from memory writes and return the indices of the
    stores that fill it.

    Many targets write their result (a digest / cipher block) byte- or
    word-wise into a caller-supplied buffer via a run of `str`s to adjacent
    addresses. `default_sinks` (ABI return regs + last instruction) never picks
    these up. Here we cluster every memory WRITE by address adjacency and treat
    the largest contiguous cluster (spanning >= ``min_span_bytes``) as the
    output buffer; the instructions that wrote it are the sinks.

    Needs the trace to record ``mem`` ops (writes). Empty when it doesn't —
    callers fall back to :func:`default_sinks`.
    """
    items_list = list(items) if not isinstance(items, list) else items
    writes: list[tuple[int, int, int]] = []   # (idx, addr, size)
    for ins in items_list:
        for op in ins.mem:
            if op.rw == "w" and op.size > 0:
                writes.append((ins.idx, op.addr, op.size))
    if not writes:
        return []
    writes.sort(key=lambda w: (w[1], w[0]))
    clusters: list[list[tuple[int, int, int]]] = [[writes[0]]]
    for w in writes[1:]:
        prev = clusters[-1][-1]
        if w[1] <= prev[1] + prev[2]:        # contiguous or overlapping
            clusters[-1].append(w)
        else:
            clusters.append([w])
    # span = highest end - lowest start within the cluster
    def _span(cl: list[tuple[int, int, int]]) -> int:
        return (cl[-1][1] + cl[-1][2]) - cl[0][1]
    best = max(clusters, key=_span)
    if _span(best) < min_span_bytes:
        return []
    out: list[int] = []
    for idx, _, _ in best:
        if idx not in out:
            out.append(idx)
    return sorted(out)


def sinks_with_hints(
    items: Iterable[Instruction],
    extra_idxs: Iterable[int] | None = None,
    *,
    include_outparam: bool = True,
) -> list[int]:
    """default_sinks plus caller-supplied "interesting" indices (e.g. fingerprint
    hit sites from S1.5 — they're guaranteed-relevant to the algorithm), plus
    the auto-detected output-buffer stores (:func:`outparam_sinks`).

    ``default_sinks`` is always included as the fallback floor — the new
    memory-write sinks are additive, never a replacement.
    """
    s: set[int] = set(default_sinks(items))
    if include_outparam:
        s |= set(outparam_sinks(items))
    if extra_idxs:
        s |= set(int(x) for x in extra_idxs)
    return sorted(s)


def slice_backward(dfg: list[DfgNode], sinks: list[int]) -> dict[int, str]:
    """Return mapping idx → reason (which sink first reached this node)."""
    by_idx = {n.idx: n for n in dfg}
    keep: dict[int, str] = {}
    for sink in sinks:
        if sink not in by_idx:
            continue
        reason = f"sink:{sink}"
        stack = [sink]
        while stack:
            cur = stack.pop()
            if cur in keep:
                continue
            keep[cur] = reason
            node = by_idx.get(cur)
            if node is None:
                continue
            for producer in node.reg_deps.values():
                if producer is not None and producer not in keep:
                    stack.append(producer)
            # Follow concrete memory dependencies too: a load's value source is
            # the store that wrote those bytes — reachable only via mem_deps
            # (the producing store writes no register).
            for producer in node.mem_deps:
                if producer not in keep:
                    stack.append(producer)
    return keep


def run(ctx) -> dict:
    items = ctx["items"]
    work = ctx["work"]
    dfg = read_dfg(work)
    # Feedback context: S1.5 stashed fingerprint hit instr indices into the
    # session under 'fingerprint_anchor_idxs'. Pull them in as extra sinks.
    session = ctx.get("session") or {}
    extra = list(session.get("fingerprint_anchor_idxs") or [])
    # capability_request.md §P0-2: any instruction whose PC falls inside
    # an `extra_trace_windows` band is treated as a sink hint so the
    # backward slice reaches data flowing through main-VMP bands like
    # 0x32302c..0x325708. The band is stored as a list of `[start_hex,
    # end_hex]` pairs in session.json.
    windows = session.get("extra_trace_windows") or []
    band_idxs = _idxs_in_windows(items, windows)
    if band_idxs:
        extra.extend(band_idxs)
    sinks = ctx.get("sinks") or sinks_with_hints(items, extra)

    # Oracle sink gate (opt-in: ONLY when the caller supplies expected_output).
    # Back-compat boundary: no expected_output -> this block is skipped and s4
    # behaves byte-for-byte as before. With expected_output the verdict is
    # AUTHORITATIVE (it redirects / blocks; never a mere warning) — the caller
    # opted into validation, so we honour the result.
    sink_validation = None
    sink_redirect = None
    expected = ctx.get("expected_output")
    if expected is not None:
        from ..oracle_sink import apply_sink_gate  # SinkGateError propagates out
        rec_path = work.root / "stage_outputs" / "s4_sink_validation.json"
        sinks, sv, sink_redirect = apply_sink_gate(
            items, sinks, bytes(expected),
            candidate_base=ctx.get("candidate_sink_base"),
            snapshots=ctx.get("snapshots"),
            record_to=rec_path,
        )
        sink_validation = sv.to_dict()

    kept = slice_backward(dfg, sinks)

    out_path: Path = work.root / "stage_outputs" / "s4_slice.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for idx in sorted(kept):
            f.write(json.dumps({"idx": idx, "reason": kept[idx]}) + "\n")

    work.mark_stage_done("s4", CODE_VERSION)
    summary = {
        "stage": "s4",
        "sinks": sinks,
        "total_nodes": len(dfg),
        "kept_nodes": len(kept),
        "out": str(out_path),
    }
    if sink_validation is not None:
        summary["sink_validation"] = sink_validation   # recorded verdict
    if sink_redirect is not None:
        summary["sink_redirect"] = sink_redirect        # WRONG_SINK auto-fix
    return summary


def read_slice(work) -> dict[int, str]:
    path: Path = work.root / "stage_outputs" / "s4_slice.jsonl"
    out: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            out[o["idx"]] = o["reason"]
    return out
