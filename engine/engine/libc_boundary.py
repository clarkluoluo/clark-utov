"""#5 — auto-synthesize a libc BoundaryEdge (sink → source) from the ABI.

Depends on #4 (:mod:`engine.import_map`): you must know a call is ``memcpy`` and
know its ABI before you can map a sink region back to a source buffer.

Universal principle (spec_tc2_libc_boundary_autosynth): raw provenance stops when
the producer writes outside the bundled trace — a libc ``memcpy(dst, src, n)``
whose copy happens in unbundled code. Today the agent hand-declares the boundary
edge :data:`oracle_provenance.BoundaryEdge` that ``trace_provenance`` already
accepts. For a KNOWN call with a KNOWN ABI, utov can synthesize the edge
"output sink ⊆ ``dst`` ⇒ source = ``src[..]``" itself, so the backtrace continues
into the source buffer instead of stalling / asking the agent.

Inventory (A8①, don't rebuild): the :class:`BoundaryEdge` concept + the
``trace_provenance(..., boundary_edge=)`` consumer already exist — this only
CONSTRUCTS the edge. ABI mapping comes from #4's ``ExternSummary.abi_args``; the
call-site identification comes from #4's :func:`import_map.annotate_calls`.

Degenerate (A8④, no silent wrong edge): unknown call / no import-map hit / ABI args
not concretely available at the call site (``n`` symbolic, ``dst`` unresolvable) ⇒
do NOT fabricate an edge. Return :class:`BoundaryEdgeUnresolved` naming exactly what
is missing, and fall back to the agent-declared edge. A wrong silent edge corrupts
provenance — forbidden.

Generic — any known external whose summary maps a destination region to a source
region (``memcpy``/``memmove``/``memset``/byte-map copies), not just TC2's memcpy.
No TC2 address is baked in.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .import_map import ExternSummary, ImportMap, extern_summary
from .oracle_provenance import BoundaryEdge, _is_call, _resolve_call_target

__all__ = [
    "BoundaryEdgeUnresolved",
    "synthesize_boundary_edge",
]


# The edge KIND, recorded on the synthesized edge's decode_meta for the audit so an
# export shows WHY the edge exists (a COPY edge points at a source buffer; a CONST
# edge points at a constant fill).
EDGE_KIND_COPY = "COPY"
EDGE_KIND_CONST = "CONST"


@dataclass(frozen=True)
class BoundaryEdgeUnresolved:
    """The honest "cannot synthesize" verdict (A8④) — NOT a fabricated edge.

    ``missing`` names exactly what the agent must supply (e.g. ``["src_concrete",
    "n_concrete"]``) so it knows what to add (a same-execution watch on the source
    pointer = #3 territory), rather than getting a silent wrong edge."""

    symbol: str
    reason: str
    missing: tuple[str, ...] = ()
    call_pc: int | None = None
    detail: str = ""

    # Sentinel a caller can string-match on the structured result.
    verdict: str = "BOUNDARY_EDGE_UNRESOLVED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "symbol": self.symbol,
            "reason": self.reason,
            "missing": list(self.missing),
            "call_pc": None if self.call_pc is None else f"0x{self.call_pc:x}",
            "detail": self.detail,
        }


def _read_reg_at(ins: Any, reg: str | None) -> int | None:
    """The CONCRETE value of ``reg`` at the call instruction, from the trace.

    A call's arg registers are READ at the call site, so ``regs_read`` carries the
    concrete dst/src/n. ``w<N>`` aliases ``x<N>``'s low 32 bits — accept either
    spelling. Returns None when the value is not present in the trace (the
    degenerate "args not concretely available" path — never a guess)."""
    if reg is None:
        return None
    if reg in ins.regs_read:
        return ins.regs_read[reg]
    # x<->w aliasing: try the sibling spelling.
    if reg.startswith("x"):
        w = "w" + reg[1:]
        if w in ins.regs_read:
            return ins.regs_read[w] & 0xFFFF_FFFF
    elif reg.startswith("w"):
        x = "x" + reg[1:]
        if x in ins.regs_read:
            return ins.regs_read[x] & 0xFFFF_FFFF
    return None


def _find_call_ins(trace: list[Any], call_site: int) -> Any | None:
    """Locate the call instruction by idx (preferred — unambiguous) or by PC.

    ``call_site`` may be a trace ``idx`` or a ``pc``. We try idx first (a single
    record), then fall back to the LAST call instruction at that PC (the most
    recent execution of the site)."""
    by_idx = {ins.idx: ins for ins in trace}
    if call_site in by_idx and _is_call(by_idx[call_site].mnemonic):
        return by_idx[call_site]
    hit = None
    for ins in trace:
        if ins.pc == call_site and _is_call(ins.mnemonic):
            hit = ins  # keep last
    return hit


def synthesize_boundary_edge(
    trace: Iterable[Any],
    call_site: int,
    sink_region: tuple[int, int],
    import_map: ImportMap,
    *,
    summary: ExternSummary | None = None,
    boundary_edge: BoundaryEdge | dict | None = None,
) -> BoundaryEdge | BoundaryEdgeUnresolved:
    """Auto-build the sink→source :class:`BoundaryEdge` for a known libc call (#5).

    Args:
      trace: the instruction stream (the call's arg regs are READ at the call site).
      call_site: the call instruction's trace ``idx`` (preferred) or ``pc``.
      sink_region: ``(base, len)`` — the output bytes whose provenance dead-ends at
        this call (must be ⊆ the call's dst region for a COPY edge).
      import_map: #4's resolved import map (identifies the symbol at the call).
      summary: an explicit :class:`ExternSummary` override; auto-looked-up from the
        resolved symbol when omitted.
      boundary_edge: an explicit agent-declared edge — honored VERBATIM (A8③
        explicit override). Auto-synthesis only fills when none was supplied.

    Returns a synthesized :class:`BoundaryEdge` (feed straight into
    ``trace_provenance(..., boundary_edge=)``), OR a :class:`BoundaryEdgeUnresolved`
    naming what is missing (A8④ — never a silently-wrong edge).

    Edge shapes (A8②):
      * ``memcpy``/``memmove`` (sink⊆dst): a COPY edge to ``src + (sink.base - dst)``
        at the same length — the backtrace continues from the source buffer.
      * ``memset`` (sink⊆dst): a CONST edge — the source is the constant byte ``c``
        (``x1``), NOT a buffer; recorded as ``kind=CONST`` with the const byte so the
        backtrace knows the bytes are a constant fill, not a dangling pointer.
      * a call whose summary has no dst/src mapping ⇒ no auto-edge (Unresolved).

    The synthesized edge carries WHY it exists (``via``, the concrete dst/src/n it
    was built from) in ``decode_meta`` so the export/audit shows the construction —
    never an unexplained jump."""
    # A8③: explicit edge → verbatim, no auto-synthesis.
    if boundary_edge is not None:
        return (boundary_edge if isinstance(boundary_edge, BoundaryEdge)
                else BoundaryEdge.from_wire(boundary_edge))

    trace = list(trace)
    sink_base, sink_len = int(sink_region[0]), int(sink_region[1])

    call_ins = _find_call_ins(trace, call_site)
    if call_ins is None:
        return BoundaryEdgeUnresolved(
            symbol="<no-call>", reason="no_call_at_site", missing=("call_site",),
            call_pc=None,
            detail=(f"no call instruction found at idx/pc 0x{int(call_site):x} — "
                    "supply the trace idx or pc of the boundary call"),
        )

    target = _resolve_call_target(call_ins)
    symbol = import_map.symbol_for(target) if target is not None else None
    if symbol is None:
        # A8④: unknown call / no import-map hit → no auto-edge. Fall back to the
        # agent-declared edge (which the caller supplies as boundary_edge above).
        missing = ("import_map_symbol",) if target is not None else ("call_target",)
        return BoundaryEdgeUnresolved(
            symbol=(f"unknown@0x{target:x}" if target is not None
                    else "unknown@<unresolved>"),
            reason="unknown_call_no_import_hit", missing=missing,
            call_pc=call_ins.pc,
            detail=("the call at the provenance boundary is not in the import map "
                    "(no symbol) — cannot map sink→source from an unknown ABI; "
                    "supply the import map symbol or an explicit boundary_edge"),
        )

    summ = summary or extern_summary(symbol)
    if summ is None:
        return BoundaryEdgeUnresolved(
            symbol=symbol, reason="no_extern_summary", missing=("extern_summary",),
            call_pc=call_ins.pc,
            detail=(f"{symbol} has no ABI summary — cannot map its dst/src roles; "
                    "add an ExternSummary or supply an explicit boundary_edge"),
        )

    dst_reg = summ.role_reg("dst")
    src_reg = summ.role_reg("src")
    n_reg = summ.role_reg("n")
    c_reg = summ.role_reg("c")

    # memset: CONST edge (source is a constant byte, not a buffer).
    if dst_reg is not None and src_reg is None and c_reg is not None:
        dst = _read_reg_at(call_ins, dst_reg)
        c = _read_reg_at(call_ins, c_reg)
        n = _read_reg_at(call_ins, n_reg)
        missing = [name for name, v in
                   (("dst_concrete", dst), ("const_byte", c), ("n_concrete", n))
                   if v is None]
        if missing:
            return BoundaryEdgeUnresolved(
                symbol=symbol, reason="args_not_concrete_at_call_site",
                missing=tuple(missing), call_pc=call_ins.pc,
                detail=(f"{symbol}({dst_reg},{c_reg},{n_reg}): one or more args not "
                        "readable at the call site (symbolic / not in trace) — "
                        "watch the source pointer same-execution (#3) or declare the "
                        "edge; no edge fabricated"),
            )
        if not (dst <= sink_base and sink_base + sink_len <= dst + n):
            return BoundaryEdgeUnresolved(
                symbol=symbol, reason="sink_not_subset_of_dst",
                missing=("sink_within_dst",), call_pc=call_ins.pc,
                detail=(f"sink [0x{sink_base:x},+{sink_len}) is not ⊆ dst "
                        f"[0x{dst:x},+{n}) — this call did not produce the sink; "
                        "no edge fabricated"),
            )
        const_byte = c & 0xFF
        return BoundaryEdge(
            sink_surface=sink_base,
            boundary_pc_from=call_ins.pc, boundary_pc_to=call_ins.pc,
            source_ptr=sink_base,           # CONST: no source buffer — anchor self
            transform="raw",
            decode_meta={
                "via": symbol, "kind": EDGE_KIND_CONST, "const_byte": const_byte,
                "dst": f"0x{dst:x}", "n": n,
                "synthesized": True,
                "explain": (f"sink ⊆ dst of {symbol}; source is the CONSTANT byte "
                            f"0x{const_byte:02x} (x1), not a buffer"),
            },
        )

    # memcpy / memmove: COPY edge (sink⊆dst ⇒ source = src at the matching offset).
    if dst_reg is not None and src_reg is not None:
        dst = _read_reg_at(call_ins, dst_reg)
        src = _read_reg_at(call_ins, src_reg)
        n = _read_reg_at(call_ins, n_reg)
        missing = [name for name, v in
                   (("dst_concrete", dst), ("src_concrete", src), ("n_concrete", n))
                   if v is None]
        if missing:
            return BoundaryEdgeUnresolved(
                symbol=symbol, reason="args_not_concrete_at_call_site",
                missing=tuple(missing), call_pc=call_ins.pc,
                detail=(f"{symbol}({dst_reg},{src_reg},{n_reg}): one or more args not "
                        "readable at the call site (symbolic / not in trace) — "
                        "watch the source pointer same-execution (#3) or declare the "
                        "edge; no edge fabricated (a wrong silent edge corrupts "
                        "provenance — forbidden)"),
            )
        if not (dst <= sink_base and sink_base + sink_len <= dst + n):
            return BoundaryEdgeUnresolved(
                symbol=symbol, reason="sink_not_subset_of_dst",
                missing=("sink_within_dst",), call_pc=call_ins.pc,
                detail=(f"sink [0x{sink_base:x},+{sink_len}) is not ⊆ dst "
                        f"[0x{dst:x},+{n}) of {symbol} — this call did not produce "
                        "the sink; no edge fabricated"),
            )
        # The offset of the sink within dst maps to the SAME offset within src.
        offset = sink_base - dst
        source_ptr = src + offset
        return BoundaryEdge(
            sink_surface=sink_base,
            boundary_pc_from=call_ins.pc, boundary_pc_to=call_ins.pc,
            source_ptr=source_ptr,
            transform="raw",
            decode_meta={
                "via": symbol, "kind": EDGE_KIND_COPY,
                "dst": f"0x{dst:x}", "src": f"0x{src:x}", "n": n,
                "sink_offset_in_dst": offset, "len": sink_len,
                "synthesized": True,
                "explain": (f"sink [0x{sink_base:x},+{sink_len}) ⊆ dst "
                            f"[0x{dst:x},+{n}) of {symbol}; source = src+offset "
                            f"= 0x{source_ptr:x} (same offset {offset})"),
            },
        )

    # A call with no dst/src mapping (e.g. time/rand/strlen) ⇒ no auto-edge.
    return BoundaryEdgeUnresolved(
        symbol=symbol, reason="no_dst_src_mapping", missing=("dst_src_roles",),
        call_pc=call_ins.pc,
        detail=(f"{symbol} has no dst/src ABI mapping (not a region copy) — there is "
                "no sink→source edge to synthesize; falls through"),
    )
