"""#5 — auto-synthesize a libc BoundaryEdge (sink → source) from the ABI.

Regression fixtures map 1:1 to spec_tc2_libc_boundary_autosynth "Regression
fixtures" + A8④ degenerate paths. Composes #4 (import map + extern summary) with
the existing oracle_provenance.BoundaryEdge.
"""

from __future__ import annotations

from engine.import_map import ImportMap, build_import_map
from engine.libc_boundary import (
    BoundaryEdgeUnresolved,
    EDGE_KIND_CONST,
    EDGE_KIND_COPY,
    synthesize_boundary_edge,
)
from engine.oracle_provenance import (
    BoundaryEdge,
    ProvenanceVerdict,
    trace_provenance,
)
from engine.types import Instruction, MemOp


def _ins(idx, mnem, reads=None, mem=(), pc=None):
    return Instruction(idx=idx, pc=pc if pc is not None else 0x400000 + idx * 4,
                       bytes_=b"\x00\x00\x00\x00", mnemonic=mnem,
                       regs_read=dict(reads or {}), regs_write={}, mem=mem)


# A small import map mapping the memcpy / memset PLT stubs.
_IM = build_import_map(plt_map={0x400a90: "memcpy", 0x400ab0: "memset"})

DST = 0x80000
SRC = 0x90000


def test_memcpy_sink_subset_dst_synthesizes_edge_to_matching_src_offset():
    # memcpy(dst=DST, src=SRC, n=32); sink = [DST+4, +8) ⊆ dst.
    trace = [_ins(0, "bl 0x400a90", reads={"x0": DST, "x1": SRC, "x2": 32})]
    edge = synthesize_boundary_edge(trace, 0, (DST + 4, 8), _IM)
    assert isinstance(edge, BoundaryEdge)
    assert edge.sink_surface == DST + 4
    # source = src + (sink.base - dst) = SRC + 4
    assert edge.source_ptr == SRC + 4
    assert edge.decode_meta["kind"] == EDGE_KIND_COPY
    assert edge.decode_meta["synthesized"] is True
    assert edge.decode_meta["via"] == "memcpy"


def test_synthesized_edge_continues_trace_provenance_past_needs_observation():
    # The sink has NO native writer in the bundled trace (the memcpy copy happened
    # in unbundled code) → trace_provenance would dead-end. The synthesized edge
    # anchors it at the boundary and surfaces the pre-transform source as next anchor.
    expected = bytes(range(8))
    sink_base = DST + 4
    # A trace where the sink is never written (only the memcpy call instruction).
    trace = [_ins(0, "bl 0x400a90", reads={"x0": DST, "x1": SRC, "x2": 32})]
    edge = synthesize_boundary_edge(trace, 0, (sink_base, 8), _IM)
    assert isinstance(edge, BoundaryEdge)
    res = trace_provenance(trace, expected, sink_base=sink_base, boundary_edge=edge)
    assert res.verdict is ProvenanceVerdict.BOUNDARY_EDGE
    assert res.anchored_edge is not None
    # The pre-transform source pointer is surfaced as the next anchor.
    assert any(w.get("addr") == f"0x{SRC + 4:x}" for w in res.next_watch)


def test_explicit_boundary_edge_used_verbatim_no_autosynth():
    declared = BoundaryEdge(
        sink_surface=DST, boundary_pc_from=0x1, boundary_pc_to=0x2,
        source_ptr=0xDEAD, transform="raw")
    trace = [_ins(0, "bl 0x400a90", reads={"x0": DST, "x1": SRC, "x2": 32})]
    edge = synthesize_boundary_edge(trace, 0, (DST, 8), _IM, boundary_edge=declared)
    assert edge is declared          # verbatim — no auto-synthesis


def test_explicit_edge_from_wire_dict_honored_verbatim():
    wire = {"sink_surface": "0x80000", "boundary_pc_from": "0x1",
            "boundary_pc_to": "0x2", "source_ptr": "0xdead"}
    trace = [_ins(0, "bl 0x400a90", reads={"x0": DST, "x1": SRC, "x2": 32})]
    edge = synthesize_boundary_edge(trace, 0, (DST, 8), _IM, boundary_edge=wire)
    assert isinstance(edge, BoundaryEdge)
    assert edge.source_ptr == 0xDEAD


def test_symbolic_n_unresolved_no_fabricated_edge():
    # n is not captured at the call site (symbolic) → cannot prove sink ⊆ dst.
    trace = [_ins(0, "bl 0x400a90", reads={"x0": DST, "x1": SRC})]  # no x2
    res = synthesize_boundary_edge(trace, 0, (DST + 4, 8), _IM)
    assert isinstance(res, BoundaryEdgeUnresolved)
    assert "n_concrete" in res.missing
    assert res.verdict == "BOUNDARY_EDGE_UNRESOLVED"


def test_dst_unresolved_no_fabricated_edge():
    trace = [_ins(0, "bl 0x400a90", reads={"x1": SRC, "x2": 32})]  # no x0
    res = synthesize_boundary_edge(trace, 0, (DST + 4, 8), _IM)
    assert isinstance(res, BoundaryEdgeUnresolved)
    assert "dst_concrete" in res.missing


def test_unknown_call_at_boundary_no_autoedge_falls_back():
    # A call to an address NOT in the import map → unknown, no auto-edge.
    im = build_import_map(plt_map={})
    trace = [_ins(0, "bl 0x99999", reads={"x0": DST, "x1": SRC, "x2": 32})]
    res = synthesize_boundary_edge(trace, 0, (DST + 4, 8), im)
    assert isinstance(res, BoundaryEdgeUnresolved)
    assert res.reason == "unknown_call_no_import_hit"


def test_sink_not_subset_of_dst_no_fabricated_edge():
    # sink is outside [dst, dst+n) → this call did not produce it.
    trace = [_ins(0, "bl 0x400a90", reads={"x0": DST, "x1": SRC, "x2": 8})]
    res = synthesize_boundary_edge(trace, 0, (DST + 100, 8), _IM)
    assert isinstance(res, BoundaryEdgeUnresolved)
    assert res.reason == "sink_not_subset_of_dst"


def test_memset_synthesizes_const_edge_not_dangling_buffer():
    # memset(dst=DST, c=0x41, n=16); sink ⊆ dst → a CONST edge (source is the byte).
    trace = [_ins(0, "bl 0x400ab0", reads={"x0": DST, "x1": 0x41, "x2": 16})]
    edge = synthesize_boundary_edge(trace, 0, (DST, 4), _IM)
    assert isinstance(edge, BoundaryEdge)
    assert edge.decode_meta["kind"] == EDGE_KIND_CONST
    assert edge.decode_meta["const_byte"] == 0x41


def test_no_call_at_site_is_honest():
    trace = [_ins(0, "mov x0, x0")]
    res = synthesize_boundary_edge(trace, 0, (DST, 8), _IM)
    assert isinstance(res, BoundaryEdgeUnresolved)
    assert res.reason == "no_call_at_site"


def test_call_with_no_dst_src_mapping_falls_through():
    # A known symbol with no dst/src roles (e.g. rand) → no edge to synthesize.
    im = build_import_map(plt_map={0x400a80: "rand"})
    trace = [_ins(0, "bl 0x400a80")]
    res = synthesize_boundary_edge(trace, 0, (DST, 8), im)
    assert isinstance(res, BoundaryEdgeUnresolved)
    assert res.reason == "no_dst_src_mapping"


def test_w_register_aliasing_for_n():
    # n supplied as w2 (32-bit) rather than x2 — accept the alias.
    trace = [_ins(0, "bl 0x400a90", reads={"x0": DST, "x1": SRC, "w2": 32})]
    edge = synthesize_boundary_edge(trace, 0, (DST, 8), _IM)
    assert isinstance(edge, BoundaryEdge)
