"""Oracle-anchored provenance mode (#3): backtrace from a located sink, classify
the production. Synthetic traces only — one per verdict.
"""

from __future__ import annotations

from engine.oracle_provenance import (
    BoundaryEdge,
    ProvenanceVerdict,
    trace_provenance,
)
from engine.types import Instruction, MemOp, MemSnapshot


def _ins(idx, mnem, reads=None, writes=None, mem=()):
    return Instruction(idx=idx, pc=0x70000 + idx * 4, bytes_=b"\x00\x00\x00\x00",
                       mnemonic=mnem, regs_read=dict(reads or {}),
                       regs_write=dict(writes or {}), mem=mem)


def _le(b: bytes) -> int:
    return int.from_bytes(b, "little")


EXPECTED = bytes([0x34, 0x15, 0x5f, 0xe9])
OUT = 0x72b18      # located sink base (from #1)
TMP = 0x70f80      # a 1-byte transient streaming slot
UNK = 0x9000       # an un-captured memory address


def test_continuous_buffer_output_in_one_contiguous_write_buffer():
    trace = [
        _ins(0, "mov x8, #v", writes={"x8": _le(EXPECTED)}),
        _ins(1, "str x8, [x9]", reads={"x8": _le(EXPECTED), "x9": OUT},
             mem=(MemOp("w", OUT, _le(EXPECTED), 4),)),
    ]
    res = trace_provenance(trace, EXPECTED, sink_base=OUT)
    assert res.verdict is ProvenanceVerdict.CONTINUOUS_BUFFER
    assert res.base == OUT
    assert res.producer_pcs == (0x70004,)           # the str
    assert res.next_watch == []


def test_streaming_output_written_chunkwise_no_contiguous_buffer():
    # each output byte is computed then written to the SAME 1-byte transient slot,
    # in order; the full output only exists in a snapshot (located sink), never as
    # a contiguous write buffer.
    trace = [
        _ins(0, "strb w0, [x10]", reads={"x0": EXPECTED[0], "x10": TMP},
             mem=(MemOp("w", TMP, EXPECTED[0], 1),)),
        _ins(1, "strb w0, [x10]", reads={"x0": EXPECTED[1], "x10": TMP},
             mem=(MemOp("w", TMP, EXPECTED[1], 1),)),
        _ins(2, "strb w0, [x10]", reads={"x0": EXPECTED[2], "x10": TMP},
             mem=(MemOp("w", TMP, EXPECTED[2], 1),)),
        _ins(3, "strb w0, [x10]", reads={"x0": EXPECTED[3], "x10": TMP},
             mem=(MemOp("w", TMP, EXPECTED[3], 1),)),
    ]
    snaps = [MemSnapshot(addr=OUT, data=EXPECTED, label="streamed_output")]
    res = trace_provenance(trace, EXPECTED, sink_base=OUT, snapshots=snaps)
    assert res.verdict is ProvenanceVerdict.STREAMING
    assert res.base is None                          # no contiguous buffer
    assert res.producer_pcs == (0x70000, 0x70004, 0x70008, 0x7000c)  # the 4 chunks
    assert "no contiguous buffer" in res.detail


def test_needs_observation_chain_breaks_at_uncaptured_read():
    # the output value is loaded from an un-captured address, then written; its
    # production is not visible -> next-watch points at the un-captured source.
    trace = [
        _ins(0, "ldr x8, [x10]", reads={"x10": UNK}, writes={"x8": 0x34},
             mem=(MemOp("r", UNK, 0x34, 4),)),                # un-captured source
        _ins(1, "strb w8, [x12]", reads={"x8": 0x34, "x12": OUT},
             mem=(MemOp("w", OUT, EXPECTED[0], 1),)),          # only 1 byte at OUT
    ]
    snaps = [MemSnapshot(addr=OUT, data=EXPECTED)]            # full output only in snapshot
    res = trace_provenance(trace, EXPECTED, sink_base=OUT, snapshots=snaps)
    assert res.verdict is ProvenanceVerdict.NEEDS_OBSERVATION
    assert any(w["addr"] == f"0x{UNK:x}" for w in res.next_watch)
    assert res.next_watch[0]["pc"] == "0x70000"              # the ldr that read it


def test_opaque_callee_data_appears_after_call_no_traced_producer():
    # nothing writes OUT in the trace; the output is first OBSERVED (read) only
    # AFTER a bl returns — produced by an un-traced callee/bridge.
    trace = [
        _ins(0, "mov x0, #1", writes={"x0": 1}),
        _ins(1, "bl #0x72ecc"),                                   # call boundary
        _ins(2, "ldr x8, [x12]", reads={"x12": OUT},
             mem=(MemOp("r", OUT, _le(EXPECTED), 4),)),           # data now present
    ]
    res = trace_provenance(trace, EXPECTED, sink_base=OUT)
    assert res.verdict is ProvenanceVerdict.OPAQUE_CALLEE
    assert res.boundary_pcs == (0x70004,)                         # the bl at idx1
    assert "0x72ecc" in res.detail                                # callee target named
    assert res.base is None


def test_transient_scrub_buffer_recovered_by_temporal_scan():
    # OUT briefly holds the full output, then is scrubbed to zero. last-write-wins
    # sees only the scrub; the temporal write-history scan recovers the buffer.
    trace = [
        _ins(0, "str x8, [x9]", reads={"x8": _le(EXPECTED), "x9": OUT},
             mem=(MemOp("w", OUT, _le(EXPECTED), 4),)),           # buffer == expected
        _ins(1, "str xzr, [x9]", reads={"x9": OUT},
             mem=(MemOp("w", OUT, 0, 4),)),                       # scrubbed to 0
    ]
    res = trace_provenance(trace, EXPECTED, sink_base=OUT)
    assert res.verdict is ProvenanceVerdict.CONTINUOUS_BUFFER
    assert res.transient is True
    assert res.producer_pcs == (0x70000,)                         # the producing str
    assert "scrubbed" in res.detail


def test_opaque_callee_carries_nonempty_chain_with_boundary_and_first_seen():
    # round-3: snapshot/observed sink, first_seen but no traced writer (appears
    # after a call return) -> non-empty chain (boundary + first-seen steps), not 0.
    trace = [
        _ins(0, "mov x0, #1", writes={"x0": 1}),
        _ins(1, "bl #0x72ecc"),                                   # call boundary
        _ins(2, "ldr x8, [x12]", reads={"x12": OUT},
             mem=(MemOp("r", OUT, _le(EXPECTED), 4),)),           # data first observed
    ]
    res = trace_provenance(trace, EXPECTED, sink_base=OUT)
    assert res.verdict is ProvenanceVerdict.OPAQUE_CALLEE
    assert len(res.chain) >= 2                                    # NOT chain_len=0
    idxs = [s["idx"] for s in res.chain]
    assert 1 in idxs and 2 in idxs                                # boundary + first_seen
    assert res.boundary_pcs == (0x70004,)                         # boundary PC
    assert any(s.get("note", "").startswith("call boundary") for s in res.chain)


def test_blr_indirect_call_target_resolved_from_register_value():
    # round-3: localize the indirect call AND name the concrete callee read from
    # the register value at the blr site (trace is concrete).
    CALLEE = 0x72ecc
    trace = [
        _ins(0, "mov x8, #t", writes={"x8": CALLEE}),
        _ins(1, "blr x8", reads={"x8": CALLEE}),                  # indirect call
        _ins(2, "ldr x9, [x12]", reads={"x12": OUT},
             mem=(MemOp("r", OUT, _le(EXPECTED), 4),)),
    ]
    res = trace_provenance(trace, EXPECTED, sink_base=OUT)
    assert res.verdict is ProvenanceVerdict.OPAQUE_CALLEE
    assert res.boundary_pcs == (0x70004,)                         # the blr site
    assert res.callee_targets == (CALLEE,)                        # resolved from x8
    assert f"0x{CALLEE:x}" in res.detail
    # the boundary step also names the resolved callee
    assert any(s.get("callee") == f"0x{CALLEE:x}" for s in res.chain)


def test_needs_observation_streaming_unprovable_when_no_producer_no_call():
    # output only in a snapshot; no traced write, no call boundary, no reads.
    trace = [_ins(0, "mov x0, #1", writes={"x0": 1})]
    snaps = [MemSnapshot(addr=OUT, data=EXPECTED)]
    res = trace_provenance(trace, EXPECTED, sink_base=OUT, snapshots=snaps)
    assert res.verdict is ProvenanceVerdict.NEEDS_OBSERVATION
    assert res.streaming == "unprovable"
    assert "neither confirmed nor refuted" in res.detail


def test_to_dict_shape():
    trace = [_ins(0, "str x8, [x9]", reads={"x8": _le(EXPECTED), "x9": OUT},
                  mem=(MemOp("w", OUT, _le(EXPECTED), 4),))]
    d = trace_provenance(trace, EXPECTED, sink_base=OUT).to_dict()
    assert d["verdict"] == "CONTINUOUS_BUFFER"
    assert d["base"] == f"0x{OUT:x}"
    assert d["expected"] == EXPECTED.hex()


# --- boundary-edge provenance (dev-boundary-edge-provenance-spec) -----------
#
# A logical/transform output has a VALUE but NO native writer in the trace, so the
# producer backtrace dead-ends UNPLACEABLE. A case that KNOWS the boundary declares
# an edge; provenance anchors at the boundary and surfaces the pre-transform
# source_ptr as the next backtrace anchor. Synthetic multi-form fixtures (pure
# call-return / base64 / framing) — ZERO case-specific addresses; the edge is DATA.
# Anchoring is PROGRESS, not closure: the verdict never auto-CLOSED/CONFIRMED.

# A logical sink whose full output is only observed (snapshot) — no traced write,
# no observed call-return appearance with reads → today this is the UNPLACEABLE
# "streaming unprovable" dead-end (path #5).
SRC = 0x12312480      # a synthetic pre-transform native source pointer (NOT a real
                      # case address — purely a fixture stand-in)


def _logical_sink_trace():
    """A trace where the sink has NO native writer and NO call-return observation —
    the genuine logical/transform UNPLACEABLE dead-end. Output lives only in a
    snapshot (a runner dump of the post-transform buffer)."""
    trace = [_ins(0, "mov x0, #1", writes={"x0": 1})]
    snaps = [MemSnapshot(addr=OUT, data=EXPECTED, label="post_transform_output")]
    return trace, snaps


def _edge(transform, decode_meta=None):
    # Generic edge shape — boundary_pc_from→to, source_ptr, transform. No address
    # here is a real case address; all are synthetic fixture stand-ins.
    return BoundaryEdge(
        sink_surface=OUT, boundary_pc_from=0xB2128, boundary_pc_to=0xB212C,
        source_ptr=SRC, transform=transform, decode_meta=decode_meta or {})


def test_boundary_edge_call_return_anchors_logical_sink():
    # pure call-return boundary: logical output, no native writer; declared edge
    # anchors at boundary_pc_to and surfaces source_ptr as next anchor.
    trace, snaps = _logical_sink_trace()
    res = trace_provenance(trace, EXPECTED, sink_base=OUT, snapshots=snaps,
                           boundary_edge=_edge("raw"))
    assert res.verdict is ProvenanceVerdict.BOUNDARY_EDGE
    assert res.boundary_pcs == (0xB212C,)                       # anchored at to-pc
    # source_ptr surfaced as next backtrace anchor:
    assert any(w["addr"] == f"0x{SRC:x}"
               and w.get("role") == "pre_transform_source_ptr"
               for w in res.next_watch)
    # explicit, WARN-loud, case-asserted marking (audit can tell it apart):
    assert "DECLARED boundary edge" in res.detail
    assert "NOT a traced writer" in res.detail
    assert res.anchored_edge is not None
    assert res.anchored_edge.source_ptr == SRC


def test_boundary_edge_base64_form_anchors():
    # base64 transform edge (utov does NOT decode — annotation only) still anchors.
    trace, snaps = _logical_sink_trace()
    res = trace_provenance(trace, EXPECTED, sink_base=OUT, snapshots=snaps,
                           boundary_edge=_edge("base64",
                                               decode_meta={"raw_len": 3}))
    assert res.verdict is ProvenanceVerdict.BOUNDARY_EDGE
    assert res.anchored_edge.transform == "base64"
    assert res.anchored_edge.decode_meta == {"raw_len": 3}
    assert res.boundary_pcs == (0xB212C,)


def test_boundary_edge_framing_form_anchors():
    # framing transform edge (e.g. length-prefixed body) anchors the same way.
    trace, snaps = _logical_sink_trace()
    res = trace_provenance(trace, EXPECTED, sink_base=OUT, snapshots=snaps,
                           boundary_edge=_edge("framing",
                                               decode_meta={"body_offset": 4}))
    assert res.verdict is ProvenanceVerdict.BOUNDARY_EDGE
    assert res.anchored_edge.transform == "framing"
    assert res.boundary_pcs == (0xB212C,)


def test_boundary_edge_accepts_wire_dict_form():
    # case-declared DATA may arrive as a wire dict (hex strings); utov normalises it.
    trace, snaps = _logical_sink_trace()
    edge = {
        "sink_surface": f"0x{OUT:x}",
        "boundary_pc_from": "0xb2128", "boundary_pc_to": "0xb212c",
        "source_ptr": f"0x{SRC:x}", "transform": "jni_string",
        "decode_meta": {"marker": "utf"},
    }
    res = trace_provenance(trace, EXPECTED, sink_base=OUT, snapshots=snaps,
                           boundary_edge=edge)
    assert res.verdict is ProvenanceVerdict.BOUNDARY_EDGE
    assert res.anchored_edge.transform == "jni_string"
    assert res.anchored_edge.boundary_pc_to == 0xB212C


def test_no_edge_same_logical_sink_regresses_to_unplaceable():
    # SAFETY (invariant 7): the SAME logical sink with NO declared edge behaves
    # exactly as today — honest UNPLACEABLE BLOCK (streaming unprovable).
    trace, snaps = _logical_sink_trace()
    res = trace_provenance(trace, EXPECTED, sink_base=OUT, snapshots=snaps)
    assert res.verdict is ProvenanceVerdict.NEEDS_OBSERVATION
    assert res.streaming == "unprovable"
    assert res.anchored_edge is None
    assert "anchored_edge" not in res.to_dict()                 # additive serialization


def test_boundary_edge_does_not_auto_close():
    # SAFETY: anchoring is NOT closure. The verdict is BOUNDARY_EDGE — never a
    # CLOSED/CONFIRMED-equivalent (CONTINUOUS_BUFFER / STREAMING). It carries
    # next_watch (still work to do) and is explicitly marked "NOT closure".
    trace, snaps = _logical_sink_trace()
    res = trace_provenance(trace, EXPECTED, sink_base=OUT, snapshots=snaps,
                           boundary_edge=_edge("raw"))
    assert res.verdict not in (ProvenanceVerdict.CONTINUOUS_BUFFER,
                               ProvenanceVerdict.STREAMING)
    assert res.base is None                                     # not a placed buffer
    assert res.next_watch != []                                 # frontier still open
    assert "NOT closure" in res.detail


def test_boundary_edge_source_ptr_with_no_producer_stays_unplaceable():
    # source_ptr ALSO has no producer → surfaced as a pc:null watch. The wall moved
    # one notch forward; it is NOT falsely re-anchored / closed (still unplaceable).
    trace, snaps = _logical_sink_trace()
    res = trace_provenance(trace, EXPECTED, sink_base=OUT, snapshots=snaps,
                           boundary_edge=_edge("raw"))
    w = next(x for x in res.next_watch if x["addr"] == f"0x{SRC:x}")
    assert w["pc"] is None                                      # no producer for source
    assert "still" in w["reason"] and "unplaceable" in w["reason"]


def test_boundary_edge_source_ptr_with_producer_continues_backtrace():
    # source_ptr DOES have a traced native producer → it is surfaced WITH its PC, so
    # the backtrace continues from a real, walkable surface (frontier pushed onto a
    # producible address). Still BOUNDARY_EDGE (anchored), not auto-closed.
    trace = [
        _ins(0, "str x8, [x9]", reads={"x8": 0xAA, "x9": SRC},
             mem=(MemOp("w", SRC, 0xAA, 1),)),                  # writes source_ptr
        _ins(1, "mov x0, #1", writes={"x0": 1}),
    ]
    snaps = [MemSnapshot(addr=OUT, data=EXPECTED, label="post_transform_output")]
    res = trace_provenance(trace, EXPECTED, sink_base=OUT, snapshots=snaps,
                           boundary_edge=_edge("raw"))
    assert res.verdict is ProvenanceVerdict.BOUNDARY_EDGE
    w = next(x for x in res.next_watch if x["addr"] == f"0x{SRC:x}")
    assert w["pc"] == "0x70000"                                 # the str that wrote it
    assert "traced producer" in w["reason"]


def test_boundary_edge_ignored_when_surface_mismatches():
    # an edge whose sink_surface is OUTSIDE the sink window must NOT mis-anchor a
    # different surface — it is ignored, behaviour falls back to today's UNPLACEABLE.
    trace, snaps = _logical_sink_trace()
    mism = BoundaryEdge(sink_surface=OUT + 0x1000, boundary_pc_from=0xB2128,
                        boundary_pc_to=0xB212C, source_ptr=SRC, transform="raw")
    res = trace_provenance(trace, EXPECTED, sink_base=OUT, snapshots=snaps,
                           boundary_edge=mism)
    assert res.verdict is ProvenanceVerdict.NEEDS_OBSERVATION
    assert res.anchored_edge is None


def test_boundary_edge_does_not_override_real_traced_production():
    # SAFETY: even WITH an edge declared, a real traced CONTINUOUS_BUFFER production
    # wins — an edge never overrides observed native production (it only fills the
    # no-native-writer dead-end).
    trace = [
        _ins(0, "mov x8, #v", writes={"x8": _le(EXPECTED)}),
        _ins(1, "str x8, [x9]", reads={"x8": _le(EXPECTED), "x9": OUT},
             mem=(MemOp("w", OUT, _le(EXPECTED), 4),)),         # real traced buffer
    ]
    res = trace_provenance(trace, EXPECTED, sink_base=OUT,
                           boundary_edge=_edge("raw"))
    assert res.verdict is ProvenanceVerdict.CONTINUOUS_BUFFER   # observed wins
    assert res.anchored_edge is None


def test_boundary_edge_to_dict_roundtrip_and_serialization():
    trace, snaps = _logical_sink_trace()
    res = trace_provenance(trace, EXPECTED, sink_base=OUT, snapshots=snaps,
                           boundary_edge=_edge("base64", decode_meta={"raw_len": 3}))
    d = res.to_dict()
    assert d["verdict"] == "BOUNDARY_EDGE"
    assert d["boundary_pcs"] == ["0xb212c"]
    assert d["anchored_edge"]["source_ptr"] == f"0x{SRC:x}"
    assert d["anchored_edge"]["transform"] == "base64"
    assert d["anchored_edge"]["decode_meta"] == {"raw_len": 3}
