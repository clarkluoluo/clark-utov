"""Item ① — multi-source trace merge.

Synthetic, zero case-specific: every address/idx is a fixture constant chosen to
exercise a mechanism (align by idx/pc, snapshot carried not forged, unaligned
surfaced, reg overlay fill/conflict, no-sidecar identity). Asserts validate the
mechanism, not a case solution.
"""

from __future__ import annotations

from engine.trace_merge import MemEvent, merge_trace_sources
from engine.types import Instruction, MemOp, MemSnapshot


def ins(idx, pc, mnem="nop", *, reads=None, writes=None, mem=()):
    return Instruction(
        idx=idx, pc=pc, bytes_=b"\x00\x00\x00\x00", mnemonic=mnem,
        regs_read=reads or {}, regs_write=writes or {}, mem=tuple(mem))


# tc4-form synthetic coordinates.
_A = 0x4000
_B = 0x4008


def test_mem_events_merge_by_idx():
    # main trace has empty mem; the sidecar carries the memory writes by idx.
    main = [ins(0, 0x1000, "str x8,[x10]"), ins(1, 0x1004, "ldr x9,[x10]")]
    mem_events = [
        MemEvent(op=MemOp("w", _A, 0x41, 8), idx=0),
        MemEvent(op=MemOp("r", _A, 0x41, 8), idx=1),
    ]
    merged = merge_trace_sources(main, mem_events=mem_events)
    assert merged.items[0].mem == (MemOp("w", _A, 0x41, 8),)
    assert merged.items[1].mem == (MemOp("r", _A, 0x41, 8),)
    assert merged.report.mem_events_merged == 2
    assert not merged.report.unaligned


def test_mem_events_merge_by_pc_fallback():
    # event with no idx aligns by pc (first instruction at that pc).
    main = [ins(0, 0x1000), ins(1, 0x1004)]
    mem_events = [MemEvent(op=MemOp("w", _B, 0x7, 4), pc=0x1004)]
    merged = merge_trace_sources(main, mem_events=mem_events)
    assert merged.items[1].mem == (MemOp("w", _B, 0x7, 4),)
    assert merged.report.mem_events_merged == 1


def test_two_cohort_form_memwrite_visible_across_inputs():
    # tc4-style: two cohort traces whose memory write VALUE differs across inputs,
    # only visible AFTER merge (main mem is empty before).
    def cohort(val):
        main = [ins(0, 0x1000, "str x8,[x10]")]
        ev = [MemEvent(op=MemOp("w", _A, val, 8), idx=0)]
        return merge_trace_sources(main, mem_events=ev).items
    a = cohort(0x11)
    b = cohort(0x22)
    assert a[0].mem[0].val != b[0].mem[0].val   # the seed-varying write is now visible


def test_unaligned_events_surface_not_dropped():
    main = [ins(0, 0x1000)]
    mem_events = [
        MemEvent(op=MemOp("w", _A, 1, 4), idx=99),     # no such idx
        MemEvent(op=MemOp("w", _B, 2, 4), pc=0xDEAD),  # no such pc
    ]
    merged = merge_trace_sources(main, mem_events=mem_events)
    assert merged.report.mem_events_merged == 0
    assert len(merged.report.unaligned) == 2
    assert merged.items[0].mem == ()   # untouched


def test_snapshot_carried_not_forged_into_memop():
    main = [ins(0, 0x1000)]
    snap = MemSnapshot(addr=_A, data=b"\xde\xad", label="out")
    merged = merge_trace_sources(main, snapshots=[snap])
    assert merged.snapshots == (snap,)
    assert merged.report.snapshots_attached == 1
    # snapshot is NOT turned into a MemOp on any instruction.
    assert merged.items[0].mem == ()


def test_no_sidecar_is_identity_same_objects():
    # invariant 7: no sidecar -> output is the input verbatim (same objects).
    main = [ins(0, 0x1000, "mov", reads={"x1": 5}), ins(1, 0x1004, "add", writes={"x2": 9})]
    merged = merge_trace_sources(main)
    assert len(merged.items) == 2
    for orig, out in zip(main, merged.items):
        assert out is orig          # SAME object, not a copy
    assert merged.report.mem_events_merged == 0
    assert merged.snapshots == ()


def test_reg_overlay_fills_missing_only():
    main = [ins(0, 0x1000, reads={"x1": 5})]
    merged = merge_trace_sources(main, reg_overlay={0: {"x2": 7}})
    assert merged.items[0].regs_read == {"x1": 5, "x2": 7}
    assert merged.report.reg_overlay_filled == 1


def test_reg_overlay_conflict_surfaced_not_applied():
    main = [ins(0, 0x1000, reads={"x1": 5})]
    merged = merge_trace_sources(main, reg_overlay={0: {"x1": 999}})
    assert merged.items[0].regs_read == {"x1": 5}   # NOT overwritten
    assert merged.report.reg_overlay_conflicts
    assert merged.report.reg_overlay_conflicts[0]["reg"] == "x1"


def test_multiple_events_same_idx_appended_in_order():
    main = [ins(0, 0x1000)]
    mem_events = [
        MemEvent(op=MemOp("r", _A, 1, 4), idx=0),
        MemEvent(op=MemOp("w", _B, 2, 4), idx=0),
    ]
    merged = merge_trace_sources(main, mem_events=mem_events)
    assert merged.items[0].mem == (MemOp("r", _A, 1, 4), MemOp("w", _B, 2, 4))


def test_to_dict_shapes_present():
    main = [ins(0, 0x1000)]
    merged = merge_trace_sources(main, mem_events=[MemEvent(op=MemOp("w", _A, 1, 4), idx=5)])
    d = merged.report.to_dict()
    assert d["kind"] == "trace_merge_report"
    assert d["unaligned"]   # the idx=5 event did not align


# --- reader-side canonical mem-event sidecar (item ① adapter boundary) --------

def test_parse_mem_events_canonical():
    from engine.obs_readers import parse_mem_events
    text = (
        '{"idx": 0, "rw": "w", "addr": "0x4000", "val": "0x41", "size": 8}\n'
        '{"pc": "0x1004", "rw": "r", "addr": "0x4000", "val": 65, "size": 8}\n'
        '{"rw": "w", "addr": "0x4000", "size": 8}\n'        # no locating key -> skipped
        'garbage line not json\n'
    )
    events = parse_mem_events(text)
    assert len(events) == 2
    assert events[0].idx == 0 and events[0].op.rw == "w"
    assert events[1].pc == 0x1004 and events[1].op.val == 65


def test_jsonl_reader_merged_sidecar(tmp_path):
    from engine.runner_client import JsonlTraceReader
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        '{"idx": 0, "pc": "0x1000", "bytes": "00000000", "mnemonic": "str x8,[x10]"}\n'
        '{"idx": 1, "pc": "0x1004", "bytes": "00000000", "mnemonic": "ldr x9,[x10]"}\n')
    memsc = tmp_path / "trace_mem.jsonl"
    memsc.write_text(
        '{"idx": 0, "rw": "w", "addr": "0x4000", "val": "0x41", "size": 8}\n'
        '{"idx": 1, "rw": "r", "addr": "0x4000", "val": "0x41", "size": 8}\n')
    # default iteration: mem empty (sidecar not folded by __iter__).
    plain = list(JsonlTraceReader(trace))
    assert all(i.mem == () for i in plain)
    # merged(): sidecar folded in.
    merged = JsonlTraceReader(trace, mem_sidecar=memsc).merged()
    assert merged.items[0].mem == (MemOp("w", 0x4000, 0x41, 8),)
    assert merged.items[1].mem == (MemOp("r", 0x4000, 0x41, 8),)
