"""Oracle sink-validator + the verdict -> action gate policy. Synthetic traces
only. Pins all three verdicts AND that the gate ACTS on them (redirects /
blocks), not just reports — the value of the gate is correcting s4's input.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from engine.oracle_sink import (
    SinkGateError,
    SinkVerdict,
    apply_sink_gate,
    validate_sink,
)
from engine.types import Instruction, MemOp, MemSnapshot


def _ins(idx, mnem, mem=()):
    return Instruction(idx=idx, pc=0x70000 + idx * 4, bytes_=b"\x00\x00\x00\x00",
                       mnemonic=mnem, regs_read={}, regs_write={}, mem=mem)


def _le(b: bytes) -> int:
    return int.from_bytes(b, "little")


EXPECTED = bytes([0x34, 0x15, 0x5f, 0xe9])     # the "oracle" output (4 bytes)
SCRATCH = bytes([0x6a, 0xd9, 0x5f, 0x40])      # wrong bytes a scratch store holds
OUT = 0x72b18                                  # real sink base
SCR = 0x70f80                                  # scratch base


# --- validate_sink: the three verdicts --------------------------------------

def test_sink_confirmed_when_candidate_reconstructs_expected():
    trace = [_ins(0, "str x9, [x12]", mem=(MemOp("w", OUT, _le(EXPECTED), 4),))]
    sv = validate_sink(trace, EXPECTED, candidate_idxs=[0])
    assert sv.verdict is SinkVerdict.SINK_CONFIRMED
    assert sv.base == OUT
    assert sv.located_via == "write"


def test_sink_confirmed_auto_located_without_candidate():
    trace = [_ins(0, "str x9, [x12]", mem=(MemOp("w", OUT, _le(EXPECTED), 4),))]
    sv = validate_sink(trace, EXPECTED)            # no candidate
    assert sv.verdict is SinkVerdict.SINK_CONFIRMED
    assert sv.base == OUT


def test_wrong_sink_candidate_is_scratch_real_sink_elsewhere():
    trace = [
        _ins(0, "str x8, [x10]", mem=(MemOp("w", SCR, _le(SCRATCH), 4),)),   # scratch
        _ins(1, "str x9, [x12]", mem=(MemOp("w", OUT, _le(EXPECTED), 4),)),  # real sink
    ]
    sv = validate_sink(trace, EXPECTED, candidate_idxs=[0])
    assert sv.verdict is SinkVerdict.WRONG_SINK
    assert sv.candidate_base == SCR
    assert sv.base == OUT                       # located the real sink
    assert sv.first_diff_offset == 0            # byte0 differs (0x6a vs 0x34)


def test_output_not_observable_when_no_region_matches():
    # only partially-matching scratch present; expected never materialises.
    trace = [_ins(0, "str x8, [x10]", mem=(MemOp("w", SCR, _le(SCRATCH), 4),))]
    sv = validate_sink(trace, EXPECTED)
    assert sv.verdict is SinkVerdict.OUTPUT_NOT_OBSERVABLE
    assert sv.longest_partial["match_count"] == 1   # only byte2 (0x5f) matches
    assert "capability gap" in sv.detail.lower()


def test_search_includes_read_observed_memory():
    # expected appears in a READ-observed region, never written: observable -> not
    # a gap. Reads are now a distinct source from snapshots (located_via="read").
    trace = [_ins(0, "ldr x8, [x10]", mem=(MemOp("r", OUT, _le(EXPECTED), 4),))]
    sv = validate_sink(trace, EXPECTED)
    assert sv.verdict is SinkVerdict.SINK_CONFIRMED
    assert sv.located_via == "read"


# --- snapshot source coverage (the false-negative this task fixes) ----------

def test_output_only_in_snapshot_is_found_not_false_negative():
    # the framed output lives ONLY in a snapshot observation, not in any mem op.
    trace = [_ins(0, "bl x9")]                       # no memory op at all
    snaps = [MemSnapshot(addr=OUT, data=EXPECTED, label="output_buffer")]
    sv = validate_sink(trace, EXPECTED, snapshots=snaps)
    assert sv.verdict is SinkVerdict.SINK_CONFIRMED   # was a false OUTPUT_NOT_OBSERVABLE
    assert sv.base == OUT
    assert sv.located_via == "snapshot"
    assert "snapshots" in sv.scanned_sources


def test_wrong_sink_when_candidate_scratch_but_output_in_snapshot():
    trace = [_ins(0, "str x8, [x10]", mem=(MemOp("w", SCR, _le(SCRATCH), 4),))]
    snaps = [MemSnapshot(addr=OUT, data=EXPECTED)]
    sv = validate_sink(trace, EXPECTED, candidate_idxs=[0], snapshots=snaps)
    assert sv.verdict is SinkVerdict.WRONG_SINK
    assert sv.located_via == "snapshot"
    assert sv.base == OUT


def test_not_observable_flags_missing_snapshot_source():
    # nowhere to be found AND no snapshots provided -> the verdict must say so,
    # not be read as "unobservable by any means".
    trace = [_ins(0, "str x8, [x10]", mem=(MemOp("w", SCR, _le(SCRATCH), 4),))]
    sv = validate_sink(trace, EXPECTED)               # no snapshots
    assert sv.verdict is SinkVerdict.OUTPUT_NOT_OBSERVABLE
    assert "snapshots" not in sv.scanned_sources
    assert "no snapshot observations were provided" in sv.detail


def test_scanned_sources_reported():
    trace = [_ins(0, "str x9, [x12]", mem=(MemOp("w", OUT, _le(EXPECTED), 4),))]
    with_snap = validate_sink(trace, EXPECTED, snapshots=[MemSnapshot(OUT, EXPECTED)])
    assert with_snap.scanned_sources == ("writes", "reads", "snapshots")
    without = validate_sink(trace, EXPECTED)
    assert without.scanned_sources == ("writes", "reads")


# --- task 8: candidate-write fail must not ban same-base snapshot/read ------

def test_candidate_write_fails_but_same_base_snapshot_confirms_sink():
    # ① the candidate WRITE at the base reconstructs WRONG bytes (scratch/partial
    #    write at that base), but a snapshot at the SAME base fully reconstructs
    #    expected. Must be SINK_CONFIRMED via snapshot, NOT OUTPUT_NOT_OBSERVABLE.
    trace = [_ins(0, "str x8, [x12]", mem=(MemOp("w", OUT, _le(SCRATCH), 4),))]
    snaps = [MemSnapshot(addr=OUT, data=EXPECTED, label="output_buffer")]
    sv = validate_sink(trace, EXPECTED, candidate_idxs=[0], snapshots=snaps)
    assert sv.verdict is SinkVerdict.SINK_CONFIRMED
    assert sv.located_via == "snapshot"
    assert sv.base == OUT
    assert sv.candidate_base == OUT


def test_candidate_write_fails_but_same_base_read_confirms_sink():
    # same as above but the independent observation source is a READ at cand_base.
    trace = [
        _ins(0, "str x8, [x12]", mem=(MemOp("w", OUT, _le(SCRATCH), 4),)),  # bad write
        _ins(1, "ldr x9, [x12]", mem=(MemOp("r", OUT, _le(EXPECTED), 4),)),  # read sees real
    ]
    sv = validate_sink(trace, EXPECTED, candidate_idxs=[0])
    assert sv.verdict is SinkVerdict.SINK_CONFIRMED
    assert sv.located_via == "read"
    assert sv.base == OUT


def test_same_base_snapshot_only_partial_still_not_observable():
    # ② the snapshot at cand_base reconstructs only PART of expected (<L) ->
    #    must NOT be confirmed; falls through to OUTPUT_NOT_OBSERVABLE old path.
    trace = [_ins(0, "str x8, [x12]", mem=(MemOp("w", OUT, _le(SCRATCH), 4),))]
    # snapshot holds only 2 of the 4 expected bytes at OUT (partial coverage)
    snaps = [MemSnapshot(addr=OUT, data=EXPECTED[:2], label="partial")]
    sv = validate_sink(trace, EXPECTED, candidate_idxs=[0], snapshots=snaps)
    assert sv.verdict is SinkVerdict.OUTPUT_NOT_OBSERVABLE


def test_no_snapshot_only_failed_write_behavior_unchanged():
    # ③ no snapshot, candidate write fails, expected nowhere -> unchanged
    #    OUTPUT_NOT_OBSERVABLE (zero regression vs prior behavior).
    trace = [_ins(0, "str x8, [x12]", mem=(MemOp("w", OUT, _le(SCRATCH), 4),))]
    sv = validate_sink(trace, EXPECTED, candidate_idxs=[0])
    assert sv.verdict is SinkVerdict.OUTPUT_NOT_OBSERVABLE
    assert sv.candidate_base == OUT


def test_gate_confirmed_via_same_base_snapshot_does_not_block():
    # ④ task-6 link: adapter emits a snapshot at the candidate base; the gate
    #    confirms the sink via it and does NOT raise / does NOT recapture.
    trace = [_ins(0, "str x8, [x12]", mem=(MemOp("w", OUT, _le(SCRATCH), 4),))]
    snaps = [MemSnapshot(addr=OUT, data=EXPECTED)]
    sinks, sv, redirect = apply_sink_gate(
        trace, [0], EXPECTED, snapshots=snaps, value_name="output")
    assert sv.verdict is SinkVerdict.SINK_CONFIRMED
    assert sv.located_via == "snapshot"
    assert sinks == [0]
    assert redirect is None


# --- verdict -> action policy (the value-delivering part) -------------------

def test_gate_confirmed_leaves_sinks_unchanged():
    trace = [_ins(0, "str x9, [x12]", mem=(MemOp("w", OUT, _le(EXPECTED), 4),))]
    sinks, sv, redirect = apply_sink_gate(trace, [0], EXPECTED)
    assert sv.verdict is SinkVerdict.SINK_CONFIRMED
    assert sinks == [0]
    assert redirect is None


def test_gate_wrong_sink_redirects_to_real_sink():
    trace = [
        _ins(0, "str x8, [x10]", mem=(MemOp("w", SCR, _le(SCRATCH), 4),)),   # idx0 scratch
        _ins(1, "str x9, [x12]", mem=(MemOp("w", OUT, _le(EXPECTED), 4),)),  # idx1 real
    ]
    sinks, sv, redirect = apply_sink_gate(trace, [0], EXPECTED)  # caller passed scratch
    assert sv.verdict is SinkVerdict.WRONG_SINK
    assert sinks == [1]                          # REDIRECTED to the real sink instr
    assert redirect["from"] == [0] and redirect["to"] == [1]
    assert redirect["located_base"] == f"0x{OUT:x}"


def test_gate_not_observable_blocks_with_exception():
    trace = [_ins(0, "str x8, [x10]", mem=(MemOp("w", SCR, _le(SCRATCH), 4),))]
    with pytest.raises(SinkGateError) as ei:
        apply_sink_gate(trace, [0], EXPECTED)
    assert ei.value.validation.verdict is SinkVerdict.OUTPUT_NOT_OBSERVABLE


def test_gate_records_verdict_to_disk_before_acting():
    trace = [_ins(0, "str x8, [x10]", mem=(MemOp("w", SCR, _le(SCRATCH), 4),))]
    with tempfile.TemporaryDirectory() as td:
        rec = Path(td) / "s4_sink_validation.json"
        with pytest.raises(SinkGateError):
            apply_sink_gate(trace, [0], EXPECTED, record_to=rec)
        # verdict was recorded even though the gate then blocked
        assert rec.exists()
        import json
        assert json.loads(rec.read_text())["verdict"] == "OUTPUT_NOT_OBSERVABLE"


def test_no_hardcoded_address_in_module():
    import inspect
    import re
    from engine import oracle_sink
    big = re.findall(r"0x[0-9a-fA-F]{4,}", inspect.getsource(oracle_sink))
    assert big == [], f"unexpected hardcoded address literal(s): {big}"
