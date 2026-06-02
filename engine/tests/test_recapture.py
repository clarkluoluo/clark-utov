"""recapture front half (no side effects): prefill observe_points from a #3
provenance gap, validate the spec, and fold a runner result back into snapshots —
demonstrated end-to-end with a FAKE adapter (no real runner, no target run).
"""

from __future__ import annotations

import pytest

from engine.oracle_provenance import ProvenanceVerdict, trace_provenance
from engine.recapture import (
    AARCH64_ARG_REGS,
    RecaptureSpec,
    dispatch_recapture,
    observations_to_snapshots,
    observe_points_from_provenance,
    plan_recapture,
    validate_spec,
)
from engine.runner_client import ObservedState, ObservePoint, RerunResult
from engine.types import Instruction, MemOp, MemSnapshot


def _ins(idx, mnem, reads=None, writes=None, mem=()):
    return Instruction(idx=idx, pc=0x70000 + idx * 4, bytes_=b"\x00\x00\x00\x00",
                       mnemonic=mnem, regs_read=dict(reads or {}),
                       regs_write=dict(writes or {}), mem=mem)


def _le(b: bytes) -> int:
    return int.from_bytes(b, "little")


EXPECTED = bytes([0x34, 0x15, 0x5f, 0xe9])
OUT = 0x72b18      # located sink base
UNK = 0x9000       # an un-captured native address read by the producer chain


# --- prefill: NEEDS_OBSERVATION -> mem ObservePoints at the reading PC -------

def _needs_observation_trace():
    # A traced str DOES write the sink, but the value it stores is WRONG (it never
    # reconstructs to expected) because the real value is sourced from a load of an
    # un-captured native address UNK -> NEEDS_OBSERVATION with a next_watch on UNK.
    return [
        _ins(0, "ldr x8, [x9]", reads={"x9": UNK}, writes={"x8": 0},
             mem=(MemOp("r", UNK, 0, 4),)),
        _ins(1, "str x8, [x10]", reads={"x8": 0, "x10": OUT},
             mem=(MemOp("w", OUT, 0, 4),)),
    ]


def test_prefill_needs_observation_makes_mem_points_at_reading_pc():
    trace = _needs_observation_trace()
    prov = trace_provenance(trace, EXPECTED, sink_base=OUT)
    assert prov.verdict is ProvenanceVerdict.NEEDS_OBSERVATION
    assert prov.next_watch and prov.next_watch[0]["addr"] == f"0x{UNK:x}"

    pre = observe_points_from_provenance(prov)
    assert len(pre.observe_points) == 1
    op = pre.observe_points[0]
    assert op.when == "before"
    assert op.capture == ("mem",)
    # 4 contiguous watched bytes coalesce into one (addr, size=4) range
    assert op.mem == ((UNK, 4),)
    # the point is hung on the PC that READS the gap (the ldr at idx 0)
    assert op.pc == 0x70000
    assert pre.unplaceable_addrs == ()


def test_prefill_opaque_callee_captures_arg_regs_at_boundary():
    # sink never written by any traced instr; appears only AFTER a bl returns.
    trace = [
        _ins(0, "bl #0x221068", reads={}),
        _ins(1, "ldr x0, [x10]", reads={"x10": OUT}, mem=(MemOp("r", OUT, _le(EXPECTED), 4),)),
    ]
    prov = trace_provenance(trace, EXPECTED, sink_base=OUT)
    assert prov.verdict is ProvenanceVerdict.OPAQUE_CALLEE

    pre = observe_points_from_provenance(prov)
    assert len(pre.observe_points) == 1
    op = pre.observe_points[0]
    assert op.pc == 0x70000              # the call boundary
    assert op.when == "before"           # capture the call-time INPUTS
    assert set(op.regs) == set(AARCH64_ARG_REGS)


def test_prefill_unplaceable_when_gap_has_no_reading_pc():
    # NEEDS_OBSERVATION (streaming-unprovable) path yields next_watch entries with
    # pc=None — they cannot become a code hook and are reported as unplaceable.
    trace = [_ins(0, "nop", reads={})]
    prov = trace_provenance(trace, EXPECTED, sink_base=OUT)
    assert prov.verdict is ProvenanceVerdict.NEEDS_OBSERVATION
    assert all(w["pc"] is None for w in prov.next_watch)

    pre = observe_points_from_provenance(prov)
    assert pre.observe_points == []
    assert pre.unplaceable_addrs == tuple(OUT + i for i in range(len(EXPECTED)))


# --- validation: pure diagnosis, never a run --------------------------------

def test_validate_flags_pc_not_in_trace_and_bad_window():
    trace = _needs_observation_trace()
    spec = RecaptureSpec(
        input=b"\x01",
        window=(0x80000, 0x70000),                         # front >= rear -> error
        observe_points=[ObservePoint(pc=0xDEAD, when="before",
                                     capture=("mem",), mem=((UNK, 4),))])  # pc absent
    rep = validate_spec(spec, trace)
    assert rep.ok is False
    codes = {f["code"] for f in rep.findings}
    assert "bad_window" in codes
    assert "pc_not_in_trace" in codes


def test_validate_focus_not_in_trace_is_warning_not_error():
    trace = _needs_observation_trace()
    spec = RecaptureSpec(input=b"\x01", focus_pcs=(0x221068,),
                         observe_points=[])
    rep = validate_spec(spec, trace)
    # an un-traced callee focus is expected for OPAQUE_CALLEE -> must not block
    assert rep.ok is True
    assert any(f["code"] == "focus_not_in_trace" and f["severity"] == "warning"
               for f in rep.findings)


def test_validate_empty_input_is_error():
    rep = validate_spec(RecaptureSpec(input=b""), [_ins(0, "nop")])
    assert rep.ok is False
    assert any(f["code"] == "no_input" for f in rep.findings)


# --- plan: prefill + validate together, no side effects ---------------------

def test_plan_recapture_prefills_focus_and_validates():
    trace = _needs_observation_trace()
    prov = trace_provenance(trace, EXPECTED, sink_base=OUT)
    plan = plan_recapture(prov, b"\x01\x02", window=(0x70000, 0x72b18), items=trace)
    assert plan.validation is not None and plan.validation.ok is True
    assert plan.spec.observe_points[0].mem == ((UNK, 4),)
    assert plan.spec.window == (0x70000, 0x72b18)


# --- ingest + round trip with a FAKE adapter (no real runner) ---------------

class _FakeAdapter:
    """Stands in for a Live-mode runner: at the requested observe point it hands
    back the bytes that were missing — proving the ingest half closes the loop."""
    def rerun(self, input_bytes, observe_points=None):
        obs = ObservedState(pc=0x70000, when="before", regs={},
                            mem={UNK: EXPECTED})
        return RerunResult(output=EXPECTED, observations=(obs,))


def test_observations_to_snapshots_closes_the_watch_gap():
    trace = _needs_observation_trace()
    prov = trace_provenance(trace, EXPECTED, sink_base=OUT)
    assert prov.verdict is ProvenanceVerdict.NEEDS_OBSERVATION
    assert any(w["addr"] == f"0x{UNK:x}" for w in prov.next_watch)

    # the deferred dispatch would call adapter.rerun(spec.input, spec.observe_points);
    # here we simulate exactly that hand-off with the fake adapter.
    plan = plan_recapture(prov, b"\x01", items=trace)
    result = _FakeAdapter().rerun(plan.spec.input, plan.spec.observe_points)
    snaps = observations_to_snapshots(result)
    assert snaps == [MemSnapshot(addr=UNK, data=EXPECTED,
                                 label="recapture@0x70000:before", source="recapture")]

    # feeding the recaptured snapshot back, the UNK producer gap is now CLOSED —
    # it no longer appears in next_watch. (The sink buffer in THIS trace still holds
    # the wrong bytes, so the verdict stays NEEDS_OBSERVATION; what recapture buys is
    # the closed gap — honest about what one observation does.)
    re = trace_provenance(trace, EXPECTED, sink_base=OUT, snapshots=snaps)
    assert all(w["addr"] != f"0x{UNK:x}" for w in re.next_watch)


def test_dispatch_is_deferred():
    trace = _needs_observation_trace()
    prov = trace_provenance(trace, EXPECTED, sink_base=OUT)
    plan = plan_recapture(prov, b"\x01", items=trace)
    with pytest.raises(NotImplementedError):
        dispatch_recapture(plan, _FakeAdapter())
