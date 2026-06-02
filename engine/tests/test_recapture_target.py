"""Recapture-target orchestration — auto-produce a reg-relative recapture
directive when the target output buffer was never observed.

Covers (spec fixtures):
  ① target unobserved + an observed pointer reaches the buffer -> reg-relative
     WatchFirstWriteSpec + recapture_directive covering the target full length.
  ② degrade: no pointer reaches the buffer -> explicit cannot-derive report.
  ③ invariant 7: the concrete WatchFirstWriteSpec path is byte-unchanged.
  ④ the auto-derivation only fires on NEEDS_OBSERVATION / OUTPUT_NOT_OBSERVABLE.
Synthetic traces only; zero target-specific value baked in.
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
from engine.recapture_target import (
    RecaptureTargetStatus,
    derive_recapture_directive,
)
from engine.types import Instruction, MemOp
from engine.watch_first_write import (
    WatchFirstWriteConfig,
    WatchFirstWriteSpec,
    request_watch_first_write,
)


def _ins(idx, mnem, *, regs_write=None, mem=()):
    return Instruction(idx=idx, pc=0x70000 + idx * 4, bytes_=b"\x00\x00\x00\x00",
                       mnemonic=mnem, regs_read={}, regs_write=regs_write or {}, mem=mem)


def _le(b: bytes) -> int:
    return int.from_bytes(b, "little")


# The target output (oracle) — 8 bytes; the buffer is only PARTIALLY observed
# on the deriving run (heap address, not stable across runs).
TARGET = bytes([0xde, 0xad, 0xbe, 0xef, 0x01, 0x02, 0x03, 0x04])
BUF_BASE = 0x7f00_1240        # where the target lands on THIS run (heap)
PTR_VALUE = 0x7f00_1200       # a pointer register held this; offset = 0x40


# ---------------------------------------------------------------------------
# Invariant 7 — concrete WatchFirstWriteSpec path byte-identical
# ---------------------------------------------------------------------------

def test_concrete_spec_dict_byte_identical():
    """A concrete (base_reg=None) spec emits EXACTLY the original dict — no new
    keys — so today's runner contract is unchanged."""
    spec = request_watch_first_write(0xbadc0de, "v", cfg=WatchFirstWriteConfig())
    assert spec.base_reg is None
    assert spec.is_reg_relative is False
    assert spec.to_dict() == {
        "addr": 0xbadc0de,
        "addr_hex": "0xbadc0de",
        "width_bytes": 8,
        "value_name": "v",
        "reason": "trace producer of observed value",
        "kind": "watch_first_write",
    }
    assert spec.addr_expr == "0xbadc0de"


def test_reg_relative_spec_dict_carries_expression():
    spec = request_watch_first_write(
        BUF_BASE, "cipher_body",
        base_reg="x25", offset=0x40, width_bytes=65,
        cfg=WatchFirstWriteConfig())
    assert spec.is_reg_relative is True
    assert spec.addr_expr == "[x25 + 0x40]"
    d = spec.to_dict()
    assert d["addressing"] == "reg_relative"
    assert d["base_reg"] == "x25"
    assert d["offset"] == 0x40
    assert d["addr_expr"] == "[x25 + 0x40]"
    assert d["width_bytes"] == 65            # covers target full length
    # addr still carried (diagnostic-only observed address)
    assert d["addr"] == BUF_BASE


def test_reg_relative_rejects_empty_base_reg():
    with pytest.raises(ValueError):
        request_watch_first_write(0x1000, "v", base_reg="", cfg=WatchFirstWriteConfig())


def test_reg_relative_negative_offset_expr():
    spec = request_watch_first_write(0x1000, "v", base_reg="x0", offset=-0x10,
                                     cfg=WatchFirstWriteConfig())
    assert spec.addr_expr == "[x0 - 0x10]"


# ---------------------------------------------------------------------------
# ① target unobserved + observed pointer reaches buffer -> reg-relative spec
# ---------------------------------------------------------------------------

def _unobserved_target_trace():
    """A trace where:
      - a register (x25) is observed holding PTR_VALUE (the buffer pointer),
      - the buffer at BUF_BASE is only PARTIALLY written (target not captured).
    So validate_sink returns OUTPUT_NOT_OBSERVABLE with a longest_partial anchor.
    """
    # Partially write the first 3 target bytes at BUF_BASE (partial observation).
    # Only x25 (the BASE pointer) is observed; the buffer-base address itself is
    # never materialised into a register — exactly the F0 [x25 + off] shape.
    partial = bytes(TARGET[:3])
    return [
        _ins(0, "mov x25, x9", regs_write={"x25": PTR_VALUE}),          # base pointer
        _ins(1, "str x8, [x25, #0x40]",
             mem=(MemOp("w", BUF_BASE, _le(partial), 3),)),             # partial write
    ]


def test_derive_reg_relative_when_pointer_reaches_buffer():
    items = _unobserved_target_trace()
    sv = validate_sink(items, TARGET)
    assert sv.verdict is SinkVerdict.OUTPUT_NOT_OBSERVABLE   # target not captured

    d = derive_recapture_directive(items, TARGET, "cipher_body", sink_validation=sv)
    assert d.status == RecaptureTargetStatus.DERIVED
    assert d.derived is True
    assert d.buffer_base == BUF_BASE
    assert d.pointer is not None
    # nearest observed pointer is x25 (value PTR_VALUE), offset = 0x40
    assert d.pointer.reg == "x25"
    assert d.pointer.reg_value == PTR_VALUE
    assert d.pointer.offset == 0x40

    spec = d.spec
    assert isinstance(spec, WatchFirstWriteSpec)
    assert spec.is_reg_relative
    assert spec.base_reg == "x25"
    assert spec.offset == 0x40
    assert spec.width_bytes == len(TARGET)          # covers target FULL length
    assert spec.addr_expr == "[x25 + 0x40]"

    dd = d.to_dict()
    assert dd["kind"] == "recapture_directive"
    assert dd["spec"]["addressing"] == "reg_relative"
    assert dd["target_length"] == len(TARGET)


def test_offset_zero_when_pointer_is_buffer_base_itself():
    items = [
        _ins(0, "mov x0, x9", regs_write={"x0": BUF_BASE}),   # reg holds base exactly
        _ins(1, "str x8, [x0]", mem=(MemOp("w", BUF_BASE, _le(TARGET[:2]), 2),)),
    ]
    sv = validate_sink(items, TARGET)
    d = derive_recapture_directive(items, TARGET, "out", sink_validation=sv)
    assert d.status == RecaptureTargetStatus.DERIVED
    assert d.pointer.offset == 0
    assert d.spec.addr_expr == "[x0]"


# ---------------------------------------------------------------------------
# ② degrade — no observed pointer reaches the buffer -> explicit cannot-derive
# ---------------------------------------------------------------------------

def test_degrade_no_pointer_reaches_buffer():
    """Buffer is located (partial write) but NO register ever held a value within
    range of the base. Explicit cannot-derive, spec=None, not silent."""
    items = [
        # write the partial buffer, but no register holds anything near BUF_BASE
        _ins(0, "str x8, [x10]", mem=(MemOp("w", BUF_BASE, _le(TARGET[:3]), 3),)),
    ]
    sv = validate_sink(items, TARGET)
    assert sv.verdict is SinkVerdict.OUTPUT_NOT_OBSERVABLE

    d = derive_recapture_directive(items, TARGET, "out",
                                   sink_validation=sv, max_offset=0x100)
    assert d.status == RecaptureTargetStatus.NO_POINTER
    assert d.derived is False
    assert d.spec is None
    assert d.buffer_base == BUF_BASE
    assert "no observed pointer reaches the buffer" in d.detail


def test_degrade_buffer_not_located_anywhere():
    """The target bytes appear NOWHERE on this run -> no buffer base to anchor.
    Explicit cannot-derive, not silent."""
    items = [
        _ins(0, "str x8, [x10]",
             mem=(MemOp("w", 0x9000, _le(b"\xaa\xbb\xcc\xdd"), 4),)),  # unrelated bytes
    ]
    sv = validate_sink(items, TARGET)
    assert sv.verdict is SinkVerdict.OUTPUT_NOT_OBSERVABLE
    d = derive_recapture_directive(items, TARGET, "out", sink_validation=sv)
    assert d.status == RecaptureTargetStatus.BUFFER_NOT_LOCATED
    assert d.spec is None
    assert d.buffer_base is None
    assert "cannot derive recapture target" in d.detail


# ---------------------------------------------------------------------------
# Invariant 8 — derived reg/offset come from REAL dataflow, never fabricated
# ---------------------------------------------------------------------------

def test_offset_derivation_from_real_dataflow():
    """The offset must equal buffer_base - the actual observed register value —
    it is read off the trace, never invented."""
    items = _unobserved_target_trace()
    sv = validate_sink(items, TARGET)
    d = derive_recapture_directive(items, TARGET, "v", sink_validation=sv)
    assert d.pointer.offset == d.buffer_base - d.pointer.reg_value


# ---------------------------------------------------------------------------
# ④ auto-derive in apply_sink_gate ONLY on OUTPUT_NOT_OBSERVABLE
# ---------------------------------------------------------------------------

def test_gate_attaches_directive_on_unobservable():
    items = _unobserved_target_trace()
    with tempfile.TemporaryDirectory() as tmp:
        rec = Path(tmp) / "gap.json"
        with pytest.raises(SinkGateError) as ei:
            apply_sink_gate(items, sinks=[], expected_output=TARGET,
                            record_to=rec, value_name="cipher_body")
        err = ei.value
        assert err.recapture_directive is not None
        assert err.recapture_directive["status"] == RecaptureTargetStatus.DERIVED
        assert err.recapture_directive["spec"]["base_reg"] == "x25"
        # the recorded gap-map JSON carries it too
        import json
        payload = json.loads(rec.read_text())
        assert payload["verdict"] == "OUTPUT_NOT_OBSERVABLE"
        assert payload["recapture_directive"]["spec"]["addressing"] == "reg_relative"


def test_gate_no_directive_without_value_name_invariant7():
    """Invariant 7: omitting value_name -> the gate behaves byte-for-byte as
    before (no directive, error has none)."""
    items = _unobserved_target_trace()
    with tempfile.TemporaryDirectory() as tmp:
        rec = Path(tmp) / "gap.json"
        with pytest.raises(SinkGateError) as ei:
            apply_sink_gate(items, sinks=[], expected_output=TARGET, record_to=rec)
        assert ei.value.recapture_directive is None
        import json
        payload = json.loads(rec.read_text())
        assert "recapture_directive" not in payload


def test_gate_does_not_derive_when_sink_confirmed():
    """④ auto-derive does NOT fire when the sink IS observed (SINK_CONFIRMED) —
    no NEEDS_OBSERVATION, nothing to recapture; verdict unchanged."""
    items = [_ins(0, "str x8, [x10]", mem=(MemOp("w", BUF_BASE, _le(TARGET), 8),))]
    # full target captured -> SINK_CONFIRMED, no raise, no directive
    eff, sv, redirect = apply_sink_gate(
        items, sinks=[], expected_output=TARGET, value_name="out")
    assert sv.verdict is SinkVerdict.SINK_CONFIRMED


def test_direct_derive_requires_non_empty_target():
    with pytest.raises(ValueError):
        derive_recapture_directive([], b"", "v")


# ---------------------------------------------------------------------------
# Task 4 — coverage / confidence gate: a too-weak partial match must NOT派 a
# (confidently-wrong) reg-relative watch; surface INSUFFICIENT_COVERAGE instead.
# ---------------------------------------------------------------------------

# A 65-byte target (the F0 cipher_body length) — only 3 bytes match anywhere ⇒
# the 3/65 (≈4.6%) anchor the spec calls out as "likely the WRONG buffer".
TARGET_65 = bytes(range(1, 66))
BIG_BUF = 0x7f00_2000


def _weak_partial_trace():
    """A run where only the first 3 of the 65 target bytes land at BIG_BUF, with a
    pointer register reaching the base — DERIVE-able geometry, but the partial match
    is far too thin to trust as the target buffer."""
    return [
        _ins(0, "mov x25, x9", regs_write={"x25": BIG_BUF}),
        _ins(1, "str x8, [x25]",
             mem=(MemOp("w", BIG_BUF, _le(bytes(TARGET_65[:3])), 3),)),
    ]


def test_weak_partial_match_yields_insufficient_coverage_not_derive():
    items = _weak_partial_trace()
    sv = validate_sink(items, TARGET_65)
    assert sv.verdict is SinkVerdict.OUTPUT_NOT_OBSERVABLE
    d = derive_recapture_directive(items, TARGET_65, "cipher_body", sink_validation=sv)
    # 3/65 ≈ 4.6% < the 0.25 default floor → NO watch派, explicit state instead.
    assert d.status == RecaptureTargetStatus.INSUFFICIENT_COVERAGE
    assert d.spec is None
    assert d.derived is False
    assert d.coverage["match_count"] == 3
    assert d.coverage["length"] == 65
    assert d.coverage["coverage"] < 0.25
    assert "too weak to trust" in d.detail
    dd = d.to_dict()
    assert dd["status"] == "INSUFFICIENT_COVERAGE"
    assert dd["coverage"]["match_count"] == 3


def test_threshold_parameterised_by_total_can_admit_weak_partial():
    """The threshold is a RATIO and is overridable — driving it low admits the
    same 3/65 partial (proves it is data-driven, not a hardcoded 65/3)."""
    items = _weak_partial_trace()
    sv = validate_sink(items, TARGET_65)
    d = derive_recapture_directive(
        items, TARGET_65, "cipher_body", sink_validation=sv,
        min_partial_coverage=0.01)          # accept anything > 1%
    assert d.status == RecaptureTargetStatus.DERIVED
    assert d.spec is not None
    assert d.coverage["match_count"] == 3


def test_strong_partial_still_derives_not_mis_killed():
    """验收② — a genuinely-partial buffer (well above the floor) still派s a watch:
    the gate does not over-fire."""
    # 6/8 (75%) of an 8-byte target match — a real partially-observed buffer.
    items = [
        _ins(0, "mov x25, x9", regs_write={"x25": BUF_BASE}),
        _ins(1, "str x8, [x25]", mem=(MemOp("w", BUF_BASE, _le(TARGET[:6]), 6),)),
    ]
    sv = validate_sink(items, TARGET)
    d = derive_recapture_directive(items, TARGET, "out", sink_validation=sv)
    assert d.status == RecaptureTargetStatus.DERIVED
    assert d.coverage["coverage"] >= 0.5


# ---------------------------------------------------------------------------
# Point-watch preference (spec ①②) — clean PC-gated single-point capture when
# the arm PC is known; explicitly-marked NOISY wide-range fallback otherwise.
# ---------------------------------------------------------------------------

def test_point_watch_emitted_when_arm_pc_known():
    """① given (pc, base_reg, offset, width) all known -> a point-watch directive
    (NOT a wide range): kind=point_watch, watch_kind + pc on the spec, no risk."""
    items = _unobserved_target_trace()
    sv = validate_sink(items, TARGET)
    ARM_PC = 0x70ec4
    d = derive_recapture_directive(
        items, TARGET, "cipher_body", sink_validation=sv,
        point_watch_pc=ARM_PC, point_watch_kind="read")
    assert d.status == RecaptureTargetStatus.DERIVED
    assert d.capture_mode == "point_watch"
    assert d.is_point_watch is True
    assert d.capture_risk is None                       # clean: no noise/cap risk
    spec = d.spec
    assert spec.is_point_watch is True
    assert spec.pc == ARM_PC
    assert spec.kind == "read"
    assert spec.base_reg == "x25"
    assert spec.offset == 0x40
    assert spec.width_bytes == len(TARGET)              # single point, full width
    dd = d.to_dict()
    assert dd["capture_mode"] == "point_watch"
    assert "capture_risk" not in dd                     # absent on clean path
    assert dd["spec"]["kind"] == "point_watch"
    assert dd["spec"]["watch_kind"] == "read"
    assert dd["spec"]["pc_hex"] == "0x70ec4"


def test_point_watch_write_direction():
    items = _unobserved_target_trace()
    sv = validate_sink(items, TARGET)
    d = derive_recapture_directive(
        items, TARGET, "v", sink_validation=sv,
        point_watch_pc=0x1234, point_watch_kind="write")
    assert d.capture_mode == "point_watch"
    assert d.spec.kind == "write"


def test_wide_range_fallback_explicitly_marks_noise_and_cap_risk():
    """② arm PC NOT known -> wide reg-relative range fallback, EXPLICITLY marked
    noisy + possible cap hit (never silent degradation)."""
    items = _unobserved_target_trace()
    sv = validate_sink(items, TARGET)
    d = derive_recapture_directive(items, TARGET, "cipher_body", sink_validation=sv)
    assert d.status == RecaptureTargetStatus.DERIVED
    assert d.capture_mode == "reg_relative_range"
    assert d.is_point_watch is False
    # The risk annotation must be present and name both hazards (noise + cap).
    assert d.capture_risk is not None
    assert "NOISY" in d.capture_risk
    assert "record cap" in d.capture_risk
    # And the spec is the plain reg-relative (first-write) watch, not a point-watch.
    assert d.spec.is_point_watch is False
    assert d.spec.is_reg_relative is True
    dd = d.to_dict()
    assert dd["capture_mode"] == "reg_relative_range"
    assert "NOISY" in dd["capture_risk"]


def test_point_watch_rejects_bad_kind():
    items = _unobserved_target_trace()
    with pytest.raises(ValueError):
        derive_recapture_directive(items, TARGET, "v",
                                   point_watch_pc=0x1000, point_watch_kind="first_write")


def test_point_watch_rejects_negative_pc():
    items = _unobserved_target_trace()
    with pytest.raises(ValueError):
        derive_recapture_directive(items, TARGET, "v", point_watch_pc=-1)


def test_point_watch_pc_does_not_rescue_no_pointer():
    """Supplying an arm PC does NOT fabricate a pointer: if no observed pointer
    reaches the buffer, it still degrades to NO_POINTER (spec: never fabricate)."""
    items = [
        _ins(0, "str x8, [x10]", mem=(MemOp("w", BUF_BASE, _le(TARGET[:3]), 3),)),
    ]
    sv = validate_sink(items, TARGET)
    d = derive_recapture_directive(items, TARGET, "out", sink_validation=sv,
                                   max_offset=0x100, point_watch_pc=0x70ec4)
    assert d.status == RecaptureTargetStatus.NO_POINTER
    assert d.spec is None
