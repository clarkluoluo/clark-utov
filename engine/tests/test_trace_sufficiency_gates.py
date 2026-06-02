"""Item ④ — analysis layers consume the ③ profile before concluding.

Synthetic, zero case-specific. Verifies the trust-gate has BOTH directions
(sufficient -> original conclusion; insufficient -> inconclusive + precise
reason), oracle_provenance reports output-not-observed via sink_captured, and the
default (no opt-in) path is byte-for-byte unchanged.
"""

from __future__ import annotations

import dataclasses

from engine.oracle_provenance import ProvenanceVerdict, trace_provenance
from engine.opaque_staging import VERDICT_INCONCLUSIVE, VERDICT_KNOWN_ADDR, diagnose_opaque_staging
from engine.setup_symex import derive_window_symbolic_regs
from engine.types import Instruction, MemOp


def ins(idx, pc, mnem="nop", *, reads=None, writes=None, mem=()):
    return Instruction(
        idx=idx, pc=pc, bytes_=b"\x00\x00\x00\x00", mnemonic=mnem,
        regs_read=reads or {}, regs_write=writes or {}, mem=tuple(mem))


_SINK = 0x6000
_STAGING = 0x10020


# --- opaque_staging gate: positive vs negative -------------------------------

def test_opaque_gate_sufficient_keeps_conclusion():
    items = [
        ins(0, 0x1000, "str x8,[x10]", reads={"x8": 0x41, "x10": _STAGING},
            writes={"x8": 0x41}, mem=[MemOp("w", _STAGING, 0x41, 8)]),
        ins(1, 0x1004, "ldr x9,[x10]", reads={"x10": _STAGING},
            writes={"x9": 0x41}, mem=[MemOp("r", _STAGING, 0x41, 8)]),
    ]
    diag = diagnose_opaque_staging(items, window=(0, 1), window_is_idx=True,
                                   symbolic_inputs=("x8",))
    assert diag.verdict == VERDICT_KNOWN_ADDR     # well-populated -> original path


def test_opaque_gate_insufficient_inconclusive():
    items = [
        ins(0, 0x1000, "str x8,[x10]", reads={"x8": 0x41, "x10": _STAGING},
            mem=[MemOp("w", _STAGING, 0x41, 8)]),
        ins(1, 0x1004, "ldr x9,[x10]", reads={"x10": _STAGING},
            mem=[MemOp("r", _STAGING, 0x41, 8)]),    # regs_write empty
    ]
    diag = diagnose_opaque_staging(items, window=(0, 1), window_is_idx=True)
    assert diag.verdict == VERDICT_INCONCLUSIVE
    assert any("regs_write coverage" in r for r in diag.reasons)


# --- derive_window_symbolic_regs readiness note ------------------------------

def test_derive_readiness_sufficient():
    items = [
        ins(0, 0x1000, writes={"x1": 1}, reads={"x0": 5}),
        ins(1, 0x1004, writes={"x2": 2}, reads={"x1": 1}),
    ]
    _live, info = derive_window_symbolic_regs(items, window=(0, 1), window_is_idx=True)
    assert info["regs_write_sufficient"] is True
    assert info["readiness_note"] == ""


def test_derive_readiness_low_regs_write_flags_inconclusive():
    items = [ins(i, 0x1000 + i, reads={"x9": 0x100 + i}) for i in range(20)]  # no writes
    _live, info = derive_window_symbolic_regs(items, window=(0, 19), window_is_idx=True)
    assert info["regs_write_sufficient"] is False
    assert "INCONCLUSIVE" in info["readiness_note"]
    assert info["n_inferred_edges"] >= 1   # x9 value changes -> inferred edges


# --- oracle_provenance sink_captured (④) -------------------------------------

def test_oracle_default_unchanged_no_sink_captured_key():
    items = [ins(0, 0x1000, "str x8,[x9]", mem=[MemOp("w", _SINK, 0xAB, 1)])]
    res = trace_provenance(items, b"\xab", sink_base=_SINK)
    assert res.sink_captured is None              # not assessed by default
    assert "sink_captured" not in res.to_dict()   # serialization unchanged


def test_oracle_reports_output_not_observed():
    # The sink is never written/snapshotted -> sink_captured False = needs re-capture.
    items = [ins(0, 0x1000, "nop")]
    res = trace_provenance(items, b"\xab", sink_base=_SINK, assess_observability=True)
    assert res.verdict == ProvenanceVerdict.NEEDS_OBSERVATION
    assert res.sink_captured is False
    assert res.to_dict()["sink_captured"] is False


def test_oracle_sink_captured_true_when_written():
    items = [ins(0, 0x1000, "str x8,[x9]", mem=[MemOp("w", _SINK, 0xAB, 1)])]
    res = trace_provenance(items, b"\xab", sink_base=_SINK, assess_observability=True)
    assert res.sink_captured is True
    assert res.verdict == ProvenanceVerdict.CONTINUOUS_BUFFER   # conclusion intact


def test_oracle_opt_in_does_not_change_verdict():
    items = [ins(0, 0x1000, "str x8,[x9]", mem=[MemOp("w", _SINK, 0xAB, 1)])]
    plain = trace_provenance(items, b"\xab", sink_base=_SINK)
    gated = trace_provenance(items, b"\xab", sink_base=_SINK, assess_observability=True)
    # same verdict + chain; the only delta is the additive sink_captured field.
    assert plain.verdict == gated.verdict
    assert dataclasses.replace(gated, sink_captured=None) == plain
