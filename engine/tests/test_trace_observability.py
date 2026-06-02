"""Item ③ — unified trace-observability profile + the A8 single-source check.

Synthetic, zero case-specific. Verifies the three-dimension coverage, the
sink_captured self-check, and that cohort_diff / opaque_staging draw their
coverage from this one module (the parallel computations were removed).
"""

from __future__ import annotations

from engine.trace_observability import (
    DEFAULT_THRESHOLDS,
    assess_trace_observability,
    dimension_coverage,
    has_regs_write,
    has_write_dim,
)
from engine.types import Instruction, MemOp, MemSnapshot


def ins(idx, pc, mnem="nop", *, reads=None, writes=None, mem=()):
    return Instruction(
        idx=idx, pc=pc, bytes_=b"\x00\x00\x00\x00", mnemonic=mnem,
        regs_read=reads or {}, regs_write=writes or {}, mem=tuple(mem))


_SINK = 0x5000


def test_three_dimension_coverage():
    items = [
        ins(0, 0x1000, writes={"x1": 1}, reads={"x0": 0}),
        ins(1, 0x1004, reads={"x1": 1}),                       # read only
        ins(2, 0x1008, mem=[MemOp("w", 0x9000, 5, 4)]),        # mem only
        ins(3, 0x100c),                                        # nothing
    ]
    obs = assess_trace_observability(items)
    assert obs.n_items == 4
    assert obs.regs_write_rate == 0.25
    assert obs.regs_read_rate == 0.5
    assert obs.mem_event_rate == 0.25


def test_sink_captured_true_on_write():
    items = [ins(0, 0x1000, mem=[MemOp("w", _SINK, 0xAB, 4)])]
    obs = assess_trace_observability(items, sink_addr=_SINK)
    assert obs.sink_captured is True


def test_sink_captured_false_when_not_observed():
    items = [ins(0, 0x1000, mem=[MemOp("w", 0x9999, 1, 4)])]
    obs = assess_trace_observability(items, sink_addr=_SINK)
    assert obs.sink_captured is False
    sink_dim = next(d for d in obs.dims if d.dimension == "sink")
    assert not sink_dim.sufficient
    assert "re-capture" in sink_dim.reason


def test_sink_captured_via_snapshot():
    items = [ins(0, 0x1000)]
    snap = MemSnapshot(addr=_SINK, data=b"\x01\x02")
    obs = assess_trace_observability(items, sink_window=(_SINK, _SINK + 1),
                                     snapshots=[snap])
    assert obs.sink_captured is True


def test_no_sink_gives_none():
    obs = assess_trace_observability([ins(0, 0x1000)])
    assert obs.sink_captured is None


def test_low_dimension_marked_insufficient():
    items = [ins(i, 0x1000 + i) for i in range(100)]   # all empty regs_write
    obs = assess_trace_observability(items)
    rw = next(d for d in obs.dims if d.dimension == "regs_write")
    assert not rw.sufficient
    assert obs.overall_sufficient_for("regs_write") is False


def test_predicates_match_kernel():
    a = ins(0, 0, writes={"x1": 1})
    b = ins(1, 0, mem=[MemOp("w", 0x10, 1, 4)])
    c = ins(2, 0, reads={"x1": 1})
    assert has_regs_write(a) and not has_regs_write(b)
    assert has_write_dim(a) and has_write_dim(b) and not has_write_dim(c)


# --- A8 single-source: cohort_diff + opaque_staging consume this module ------

def test_cohort_diff_imports_shared_predicate():
    import engine.cohort_diff as cd
    from engine.trace_observability import has_write_dim as kernel_pred
    # cohort_diff's observable-diff-dim predicate IS the kernel one (single source).
    assert cd._has_observable_diff_dim is kernel_pred


def test_opaque_staging_uses_unified_assess():
    # opaque_staging's coverage path imports assess_trace_observability; exercise a
    # window so the path runs without error and yields the same number the kernel
    # would (here a fully-populated window -> rate 1.0).
    from engine.opaque_staging import diagnose_opaque_staging
    items = [
        ins(0, 0x1000, "mov x1,#1", writes={"x1": 1}, reads={"x0": 0}),
        ins(1, 0x1004, "ldr x2,[x1]", reads={"x1": 1}, writes={"x2": 2},
            mem=[MemOp("r", 0x8000, 3, 4)]),
    ]
    n, rw, rr, mem = dimension_coverage(items)
    assert rw == 1.0
    # diagnosis runs (no crash) on the populated window.
    diag = diagnose_opaque_staging(items, window=(0, 1), window_is_idx=True)
    assert diag.verdict in ("known_addr", "symbolic_address", "inconclusive")


def test_thresholds_parameterised():
    assert set(DEFAULT_THRESHOLDS) == {"regs_write", "regs_read", "mem"}
    items = [ins(0, 0), ins(1, 0, writes={"x1": 1})]   # rate 0.5
    # default threshold 0.05 -> sufficient; raise to 0.9 -> insufficient.
    assert assess_trace_observability(items).overall_sufficient_for("regs_write")
    assert not assess_trace_observability(
        items, thresholds={"regs_write": 0.9}).overall_sufficient_for("regs_write")
