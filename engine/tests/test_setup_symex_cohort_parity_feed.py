"""cohort→parity feed leg — supplied goes from 0/fallback to a REAL N.

todo/dev-cohort-parity-feed-leg.md. A parity-blocked drive used to surface
``supplied=0 (fallback 1/1), need>=3`` for every window because the cohort's
per-window observed outputs were never assembled into ``ParityVector``s — that is
a FEED pit, not an F error. This leg runs the runner once per cohort vector (same
entry/window, items swapped), builds one REAL cross-run vector each (observed =
that vector's true exit sink from ``trace_self_check.sink_value``, predicted =
``emitted_F`` on its own ``seed_values``), and feeds them to the multi-vector
gate so ``supplied`` reflects the real vector count.

Synthetic shapes only — a stub runner that returns a different
``trace_self_check`` per cohort vector. No case coordinates.
"""

from __future__ import annotations

from engine.setup_symex import (
    CaseConfig,
    DriveResult,
    build_concrete_backing,
    drive,
)
from engine.types import Instruction


# --------------------------------------------------------------------------- #
# Fixtures: a 2-instr backed window + a stub runner keyed on the trace's marker.
# --------------------------------------------------------------------------- #

def _ins(idx, pc, mnem, *, reads=None, writes=None, mem=()):
    return Instruction(idx=idx, pc=pc, bytes_=b"\x00\x00\x00\x00", mnemonic=mnem,
                       regs_read=reads or {}, regs_write=writes or {}, mem=tuple(mem))


def _items(marker: int):
    # ``marker`` rides on x16 (the backed pointer base) so the stub runner can tell
    # the main trace from each cohort vector by reading items[0].regs_read.
    return [
        _ins(0, 0x1000, "ldr w0, [x16]", reads={"x16": marker}),
        _ins(1, 0x1004, "eor w0, w0, w1", reads={"w0": 0, "w1": 0}, writes={"w0": 0}),
    ]


# F under test: carrier XOR a fixed key. predicted = F(seed) per vector.
_KEY = 0x5A5A5A5A
_EXPR = f"def f(carrier):\n    return (carrier ^ {_KEY:#x}) & 0xffffffff\n"


def _cc(marker: int) -> CaseConfig:
    return CaseConfig(
        target="synthetic.so", input_hash="ab12", run_id="run-1",
        seed_hint_addr=0x100, sink_hint_addr=0x200, entry_pc=0x0FFF,
        window=(0x1000, 0x10FF), reg_file=("x0", "x1", "x16"),
        inputs=("carrier",), parity_min=1, symbolic_regs=("x0", "x1"),
        concrete_backing=build_concrete_backing(reg_values={"x16": marker}),
        task="cohort_parity_smoke")


_BOTH = {"alias_vs_compute": "compute", "which_static": []}

# Main trace marker + four cohort vectors, each with a DISTINCT seed → distinct
# (observed, predicted) pair. observed is the TRUE F(seed) so predicted matches.
_MAIN = 0x9000
_COHORT_SEED = {0xA000: 0x01, 0xB000: 0x02, 0xC000: 0x03, 0xD000: 0x04}


def _sink_for(seed: int) -> int:
    return (seed ^ _KEY) & 0xFFFFFFFF


def _stub_runner(*, cohort_observed=None, cohort_exec_id=None):
    """A runner keyed on items[0].regs_read['x16'] (the per-trace marker).

    The main trace emits the F + a single tautological-ish parity vector (1/1).
    Each cohort vector surfaces its own ``trace_self_check`` (distinct seed/sink),
    so the feed leg builds a REAL vector per cohort. ``cohort_observed`` can
    override a vector's observed sink to inject a mismatch; ``cohort_exec_id`` can
    force a runner-supplied ``exec_id`` per marker (to pin the mixing gate).
    """
    cohort_observed = cohort_observed or {}
    cohort_exec_id = cohort_exec_id or {}

    def runner(ctx):
        items = list(ctx.get("items", []))
        marker = items[0].regs_read.get("x16") if items else None
        if marker == _MAIN:
            return {
                "propagated": True, "gold_parity": "1/1",
                "expr_source": _EXPR,
                # main run's own vector (carries the deriving trace's exec_id via
                # gold fallback → no exec_id; 1/1 → tautological, supplied=1 today).
                "trace_self_check": {
                    "seed_values": {"carrier": _MAIN},
                    "sink_value": _sink_for(_MAIN), "sink_mask": 0xFFFFFFFF},
            }
        # A cohort vector: surface its own seed + TRUE exit sink (the runner's live
        # oracle output). observed overridable for the mismatch / mixing pins.
        seed = _COHORT_SEED.get(marker, 0)
        observed = cohort_observed.get(marker, _sink_for(seed))
        out = {
            "propagated": True, "gold_parity": "1/1", "expr_source": _EXPR,
            "trace_self_check": {
                "seed_values": {"carrier": seed},
                "sink_value": observed, "sink_mask": 0xFFFFFFFF},
        }
        if marker in cohort_exec_id:
            out["exec_id"] = cohort_exec_id[marker]
        return out

    return runner


def _cohort_traces():
    return [_items(m) for m in _COHORT_SEED]


def _cohort_keys():
    return [f"seed-{m:#x}" for m in _COHORT_SEED]


# --------------------------------------------------------------------------- #
# CORE: supplied 0/fallback → real 4 (the clark acceptance anchor).
# --------------------------------------------------------------------------- #

def test_supplied_goes_from_fallback_to_real_four():
    res = drive(trace=_items(_MAIN), case_config=_cc(_MAIN),
                triton_runner=_stub_runner(), decisions=_BOTH,
                cohort_traces=_cohort_traces(), cohort_keys=_cohort_keys())
    assert isinstance(res, DriveResult)
    pr = res.parity_report
    assert pr is not None
    # supplied = main's 1 fallback vector + 4 REAL cohort vectors.
    assert pr["total"] == 5
    # the 4 cohort vectors are all independent (distinct input_keys, none the
    # deriving trace) and match (observed == F(seed)) → matched >= 4 → EXACT.
    assert pr["independent_pass"] >= 4
    assert pr["verdict"] == "EXACT"
    assert res.closed is True


def test_no_cohort_is_byte_for_byte_today():
    # Invariant 7: no cohort_traces → only the main trace's fallback vector.
    res = drive(trace=_items(_MAIN), case_config=_cc(_MAIN),
                triton_runner=_stub_runner(), decisions=_BOTH)
    assert isinstance(res, DriveResult)
    pr = res.parity_report
    assert pr is not None
    assert pr["total"] == 1          # main fallback only — supplied unchanged
    # The lone tautological 1/1 fallback is below the independent floor (need 3) —
    # supplied < need, the FEED pit the cohort leg exists to close. BLOCK, no close.
    assert pr["independent_pass"] < pr["min_vectors"]
    assert pr["verdict"] == "BLOCK"
    assert res.closed is False


def test_empty_cohort_list_is_byte_for_byte_today():
    # Invariant 7: an empty cohort list behaves exactly like no cohort.
    res = drive(trace=_items(_MAIN), case_config=_cc(_MAIN),
                triton_runner=_stub_runner(), decisions=_BOTH,
                cohort_traces=[], cohort_keys=[])
    assert res.parity_report["total"] == 1
    assert res.parity_report["verdict"] == "BLOCK"


# --------------------------------------------------------------------------- #
# Invariant 8: real oracle observed + real eval predicted → real comparison.
# --------------------------------------------------------------------------- #

def test_f_wrong_surfaces_as_real_mismatch_not_feed_pit():
    # One cohort vector's TRUE exit sink disagrees with F(seed) → a real mismatch
    # (matched < supplied), not the supplied<need feed pit. supplied still 5.
    bad_marker = 0xC000
    res = drive(
        trace=_items(_MAIN), case_config=_cc(_MAIN),
        triton_runner=_stub_runner(cohort_observed={bad_marker: 0xDEADBEEF}),
        decisions=_BOTH, cohort_traces=_cohort_traces(), cohort_keys=_cohort_keys())
    pr = res.parity_report
    assert pr["total"] == 5                      # feed still arrived (not a pit)
    # main fallback (1) + 3 good cohort vectors matched; the bad one did NOT.
    assert pr["independent_pass"] == 4
    assert any("seed-0xc000" in m for m in pr["mismatches"])


def test_synthesized_exec_ids_are_distinct_per_input_no_false_mixing():
    # Default: each cohort vector gets a STABLE synthesized exec_id (cohort-k:key),
    # distinct per input, so even a numeric observed coincidence does NOT trip the
    # mixing gate — distinct inputs never share one execution.
    res = drive(
        trace=_items(_MAIN), case_config=_cc(_MAIN),
        triton_runner=_stub_runner(cohort_observed={0xA000: _sink_for(0x01),
                                                    0xB000: _sink_for(0x01)}),
        decisions=_BOTH, cohort_traces=_cohort_traces(), cohort_keys=_cohort_keys())
    pr = res.parity_report
    assert pr["determinism_seen"] is True        # cohort vectors carry exec_ids
    assert pr["determinism_ok"] is True          # distinct exec_ids → no mixing


def test_determinism_mixing_gate_catches_shared_exec_id():
    # Invariant 8: when the runner surfaces the SAME exec_id for two DISTINCT cohort
    # inputs (one execution's observed reused for another input), the mixing gate
    # catches it → determinism_ok False → BLOCK, never silently EXACT.
    res = drive(
        trace=_items(_MAIN), case_config=_cc(_MAIN),
        triton_runner=_stub_runner(
            cohort_exec_id={0xA000: "shared-exec", 0xB000: "shared-exec"}),
        decisions=_BOTH, cohort_traces=_cohort_traces(), cohort_keys=_cohort_keys())
    pr = res.parity_report
    assert pr["determinism_seen"] is True
    assert pr["determinism_ok"] is False         # the mixing gate fired
    assert pr["verdict"] == "BLOCK"              # gate not loosened — no close
    assert res.closed is False


def test_each_cohort_vector_carries_its_own_exec_id():
    res = drive(trace=_items(_MAIN), case_config=_cc(_MAIN),
                triton_runner=_stub_runner(), decisions=_BOTH,
                cohort_traces=_cohort_traces(), cohort_keys=_cohort_keys())
    pr = res.parity_report
    vecs = pr["vectors"]
    if isinstance(vecs, dict):              # trimmed-list form
        vecs = vecs["sample"]
    cohort_keys = {f"seed-{m:#x}" for m in _COHORT_SEED}
    seen = {v["input_key"] for v in vecs} & cohort_keys
    # the four cohort input_keys all surfaced as real vectors.
    assert seen == cohort_keys
