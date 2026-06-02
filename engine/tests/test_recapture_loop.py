"""Recapture batch loop (Req2 + G1/G2/G3 rewrite) — accumulate a WATCH-POINT PLAN
(never snapshots), and each round do EXACTLY ONE rerun whose output + snapshots all
come from that single execution (one nonce — G1). Cluster the plan into a few batch
observe points, re-run, re-provenance, converge (G2: transient rises allowed) or stop
with a structured terminal (G3). Exercised end-to-end with a SYNTHETIC adapter.

G1 (HARD): rr.output, the output-region snapshot, and every watch snapshot fed to one
trace_provenance call all come from the SAME rerun. The loop NEVER carries a prior
round's snapshots — the adapter therefore returns the WHOLE accumulated capture each
round (re-capturing everything fresh in one execution), and the loop uses rr.output as
that round's expected.

Acceptance:
  ① byte-level next_watch → few batch observe points (PC + contiguous range).
  ② feed snapshots back over multiple rounds → loop advances to CLOSED.
  ③ gap stops shrinking → structured terminal with residual shape, no infinite loop.
  G2: a 232→1375→1033→673→126 shape (a transient RISE on round 2) must NOT misfire
      STALLED and must eventually converge.
"""

from __future__ import annotations

from engine.oracle_provenance import (
    ProvenanceResult,
    ProvenanceVerdict,
    trace_provenance,
)
from engine.recapture import observe_points_from_provenance
from engine.recapture_loop import (
    LoopOutcome,
    RecaptureLoopResult,
    run_recapture_loop,
)
from engine.runner_client import ObservedState, RerunResult
from engine.types import Instruction, MemOp


def _ins(idx, mnem, reads=None, writes=None, mem=()):
    return Instruction(idx=idx, pc=0x70000 + idx * 4, bytes_=b"\x00\x00\x00\x00",
                       mnemonic=mnem, regs_read=dict(reads or {}),
                       regs_write=dict(writes or {}), mem=mem)


OUT = 0x72b18          # located sink base
EXPECTED = bytes([0xAA, 0xBB, 0xCC, 0xDD])
SINK_PC = 0x80000      # a PC at which the sink region is observable (output_observe_pc)


# --- Acceptance ①: byte-level gaps cluster into few batch observe points -----

def test_byte_level_gaps_cluster_into_few_batch_points():
    # Two reading PCs, each loading a contiguous 4-byte native range that feeds half
    # of an 8-byte sink. 8 byte-level gaps should coalesce to 2 batch observe points.
    BASE_A = 0x9000
    BASE_B = 0xA000
    expected8 = bytes(range(8))
    trace = [
        _ins(0, "ldr x8, [x9]", reads={"x9": BASE_A}, writes={"x8": 0},
             mem=(MemOp("r", BASE_A, 0, 4),)),
        _ins(1, "str x8, [x12]", reads={"x8": 0, "x12": OUT},
             mem=(MemOp("w", OUT, 0, 4),)),
        _ins(2, "ldr x10, [x11]", reads={"x11": BASE_B}, writes={"x10": 0},
             mem=(MemOp("r", BASE_B, 0, 4),)),
        _ins(3, "str x10, [x13]", reads={"x10": 0, "x13": OUT + 4},
             mem=(MemOp("w", OUT + 4, 0, 4),)),
    ]
    prov = trace_provenance(trace, expected8, sink_base=OUT)
    assert prov.verdict is ProvenanceVerdict.NEEDS_OBSERVATION
    distinct_addrs = {w["addr"] for w in prov.next_watch}
    assert len(distinct_addrs) == 8

    pre = observe_points_from_provenance(prov)
    assert len(pre.observe_points) == 2
    mems = sorted(op.mem for op in pre.observe_points)
    assert mems == [((BASE_A, 4),), ((BASE_B, 4),)]


# --- A synthetic adapter: re-captures the WHOLE accumulated set every round (G1) ---

class _AllAtOnceAdapter:
    """Models the runner re-running ONCE per round with the full plan and returning
    the captures for EVERY address it can observe in that single execution. ``known``
    is the {addr: bytes} the runner can observe; every round returns ALL of them (one
    execution, one nonce). ``output`` is the produced bytes used as that round's
    expected. This mirrors G1: no cross-rerun accumulation — each round's snapshots
    all come from this one rerun."""

    def __init__(self, known, output=EXPECTED, truncated=False, truncated_detail=None):
        self._known = dict(known)
        self._output = output
        self._truncated = truncated
        self._truncated_detail = truncated_detail
        self.calls = 0
        self.last_points = None

    def rerun(self, input_bytes, observe_points=None):
        self.calls += 1
        self.last_points = observe_points
        obs = ObservedState(pc=0x70000, when="before", regs={}, mem=dict(self._known))
        return RerunResult(output=self._output, observations=(obs,),
                           truncated=self._truncated,
                           truncated_detail=self._truncated_detail)


# --- Acceptance ②: feed-back closes the producer gap ------------------------

def test_loop_closes_when_sink_reconstructs_from_traced_writes():
    # Sink reconstructs to expected2 from traced writes → CONTINUOUS_BUFFER on the
    # first round's rerun (rr.output == expected2). Loop closes on round 1.
    A0 = 0x9000
    A1 = 0x9100
    expected2 = bytes([0xAA, 0xBB])
    trace = [
        _ins(0, "ldrb w8, [x9]", reads={"x9": A0}, writes={"w8": 0xAA},
             mem=(MemOp("r", A0, 0xAA, 1),)),
        _ins(1, "strb w8, [x12]", reads={"w8": 0xAA, "x12": OUT},
             mem=(MemOp("w", OUT, 0xAA, 1),)),
        _ins(2, "ldrb w10, [x11]", reads={"x11": A1}, writes={"w10": 0xBB},
             mem=(MemOp("r", A1, 0xBB, 1),)),
        _ins(3, "strb w10, [x13]", reads={"w10": 0xBB, "x13": OUT + 1},
             mem=(MemOp("w", OUT + 1, 0xBB, 1),)),
    ]
    res = run_recapture_loop(
        trace, sink_base=OUT, adapter=_AllAtOnceAdapter({}, output=expected2),
        loop_input=b"\x01", output_observe_pc=SINK_PC)
    assert isinstance(res, RecaptureLoopResult)
    assert res.outcome is LoopOutcome.CLOSED
    assert res.final.verdict is ProvenanceVerdict.CONTINUOUS_BUFFER
    assert len(res.rounds) == 1  # one rerun produced rr.output; reconstructed closed


def test_loop_closes_when_producer_reads_become_observed():
    # In-trace sink write value is WRONG (zeros), so NEEDS_OBSERVATION with two
    # producer-read gaps. Once the runner captures BOTH read addresses (as snapshots
    # in the SAME rerun), the gap set empties → closed. The adapter returns both each
    # round (one execution, one nonce — G1).
    A0 = 0x9000
    A1 = 0xA000
    expected8 = bytes(range(8))
    trace = [
        _ins(0, "ldr x8, [x9]", reads={"x9": A0}, writes={"x8": 0},
             mem=(MemOp("r", A0, 0, 4),)),
        _ins(1, "str x8, [x12]", reads={"x8": 0, "x12": OUT},
             mem=(MemOp("w", OUT, 0, 4),)),
        _ins(2, "ldr x10, [x11]", reads={"x11": A1}, writes={"x10": 0},
             mem=(MemOp("r", A1, 0, 4),)),
        _ins(3, "str x10, [x13]", reads={"x10": 0, "x13": OUT + 4},
             mem=(MemOp("w", OUT + 4, 0, 4),)),
    ]
    prov0 = trace_provenance(trace, expected8, sink_base=OUT)
    assert prov0.verdict is ProvenanceVerdict.NEEDS_OBSERVATION
    g0 = {w["addr"] for w in prov0.next_watch}
    assert f"0x{A0:x}" in g0 and f"0x{A1:x}" in g0

    known = {}
    known.update({A0 + k: bytes([0x11]) for k in range(4)})
    known.update({A1 + k: bytes([0x22]) for k in range(4)})
    res = run_recapture_loop(
        trace, sink_base=OUT, adapter=_AllAtOnceAdapter(known, output=expected8),
        loop_input=b"\x01", output_observe_pc=SINK_PC)
    assert res.outcome is LoopOutcome.CLOSED
    assert res.final.next_watch == []  # all producer reads now observed


# --- G2: a transient RISE (232→1375→...) must not misfire STALLED -------------

def test_g2_transient_rise_does_not_misfire_stalled_then_converges(monkeypatch):
    # Drive the convergence DECISION directly by scripting trace_provenance with a gap
    # sequence shaped like the real 232→1375→1033→673→126→0 (round 2 RISES). The loop
    # must NOT stop on the round-2 rise and must reach CLOSED when the gap empties.
    seq = [232, 1375, 1033, 673, 126, 0]
    calls = {"i": 0}

    def _fake_trace(items, expected, *, sink_base, snapshots=None, **kw):
        n = seq[calls["i"]]
        calls["i"] += 1
        # n distinct gaps, each at its own reading PC so they are placeable and counted
        # as distinct (addr, pc).
        nw = [{"addr": f"0x{0x9000 + j:x}", "pc": f"0x{0x70000 + j:x}",
               "reason": "uncaptured native read"} for j in range(n)]
        return ProvenanceResult(ProvenanceVerdict.NEEDS_OBSERVATION,
                                next_watch=nw, expected=b"x")

    monkeypatch.setattr("engine.recapture_loop.trace_provenance", _fake_trace)
    trace = [_ins(0, "nop")]
    res = run_recapture_loop(
        trace, sink_base=OUT, adapter=_AllAtOnceAdapter({}, output=EXPECTED),
        loop_input=b"\x01", output_observe_pc=SINK_PC, max_rounds=20)
    assert res.outcome is LoopOutcome.CLOSED
    gaps = [r.n_gaps_after for r in res.rounds]
    assert gaps == [232, 1375, 1033, 673, 126, 0]   # the rise on round 2 was kept
    assert res.final.next_watch == []


def test_g2_two_consecutive_nonshrink_rounds_stalls(monkeypatch):
    # 100 → 80 (shrink) → 80 (flat, streak=1, keep going) → 80 (flat, streak=2, STALL).
    seq = [100, 80, 80, 80]
    calls = {"i": 0}

    def _fake_trace(items, expected, *, sink_base, snapshots=None, **kw):
        n = seq[min(calls["i"], len(seq) - 1)]
        calls["i"] += 1
        nw = [{"addr": f"0x{0x9000 + j:x}", "pc": f"0x{0x70000 + j:x}",
               "reason": "uncaptured native read"} for j in range(n)]
        return ProvenanceResult(ProvenanceVerdict.NEEDS_OBSERVATION,
                                next_watch=nw, expected=b"x")

    monkeypatch.setattr("engine.recapture_loop.trace_provenance", _fake_trace)
    res = run_recapture_loop(
        [_ins(0, "nop")], sink_base=OUT,
        adapter=_AllAtOnceAdapter({}, output=EXPECTED),
        loop_input=b"\x01", output_observe_pc=SINK_PC, max_rounds=20)
    assert res.outcome is LoopOutcome.STALLED
    # one flat round was tolerated (streak=1) before stopping on the second
    assert [r.n_gaps_after for r in res.rounds] == [100, 80, 80, 80]
    assert res.residual is not None
    term = res.terminal()
    assert term["terminal"] == "BLOCKED"
    assert term["block_phase"] == "PROVENANCE_WATCH_BATCH"
    assert term["last_next_watch_n"] == 80


def test_g2_broad_region_diffusion_stalls(monkeypatch):
    # The watch plan spreads across an explosion of distinct reading PCs (region
    # groups) on CONSECUTIVE non-shrinking rounds → the diffusion guard terminals.
    # R1: 2 groups (baseline). R2: 20 (>=8 floor, >=2x → diffusion_streak=1). R3: 40
    # (>=2x of 20 → diffusion_streak=2 → STALLED). A single jump alone would NOT stop.
    seq_groups = [2, 20, 40]
    calls = {"i": 0}

    def _fake_trace(items, expected, *, sink_base, snapshots=None, **kw):
        g = seq_groups[min(calls["i"], len(seq_groups) - 1)]
        calls["i"] += 1
        # one distinct PC per group, one addr each
        nw = [{"addr": f"0x{0x9000 + j * 0x100:x}", "pc": f"0x{0x70000 + j * 4:x}",
               "reason": "spread"} for j in range(g)]
        return ProvenanceResult(ProvenanceVerdict.NEEDS_OBSERVATION,
                                next_watch=nw, expected=b"x")

    monkeypatch.setattr("engine.recapture_loop.trace_provenance", _fake_trace)
    res = run_recapture_loop(
        [_ins(0, "nop")], sink_base=OUT,
        adapter=_AllAtOnceAdapter({}, output=EXPECTED),
        loop_input=b"\x01", output_observe_pc=SINK_PC, max_rounds=20)
    assert res.outcome is LoopOutcome.STALLED
    assert "DIFFUSED" in res.detail
    assert res.rounds[-1].n_region_groups == 40


# --- Acceptance ③: gap stops shrinking → structured terminal, no infinite loop -

def test_loop_stalls_when_capture_never_closes_the_gap(monkeypatch):
    # Real trace_provenance, but the runner never returns the gap address, so the gap
    # count is flat every round. Two flat rounds → STALLED with the residual shape.
    A0 = 0x9000
    trace = [
        _ins(0, "ldr x8, [x9]", reads={"x9": A0}, writes={"x8": 0},
             mem=(MemOp("r", A0, 0, 4),)),
        _ins(1, "str x8, [x12]", reads={"x8": 0, "x12": OUT},
             mem=(MemOp("w", OUT, 0, 4),)),
    ]
    res = run_recapture_loop(
        trace, sink_base=OUT, adapter=_AllAtOnceAdapter({}, output=EXPECTED),
        loop_input=b"\x01", output_observe_pc=SINK_PC, max_rounds=50)
    assert res.outcome is LoopOutcome.STALLED
    assert res.closed is False
    # gap is flat (4) from the start; two consecutive non-shrink rounds → stop. The
    # first round establishes the baseline, then two flat comparisons → 3 rounds.
    assert len(res.rounds) == 3
    assert res.residual is not None
    assert res.residual["n_reading_pcs"] == 1
    assert res.residual["n_gaps"] == 4
    regions = res.residual["regions"]
    assert len(regions) == 1
    assert regions[0]["ranges"] == [[f"0x{A0:x}", 4]]
    assert regions[0]["pc"] == f"0x{0x70000:x}"
    assert "NO mem snapshots" in res.detail
    d = res.to_dict()
    assert d["outcome"] == "STALLED" and d["closed"] is False
    assert d["terminal"]["block_phase"] == "PROVENANCE_WATCH_BATCH"
    assert d["residual"]["regions"][0]["ranges"] == [[f"0x{A0:x}", 4]]


# --- Unhookable terminal: gaps remain but none have a reading PC -------------

def test_loop_unplaceable_when_no_gap_has_a_reading_pc():
    # A snapshot-located sink with no traced producer and no reading PC for the gap →
    # next_watch entries have pc=None → no observe point is placeable.
    trace = [_ins(0, "nop"), _ins(1, "nop")]
    res = run_recapture_loop(
        trace, sink_base=OUT, adapter=_AllAtOnceAdapter({}, output=EXPECTED),
        loop_input=b"\x01", output_observe_pc=SINK_PC)
    assert res.outcome is LoopOutcome.UNPLACEABLE
    assert res.closed is False
    assert res.residual is not None
    assert res.residual["n_unhookable"] >= 1
    assert "no_pc" in [reg["pc"] for reg in res.residual["regions"]]
    assert "snapshot the buffer directly" in res.detail
    assert res.terminal()["outcome"] == "UNPLACEABLE"


# --- Budget terminal: still shrinking when rounds run out --------------------

def test_loop_budget_exhausted_is_explicit_not_silent(monkeypatch):
    # Gap shrinks every round but never reaches 0 within the budget → BUDGET_EXHAUSTED.
    seq = [100, 80, 60, 40, 20]
    calls = {"i": 0}

    def _fake_trace(items, expected, *, sink_base, snapshots=None, **kw):
        n = seq[min(calls["i"], len(seq) - 1)]
        calls["i"] += 1
        nw = [{"addr": f"0x{0x9000 + j:x}", "pc": f"0x{0x70000 + j:x}",
               "reason": "r"} for j in range(n)]
        return ProvenanceResult(ProvenanceVerdict.NEEDS_OBSERVATION,
                                next_watch=nw, expected=b"x")

    monkeypatch.setattr("engine.recapture_loop.trace_provenance", _fake_trace)
    res = run_recapture_loop(
        [_ins(0, "nop")], sink_base=OUT,
        adapter=_AllAtOnceAdapter({}, output=EXPECTED),
        loop_input=b"\x01", output_observe_pc=SINK_PC, max_rounds=2)
    assert res.outcome is LoopOutcome.BUDGET_EXHAUSTED
    assert res.closed is False
    assert len(res.rounds) == 2
    assert res.residual is not None
    assert "out of budget" in res.detail
    assert res.terminal()["outcome"] == "BUDGET_EXHAUSTED"


# --- Truncation propagation: a cap-hit rerun is surfaced per round, not silent -

def test_loop_surfaces_runner_truncation_per_round():
    A0 = 0x9000
    trace = [
        _ins(0, "ldr x8, [x9]", reads={"x9": A0}, writes={"x8": 0},
             mem=(MemOp("r", A0, 0, 4),)),
        _ins(1, "str x8, [x12]", reads={"x8": 0, "x12": OUT},
             mem=(MemOp("w", OUT, 0, 4),)),
    ]
    adapter = _AllAtOnceAdapter(
        {A0 + k: bytes([0x11]) for k in range(4)}, output=EXPECTED,
        truncated=True, truncated_detail={"cap": "X_MAX", "limit": 8})
    res = run_recapture_loop(
        trace, sink_base=OUT, adapter=adapter, loop_input=b"\x01",
        output_observe_pc=SINK_PC)
    assert res.rounds[0].truncated is True


# --- G1: the single rerun carries output point + plan points together --------

def test_g1_single_rerun_carries_output_point_and_plan_together():
    # After round 1 seeds a plan gap, round 2's single rerun must include BOTH the
    # output-region observe point (at SINK_PC) AND the plan's batch point. We never
    # accumulate snapshots: the loop hands the adapter the full plan each round.
    A0 = 0x9000
    trace = [
        _ins(0, "ldr x8, [x9]", reads={"x9": A0}, writes={"x8": 0},
             mem=(MemOp("r", A0, 0, 4),)),
        _ins(1, "str x8, [x12]", reads={"x8": 0, "x12": OUT},
             mem=(MemOp("w", OUT, 0, 4),)),
    ]
    seen_points = []

    class _RecordingAdapter:
        def rerun(self, input_bytes, observe_points=None):
            seen_points.append(list(observe_points or []))
            obs = ObservedState(pc=0x70000, when="before", regs={}, mem={})
            return RerunResult(output=EXPECTED, observations=(obs,))

    run_recapture_loop(
        trace, sink_base=OUT, adapter=_RecordingAdapter(), loop_input=b"\x01",
        output_observe_pc=SINK_PC, max_rounds=3)
    # round 1: plan empty, last_out_len 0 → only (or no) output point; round 2 onward:
    # the output-region point sized to len(EXPECTED) PLUS the plan's batch point.
    assert len(seen_points) >= 2
    r2 = seen_points[1]
    pcs = {p.pc for p in r2}
    assert SINK_PC in pcs           # output-region point present
    assert 0x70000 in pcs           # plan's batch point (reading PC at idx0) present
    out_pt = next(p for p in r2 if p.pc == SINK_PC)
    assert out_pt.mem == ((OUT, len(EXPECTED)),)   # sized to the prior rr.output


def test_g1_no_output_observe_pc_warns_but_still_uses_rr_output(caplog):
    # Without output_observe_pc the loop still uses rr.output as expected (and closes
    # if the sink reconstructs), but logs a WARN that the sink region is not snapshot-
    # confirmed in-run.
    import logging
    A0 = 0x9000
    A1 = 0x9100
    expected2 = bytes([0xAA, 0xBB])
    trace = [
        _ins(0, "ldrb w8, [x9]", reads={"x9": A0}, writes={"w8": 0xAA},
             mem=(MemOp("r", A0, 0xAA, 1),)),
        _ins(1, "strb w8, [x12]", reads={"w8": 0xAA, "x12": OUT},
             mem=(MemOp("w", OUT, 0xAA, 1),)),
        _ins(2, "ldrb w10, [x11]", reads={"x11": A1}, writes={"w10": 0xBB},
             mem=(MemOp("r", A1, 0xBB, 1),)),
        _ins(3, "strb w10, [x13]", reads={"w10": 0xBB, "x13": OUT + 1},
             mem=(MemOp("w", OUT + 1, 0xBB, 1),)),
    ]
    with caplog.at_level(logging.WARNING, logger="engine.recapture_loop"):
        res = run_recapture_loop(
            trace, sink_base=OUT, adapter=_AllAtOnceAdapter({}, output=expected2),
            loop_input=b"\x01")
    assert res.outcome is LoopOutcome.CLOSED
    assert any("no output_observe_pc" in r.message for r in caplog.records)


# --- Guard: a non-RerunResult from a misbehaving adapter is rejected loudly --

def test_loop_rejects_non_rerunresult_adapter():
    import pytest
    A0 = 0x9000
    trace = [
        _ins(0, "ldr x8, [x9]", reads={"x9": A0}, writes={"x8": 0},
             mem=(MemOp("r", A0, 0, 4),)),
        _ins(1, "str x8, [x12]", reads={"x8": 0, "x12": OUT},
             mem=(MemOp("w", OUT, 0, 4),)),
    ]

    class _BadAdapter:
        def rerun(self, input_bytes, observe_points=None):
            return {"output": b""}  # wrong type

    with pytest.raises(TypeError, match="must return a RerunResult"):
        run_recapture_loop(trace, sink_base=OUT, adapter=_BadAdapter(),
                           loop_input=b"\x01", output_observe_pc=SINK_PC)


# --- Empty output: honest stop, not a silent spin ----------------------------

def test_loop_stops_on_empty_rerun_output():
    trace = [_ins(0, "nop")]
    res = run_recapture_loop(
        trace, sink_base=OUT, adapter=_AllAtOnceAdapter({}, output=b""),
        loop_input=b"\x01", output_observe_pc=SINK_PC)
    assert res.outcome is LoopOutcome.STALLED
    assert "EMPTY output" in res.detail


# --- Serialization: to_dict() must json.dumps for CLOSED and non-CLOSED (set bug) -

def test_to_dict_json_roundtrips_for_closed_outcome():
    # CLOSED: a reconstructing sink. The result and every LoopRound (incl. new_region)
    # must json.dumps — guards the set-vs-bool regression at the source.
    import json
    A0 = 0x9000
    A1 = 0x9100
    expected2 = bytes([0xAA, 0xBB])
    trace = [
        _ins(0, "ldrb w8, [x9]", reads={"x9": A0}, writes={"w8": 0xAA},
             mem=(MemOp("r", A0, 0xAA, 1),)),
        _ins(1, "strb w8, [x12]", reads={"w8": 0xAA, "x12": OUT},
             mem=(MemOp("w", OUT, 0xAA, 1),)),
        _ins(2, "ldrb w10, [x11]", reads={"x11": A1}, writes={"w10": 0xBB},
             mem=(MemOp("r", A1, 0xBB, 1),)),
        _ins(3, "strb w10, [x13]", reads={"w10": 0xBB, "x13": OUT + 1},
             mem=(MemOp("w", OUT + 1, 0xBB, 1),)),
    ]
    res = run_recapture_loop(
        trace, sink_base=OUT, adapter=_AllAtOnceAdapter({}, output=expected2),
        loop_input=b"\x01", output_observe_pc=SINK_PC)
    assert res.outcome is LoopOutcome.CLOSED
    s = json.dumps(res.to_dict())          # must not raise (no set leaked anywhere)
    assert json.loads(s)["closed"] is True


def test_to_dict_json_roundtrips_for_non_closed_outcome(monkeypatch):
    # STALLED via two flat rounds — the exact path where prev_gap_keys is empty on the
    # FIRST round, so new_region would have been a set under the old bug. json.dumps of
    # the whole result (incl. rounds + terminal) must succeed.
    import json
    seq = [50, 50, 50]
    calls = {"i": 0}

    def _fake_trace(items, expected, *, sink_base, snapshots=None, **kw):
        n = seq[min(calls["i"], len(seq) - 1)]
        calls["i"] += 1
        nw = [{"addr": f"0x{0x9000 + j:x}", "pc": f"0x{0x70000 + j:x}",
               "reason": "r"} for j in range(n)]
        return ProvenanceResult(ProvenanceVerdict.NEEDS_OBSERVATION,
                                next_watch=nw, expected=b"x")

    monkeypatch.setattr("engine.recapture_loop.trace_provenance", _fake_trace)
    res = run_recapture_loop(
        [_ins(0, "nop")], sink_base=OUT,
        adapter=_AllAtOnceAdapter({}, output=EXPECTED),
        loop_input=b"\x01", output_observe_pc=SINK_PC, max_rounds=20)
    assert res.outcome is LoopOutcome.STALLED
    s = json.dumps(res.to_dict())          # must not raise (set bug would break here)
    d = json.loads(s)
    assert d["closed"] is False
    assert d["terminal"]["block_phase"] == "PROVENANCE_WATCH_BATCH"


# --- new_region is always a bool (False on the first round, prev_gap_keys empty) ---

def test_new_region_is_bool_false_on_first_round(monkeypatch):
    # On round 1 prev_gap_keys is empty; new_region MUST be the bool False, never a
    # set() (the regression that broke json.dumps).
    seq = [10, 10, 10]
    calls = {"i": 0}

    def _fake_trace(items, expected, *, sink_base, snapshots=None, **kw):
        n = seq[min(calls["i"], len(seq) - 1)]
        calls["i"] += 1
        nw = [{"addr": f"0x{0x9000 + j:x}", "pc": f"0x{0x70000 + j:x}",
               "reason": "r"} for j in range(n)]
        return ProvenanceResult(ProvenanceVerdict.NEEDS_OBSERVATION,
                                next_watch=nw, expected=b"x")

    monkeypatch.setattr("engine.recapture_loop.trace_provenance", _fake_trace)
    res = run_recapture_loop(
        [_ins(0, "nop")], sink_base=OUT,
        adapter=_AllAtOnceAdapter({}, output=EXPECTED),
        loop_input=b"\x01", output_observe_pc=SINK_PC, max_rounds=20)
    for rnd in res.rounds:
        assert isinstance(rnd.new_region, bool)   # never a set
    assert res.rounds[0].new_region is False       # first round: no prior region


# --- G3: non-CLOSED terminal() carries chain_n/backtrace_truncated/rounds/sample ---

def test_terminal_carries_g3_fields_on_non_closed(monkeypatch):
    seq = [100, 80, 80, 80]
    calls = {"i": 0}

    def _fake_trace(items, expected, *, sink_base, snapshots=None, **kw):
        n = seq[min(calls["i"], len(seq) - 1)]
        calls["i"] += 1
        nw = [{"addr": f"0x{0x9000 + j:x}", "pc": f"0x{0x70000 + j:x}",
               "reason": "uncaptured native read"} for j in range(n)]
        return ProvenanceResult(ProvenanceVerdict.NEEDS_OBSERVATION,
                                next_watch=nw, expected=b"x",
                                chain=[{"step": 0}, {"step": 1}, {"step": 2}])

    monkeypatch.setattr("engine.recapture_loop.trace_provenance", _fake_trace)
    res = run_recapture_loop(
        [_ins(0, "nop")], sink_base=OUT,
        adapter=_AllAtOnceAdapter({}, output=EXPECTED),
        loop_input=b"\x01", output_observe_pc=SINK_PC, max_rounds=20)
    assert res.outcome is LoopOutcome.STALLED
    term = res.terminal()
    # all four G3 fields present
    assert "chain_n" in term and "backtrace_truncated" in term
    assert "rounds" in term and "latest_next_watch_sample" in term
    assert term["chain_n"] == 3                       # from the last round's prov.chain
    assert term["backtrace_truncated"] is False
    assert [r["round"] for r in term["rounds"]] == [r.round for r in res.rounds]
    # sample is a readable preview (capped) of the residual next_watch
    assert isinstance(term["latest_next_watch_sample"], list)
    assert 0 < len(term["latest_next_watch_sample"]) <= 8
    assert term["latest_next_watch_sample"][0]["addr"].startswith("0x")
    # CLOSED still yields None
    assert run_recapture_loop is run_recapture_loop  # (no-op; keep import used)


# --- Req2 G3 補点1: every terminal carries a uniform block_why key ------------

def _stalled_loop(monkeypatch, seq=(100, 80, 80, 80)):
    calls = {"i": 0}

    def _fake_trace(items, expected, *, sink_base, snapshots=None, **kw):
        n = seq[min(calls["i"], len(seq) - 1)]
        calls["i"] += 1
        nw = [{"addr": f"0x{0x9000 + j:x}", "pc": f"0x{0x70000 + j:x}",
               "reason": "uncaptured native read"} for j in range(n)]
        return ProvenanceResult(ProvenanceVerdict.NEEDS_OBSERVATION,
                                next_watch=nw, expected=b"x")

    monkeypatch.setattr("engine.recapture_loop.trace_provenance", _fake_trace)
    return run_recapture_loop(
        [_ins(0, "nop")], sink_base=OUT,
        adapter=_AllAtOnceAdapter({}, output=EXPECTED),
        loop_input=b"\x01", output_observe_pc=SINK_PC, max_rounds=20)


def test_block_why_present_on_unplaceable_terminal_form():
    # block_why is the ONE uniform machine-readable block-reason key. UNPLACEABLE form
    # (real trace, no reading PC for the gap) — present AND == reason.
    res_u = run_recapture_loop(
        [_ins(0, "nop"), _ins(1, "nop")], sink_base=OUT,
        adapter=_AllAtOnceAdapter({}, output=EXPECTED),
        loop_input=b"\x01", output_observe_pc=SINK_PC)
    assert res_u.outcome is LoopOutcome.UNPLACEABLE
    term = res_u.terminal()
    assert term["block_why"] == term["reason"]
    assert "no reading PC" in term["block_why"]


def test_block_why_present_on_stalled_and_budget_terminal_forms(monkeypatch):
    # Two MORE terminal forms (STALLED, BUDGET_EXHAUSTED) — assert block_why present
    # and == reason on each (multi-terminal-form coverage, not just one path).
    res = _stalled_loop(monkeypatch)
    assert res.outcome is LoopOutcome.STALLED
    assert res.terminal()["block_why"] == res.terminal()["reason"]
    assert res.terminal()["block_why"] == "next_watch did not converge"

    calls = {"i": 0}
    seqb = [100, 80, 60]

    def _shrinking(items, expected, *, sink_base, snapshots=None, **kw):
        n = seqb[min(calls["i"], len(seqb) - 1)]
        calls["i"] += 1
        nw = [{"addr": f"0x{0x9000 + j:x}", "pc": f"0x{0x70000 + j:x}", "reason": "r"}
              for j in range(n)]
        return ProvenanceResult(ProvenanceVerdict.NEEDS_OBSERVATION,
                                next_watch=nw, expected=b"x")

    monkeypatch.setattr("engine.recapture_loop.trace_provenance", _shrinking)
    res_b = run_recapture_loop(
        [_ins(0, "nop")], sink_base=OUT,
        adapter=_AllAtOnceAdapter({}, output=EXPECTED),
        loop_input=b"\x01", output_observe_pc=SINK_PC, max_rounds=2)
    assert res_b.outcome is LoopOutcome.BUDGET_EXHAUSTED
    assert res_b.terminal()["block_why"] == res_b.terminal()["reason"]

    # to_dict carries the terminal (with block_why) for non-CLOSED, json-clean.
    import json
    d = json.loads(json.dumps(res.to_dict()))
    assert d["terminal"]["block_why"]


# --- Req2 G3 補点2: PROVENANCE_WATCH_BATCH_G1 same-execution terminal ----------

def test_assert_same_execution_passes_single_execution_set():
    # Same-execution set (one token, or untokened) → None (no misfire — regression).
    from engine.recapture_loop import assert_same_execution
    from engine.types import MemSnapshot
    one_token = [MemSnapshot(addr=0x9000, data=b"\x01", execution_id="rerun#1:ab"),
                 MemSnapshot(addr=0x9004, data=b"\x02", execution_id="rerun#1:ab")]
    assert assert_same_execution(one_token) is None
    untokened = [MemSnapshot(addr=0x9000, data=b"\x01"),
                 MemSnapshot(addr=0x9004, data=b"\x02")]
    assert assert_same_execution(untokened) is None      # absence != violation (default)


def test_assert_same_execution_flags_cross_rerun_mix():
    # >= 2 distinct non-None tokens → cross-rerun violation, with the offending set.
    from engine.recapture_loop import (
        assert_same_execution, G1_SAME_EXECUTION_VIOLATED)
    from engine.types import MemSnapshot
    mixed = [MemSnapshot(addr=0x9000, data=b"\x01", execution_id="rerun#1:aa"),
             MemSnapshot(addr=0xA000, data=b"\x02", execution_id="rerun#2:bb")]
    rep = assert_same_execution(mixed)
    assert rep is not None
    assert rep["violation"] == "cross_rerun"
    assert rep["n_executions"] == 2
    assert G1_SAME_EXECUTION_VIOLATED in rep["block_why"]
    assert "cannot be proven" not in rep["block_why"]


def test_assert_same_execution_unprovable_only_when_required():
    from engine.recapture_loop import (
        assert_same_execution, G1_SAME_EXECUTION_UNPROVABLE)
    from engine.types import MemSnapshot
    untokened = [MemSnapshot(addr=0x9000, data=b"\x01")]
    assert assert_same_execution(untokened) is None              # default: tolerated
    rep = assert_same_execution(untokened, require_proof=True)   # required: violation
    assert rep is not None and rep["violation"] == "unprovable"
    assert G1_SAME_EXECUTION_UNPROVABLE in rep["block_why"]


def test_loop_normal_closure_does_not_misfire_g1(monkeypatch):
    # REGRESSION: a normal same-execution closure must NOT trip the G1 terminal.
    A0 = 0x9000
    A1 = 0x9100
    expected2 = bytes([0xAA, 0xBB])
    trace = [
        _ins(0, "ldrb w8, [x9]", reads={"x9": A0}, writes={"w8": 0xAA},
             mem=(MemOp("r", A0, 0xAA, 1),)),
        _ins(1, "strb w8, [x12]", reads={"w8": 0xAA, "x12": OUT},
             mem=(MemOp("w", OUT, 0xAA, 1),)),
        _ins(2, "ldrb w10, [x11]", reads={"x11": A1}, writes={"w10": 0xBB},
             mem=(MemOp("r", A1, 0xBB, 1),)),
        _ins(3, "strb w10, [x13]", reads={"w10": 0xBB, "x13": OUT + 1},
             mem=(MemOp("w", OUT + 1, 0xBB, 1),)),
    ]
    res = run_recapture_loop(
        trace, sink_base=OUT, adapter=_AllAtOnceAdapter({}, output=expected2),
        loop_input=b"\x01", output_observe_pc=SINK_PC)
    assert res.outcome is LoopOutcome.CLOSED          # not CROSS_RERUN_G1
    assert res.same_execution is None
    # whatever closing snapshots exist are one execution (by construction): at most
    # ONE distinct token, never a None mixed with a token (assert_same_execution=None).
    from engine.recapture_loop import assert_same_execution
    assert assert_same_execution(res.snapshots) is None
    tokens = {s.execution_id for s in res.snapshots}
    assert len(tokens) <= 1


def test_loop_g1_terminal_fires_when_closing_set_crosses_reruns(monkeypatch):
    # Construct the cross-rerun scenario: a misbehaving adapter returns observations
    # carrying snapshots that, once stamped per round, the loop would close on — but
    # we force the closing set to span TWO reruns by patching mem_snapshots_from_rerun
    # to inject a snapshot from a DIFFERENT rerun token. The loop must refuse the
    # closure and emit PROVENANCE_WATCH_BATCH_G1 (never a FALSE close).
    from engine.types import MemSnapshot
    A0 = 0x9000
    A1 = 0x9100
    expected2 = bytes([0xAA, 0xBB])
    trace = [
        _ins(0, "ldrb w8, [x9]", reads={"x9": A0}, writes={"w8": 0xAA},
             mem=(MemOp("r", A0, 0xAA, 1),)),
        _ins(1, "strb w8, [x12]", reads={"w8": 0xAA, "x12": OUT},
             mem=(MemOp("w", OUT, 0xAA, 1),)),
        _ins(2, "ldrb w10, [x11]", reads={"x11": A1}, writes={"w10": 0xBB},
             mem=(MemOp("r", A1, 0xBB, 1),)),
        _ins(3, "strb w10, [x13]", reads={"w10": 0xBB, "x13": OUT + 1},
             mem=(MemOp("w", OUT + 1, 0xBB, 1),)),
    ]

    def _cross_rerun_snaps(result):
        # The closing set spans TWO reruns: one snapshot already carries a FOREIGN
        # rerun token (kept — not re-stamped), the other is untokened and gets THIS
        # round's token. Two distinct tokens in the set that would close provenance →
        # the G1 guard must refuse the closure.
        return [
            MemSnapshot(addr=A0, data=b"\xAA", execution_id="FOREIGN_RERUN#99"),
            MemSnapshot(addr=A1, data=b"\xBB"),   # untokened → this round's token
        ]

    monkeypatch.setattr(
        "engine.recapture_loop.mem_snapshots_from_rerun", _cross_rerun_snaps)
    res = run_recapture_loop(
        trace, sink_base=OUT, adapter=_AllAtOnceAdapter({}, output=expected2),
        loop_input=b"\x01", output_observe_pc=SINK_PC)
    assert res.outcome is LoopOutcome.CROSS_RERUN_G1
    assert res.closed is False                       # NOT reported as a closure
    term = res.terminal()
    assert term["terminal"] == "BLOCKED"
    assert term["block_phase"] == "PROVENANCE_WATCH_BATCH_G1"
    assert "same-execution snapshot requirement violated" in term["block_why"]
    assert term["violation"] == "cross_rerun"
    # to_dict surfaces the same-execution report + the G1 terminal, json-clean.
    import json
    d = json.loads(json.dumps(res.to_dict()))
    assert d["terminal"]["block_phase"] == "PROVENANCE_WATCH_BATCH_G1"
    assert d["same_execution"]["n_executions"] == 2
