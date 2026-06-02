"""spec #4 — provenance-driven observation planner.

The instruction/edge-SHAPE heuristic layer that turns a stalled provenance result
into a concrete next-batch of observe points (each with a reason + which rule fired),
plus the thin standalone ``run_plan`` accept-&-rerun helper.

Synthetic traces / synthetic provenance results only — ZERO real-case addresses; the
rules key off generic aarch64 shapes + the target-agnostic import_map summary table.
Fixtures mirror spec_provenance_observation_planner.md §Fixtures (a)-(d).
"""

from __future__ import annotations

from engine.import_map import ImportMap
from engine.observation_planner import (
    DEFAULT_RULES,
    ObserveProposal,
    plan_for_result,
    rule_boundary_copy,
    rule_extern_call,
    rule_write_chain,
    run_plan,
    suggest_observations,
    suggest_proposals,
)
from engine.oracle_provenance import (
    BoundaryEdge,
    ProvenanceResult,
    ProvenanceVerdict,
    trace_provenance,
)
from engine.recapture_loop import _stamp_execution, _round_exec_id
from engine.runner_client import (
    ObservedState,
    ObservePoint,
    RerunResult,
    mem_snapshots_from_rerun,
)
from engine.types import Instruction, MemOp, MemSnapshot


def _ins(idx, pc, mnem, reads=None, writes=None, mem=()):
    return Instruction(idx=idx, pc=pc, bytes_=b"\x00\x00\x00\x00", mnemonic=mnem,
                       regs_read=dict(reads or {}), regs_write=dict(writes or {}),
                       mem=mem)


def _le(b: bytes) -> int:
    return int.from_bytes(b, "little")


EXPECTED = bytes([0x34, 0x15, 0x5f, 0xe9])
OUT = 0x72b18
UNK = 0x9000


# --------------------------------------------------------------------------- #
# Fixture (a) — each seed rule produces the right proposal + reason (+heuristic).
# --------------------------------------------------------------------------- #


def test_write_chain_rule_proposes_value_before_and_target_after():
    # A post-indexed store on the producer chain: capture the stored VALUE wN BEFORE
    # and the TARGET memory at [xK] (reg-relative) AFTER.
    store = _ins(5, 0x12002c98, "strb w0, [x19], #1")
    prov = ProvenanceResult(
        ProvenanceVerdict.NEEDS_OBSERVATION,
        chain=[{"idx": 5, "pc": "0x12002c98", "mnemonic": "strb w0, [x19], #1",
                "reads": []}],
        next_watch=[{"addr": "0x9000", "pc": "0x12002c98", "reason": "x"}],
    )
    props = suggest_proposals(prov, [store], rules=[rule_write_chain])
    assert len(props) == 2
    assert {p.heuristic for p in props} == {"write_chain"}

    before = next(p for p in props if p.when == "before")
    assert before.capture == ("regs",)
    assert before.regs == ("w0",)
    assert "produced value w0" in before.reason

    after = next(p for p in props if p.when == "after")
    assert after.capture == ("mem",)
    assert after.regs == ()
    assert len(after.mem_regrel) == 1
    rr = after.mem_regrel[0]
    assert rr.base_reg == "x19" and rr.width == 1 and rr.kind == "write"
    assert rr.pc == 0x12002c98
    assert "target memory at [x19]" in after.reason


def test_write_chain_rule_handles_plain_str_and_sized_widths():
    # A reg-indirect (non-post-indexed) `str x8, [x9]` is also a write-chain shape;
    # the width follows the mnemonic (str=8).
    store = _ins(2, 0x80000, "str x8, [x9]")
    prov = ProvenanceResult(
        ProvenanceVerdict.NEEDS_OBSERVATION,
        chain=[{"idx": 2, "pc": "0x80000", "mnemonic": "str x8, [x9]", "reads": []}],
    )
    props = suggest_proposals(prov, [store], rules=[rule_write_chain])
    before = next(p for p in props if p.when == "before")
    after = next(p for p in props if p.when == "after")
    assert before.regs == ("x8",)
    assert after.mem_regrel[0].base_reg == "x9"
    assert after.mem_regrel[0].width == 8


def test_extern_call_rule_proposes_abi_args_before_and_return_after():
    # A `bl` to a PLT stub resolved (via the import_map) to memcpy: capture the
    # memcpy ABI args (x0=dst,x1=src,x2=n) BEFORE and x0 AFTER.
    call = _ins(3, 0x70010, "bl #0x40a90")
    imap = ImportMap(by_plt_addr={0x40A90: "memcpy"})
    prov = ProvenanceResult(
        ProvenanceVerdict.OPAQUE_CALLEE, boundary_pcs=(0x70010,), chain=[])
    props = suggest_proposals(prov, [call], rules=[rule_extern_call], import_map=imap)
    assert len(props) == 2
    assert {p.heuristic for p in props} == {"extern_call"}

    before = next(p for p in props if p.when == "before")
    assert before.regs == ("x0", "x1", "x2")          # memcpy's declared abi_args
    assert "memcpy" in before.reason
    assert "x1=src" in before.reason                  # ABI roles surfaced

    after = next(p for p in props if p.when == "after")
    assert after.regs == ("x0",)
    assert "return register x0" in after.reason


def test_extern_call_rule_falls_back_to_generic_aapcs_when_no_summary():
    # A resolved extern with NO #6 summary entry is never silently skipped (A8④):
    # propose the generic AAPCS arg regs x0..x3 with an honest "no ABI summary" note.
    call = _ins(1, 0x70004, "bl #0x40b00")
    imap = ImportMap(by_plt_addr={0x40B00: "some_unknown_helper"})
    prov = ProvenanceResult(
        ProvenanceVerdict.OPAQUE_CALLEE, boundary_pcs=(0x70004,), chain=[])
    props = suggest_proposals(prov, [call], rules=[rule_extern_call], import_map=imap)
    before = next(p for p in props if p.when == "before")
    assert before.regs == ("x0", "x1", "x2", "x3")
    assert "no ABI summary" in before.reason


def test_extern_call_rule_does_not_fire_without_import_map():
    # No import_map ⇒ a `bl` cannot be resolved to a symbol ⇒ the rule does not fire
    # (the gap stays in next_watch — A8④).
    call = _ins(3, 0x70010, "bl #0x40a90")
    prov = ProvenanceResult(
        ProvenanceVerdict.OPAQUE_CALLEE, boundary_pcs=(0x70010,), chain=[])
    assert suggest_proposals(prov, [call], rules=[rule_extern_call]) == []
    # ... and with a map that does NOT resolve the target, also no proposal.
    imap = ImportMap(by_plt_addr={0x9999: "rand"})
    assert suggest_proposals(prov, [call], rules=[rule_extern_call],
                             import_map=imap) == []


def test_boundary_copy_rule_proposes_pre_transform_source_buffer():
    # Near a declared boundary edge: propose capturing the pre-transform SOURCE
    # buffer at the boundary's start PC, width from decode_meta.
    edge = BoundaryEdge(
        sink_surface=OUT, boundary_pc_from=0xB2128, boundary_pc_to=0xB212C,
        source_ptr=0x12312480, transform="base64", decode_meta={"raw_len": 3})
    prov = ProvenanceResult(
        ProvenanceVerdict.BOUNDARY_EDGE, anchored_edge=edge, chain=[])
    props = suggest_proposals(prov, [], rules=[rule_boundary_copy])
    assert len(props) == 1
    p = props[0]
    assert p.heuristic == "boundary_copy"
    assert p.when == "before"
    assert p.pc == 0xB2128
    assert p.capture == ("mem",)
    assert p.mem == ((0x12312480, 3),)               # width from raw_len
    assert "pre-transform source buffer" in p.reason


def test_boundary_copy_width_defaults_when_no_decode_meta_hint():
    edge = BoundaryEdge(
        sink_surface=OUT, boundary_pc_from=0xB2128, boundary_pc_to=0xB212C,
        source_ptr=0x12312480, transform="raw")
    prov = ProvenanceResult(
        ProvenanceVerdict.BOUNDARY_EDGE, anchored_edge=edge, chain=[])
    p = suggest_proposals(prov, [], rules=[rule_boundary_copy])[0]
    assert p.mem == ((0x12312480, 32),)              # conservative default


def test_boundary_copy_rule_does_not_fire_without_anchored_edge():
    prov = ProvenanceResult(ProvenanceVerdict.NEEDS_OBSERVATION, chain=[])
    assert suggest_proposals(prov, [], rules=[rule_boundary_copy]) == []


# --------------------------------------------------------------------------- #
# Fixture (a, cont.) — suggest_observations lowers proposals to ObservePoints,
# plan_for_result emits the observation_plan dicts (the field shape).
# --------------------------------------------------------------------------- #


def test_suggest_observations_lowers_to_runner_observe_points():
    store = _ins(5, 0x12002c98, "strb w0, [x19], #1")
    prov = ProvenanceResult(
        ProvenanceVerdict.NEEDS_OBSERVATION,
        chain=[{"idx": 5, "pc": "0x12002c98", "mnemonic": "strb w0, [x19], #1",
                "reads": []}])
    ops = suggest_observations(prov, [store], rules=[rule_write_chain])
    assert all(isinstance(o, ObservePoint) for o in ops)
    assert ops[0].pc == 0x12002c98 and ops[0].when == "before"
    assert ops[0].regs == ("w0",)
    assert ops[1].when == "after" and len(ops[1].mem_regrel) == 1


def test_plan_for_result_emits_audit_dicts_with_reason_and_heuristic():
    store = _ins(5, 0x12002c98, "strb w0, [x19], #1")
    prov = ProvenanceResult(
        ProvenanceVerdict.NEEDS_OBSERVATION,
        chain=[{"idx": 5, "pc": "0x12002c98", "mnemonic": "strb w0, [x19], #1",
                "reads": []}])
    plan = plan_for_result(prov, [store], rules=[rule_write_chain])
    assert all(set(("pc", "when", "capture", "reason", "heuristic")) <= set(d)
               for d in plan)
    assert {d["heuristic"] for d in plan} == {"write_chain"}
    assert plan[0]["pc"] == "0x12002c98"


# --------------------------------------------------------------------------- #
# Fixture (b) — run_plan round-trips a plan to snapshots under ONE execution_id.
# --------------------------------------------------------------------------- #


SRC32 = 0x1235D000      # synthetic source buffer the runner hands back


class _FakeAdapter:
    """Stands in for a Live-mode runner: when reran with a plan it hands back the
    bytes at the proposed observe points — proving run_plan closes the round-trip.
    A single rerun() call ⇒ ONE execution. Records the observe_points it was given so
    the test can assert the plan reached the adapter verbatim."""

    def __init__(self, output=EXPECTED, mem=None):
        self.output = bytes(output)
        self._mem = mem if mem is not None else {SRC32: EXPECTED, UNK: EXPECTED}
        self.seen_points = None

    def rerun(self, input_bytes, observe_points=None):
        self.seen_points = list(observe_points or [])
        # one ObservedState carrying every captured region — one execution.
        obs = ObservedState(pc=0x12002c98, when="after", regs={}, mem=dict(self._mem))
        return RerunResult(output=self.output, observations=(obs,))


def test_run_plan_reruns_the_plan_and_folds_snapshots_one_execution():
    plan = [
        ObservePoint(pc=0x12002c98, when="after", capture=("mem",), mem=((SRC32, 4),)),
        ObservePoint(pc=0x12002c90, when="before", capture=("regs",), regs=("x0",)),
    ]
    adapter = _FakeAdapter()
    result = run_plan(adapter, b"\x01\x02", plan)
    assert isinstance(result, RerunResult)
    assert result.output == EXPECTED
    # the plan reached the adapter verbatim (run_plan is a thin wrapper).
    assert adapter.seen_points == plan

    # the captured mem folds into canonical snapshots — the same call the loop makes.
    snaps = mem_snapshots_from_rerun(result)
    assert {s.addr for s in snaps} == {SRC32, UNK}
    assert all(isinstance(s, MemSnapshot) for s in snaps)

    # ONE rerun = ONE execution: stamped with the single round token, every snapshot
    # carries the SAME execution_id (the same-execution G1 invariant the loop relies
    # on, inherited by run_plan's single-rerun path).
    token = _round_exec_id(1, result)
    stamped = _stamp_execution(snaps, token)
    ids = {s.execution_id for s in stamped}
    assert ids == {token}                            # exactly one distinct id


def test_run_plan_snapshots_feed_back_into_trace_provenance():
    # The full intended use: run a plan, fold to snapshots, hand them to
    # trace_provenance by hand (run_plan does NOT drive a convergence loop).
    trace = [
        _ins(0, 0x70000, "ldr x8, [x9]", reads={"x9": UNK}, writes={"x8": 0},
             mem=(MemOp("r", UNK, 0, 4),)),
        _ins(1, 0x70004, "str x8, [x10]", reads={"x8": 0, "x10": OUT},
             mem=(MemOp("w", OUT, 0, 4),)),
    ]
    prov = trace_provenance(trace, EXPECTED, sink_base=OUT)
    assert prov.verdict is ProvenanceVerdict.NEEDS_OBSERVATION
    assert any(w["addr"] == f"0x{UNK:x}" for w in prov.next_watch)

    result = run_plan(_FakeAdapter(mem={UNK: EXPECTED}), b"\x01", [
        ObservePoint(pc=0x70000, when="before", capture=("mem",), mem=((UNK, 4),))])
    snaps = mem_snapshots_from_rerun(result)
    again = trace_provenance(trace, EXPECTED, sink_base=OUT, snapshots=snaps)
    # the UNK producer gap is now CLOSED — no longer surfaced.
    assert all(w["addr"] != f"0x{UNK:x}" for w in again.next_watch)


def test_run_plan_rejects_misbehaving_adapter():
    class _BadAdapter:
        def rerun(self, input_bytes, observe_points=None):
            return {"output": b"x"}      # not a RerunResult

    import pytest
    with pytest.raises(TypeError, match="must return a RerunResult"):
        run_plan(_BadAdapter(), b"\x01", [])


# --------------------------------------------------------------------------- #
# Fixture (c) — an unmatched gap is NEVER silently dropped: it stays in next_watch,
# and the plan does not fabricate an entry for it (A8④).
# --------------------------------------------------------------------------- #


def test_unmatched_gap_stays_in_next_watch_and_yields_no_plan_entry():
    # A NEEDS_OBSERVATION gap at a PLAIN load (not a store, not a call) — no seed rule
    # matches its shape. The gap MUST remain in next_watch; the plan stays empty.
    trace = [
        _ins(0, 0x70000, "ldr x8, [x9]", reads={"x9": UNK}, writes={"x8": 0x34},
             mem=(MemOp("r", UNK, 0x34, 4),)),
        _ins(1, 0x70004, "strb w8, [x12]", reads={"x8": 0x34, "x12": OUT},
             mem=(MemOp("w", OUT, EXPECTED[0], 1),)),
    ]
    snaps = [MemSnapshot(addr=OUT, data=EXPECTED)]
    prov = trace_provenance(trace, EXPECTED, sink_base=OUT, snapshots=snaps)
    assert prov.verdict is ProvenanceVerdict.NEEDS_OBSERVATION
    # the gap on UNK (read at the ldr) is surfaced in next_watch ...
    assert any(w["addr"] == f"0x{UNK:x}" for w in prov.next_watch)

    # ... but no write_chain rule matches the ldr shape (the strb is on the chain,
    # though — so to make a TRULY unmatched gap, run only extern_call without a map).
    plan = plan_for_result(prov, trace, rules=[rule_extern_call])
    assert plan == []                                # no fabricated entry
    # the gap is STILL there for the consumer to act on — never hidden by an empty plan.
    assert any(w["addr"] == f"0x{UNK:x}" for w in prov.next_watch)


def test_empty_plan_never_replaces_a_nonempty_next_watch():
    # Even with the FULL default rule set, a result whose gap no rule covers yields an
    # empty plan while next_watch keeps the gap (the plan is additive, never a swap).
    trace = [_ins(0, 0x70000, "nop")]
    snaps = [MemSnapshot(addr=OUT, data=EXPECTED)]
    prov = trace_provenance(trace, EXPECTED, sink_base=OUT, snapshots=snaps)
    assert prov.verdict is ProvenanceVerdict.NEEDS_OBSERVATION
    assert prov.next_watch != []
    assert plan_for_result(prov, trace, rules=DEFAULT_RULES) == []


def test_proposals_are_deduplicated_preserving_order():
    # Two chain steps at the SAME store PC must not double-propose.
    store = _ins(5, 0x12002c98, "strb w0, [x19], #1")
    prov = ProvenanceResult(
        ProvenanceVerdict.NEEDS_OBSERVATION,
        chain=[{"idx": 5, "pc": "0x12002c98", "mnemonic": "strb w0, [x19], #1",
                "reads": []}],
        next_watch=[
            {"addr": "0x9000", "pc": "0x12002c98", "reason": "x"},
            {"addr": "0x9004", "pc": "0x12002c98", "reason": "y"},
        ],
    )
    props = suggest_proposals(prov, [store], rules=[rule_write_chain])
    assert len(props) == 2                            # not 4 (deduped across gaps)


# --------------------------------------------------------------------------- #
# Fixture (d) — regression: default plan_observations=False ⇒ observation_plan is
# NOT generated, NOT in to_dict, serialization byte-for-byte unchanged.
# --------------------------------------------------------------------------- #


def _write_chain_full_trace():
    # A real NEEDS_OBSERVATION trace whose producer chain holds a post-indexed store —
    # i.e. the write_chain rule WOULD fire if planning were enabled.
    return [
        _ins(0, 0x12002c98, "strb w0, [x19], #1", reads={"w0": 0x99, "x19": OUT},
             mem=(MemOp("w", OUT, EXPECTED[0], 1),)),
    ]


def test_default_off_omits_observation_plan_from_to_dict():
    trace = _write_chain_full_trace()
    snaps = [MemSnapshot(addr=OUT, data=EXPECTED)]
    res = trace_provenance(trace, EXPECTED, sink_base=OUT, snapshots=snaps)
    assert res.observation_plan == []
    assert "observation_plan" not in res.to_dict()


def test_default_off_serialization_byte_for_byte_unchanged():
    # The to_dict of a default (planning-off) result must be identical to one built
    # with plan_observations explicitly False — additive field, zero drift.
    trace = _write_chain_full_trace()
    snaps = [MemSnapshot(addr=OUT, data=EXPECTED)]
    import json
    a = trace_provenance(trace, EXPECTED, sink_base=OUT, snapshots=snaps)
    b = trace_provenance(trace, EXPECTED, sink_base=OUT, snapshots=snaps,
                         plan_observations=False)
    assert json.dumps(a.to_dict(), sort_keys=True) == \
        json.dumps(b.to_dict(), sort_keys=True)


def test_plan_observations_on_attaches_plan_alongside_next_watch():
    # The opt-in path: with plan_observations=True the heuristic plan is attached and
    # appears in to_dict ALONGSIDE next_watch (never replacing it).
    trace = _write_chain_full_trace()
    snaps = [MemSnapshot(addr=OUT, data=EXPECTED)]
    res = trace_provenance(trace, EXPECTED, sink_base=OUT, snapshots=snaps,
                           plan_observations=True)
    assert res.observation_plan != []
    assert {d["heuristic"] for d in res.observation_plan} == {"write_chain"}
    d = res.to_dict()
    assert "observation_plan" in d
    assert "next_watch" in d                          # plan is ADDITIVE


def test_plan_observations_on_extern_uses_import_map():
    # opt-in plan with an import_map resolves a bl → symbol (the extern_call rule).
    trace = [
        _ins(0, 0x70000, "bl #0x40a90", reads={}),
        _ins(1, 0x70004, "ldr x0, [x10]", reads={"x10": OUT},
             mem=(MemOp("r", OUT, _le(EXPECTED), 4),)),
    ]
    imap = ImportMap(by_plt_addr={0x40A90: "memcpy"})
    res = trace_provenance(trace, EXPECTED, sink_base=OUT,
                           plan_observations=True, import_map=imap)
    assert res.verdict is ProvenanceVerdict.OPAQUE_CALLEE
    assert any(d["heuristic"] == "extern_call" for d in res.observation_plan)


# --------------------------------------------------------------------------- #
# Proof-point flavour (TC2 sink→src32 shape) — the planner auto-proposes a
# write_chain capture at the post-indexed store, target reg-relative. Generic
# shapes only (the spec's exact addresses are an illustration, not baked here).
# --------------------------------------------------------------------------- #


def test_proof_point_write_chain_shape_autoproposes_value_and_target():
    store = _ins(7, 0x12002c98, "strb w0, [x19], #1")
    prov = ProvenanceResult(
        ProvenanceVerdict.NEEDS_OBSERVATION,
        chain=[{"idx": 7, "pc": "0x12002c98", "mnemonic": "strb w0, [x19], #1",
                "reads": []}],
        next_watch=[{"addr": f"0x{UNK:x}", "pc": "0x12002c98", "reason": "gap"}])
    plan = plan_for_result(prov, [store], rules=DEFAULT_RULES)
    whens = {(d["pc"], d["when"]) for d in plan}
    assert ("0x12002c98", "before") in whens          # capture the value w0
    assert ("0x12002c98", "after") in whens           # capture the target buffer
    assert all(d.get("reason") for d in plan)         # every proposal carries a reason


def test_observe_proposal_to_observe_point_preserves_fields():
    p = ObserveProposal(pc=0x1000, when="before", capture=("regs",),
                        heuristic="write_chain", reason="r", regs=("w0",))
    op = p.to_observe_point()
    assert op.pc == 0x1000 and op.when == "before"
    assert op.capture == ("regs",) and op.regs == ("w0",)
