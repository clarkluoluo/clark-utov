"""Cohort-diff input/seed-dependence localization.

Pins the mechanical "which windows vary with the seed" analysis and the three
cases a naive register-diff misses (dev verification-localization addendum):
memory/staging (M-mem), control-flow divergence (M5), and the all-invariant
opaque outcome (M6). All synthetic; zero case-specific coordinates.
"""

from __future__ import annotations

from engine.cohort_diff import (
    InputDependenceMap,
    localize_input_dependence,
)
from engine.types import Instruction, MemOp


def _i(idx, pc, mnem, *, reads=None, writes=None, mem=()):
    return Instruction(idx=idx, pc=pc, bytes_=b"\x00\x00\x00\x00", mnemonic=mnem,
                       regs_read=reads or {}, regs_write=writes or {}, mem=tuple(mem))


def test_insufficient_cohort_under_two_traces():
    rep = localize_input_dependence([[_i(0, 0x1000, "nop")]])
    assert isinstance(rep, InputDependenceMap)
    assert rep.verdict == "insufficient" and rep.divergence_idx is None


def test_insufficient_when_cohort_did_not_vary_the_seed():
    # Same input_key in every vector → the cohort did not vary the seed; can't tell.
    t = [_i(0, 0x1000, "add x0, x0, 1", writes={"x0": 5})]
    rep = localize_input_dependence([t, t], input_keys=["same", "same"])
    assert rep.verdict == "insufficient"
    assert "did not vary the seed" in " ".join(rep.reasons)


def test_localize_register_varying_window():
    # Reg write value differs across vectors → that position is seed-varying.
    a = [_i(0, 0x1000, "mov w0, ?", writes={"w0": 0x11}),
         _i(1, 0x1004, "add w1, w1, 1", writes={"w1": 7})]   # constant
    b = [_i(0, 0x1000, "mov w0, ?", writes={"w0": 0x22}),
         _i(1, 0x1004, "add w1, w1, 1", writes={"w1": 7})]
    rep = localize_input_dependence([a, b], input_keys=["A", "B"])
    assert rep.verdict == "localized"
    assert rep.divergence_idx == 0
    assert rep.varying_idxs == (0,)
    assert rep.varying[0].varying_regs == ("w0",)
    assert rep.window_is_seed_varying(0, 0) and not rep.window_is_seed_varying(1, 1)


def test_localize_memory_varying_window_not_just_registers():
    # dev addendum M-mem / M6: the seed enters through a STORE — invisible to a
    # register-only diff. The memory write value must be diffed.
    a = [_i(0, 0x1000, "str w0, [x9]", mem=[MemOp("w", 0x9000, 0xAA, 4)])]
    b = [_i(0, 0x1000, "str w0, [x9]", mem=[MemOp("w", 0x9000, 0xBB, 4)])]
    rep = localize_input_dependence([a, b], input_keys=["A", "B"])
    assert rep.verdict == "localized"
    assert rep.varying[0].varying_mem == (0x9000,)
    assert rep.varying[0].varying_regs == ()


def test_all_invariant_with_varied_seed_is_opaque_not_no_dependence():
    # dev addendum M6 (THE F0 case): the cohort genuinely differs, yet no
    # observable reg/mem state varies → OPAQUE (seed hidden in staging), NOT a
    # silent "no dependence". Must steer the agent to symex, not to "constant".
    a = [_i(0, 0x1000, "add x0, x0, 1", writes={"x0": 0xFB9881B1}),
         _i(1, 0x1004, "eor x1, x1, x2", writes={"x1": 0})]
    b = [_i(0, 0x1000, "add x0, x0, 1", writes={"x0": 0xFB9881B1}),
         _i(1, 0x1004, "eor x1, x1, x2", writes={"x1": 0})]
    rep = localize_input_dependence([a, b], input_keys=["dm1", "dm2"])  # seeds differ
    assert rep.verdict == "opaque" and rep.is_opaque
    assert rep.varying == () and rep.divergence_idx is None
    assert "opaque" in rep.advisory.lower()
    # opaque must NOT read as "this window is constant/safe":
    assert rep.window_is_seed_varying(0, 1) is False   # but is_opaque flags the caveat


def test_control_flow_divergence_is_a_seed_signal_not_a_crash():
    # dev addendum M5: different seeds take different paths → traces don't align by
    # index. The first PC mismatch is a seed-dependent branch, recorded, scan stops.
    a = [_i(0, 0x1000, "cmp w0, w1"),
         _i(1, 0x1004, "b.eq #0x1010"),
         _i(2, 0x1008, "mov w2, #1", writes={"w2": 1})]
    b = [_i(0, 0x1000, "cmp w0, w1"),
         _i(1, 0x1004, "b.eq #0x1010"),
         _i(2, 0x1010, "mov w2, #2", writes={"w2": 2})]   # different path at pos 2
    rep = localize_input_dependence([a, b], input_keys=["A", "B"])
    assert rep.verdict == "localized"
    assert rep.divergence_idx == 2
    assert rep.varying[-1].control_flow is True
    assert "control-flow divergence" in " ".join(rep.reasons)


def test_partial_windows_vary_others_invariant():
    a = [_i(0, 0x1000, "mov w0, ?", writes={"w0": 1}),       # constant
         _i(1, 0x1004, "mov w1, ?", writes={"w1": 0xAA}),     # varies
         _i(2, 0x1008, "mov w2, ?", writes={"w2": 9})]        # constant
    b = [_i(0, 0x1000, "mov w0, ?", writes={"w0": 1}),
         _i(1, 0x1004, "mov w1, ?", writes={"w1": 0xBB}),
         _i(2, 0x1008, "mov w2, ?", writes={"w2": 9})]
    rep = localize_input_dependence([a, b], input_keys=["A", "B"])
    assert rep.verdict == "localized" and rep.varying_idxs == (1,)
    assert rep.divergence_idx == 1


def test_low_observability_not_opaque_when_diff_dims_empty():
    # dev opaque trust-gate (the reference case false-opaque): the trace's regs_write
    # and mem are basically EMPTY — the input difference lives in regs_read. The
    # old code would diff the empty written dims, see no variation, and FALSELY
    # report opaque. The trust-gate must instead call this a measurement blind
    # spot: inconclusive_low_observability, NOT a real opaque frontier.
    a = [_i(0, 0x1000, "ldr w0, [x9]", reads={"w0": 0xAA}),
         _i(1, 0x1004, "eor w1, w0, w2", reads={"w0": 0xAA, "w2": 0x01}),
         _i(2, 0x1008, "ret", reads={"w0": 0xAA})]
    b = [_i(0, 0x1000, "ldr w0, [x9]", reads={"w0": 0xBB}),
         _i(1, 0x1004, "eor w1, w0, w2", reads={"w0": 0xBB, "w2": 0x01}),
         _i(2, 0x1008, "ret", reads={"w0": 0xBB})]
    rep = localize_input_dependence([a, b], input_keys=["A", "B"])
    assert rep.verdict == "inconclusive_low_observability"
    assert rep.is_low_observability and not rep.is_opaque
    assert rep.observability_rate == 0.0 and rep.observable_positions == 0
    assert rep.regs_read_varies is True
    assert "blind spot" in rep.advisory.lower()
    assert any("trust-gate" in r for r in rep.reasons)


def test_low_observability_triggers_on_sparse_coverage_alone():
    # Coverage below threshold even without regs_read variation: one populated
    # write position out of many empty ones → diffed dims too sparse to trust a
    # "no variation" conclusion. Honest inconclusive, not opaque.
    a = ([_i(0, 0x1000, "mov w0, ?", writes={"w0": 5})]
         + [_i(i, 0x1000 + 4 * i, "nop") for i in range(1, 40)])
    b = ([_i(0, 0x1000, "mov w0, ?", writes={"w0": 5})]
         + [_i(i, 0x1000 + 4 * i, "nop") for i in range(1, 40)])
    rep = localize_input_dependence([a, b], input_keys=["A", "B"])
    assert rep.verdict == "inconclusive_low_observability"
    assert rep.observable_positions == 1 and rep.observability_rate < 0.05
    assert rep.regs_read_varies is False


def test_true_opaque_survives_trust_gate_when_well_observed():
    # dev opaque trust-gate: regs_write populated at EVERY position (coverage 100%),
    # values genuinely constant across vectors, regs_read also flat → this is a
    # REAL opaque frontier and must NOT be downgraded by the trust-gate.
    a = [_i(0, 0x1000, "add x0, x0, 1", reads={"x0": 9}, writes={"x0": 0xFB9881B1}),
         _i(1, 0x1004, "eor x1, x1, x2", reads={"x1": 0}, writes={"x1": 0})]
    b = [_i(0, 0x1000, "add x0, x0, 1", reads={"x0": 9}, writes={"x0": 0xFB9881B1}),
         _i(1, 0x1004, "eor x1, x1, x2", reads={"x1": 0}, writes={"x1": 0})]
    rep = localize_input_dependence([a, b], input_keys=["dm1", "dm2"])
    assert rep.verdict == "opaque" and rep.is_opaque
    assert not rep.is_low_observability
    assert rep.observability_rate == 1.0
    assert rep.regs_read_varies is False


def test_observability_threshold_is_parameterized():
    # Zero case-specific constants: the gate threshold is a parameter. A cohort
    # just under a stricter threshold flips to inconclusive; relaxing it keeps the
    # original opaque path. 1/2 observable positions = 50% coverage.
    a = [_i(0, 0x1000, "mov w0, ?", writes={"w0": 7}),
         _i(1, 0x1004, "nop")]
    b = [_i(0, 0x1000, "mov w0, ?", writes={"w0": 7}),
         _i(1, 0x1004, "nop")]
    strict = localize_input_dependence([a, b], input_keys=["A", "B"],
                                       min_observability=0.75)
    assert strict.verdict == "inconclusive_low_observability"
    lax = localize_input_dependence([a, b], input_keys=["A", "B"],
                                    min_observability=0.25)
    assert lax.verdict == "opaque"


def test_coupling_axis_controlled_out_via_ignore():
    # dev addendum M4: the cohort co-varies a nonce (w30) alongside the seed (w0).
    # Without control, both look seed-varying. ignore_regs scopes the diff to the
    # seed's own footprint.
    a = [_i(0, 0x1000, "mov w0, ?", writes={"w0": 0x11, "w30": 0x1001})]
    b = [_i(0, 0x1000, "mov w0, ?", writes={"w0": 0x22, "w30": 0x2002})]
    raw = localize_input_dependence([a, b], input_keys=["A", "B"])
    assert set(raw.varying[0].varying_regs) == {"w0", "w30"}   # both, uncontrolled
    scoped = localize_input_dependence([a, b], input_keys=["A", "B"], ignore_regs=["w30"])
    assert scoped.varying[0].varying_regs == ("w0",)           # nonce controlled out
    assert scoped.ignored_regs == ("w30",)


# --- Phase 3: localize-side opaque advisory (EA-varying staging PCs) ----------

def test_phase3_form_f_address_varying_staging_surfaces_advisory():
    # Form F: 2 vectors, each accessing a DIFFERENT effective address at the same
    # store PC and the same load PC. The stored *value* lands at addresses that
    # never enter the common-address intersection a value diff sees → the cohort
    # reads opaque (no observable variation), yet the EA itself varies per vector.
    # regs_write is populated and identical across vectors (trust-gate passes),
    # so the verdict stays a TRUE opaque — and Phase 3 surfaces the EA-varying PCs.
    a = [_i(0, 0x1000, "str x8, [x10]", writes={"x9": 1},
            mem=[MemOp("w", 0xA000, 0x41, 8)]),
         _i(1, 0x1004, "ldr x11, [x10]", writes={"x12": 1},
            mem=[MemOp("r", 0xA000, 0x41, 8)])]
    b = [_i(0, 0x1000, "str x8, [x10]", writes={"x9": 1},
            mem=[MemOp("w", 0xB000, 0x41, 8)]),   # different store EA
         _i(1, 0x1004, "ldr x11, [x10]", writes={"x12": 1},
            mem=[MemOp("r", 0xB000, 0x41, 8)])]   # different load EA
    rep = localize_input_dependence([a, b], input_keys=["A", "B"])
    assert rep.verdict == "opaque" and rep.is_opaque
    adv = rep.opaque_staging_advisory
    assert adv is not None
    sites = adv["ea_varying_sites"]
    assert isinstance(sites, list) and len(sites) == 2     # one store PC, one load PC
    by_rw = {s["rw"]: s for s in sites}
    assert set(by_rw) == {"w", "r"}
    assert by_rw["w"]["pc"] == "0x1000" and by_rw["w"]["idx"] == 0
    assert by_rw["r"]["pc"] == "0x1004" and by_rw["r"]["idx"] == 1
    assert by_rw["w"]["n_distinct_ea"] == 2
    assert set(by_rw["w"]["sample_eas"]) == {"0xa000", "0xb000"}
    # the advisory is also carried in the serialized map (opaque path only).
    assert rep.to_dict()["opaque_staging_advisory"]["ea_varying_sites"]


def test_phase3_form_g_truly_invisible_opaque_empty_advisory_with_note():
    # Form G: a TRUE opaque with no address/EA-level visible structure at all
    # (the F0-style staging — every reg/mem state constant, no mem accesses to
    # vary). Advisory must be empty + an honest note, never a fabricated locator.
    a = [_i(0, 0x1000, "add x0, x0, 1", writes={"x0": 0xFB9881B1}),
         _i(1, 0x1004, "eor x1, x1, x2", writes={"x1": 0})]
    b = [_i(0, 0x1000, "add x0, x0, 1", writes={"x0": 0xFB9881B1}),
         _i(1, 0x1004, "eor x1, x1, x2", writes={"x1": 0})]
    rep = localize_input_dependence([a, b], input_keys=["dm1", "dm2"])
    assert rep.verdict == "opaque"
    adv = rep.opaque_staging_advisory
    assert adv is not None
    assert adv["ea_varying_sites"] == []
    assert "genuinely invisible" in adv["note"] or "no store/load" in adv["note"]


def test_phase3_advisory_is_none_on_non_opaque_paths():
    # Regression: localized / inconclusive_low_observability / insufficient all
    # leave opaque_staging_advisory None, and the serialized dict has no such key.
    loc_a = [_i(0, 0x1000, "mov w0, ?", writes={"w0": 0x11})]
    loc_b = [_i(0, 0x1000, "mov w0, ?", writes={"w0": 0x22})]
    loc = localize_input_dependence([loc_a, loc_b], input_keys=["A", "B"])
    assert loc.verdict == "localized"
    assert loc.opaque_staging_advisory is None
    assert "opaque_staging_advisory" not in loc.to_dict()

    insf = localize_input_dependence([loc_a])    # < 2 vectors
    assert insf.verdict == "insufficient"
    assert insf.opaque_staging_advisory is None
    assert "opaque_staging_advisory" not in insf.to_dict()

    lo_a = [_i(0, 0x1000, "nop"), _i(1, 0x1004, "nop")]
    lo = localize_input_dependence([lo_a, lo_a], input_keys=["A", "B"])
    assert lo.verdict == "inconclusive_low_observability"
    assert lo.opaque_staging_advisory is None
    assert "opaque_staging_advisory" not in lo.to_dict()
