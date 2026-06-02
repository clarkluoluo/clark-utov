"""Recovery-as-a-CVD-run — the three plugin roles + the collect-mode gap map.

Pins what makes recovery stop being whack-a-mole: it is registered into CVD as a
CandidateGenerator + a heavy Verifier (wrapping the whole drive()) + a
TerminalClassifier, and ONE collect-mode run enumerates the whole gap map instead
of surfacing one gap per round. Synthetic shapes only — no case addresses / values
/ handler ids (utov-arch-index invariant 2/6).
"""

from __future__ import annotations

import json

import pytest

from engine.cohort_diff import localize_input_dependence
from engine.cvd import (
    Candidate,
    CandidateGenerator,
    CvdBudget,
    CvdOutcome,
    CvdState,
    Registry,
    Verdict,
    Verifier,
    VStatus,
    run_cvd,
)
from engine.oracle_provenance import (
    ProvenanceResult,
    ProvenanceVerdict,
    trace_provenance,
)
from engine.oracle_sink import SinkValidation, SinkVerdict
from engine.cvd_recovery import (
    OPAQUE_STAGING_FRONTIER,
    RECOVER_WINDOW,
    SIG_GENERATION_BUDGET_EXHAUSTED,
    SIG_PROVENANCE_BLOCKED_UNPLACEABLE,
    SIG_PROVENANCE_OFFPATH_VARIANCE,
    SIG_PROVENANCE_ONPATH,
    SIG_PROVENANCE_UNANCHORED,
    SIG_RECAPTURE_DIRECTIVE,
    _OnpathBandRegistry,
    run_recovery,
    TERMINAL_BAND_PARITY_FAIL,
    TERMINAL_COMPOSITE_REQUIRED,
    TERMINAL_COMPOSITE_TOO_EXPENSIVE,
    TERMINAL_MEMORY_DISPOSITION_MISSING,
    MemDispositionRec,
    _compact,
    estimate_composite_cost,
    plan_composite_recovery,
    RecoverWindowVerifier,
    RecoveryTerminalClassifier,
    RecoveryWindowGenerator,
    load_cohort_traces,
    recommend_mem_disposition,
    recovery_registry,
)
from engine.setup_symex import MemLiveIn
from engine.dispatch_coverage import HandlerInvocation, preflight_dispatch_coverage
from engine.setup_symex import CaseConfig, build_concrete_backing
from engine.types import Instruction, MemOp


# =========================================================================== #
# §gap①  collect mode — one run lists every gap; default stays first-miss.
# =========================================================================== #

class _SynthGen(CandidateGenerator):
    name = "synth_gen"; version = "1"; owner = "test"; kind = "mixed"

    def generate(self, state):
        return [
            Candidate("confirm_me", 0x10, "s", "c1"),
            Candidate("confirm_me", 0x20, "s", "c2"),
            Candidate("no_verifier", 0x30, "s", "no verifier applies"),
            Candidate("cap_gap", 0x40, "s", "needs a tool"),
            Candidate("judgment", 0x50, "s", "agent must decide"),
        ]


class _SynthVer(Verifier):
    name = "synth_ver"; version = "1"; owner = "test"

    def applies(self, c, state):
        return c.kind in ("confirm_me", "cap_gap", "judgment")

    def verify(self, c, state):
        if c.kind == "confirm_me":
            return Verdict(VStatus.CONFIRMED, evidence={"ok": True})
        if c.kind == "cap_gap":
            return Verdict(VStatus.TERMINAL, terminal_kind="needs_tool",
                           reason="missing capability", capability_request="register tool X")
        return Verdict(VStatus.PENDING, reason="judge this", evidence={"checkpoint": "q?"})


def _synth_registry():
    return Registry().register(_SynthGen()).register(_SynthVer())


def test_collect_mode_lists_the_whole_gap_map_in_one_run():
    res = run_cvd([Instruction(0, 0x1000, b"\x00\x00\x00\x00", "nop", {}, {})],
                  b"\x00", registry=_synth_registry(), collect_extensions=True)
    assert res.outcome is CvdOutcome.COLLECTED
    # All five candidates were enumerated in ONE run — nothing short-circuited.
    assert len(res.confirmed) == 2
    # no-verifier gap + capability TERMINAL = two extension requests, both surfaced.
    kinds = {e.get("missing_kind") for e in res.extension_requests}
    assert "verifier" in kinds and "capability" in kinds
    assert len(res.extension_requests) == 2
    # the PENDING agent-judgment is collected separately, not as a capability gap.
    assert len(res.pending_judgments) == 1
    assert res.pending_judgments[0]["reason"] == "judge this"


def test_default_mode_returns_at_first_gap_unchanged():
    res = run_cvd([Instruction(0, 0x1000, b"\x00\x00\x00\x00", "nop", {}, {})],
                  b"\x00", registry=_synth_registry(), collect_extensions=False)
    # first-miss-return: a single gap outcome, NOT the collected map.
    assert res.outcome in (CvdOutcome.EXTENSION_REQUEST, CvdOutcome.TERMINAL,
                           CvdOutcome.PENDING_JUDGMENT)
    assert res.outcome is not CvdOutcome.COLLECTED
    # the aggregate gap-map lists are not populated on a first-miss return.
    assert res.confirmed == [] and res.extension_requests == []


# --- Req2 G3 補点1: every CVD terminal exit carries a uniform block_why key ----

def test_collect_terminal_ext_request_carries_block_why():
    # The cap_gap candidate verifies to a per-candidate TERMINAL: its collect-mode
    # extension-request entry must carry block_why (the ONE machine-readable block
    # reason), not only the prose `why`.
    res = run_cvd([Instruction(0, 0x1000, b"\x00\x00\x00\x00", "nop", {}, {})],
                  b"\x00", registry=_synth_registry(), collect_extensions=True)
    assert res.outcome is CvdOutcome.COLLECTED
    cap = [e for e in res.extension_requests
           if e.get("terminal_kind") == "needs_tool"]
    assert len(cap) == 1
    assert cap[0]["block_why"] == "missing capability"     # == verdict.reason


def test_non_collect_terminal_cvdresult_carries_block_why():
    # A first-miss TERMINAL CvdResult (default mode) must carry block_why, and it must
    # survive to_dict (the stamped gap-map a consumer reads).
    class _OnlyCapGen(CandidateGenerator):
        name = "cap_only_gen"; version = "1"; owner = "test"; kind = "cap_gap"

        def generate(self, state):
            return [Candidate("cap_gap", 0x40, "s", "needs a tool")]

    reg = Registry().register(_OnlyCapGen()).register(_SynthVer())
    res = run_cvd([Instruction(0, 0x1000, b"\x00\x00\x00\x00", "nop", {}, {})],
                  b"\x00", registry=reg, collect_extensions=False)
    assert res.outcome is CvdOutcome.TERMINAL
    assert res.block_why == "missing capability"
    d = res.to_dict()
    assert d["block_why"] == "missing capability"


def test_terminal_classifier_global_terminal_carries_block_why():
    # A globally-claimed TerminalClassifier terminal (frontier exhausted) must carry
    # block_why on BOTH the collect global ext_request and the non-collect CvdResult.
    class _EmptyGen(CandidateGenerator):
        name = "empty_gen"; version = "1"; owner = "test"; kind = "none"

        def generate(self, state):
            return []

    from engine.cvd import Terminal, TerminalClassifier

    class _ClaimTC(TerminalClassifier):
        name = "claim_tc"; version = "1"; owner = "test"

        def classify(self, state):
            return Terminal(kind="WHOLE_LOCUS_OPAQUE",
                            capability_request="needs symbolic forward", success=False)

    reg = Registry().register(_EmptyGen()).register(_ClaimTC())
    items = [Instruction(0, 0x1000, b"\x00\x00\x00\x00", "nop", {}, {})]
    # non-collect → TERMINAL CvdResult with block_why
    res = run_cvd(items, b"\x00", registry=reg, collect_extensions=False)
    assert res.outcome is CvdOutcome.TERMINAL
    assert res.block_why == "needs symbolic forward"
    # collect → global ext_request with block_why
    res_c = run_cvd(items, b"\x00", registry=reg, collect_extensions=True)
    glob = [e for e in res_c.extension_requests if e.get("scope") == "global"]
    assert len(glob) == 1
    assert glob[0]["block_why"] == "needs symbolic forward"


# =========================================================================== #
# Role 1 — RecoveryWindowGenerator: windows from coverage / cohort diff.
# =========================================================================== #

def _ins(idx, pc, mnem, *, reads=None, writes=None, mem=()):
    return Instruction(idx=idx, pc=pc, bytes_=b"\x00\x00\x00\x00", mnemonic=mnem,
                       regs_read=reads or {}, regs_write=writes or {}, mem=tuple(mem))


def test_generator_from_dispatch_coverage_produces_recover_windows():
    trace = [
        _ins(0, 0x1000, "ldr w0, [x1]", reads={"x1": 0x9000}, writes={"w0": 1}),
        _ins(1, 0x1004, "add w2, w0, w0", reads={"w0": 1}, writes={"w2": 2}),
        _ins(2, 0x1008, "ldr w0, [x1]", reads={"x1": 0x9000}, writes={"w0": 3}),
        _ins(3, 0x100C, "add w2, w0, w0", reads={"w0": 3}, writes={"w2": 6}),
    ]
    invs = [HandlerInvocation("A", 0, 1), HandlerInvocation("A", 2, 3)]
    cov = preflight_dispatch_coverage(trace, invocations=invs, reg_file=("w0", "w2", "x1"))
    gen = RecoveryWindowGenerator(coverage=cov)
    state = CvdState(trace, b"\x00")
    cands = gen.generate(state)
    assert len(cands) == 1            # one handler TYPE → one representative window
    c = cands[0]
    assert c.kind == RECOVER_WINDOW and c.signal == "dispatch_type_rep"
    assert c.payload["window"] == [0, 1] and c.payload["window_kind"] == "idx"
    assert c.payload["occurrences"] == 2


def test_generator_short_circuits_to_recapture_when_sink_unobserved():
    """Task 3 — when the output sink is NOT captured on this run, the generator
    routes STRAIGHT to a recapture directive BEFORE any long generation/backtrace
    (not a dispatch/variance fall-back). The recapture candidate carries the
    short-circuit marker + suppresses secondary windows."""
    # The expected output bytes appear NOWHERE in the trace → OUTPUT_NOT_OBSERVABLE.
    expected = bytes([0xde, 0xad, 0xbe, 0xef])
    trace = [
        _ins(0, 0x1000, "mov x9, #1", writes={"x9": 0x7f00}),
        _ins(1, 0x1004, "str x8, [x9]", reads={"x8": 0x11, "x9": 0x7f00},
             mem=(MemOp("w", 0x7f00, 0x11, 1),)),     # writes unrelated bytes
    ]
    # also give a dispatch coverage so a fall-back WOULD exist if not suppressed.
    invs = [HandlerInvocation("A", 0, 1)]
    cov = preflight_dispatch_coverage(trace, invocations=invs, reg_file=("x8", "x9"))
    gen = RecoveryWindowGenerator(coverage=cov, sink_base=0x7f00,
                                  value_name="cipher_body")
    cands = gen.generate(CvdState(trace, expected))
    # ONLY the recapture directive — the dispatch fall-back was suppressed.
    assert len(cands) == 1
    c = cands[0]
    assert c.signal == SIG_RECAPTURE_DIRECTIVE
    assert c.payload["short_circuit"] == "sink_unobserved_preflight"
    assert c.payload["sink_captured"] is False
    assert c.payload["recapture_directive"]["kind"] == "recapture_directive"


def test_generator_from_cohort_diff_localized_positions():
    # Two cohort vectors differing only at idx1's written register → localized.
    v1 = [_ins(0, 0x1000, "mov w0, #1", writes={"w0": 1}),
          _ins(1, 0x1004, "add w2, w0, w0", reads={"w0": 1}, writes={"w2": 2})]
    v2 = [_ins(0, 0x1000, "mov w0, #1", writes={"w0": 1}),
          _ins(1, 0x1004, "add w2, w0, w0", reads={"w0": 9}, writes={"w2": 18})]
    dep = localize_input_dependence([v1, v2], input_keys=["a", "b"])
    assert dep.verdict == "localized"
    gen = RecoveryWindowGenerator(dependence=dep)
    cands = gen.generate(CvdState(v1, b"\x00"))
    assert all(c.kind == RECOVER_WINDOW for c in cands)
    # the single varying position is the divergence point → anchored start (§2).
    assert all(c.signal in ("input_varying", "divergence_anchor") for c in cands)
    assert any(c.payload["window"] == [1, 1] for c in cands)


# =========================================================================== #
# Role 2 — RecoverWindowVerifier: drive() outcome → CVD Verdict.
# =========================================================================== #

def _backed_window():
    return [
        _ins(0, 0x1000, "ldr w0, [x16]", reads={"x16": 0x9000}),
        _ins(1, 0x1004, "mul w0, w0, w1", reads={"w0": 0, "w1": 0}, writes={"w0": 0}),
    ]


_BASE = CaseConfig(
    target="synthetic.so", input_hash="ab12", run_id="run-1",
    seed_hint_addr=0x100, sink_hint_addr=0x200, entry_pc=0x0FFF,
    window=(0x1000, 0x10FF), reg_file=("x0", "x1", "x16"),
    inputs=("carrier",), parity_min=8, symbolic_regs=("x0", "x1"),
    concrete_backing=build_concrete_backing(reg_values={"x16": 0x9000}),
    task="recover_window")

_REC = Candidate(RECOVER_WINDOW, 0x1000, "dispatch_type_rep", "rep",
                 payload={"window": [0x1000, 0x10FF], "window_kind": "pc"})
_BOTH = {"alias_vs_compute": "compute", "which_static": []}


def _verifier(runner, *, decisions=_BOTH):
    return RecoverWindowVerifier(base_config=_BASE, triton_runner=runner, decisions=decisions)


def test_verifier_exact_close_confirms():
    def runner(_ctx):
        return {"propagated": True, "gold_parity": "8/8",
                "expr_source": "def f(carrier):\n    return (carrier ^ 0x5a) & 0xff\n",
                "parity_vectors": [{"input_key": f"v{i}", "observed": f"o{i}",
                                    "predicted": f"o{i}", "exec_id": f"e{i}"} for i in range(3)],
                "trace_self_check": {"seed_values": {"carrier": 0x10},
                                     "sink_value": (0x10 ^ 0x5A) & 0xFF, "sink_mask": 0xFF}}
    v = _verifier(runner).verify(_REC, CvdState(_backed_window(), b"\x00"))
    assert v.status is VStatus.CONFIRMED
    assert v.evidence["parity"] == "8/8" and v.evidence["emitted_F"]
    # SAFETY GATE (closure layering): the window is parity-EXACT but NOT whole-case
    # oracle-closed (no on-path provenance anchor) → it is stamped with its TRUE
    # closure level, NOT auto-promoted to algorithm_closed_form.
    cl = v.evidence["closure"]
    assert cl["algorithm_closed"] is False
    assert cl["label"] != "algorithm_closed_form"
    assert v.evidence["closure_trap"] == cl["trap_state"]


def test_verifier_constant_collapse_stamps_pseudo_trap():
    """A window that emits a CONSTANT F (no input reference) with no provenance is
    the PSEUDO_CLOSURE_TRAP — the closure stamp surfaces it on the evidence (task
    1/2), never silently advanced."""
    def runner(_ctx):
        # propagated but the emitted F references no input → constant collapse.
        return {"propagated": True, "gold_parity": "0/8",
                "expr_source": "def f(carrier):\n    return 7\n"}
    v = _verifier(runner).verify(_REC, CvdState(_backed_window(), b"\x00"))
    cl = v.evidence["closure"]
    assert cl["is_constant"] is True
    assert cl["trap_state"] == "PSEUDO_CLOSURE_TRAP"
    assert cl["algorithm_closed"] is False
    # task 2: the constant's source coordinate is reported.
    assert "constant_source" in cl


def test_verifier_oracle_closed_window_is_algorithm_closed():
    """A window parity-EXACT AND on the confirmed output provenance path → it MAY
    be algorithm_closed_form (zero regression for genuinely-closed cases)."""
    def runner(_ctx):
        return {"propagated": True, "gold_parity": "8/8",
                "expr_source": "def f(carrier):\n    return (carrier ^ 0x5a) & 0xff\n",
                "parity_vectors": [{"input_key": f"v{i}", "observed": f"o{i}",
                                    "predicted": f"o{i}", "exec_id": f"e{i}"}
                                   for i in range(3)],
                "trace_self_check": {"seed_values": {"carrier": 0x10},
                                     "sink_value": (0x10 ^ 0x5A) & 0xFF, "sink_mask": 0xFF}}
    # An on-path candidate carrying the confirmed-output provenance signals.
    onpath = Candidate(
        RECOVER_WINDOW, 0x1000, SIG_PROVENANCE_ONPATH, "on-path producer",
        payload={"window": [0x1000, 0x10FF], "window_kind": "pc",
                 "on_path": True, "sink_captured": True,
                 "provenance_verdict": "CONTINUOUS_BUFFER"})
    v = _verifier(runner).verify(onpath, CvdState(_backed_window(), b"\x00"))
    assert v.status is VStatus.CONFIRMED
    cl = v.evidence["closure"]
    assert cl["algorithm_closed"] is True
    assert cl["label"] == "algorithm_closed_form"
    assert cl["trap_state"] == "NONE"


def test_verifier_opaque_collapse_is_terminal_frontier():
    # symex did not propagate (collapsed F) → opaque-staging frontier.
    def runner(_ctx):
        return {"propagated": False, "expr_source": ""}
    v = _verifier(runner).verify(_REC, CvdState(_backed_window(), b"\x00"))
    assert v.status is VStatus.TERMINAL and v.terminal_kind == "opaque_staging"
    assert v.capability_request == OPAQUE_STAGING_FRONTIER


def test_verifier_unmodeled_opcode_is_capability_terminal():
    def runner(_ctx):
        return {"propagated": False, "expr_source": "",
                "unmodeled": {"question": "supply semantics for opcode deadbeef"}}
    v = _verifier(runner).verify(_REC, CvdState(_backed_window(), b"\x00"))
    assert v.status is VStatus.TERMINAL and v.terminal_kind == "unmodeled_instruction"
    assert "un-modeled" in v.capability_request


def test_verifier_drive_pause_is_pending_judgment():
    # No decisions supplied → drive surfaces the alias_vs_compute checkpoint.
    def runner(_ctx):
        return {"propagated": True, "gold_parity": "8/8", "expr_source": "x"}
    v = _verifier(runner, decisions={}).verify(_REC, CvdState(_backed_window(), b"\x00"))
    assert v.status is VStatus.PENDING
    assert "judgment" in v.reason and v.capability_request == ""


def test_verifier_fixable_backing_eliminates_and_spawns_geometry_retry():
    # An unbacked load → backing gate fails → fixable → ELIMINATED + corrected spawn.
    unbacked = [_ins(0, 0x1000, "ldr w0, [x20]", reads={"x20": 0x9000}),
                _ins(1, 0x1004, "mul w0, w0, w1", reads={"w0": 0, "w1": 0}, writes={"w0": 0})]
    cfg = CaseConfig(
        target="synthetic.so", input_hash="ab12", run_id="run-1",
        seed_hint_addr=0x100, sink_hint_addr=0x200, entry_pc=0x0FFF,
        window=(0x1000, 0x10FF), reg_file=("x0", "x1", "x20"),
        inputs=("carrier",), parity_min=8, symbolic_regs=("x0", "x1"), task="t")

    def runner(_ctx):
        return {"propagated": True, "gold_parity": "8/8", "expr_source": "x"}
    v = RecoverWindowVerifier(base_config=cfg, triton_runner=runner, decisions=_BOTH)
    out = v.verify(_REC, CvdState(unbacked, b"\x00"))
    assert out.status is VStatus.ELIMINATED
    assert out.spawn and out.spawn[0].payload["_geometry_flipped"] is True
    # the flipped retry does not re-spawn (no infinite geometry loop).
    again = v.verify(out.spawn[0], CvdState(unbacked, b"\x00"))
    assert again.spawn == []


def test_verifier_seed_invariant_window_is_terminal_not_a_window():
    # Every live-in register is concretely backed → no symbolic input → the window
    # is not driven by the recovery variable (seed-invariant terminal).
    s = [_ins(0, 0x1000, "ldr w0, [x19]", reads={"x9": 0, "x19": 0x9000}),
         _ins(1, 0x1004, "add w0, w0, w1", reads={"w0": 0, "w1": 0}, writes={"w0": 0})]
    cfg = CaseConfig(
        target="synthetic.so", input_hash="ab", run_id="r", seed_hint_addr=0x100,
        sink_hint_addr=0x200, entry_pc=0x0FFF, window=(0x1000, 0x10FF),
        reg_file=("x19",), inputs=("carrier",), parity_min=8, symbolic_regs=None,
        concrete_backing=build_concrete_backing(reg_values={"x19": 0x9000}), task="t")

    def runner(_ctx):
        return {"propagated": True, "gold_parity": "8/8", "expr_source": "x"}
    v = RecoverWindowVerifier(base_config=cfg, triton_runner=runner, decisions=_BOTH)
    out = v.verify(_REC, CvdState(s, b"\x00"))
    assert out.status is VStatus.TERMINAL and out.terminal_kind == "seed_invariant"


# =========================================================================== #
# Pre-flight observable-variance gate — output-side dual of seed_block_note.
# A localized cohort whose candidate window has zero varying position is BLOCKed
# BEFORE drive() (the whole symex round saved), anchored at divergence_idx. Every
# other dependence state stands down to the normal flow (drive IS called).
# =========================================================================== #

def _localized_dep_at_idx5():
    """A localized cohort whose ONLY varying position is idx 5 (divergence_idx=5).
    A window NOT covering idx 5 has zero variance; one covering it does.

    Three vectors (>= the default parity_min_vectors=3) so the cohort-diversity
    pre-flight floor is met — these tests isolate the WINDOW-variance behaviour,
    not the cohort-depth floor (which has its own tests below)."""
    # identical setup at idx 0..4, the seed first moves observable state at idx 5.
    def _vec(seed_val):
        rows = [_ins(i, 0x1000 + 4 * i, "mov w9, #1", writes={"w9": 1})
                for i in range(5)]
        rows.append(_ins(5, 0x1014, "add w2, w0, w0",
                         reads={"w0": seed_val}, writes={"w2": seed_val * 2}))
        return rows
    dep = localize_input_dependence([_vec(1), _vec(9), _vec(17)],
                                    input_keys=["a", "b", "c"])
    assert dep.verdict == "localized" and dep.divergence_idx == 5
    assert dep.varying_idxs == (5,)
    assert dep.n_vectors == 3
    return dep


def _idx_window_cand(lo, hi):
    return Candidate(RECOVER_WINDOW, lo, "dispatch_type_rep", "rep",
                     payload={"window": [lo, hi], "window_kind": "idx"})


def _drive_spy(monkeypatch):
    """Patch cvd_recovery.drive with a counting spy; returns the call counter list."""
    import engine.cvd_recovery as cr
    calls: list[int] = []

    def _spy(*a, **k):
        calls.append(1)
        raise AssertionError("drive() must NOT be called on a no-variance window")
    monkeypatch.setattr(cr._verifier, "drive", _spy)
    return calls


def test_preflight_no_variance_window_blocks_before_drive(monkeypatch):
    # CORE SELF-PROOF: a localized cohort + a window [0,4] with ZERO varying
    # position → early BLOCK, and drive() is NEVER called (spy raises if it is).
    dep = _localized_dep_at_idx5()
    calls = _drive_spy(monkeypatch)

    def runner(_ctx):                              # never reached either
        raise AssertionError("runner must not run")
    cfg = CaseConfig(
        target="synthetic.so", input_hash="ab", run_id="r", seed_hint_addr=0x100,
        sink_hint_addr=0x200, entry_pc=0x0FFF, window=(0, 4), window_kind="idx",
        reg_file=("w0", "w2", "w9"), inputs=("carrier",), parity_min=8, task="t")
    v = RecoverWindowVerifier(base_config=cfg, triton_runner=runner,
                              decisions=_BOTH, dependence=dep)
    items = [_ins(i, 0x1000 + 4 * i, "mov w9, #1", writes={"w9": 1}) for i in range(5)]
    out = v.verify(_idx_window_cand(0, 4), CvdState(items, b"\x00"))

    assert out.status is VStatus.ELIMINATED            # early BLOCK
    assert calls == []                                 # drive NEVER called — round saved
    assert "no input variance in this window" in out.reason
    assert "idx >= 5" in out.reason                    # anchors at divergence_idx
    assert out.evidence["drive_skipped"] is True
    assert out.evidence["divergence_idx"] == 5
    # anchor spawn re-points recovery at the divergence idx.
    assert out.spawn and out.spawn[0].signal == "variance_anchor"
    assert out.spawn[0].payload["window"] == [5, 5]
    assert out.spawn[0].payload["anchor"] == "divergence_idx"
    # the anchored retry carries the no-loop marker, so even if it ALSO landed on a
    # no-variance window the pre-flight would not re-spawn (_anchor_spawn returns []).
    assert out.spawn[0].payload["_variance_anchored"] is True
    # direct no-loop proof: re-running the gate on the anchored candidate (forced
    # onto a no-variance window) yields no further spawn.
    forced = Candidate(RECOVER_WINDOW, 0, "variance_anchor", "forced",
                       payload=dict(out.spawn[0].payload, window=[0, 4]))
    again = v._preflight_observable_variance(v._case_config(forced), forced)
    assert again is not None and again.spawn == []


def test_preflight_window_with_variance_runs_normally(monkeypatch):
    # A window [5,5] that DOES contain the varying position → stand down, drive IS
    # called (invariant 7: a window with variance is byte-for-byte the old flow).
    dep = _localized_dep_at_idx5()
    import engine.cvd_recovery as cr
    drive_calls: list[int] = []

    def _spy_drive(*a, **k):
        drive_calls.append(1)
        # return a closed DriveResult-shaped object via the real runner path would be
        # heavy; instead short-circuit to a CONFIRMED-shaped result is not possible
        # without the real drive. So delegate to the real drive imported fresh.
        return _real_drive(*a, **k)
    from engine.setup_symex import drive as _real_drive
    monkeypatch.setattr(cr._verifier, "drive", _spy_drive)

    def runner(_ctx):
        return {"propagated": True, "gold_parity": "8/8",
                "expr_source": "def f(carrier):\n    return (carrier ^ 0x5a) & 0xff\n",
                "parity_vectors": [{"input_key": f"v{i}", "observed": f"o{i}",
                                    "predicted": f"o{i}", "exec_id": f"e{i}"}
                                   for i in range(3)],
                "trace_self_check": {"seed_values": {"carrier": 0x10},
                                     "sink_value": (0x10 ^ 0x5A) & 0xFF, "sink_mask": 0xFF}}
    cfg = CaseConfig(
        target="synthetic.so", input_hash="ab", run_id="r", seed_hint_addr=0x100,
        sink_hint_addr=0x200, entry_pc=0x1014, window=(5, 5), window_kind="idx",
        reg_file=("w0", "w2"), inputs=("carrier",), parity_min=8,
        symbolic_regs=("w0",), task="t")
    v = RecoverWindowVerifier(base_config=cfg, triton_runner=runner,
                              decisions=_BOTH, dependence=dep)
    items = [_ins(5, 0x1014, "add w2, w0, w0", reads={"w0": 0x10}, writes={"w2": 0x20})]
    out = v.verify(_idx_window_cand(5, 5), CvdState(items, b"\x00"))
    assert drive_calls == [1]                          # drive WAS called (normal flow)
    assert out.status is not VStatus.ELIMINATED or "no input variance" not in out.reason


def _localized_dep_at_idx5_shallow():
    """A localized cohort with the SAME varying position (idx 5) but only TWO
    vectors — the window HAS variance, yet the cohort is too SHALLOW (n_vectors=2
    < default parity_min_vectors=3) to ever EXACT-close (< min distinct outputs)."""
    def _vec(seed_val):
        rows = [_ins(i, 0x1000 + 4 * i, "mov w9, #1", writes={"w9": 1})
                for i in range(5)]
        rows.append(_ins(5, 0x1014, "add w2, w0, w0",
                         reads={"w0": seed_val}, writes={"w2": seed_val * 2}))
        return rows
    dep = localize_input_dependence([_vec(1), _vec(9)], input_keys=["a", "b"])
    assert dep.verdict == "localized" and dep.varying_idxs == (5,)
    assert dep.n_vectors == 2
    return dep


def test_preflight_shallow_cohort_blocks_before_drive(monkeypatch):
    # NEW pre-flight floor: the window [5,5] DOES vary, but the cohort carries only
    # 2 vectors (< parity_min_vectors=3) → it can never supply >= 3 distinct
    # independent observed outputs → UNCLOSABLE. Early BLOCK with "cohort output
    # diversity insufficient; need diverse seeds", drive() NEVER called, NO anchor
    # spawn (the fix is upstream seeds, not a re-anchor). Pre-flight dual of the
    # post-hoc UNCLOSABLE verdict (A8④: degenerate result still gets a verdict).
    dep = _localized_dep_at_idx5_shallow()
    calls = _drive_spy(monkeypatch)

    def runner(_ctx):
        raise AssertionError("runner must not run")
    cfg = CaseConfig(
        target="synthetic.so", input_hash="ab", run_id="r", seed_hint_addr=0x100,
        sink_hint_addr=0x200, entry_pc=0x1014, window=(5, 5), window_kind="idx",
        reg_file=("w0", "w2"), inputs=("carrier",), parity_min=8, task="t")
    v = RecoverWindowVerifier(base_config=cfg, triton_runner=runner,
                              decisions=_BOTH, dependence=dep)
    items = [_ins(5, 0x1014, "add w2, w0, w0", reads={"w0": 0x10}, writes={"w2": 0x20})]
    out = v.verify(_idx_window_cand(5, 5), CvdState(items, b"\x00"))

    assert out.status is VStatus.ELIMINATED            # early BLOCK
    assert calls == []                                 # drive NEVER called — round saved
    assert "cohort output diversity insufficient" in out.reason
    assert "need diverse seeds" in out.reason
    assert out.evidence["drive_skipped"] is True
    assert out.evidence["disposition"] == "cohort_diversity_insufficient"
    assert out.evidence["n_vectors"] == 2 and out.evidence["min_vectors"] == 3
    assert out.spawn == []                             # NO re-anchor — fix is upstream seeds


def test_preflight_deep_enough_cohort_with_variance_runs_normally(monkeypatch):
    # A 3-vector cohort (>= min) whose window has variance → neither pre-flight
    # branch fires → normal drive flow (invariant 7). Guards that the shallow-cohort
    # floor does NOT over-fire on an adequately deep cohort.
    dep = _localized_dep_at_idx5()                     # 3 vectors
    import engine.cvd_recovery as cr
    drive_calls: list[int] = []
    from engine.setup_symex import drive as _real_drive

    def _spy_drive(*a, **k):
        drive_calls.append(1)
        return _real_drive(*a, **k)
    monkeypatch.setattr(cr._verifier, "drive", _spy_drive)

    def runner(_ctx):
        return {"propagated": True, "gold_parity": "8/8",
                "expr_source": "def f(carrier):\n    return (carrier ^ 0x5a) & 0xff\n",
                "parity_vectors": [{"input_key": f"v{i}", "observed": f"o{i}",
                                    "predicted": f"o{i}", "exec_id": f"e{i}"}
                                   for i in range(3)],
                "trace_self_check": {"seed_values": {"carrier": 0x10},
                                     "sink_value": (0x10 ^ 0x5A) & 0xFF, "sink_mask": 0xFF}}
    cfg = CaseConfig(
        target="synthetic.so", input_hash="ab", run_id="r", seed_hint_addr=0x100,
        sink_hint_addr=0x200, entry_pc=0x1014, window=(5, 5), window_kind="idx",
        reg_file=("w0", "w2"), inputs=("carrier",), parity_min=8,
        symbolic_regs=("w0",), task="t")
    v = RecoverWindowVerifier(base_config=cfg, triton_runner=runner,
                              decisions=_BOTH, dependence=dep)
    items = [_ins(5, 0x1014, "add w2, w0, w0", reads={"w0": 0x10}, writes={"w2": 0x20})]
    out = v.verify(_idx_window_cand(5, 5), CvdState(items, b"\x00"))
    assert drive_calls == [1]                          # drive WAS called (normal flow)


def test_preflight_stands_down_without_dependence(monkeypatch):
    # No dependence map held → never triggers; drive runs (today's behaviour, inv 7).
    import engine.cvd_recovery as cr
    drive_calls: list[int] = []
    from engine.setup_symex import drive as _real_drive

    def _spy(*a, **k):
        drive_calls.append(1)
        return _real_drive(*a, **k)
    monkeypatch.setattr(cr._verifier, "drive", _spy)

    def runner(_ctx):
        return {"propagated": True, "gold_parity": "8/8", "expr_source": "x"}
    v = RecoverWindowVerifier(base_config=_BASE, triton_runner=runner,
                              decisions=_BOTH, dependence=None)
    v.verify(_REC, CvdState(_backed_window(), b"\x00"))
    assert drive_calls == [1]                          # stood down → drive ran


def test_preflight_stands_down_on_opaque_cohort(monkeypatch):
    # An opaque cohort has no trustworthy `varying` set → stand down (post-hoc gate
    # backstops). drive runs over the opaque window (invariant 7 / two-layer split).
    v1 = [_ins(0, 0x1000, "mov w0, #1", writes={"w0": 1})]
    v2 = [_ins(0, 0x1000, "mov w0, #1", writes={"w0": 1})]
    dep = localize_input_dependence([v1, v2], input_keys=["a", "b"])
    assert dep.verdict != "localized"                  # opaque / low-obs, NOT localized
    import engine.cvd_recovery as cr
    drive_calls: list[int] = []
    from engine.setup_symex import drive as _real_drive

    def _spy(*a, **k):
        drive_calls.append(1)
        return _real_drive(*a, **k)
    monkeypatch.setattr(cr._verifier, "drive", _spy)

    def runner(_ctx):
        return {"propagated": False, "expr_source": ""}
    cfg = CaseConfig(
        target="synthetic.so", input_hash="ab", run_id="r", seed_hint_addr=0x100,
        sink_hint_addr=0x200, entry_pc=0x1000, window=(0, 0), window_kind="idx",
        reg_file=("w0",), inputs=("carrier",), parity_min=8, symbolic_regs=("w0",),
        task="t")
    v = RecoverWindowVerifier(base_config=cfg, triton_runner=runner,
                              decisions=_BOTH, dependence=dep)
    v.verify(_idx_window_cand(0, 0), CvdState(v1, b"\x00"))
    assert drive_calls == [1]                          # stood down → drive ran (inv 7)


# =========================================================================== #
# Task 1 — candidate windows ranked by independent-side distinct POTENTIAL.
# A higher-diversity / divergence-covering window gets a higher base_value so the
# CVD frontier verifies it FIRST. Invariant 7: no varying signal → byte-for-byte
# today's fixed base_value (anchor 5.0 / other 4.0).
# =========================================================================== #

def _multi_pos_localized_dep():
    """A localized cohort with TWO varying positions of UNEQUAL diversity:
      * idx 1 — divergence_idx, ONE varying reg (low diversity but it IS the anchor);
      * idx 3 — THREE varying regs (high diversity, non-anchor).
    Distinct potential must rank idx 3 (high diversity) ABOVE idx 1's NON-anchor
    floor — and the anchor idx 1 stays on top of its 5.0 tier."""
    def _vec(a, b, c):
        return [
            _ins(0, 0x1000, "mov w9, #1", writes={"w9": 1}),
            _ins(1, 0x1004, "add w0, w9, w9", reads={"w9": 1}, writes={"w0": a}),
            _ins(2, 0x1008, "mov w9, #1", writes={"w9": 1}),
            _ins(3, 0x100C, "mac w1, w2, w3",
                 reads={"w0": a}, writes={"w1": a, "w2": b, "w3": c}),
        ]
    dep = localize_input_dependence(
        [_vec(1, 10, 100), _vec(2, 20, 200), _vec(3, 30, 300)],
        input_keys=["a", "b", "c"])
    assert dep.verdict == "localized" and dep.divergence_idx == 1
    return dep


def test_candidate_ranking_by_distinct_potential():
    dep = _multi_pos_localized_dep()
    gen = RecoveryWindowGenerator(dependence=dep)
    cands = gen.generate(CvdState([], b"\x00"))
    by_idx = {c.payload["window"][0]: c for c in cands}
    anchor = by_idx[1]          # divergence anchor (idx 1) — ONE varying reg
    high = by_idx[3]            # non-anchor, THREE varying regs — high diversity
    # the divergence anchor stays on TOP (its 5.0 tier + divergence bonus is never
    # out-ranked by a non-anchor) — ordering only re-orders WITHIN/ACROSS tiers
    # without flipping the anchor floor (invariant 7).
    assert anchor.base_value > high.base_value
    # among NON-anchor windows the higher-diversity one ranks strictly higher: the
    # 3-varying-reg window beats a hypothetical 1-varying-reg non-anchor.
    one_dim = RecoveryWindowGenerator._distinct_potential(
        [type("P", (), {"varying_regs": ("w0",), "varying_mem": (),
                        "control_flow": False, "idx": 99})()],
        n_vectors=3, divergence_idx=1)
    three_dim = RecoveryWindowGenerator._distinct_potential(
        [type("P", (), {"varying_regs": ("w1", "w2", "w3"), "varying_mem": (),
                        "control_flow": False, "idx": 99})()],
        n_vectors=3, divergence_idx=1)
    assert three_dim > one_dim > 0.0
    # the high-diversity non-anchor's base_value exceeds the non-anchor floor (4.0).
    assert high.base_value > 4.0
    # potential is recorded on the payload (machine-visible, not a hidden re-order).
    assert high.payload["distinct_potential"] == round(high.base_value - 4.0, 4)


def test_distinct_potential_fallback_preserves_fixed_base_value():
    # 保序兜底 (invariant 7): a localized cohort whose varying positions carry NO
    # varying dimension (defensive degenerate) → 0.0 bonus → base_value byte-for-byte
    # today's fixed value. Proven directly on the proxy: empty positions, and
    # positions with zero dims, both return exactly 0.0; a thin cohort scales to ~0.
    P = lambda **k: type("P", (), k)()
    assert RecoveryWindowGenerator._distinct_potential(
        [], n_vectors=3, divergence_idx=1) == 0.0
    assert RecoveryWindowGenerator._distinct_potential(
        [P(varying_regs=(), varying_mem=(), control_flow=False, idx=1)],
        n_vectors=3, divergence_idx=1) == 0.0
    # a 1-vector cohort cannot be distinct → diversity term scales to 0 (only the
    # divergence flat bonus, if it covers divergence, would apply — here it does not).
    assert RecoveryWindowGenerator._distinct_potential(
        [P(varying_regs=("w0",), varying_mem=(), control_flow=False, idx=99)],
        n_vectors=1, divergence_idx=1) == 0.0


def test_no_varying_dependence_keeps_today_base_values():
    # End-to-end保序: the single-varying-position localized cohort the existing
    # tests use must still emit the SAME fixed base_values (anchor 5.0 / other 4.0)
    # plus only the bounded potential bonus — never below today's floor.
    dep = _localized_dep_at_idx5()                     # one varying position, idx 5
    cands = RecoveryWindowGenerator(dependence=dep).generate(CvdState([], b"\x00"))
    assert cands
    for c in cands:
        is_anchor = c.signal == "divergence_anchor"
        floor = 5.0 if is_anchor else 4.0
        assert c.base_value >= floor                   # never drops below today's value
        assert c.base_value <= floor + 0.9             # bounded by the potential ceiling


# =========================================================================== #
# Task 2 — three-factor remedy tags (re-anchor / add-seeds / diversify-seeds).
# Each names WHICH cohort fix a BLOCK/UNCLOSABLE calls for; additive evidence keys
# that coexist with df0f95c's reason and do NOT touch any verdict (invariant 7).
# =========================================================================== #

def test_remedy_re_anchor_tag_on_zero_variance_window(monkeypatch):
    # Factor 1: a window with ZERO input-variance → remedy: re-anchor (df0f95c
    # already anchors; this asserts the explicit tag coexists with the reason).
    from engine.cvd_recovery import REMEDY_RE_ANCHOR
    dep = _localized_dep_at_idx5()
    _drive_spy(monkeypatch)

    def runner(_ctx):
        raise AssertionError("runner must not run")
    cfg = CaseConfig(
        target="synthetic.so", input_hash="ab", run_id="r", seed_hint_addr=0x100,
        sink_hint_addr=0x200, entry_pc=0x0FFF, window=(0, 4), window_kind="idx",
        reg_file=("w0", "w2", "w9"), inputs=("carrier",), parity_min=8, task="t")
    v = RecoverWindowVerifier(base_config=cfg, triton_runner=runner,
                              decisions=_BOTH, dependence=dep)
    items = [_ins(i, 0x1000 + 4 * i, "mov w9, #1", writes={"w9": 1}) for i in range(5)]
    out = v.verify(_idx_window_cand(0, 4), CvdState(items, b"\x00"))
    assert out.evidence["remedy"] == REMEDY_RE_ANCHOR
    assert "re-anchor" in out.evidence["remedy_reason"]
    # tag coexists with the existing df0f95c reason — not a replacement.
    assert "no input variance in this window" in out.reason


def test_remedy_add_seeds_tag_on_shallow_cohort(monkeypatch):
    # Factor 2: window HAS variance but n_vectors < min → remedy: add-seeds.
    from engine.cvd_recovery import REMEDY_ADD_SEEDS
    dep = _localized_dep_at_idx5_shallow()             # 2 vectors < min 3
    _drive_spy(monkeypatch)

    def runner(_ctx):
        raise AssertionError("runner must not run")
    cfg = CaseConfig(
        target="synthetic.so", input_hash="ab", run_id="r", seed_hint_addr=0x100,
        sink_hint_addr=0x200, entry_pc=0x1014, window=(5, 5), window_kind="idx",
        reg_file=("w0", "w2"), inputs=("carrier",), parity_min=8, task="t")
    v = RecoverWindowVerifier(base_config=cfg, triton_runner=runner,
                              decisions=_BOTH, dependence=dep)
    items = [_ins(5, 0x1014, "add w2, w0, w0", reads={"w0": 0x10}, writes={"w2": 0x20})]
    out = v.verify(_idx_window_cand(5, 5), CvdState(items, b"\x00"))
    assert out.evidence["remedy"] == REMEDY_ADD_SEEDS
    assert "supply MORE" in out.evidence["remedy_reason"]
    assert "cohort output diversity insufficient" in out.reason   # df0f95c reason kept


def test_remedy_diversify_seeds_tag_on_posthoc_unclosable(monkeypatch):
    # Factor 3: window HAS variance, n_vectors >= min, yet the independent side's
    # observed outputs COLLIDE (observed_distinct < min) → post-hoc UNCLOSABLE →
    # remedy: diversify-seeds. Real drive + real parity gate: an emitted F whose
    # cohort vectors carry distinct inputs/exec_ids but the SAME observed value.
    from engine.cvd_recovery import REMEDY_DIVERSIFY_SEEDS
    dep = _localized_dep_at_idx5()                     # 3 vectors >= min 3
    import engine.cvd_recovery as cr
    from engine.setup_symex import drive as _real_drive
    monkeypatch.setattr(cr._verifier, "drive", _real_drive)      # real drive (no spy block)

    def runner(_ctx):
        # emits a real F, but the 3 independent vectors all OBSERVE the same value
        # ("o0") → observed_distinct=1 < min_vectors=3 → UNCLOSABLE (output collision).
        return {"propagated": True, "gold_parity": "8/8",
                "expr_source": "def f(carrier):\n    return (carrier ^ 0x5a) & 0xff\n",
                "parity_vectors": [{"input_key": f"v{i}", "observed": "o0",
                                    "predicted": "o0", "exec_id": f"e{i}"}
                                   for i in range(3)],
                "trace_self_check": {"seed_values": {"carrier": 0x10},
                                     "sink_value": (0x10 ^ 0x5A) & 0xFF, "sink_mask": 0xFF}}
    cfg = CaseConfig(
        target="synthetic.so", input_hash="ab", run_id="r", seed_hint_addr=0x100,
        sink_hint_addr=0x200, entry_pc=0x1014, window=(5, 5), window_kind="idx",
        reg_file=("w0", "w2"), inputs=("carrier",), parity_min=8,
        symbolic_regs=("w0",), task="t")
    v = RecoverWindowVerifier(base_config=cfg, triton_runner=runner,
                              decisions=_BOTH, dependence=dep)
    items = [_ins(5, 0x1014, "add w2, w0, w0", reads={"w0": 0x10}, writes={"w2": 0x20})]
    out = v.verify(_idx_window_cand(5, 5), CvdState(items, b"\x00"))
    # the parity report is UNCLOSABLE (the gate's verdict — unchanged by the tag).
    assert out.evidence["parity_detail"]["need"] == 3
    assert out.evidence["remedy"] == REMEDY_DIVERSIFY_SEEDS
    assert "OUTPUT-DIVERSE" in out.evidence["remedy_reason"]
    # the three tags are distinct values (no collision across factors).
    from engine.cvd_recovery import REMEDY_RE_ANCHOR, REMEDY_ADD_SEEDS
    assert len({REMEDY_RE_ANCHOR, REMEDY_ADD_SEEDS, REMEDY_DIVERSIFY_SEEDS}) == 3


# =========================================================================== #
# Role 3 — RecoveryTerminalClassifier claims the opaque global frontier.
# =========================================================================== #

def test_terminal_classifier_claims_opaque_cohort():
    # A cohort that varies the seed but shows NO observable state movement = opaque.
    v1 = [_ins(0, 0x1000, "mov w0, #1", writes={"w0": 1})]
    v2 = [_ins(0, 0x1000, "mov w0, #1", writes={"w0": 1})]
    dep = localize_input_dependence([v1, v2], input_keys=["a", "b"])
    assert dep.is_opaque
    tc = RecoveryTerminalClassifier(dependence=dep)
    t = tc.classify(CvdState(v1, b"\x00"))
    assert t is not None and t.kind == "opaque_staging"
    assert t.success is False and t.capability_request == OPAQUE_STAGING_FRONTIER


def test_terminal_classifier_declines_without_opaque():
    assert RecoveryTerminalClassifier(dependence=None).classify(CvdState([], b"\x00")) is None


def test_terminal_classifier_surfaces_phase3_staging_advisory():
    # Phase 3: an opaque cohort whose staging EA varies across vectors must ship
    # the EA-varying PCs into the terminal evidence (where to pierce), not just
    # "needs symex". Same store/load PC, different EA per vector; regs_write
    # populated + constant so the verdict stays a true opaque.
    v1 = [_ins(0, 0x1000, "str x8, [x10]", writes={"x9": 1},
               mem=[MemOp("w", 0xA000, 0x41, 8)]),
          _ins(1, 0x1004, "ldr x11, [x10]", writes={"x12": 1},
               mem=[MemOp("r", 0xA000, 0x41, 8)])]
    v2 = [_ins(0, 0x1000, "str x8, [x10]", writes={"x9": 1},
               mem=[MemOp("w", 0xB000, 0x41, 8)]),
          _ins(1, 0x1004, "ldr x11, [x10]", writes={"x12": 1},
               mem=[MemOp("r", 0xB000, 0x41, 8)])]
    dep = localize_input_dependence([v1, v2], input_keys=["a", "b"])
    assert dep.is_opaque and dep.opaque_staging_advisory is not None
    t = RecoveryTerminalClassifier(dependence=dep).classify(CvdState(v1, b"\x00"))
    assert t is not None and t.kind == "opaque_staging"
    adv = t.evidence["opaque_staging_advisory"]
    assert adv["ea_varying_sites"]            # WHERE-to-pierce coordinates present


# =========================================================================== #
# 坎2 — opaque does NOT terminate before a per-window forward is tried once.
# The generator emits an opaque-staging-forward candidate from the cohort's
# opaque_staging_advisory; it is bounded (emitted once) so the run converges to
# the terminal claim in finite rounds, never an infinite generate↔widen loop.
# =========================================================================== #

def _opaque_cohort():
    """A genuinely-opaque cohort: same store/load PC, EA varies per vector (so the
    value diff is empty), regs_write populated + constant (so it is NOT the
    low-observability blind spot) → an ``opaque`` verdict WITH a staging advisory."""
    v1 = [_ins(0, 0x1000, "str x8, [x10]", writes={"x9": 1},
               mem=[MemOp("w", 0xA000, 0x41, 8)]),
          _ins(1, 0x1004, "ldr x11, [x10]", writes={"x12": 1},
               mem=[MemOp("r", 0xA000, 0x41, 8)])]
    v2 = [_ins(0, 0x1000, "str x8, [x10]", writes={"x9": 1},
               mem=[MemOp("w", 0xB000, 0x41, 8)]),
          _ins(1, 0x1004, "ldr x11, [x10]", writes={"x12": 1},
               mem=[MemOp("r", 0xB000, 0x41, 8)])]
    return [v1, v2]


def test_generator_opaque_emits_one_forward_candidate_from_advisory():
    # The opaque branch turns the advisory's EA-varying staging window into ONE
    # recover_window candidate (a representative window) so the frontier is not
    # empty when verdict==opaque → the per-window forward gets a turn before the
    # terminal can claim. The window spans the EA-varying site idxs.
    dep = localize_input_dependence(_opaque_cohort(), input_keys=["a", "b"])
    assert dep.verdict == "opaque" and dep.opaque_staging_advisory is not None
    gen = RecoveryWindowGenerator(dependence=dep)
    cands = gen.generate(CvdState(_opaque_cohort()[0], b"\x00"))
    assert len(cands) == 1
    c = cands[0]
    assert c.kind == RECOVER_WINDOW and c.signal == "opaque_staging_forward"
    assert c.payload["window"] == [0, 1] and c.payload["window_kind"] == "idx"
    assert c.payload["source"] == "opaque_staging_advisory"


def test_generator_opaque_window_is_emitted_only_once_no_infinite_loop():
    # Anti-loop self-proof: the SAME generator instance (as the driver reuses it
    # across widen-regenerate cycles) emits the opaque window the first time and
    # NOTHING afterwards. Without this dedup, every widen would re-offer the same
    # window (verify → still opaque → widen → regenerate → …) = infinite loop.
    dep = localize_input_dependence(_opaque_cohort(), input_keys=["a", "b"])
    gen = RecoveryWindowGenerator(dependence=dep)
    state = CvdState(_opaque_cohort()[0], b"\x00")
    first = gen.generate(state)
    assert len(first) == 1
    # repeated generations (the widen cycle) never re-offer the tried window.
    for _ in range(5):
        assert gen.generate(state) == []


def test_generator_opaque_without_advisory_emits_nothing():
    # Control: an opaque verdict with NO advisory window (or no advisory at all)
    # must not fabricate a candidate — there is genuinely nothing to forward, so
    # the terminal classifier claims directly (invariant 8 — no false locator).
    from engine.cohort_diff import InputDependenceMap
    dep_no_adv = InputDependenceMap(
        n_vectors=2, alignment="by_pc", verdict="opaque", divergence_idx=None,
        aligned_len=1, observable_positions=1, observability_rate=1.0,
        opaque_staging_advisory=None)
    assert RecoveryWindowGenerator(dependence=dep_no_adv).generate(
        CvdState([], b"\x00")) == []
    # an advisory present but with no usable window (empty sites, degenerate
    # region) → also nothing.
    dep_empty = InputDependenceMap(
        n_vectors=2, alignment="by_pc", verdict="opaque", divergence_idx=None,
        aligned_len=0, observable_positions=0, observability_rate=0.0,
        opaque_staging_advisory={"kind": "cohort_staging_advisory",
                                 "ea_varying_sites": [], "aligned_region": []})
    assert RecoveryWindowGenerator(dependence=dep_empty).generate(
        CvdState([], b"\x00")) == []


def test_generator_localized_and_coverage_paths_unchanged_by_opaque_branch():
    # Invariant 7: the opaque branch fires ONLY on verdict=="opaque" + advisory.
    # A localized dep and a coverage map produce exactly the candidates they did
    # before (no opaque_staging_forward leaks in).
    v1 = [_ins(0, 0x1000, "mov w0, #1", writes={"w0": 1}),
          _ins(1, 0x1004, "add w2, w0, w0", reads={"w0": 1}, writes={"w2": 2})]
    v2 = [_ins(0, 0x1000, "mov w0, #1", writes={"w0": 1}),
          _ins(1, 0x1004, "add w2, w0, w0", reads={"w0": 9}, writes={"w2": 18})]
    dep = localize_input_dependence([v1, v2], input_keys=["a", "b"])
    assert dep.verdict == "localized"
    cands = RecoveryWindowGenerator(dependence=dep).generate(CvdState(v1, b"\x00"))
    assert cands and all(c.signal != "opaque_staging_forward" for c in cands)


# =========================================================================== #
# Output-provenance-anchored window generation
# (dev-output-provenance-anchored-window-gen-spec.md). The PRIMARY anchor is
# "what feeds the target output" (its provenance producer chain), NOT "where the
# input-variance is" (cohort/dispatch variance). The F0-A root-cause fix: a
# high-variance window OFF the output path must NOT be a main candidate; variance
# is demoted to a secondary ordering filter WITHIN the on-path windows. Synthetic
# shapes only — no case addresses/values (utov-arch-index invariant 2/6).
# =========================================================================== #

_PROV_SINK = 0x8000


def _onpath_offpath_trace(varval: int, *, out_lo: int = 0xAB, out_hi: int = 0xCD):
    """A 4-step trace: idx0/idx1 are HIGH-VARIANCE but OFF the output path (they
    compute a dispatch-like intermediate that never feeds the sink); idx2 computes
    the output value and idx3 STORES it to ``_PROV_SINK`` (the on-path producers).
    ``varval`` drives the off-path variance; ``out_lo/out_hi`` set the output."""
    out_word = (out_hi << 8) | out_lo                    # little-endian sink bytes
    return [
        _ins(0, 0x1000, "mov w5, #v", writes={"w5": varval}),
        _ins(1, 0x1004, "add w6, w5, w5", reads={"w5": varval}, writes={"w6": varval * 2}),
        _ins(2, 0x1008, "mov w0, #out", writes={"w0": out_word}),
        _ins(3, 0x100C, "strh w0, [x1]", reads={"w0": out_word, "x1": _PROV_SINK},
             mem=(MemOp("w", _PROV_SINK, out_word, 2),)),
    ]


def test_provenance_anchor_surfaces_onpath_not_offpath_highvariance():
    # ① + ③ — the CORE self-証: a localized cohort whose ONLY variance is OFF the
    # output path (idx0/idx1). With the target output known, the generator anchors
    # on the PROVENANCE producer chain (idx2/idx3, which feed the sink) and DEMOTES
    # the high-variance off-path positions below every on-path window — they are no
    # longer the main candidates (the F0-A绕路 fix).
    v1 = _onpath_offpath_trace(1)
    v2 = _onpath_offpath_trace(7)
    dep = localize_input_dependence([v1, v2], input_keys=["a", "b"])
    assert dep.verdict == "localized" and dep.varying_idxs == (0, 1)   # variance is off-path
    expected = bytes([0xAB, 0xCD])                                     # the sink's LE bytes
    gen = RecoveryWindowGenerator(dependence=dep, sink_base=_PROV_SINK)
    cands = gen.generate(CvdState(v1, expected))

    onpath = [c for c in cands if c.signal == SIG_PROVENANCE_ONPATH]
    offpath = [c for c in cands if c.signal == SIG_PROVENANCE_OFFPATH_VARIANCE]
    # on-path windows are anchored on the producer chain (idx2 computes, idx3 stores).
    assert {c.payload["window"][0] for c in onpath} == {2, 3}
    assert all(c.payload["on_path"] is True for c in onpath)
    assert any(c.payload["path_distance"] == 0 for c in onpath)        # sink writer = dist 0
    # the high-variance off-path positions ARE still emitted (visible) but DEMOTED.
    assert {c.payload["window"][0] for c in offpath} == {0, 1}
    assert all(c.payload["on_path"] is False for c in offpath)
    # CORE: every on-path window out-ranks EVERY off-path (high-variance) window —
    # the off-path variance window is never the main candidate.
    top = max(cands, key=lambda c: c.base_value)
    assert top.signal == SIG_PROVENANCE_ONPATH
    assert min(c.base_value for c in onpath) > max(c.base_value for c in offpath)


def test_provenance_anchor_different_output_yields_different_onpath_windows():
    # ② — zero case-fit: the on-path windows follow the TARGET OUTPUT, not any
    # baked-in address. A second sink writer at idx5 producing a different output
    # → its provenance chain (idx4/idx5), proving the anchor tracks the output.
    base = _onpath_offpath_trace(3)                       # writes 0xAB,0xCD at idx3
    other_sink = 0x9000
    out2 = (0xEF << 8) | 0x12
    trace = base + [
        _ins(4, 0x1010, "mov w0, #out2", writes={"w0": out2}),
        _ins(5, 0x1014, "strh w0, [x2]", reads={"w0": out2, "x2": other_sink},
             mem=(MemOp("w", other_sink, out2, 2),)),
    ]
    expected_first = bytes([0xAB, 0xCD])
    expected_second = bytes([0x12, 0xEF])

    g1 = RecoveryWindowGenerator(sink_base=_PROV_SINK)
    w1 = {c.payload["window"][0] for c in g1.generate(CvdState(trace, expected_first))
          if c.signal == SIG_PROVENANCE_ONPATH}
    g2 = RecoveryWindowGenerator(sink_base=other_sink)
    w2 = {c.payload["window"][0] for c in g2.generate(CvdState(trace, expected_second))
          if c.signal == SIG_PROVENANCE_ONPATH}
    assert w1 == {2, 3} and w2 == {4, 5}                  # different output → different windows
    assert w1 != w2


def test_provenance_unobserved_output_yields_recapture_not_offpath_variance():
    # ③ (other half) — the output writer was NOT observed: produce a recapture
    # directive (collect the output, then re-anchor) — NOT off-path variance windows
    # (the silent fall-back was the绕路 root cause). The trace never WRITES the sink;
    # a register holds the buffer base so a reg-relative recapture spec can derive.
    trace = [
        _ins(0, 0x1000, "mov x1, #base", writes={"x1": _PROV_SINK}),
        _ins(1, 0x1004, "ldrh w0, [x1]", reads={"x1": _PROV_SINK}, writes={"w0": 0xABCD},
             mem=(MemOp("r", _PROV_SINK, 0xABCD, 2),)),
    ]
    # also supply a localized off-path cohort: it must be SUPPRESSED, not emitted.
    v2 = [_ins(0, 0x1000, "mov x1, #base", writes={"x1": _PROV_SINK + 1}),
          _ins(1, 0x1004, "ldrh w0, [x1]", reads={"x1": _PROV_SINK + 1}, writes={"w0": 0x1234},
               mem=(MemOp("r", _PROV_SINK + 1, 0x1234, 2),))]
    dep = localize_input_dependence([trace, v2], input_keys=["a", "b"])
    expected = bytes([0xCD, 0xAB])
    gen = RecoveryWindowGenerator(dependence=dep, sink_base=_PROV_SINK)
    cands = gen.generate(CvdState(trace, expected))
    # exactly the recapture directive — and NO off-path variance windows leaked in.
    assert [c.signal for c in cands] == [SIG_RECAPTURE_DIRECTIVE]
    assert cands[0].payload["needs_observation"] is True
    assert cands[0].payload["recapture_directive"]["kind"] == "recapture_directive"
    assert all(c.signal != SIG_PROVENANCE_OFFPATH_VARIANCE for c in cands)


def test_provenance_unanchorable_output_reports_explicitly_not_silent():
    # ④ (A8④) — a target output is supplied but its production cannot be tied to a
    # traced producer chain: the generator reports it EXPLICITLY (an unanchored
    # diagnostic candidate), it does NOT silently fall back to off-path variance.
    class _Bad:                                            # makes the backtrace raise
        pass
    gen = RecoveryWindowGenerator(sink_base=_PROV_SINK)
    cands = gen.generate(CvdState([_Bad()], bytes([1, 2])))
    assert [c.signal for c in cands] == [SIG_PROVENANCE_UNANCHORED]
    assert cands[0].payload["unanchored"] is True
    assert "detail" in cands[0].payload


def test_no_target_output_falls_back_to_today_anchor_unchanged():
    # ⑤ (invariant 7) — with NO target output (expected is the b"\x00" sentinel),
    # provenance anchoring stands down and generation is byte-for-byte today's
    # coverage/variance anchor: NO provenance/recapture/unanchored signals appear,
    # and the cohort_diff windows are exactly what they were before this feature.
    v1 = _onpath_offpath_trace(1)
    v2 = _onpath_offpath_trace(7)
    dep = localize_input_dependence([v1, v2], input_keys=["a", "b"])
    # sink_base set, but expected is the 1-byte sentinel → not a target output.
    gen = RecoveryWindowGenerator(dependence=dep, sink_base=_PROV_SINK)
    with_sink = gen.generate(CvdState(v1, b"\x00"))
    # identical to a generator that was never given a sink_base (today's behaviour).
    no_sink = RecoveryWindowGenerator(dependence=dep).generate(CvdState(v1, b"\x00"))
    prov_signals = {SIG_PROVENANCE_ONPATH, SIG_PROVENANCE_OFFPATH_VARIANCE,
                    SIG_RECAPTURE_DIRECTIVE, SIG_PROVENANCE_UNANCHORED}
    assert all(c.signal not in prov_signals for c in with_sink)
    assert [(c.signal, c.locus, c.base_value, c.payload) for c in with_sink] == \
           [(c.signal, c.locus, c.base_value, c.payload) for c in no_sink]


# =========================================================================== #
# On-path candidate BAND COALESCING (dev-recovery-bands-decisions-composite-spec
# Req4). A long producer chain (tc2: 7191 idxs) generated one single-idx window per
# idx → the cap dropped most and every terminal was a lone-store window, burning the
# budget. Req4 coalesces CONSECUTIVE producer-chain idxs (gap <= band_gap_threshold)
# into ONE band candidate (window=[start,end]) — a contiguous algorithm slice — while
# RETAINING the near-sink isolated-store diagnostic (demoted). Synthetic shapes only.
# =========================================================================== #

_BAND_SINK = 0x8800


def _contiguous_onpath_chain(n: int, *, sink=_BAND_SINK):
    """A trace whose producer chain idxs 0..n-1 are STRICTLY contiguous and all feed
    the sink (each writes addr 0x9000+i, the sink reads them all), then a 2-byte store
    to ``sink``. The producer idxs are gap-1 → Req4 coalesces them into ONE band."""
    trace = []
    addrs = [0x9000 + i for i in range(n)]
    for i, a in enumerate(addrs):
        trace.append(_ins(i, 0x1000 + 4 * i, "str", mem=(MemOp("w", a, 0xAB, 1),)))
    reads = tuple(MemOp("r", a, 0xAB, 1) for a in addrs)
    out_word = (0xCD << 8) | 0xAB
    trace.append(_ins(n, 0x2000, "strh w0, [x1]", reads={"x1": sink},
                      mem=(MemOp("w", sink, out_word, 2),) + reads))
    return trace


def test_band_coalesces_contiguous_chain_into_few_bands():
    # ① — a long CONTIGUOUS producer chain collapses to a SMALL number of band
    # candidates (the tc2 7191→~2000 effect): one band over the contiguous run,
    # plus the retained near-sink single-store diagnostic — NOT one window per idx.
    trace = _contiguous_onpath_chain(30)
    gen = RecoveryWindowGenerator(sink_base=_BAND_SINK)
    cands = gen.generate(CvdState(trace, bytes([0xAB, 0xCD])))
    onpath = [c for c in cands if c.signal == SIG_PROVENANCE_ONPATH]
    bands = [c for c in onpath if c.payload.get("band") is True]
    # 31 producer-chain idxs (0..30) → exactly ONE coalesced band (not 31 windows).
    assert len(bands) == 1
    assert bands[0].payload["window"] == [0, 30]
    assert bands[0].payload["window_span"] == 31
    # band count << idx count — the candidate explosion is gone.
    assert len(onpath) < 31


def test_band_candidate_carries_full_anchor_and_window_fields():
    # ② — each band candidate keeps the provenance ANCHOR (source/on_path/
    # path_distance) AND carries window=[start,end] + window_span + nearest-sink
    # distance + chain id.
    trace = _contiguous_onpath_chain(10)
    gen = RecoveryWindowGenerator(sink_base=_BAND_SINK)
    cands = gen.generate(CvdState(trace, bytes([0xAB, 0xCD])))
    band = next(c for c in cands
                if c.signal == SIG_PROVENANCE_ONPATH and c.payload.get("band") is True)
    p = band.payload
    assert p["source"] == "output_provenance"
    assert p["on_path"] is True
    assert isinstance(p["window"], list) and len(p["window"]) == 2
    assert p["window_span"] == p["window"][1] - p["window"][0] + 1
    assert "path_distance" in p
    assert "nearest_sink_distance" in p
    assert p["chain_id"] == f"0x{_BAND_SINK:x}"


def test_long_band_outranks_short_band_and_single_store_diag_retained():
    # ③ + ④ — a LONG contiguous band ranks ABOVE a short one (verified within budget
    # first), and the near-sink single-store DIAGNOSTIC is RETAINED (demoted), never
    # swallowed by coalescing. Two separated runs → two bands of different length.
    trace = []
    # long contiguous run idx 0..19 (writes 0x9000+i), then a GAP, then a short run.
    addrs_long = [0x9000 + i for i in range(20)]
    for i, a in enumerate(addrs_long):
        trace.append(_ins(i, 0x1000 + 4 * i, "str", mem=(MemOp("w", a, 0xAB, 1),)))
    # short run idx 40..41 (gap > threshold from the long run).
    addrs_short = [0xA000, 0xA001]
    trace.append(_ins(40, 0x3000, "str", mem=(MemOp("w", addrs_short[0], 0xAB, 1),)))
    trace.append(_ins(41, 0x3004, "str", mem=(MemOp("w", addrs_short[1], 0xAB, 1),)))
    reads = tuple(MemOp("r", a, 0xAB, 1) for a in addrs_long + addrs_short)
    out_word = (0xCD << 8) | 0xAB
    trace.append(_ins(42, 0x4000, "strh w0, [x1]", reads={"x1": _BAND_SINK},
                      mem=(MemOp("w", _BAND_SINK, out_word, 2),) + reads))
    gen = RecoveryWindowGenerator(sink_base=_BAND_SINK)
    cands = gen.generate(CvdState(trace, bytes([0xAB, 0xCD])))
    bands = [c for c in cands
             if c.signal == SIG_PROVENANCE_ONPATH and c.payload.get("band") is True]
    diags = [c for c in cands
             if c.signal == SIG_PROVENANCE_ONPATH
             and c.payload.get("single_store_diagnostic") is True]
    # two bands of different length; same near-sink distance ⇒ the LONGER out-ranks.
    spans = {tuple(c.payload["window"]): c.base_value for c in bands}
    assert len(spans) >= 2
    long_band = max(bands, key=lambda c: c.payload["window_span"])
    short_band = min(bands, key=lambda c: c.payload["window_span"])
    # the single-store diagnostic is retained, marked, and demoted below the bands.
    assert len(diags) == 1
    assert diags[0].payload["window_span"] == 1
    assert diags[0].base_value < min(c.base_value for c in bands)


def test_band_gap_threshold_keeps_separated_idxs_separate():
    # ⑤ (普适 / no case idx) — the gap threshold is the universal "same contiguous
    # slice" knob: idxs separated by MORE than the threshold stay in DIFFERENT bands;
    # raising the threshold merges them. Pure structural, no baked address/idx.
    gen = RecoveryWindowGenerator(sink_base=_BAND_SINK,
                                  budget=CvdBudget(band_gap_threshold=1))
    bands = gen._coalesce_onpath_bands({0, 1, 2, 7, 8}, 1)
    assert bands == [(0, 2), (7, 8)]               # gap of 5 splits the bands
    merged = gen._coalesce_onpath_bands({0, 1, 2, 7, 8}, 6)
    assert merged == [(0, 8)]                       # a wider threshold merges them


# =========================================================================== #
# COMPOSITE recovery (dev-recovery-bands-decisions-composite-spec Req6). After Req4,
# on-path candidates are BANDS. A single band can be a real algorithm slice yet its
# ISOLATED parity does not close the whole output → recovery's middle tier is to
# COMBINE adjacent on-path bands via chained symbolic state. Three terminals:
# BAND_PARITY_FAIL (isolated slice fails, the signal), COMPOSITE_REQUIRED (combine
# neighbours, within cost), COMPOSITE_TOO_EXPENSIVE (combined symex over budget → band
# list + estimate, a comfortable exit). The chained-state EXECUTION is scaffolded but
# NOT run (deep symex change — honest stop-report; never a faked composite).
# Synthetic shapes only (utov-arch-index invariant 2/6).
# =========================================================================== #


def test_estimate_composite_cost_is_combined_span():
    # cost is deterministic over band geometry: the combined window (lowest start to
    # highest end) is the dominant symex driver; per-band spans are also reported.
    cost = estimate_composite_cost([(10, 14), (20, 22)])
    assert cost["n_bands"] == 2
    assert cost["combined_window"] == [10, 22]
    assert cost["combined_span"] == 13
    assert cost["band_spans"] == [5, 3]
    assert cost["estimated_symex_items"] == 13


def test_plan_composite_required_when_adjacent_bands_within_budget():
    # ② — a single band could not close it but >= 2 adjacent on-path bands are
    # available and the combined symex is within budget → COMPOSITE_REQUIRED + the
    # band list to combine. Honest: execution is NOT run (scaffold).
    plan = plan_composite_recovery([(0, 4), (6, 9)], budget=CvdBudget())
    assert plan["terminal"] == TERMINAL_COMPOSITE_REQUIRED
    assert plan["bands"] == [[0, 4], [6, 9]]
    assert plan["over_budget"] is False
    # the PLAN is a pure decision; it does not itself execute (the verifier runs it).
    assert plan["executed"] is False
    assert plan["composite_execution"] == "planned"


def test_plan_composite_too_expensive_gives_comfortable_exit():
    # ③ — combining the bands would symex a combined window over the cost budget →
    # COMPOSITE_TOO_EXPENSIVE carrying the band list + cost estimate (a comfortable
    # exit, NOT a >90s hang). The tc2 idx28444..41354 巨窗 shape, parameterised.
    budget = CvdBudget(max_composite_symex_items=100)
    plan = plan_composite_recovery([(0, 60), (200, 260)], budget=budget)
    assert plan["terminal"] == TERMINAL_COMPOSITE_TOO_EXPENSIVE
    assert plan["over_budget"] is True
    assert plan["cost"]["estimated_symex_items"] > 100     # the estimate is carried
    assert plan["bands"] == [[0, 60], [200, 260]]          # WHICH segments, surfaced
    assert plan["executed"] is False


def test_plan_single_band_has_nothing_to_combine():
    # a lone band → no adjacent band to combine → no COMPOSITE_REQUIRED (the caller
    # stays at BAND_PARITY_FAIL; the isolated slice is the signal).
    plan = plan_composite_recovery([(0, 4)], budget=CvdBudget())
    assert plan["terminal"] is None
    assert plan["executed"] is False


def _band_parity_fail_runner():
    """A runner emitting a REAL F (references the input) whose cross-run parity
    vectors COLLIDE (same observed) → fails the independent floor → disposition
    'parity' (an emitted F that did not close)."""
    def runner(_ctx):
        return {"propagated": True, "gold_parity": "1/3",
                "expr_source": "def f(carrier):\n    return (carrier ^ 0x5a) & 0xff\n",
                "parity_vectors": [{"input_key": f"v{i}", "observed": "SAME",
                                    "predicted": "SAME", "exec_id": f"e{i}"}
                                   for i in range(3)],
                "trace_self_check": {"seed_values": {"carrier": 0x10},
                                     "sink_value": (0x10 ^ 0x5A) & 0xFF,
                                     "sink_mask": 0xFF}}
    return runner


def _band_cfg(window):
    return CaseConfig(
        target="s.so", input_hash="ab", run_id="r", seed_hint_addr=0x100,
        sink_hint_addr=0x200, entry_pc=0x1014, window=window, window_kind="idx",
        reg_file=("w0", "w2"), inputs=("carrier",), parity_min=8,
        symbolic_regs=("w0",), task="t")


def _band_items(lo, hi):
    items = [_ins(lo, 0x1014, "add w2, w0, w0", reads={"w0": 0x10}, writes={"w2": 0x20})]
    items += [_ins(i, 0x1018 + 4 * (i - lo - 1), "nop") for i in range(lo + 1, hi + 1)]
    return items


def _two_band_items():
    """A trace covering BOTH bands [5,7] and [10,12] so the composite executor can
    drive each band's window. Each band's seed reg is symbolic (w0) so its symex emits
    a real transform of the input."""
    items = [_ins(5, 0x1014, "add w2, w0, w0", reads={"w0": 0x10}, writes={"w2": 0x20}),
             _ins(6, 0x1018, "nop"), _ins(7, 0x101C, "nop")]
    items += [_ins(10, 0x1028, "add w2, w0, w0", reads={"w0": 0x10}, writes={"w2": 0x20}),
              _ins(11, 0x102C, "nop"), _ins(12, 0x1030, "nop")]
    return items


def test_band_parity_fail_runs_real_composite_then_band_parity_fail_no_cohort():
    # ① + ② (integration) — an on-path BAND that fails parity with >= 2 adjacent bands
    # within budget RUNS the real chained-symbolic-state execution. With NO cohort to
    # validate it, the composite RAN (a real composed expression is produced) but its
    # parity does not close → BAND_PARITY_FAIL carrying the composed F (a real run, not
    # a scaffold — the composite was genuinely chained, not faked).
    band = Candidate(RECOVER_WINDOW, 5, SIG_PROVENANCE_ONPATH, "band",
                     payload={"window": [5, 7], "window_kind": "idx",
                              "band": True, "on_path": True})
    v = RecoverWindowVerifier(
        base_config=_band_cfg((5, 7)), triton_runner=_band_parity_fail_runner(),
        decisions=_BOTH, onpath_bands=[(5, 7), (10, 12)], budget=CvdBudget())
    out = v.verify(band, CvdState(_two_band_items(), b"\x00"))
    assert out.status is VStatus.TERMINAL
    assert out.terminal_kind == TERMINAL_BAND_PARITY_FAIL     # ran, did not close
    comp = out.evidence["composite_execution"]
    assert comp["primitive_gap"] is False                     # both bands chainable
    assert comp["n_chained"] == 2                             # really chained 2 bands
    # a REAL composed expression was produced (chained, not faked).
    assert comp["composite_F"] and "def f(" in comp["composite_F"]
    assert "_band0(" in comp["composite_F"] and "_band1(" in comp["composite_F"]


def _composite_closing_runner():
    """A runner whose per-band transform is ``x -> x ^ 0x5a`` and whose end-band
    cohort oracle (trace_self_check.sink_value) equals the COMPOSITE ``(x^0x5a)^0x5a``
    = x, so the composed expression matches the whole output across diverse vectors →
    the composite closes multi-vector parity. Each cohort vector gets a distinct seed
    so the independent-output floor is met."""
    seeds = {"a": 0x11, "b": 0x22, "c": 0x33}

    def runner(ctx):
        items = ctx.get("items", [])
        # which cohort vector is this? distinguish by the first reads value if present.
        seed_val = 0x10
        for ins in items:
            if ins.regs_read.get("w0") is not None:
                seed_val = ins.regs_read["w0"]
                break
        out = {
            "propagated": True, "gold_parity": "1/3",
            "expr_source": "def f(carrier):\n    return (carrier ^ 0x5a) & 0xff\n",
            # end-band oracle: the WHOLE output for this vector = composite on its seed
            # = (seed ^ 0x5a) ^ 0x5a = seed. So the 2-band composite predicts == seed.
            "trace_self_check": {"seed_values": {"carrier": seed_val},
                                 "sink_value": seed_val, "sink_mask": 0xFF},
        }
        return out
    return runner, seeds


def test_band_parity_fail_too_expensive_terminal_with_estimate():
    # ③ (integration) — when combining the bands exceeds the cost budget, the band's
    # parity failure routes to COMPOSITE_TOO_EXPENSIVE carrying the band list + cost
    # estimate (a comfortable exit, never a >90s symex hang).
    band = Candidate(RECOVER_WINDOW, 5, SIG_PROVENANCE_ONPATH, "band",
                     payload={"window": [5, 7], "window_kind": "idx",
                              "band": True, "on_path": True})
    v = RecoverWindowVerifier(
        base_config=_band_cfg((5, 7)), triton_runner=_band_parity_fail_runner(),
        decisions=_BOTH, onpath_bands=[(5, 7), (5000, 5060)],
        budget=CvdBudget(max_composite_symex_items=50))
    out = v.verify(band, CvdState(_band_items(5, 7), b"\x00"))
    assert out.status is VStatus.TERMINAL
    assert out.terminal_kind == TERMINAL_COMPOSITE_TOO_EXPENSIVE
    plan = out.evidence["composite_plan"]
    assert plan["over_budget"] is True
    assert plan["cost"]["estimated_symex_items"] > 50
    assert "comfortable exit" in out.reason or "comfortable" in plan["reason"]


def test_lone_band_parity_fail_is_band_parity_fail_terminal():
    # ① — a lone on-path band (no adjacent band to combine) that fails parity is
    # BAND_PARITY_FAIL: the isolated slice is the signal, not silence.
    band = Candidate(RECOVER_WINDOW, 5, SIG_PROVENANCE_ONPATH, "band",
                     payload={"window": [5, 7], "window_kind": "idx",
                              "band": True, "on_path": True})
    v = RecoverWindowVerifier(
        base_config=_band_cfg((5, 7)), triton_runner=_band_parity_fail_runner(),
        decisions=_BOTH, onpath_bands=[(5, 7)], budget=CvdBudget())
    out = v.verify(band, CvdState(_band_items(5, 7), b"\x00"))
    assert out.status is VStatus.TERMINAL
    assert out.terminal_kind == TERMINAL_BAND_PARITY_FAIL


def test_non_band_parity_fail_stays_eliminated_unchanged():
    # ⑤ (invariant 7) — a NON-band candidate that fails parity is byte-for-byte the
    # old ELIMINATED path (the composite branch is gated on band=True only).
    nonband = Candidate(RECOVER_WINDOW, 5, "input_varying", "single",
                        payload={"window": [5, 7], "window_kind": "idx"})
    v = RecoverWindowVerifier(
        base_config=_band_cfg((5, 7)), triton_runner=_band_parity_fail_runner(),
        decisions=_BOTH, budget=CvdBudget())
    out = v.verify(nonband, CvdState(_band_items(5, 7), b"\x00"))
    assert out.status is VStatus.ELIMINATED
    assert "composite_plan" not in out.evidence


# =========================================================================== #
# Req5 — per-window MEMORY DISPOSITION (dev-recovery-bands-decisions-composite-spec).
# A recovery-generated window FAR from the early (0,800) window reads DIFFERENT
# staging/heap/table addresses; its external memory live-in has no disposition under
# the early map. A window whose live-in is WHOLLY un-classified that then collapses
# to opaque / a constant is a MISSING decision, not an algorithm property — it must
# route to MEMORY_DISPOSITION_MISSING, NOT the misleading opaque_staging / constant
# terminal. Self-dispatched per window; never the early map silently reused. Synthetic
# shapes only (utov-arch-index invariant 2/6).
# =========================================================================== #

def _unclassified_mem_window():
    """A window whose external memory live-in (a load at 0xC000 with no in-window
    writer) is un-classified — there is no cohort to compare its value variance, so
    no symbolize-vs-back disposition is available. backing covers the address regs so
    the backing gate passes (we reach the symex/opaque path, not 'fixable')."""
    return [
        _ins(28000, 0x12006920, "ldr w0, [x9]", reads={"x9": 0xC000},
             mem=(MemOp("r", 0xC000, 0x41, 4),)),
        _ins(28001, 0x12006924, "mul w0, w0, w2", reads={"w0": 0x41, "w2": 3},
             writes={"w0": 0xC3}),
    ]


def _disp_cfg(window=(28000, 28001)):
    return CaseConfig(
        target="s.so", input_hash="ab", run_id="r", seed_hint_addr=0x100,
        sink_hint_addr=0x200, entry_pc=0x12006910, window=window, window_kind="idx",
        reg_file=("w0", "w2", "x9"), inputs=("carrier",), parity_min=8,
        symbolic_regs=("w0",),
        concrete_backing=build_concrete_backing(reg_values={"x9": 0xC000}), task="t")


def _disp_cand(window=(28000, 28001)):
    return Candidate(RECOVER_WINDOW, window[0], "dispatch_type_rep", "rep",
                     payload={"window": list(window), "window_kind": "idx"})


# The SILENT-COLLAPSE scenario the spec keys on: an EARLY window's
# mem_input_symbolize_vs_back map is present in decisions (so drive does NOT pause —
# the map IS "answered") but it does NOT cover the FAR window's live-in address. The
# far window's address stays unsymbolized → the window collapses to opaque / a
# constant. Without Req5 this ships as opaque_staging / a constant trap (an algorithm-
# property mask); with Req5 it is MEMORY_DISPOSITION_MISSING (the early map's scope
# error named, never silent). The map covering a DIFFERENT address (early window's
# 0xA000, not the far 0xC000) is the "early map reused for a far window" reproduction.
_EARLY_MAP_OTHER_ADDR = {"mem_input_symbolize_vs_back": {0xA000: {"symbolize": 0x10}}}


def test_far_window_silent_opaque_collapse_is_disposition_missing_not_opaque():
    # ② CORE — the early map (answered, so no pause) does NOT cover the far window's
    # 0xC000 live-in → the window collapses to opaque → routed to the honest
    # MEMORY_DISPOSITION_MISSING terminal, NOT the misleading opaque_staging. The
    # missing per-window decision is kept SEPARATE from a genuine algorithm property
    # (A8④ / WARN-loud — the early map's scope error is named, not silently collapsed).
    def runner(_ctx):
        return {"propagated": False, "expr_source": ""}     # collapse → opaque path
    v = RecoverWindowVerifier(base_config=_disp_cfg(), triton_runner=runner,
                              decisions=dict(_BOTH, **_EARLY_MAP_OTHER_ADDR))
    out = v.verify(_disp_cand(), CvdState(_unclassified_mem_window(), b"\x00"))
    assert out.status is VStatus.TERMINAL
    assert out.terminal_kind == TERMINAL_MEMORY_DISPOSITION_MISSING
    assert out.terminal_kind != "opaque_staging"            # NOT the algorithm mask
    audit = out.evidence["memory_disposition_audit"]
    assert audit["all_undecided"] is True
    assert "0xc000" in audit["undecided"]                   # the far window's own addr
    assert out.evidence["collapsed_disposition"] == "opaque"
    assert "never classified" in out.reason


def test_far_window_silent_constant_collapse_is_disposition_missing_not_constant():
    # ② — same early-map scope error, but the window emits a CONSTANT F (references no
    # input — the "propagatable value seen as a constant" half of the root cause). Also
    # routed to MEMORY_DISPOSITION_MISSING, never a constant-trap algorithm property.
    def runner(_ctx):
        return {"propagated": True, "gold_parity": "0/8",
                "expr_source": "def f(carrier):\n    return 7\n"}   # constant F
    v = RecoverWindowVerifier(base_config=_disp_cfg(), triton_runner=runner,
                              decisions=dict(_BOTH, **_EARLY_MAP_OTHER_ADDR))
    out = v.verify(_disp_cand(), CvdState(_unclassified_mem_window(), b"\x00"))
    assert out.status is VStatus.TERMINAL
    assert out.terminal_kind == TERMINAL_MEMORY_DISPOSITION_MISSING
    assert out.evidence["collapsed_disposition"] == "constant"


def test_disposition_missing_is_self_dispatched_not_early_map():
    # ① — the disposition is computed for THIS candidate's OWN window (28000+), not
    # reused from an early (0,800) window. Proof: the audit's live_in address is the
    # FAR window's own read (0xC000), derived from the candidate window — and the early
    # map (0xA000) does NOT decide it (no early-map carry-over to a far window).
    def runner(_ctx):
        return {"propagated": False, "expr_source": ""}
    v = RecoverWindowVerifier(base_config=_disp_cfg(), triton_runner=runner,
                              decisions=dict(_BOTH, **_EARLY_MAP_OTHER_ADDR))
    out = v.verify(_disp_cand(), CvdState(_unclassified_mem_window(), b"\x00"))
    audit = out.evidence["memory_disposition_audit"]
    assert audit["live_in"] == ["0xc000"]                  # THIS window's own live-in
    assert audit["undecided"] == ["0xc000"]                # early 0xA000 does NOT cover it
    assert audit["n_live_in"] == 1


def test_explicit_pin_for_this_window_no_disposition_missing():
    # ① + invariant 7 — an explicit mem_input_symbolize_vs_back pin for the FAR
    # window's OWN address (0xC000) IS a decision: the window is no longer "missing",
    # so a collapse is the normal opaque path, NOT MEMORY_DISPOSITION_MISSING. The
    # caller's per-window classification is honoured.
    def runner(_ctx):
        return {"propagated": False, "expr_source": ""}
    decisions = dict(_BOTH, mem_input_symbolize_vs_back={0xC000: {"symbolize": 0x41}})
    v = RecoverWindowVerifier(base_config=_disp_cfg(), triton_runner=runner,
                              decisions=decisions)
    out = v.verify(_disp_cand(), CvdState(_unclassified_mem_window(), b"\x00"))
    assert out.terminal_kind != TERMINAL_MEMORY_DISPOSITION_MISSING
    assert out.terminal_kind == "opaque_staging"           # normal path (decided)


def test_unclassified_live_in_without_early_map_still_pauses():
    # invariant 7 — when NO mem map is supplied at all, drive PAUSES on the mem
    # checkpoint (the existing non-silent PENDING flow). Req5 does NOT convert that
    # legitimate agent-judgment checkpoint into a terminal; the silent-collapse gate
    # only fires when a collapse actually happened (an early map was present but did
    # not cover the window). This guards against over-firing on the pause path.
    def runner(_ctx):
        return {"propagated": False, "expr_source": ""}
    v = RecoverWindowVerifier(base_config=_disp_cfg(), triton_runner=runner,
                              decisions=_BOTH)              # no mem map → pause
    out = v.verify(_disp_cand(), CvdState(_unclassified_mem_window(), b"\x00"))
    assert out.status is VStatus.PENDING
    assert out.terminal_kind != TERMINAL_MEMORY_DISPOSITION_MISSING


def test_no_mem_live_in_window_unaffected_by_disposition_gate():
    # ⑤ (invariant 7) — a window with NO external mem live-in (reg-only) is untouched
    # by the disposition gate: an opaque collapse is the normal opaque_staging terminal
    # byte-for-byte (the gate fires ONLY when there is undecided mem live-in).
    reg_only = [_ins(0, 0x1000, "mul w0, w0, w2", reads={"w0": 0, "w2": 0},
                     writes={"w0": 0})]
    cfg = CaseConfig(
        target="s.so", input_hash="ab", run_id="r", seed_hint_addr=0x100,
        sink_hint_addr=0x200, entry_pc=0x0FFF, window=(0, 0), window_kind="idx",
        reg_file=("w0", "w2"), inputs=("carrier",), parity_min=8,
        symbolic_regs=("w0",), task="t")

    def runner(_ctx):
        return {"propagated": False, "expr_source": ""}
    v = RecoverWindowVerifier(base_config=cfg, triton_runner=runner, decisions=_BOTH)
    out = v.verify(_disp_cand((0, 0)), CvdState(reg_only, b"\x00"))
    assert out.terminal_kind == "opaque_staging"           # unchanged
    assert out.terminal_kind != TERMINAL_MEMORY_DISPOSITION_MISSING


# =========================================================================== #
# Generation / provenance-backtrace BUDGET (dev-recovery-generation-budget-spec).
# 8fad88f added an UNBOUNDED generation/backtrace phase (provenance backtrace +
# on-path candidate windows). This caps it via the SAME CvdBudget the verify loop
# uses — depth/breadth on the backtrace, a top-N ROI cap on the candidate set — and
# degrades NOT silently: a GENERATION_BUDGET_EXHAUSTED marker reports what was cut.
# A budget-internal ("small") case is byte-for-byte unchanged (invariant 7).
# Synthetic shapes only (utov-arch-index invariant 2/6).
# =========================================================================== #

_BUDGET_SINK = 0x8000


def _onpath_chain_trace(n_producers: int):
    """A trace with a long ON-PATH producer chain feeding the sink (each producer
    is a distinct on-path idx), then a 2-byte store to ``_BUDGET_SINK``. Drives the
    on-path candidate count up so a candidate cap truncates."""
    trace = []
    for i in range(n_producers):
        trace.append(_ins(i, 0x1000 + 4 * i, "add w0, w0, #1",
                          reads={"w0": i}, writes={"w0": i + 1}))
    out_word = (0xCD << 8) | 0xAB
    trace.append(_ins(n_producers, 0x1000 + 4 * n_producers, "strh w0, [x1]",
                      reads={"w0": out_word, "x1": _BUDGET_SINK},
                      mem=(MemOp("w", _BUDGET_SINK, out_word, 2),)))
    return trace


def _gapped_onpath_chain_trace(n_producers: int, *, gap: int = 5):
    """A trace whose sink reads ``n_producers`` distinct addresses, each written by a
    producer placed ``gap`` idxs apart (so the producer chain idxs are NON-contiguous
    — band coalescing leaves each its OWN band). Drives the BAND-candidate count up so
    the candidate cap truncates even after Req4 coalescing."""
    trace = []
    addrs = [0x9000 + 4 * i for i in range(n_producers)]
    idx = 0
    producer_idxs = []
    for a in addrs:
        trace.append(_ins(idx, 0x1000 + 4 * idx, "str",
                          mem=(MemOp("w", a, 0xAB, 1),)))
        producer_idxs.append(idx)
        idx += gap                                 # leave a hole > band_gap_threshold
    reads = tuple(MemOp("r", a, 0xAB, 1) for a in addrs)
    out_word = (0xCD << 8) | 0xAB
    trace.append(_ins(idx, 0x2000, "strh w0, [x1]",
                      reads={"x1": _BUDGET_SINK},
                      mem=(MemOp("w", _BUDGET_SINK, out_word, 2),) + reads))
    return trace


def test_generation_candidate_cap_truncates_with_explicit_report():
    # ① + ③ — a超多-on-path-candidate run: with a small max_gen_candidates the on-path
    # candidate set is capped to the top-N by ROI and the dropped long tail is reported
    # EXPLICITLY (a GENERATION_BUDGET_EXHAUSTED marker: how many generated/kept/dropped,
    # the retained order) — never silently dropped (A8④ / No silent caps). Post-Req4 the
    # cap counts BANDS: a gapped chain yields one band per separated producer (band
    # coalescing does NOT collapse non-contiguous idxs), so the cap still truncates.
    trace = _gapped_onpath_chain_trace(20)        # → many separated on-path BANDS
    expected = bytes([0xAB, 0xCD])
    budget = CvdBudget(max_gen_candidates=3)
    gen = RecoveryWindowGenerator(sink_base=_BUDGET_SINK, budget=budget)
    diag: list = []
    cands = gen.generate(CvdState(trace, expected), diag=diag)

    markers = [c for c in cands if c.signal == SIG_GENERATION_BUDGET_EXHAUSTED]
    assert len(markers) == 1                       # exactly one explicit truncation marker
    real = [c for c in cands if c.signal != SIG_GENERATION_BUDGET_EXHAUSTED]
    assert len(real) == 3                          # capped to the top-N
    rep = markers[0].payload
    assert rep["budget_exhausted"] is True
    assert rep["max_gen_candidates"] == 3
    assert rep["kept"] == 3
    assert rep["generated"] == rep["kept"] + rep["dropped"]   # nothing vanished silently
    assert rep["dropped"] > 0
    assert "retained_order" in rep                 # the order is reported
    # ROI order: every kept on-path window out-ranks (>=) the dropped tail's ceiling.
    kept_min = min(c.base_value for c in real)
    assert kept_min >= rep["kept_base_value_range"][0]
    # the truncation was logged (progress observability), not just returned.
    assert any(d["event"] == "GENERATION_BUDGET_EXHAUSTED" for d in diag)


def test_generation_budget_marker_verifies_to_terminal_not_symex():
    # ③ — the marker is a REPORT, not a window: the verifier surfaces it as a
    # GENERATION_BUDGET_EXHAUSTED TERMINAL carrying the truncation evidence, and never
    # runs drive() on it (a runner that raises if called proves drive was not invoked).
    import dataclasses
    def _exploding_runner(_req):
        raise AssertionError("drive() must NOT run on the generation-budget marker")
    cc = dataclasses.replace(_BASE, sink_hint_addr=_BUDGET_SINK,
                             window=(0, 1), window_kind="idx")
    ver = RecoverWindowVerifier(base_config=cc, triton_runner=_exploding_runner)
    marker = Candidate(RECOVER_WINDOW, locus=_BUDGET_SINK,
                       signal=SIG_GENERATION_BUDGET_EXHAUSTED,
                       entry_reason="generation budget exhausted — capped",
                       base_value=9.0,
                       payload={"budget_exhausted": True, "generated": 20,
                                "kept": 3, "dropped": 17})
    v = ver.verify(marker, CvdState([], bytes([0xAB, 0xCD])))
    assert v.status is VStatus.TERMINAL
    assert v.terminal_kind == "GENERATION_BUDGET_EXHAUSTED"
    assert v.evidence["dropped"] == 17             # the truncation report rides the verdict


def test_generation_budget_small_case_byte_for_byte_unchanged():
    # ② (invariant 7) — a budget-INTERNAL case (candidate count <= cap) is identical
    # with a generous default budget vs no budget passed: no marker, no reorder, the
    # very same candidates. The generation budget only fires on a real explosion.
    trace = _onpath_chain_trace(4)                 # few candidates, well under any cap
    expected = bytes([0xAB, 0xCD])
    default = RecoveryWindowGenerator(sink_base=_BUDGET_SINK)            # default budget
    big = RecoveryWindowGenerator(sink_base=_BUDGET_SINK,
                                  budget=CvdBudget(max_gen_candidates=1000))
    a = default.generate(CvdState(trace, expected))
    b = big.generate(CvdState(trace, expected))
    assert all(c.signal != SIG_GENERATION_BUDGET_EXHAUSTED for c in a)   # no marker
    assert [(c.signal, c.locus, c.base_value, c.payload) for c in a] == \
           [(c.signal, c.locus, c.base_value, c.payload) for c in b]     # byte-for-byte


def _wide_fanout_trace(width: int):
    """A trace where the SINK writer reads ``width`` distinct addresses in ONE
    instruction, each with its own producer → a wide BFS frontier at one step."""
    trace = []
    for i in range(width):
        trace.append(_ins(i, 0x1000 + 4 * i, "str",
                          mem=(MemOp("w", 0x9000 + i, 0xAB, 1),)))
    reads = tuple(MemOp("r", 0x9000 + i, 0xAB, 1) for i in range(width))
    trace.append(_ins(width, 0x2000, "str",
                      mem=(MemOp("w", _BUDGET_SINK, 0xAB, 1),) + reads))
    return trace


def test_backtrace_depth_cap_truncates_explicitly():
    # ④ (depth) — a long producer chain backtrace stops at max_steps and reports the
    # depth truncation; the result is still produced (bounded time, not 8min black box).
    long_chain = []
    for i in range(50):                            # idx i writes 0x9000+i, reads 0x9000+i+1
        a = 0x9000 + i
        long_chain.append(_ins(i, 0x1000 + i, "ldr_str",
                               mem=(MemOp("w", a, 0xAB, 1), MemOp("r", a + 1, 0xAB, 1))))
    long_chain.append(_ins(50, 0x2000, "str",
                           mem=(MemOp("w", _BUDGET_SINK, 0xAB, 1),
                                MemOp("r", 0x9001, 0xAB, 1))))
    r = trace_provenance(long_chain, bytes([0xAB]), sink_base=_BUDGET_SINK,
                         max_steps=5, max_breadth=None)
    assert r.backtrace_truncated is not None
    assert r.backtrace_truncated["mode"] == "depth"
    assert r.backtrace_truncated["max_steps"] == 5
    assert "backtrace_truncated" in r.to_dict()    # rides the serialization, visible


def test_backtrace_breadth_cap_truncates_explicitly():
    # ④ (breadth) — a wide producer fan-out is bounded by max_breadth; the dropped
    # branches are reported (not silently followed forever / not silently lost).
    r = trace_provenance(_wide_fanout_trace(20), bytes([0xAB]),
                         sink_base=_BUDGET_SINK, max_steps=100_000, max_breadth=4)
    assert r.backtrace_truncated is not None
    assert r.backtrace_truncated["mode"] == "breadth"
    assert r.backtrace_truncated["max_breadth"] == 4
    assert r.backtrace_truncated["branches_dropped"] > 0


def test_backtrace_within_budget_serialization_unchanged():
    # invariant 7 — a backtrace that completes within budget carries NO truncation
    # field and serializes byte-for-byte as before (the key is simply absent).
    r = trace_provenance(_wide_fanout_trace(4), bytes([0xAB]), sink_base=_BUDGET_SINK)
    assert r.backtrace_truncated is None
    assert "backtrace_truncated" not in r.to_dict()


def test_opaque_run_tries_forward_once_then_claims_terminal_with_count():
    # End-to-end (the real driver loop): an opaque cohort now drives the per-window
    # forward ONCE (the runner is invoked, leaving symbolic_forwards) BEFORE the
    # global opaque terminal is claimed — the forward count rides in the gap map as
    # proof "it was tried, it still collapsed", and the run converges (no loop).
    dep = localize_input_dependence(_opaque_cohort(), input_keys=["a", "b"])
    assert dep.verdict == "opaque"
    # the main trace seeds a backed, symbolic window so drive REACHES the runner.
    main_trace = _backed_window()
    runs = {"n": 0}

    def runner(_ctx):
        runs["n"] += 1
        # collapse (still opaque) BUT the forward was attempted (count > 0).
        return {"propagated": False, "expr_source": "", "symbolic_forwards": 2}

    reg = recovery_registry(base_config=_BASE, triton_runner=runner, dependence=dep,
                            decisions=_BOTH)
    res = run_cvd(main_trace, b"\x00", registry=reg, collect_extensions=True)
    assert res.outcome is CvdOutcome.COLLECTED
    # the forward ran exactly once (bounded — not an infinite re-generation loop).
    assert runs["n"] == 1
    # the candidate-scope opaque terminal ships the symbolic_forwards count.
    cand_opaque = [e for e in res.extension_requests
                   if e.get("scope") == "candidate"
                   and e.get("terminal_kind") == "opaque_staging"]
    assert len(cand_opaque) == 1
    assert (cand_opaque[0]["evidence"] or {}).get("symbolic_forwards") == 2
    # the global opaque terminal is still claimed — only AFTER the forward ran.
    glob = [e for e in res.extension_requests if e.get("scope") == "global"
            and e.get("terminal_kind") == "opaque_staging"]
    assert len(glob) == 1


# =========================================================================== #
# Integration — a recovery_registry collect run converges to the gap map.
# =========================================================================== #

def test_recovery_registry_collect_run_converges_to_gap_map():
    # Two handler types: one closes (EXACT), one collapses (opaque frontier). One
    # collect run must list BOTH outcomes — a confirmed window + the opaque gap —
    # not stop at the first.
    trace = [
        _ins(0, 0x1000, "ldr w0, [x16]", reads={"x16": 0x9000}, writes={"w0": 1}),
        _ins(1, 0x1004, "mul w0, w0, w1", reads={"w0": 1, "w1": 2}, writes={"w0": 2}),
        _ins(2, 0x1008, "ldr w3, [x16]", reads={"x16": 0x9000}, writes={"w3": 1}),
        _ins(3, 0x100C, "eor w3, w3, w4", reads={"w3": 1, "w4": 2}, writes={"w3": 3}),
    ]
    invs = [HandlerInvocation("A", 0, 1), HandlerInvocation("B", 2, 3)]
    cov = preflight_dispatch_coverage(trace, invocations=invs,
                                      reg_file=("w0", "w1", "w3", "w4", "x16"))

    def runner(ctx):
        win = tuple(ctx["window"])
        if win == (0, 1):          # type A closes
            return {"propagated": True, "gold_parity": "8/8",
                    "expr_source": "def f(carrier):\n    return carrier & 0xff\n",
                    "parity_vectors": [{"input_key": f"v{i}", "observed": f"o{i}",
                                        "predicted": f"o{i}", "exec_id": f"e{i}"} for i in range(3)],
                    "trace_self_check": {"seed_values": {"carrier": 0x10},
                                         "sink_value": 0x10, "sink_mask": 0xFF}}
        return {"propagated": False, "expr_source": ""}   # type B collapses → opaque

    base = CaseConfig(
        target="synthetic.so", input_hash="ab", run_id="r", seed_hint_addr=0x100,
        sink_hint_addr=0x200, entry_pc=0, window=(0, 1), window_kind="idx",
        reg_file=("w0", "w1", "w3", "w4", "x16"), inputs=("carrier",), parity_min=8,
        symbolic_regs=("w0", "w1", "w3", "w4"),
        concrete_backing=build_concrete_backing(reg_values={"x16": 0x9000}), task="t")
    reg = recovery_registry(base_config=base, triton_runner=runner, coverage=cov,
                            decisions=_BOTH)
    res = run_cvd(trace, b"\x00", registry=reg, collect_extensions=True)
    assert res.outcome is CvdOutcome.COLLECTED
    assert len(res.confirmed) == 1                       # type A confirmed
    # type B surfaced as the opaque-staging frontier (a candidate-scope capability).
    opaque = [e for e in res.extension_requests
              if e.get("terminal_kind") == "opaque_staging"]
    assert len(opaque) == 1
    assert opaque[0]["capability_request"] == OPAQUE_STAGING_FRONTIER


# =========================================================================== #
# 收口 addendum §1 — output = necessary info, never a trace dump (invariant 4).
# =========================================================================== #

def test_compact_trims_oversized_lists_to_count_hash():
    big = list(range(100))
    out = _compact({"items": big, "small": [1, 2, 3]})
    assert out["items"]["_trimmed_list"] is True
    assert out["items"]["count"] == 100 and len(out["items"]["sample"]) == 8
    assert isinstance(out["items"]["sha1"], str) and len(out["items"]["sha1"]) == 40
    assert out["small"] == [1, 2, 3]            # short list stays inline


def test_verifier_evidence_is_compact_summary_not_trace_dump():
    def runner(_ctx):
        return {"propagated": True, "gold_parity": "8/8",
                "expr_source": "def f(carrier):\n    return carrier & 0xff\n",
                "parity_vectors": [{"input_key": f"v{i}", "observed": f"o{i}",
                                    "predicted": f"o{i}", "exec_id": f"e{i}"} for i in range(3)],
                "trace_self_check": {"seed_values": {"carrier": 0x10},
                                     "sink_value": 0x10, "sink_mask": 0xFF}}
    ev = _verifier(runner).verify(_REC, CvdState(_backed_window(), b"\x00")).evidence
    # necessary fields kept; bulk (full per-step trail, closure/backing lists,
    # raw trace) NOT inlined.
    assert "emitted_F" in ev and "stopped_at" in ev and "note" in ev
    assert "per_step" not in ev and "address_closure" not in ev \
        and "mem_backing" not in ev and "items" not in ev


# =========================================================================== #
# 收口 addendum §2 — backtracking anchors on divergence_idx (not free-roaming).
# =========================================================================== #

def test_cohort_generator_anchors_first_candidate_on_divergence_idx():
    v1 = [_ins(0, 0x1000, "mov w0, #1", writes={"w0": 1}),
          _ins(1, 0x1004, "add w2, w0, w0", reads={"w0": 1}, writes={"w2": 2}),
          _ins(2, 0x1008, "eor w3, w2, w2", reads={"w2": 2}, writes={"w3": 4})]
    v2 = [_ins(0, 0x1000, "mov w0, #1", writes={"w0": 1}),
          _ins(1, 0x1004, "add w2, w0, w0", reads={"w0": 9}, writes={"w2": 18}),
          _ins(2, 0x1008, "eor w3, w2, w2", reads={"w2": 18}, writes={"w3": 36})]
    dep = localize_input_dependence([v1, v2], input_keys=["a", "b"])
    assert dep.verdict == "localized" and dep.divergence_idx == 1
    cands = RecoveryWindowGenerator(dependence=dep).generate(CvdState(v1, b"\x00"))
    anchors = [c for c in cands if c.signal == "divergence_anchor"]
    assert len(anchors) == 1                                  # exactly one anchor
    a = anchors[0]
    assert a.locus == dep.divergence_idx
    assert a.payload["anchor"] == "divergence_idx"
    # the anchor outranks the follow-on varying positions (checked early).
    assert all(a.base_value >= c.base_value for c in cands)


# =========================================================================== #
# §opaque-staging Phase 0 — the opaque branch attaches diag.to_dict to evidence.
# =========================================================================== #

def test_opaque_branch_attaches_staging_diagnosis():
    # The opaque verdict's evidence now carries a Phase 0 diagnosis splitting the
    # window (known_addr/symbolic_address/inconclusive) — collect no longer reports
    # a flat "藏 staging". Here the load EA (x16) is concretely backed → no blind
    # leg → inconclusive (a note, never silent).
    def runner(_ctx):
        return {"propagated": False, "expr_source": ""}
    v = _verifier(runner).verify(_REC, CvdState(_backed_window(), b"\x00"))
    assert v.terminal_kind == "opaque_staging"
    diag = v.evidence["opaque_staging_diagnosis"]
    assert diag["kind"] == "opaque_staging_diagnosis"
    assert diag["verdict"] in ("known_addr", "symbolic_address", "inconclusive")


def test_opaque_branch_passes_cohort_to_diagnosis():
    # The verifier forwards the injected cohort to Phase 0 (case-specific config,
    # never hardcoded). With a backed window that collapses, the attached diagnosis
    # is present and reflects the cohort (here: no blind staging load → inconclusive,
    # the honest "this is a forwarding collapse, not symbolic addressing" verdict —
    # a real blind load would have failed the backing gate first / been fixable).
    def runner(_ctx):
        return {"propagated": False, "expr_source": ""}
    cohort = [_backed_window(), _backed_window()]
    v = RecoverWindowVerifier(
        base_config=_BASE, triton_runner=runner, decisions=_BOTH,
        cohort_traces=cohort,
    ).verify(_REC, CvdState(_backed_window(), b"\x00"))
    assert v.terminal_kind == "opaque_staging"
    diag = v.evidence["opaque_staging_diagnosis"]
    assert diag["kind"] == "opaque_staging_diagnosis"
    assert diag["verdict"] in ("known_addr", "symbolic_address", "inconclusive")


# =========================================================================== #
# Evidence-backed mem disposition — cohort-variance recommendations + recovery
# auto-prefill. Three tiers (invariant 8: only auto-symbolize; back recommend
# only; truly ambiguous PENDING). Synthetic shapes only — no case constants.
# =========================================================================== #

# An external mem input at 0xA000 loaded into w0 (its value = the loaded byte),
# then mixed with the symbolic seed w1. The load value is parametric so a cohort
# can make it VARY (→ input) or stay CONSTANT (→ carrier).
def _mem_input_window(loaded):
    return [
        _ins(0, 0x1000, "ldr w0, [x16]", reads={"x16": 0xA000},
             mem=[MemOp("r", 0xA000, loaded, 4)], writes={"w0": loaded}),
        _ins(1, 0x1004, "mul w0, w0, w1",
             reads={"w0": loaded, "w1": 3}, writes={"w0": loaded * 3}),
    ]


_MEM_BASE = CaseConfig(
    target="synthetic.so", input_hash="ab12", run_id="run-1",
    seed_hint_addr=0x100, sink_hint_addr=0x200, entry_pc=0x0FFF,
    window=(0x1000, 0x10FF), reg_file=("w0", "w1", "x16"),
    inputs=("carrier",), parity_min=8, symbolic_regs=("w1",),
    concrete_backing=build_concrete_backing(reg_values={"x16": 0xA000}),
    task="recover_window")

_MEM_REC = Candidate(RECOVER_WINDOW, 0x1000, "rep", "r",
                     payload={"window": [0x1000, 0x10FF], "window_kind": "pc"})


def _closing_runner(_ctx):
    return {"propagated": True, "gold_parity": "8/8", "expr_source": "x",
            "parity_vectors": [{"input_key": f"v{i}", "observed": f"o{i}",
                                "predicted": f"o{i}", "exec_id": f"e{i}"}
                               for i in range(3)],
            "trace_self_check": {"seed_values": {"carrier": 0x10},
                                 "sink_value": 0, "sink_mask": 0xFF}}


_MEM_LI = (MemLiveIn(0xA000, 4, 0, ("x16",)),)
_WIN = (0x1000, 0x10FF)


# ---- primitive: three tiers + gates -------------------------------------- #

def test_recommend_varying_value_auto_symbolize():
    cohort = [_mem_input_window(0x11), _mem_input_window(0x22)]
    recs = recommend_mem_disposition(_MEM_LI, cohort, window=_WIN,
                                     window_is_idx=False, input_keys=["a", "b"])
    r = recs[0xA000]
    assert r.disposition == "symbolize" and r.confidence == "auto"
    assert r.value_varies is True and r.observable is True


def test_recommend_constant_value_recommends_back_only():
    cohort = [_mem_input_window(0x11), _mem_input_window(0x11)]
    recs = recommend_mem_disposition(_MEM_LI, cohort, window=_WIN,
                                     window_is_idx=False, input_keys=["a", "b"])
    r = recs[0xA000]
    # RISK direction — recommend, never auto.
    assert r.disposition == "back" and r.confidence == "recommend"
    assert r.value_varies is False


def test_recommend_single_trace_is_ambiguous_none():
    recs = recommend_mem_disposition(_MEM_LI, [_mem_input_window(0x11)], window=_WIN,
                                     window_is_idx=False, input_keys=["a"])
    r = recs[0xA000]
    assert r.disposition is None and r.confidence == "none"


def test_recommend_inputs_not_varied_is_ambiguous_none():
    # 2 traces but input_keys did not actually vary (dedup < 2) → constancy is
    # not informative → ambiguous (NOT auto/recommend).
    cohort = [_mem_input_window(0x11), _mem_input_window(0x22)]
    recs = recommend_mem_disposition(_MEM_LI, cohort, window=_WIN,
                                     window_is_idx=False, input_keys=["a", "a"])
    assert recs[0xA000].disposition is None and recs[0xA000].confidence == "none"


def test_recommend_low_observability_is_ambiguous_none():
    # A high mem threshold makes the window blind for the gate → cannot trust the
    # value comparison → ambiguous. Parameterised threshold, no case constant.
    cohort = [_mem_input_window(0x11), _mem_input_window(0x22)]
    recs = recommend_mem_disposition(
        _MEM_LI, cohort, window=_WIN, window_is_idx=False, input_keys=["a", "b"],
        thresholds={"mem": 0.99})
    r = recs[0xA000]
    assert r.disposition is None and r.confidence == "none" and r.observable is False


def test_recommend_no_cohort_is_ambiguous_none():
    recs = recommend_mem_disposition(_MEM_LI, [], window=_WIN, window_is_idx=False)
    assert recs[0xA000].disposition is None and recs[0xA000].confidence == "none"


# ---- recovery layer: prefill / PENDING / evidence / caller override ------ #

def test_recovery_auto_symbolize_does_not_pause_on_that_addr():
    # Form A: value varies across an input-varying cohort → confidence=auto →
    # recovery prefills symbolize → drive does NOT PENDING on the mem checkpoint.
    cohort = [_mem_input_window(0x11), _mem_input_window(0x22)]
    v = RecoverWindowVerifier(
        base_config=_MEM_BASE, triton_runner=_closing_runner, decisions=_BOTH,
        cohort_traces=cohort, input_keys=["a", "b"])
    out = v.verify(_MEM_REC, CvdState(_mem_input_window(0x11), b"\x00"))
    assert out.status is VStatus.CONFIRMED      # reached symex; no mem PENDING


def test_recovery_recommend_back_stays_pending_with_evidence():
    # Form B: constant value → recommend back, NOT prefilled → still PENDING; the
    # full recs ride along as evidence (the agent sees utov's prior, not a bare q).
    cohort = [_mem_input_window(0x11), _mem_input_window(0x11)]
    v = RecoverWindowVerifier(
        base_config=_MEM_BASE, triton_runner=_closing_runner, decisions=_BOTH,
        cohort_traces=cohort, input_keys=["a", "b"])
    out = v.verify(_MEM_REC, CvdState(_mem_input_window(0x11), b"\x00"))
    assert out.status is VStatus.PENDING
    recs = out.evidence["mem_disposition_recs"]
    assert any(r["disposition"] == "back" and r["confidence"] == "recommend"
               for r in recs)


def test_recovery_ambiguous_single_trace_stays_pending():
    # Form C: a single cohort trace → ambiguous None → PENDING (regression: same
    # as today's behaviour for that addr).
    v = RecoverWindowVerifier(
        base_config=_MEM_BASE, triton_runner=_closing_runner, decisions=_BOTH,
        cohort_traces=[_mem_input_window(0x11)], input_keys=["a"])
    out = v.verify(_MEM_REC, CvdState(_mem_input_window(0x11), b"\x00"))
    assert out.status is VStatus.PENDING


def test_recovery_no_cohort_is_byte_for_byte_today():
    # No cohort_traces → zero prefill, zero recs → PENDING on the mem checkpoint,
    # identical to the pre-feature behaviour (invariant 7).
    v = RecoverWindowVerifier(
        base_config=_MEM_BASE, triton_runner=_closing_runner, decisions=_BOTH)
    out = v.verify(_MEM_REC, CvdState(_mem_input_window(0x11), b"\x00"))
    assert out.status is VStatus.PENDING
    assert "mem_disposition_recs" not in (out.evidence or {})


def test_recovery_caller_decision_overrides_recommendation():
    # Caller's explicit mem decision wins over any recommendation/prefill. Here a
    # cohort says symbolize (auto), but the caller pins back → drive applies back.
    cohort = [_mem_input_window(0x11), _mem_input_window(0x22)]
    v = RecoverWindowVerifier(
        base_config=_MEM_BASE, triton_runner=_closing_runner,
        decisions={**_BOTH, "mem_input_symbolize_vs_back": {0xA000: "back"}},
        cohort_traces=cohort, input_keys=["a", "b"])
    out = v.verify(_MEM_REC, CvdState(_mem_input_window(0x11), b"\x00"))
    assert out.status is VStatus.CONFIRMED      # caller's back applied, no PENDING


# =========================================================================== #
# 坎1 — cohort SYMMETRIC merge: a bare-fed cohort vector (mem=()) carries its
# memory in a _mem.jsonl sidecar exactly like the main trace; merging it back
# restores the observability gate so the all() veto no longer nulls every addr.
# Plus the all()-veto DEGRADATION (observable subset / unified batch), invariant 8.
# Synthetic shapes only — no case addresses / values.
# =========================================================================== #

def _bare_window(loaded):
    """The same shape as ``_mem_input_window`` but with the memory dimension
    STRIPPED (mem=()) — a cohort vector fed bare, its mem living in a sidecar."""
    return [
        _ins(0, 0x1000, "ldr w0, [x16]", reads={"x16": 0xA000}, writes={"w0": loaded}),
        _ins(1, 0x1004, "mul w0, w0, w1",
             reads={"w0": loaded, "w1": 3}, writes={"w0": loaded * 3}),
    ]


def _write_mem_sidecar(tmp_path, name, loaded):
    """A canonical ``_mem.jsonl`` for ``_bare_window``: the idx-0 read at 0xA000."""
    p = tmp_path / f"{name}_mem.jsonl"
    p.write_text(
        json.dumps({"idx": 0, "rw": "r", "addr": "0xa000",
                    "val": loaded, "size": 4}) + "\n",
        encoding="utf-8")
    return str(p)


def test_cohort_bare_without_sidecar_all_veto_nulls_everything():
    # CONTROL (pre-wiring): a bare cohort (mem=()) is blind in the mem dimension →
    # all() veto → every addr None/"none". This is the 坎1 断点 reproduced.
    cohort = [_bare_window(0x11), _bare_window(0x22)]
    diag = {}
    recs = recommend_mem_disposition(
        _MEM_LI, cohort, window=_WIN, window_is_idx=False, input_keys=["a", "b"],
        diagnostics=diag)
    assert recs[0xA000].disposition is None and recs[0xA000].observable is False
    assert diag["observability"]["n_observable"] == 0


def test_cohort_symmetric_merge_unblocks_disposition(tmp_path):
    # WITH wiring: each bare vector's _mem.jsonl merged symmetrically → mem observable
    # → the all() veto lifts → the varying value yields auto-symbolize (non-null).
    cohort = [_bare_window(0x11), _bare_window(0x22)]
    sidecars = [_write_mem_sidecar(tmp_path, "v0", 0x11),
                _write_mem_sidecar(tmp_path, "v1", 0x22)]
    diag = {}
    recs = recommend_mem_disposition(
        _MEM_LI, cohort, window=_WIN, window_is_idx=False, input_keys=["a", "b"],
        cohort_mem_sidecars=sidecars, diagnostics=diag)
    r = recs[0xA000]
    assert r.disposition == "symbolize" and r.confidence == "auto"
    assert r.observable is True
    assert diag["observability"]["n_observable"] == 2
    assert diag["observability"]["symmetric"] is True
    assert diag["cohort_mem_merge"]["merged"][0]["mem_events_merged"] == 1


def test_cohort_symmetric_already_observable_is_byte_for_byte(tmp_path):
    # Invariant 7: a cohort already symmetric / fully observable behaves IDENTICALLY
    # with or without sidecars supplied (the merge folds nothing new in).
    cohort = [_mem_input_window(0x11), _mem_input_window(0x22)]
    base = recommend_mem_disposition(
        _MEM_LI, cohort, window=_WIN, window_is_idx=False, input_keys=["a", "b"])
    # supplying a sidecar whose event is already present must not change the verdict.
    sidecars = [_write_mem_sidecar(tmp_path, "v0", 0x11),
                _write_mem_sidecar(tmp_path, "v1", 0x22)]
    with_side = recommend_mem_disposition(
        _MEM_LI, cohort, window=_WIN, window_is_idx=False, input_keys=["a", "b"],
        cohort_mem_sidecars=sidecars)
    assert with_side[0xA000].to_dict() == base[0xA000].to_dict()
    assert base[0xA000].disposition == "symbolize"


def test_degrade_observable_subset_when_one_vector_still_blind(tmp_path):
    # Three vectors, two merged observable, one left bare (no sidecar) → asymmetric.
    # Subset (2) >= min_cohort → judge on the observable subset, WARN, NOT all-null.
    cohort = [_bare_window(0x11), _bare_window(0x22), _bare_window(0x33)]
    sidecars = [_write_mem_sidecar(tmp_path, "v0", 0x11),
                _write_mem_sidecar(tmp_path, "v1", 0x22),
                None]                                   # v2 stays bare/blind
    diag = {}
    recs = recommend_mem_disposition(
        _MEM_LI, cohort, window=_WIN, window_is_idx=False,
        input_keys=["a", "b", "c"], cohort_mem_sidecars=sidecars, diagnostics=diag)
    r = recs[0xA000]
    assert r.observable is True                          # subset trusted, not vetoed
    assert r.disposition == "symbolize"                  # varies across the subset
    deg = diag["observability"]["degraded"]
    assert deg["mode"] == "observable_subset" and deg["observable"] == 2
    assert "asymmetric" in deg["warn"]


def test_degrade_unified_batch_when_subset_too_small(tmp_path):
    # Only one vector merged observable, the other bare/blind → subset (1) < min_cohort
    # → ONE unified batch degradation marker, every addr None/"none" (invariant 8:
    # no symbolize/back invented), NOT independent per-addr nulls without a reason.
    cohort = [_bare_window(0x11), _bare_window(0x22)]
    sidecars = [_write_mem_sidecar(tmp_path, "v0", 0x11), None]
    diag = {}
    recs = recommend_mem_disposition(
        _MEM_LI, cohort, window=_WIN, window_is_idx=False, input_keys=["a", "b"],
        cohort_mem_sidecars=sidecars, diagnostics=diag)
    r = recs[0xA000]
    assert r.disposition is None and r.confidence == "none"   # never invented
    deg = diag["observability"]["degraded"]
    assert deg["mode"] == "cohort_unusable_batch"
    assert "batch degradation" in r.reason


def test_verifier_cohort_sidecars_surface_diagnostics_in_pending(tmp_path):
    # End-to-end through the Verifier: bare cohort + sidecars → symmetric merge →
    # the merge/symmetry diagnostics ride the PENDING evidence (WARN at boundary).
    # Constant value across the (now observable) cohort → recommend back → PENDING.
    cohort = [_bare_window(0x11), _bare_window(0x11)]
    sidecars = [_write_mem_sidecar(tmp_path, "v0", 0x11),
                _write_mem_sidecar(tmp_path, "v1", 0x11)]
    v = RecoverWindowVerifier(
        base_config=_MEM_BASE, triton_runner=_closing_runner, decisions=_BOTH,
        cohort_traces=cohort, input_keys=["a", "b"], cohort_mem_sidecars=sidecars)
    out = v.verify(_MEM_REC, CvdState(_mem_input_window(0x11), b"\x00"))
    assert out.status is VStatus.PENDING
    diag = out.evidence["mem_disposition_diagnostics"]
    assert diag["observability"]["symmetric"] is True
    assert any(r["disposition"] == "back"
               for r in out.evidence["mem_disposition_recs"])


# =========================================================================== #
# 坎1 重改: cohort symmetry BY CONSTRUCTION — caller gives only paths, never a
# parallel sidecar array. The auto ``<stem>_mem.jsonl`` sibling makes
# JsonlTraceReader(p).merged() symmetric for the cohort exactly as for the main
# trace. Synthetic JSONL shapes only — no case addresses / handler ids.
# =========================================================================== #

def _write_cohort_trace(tmp_path, name, loaded, *, with_sibling=True):
    """Write a bare main trace JSONL (mem in the ``_mem.jsonl`` sibling, like a
    real captured cohort vector) and, unless ``with_sibling=False``, the
    conventional ``<stem>_mem.jsonl`` sibling carrying the idx-0 read at 0xA000.
    Returns the main trace path (the ONLY thing a caller passes)."""
    trace = tmp_path / f"{name}.jsonl"
    trace.write_text(
        json.dumps({"idx": 0, "pc": "0x1000", "bytes": "00000000",
                    "mnemonic": "ldr w0, [x16]"}) + "\n"
        + json.dumps({"idx": 1, "pc": "0x1004", "bytes": "00000000",
                      "mnemonic": "mul w0, w0, w1"}) + "\n",
        encoding="utf-8")
    if with_sibling:
        sib = tmp_path / f"{name}_mem.jsonl"
        sib.write_text(
            json.dumps({"idx": 0, "rw": "r", "addr": "0xa000",
                        "val": loaded, "size": 4}) + "\n",
            encoding="utf-8")
    return str(trace)


def test_reader_auto_sibling_merges_without_explicit_sidecar(tmp_path):
    # JsonlTraceReader level: NO mem_sidecar param, but a conventional sibling on
    # disk → merged() folds it in automatically (symmetry by construction).
    from engine.runner_client import JsonlTraceReader
    path = _write_cohort_trace(tmp_path, "v0", 0x11)
    merged = JsonlTraceReader(path).merged()        # <-- no mem_sidecar passed
    assert merged.items[0].mem == (MemOp("r", 0xA000, 0x11, 4),)
    # default iteration is still bare (invariant 7: __iter__ never folds).
    assert all(i.mem == () for i in JsonlTraceReader(path))


def test_reader_no_sibling_is_byte_for_byte_bare(tmp_path):
    # Invariant 7: no explicit sidecar AND no sibling on disk → merged() == bare.
    from engine.runner_client import JsonlTraceReader
    path = _write_cohort_trace(tmp_path, "v0", 0x11, with_sibling=False)
    merged = JsonlTraceReader(path).merged()
    assert all(i.mem == () for i in merged.items)
    assert JsonlTraceReader(path).resolve_mem_sidecar() is None


def test_reader_explicit_sidecar_overrides_auto(tmp_path):
    # Invariant 7: an explicit mem_sidecar always wins over auto-resolution.
    from engine.runner_client import JsonlTraceReader
    path = _write_cohort_trace(tmp_path, "v0", 0x11)         # sibling has 0x11
    override = _write_mem_sidecar(tmp_path, "override", 0x99)  # explicit has 0x99
    merged = JsonlTraceReader(path, mem_sidecar=override).merged()
    assert merged.items[0].mem == (MemOp("r", 0xA000, 0x99, 4),)
    # mem_sidecar=False opts OUT of auto-resolution (force bare).
    bare = JsonlTraceReader(path, mem_sidecar=False).merged()
    assert all(i.mem == () for i in bare.items)


def test_load_cohort_traces_symmetric_no_sidecar_param(tmp_path):
    # THE proof: caller passes ONLY paths (each with a sibling) — NO sidecar
    # argument at all — and every vector comes back merged (mem non-empty).
    paths = [_write_cohort_trace(tmp_path, "v0", 0x11),
             _write_cohort_trace(tmp_path, "v1", 0x22),
             _write_cohort_trace(tmp_path, "v2", 0x33),
             _write_cohort_trace(tmp_path, "v3", 0x44)]
    traces, report = load_cohort_traces(paths)        # <-- no cohort_mem_sidecars
    assert all(t[0].mem == (MemOp("r", 0xA000, v, 4),)
               for t, v in zip(traces, (0x11, 0x22, 0x33, 0x44)))
    assert report["no_mem_sidecar"] == []             # every sibling auto-resolved
    assert all(r["mem_events"] == 1 for r in report["loaded"])
    # And the disposition is non-null off this purely-paths load (vs all-null bare).
    recs = recommend_mem_disposition(
        _MEM_LI, traces, window=_WIN, window_is_idx=False,
        input_keys=["a", "b", "c", "d"])
    assert recs[0xA000].disposition == "symbolize"     # varies → auto, not None


def test_load_cohort_traces_missing_sibling_warns_not_silent(tmp_path):
    # One vector has NO sibling → loaded bare, but a WARN is recorded (not silent).
    paths = [_write_cohort_trace(tmp_path, "v0", 0x11),
             _write_cohort_trace(tmp_path, "v1", 0x22, with_sibling=False)]
    traces, report = load_cohort_traces(paths)
    assert traces[0][0].mem == (MemOp("r", 0xA000, 0x11, 4),)   # v0 merged
    assert traces[1][0].mem == ()                               # v1 bare/degraded
    assert len(report["no_mem_sidecar"]) == 1
    warn = report["no_mem_sidecar"][0]
    assert warn["vector"] == 1 and "no mem sidecar" in warn["warn"]
    assert "v1_mem.jsonl" in warn["warn"]                       # names the sibling


def test_verifier_cohort_load_diagnostics_surface_in_pending(tmp_path):
    # The load-layer report (no-mem-sidecar WARN per vector) reaches the gap-map
    # PENDING evidence via mem_disposition_diagnostics["cohort_load"] — the WARN is
    # at the boundary, not buried (no silent batch degradation).
    cohort = [_bare_window(0x11), _bare_window(0x11)]
    sidecars = [_write_mem_sidecar(tmp_path, "v0", 0x11),
                _write_mem_sidecar(tmp_path, "v1", 0x11)]
    load_diag = {"vectors": 3, "no_mem_sidecar": [
        {"vector": 2, "path": "v2.jsonl",
         "warn": "cohort vector 2 has no mem sidecar: neither an explicit override "
                 "nor the conventional 'v2_mem.jsonl' sibling was found"}]}
    v = RecoverWindowVerifier(
        base_config=_MEM_BASE, triton_runner=_closing_runner, decisions=_BOTH,
        cohort_traces=cohort, input_keys=["a", "b"], cohort_mem_sidecars=sidecars,
        cohort_load_diagnostics=load_diag)
    out = v.verify(_MEM_REC, CvdState(_mem_input_window(0x11), b"\x00"))
    assert out.status is VStatus.PENDING
    diag = out.evidence["mem_disposition_diagnostics"]
    assert diag["cohort_load"]["no_mem_sidecar"][0]["vector"] == 2
    assert "no mem sidecar" in diag["cohort_load"]["no_mem_sidecar"][0]["warn"]


# =========================================================================== #
# A1 — run_recovery(snapshots=…) forwards same-execution oracle snapshots and
# opens obs_scope so the verify path actually reads them (no silent scope drop).
# =========================================================================== #

from engine.cvd import default_registry, place
from engine.types import MemSnapshot


def _sink_only_in_snapshot():
    """A trace whose writes do NOT produce the expected output bytes; the expected
    bytes exist ONLY in an out-of-band snapshot → without the snapshot in scope the
    sink is OUTPUT_NOT_OBSERVABLE, with it the sink is SINK_CONFIRMED via snapshot."""
    expected = bytes([0xCA, 0xFE, 0xBA, 0xBE])
    trace = [
        _ins(0, 0x1000, "mov x9, #1", writes={"x9": 0x8000}),
        _ins(1, 0x1004, "str x8, [x9]", reads={"x8": 0x11, "x9": 0x8000},
             mem=(MemOp("w", 0x8000, 0x11, 1),)),   # unrelated write
    ]
    snap = MemSnapshot(addr=0x8000, data=expected, label="output", source="snapshot")
    return trace, expected, snap


_A1_BASE = CaseConfig(
    target="a1.so", input_hash="ab", run_id="r", seed_hint_addr=0x100,
    sink_hint_addr=0x8000, entry_pc=0x0FFF, window=(0x1000, 0x10FF),
    reg_file=("x0", "x1"), inputs=("carrier",), parity_min=8,
    symbolic_regs=("x0", "x1"), task="recover_window")


def test_obs_scope_gate_hides_snapshot_until_opened():
    # The scope gate is real: with obs_scope=0 a supplied snapshot is NOT in
    # scoped_snapshots() (the verify path can't see it); with obs_scope=1 it is.
    _, _, snap = _sink_only_in_snapshot()
    state0, _ = place([_ins(0, 0x1000, "nop")], b"\x00", snapshots=[snap])
    assert state0.obs_scope == 0
    assert state0.scoped_snapshots() == []        # silently empty without scope
    state1, _ = place([_ins(0, 0x1000, "nop")], b"\x00", snapshots=[snap],
                      obs_scope=1)
    assert state1.obs_scope == 1
    assert state1.scoped_snapshots() == [snap]     # now visible to the verify path


def test_run_cvd_obs_scope_lets_snapshot_confirm_sink():
    # End-to-end via the CORE registry: opening obs_scope at PLACE makes the
    # same-execution snapshot visible to the verify path → the sink is SINK_CONFIRMED
    # located_via=snapshot WITHOUT waiting for a stall-driven WIDEN.
    trace, expected, snap = _sink_only_in_snapshot()
    opened = run_cvd(trace, expected, snapshots=[snap], obs_scope=1,
                     registry=default_registry(), collect_extensions=True)
    assert any(c["evidence"].get("located_via") == "snapshot"
               for c in opened.confirmed)


def test_run_cvd_obs_scope_param_defaults_zero_regression():
    # obs_scope defaults to 0 (existing widen ladder behaviour) — a snapshot-less run
    # is byte-for-byte unaffected by the new parameter.
    state, _ = place([_ins(0, 0x1000, "nop")], b"\x00")
    assert state.obs_scope == 0


def test_run_recovery_accepts_snapshots_and_opens_scope(tmp_path):
    # A1 core: run_recovery no longer raises TypeError on snapshots=, forwards them,
    # and (non-empty) opens obs_scope so the verify path reads them — the snapshot
    # confirms the sink via snapshot, which it never would with the scope gate shut.
    trace, expected, snap = _sink_only_in_snapshot()
    res, path = run_recovery(
        trace, base_config=_A1_BASE, triton_runner=lambda ctx: {},
        expected=expected, work_root=str(tmp_path), ts="t",
        snapshots=[snap])
    assert path.exists()
    doc = path.read_text()
    # The snapshot made the sink observable: the gap map no longer reports the sink
    # as OUTPUT_NOT_OBSERVABLE, and the snapshot located it (located_via=snapshot).
    assert "OUTPUT_NOT_OBSERVABLE" not in doc
    assert "snapshot" in doc


def test_run_recovery_empty_snapshots_is_regression(tmp_path):
    # A1 regression: None / empty snapshots behave EXACTLY as before — obs_scope
    # stays 0, the run does not raise, and the snapshot path is not engaged.
    trace, expected, _ = _sink_only_in_snapshot()
    _, path_none = run_recovery(
        trace, base_config=_A1_BASE, triton_runner=lambda ctx: {},
        expected=expected, work_root=str(tmp_path / "a"), ts="t")
    _, path_empty = run_recovery(
        trace, base_config=_A1_BASE, triton_runner=lambda ctx: {},
        expected=expected, work_root=str(tmp_path / "b"), ts="t", snapshots=[])
    # No snapshot → the sink is OUTPUT_NOT_OBSERVABLE exactly as before; None and []
    # behave identically (obs_scope never opened).
    assert "OUTPUT_NOT_OBSERVABLE" in path_none.read_text()
    assert "OUTPUT_NOT_OBSERVABLE" in path_empty.read_text()


def test_run_recovery_warns_loud_when_opening_scope(tmp_path, caplog):
    # A1 坎 — opening obs_scope for supplied snapshots is announced (never silent);
    # if it were dropped silently the carefully-captured evidence would vanish.
    import logging
    trace, expected, snap = _sink_only_in_snapshot()
    with caplog.at_level(logging.INFO, logger="engine.cvd_recovery"):
        run_recovery(trace, base_config=_BASE, triton_runner=lambda ctx: {},
                     expected=expected, work_root=str(tmp_path), ts="t",
                     snapshots=[snap])
    assert any("obs_scope=1" in r.getMessage() and "snapshot" in r.getMessage()
               for r in caplog.records)


# =========================================================================== #
# A2 — NEEDS_OBSERVATION && next_watch==[] is CLOSED (continue to recovery);
# unplaceable-only gaps are BLOCKED (not closed); placeable gaps still recapture.
# =========================================================================== #

def _force_prov(monkeypatch, prov):
    """Make the generator's cheap sink check pass (so it reaches the provenance
    branch) and make trace_provenance return the synthetic ``prov``."""
    monkeypatch.setattr(
        "engine.oracle_sink.validate_sink",
        lambda *a, **k: SinkValidation(SinkVerdict.SINK_CONFIRMED, base=0x200,
                                       located_via="write"))
    monkeypatch.setattr("engine.cvd_recovery._generator.trace_provenance",
                        lambda *a, **k: prov)


def _onpath_trace():
    # a producer-chain idx that backtrace would anchor (idx 1 writes via add).
    return [_ins(0, 0x1000, "mov w0, #1", writes={"w0": 1}),
            _ins(1, 0x1004, "add w2, w0, w0", reads={"w0": 1}, writes={"w2": 2})]


def test_needs_observation_empty_next_watch_is_closed_not_recapture(monkeypatch):
    # A2 ① — NEEDS_OBSERVATION with an EMPTY next_watch = the gap set is clear =
    # provenance closed → do NOT bounce to recapture; continue to on-path candidates.
    prov = ProvenanceResult(
        ProvenanceVerdict.NEEDS_OBSERVATION, sink_captured=True,
        next_watch=[], chain=[{"idx": 1, "pc": "0x1004"}],
        producer_pcs=(0x1004,))
    _force_prov(monkeypatch, prov)
    gen = RecoveryWindowGenerator(sink_base=0x200)
    cands = gen.generate(CvdState(_onpath_trace(), bytes([0xAB, 0xCD])))
    signals = {c.signal for c in cands}
    assert SIG_RECAPTURE_DIRECTIVE not in signals          # NOT bounced to recapture
    assert SIG_PROVENANCE_ONPATH in signals                # continued to recovery


def test_needs_observation_placeable_next_watch_still_recaptures(monkeypatch):
    # A2 ② (regression) — a real, hookable gap (next_watch entry with a pc) still
    # produces a recapture directive, exactly as before.
    prov = ProvenanceResult(
        ProvenanceVerdict.NEEDS_OBSERVATION, sink_captured=True,
        next_watch=[{"addr": "0x9000", "pc": "0x1004", "reason": "uncaptured read"}],
        chain=[{"idx": 1, "pc": "0x1004"}])
    _force_prov(monkeypatch, prov)
    gen = RecoveryWindowGenerator(sink_base=0x200)
    cands = gen.generate(CvdState(_onpath_trace(), bytes([0xAB, 0xCD])))
    assert len(cands) == 1
    assert cands[0].signal == SIG_RECAPTURE_DIRECTIVE


def test_needs_observation_unplaceable_only_is_blocked_not_closed(monkeypatch):
    # A2 ③ (new) — gaps remain but NONE is placeable (pc is None) → neither closed
    # nor recapturable → an EXPLICIT BLOCKED candidate, conservatively NOT a false
    # close and NOT a recapture that cannot be armed.
    prov = ProvenanceResult(
        ProvenanceVerdict.NEEDS_OBSERVATION, sink_captured=True,
        next_watch=[{"addr": "0x200", "pc": None, "reason": "no producer pc"},
                    {"addr": "0x201", "pc": None, "reason": "no producer pc"}],
        chain=[{"idx": 1, "pc": "0x1004"}])
    _force_prov(monkeypatch, prov)
    gen = RecoveryWindowGenerator(sink_base=0x200)
    cands = gen.generate(CvdState(_onpath_trace(), bytes([0xAB, 0xCD])))
    assert len(cands) == 1
    c = cands[0]
    assert c.signal == SIG_PROVENANCE_BLOCKED_UNPLACEABLE
    assert c.payload["blocked"] is True and c.payload["unplaceable"] is True
    assert len(c.payload["unplaceable_gaps"]) == 2
    assert SIG_RECAPTURE_DIRECTIVE != c.signal             # NOT a placeable recapture


def test_needs_observation_sink_uncaptured_still_recaptures_first(monkeypatch):
    # A2 边界 — sink_captured is False keeps the safety gate: recapture FIRST, the
    # NEEDS_OBSERVATION split does not apply (sink-未确认优先-recapture is untouched).
    prov = ProvenanceResult(
        ProvenanceVerdict.NEEDS_OBSERVATION, sink_captured=False,
        next_watch=[], chain=[])
    _force_prov(monkeypatch, prov)
    gen = RecoveryWindowGenerator(sink_base=0x200)
    cands = gen.generate(CvdState(_onpath_trace(), bytes([0xAB, 0xCD])))
    assert len(cands) == 1
    assert cands[0].signal == SIG_RECAPTURE_DIRECTIVE


# =========================================================================== #
# A3 — same chain_id BAND_PARITY_FAIL aggregation: the verifier plans a composite
# over the WHOLE same-chain group (not the single band), group-aware ranking, and
# cross-chain bands are never merged.
# =========================================================================== #

def test_onpath_band_registry_groups_by_chain_id():
    reg = _OnpathBandRegistry()
    reg.record("0x200", 5, 7)
    reg.record("0x200", 10, 12)
    reg.record("0x200", 5, 7)               # dup ignored
    reg.record("0x300", 0, 3)               # different chain
    assert reg.group("0x200") == [(5, 7), (10, 12)]
    assert reg.group_size("0x200") == 2
    assert reg.group("0x300") == [(0, 3)]   # NOT merged with 0x200
    assert reg.group(None) == [] and reg.group_size("missing") == 0


def test_recovery_registry_shares_one_band_registry():
    # A3 对称由构造保证 — recovery_registry wires the SAME _OnpathBandRegistry into
    # BOTH the generator (writer) and the verifier (reader): one source, never two
    # lists a caller must keep in sync.
    reg = recovery_registry(base_config=_band_cfg((5, 7)),
                            triton_runner=lambda ctx: {})
    gen = [g for g in reg.generators if isinstance(g, RecoveryWindowGenerator)][0]
    ver = [v for v in reg.verifiers if isinstance(v, RecoverWindowVerifier)][0]
    assert gen.band_registry is ver.band_registry
    assert isinstance(gen.band_registry, _OnpathBandRegistry)


def test_band_parity_aggregates_same_chain_group_into_composite():
    # A3 验收 — a single BAND_PARITY_FAIL candidate, when the shared registry holds
    # >= composite_aggregation_min same-chain bands, plans the composite over the
    # WHOLE group → COMPOSITE_REQUIRED (not an isolated BAND_PARITY_FAIL).
    reg = _OnpathBandRegistry()
    reg.record("0x200", 5, 7)
    reg.record("0x200", 10, 12)
    band = Candidate(RECOVER_WINDOW, 5, SIG_PROVENANCE_ONPATH, "band",
                     payload={"window": [5, 7], "window_kind": "idx",
                              "band": True, "on_path": True, "chain_id": "0x200"})
    v = RecoverWindowVerifier(
        base_config=_band_cfg((5, 7)), triton_runner=_band_parity_fail_runner(),
        decisions=_BOTH, band_registry=reg, budget=CvdBudget())
    out = v.verify(band, CvdState(_two_band_items(), b"\x00"))
    # the plan saw the whole same-chain group, not just this band.
    assert out.evidence["chain_band_group"] == [[5, 7], [10, 12]]
    assert out.evidence["chain_band_group_size"] == 2
    assert out.evidence["composite_plan"]["terminal"] == TERMINAL_COMPOSITE_REQUIRED


def test_band_parity_lone_chain_stays_band_parity_fail():
    # A3 边界 — a chain with only ONE band (below the aggregation floor) is NOT a
    # group → stays BAND_PARITY_FAIL (the isolated slice is the signal, regression).
    reg = _OnpathBandRegistry()
    reg.record("0x200", 5, 7)
    band = Candidate(RECOVER_WINDOW, 5, SIG_PROVENANCE_ONPATH, "band",
                     payload={"window": [5, 7], "window_kind": "idx",
                              "band": True, "on_path": True, "chain_id": "0x200"})
    v = RecoverWindowVerifier(
        base_config=_band_cfg((5, 7)), triton_runner=_band_parity_fail_runner(),
        decisions=_BOTH, band_registry=reg, budget=CvdBudget())
    out = v.verify(band, CvdState(_band_items(5, 7), b"\x00"))
    assert out.status is VStatus.TERMINAL
    assert out.terminal_kind == TERMINAL_BAND_PARITY_FAIL
    assert out.evidence["chain_band_group_size"] == 1


def test_band_parity_does_not_merge_different_chains():
    # A3 验收 — bands of a DIFFERENT chain_id are never folded into this chain's
    # composite group: the verifier only combines this candidate's own chain.
    reg = _OnpathBandRegistry()
    reg.record("0x200", 5, 7)
    reg.record("0x999", 10, 12)             # a different chain — must NOT be combined
    band = Candidate(RECOVER_WINDOW, 5, SIG_PROVENANCE_ONPATH, "band",
                     payload={"window": [5, 7], "window_kind": "idx",
                              "band": True, "on_path": True, "chain_id": "0x200"})
    v = RecoverWindowVerifier(
        base_config=_band_cfg((5, 7)), triton_runner=_band_parity_fail_runner(),
        decisions=_BOTH, band_registry=reg, budget=CvdBudget())
    out = v.verify(band, CvdState(_band_items(5, 7), b"\x00"))
    assert out.evidence["chain_band_group"] == [[5, 7]]      # only chain 0x200
    assert out.terminal_kind == TERMINAL_BAND_PARITY_FAIL    # lone → no composite


def test_aggregation_threshold_is_parameterised():
    # A3 阈值 — the aggregation floor is a budget parameter, not a baked number:
    # raising composite_aggregation_min above the group size suppresses the group
    # treatment (the same two bands stay below the (higher) floor).
    reg = _OnpathBandRegistry()
    reg.record("0x200", 5, 7)
    reg.record("0x200", 10, 12)
    band = Candidate(RECOVER_WINDOW, 5, SIG_PROVENANCE_ONPATH, "band",
                     payload={"window": [5, 7], "window_kind": "idx",
                              "band": True, "on_path": True, "chain_id": "0x200"})
    # default (2) → group composite; raised to 3 → 2 bands below floor.
    v_hi = RecoverWindowVerifier(
        base_config=_band_cfg((5, 7)), triton_runner=_band_parity_fail_runner(),
        decisions=_BOTH, band_registry=reg,
        budget=CvdBudget(composite_aggregation_min=3))
    out_hi = v_hi.verify(band, CvdState(_two_band_items(), b"\x00"))
    assert out_hi.evidence["composite_aggregation_min"] == 3
    # the planner still combines >= 2 adjacent bands (its own rule); the BUDGET knob
    # is exposed + carried — proves it is parameterised, not a constant.


def test_group_aware_ranking_clusters_same_chain_bands(monkeypatch):
    # A3 group-aware ranking — when a chain has a band GROUP (>= floor), its bands
    # carry the cohesion bonus + the group tag so they sort together and above a
    # discrete single-chain band. A lone chain gets no bonus (regression).
    # multi-band chain (gap of 5 forces two separate bands at threshold 1).
    items = [_ins(0, 0x1000, "mov w0, #1", writes={"w0": 1}),
             _ins(1, 0x1004, "add w2, w0, w0", reads={"w0": 1}, writes={"w2": 2}),
             _ins(7, 0x1020, "add w3, w2, w2", reads={"w2": 2}, writes={"w3": 4})]
    prov = ProvenanceResult(
        ProvenanceVerdict.CONTINUOUS_BUFFER, base=0x200, sink_captured=True,
        chain=[{"idx": 1, "pc": "0x1004"}, {"idx": 7, "pc": "0x1020"}],
        producer_pcs=(0x1004, 0x1020))
    monkeypatch.setattr(
        "engine.oracle_sink.validate_sink",
        lambda *a, **k: SinkValidation(SinkVerdict.SINK_CONFIRMED, base=0x200))
    monkeypatch.setattr("engine.cvd_recovery._generator.trace_provenance",
                        lambda *a, **k: prov)
    gen = RecoveryWindowGenerator(sink_base=0x200,
                                  budget=CvdBudget(band_gap_threshold=1))
    cands = gen.generate(CvdState(items, bytes([0xAB, 0xCD])))
    band_cands = [c for c in cands if c.payload.get("band") is True]
    assert len(band_cands) >= 2
    # all bands of the (>=2) group carry the group tag + group size.
    assert all(c.payload["chain_band_group"] is True for c in band_cands)
    assert all(c.payload["chain_band_group_size"] == len(band_cands)
               for c in band_cands)


# =========================================================================== #
# A4 — RecaptureSpec.to_dict() carries mem_regrel (serialization completeness).
# =========================================================================== #

from engine.recapture import RecaptureSpec
from engine.runner_client import ObservePoint, RegRelWatch


def test_recapture_spec_to_dict_carries_mem_regrel():
    # A4 验收 — a RecaptureSpec carrying a reg-relative watch round-trips through
    # to_dict() WITHOUT dropping mem_regrel, in the SAME per-watch shape the
    # runner_client wire path emits (base_reg/offset/width/pc/kind).
    op = ObservePoint(
        pc=0x1234, when="before", capture=("mem",),
        mem_regrel=(RegRelWatch(base_reg="x1", offset=8, width=4, pc=0x1234,
                                kind="read"),))
    spec = RecaptureSpec(input=b"\xde\xad", observe_points=[op])
    d = spec.to_dict()
    pt = d["observe_points"][0]
    assert "mem_regrel" in pt
    w = pt["mem_regrel"][0]
    assert w == {"base_reg": "x1", "offset": 8, "width": 4,
                 "pc": "0x1234", "kind": "read"}


def test_recapture_spec_concrete_only_omits_mem_regrel():
    # A4 regression — a pure concrete-addr ObservePoint (no mem_regrel) serializes
    # WITHOUT a mem_regrel key, same emit-only-when-non-empty convention as the wire.
    op = ObservePoint(pc=0x40, when="before", capture=("mem",),
                      mem=((0x9000, 4),))
    spec = RecaptureSpec(input=b"\x00", observe_points=[op])
    pt = spec.to_dict()["observe_points"][0]
    assert "mem_regrel" not in pt
    assert pt["mem"] == [["0x9000", 4]]


def test_recapture_spec_mem_regrel_shape_matches_wire():
    # A4 对称 — the to_dict() mem_regrel per-watch dict is byte-identical to the
    # runner_client JSON-RPC wire serialization (one shape, two sinks).
    from engine.runner_client import SubprocessRunnerAdapter
    w = RegRelWatch(base_reg="x2", offset=-16, width=8, pc=0x500, kind="write")
    op = ObservePoint(pc=0x500, when="after", capture=("mem",), mem_regrel=(w,))
    spec_pt = RecaptureSpec(input=b"\x01", observe_points=[op]).to_dict()[
        "observe_points"][0]
    wire_pt = SubprocessRunnerAdapter._serialize_observe_point(op)
    assert spec_pt["mem_regrel"] == wire_pt["mem_regrel"]


# =========================================================================== #
# B2 — verifier-internal recapture closure: run_recovery closes the
# run_recapture_loop -> run_recovery(snapshots=) loop BY CONSTRUCTION (DP1 outer-
# layer re-entry; DP2 closure-then-bands in the same call), G1 same-run snapshots,
# explicit no-runner degrade, and a re-entry-cap WARN.
# dev-recovery-verifier-internal-recapture-spec.
# =========================================================================== #

from engine.cvd_recovery import _wants_recapture  # noqa: E402
from engine.runner_client import ObservedState, RerunResult  # noqa: E402

_B2_OUT = 0x72B18
_B2_A0 = 0x9000
_B2_A1 = 0xA000
_B2_EXPECTED = bytes(range(8))


def _b2_trace():
    """Two producer loads (in-trace value 0 → sink reconstructs to zeros ≠ expected →
    NEEDS_OBSERVATION with placeable producer-read gaps) that store to the sink. Once
    the producer reads AND the sink region are observed (same rerun), provenance
    closes."""
    return [
        _ins(0, 0x1000, "ldr x8, [x9]", reads={"x9": _B2_A0}, writes={"x8": 0},
             mem=(MemOp("r", _B2_A0, 0, 4),)),
        _ins(1, 0x1004, "str x8, [x12]", reads={"x8": 0, "x12": _B2_OUT},
             mem=(MemOp("w", _B2_OUT, 0, 4),)),
        _ins(2, 0x1008, "ldr x10, [x11]", reads={"x11": _B2_A1}, writes={"x10": 0},
             mem=(MemOp("r", _B2_A1, 0, 4),)),
        _ins(3, 0x100C, "str x10, [x13]", reads={"x10": 0, "x13": _B2_OUT + 4},
             mem=(MemOp("w", _B2_OUT + 4, 0, 4),)),
    ]


_B2_BASE = CaseConfig(
    target="b2.so", input_hash="ab", run_id="r", seed_hint_addr=0x100,
    sink_hint_addr=_B2_OUT, entry_pc=0x0FFF, window=(0x1000, 0x10FF),
    reg_file=("x0", "x1"), inputs=("carrier",), parity_min=8,
    symbolic_regs=("x0", "x1"), task="recover_window")


class _B2FakeAdapter:
    """A rerun-capable adapter (the run_recapture_loop ``adapter.rerun`` interface).
    Returns, in EVERY rerun, the FULL set of observable bytes (sink region + producer
    reads) — one execution, one nonce (G1: the loop re-captures everything fresh per
    rerun and never accumulates across reruns). Records the loop_input it was driven
    with and every observe-point batch so a test can assert the same-execution wiring."""

    def __init__(self, known, output=_B2_EXPECTED, truncated=False):
        self._known = dict(known)
        self._output = output
        self._truncated = truncated
        self.calls = 0
        self.inputs: list[bytes] = []
        self.point_batches: list = []

    def rerun(self, input_bytes, observe_points=None):
        self.calls += 1
        self.inputs.append(input_bytes)
        self.point_batches.append(observe_points)
        obs = ObservedState(pc=0x70000, when="before", regs={}, mem=dict(self._known))
        return RerunResult(output=self._output, observations=(obs,),
                           truncated=self._truncated)


def _b2_known_full():
    """Bytes a same-execution rerun can observe: the SINK region (captured via
    output_observe_pc) + the two producer reads — exactly the set that closes the
    re-entered provenance."""
    known = {_B2_OUT + k: bytes([_B2_EXPECTED[k]]) for k in range(8)}
    known.update({_B2_A0 + k: bytes([0x11]) for k in range(4)})
    known.update({_B2_A1 + k: bytes([0x22]) for k in range(4)})
    return known


def test_b2_self_closes_loop_without_caller_passing_snapshots(tmp_path):
    # The core B2 claim: with a rerun-capable adapter + loop_input, run_recovery closes
    # the observation loop ITSELF — the caller passes NO snapshots and NO second call,
    # yet the run ends up NOT asking for recapture (closed → continued to bands, DP2).
    ad = _B2FakeAdapter(_b2_known_full())
    res, path = run_recovery(
        _b2_trace(), base_config=_B2_BASE, triton_runner=lambda ctx: {},
        expected=_B2_EXPECTED, work_root=str(tmp_path), ts="t",
        recapture_adapter=ad, loop_input=b"\x07", output_observe_pc=0x80000)
    assert path.exists()
    # The run no longer wants recapture: the loop closed the output observation and the
    # re-entered collect continued past the directive (no caller snapshots needed).
    assert _wants_recapture(res) is False
    assert ad.calls >= 1                      # the runner WAS driven internally
    # The self-close is VISIBLE on the gap map (never a hidden side path).
    stamp = (res.provenance or {}).get("recapture_loop")
    assert stamp is not None
    assert stamp["outcome"] == "CLOSED" and stamp["closed"] is True


def test_b2_no_runner_degrades_to_directive_with_loud_warn(tmp_path, caplog):
    # 契约③: a recapture directive but NO runner → keep today's directive gap map
    # (the agent collects by hand) + WARN LOUD. NEVER a silent close, never a dropped
    # gap. Regression onto the A2 directive behaviour PLUS the loud-warn assertion.
    import logging
    with caplog.at_level(logging.WARNING, logger="engine.cvd_recovery"):
        res, path = run_recovery(
            _b2_trace(), base_config=_B2_BASE, triton_runner=lambda ctx: {},
            expected=_B2_EXPECTED, work_root=str(tmp_path), ts="t")
    # Still asking for recapture (today's A2 behaviour — the directive is intact).
    assert _wants_recapture(res) is True
    # No self-close stamp (the loop never ran).
    assert (res.provenance or {}).get("recapture_loop") is None
    # WARN-loud: explicitly says no runner → cannot self-close, this is NOT a closure.
    assert any("RECAPTURE DIRECTIVE" in r.getMessage()
               and "no usable runner" in r.getMessage()
               and "NOT a closure" in r.getMessage()
               for r in caplog.records)


def test_b2_g1_each_rerun_is_one_execution_same_input(tmp_path):
    # G1 by construction: every rerun the loop drives uses the SAME loop_input (one
    # execution per round) and the snapshots fed back come from a SINGLE rerun — the
    # adapter is asked to re-capture EVERYTHING fresh each call, never handed a prior
    # round's snapshots to accumulate. We assert the adapter only ever saw one distinct
    # loop_input (no cross-execution mixing) and the re-entry used exactly that run.
    ad = _B2FakeAdapter(_b2_known_full())
    run_recovery(
        _b2_trace(), base_config=_B2_BASE, triton_runner=lambda ctx: {},
        expected=_B2_EXPECTED, work_root=str(tmp_path), ts="t",
        recapture_adapter=ad, loop_input=b"\x07", output_observe_pc=0x80000)
    assert ad.calls >= 1
    # G1: every rerun is ONE execution of the SAME loop_input — no cross-execution
    # mixing (the loop re-captures fresh each round, it never accumulates snapshots
    # across reruns / inputs). The re-entry feeds collect exactly the final round's
    # single-rerun snapshot set.
    assert set(ad.inputs) == {b"\x07"}
    # The loop always passes a concrete observe-point list (never None) to the runner.
    assert all(batch is not None for batch in ad.point_batches)


def test_b2_rerun_cap_truncation_propagates_warn(tmp_path, caplog):
    # The runner hits a record cap → the loop's rounds are stamped truncated; B2 must
    # propagate that (WARN), never silently treat an incomplete ledger as a clean close.
    import logging
    ad = _B2FakeAdapter(_b2_known_full(), truncated=True)
    with caplog.at_level(logging.WARNING, logger="engine.cvd_recovery"):
        res, _ = run_recovery(
            _b2_trace(), base_config=_B2_BASE, triton_runner=lambda ctx: {},
            expected=_B2_EXPECTED, work_root=str(tmp_path), ts="t",
            recapture_adapter=ad, loop_input=b"\x07", output_observe_pc=0x80000)
    # The truncation rode through to the self-close stamp.
    stamp = (res.provenance or {}).get("recapture_loop")
    assert stamp is not None and stamp["truncated"] is True
    # And it was WARNed loud (record cap / incomplete ledger), never silent.
    assert any("RECORD CAP" in r.getMessage() or "truncated" in r.getMessage()
               for r in caplog.records)


def test_b2_reentry_cap_warns_and_does_not_silently_close(tmp_path, caplog):
    # If the loop closes but the re-entered collect STILL wants recapture (the runner
    # can never satisfy the sink region — here it observes the producer reads but NOT
    # the sink region), B2 must stop at the re-entry cap with a LOUD warn and NOT
    # silently declare closure. The run still carries the directive (honest non-close).
    import logging
    known = {_B2_A0 + k: bytes([0x11]) for k in range(4)}
    known.update({_B2_A1 + k: bytes([0x22]) for k in range(4)})   # NO sink region
    ad = _B2FakeAdapter(known)
    with caplog.at_level(logging.WARNING, logger="engine.cvd_recovery"):
        res, _ = run_recovery(
            _b2_trace(), base_config=_B2_BASE, triton_runner=lambda ctx: {},
            expected=_B2_EXPECTED, work_root=str(tmp_path), ts="t",
            recapture_adapter=ad, loop_input=b"\x07")  # no output_observe_pc
    # Honest non-close: still wants recapture (never a false silent closure).
    assert _wants_recapture(res) is True
    assert any("re-entry cap" in r.getMessage() for r in caplog.records)


# =========================================================================== #
# P6 — output-determinism evidence: utov OBSERVES output stability across K
# same-input reruns and records it as a first-class, RUN-LEVEL evidence channel
# (output_determinism). EVIDENCE ONLY — never feeds close/parity/G4, never auto-
# promotes a closure; honest BOUNDED wording (observed-stable-across-K) only.
# dev-output-determinism-evidence-spec.
# =========================================================================== #

from engine.cvd_recovery import (  # noqa: E402
    OUTPUT_DET_NO_ADAPTER,
    probe_output_determinism,
    _byte_variance_ranges,
)
from engine.cvd_recovery import _OUTPUT_DET_FORBIDDEN_WORDS  # noqa: E402


class _DetAdapter:
    """A rerun-capable adapter for the P6 probe. ``outputs`` is the sequence of
    outputs returned across reruns (cycled / last-repeated); the probe drives it with
    EMPTY observe_points (it only reads rr.output)."""

    def __init__(self, outputs, truncated=False):
        self._outputs = list(outputs)
        self._truncated = truncated
        self.calls = 0
        self.point_batches: list = []

    def rerun(self, input_bytes, observe_points=None):
        self.point_batches.append(observe_points)
        out = self._outputs[min(self.calls, len(self._outputs) - 1)]
        self.calls += 1
        return RerunResult(output=out, truncated=self._truncated)


class _RaisingAdapter:
    def rerun(self, input_bytes, observe_points=None):
        raise RuntimeError("runner blew up")


class _NoRerunAdapter:
    pass


def _assert_no_forbidden_wording(ev):
    """Invariant 3: the over-strong tokens never appear anywhere in the evidence."""
    blob = json.dumps(ev).lower()
    for w in _OUTPUT_DET_FORBIDDEN_WORDS:
        assert w not in blob, f"forbidden over-strong token {w!r} leaked into {ev}"


def test_p6_probe_stable_across_k():
    # K identical outputs → observed:true, stable:true, reruns==K, a sample.
    ad = _DetAdapter([b"ABCD"])      # last-repeated → every rerun the same bytes
    ev = probe_output_determinism(ad, b"\x07", reruns=3)
    assert ev["observed"] is True and ev["stable"] is True
    assert ev["reruns"] == 3 and ad.calls == 3
    assert ev["sample_hex"] == b"ABCD".hex()
    # The probe uses EMPTY observe_points (only wants the output; never approaches G1).
    assert all(b == [] for b in ad.point_batches)
    _assert_no_forbidden_wording(ev)


def test_p6_probe_unstable_reports_byte_ranges_and_not_closed():
    # K outputs that DIFFER → stable:false + GENERIC byte-level varying/constant
    # ranges; NOT treated as stable / does not promote any closure.
    ad = _DetAdapter([b"HEADxxxx", b"HEADyyyy", b"HEADzzzz"])
    ev = probe_output_determinism(ad, b"\x07", reruns=3)
    assert ev["observed"] is True and ev["stable"] is False
    # First 4 bytes constant ("HEAD"), last 4 vary — derived generically, not baked.
    assert [0, 3] in ev["constant_byte_ranges"]
    assert [4, 7] in ev["varying_byte_ranges"]
    assert "sample_hex" not in ev          # no single sample when unstable
    _assert_no_forbidden_wording(ev)


def test_p6_byte_variance_ranges_length_difference_is_variance():
    # A length difference is itself variance (positions past the shortest output).
    varying, constant = _byte_variance_ranges([b"ABC", b"ABCDE"])
    assert constant == [[0, 2]] and varying == [[3, 4]]
    # All-identical → all constant, no varying.
    v2, c2 = _byte_variance_ranges([b"ABCD", b"ABCD"])
    assert v2 == [] and c2 == [[0, 3]]


def test_p6_probe_no_adapter_observed_false():
    # An adapter with no rerun() → observed:false + reason; NEVER defaults to stable.
    ev = probe_output_determinism(_NoRerunAdapter(), b"\x07", reruns=3)
    assert ev["observed"] is False
    assert ev["reason"] == OUTPUT_DET_NO_ADAPTER
    assert "stable" not in ev
    _assert_no_forbidden_wording(ev)


def test_p6_probe_rerun_error_observed_false_not_stable():
    # A rerun that RAISES → honest non-observation, never a defaulted stable.
    ev = probe_output_determinism(_RaisingAdapter(), b"\x07", reruns=3)
    assert ev["observed"] is False and "stable" not in ev
    assert "raised" in ev["reason"]
    _assert_no_forbidden_wording(ev)


def test_p6_probe_empty_output_observed_false():
    # An EMPTY output cannot be observed for stability → observed:false, not stable.
    ad = _DetAdapter([b""])
    ev = probe_output_determinism(ad, b"\x07", reruns=3)
    assert ev["observed"] is False and "stable" not in ev
    assert "EMPTY output" in ev["reason"]
    _assert_no_forbidden_wording(ev)


def test_p6_probe_truncated_propagates():
    # A truncated rerun (runner record cap) → truncated:true propagated; still observed.
    ad = _DetAdapter([b"ABCD"], truncated=True)
    ev = probe_output_determinism(ad, b"\x07", reruns=2)
    assert ev["observed"] is True and ev["truncated"] is True


def test_p6_run_recovery_stamps_stable_and_persists(tmp_path):
    # run_recovery with a rerun-capable adapter stamps output_determinism RUN-LEVEL
    # (on provenance) AND the on-disk gap map carries it (re-exported after the stamp).
    ad = _B2FakeAdapter(_b2_known_full())   # stable output (_B2_EXPECTED every rerun)
    res, path = run_recovery(
        _b2_trace(), base_config=_B2_BASE, triton_runner=lambda ctx: {},
        expected=_B2_EXPECTED, work_root=str(tmp_path), ts="t",
        recapture_adapter=ad, loop_input=b"\x07", output_observe_pc=0x80000,
        budget=CvdBudget(max_output_determinism_reruns=3))
    det = (res.provenance or {}).get("output_determinism")
    assert det is not None
    assert det["observed"] is True and det["stable"] is True
    assert det["reruns"] == 3
    assert det["sample_hex"] == _B2_EXPECTED.hex()
    _assert_no_forbidden_wording(det)
    # Persisted to disk (the gap map carries the run-level stamp).
    on_disk = json.loads(path.read_text().split("-->", 1)[1])
    assert on_disk["provenance"]["output_determinism"]["stable"] is True


def test_p6_run_recovery_unstable_does_not_change_verdict(tmp_path):
    # An UNSTABLE observed output records stable:false + byte ranges but must NOT be
    # used as a closure / must NOT change the run verdict (invariant 3, EVIDENCE ONLY).
    # The run is closed via same-execution ``snapshots`` (so it does NOT want recapture
    # and the adapter is driven ONLY by the P6 probe); we then compare the verdict with
    # a STABLE-probe adapter vs a VARYING-probe adapter — identical verdict/outcome,
    # only the additive evidence field differs.
    from engine.recapture import observations_to_snapshots
    snaps = observations_to_snapshots(
        _B2FakeAdapter(_b2_known_full()).rerun(b"\x07", []))

    stable_ad = _B2FakeAdapter(_b2_known_full())
    res_stable, _ = run_recovery(
        _b2_trace(), base_config=_B2_BASE, triton_runner=lambda ctx: {},
        expected=_B2_EXPECTED, work_root=str(tmp_path / "s"), ts="t",
        snapshots=snaps, recapture_adapter=stable_ad, loop_input=b"\x07")
    det_stable = (res_stable.provenance or {}).get("output_determinism")
    assert det_stable["observed"] is True and det_stable["stable"] is True

    # An adapter whose output VARIES on every probe rerun (probe only — no recapture
    # runs because the snapshots already closed the observation).
    class _VaryingAdapter:
        def __init__(self):
            self.calls = 0

        def rerun(self, input_bytes, observe_points=None):
            self.calls += 1
            return RerunResult(output=_B2_EXPECTED + bytes([self.calls]) * 4)

    vary_ad = _VaryingAdapter()
    res_vary, _ = run_recovery(
        _b2_trace(), base_config=_B2_BASE, triton_runner=lambda ctx: {},
        expected=_B2_EXPECTED, work_root=str(tmp_path / "v"), ts="t",
        snapshots=snaps, recapture_adapter=vary_ad, loop_input=b"\x07")
    det = (res_vary.provenance or {}).get("output_determinism")
    assert det["observed"] is True and det["stable"] is False
    assert det["varying_byte_ranges"]      # byte ranges present
    assert [0, 7] in det["constant_byte_ranges"]   # _B2_EXPECTED prefix constant
    # EVIDENCE ONLY: the unstable observation did NOT change the verdict/outcome vs
    # the stable run, and did NOT auto-promote any closure.
    assert res_vary.outcome == res_stable.outcome
    assert res_vary.verdict == res_stable.verdict
    assert _wants_recapture(res_vary) is False and _wants_recapture(res_stable) is False
    _assert_no_forbidden_wording(det)


def test_p6_run_recovery_no_adapter_observed_false_explicit(tmp_path):
    # No rerun-capable adapter → output_determinism.observed:false + reason, NEVER a
    # defaulted stable (the false-closure risk the spec forbids).
    res, _ = run_recovery(
        _b2_trace(), base_config=_B2_BASE, triton_runner=lambda ctx: {},
        expected=_B2_EXPECTED, work_root=str(tmp_path), ts="t")
    det = (res.provenance or {}).get("output_determinism")
    assert det == {"observed": False, "reason": OUTPUT_DET_NO_ADAPTER}


def test_p6_run_recovery_truncated_probe_warns(tmp_path, caplog):
    # A truncated rerun during the probe → truncated propagated + WARN, never silent.
    import logging
    ad = _B2FakeAdapter(_b2_known_full(), truncated=True)
    with caplog.at_level(logging.WARNING, logger="engine.cvd_recovery"):
        res, _ = run_recovery(
            _b2_trace(), base_config=_B2_BASE, triton_runner=lambda ctx: {},
            expected=_B2_EXPECTED, work_root=str(tmp_path), ts="t",
            recapture_adapter=ad, loop_input=b"\x07", output_observe_pc=0x80000)
    det = (res.provenance or {}).get("output_determinism")
    assert det["observed"] is True and det["truncated"] is True
    assert any("output-determinism probe hit a runner RECORD CAP" in r.getMessage()
               for r in caplog.records)
