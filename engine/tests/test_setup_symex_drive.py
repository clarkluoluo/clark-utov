"""setup_symex.drive — the Level-1 executing driver.

Pins what the driver makes utov-enforced so the agent stops hand-writing a
run_*.py each round: the plan runs end to end, the backing gate is never
bypassed, the two checkpoints are surfaced (DrivePause), and recording follows
the policy (durable findings + one roll-up, never per-step). Addresses are
inline here — the driver itself is target-agnostic.
"""

from __future__ import annotations

from dataclasses import replace

from engine.cvd_ledger import open_ledger
from engine.export_stamp import is_utov_export
from engine.setup_symex import (
    CaseConfig,
    DrivePause,
    DriveResult,
    build_concrete_backing,
    derive_window_symbolic_regs,
    drive,
)
from engine.types import Instruction, MemOp


def ins(idx, pc, mnem, *, reads=None, writes=None, mem=()):
    return Instruction(idx=idx, pc=pc, bytes_=b"\x00\x00\x00\x00", mnemonic=mnem,
                       regs_read=reads or {}, regs_write=writes or {}, mem=tuple(mem))


def _items():
    # F0-style reg-trace window: a load off base x16 (live-in, empty mem[]) —
    # backed only by the snapshot, not the trace. Plus a mixing step.
    return [
        ins(0, 0x1000, "ldr w0, [x16]", reads={"x16": 0x9000}),
        ins(1, 0x1004, "mul w0, w0, w1", reads={"w0": 0, "w1": 0}, writes={"w0": 0}),
    ]


CC = CaseConfig(
    target="libEncryptor.so", input_hash="ab12", run_id="run-1",
    seed_hint_addr=0x100, sink_hint_addr=0x200, entry_pc=0x0FFF,
    window=(0x1000, 0x10FF), reg_file=("x0", "x1", "x16"),
    inputs=("carrier",), parity_min=8, symbolic_regs=("x0", "x1"),
    concrete_backing=build_concrete_backing(reg_values={"x16": 0x9000}),
    task="drive_smoke",
)


def _runner(_ctx):
    return {"propagated": True, "gold_parity": "8/8",
            "expr_source": "def f(carrier):\n    return bytes(8)\n"}


BOTH = {"alias_vs_compute": "compute", "which_static": []}


def test_drive_runs_full_plan_records_durable_only(tmp_path):
    led = open_ledger(str(tmp_path / "cvd_ledger.sqlite"))
    res = drive(trace=_items(), case_config=CC, triton_runner=_runner, ledger=led,
                decisions=BOTH, ts="2026-05-31T10:00:00Z")
    assert isinstance(res, DriveResult)
    assert res.backing_ok is True
    assert res.closed is True and res.parity == "8/8" and res.emitted_F
    # recording policy: ONLY durable findings land — emit + one run-summary = 2,
    # not the 8+ per-step trail.
    n = led.execute("SELECT COUNT(*) FROM cvd_ledger").fetchone()[0]
    assert n == 2
    assert len(res.entry_keys) == 2
    # the stamped view materialised at the fixed path.
    assert res.view_path and (tmp_path / "cvd_ledger_view.md").exists()
    text = (tmp_path / "cvd_ledger_view.md").read_text()
    assert is_utov_export(text) and "Run summary" in text
    led.close()


def test_drive_pauses_at_each_checkpoint():
    # no decisions -> pause at the first checkpoint
    r1 = drive(trace=_items(), case_config=CC, triton_runner=_runner)
    assert isinstance(r1, DrivePause) and r1.checkpoint.name == "alias_vs_compute"
    assert "which_static" in r1.pending
    # resolve the first -> pause at the second
    r2 = drive(trace=_items(), case_config=CC, triton_runner=_runner,
               decisions={"alias_vs_compute": "compute"})
    assert isinstance(r2, DrivePause) and r2.checkpoint.name == "which_static"
    # resolve both -> completes
    r3 = drive(trace=_items(), case_config=CC, triton_runner=_runner, decisions=BOTH)
    assert isinstance(r3, DriveResult)


def test_drive_on_checkpoint_resolver_is_asked_both():
    seen = []

    def on_cp(cp):
        seen.append(cp.name)
        return "decided"

    res = drive(trace=_items(), case_config=CC, triton_runner=_runner, on_checkpoint=on_cp)
    assert isinstance(res, DriveResult)
    assert seen == ["alias_vs_compute", "which_static"]   # both surfaced, in order


def test_drive_never_bypasses_a_blind_backing_gate():
    # No backing -> x16 is a live-in unbacked base -> blind closure. The driver
    # must NOT run symex / emit a stub (the blind_pcs==0 hand-bypass anti-pattern).
    cc = replace(CC, concrete_backing=None)
    called = []

    def runner_spy(_ctx):
        called.append(1)
        return {"propagated": True, "gold_parity": "8/8", "expr_source": "x"}

    res = drive(trace=_items(), case_config=cc, triton_runner=runner_spy, decisions=BOTH)
    assert isinstance(res, DriveResult)
    assert res.backing_ok is False
    assert res.closed is False and res.emitted_F is None
    assert not called                       # symex/emit not reached -> no stub
    assert "NOT bypassed" in res.note


def test_drive_runs_without_ledger():
    # ledger is optional — drive still returns a result (no recording).
    res = drive(trace=_items(), case_config=CC, triton_runner=_runner, decisions=BOTH)
    assert isinstance(res, DriveResult)
    assert res.closed is True
    assert res.entry_keys == () and res.view_path is None


# --- multi-vector parity gate (per-handler / window) -------------------------

def test_drive_blocks_tautological_one_over_one_parity():
    # The handler10 lesson: a single 1/1 parity ≈ verifying the transform with
    # the trace it was derived from. backing holds, but the per-window parity
    # gate must BLOCK (not stamp exact) for want of independent cross-run vectors.
    cc = replace(CC, parity_min=1)

    def runner_1of1(_ctx):
        return {"propagated": True, "gold_parity": "1/1",
                "expr_source": "def f(carrier):\n    return bytes(8)\n"}

    res = drive(trace=_items(), case_config=cc, triton_runner=runner_1of1, decisions=BOTH)
    assert isinstance(res, DriveResult)
    assert res.backing_ok is True            # backing is fine — this is the parity leg
    assert res.closed is False               # but NOT closed: parity vectors insufficient
    assert res.parity_report and res.parity_report["verdict"] == "BLOCK"
    assert "do NOT stamp exact" in res.note


def test_drive_blocks_incomplete_transform_via_cross_run_vectors():
    # handler10 incomplete transform: matches only the deriving trace, diverges on
    # every independent cross-run vector -> BLOCK, surfaced AT the gate (no need to
    # wait for compose + gold 0/8 + boundary diff). The cohort IS output-diverse
    # (observed out-B/out-C/out-D distinct >= min), so closability is NOT the issue
    # — the F-error owns the verdict (BLOCK, an F defect), distinct from UNCLOSABLE
    # (a cohort defect). The incomplete F collapses to a constant "stop107" and
    # misses every independent vector.
    def runner_incomplete(_ctx):
        return {
            "propagated": True, "gold_parity": "1/4",
            "expr_source": "def f(carrier):\n    return bytes(8)\n",
            "parity_vectors": [
                {"input_key": "A", "observed": "out-A", "predicted": "out-A",
                 "exec_id": "run-A", "derived_from": True},
                {"input_key": "B", "observed": "out-B", "predicted": "stop107",
                 "exec_id": "run-B"},
                {"input_key": "C", "observed": "out-C", "predicted": "stop107",
                 "exec_id": "run-C"},
                {"input_key": "D", "observed": "out-D", "predicted": "stop107",
                 "exec_id": "run-D"},
            ],
        }

    res = drive(trace=_items(), case_config=CC, triton_runner=runner_incomplete,
                decisions=BOTH)
    assert res.closed is False
    assert res.parity_report["verdict"] == "BLOCK"
    assert res.parity_report["independent_observed_distinct"] == 3   # cohort IS diverse
    assert res.parity_report["independent_pass"] == 0          # A excluded; B,C,D miss
    assert set(res.parity_report["mismatches"]) == {"B", "C", "D"}


def test_drive_closes_complete_transform_via_cross_run_vectors():
    # Completing the window makes the transform match every independent vector.
    cc = replace(CC, parity_min=3)           # per-handler verify: 3 gold vectors

    def runner_complete(_ctx):
        return {
            "propagated": True, "gold_parity": "3/3",
            "expr_source": "def f(carrier):\n    return bytes(8)\n",
            # Real recovery: observed VARIES per distinct input (out-A..out-D),
            # and the complete transform predicts each one — the observed-variance
            # gate's premise (a constant observed would be an UNCLOSABLE false EXACT).
            "parity_vectors": [
                {"input_key": "A", "observed": "out-A", "predicted": "out-A",
                 "exec_id": "run-A", "derived_from": True},
                {"input_key": "B", "observed": "out-B", "predicted": "out-B",
                 "exec_id": "run-B"},
                {"input_key": "C", "observed": "out-C", "predicted": "out-C",
                 "exec_id": "run-C"},
                {"input_key": "D", "observed": "out-D", "predicted": "out-D",
                 "exec_id": "run-D"},
            ],
        }

    res = drive(trace=_items(), case_config=cc, triton_runner=runner_complete,
                decisions=BOTH)
    assert res.closed is True
    assert res.parity_report["verdict"] == "EXACT"
    assert res.parity_report["independent_pass"] == 3          # B, C, D (A excluded)
    assert res.parity_report["determinism_ok"] is True


def test_drive_unclosable_when_observed_constant_does_not_close():
    # OUTPUT-side false EXACT: the input varies (B,C,D distinct) and F matches every
    # vector, BUT the INDEPENDENT-side observed output is the SAME constant on every
    # one -> independent observed collapses to 1 distinct (< min) -> UNCLOSABLE, NOT
    # closed (no F closes a collapsed independent side; it is a COHORT defect, not an
    # F defect). drive must land terminal with the cohort reason, never CONFIRMED.
    cc = replace(CC, parity_min=3)

    def runner_degenerate(_ctx):
        return {
            "propagated": True, "gold_parity": "3/3",
            "expr_source": "def f(carrier):\n    return bytes(8)\n",
            # observed CONSTANT across the input-varying cohort -> unclosable.
            "parity_vectors": [
                {"input_key": "A", "observed": "K", "predicted": "K",
                 "exec_id": "run-A", "derived_from": True},
                {"input_key": "B", "observed": "K", "predicted": "K",
                 "exec_id": "run-B"},
                {"input_key": "C", "observed": "K", "predicted": "K",
                 "exec_id": "run-C"},
                {"input_key": "D", "observed": "K", "predicted": "K",
                 "exec_id": "run-D"},
            ],
        }

    res = drive(trace=_items(), case_config=cc, triton_runner=runner_degenerate,
                decisions=BOTH)
    assert res.closed is False                                  # UNCLOSABLE never closes
    assert res.parity_report["verdict"] == "UNCLOSABLE"
    assert res.parity_report["observed_distinct"] == 1
    assert res.parity_report["independent_observed_distinct"] == 1
    assert res.parity_report["independent_pass"] == 3           # input DID vary (B,C,D)
    assert "fix the cohort" in res.note                         # cohort reason surfaced


# --- Level-2 escape hatch surfaced through drive -----------------------------

def test_drive_surfaces_unmodeled_instruction_without_emit():
    # The Level-2 runner hit an un-modeled instruction and returned the precise
    # checkpoint instead of force-concretizing. drive must NOT emit, NOT close,
    # and carry the checkpoint (same spirit as the backing/parity gates).
    def runner_unmodeled(_ctx):
        return {"propagated": False, "expr_source": "",
                "unmodeled": {"opcode_hex": "537c019b", "mnemonic": "mul",
                              "idx": 7, "pc": "0x1008",
                              "question": "insn 537c019b (mul) @ idx 7 is not modeled",
                              "kind": "setup_symex_unmodeled_insn"}}

    res = drive(trace=_items(), case_config=CC, triton_runner=runner_unmodeled,
                decisions=BOTH)
    assert res.closed is False and res.emitted_F is None
    assert res.unmodeled and res.unmodeled["mnemonic"] == "mul"
    assert "un-modeled instruction" in res.note and "NOT force-concretized" in res.note


# --- auto-seed: utov derives per-handler symbolic inputs (window live-in) -----
# A handler's symbolic inputs = its live-in regs (read inside the window with no
# producer inside it). utov derives them so the agent never hand-configs
# symbolic_regs per handler — the run-once-look-once trap that left F0 handler11
# with sym_regs_n=0 (config filled for one handler, forgotten for the next).


def _live_in_items():
    # add reads x1,x2 (external -> live-in), writes x0; mul reads x0 (internal,
    # produced at idx0) + x3 (external -> live-in). live-in = {x1, x2, x3}.
    return [
        ins(0, 0x1000, "add x0, x1, x2", reads={"x1": 1, "x2": 2}, writes={"x0": 3}),
        ins(1, 0x1004, "mul x0, x0, x3", reads={"x0": 3, "x3": 4}, writes={"x0": 12}),
    ]


def test_derive_window_symbolic_regs_is_the_live_in_set():
    regs, info = derive_window_symbolic_regs(_live_in_items(), window=(0x1000, 0x10FF))
    assert set(regs) == {"x1", "x2", "x3"}        # x0 is produced-then-read = internal
    assert info["empty"] is False and info["dropped_not_in_reg_file"] == []
    assert info["window_basis"] == "pc" and info["n_window_items"] == 2


def test_derive_window_symbolic_regs_drops_regs_absent_from_reg_file():
    regs, info = derive_window_symbolic_regs(
        _live_in_items(), window=(0x1000, 0x10FF), reg_file=("x1", "x2"))
    assert set(regs) == {"x1", "x2"}
    assert info["dropped_not_in_reg_file"] == ["x3"]     # surfaced, not swallowed


def test_derive_window_symbolic_regs_idx_window_and_empty():
    # idx basis selects only idx 0 -> live-in {x1, x2}
    regs, _ = derive_window_symbolic_regs(
        _live_in_items(), window=(0, 0), window_is_idx=True)
    assert set(regs) == {"x1", "x2"}
    # a window matching nothing -> empty live-in, flagged
    regs2, info2 = derive_window_symbolic_regs(_live_in_items(), window=(0x9000, 0x9001))
    assert regs2 == () and info2["empty"] is True and info2["n_window_items"] == 0


def test_drive_auto_seeds_when_symbolic_regs_unconfigured():
    items = [
        ins(0, 0x1000, "ldr w0, [x16]", reads={"x16": 0x9000}, writes={"w0": 0}),
        ins(1, 0x1004, "mul w0, w0, x1", reads={"w0": 0, "x1": 7}, writes={"w0": 0}),
    ]
    cc = replace(CC, symbolic_regs=None, reg_file=("x16", "x1", "w0"))
    res = drive(trace=items, case_config=cc, triton_runner=_runner, decisions=BOTH)
    assert isinstance(res, DriveResult)
    seed = next(s for s in res.per_step if s["step"] == "seed_entry_state")
    assert "auto_seed" in seed and seed["auto_seed"]["empty"] is False
    # raw live-in = x16 (ldr base) + x1 (mul src); w0 produced-then-read = internal.
    # x16 is the concrete-backed pointer base (CC backs x16) -> excluded from the
    # symbolic set (C2 split: symbolize inputs, pin pointer bases).
    assert set(seed["symbolic_regs"]) == {"x1"}
    assert seed["auto_seed"]["backed_excluded"] == ["x16"]


def test_drive_honours_hand_supplied_symbolic_regs_override():
    # CC supplies symbolic_regs=("x0","x1") explicitly -> used verbatim, no auto-seed.
    res = drive(trace=_items(), case_config=CC, triton_runner=_runner, decisions=BOTH)
    seed = next(s for s in res.per_step if s["step"] == "seed_entry_state")
    assert set(seed["symbolic_regs"]) == {"x0", "x1"} and "auto_seed" not in seed


def test_drive_auto_seed_empty_window_falls_back_and_surfaces_note():
    cc = replace(CC, symbolic_regs=None, window=(0xDEAD, 0xDEAE))
    res = drive(trace=_items(), case_config=cc, triton_runner=_runner, decisions=BOTH)
    seed = next(s for s in res.per_step if s["step"] == "seed_entry_state")
    assert seed["auto_seed"]["empty"] is True
    # degenerate window seeds nothing -> fall back to the full reg_file under the
    # same C2 split (the backed pointer base x16 stays pinned, not symbolized).
    assert set(seed["symbolic_regs"]) == set(cc.reg_file) - {"x16"}
    assert "auto-seed found no live-in" in res.note


# --- window_kind="idx" threaded through every window step --------------------

def test_drive_window_kind_idx_threads_to_runner_and_closes():
    # An idx band (0,1) selecting the SAME two steps as the pc window — drive
    # threads window_kind to the runner ctx and the run closes just like pc.
    seen_ctx = {}

    def spy(ctx):
        seen_ctx.update(ctx)
        return _runner(ctx)

    cc = replace(CC, window=(0, 1), window_kind="idx")
    res = drive(trace=_items(), case_config=cc, triton_runner=spy, decisions=BOTH)
    assert seen_ctx["window_kind"] == "idx"      # threaded to Level-2 run_window
    assert res.closed is True and res.parity == "8/8" and res.emitted_F


def test_drive_window_kind_idx_empty_window_surfaces_note_not_silent():
    # An out-of-range idx band selects 0 trace items: every window step runs empty
    # (the bug this todo closes). drive must surface it LOUDLY, never close silent.
    cc = replace(CC, window=(100, 101), window_kind="idx")
    res = drive(trace=_items(), case_config=cc, triton_runner=_no_emit_runner,
                decisions=BOTH)
    assert res.closed is False
    assert "window_kind='idx'" in res.note and "0 trace items" in res.note


# --- memory arm of the input (handler11 symbolic=0) --------------------------

def _mem_items():
    # An external memory input: a load off backed base x16 whose bytes have no
    # in-window writer (the carrier byte a previous handler / the seed put there).
    return [
        ins(0, 0x1000, "ldr x2, [x16]", reads={"x16": 0x9000}, writes={"x2": 0},
            mem=[MemOp("r", 0x9000, 0, 8)]),
        ins(1, 0x1004, "mul w0, w2, w1", reads={"w2": 0, "w1": 0}, writes={"w0": 0}),
    ]


def _no_emit_runner(_ctx):
    return {"propagated": True, "expr_source": "", "gold_parity": "0/8"}


def test_drive_pauses_at_mem_input_symbolize_vs_back_checkpoint():
    # CC backs x16 (a reg) but NOT the loaded bytes 0x9000 -> un-pinned mem input
    # -> drive BLOCKS at the named checkpoint (the symbolic=0 trap, surfaced as a
    # judgment, never auto-guessed). base_regs travels for re-pin context.
    res = drive(trace=_mem_items(), case_config=CC, triton_runner=_no_emit_runner,
                decisions=BOTH)
    assert isinstance(res, DrivePause)
    assert res.checkpoint.name == "mem_input_symbolize_vs_back"
    assert "0x9000" in res.checkpoint.question and "via ['x16']" in res.checkpoint.question


def test_drive_mem_decision_symbolize_applies_and_proceeds():
    # The agent decides SYMBOLIZE via the checkpoint -> drive applies it to the
    # entry spec's symbolic_mem and proceeds (no pause).
    dec = {**BOTH, "mem_input_symbolize_vs_back": {0x9000: {"symbolize": 0xFB9881B1}}}
    res = drive(trace=_mem_items(), case_config=CC, triton_runner=_no_emit_runner,
                decisions=dec)
    assert isinstance(res, DriveResult)
    seed = next(s for s in res.per_step if s["step"] == "seed_entry_state")
    assert seed["mem_live_in"]["unpinned"] == [] and 0x9000 in seed["mem_live_in"]["symbolized"]


def test_drive_mem_decision_back_is_acknowledged():
    dec = {**BOTH, "mem_input_symbolize_vs_back": {"0x9000": "back"}}
    res = drive(trace=_mem_items(), case_config=CC, triton_runner=_no_emit_runner,
                decisions=dec)
    assert isinstance(res, DriveResult)
    seed = next(s for s in res.per_step if s["step"] == "seed_entry_state")
    assert seed["mem_live_in"]["decided_back"] == [0x9000]
    assert seed["mem_live_in"]["unpinned"] == []


def test_drive_mem_on_checkpoint_resolver_decides():
    seen = []

    def on_cp(cp):
        seen.append(cp.name)
        return {0x9000: {"symbolize": 0xFB9881B1}}

    res = drive(trace=_mem_items(), case_config=CC, triton_runner=_no_emit_runner,
                decisions=BOTH, on_checkpoint=on_cp)
    assert isinstance(res, DriveResult)
    assert "mem_input_symbolize_vs_back" in seen


def test_drive_symbolic_mem_preconfig_skips_the_checkpoint():
    # A pre-decided / cached symbolic_mem resolves it without a pause.
    cc = replace(CC, symbolic_mem=((0x9000, 8, 0xFB9881B1),))
    res = drive(trace=_mem_items(), case_config=cc, triton_runner=_no_emit_runner,
                decisions=BOTH)
    assert isinstance(res, DriveResult)
    seed = next(s for s in res.per_step if s["step"] == "seed_entry_state")
    assert seed["mem_live_in"]["unpinned"] == [] and 0x9000 in seed["mem_live_in"]["symbolized"]


# --- §5′: DYNAMIC backing gate (the BLOCK criterion is the concolic view) -----
# A load whose EA (op.addr) AND loaded value are in the trace is DYNAMICALLY
# backed (back.sufficient True) — it is NOT blind, regardless of whether its
# STATIC address closure (clo) is symbolic. That is exactly the P2(i) forwarding
# frontier: the gate ENTERS it into symex (the clo-symbolic verdict becomes an
# INFORMATIONAL deferral note, NOT a BLOCK). Only a TRULY blind leg (op.addr AND
# value both absent) leaves back.sufficient False and BLOCKs. The CLOSE decision
# is UNCHANGED: only the parity_ok chain (P2(i) + G4 + cross-vector parity) may
# close — the gate only governs "does it enter symex".


def _sym_ea_items():
    # Form J′: load with a mem[] operand (op.addr+val present -> back ok) off base
    # x16 that is a symbolic live-in input (clo NOT ok: x16 is an un-backed symbolic
    # root). diagnose -> symbolic_address. The OLD (clo-AND) gate blocked this as
    # blind; the §5′ dynamic gate ENTERS it into symex (deferred to P2(i)).
    return [
        ins(0, 0x1000, "ldr w0, [x16]", reads={"x16": 0x9000},
            mem=[MemOp("r", 0x9000, 0, 4)]),
        ins(1, 0x1004, "mul w0, w0, w1", reads={"w0": 0, "w1": 0}, writes={"w0": 0}),
    ]


# x16 is symbolic (the addressing input); no concrete_backing (would otherwise
# make clo sufficient and skip the new path).
SYM_EA_CC = replace(CC, concrete_backing=None, symbolic_regs=("x0", "x1", "x16"))

# the load's bytes have no in-window writer -> an external mem input checkpoint;
# symbolize it so the run proceeds to the backing gate (the leg under test).
DEC_SYM_MEM = {**BOTH, "mem_input_symbolize_vs_back": {0x9000: {"symbolize": 0xFB9881B1}}}


def test_drive_form_J_symbolic_ea_load_is_deferred_into_symex():
    # §5′ The DYNAMIC gate DEFERS the symbolic-EA window into symex (op.addr+value
    # present -> back.sufficient True -> backing_ok True), instead of BLOCKing it
    # as if blind. A no-emit P2(i) (not yet resolved) -> NOT closed, but symex ran
    # and the note is the honest "deferred, NOT missing backing".
    called = []

    def runner_spy(_ctx):
        called.append(1)
        return {"propagated": True, "expr_source": "", "gold_parity": "0/8"}

    res = drive(trace=_sym_ea_items(), case_config=SYM_EA_CC,
                triton_runner=runner_spy, decisions=DEC_SYM_MEM)
    assert isinstance(res, DriveResult)
    # §5′ semantic change: backing_ok is the DYNAMIC view (op.addr+value present),
    # so it is now True even though the STATIC closure (clo) is symbolic.
    assert res.backing_ok is True
    assert res.address_closure["sufficient"] is False    # clo still flags symbolic EA
    assert called                                   # symex/emit path WAS entered
    gate = next(s for s in res.per_step if s["step"] == "symbolic_addressing_gate")
    assert gate["deferred"] is True
    assert gate["staging_verdict"] == "symbolic_address"
    assert any(s["step"] == "symex" for s in res.per_step)
    # P2(i) did not resolve -> NOT closed, and the note is honest: deferred, do
    # NOT re-capture (it is NOT the missing-backing blind note).
    assert res.closed is False
    assert "do NOT re-capture" in res.note
    assert "NOT bypassed" not in res.note


def test_drive_form_J_symbolic_ea_deferred_even_when_diagnose_inconclusive():
    # §5′ KEY difference vs the old ⑤ (diagnose-gated bypass): the gate is the
    # DYNAMIC backing view, NOT the diagnose verdict. A dynamically-backed window
    # is entered into symex even when diagnose returns INCONCLUSIVE (no symbolic
    # input configured -> staging_verdict is NOT symbolic_address) — the old flag
    # would have kept it BLOCKED here.
    cc = replace(CC, concrete_backing=None, symbolic_regs=("x0",))   # x16 NOT symbolic
    called = []

    def runner_spy(_ctx):
        called.append(1)
        return {"propagated": True, "expr_source": "", "gold_parity": "0/8"}

    res = drive(trace=_sym_ea_items(), case_config=cc,
                triton_runner=runner_spy, decisions=DEC_SYM_MEM)
    assert isinstance(res, DriveResult)
    assert res.backing_ok is True                   # dynamic view: op.addr+value present
    assert res.address_closure["sufficient"] is False
    assert called                                   # entered symex regardless of diagnose
    gate = next(s for s in res.per_step if s["step"] == "symbolic_addressing_gate")
    assert gate["deferred"] is True
    assert gate["staging_verdict"] != "symbolic_address"   # inconclusive / known_addr
    assert "do NOT re-capture" in res.note
    assert "NOT bypassed" not in res.note


def test_drive_surfaces_symbolic_forwards_from_runner():
    # Phase 2(i) "+ record-a-line": the runner's symbolic_forwards count flows into
    # the symex per_step (next to ``propagated``) and onto the DriveResult — the
    # "forwarded M loads" terminal evidence. Purely observational.
    def runner_spy(_ctx):
        return {"propagated": True, "expr_source": "", "gold_parity": "0/8",
                "symbolic_forwards": 3,
                "symbolic_forward_sites": [[0x1004, 0x10020, 8]]}

    res = drive(trace=_sym_ea_items(), case_config=SYM_EA_CC,
                triton_runner=runner_spy, decisions=DEC_SYM_MEM)
    assert isinstance(res, DriveResult)
    symex = next(s for s in res.per_step if s["step"] == "symex")
    assert symex["symbolic_forwards"] == 3
    assert res.symbolic_forwards == 3
    assert res.to_dict()["symbolic_forwards"] == 3


def test_drive_symbolic_forwards_zero_when_no_symex_runs():
    # Invariant 7 / observational default: when the symex block does not run (or the
    # runner reports no forwards) symbolic_forwards is 0 on the result — never absent,
    # never a gate input.
    def runner_spy(_ctx):
        return {"propagated": True, "expr_source": "", "gold_parity": "0/8"}

    res = drive(trace=_sym_ea_items(), case_config=SYM_EA_CC,
                triton_runner=runner_spy, decisions=DEC_SYM_MEM)
    assert isinstance(res, DriveResult)
    assert res.symbolic_forwards == 0


def test_drive_form_J_symbolic_ea_closes_only_via_parity_chain():
    # CLOSE soundness: a deferred (clo-symbolic) window closes ONLY by the same
    # parity_ok chain (P2(i) forwarding success + G4 self-check + independent
    # cross-run vectors), never relaxed by the gate. backing_ok is True (dynamic),
    # but closing still rides parity — clo plays no part in the close decision.
    cc = replace(SYM_EA_CC, parity_min=3)

    def runner_complete(_ctx):
        return {
            "propagated": True, "gold_parity": "3/3",
            "expr_source": "def f(carrier):\n    return bytes(8)\n",
            # Real recovery: observed VARIES per distinct input (a constant observed
            # would be an UNCLOSABLE false EXACT under the observed-variance gate).
            "parity_vectors": [
                {"input_key": "A", "observed": "out-A", "predicted": "out-A",
                 "exec_id": "run-A", "derived_from": True},
                {"input_key": "B", "observed": "out-B", "predicted": "out-B",
                 "exec_id": "run-B"},
                {"input_key": "C", "observed": "out-C", "predicted": "out-C",
                 "exec_id": "run-C"},
                {"input_key": "D", "observed": "out-D", "predicted": "out-D",
                 "exec_id": "run-D"},
            ],
        }

    res = drive(trace=_sym_ea_items(), case_config=cc,
                triton_runner=runner_complete, decisions=DEC_SYM_MEM)
    assert isinstance(res, DriveResult)
    assert res.backing_ok is True                    # dynamic backing holds
    assert res.address_closure["sufficient"] is False  # but static closure symbolic
    assert res.closed is True                        # closed via parity chain only
    assert res.parity_report and res.parity_report["verdict"] == "EXACT"


def test_drive_form_J_symbolic_ea_does_not_close_without_parity_vectors():
    # The deferral must NOT let a tautological 1/1 (no independent vectors) close.
    cc = replace(SYM_EA_CC, parity_min=1)

    def runner_1of1(_ctx):
        return {"propagated": True, "gold_parity": "1/1",
                "expr_source": "def f(carrier):\n    return bytes(8)\n"}

    res = drive(trace=_sym_ea_items(), case_config=cc,
                triton_runner=runner_1of1, decisions=DEC_SYM_MEM)
    assert isinstance(res, DriveResult)
    assert res.closed is False                       # parity vectors insufficient
    assert res.parity_report and res.parity_report["verdict"] == "BLOCK"


def _truly_blind_items():
    # Form K′: a mem-class step whose op.addr is absent (no mem[] operand) AND whose
    # value is absent (no regs_write -> the loaded value is nowhere on the reg side).
    # That is a TRULY blind leg -> back.sufficient False -> BLOCK + re-capture.
    return [
        ins(0, 0x1000, "ldr w0, [x16]", reads={"x16": 0x9000}),    # no operand, no write
        ins(1, 0x1004, "mul w3, w3, w1", reads={"w3": 0, "w1": 0}, writes={"w3": 0}),
    ]


def test_drive_form_K_truly_blind_still_blocks_with_recapture_note():
    # §5′ The TRULY blind leg (op.addr AND value both absent) keeps back.sufficient
    # False -> backing_ok False -> the blind-closure BLOCK + re-capture note holds.
    cc = replace(CC, concrete_backing=None, symbolic_regs=("x0", "x1", "x16"))
    called = []

    def runner_spy(_ctx):
        called.append(1)
        return {"propagated": True, "gold_parity": "8/8", "expr_source": "x"}

    res = drive(trace=_truly_blind_items(), case_config=cc,
                triton_runner=runner_spy, decisions=DEC_SYM_MEM)
    assert isinstance(res, DriveResult)
    assert res.backing_ok is False
    assert res.closed is False and res.emitted_F is None
    assert not called                                # blind short-circuit holds
    # back.sufficient False -> no clo-deferral diagnosis fired (no gate step).
    assert not any(s["step"] == "symbolic_addressing_gate" for s in res.per_step)
    assert "NOT bypassed" in res.note                # the re-capture note


def _value_on_reg_items():
    # Form M′: a load with NO mem[] operand (op.addr absent) but its loaded value IS
    # observable on the reg side (regs_write carries the dest reg — recovered via the
    # ② regs_read/regs_write fallback). The VALUE dimension makes it dynamically
    # backed -> entered into symex (proves the value-dim extension).
    return [
        ins(0, 0x1000, "ldr w0, [x16]", reads={"x16": 0x9000}, writes={"w0": 0x1234}),
        ins(1, 0x1004, "mul w0, w0, w1", reads={"w0": 0x1234, "w1": 0}, writes={"w0": 0}),
    ]


def test_drive_form_M_value_on_reg_side_is_dynamically_backed():
    # §5′ value-dimension: no op.addr, but the loaded value is in regs_write -> the
    # leg is NOT value-blind -> back.sufficient True -> entered into symex.
    cc = replace(CC, concrete_backing=None, symbolic_regs=("x0", "x1", "x16"))
    called = []

    def runner_spy(_ctx):
        called.append(1)
        return {"propagated": True, "expr_source": "", "gold_parity": "0/8"}

    res = drive(trace=_value_on_reg_items(), case_config=cc,
                triton_runner=runner_spy, decisions=DEC_SYM_MEM)
    assert isinstance(res, DriveResult)
    assert res.backing_ok is True                   # value-on-reg => dynamically backed
    assert called                                   # entered symex
    assert "NOT bypassed" not in res.note           # not the missing-backing block


def _mixed_window_items():
    # Form L′: one symbolic-EA leg (op.addr+value present, x16 symbolic) + one TRULY
    # blind leg (no operand, no regs_write -> op.addr AND value both absent) off x17.
    # The truly-blind leg makes back.sufficient False, so the WHOLE window BLOCKs
    # (a real blind leg is not carried over by the symbolic leg) — mixed-window safety.
    return [
        ins(0, 0x1000, "ldr w0, [x16]", reads={"x16": 0x9000},
            mem=[MemOp("r", 0x9000, 0, 4)]),
        ins(1, 0x1004, "ldr w2, [x17]", reads={"x17": 0xA000}),   # no operand, no write = blind
        ins(2, 0x1008, "mul w0, w0, w2", reads={"w0": 0, "w2": 0}, writes={"w0": 0}),
    ]


def test_drive_form_L_mixed_window_truly_blind_leg_still_blocks():
    # A real blind leg (op.addr AND value both absent) keeps back.sufficient False,
    # so backing_ok is False and the whole window BLOCKs — the dynamic gate is NOT
    # released on the strength of the one symbolic-EA leg.
    cc = replace(CC, concrete_backing=None,
                 symbolic_regs=("x0", "x1", "x16"),
                 window=(0x1000, 0x10FF), reg_file=("x0", "x1", "x16", "x17"))
    called = []

    def runner_spy(_ctx):
        called.append(1)
        return {"propagated": True, "gold_parity": "8/8", "expr_source": "x"}

    res = drive(trace=_mixed_window_items(), case_config=cc,
                triton_runner=runner_spy, decisions=DEC_SYM_MEM)
    assert isinstance(res, DriveResult)
    assert res.backing_ok is False
    assert res.closed is False and res.emitted_F is None
    assert not called                                # not released into symex
    # back.sufficient is False -> no clo-deferral diagnosis fired (no gate step).
    assert not any(s["step"] == "symbolic_addressing_gate" for s in res.per_step)
    assert "NOT bypassed" in res.note


def test_drive_backed_closure_window_skips_the_symbolic_ea_gate(tmp_path):
    # Regression / invariant 7: a window whose clo.sufficient is already True (the
    # vast majority — tc2 class) takes the original path byte-for-byte: the
    # diagnose call is NOT made, no gate step is recorded, and it closes normally.
    res = drive(trace=_items(), case_config=CC, triton_runner=_runner, decisions=BOTH)
    assert isinstance(res, DriveResult)
    assert res.backing_ok is True and res.closed is True
    assert not any(s["step"] == "symbolic_addressing_gate" for s in res.per_step)


# --- Phase 2(i) FEED LINE: diagnose -> entry.symbolic_staging -> forwarding ----
# The forwarding MECHANISM (runner._symbolic_staging -> _shadow_concrete_reads skip)
# is pinned Triton-side in test_setup_symex_runner (hand-filled entry). THIS section
# pins the missing producer half: drive's clo_deferred branch, on a symbolic_address
# verdict, AUTO-injects the backbone staging interval(s) into entry.symbolic_staging
# BEFORE the symex call — so the symbolic-address window forwards on the FIRST run in
# the automatic flow instead of collapsing to an opaque frontier. The injection is
# observed where it lands: the dict the runner receives (ctx["entry"]["symbolic_
# staging"]). The real Triton forward->close is already covered by the runner test;
# here the load-bearing assertion is the FEED reaching the runner, which needs no
# Triton. Synthetic, zero case-specific (addresses are mechanism fixtures).

from engine.opaque_staging import PointerChainSpec   # noqa: E402


def _staging_entry_spy():
    """A no-emit runner that captures the entry dict it is handed (so the test can
    read what drive injected into entry.symbolic_staging)."""
    seen: dict[str, object] = {}

    def runner(ctx):
        seen["entry"] = dict(ctx["entry"])
        return {"propagated": True, "expr_source": "", "gold_parity": "0/8"}

    return runner, seen


def test_drive_form_J_auto_injects_symbolic_staging_interval():
    # FORM J: a symbolic_address staging window driven through the COMPLETE drive
    # (NOT a hand-filled entry). drive must auto-inject the target load's trace
    # (op.addr, op.size) into entry.symbolic_staging, surfaced both on the runner
    # ctx and (case-agnostic) in the gate's per_step count. The load reads
    # MemOp("r", 0x9000, 0, 4) -> injected interval (0x9000, 4).
    runner, seen = _staging_entry_spy()
    res = drive(trace=_sym_ea_items(), case_config=SYM_EA_CC,
                triton_runner=runner, decisions=DEC_SYM_MEM)
    assert isinstance(res, DriveResult)
    gate = next(s for s in res.per_step if s["step"] == "symbolic_addressing_gate")
    assert gate["staging_verdict"] == "symbolic_address"
    # the FEED LINE: the interval reached the runner's entry (the producer half).
    assert seen["entry"]["symbolic_staging"] == [[0x9000, 4]]
    # observable, invariant-4-safe count (not the full list) in per_step.
    assert gate["symbolic_staging_injected"] == 1


def test_drive_form_J_old_empty_entry_is_the_collapse_baseline():
    # CONTRAST (old code = empty entry): a runner that IGNORES the injected staging
    # (the pre-feed-line behaviour) gets no forward -> F collapses (no emit) -> the
    # window does NOT close. This is the causal control: the feed line is what gives
    # the symbolic-address load its forward chance; without the runner honouring it,
    # the same window is an opaque frontier.
    runner, _seen = _staging_entry_spy()   # no-emit == collapse
    res = drive(trace=_sym_ea_items(), case_config=SYM_EA_CC,
                triton_runner=runner, decisions=DEC_SYM_MEM)
    assert isinstance(res, DriveResult)
    assert res.closed is False and res.emitted_F is None   # collapsed without forward
    # but the feed itself DID run (drive's half is done; closing is the runner's).
    gate = next(s for s in res.per_step if s["step"] == "symbolic_addressing_gate")
    assert gate["symbolic_staging_injected"] == 1


def test_drive_form_J_pointer_chain_enhancement_merges_store_landing():
    # FORM J enhancement: when a pointer_chain is supplied, drive ALSO merges the
    # chain's store-side landing (resolve_staging_address) into the injected set,
    # de-duped with the backbone. Here a separate store lands the symbol at a
    # different address than the load reads — the chain surfaces the store landing
    # too. Backbone alone would miss the store-only interval.
    store_addr = 0xA000
    items = [
        ins(0, 0x1000, "str x8, [x20]", reads={"x8": 0x41, "x20": store_addr},
            mem=[MemOp("w", store_addr, 0x41, 8)]),
        ins(1, 0x1004, "ldr w0, [x16]", reads={"x16": 0x9000},
            mem=[MemOp("r", 0x9000, 0, 4)]),
        ins(2, 0x1008, "mul w0, w0, w1", reads={"w0": 0, "w1": 0}, writes={"w0": 0}),
    ]
    chain = PointerChainSpec(store_base_regs=("x20",), load_base_regs=("x16",))
    runner, seen = _staging_entry_spy()
    res = drive(trace=items, case_config=SYM_EA_CC,
                triton_runner=runner, decisions=DEC_SYM_MEM, pointer_chain=chain)
    assert isinstance(res, DriveResult)
    gate = next(s for s in res.per_step if s["step"] == "symbolic_addressing_gate")
    assert gate["staging_verdict"] == "symbolic_address"
    injected = {tuple(iv) for iv in seen["entry"]["symbolic_staging"]}
    assert (0x9000, 4) in injected        # backbone (the load landing)
    assert (store_addr, 8) in injected    # pointer-chain store landing, merged
    assert gate["symbolic_staging_injected"] == 2


def test_drive_form_J_no_chain_self_derives_store_landing():
    # 坎3 (self-produce the shape, no caller obligation): with NO pointer_chain the
    # backbone STILL feeds (case-agnostic) AND drive SELF-DERIVES the staging shape
    # from its own diagnosis, so the store-side landing is surfaced WITHOUT the
    # caller hand-typing a chain. The store [x20] lands a staging byte → derived
    # store_base_regs={x20} → resolve_staging_address surfaces (store_addr, 8). This
    # is the CORE self-proof: caller does not pass pointer_chain yet the store-side
    # narrow has a shape (contrast: pre-坎3 this interval was永远 absent w/o a chain).
    store_addr = 0xA000
    items = [
        ins(0, 0x1000, "str x8, [x20]", reads={"x8": 0x41, "x20": store_addr},
            mem=[MemOp("w", store_addr, 0x41, 8)]),
        ins(1, 0x1004, "ldr w0, [x16]", reads={"x16": 0x9000},
            mem=[MemOp("r", 0x9000, 0, 4)]),
        ins(2, 0x1008, "mul w0, w0, w1", reads={"w0": 0, "w1": 0}, writes={"w0": 0}),
    ]
    runner, seen = _staging_entry_spy()
    res = drive(trace=items, case_config=SYM_EA_CC,
                triton_runner=runner, decisions=DEC_SYM_MEM)   # no chain → self-derive
    assert isinstance(res, DriveResult)
    injected = {tuple(iv) for iv in seen["entry"]["symbolic_staging"]}
    assert (0x9000, 4) in injected             # backbone fed
    assert (store_addr, 8) in injected         # SELF-DERIVED store landing (no caller chain)


def test_drive_form_K_known_addr_injects_empty_invariant_7():
    # FORM K (invariant 7): a clo_deferred window whose verdict is NOT
    # symbolic_address must inject NOTHING -> entry.symbolic_staging stays [] ->
    # the runner's _symbolic_staging is byte-for-byte the pre-feed-line state. Here
    # x16 is NOT a symbolic input -> diagnose returns inconclusive/known_addr.
    cc = replace(CC, concrete_backing=None, symbolic_regs=("x0",))   # x16 NOT symbolic
    runner, seen = _staging_entry_spy()
    res = drive(trace=_sym_ea_items(), case_config=cc,
                triton_runner=runner, decisions=DEC_SYM_MEM)
    assert isinstance(res, DriveResult)
    gate = next(s for s in res.per_step if s["step"] == "symbolic_addressing_gate")
    assert gate["staging_verdict"] != "symbolic_address"
    assert seen["entry"]["symbolic_staging"] == []      # nothing injected
    assert gate["symbolic_staging_injected"] == 0


def test_drive_form_K_plain_backed_window_no_gate_no_staging_field_change():
    # FORM K (invariant 7): a plain clo-sufficient window never reaches the gate, so
    # entry.symbolic_staging is the default empty — the to_dict carries [] and the
    # window closes exactly as before the feed line existed.
    runner, seen = _staging_entry_spy_emit()
    res = drive(trace=_items(), case_config=CC, triton_runner=runner, decisions=BOTH)
    assert isinstance(res, DriveResult)
    assert not any(s["step"] == "symbolic_addressing_gate" for s in res.per_step)
    assert seen["entry"]["symbolic_staging"] == []      # default empty, unchanged


def _staging_entry_spy_emit():
    """Spy that DOES emit (so a clo-sufficient window closes) and captures entry."""
    seen: dict[str, object] = {}

    def runner(ctx):
        seen["entry"] = dict(ctx["entry"])
        return {"propagated": True, "gold_parity": "8/8",
                "expr_source": "def f(carrier):\n    return bytes(8)\n"}

    return runner, seen


def test_drive_form_L_ea_truly_varies_injects_but_parity_blocks():
    # FORM L (degrade still yields a verdict — gates not relaxed): a symbolic_address
    # window whose EA TRULY varies with input. drive injects the interval and the
    # runner gets the forward chance, but a tautological 1/1 parity (no independent
    # vectors) still BLOCKs -> NOT closed -> falls to the opaque frontier. The feed
    # line opens the door; the existing parity gate still decides the close.
    cc = replace(SYM_EA_CC, parity_min=1)
    captured: dict[str, object] = {}

    def runner_1of1(ctx):
        captured["entry"] = dict(ctx["entry"])
        return {"propagated": True, "gold_parity": "1/1",
                "expr_source": "def f(carrier):\n    return bytes(8)\n"}

    res = drive(trace=_sym_ea_items(), case_config=cc,
                triton_runner=runner_1of1, decisions=DEC_SYM_MEM)
    assert isinstance(res, DriveResult)
    # the interval WAS injected (door opened) ...
    assert captured["entry"]["symbolic_staging"] == [[0x9000, 4]]
    gate = next(s for s in res.per_step if s["step"] == "symbolic_addressing_gate")
    assert gate["symbolic_staging_injected"] == 1
    # ... but the parity gate is NOT relaxed -> still does not close (opaque frontier).
    assert res.closed is False
    assert res.parity_report and res.parity_report["verdict"] == "BLOCK"


# --- Phase 2(i) re-續: OPAQUE-PATH fallback re-run (two-path merge) -------------
# The FIRST-run injection (above) only covers the clo_deferred window (back ok +
# STATIC closure symbolic). A window whose static closure IS backed
# (clo_deferred=False — e.g. a POINTER-INDIRECT staging load: ldr xN,[xM] then
# ldr w,[xN]) never got the first-run injection, ran symex blind, and collapsed to
# opaque. The opaque fallback re-diagnoses via the Phase 0b DFG path (independent of
# clo_deferred), injects the staging interval(s), and re-runs symex ONCE — gated on
# the FIRST run having forwarded NOTHING (symbolic_forwards == 0). The close criterion
# is UNCHANGED (parity_ok / G4 / cross-vector); a still-blocking re-run stays a
# frontier. These pin the BRANCH logic with stub runners (no Triton): the runner keys
# its behaviour on whether the entry it is handed already carries symbolic_staging.


def _ptr_indirect_items():
    # FORM P backbone: a pointer-indirect staging chain. ldr x16,[x20] reads the
    # pointer (x20 BACKED -> that leg's closure is backed); ldr w0,[x16] uses the
    # loaded pointer as its EA. The closure of the SECOND load resolves x16 back to
    # the (backed) x20 -> clo.sufficient True -> clo_deferred is FALSE (no first-run
    # injection). But diagnose's DFG path sees x16 produced BY a memory load -> a
    # pointer chain -> verdict symbolic_address. So the window runs symex blind,
    # collapses, and is exactly the opaque-fallback's target.
    return [
        ins(0, 0x1000, "ldr x16, [x20]", reads={"x20": 0x8000}, writes={"x16": 0x9000},
            mem=[MemOp("r", 0x8000, 0x9000, 8)]),
        ins(1, 0x1004, "ldr w0, [x16]", reads={"x16": 0x9000}, writes={"w0": 0x55},
            mem=[MemOp("r", 0x9000, 0x55, 4)]),
        ins(2, 0x1008, "mul w0, w0, w1", reads={"w0": 0, "w1": 0}, writes={"w0": 0}),
    ]


# x20 is the (backed) pointer base; x16/x0/x1 symbolic inputs. The two staging loads
# read external bytes with no in-window writer -> a mem-input checkpoint; back the
# pointer slot (0x8000) and the staging slot (0x9000) so the run proceeds to symex.
PTR_CC = replace(
    CC, concrete_backing=build_concrete_backing(reg_values={"x20": 0x8000}),
    symbolic_regs=("x0", "x1", "x16"), reg_file=("x0", "x1", "x16", "x20"))
DEC_PTR = {**BOTH, "mem_input_symbolize_vs_back": {0x8000: "back", 0x9000: "back"}}


def _staging_gated_runner(*, parity, forwards, expr, vectors=None):
    """A stub runner whose result depends on whether the entry it is handed already
    carries a symbolic_staging interval. The FIRST drive pass (no injection on a
    clo_deferred=False window) gets the collapse result (no emit, 0 forwards); the
    re-run (entry carries the fallback-injected interval) gets the forward result."""
    calls: list[dict] = []

    def runner(ctx):
        calls.append(dict(ctx["entry"]))
        if ctx["entry"].get("symbolic_staging"):
            out = {"propagated": True, "gold_parity": parity,
                   "symbolic_forwards": forwards, "expr_source": expr}
            if vectors is not None:
                out["parity_vectors"] = vectors
            return out
        # first pass on the clo_deferred=False window: blind -> collapse, 0 forwards.
        return {"propagated": True, "expr_source": "", "gold_parity": "0/8",
                "symbolic_forwards": 0}

    return runner, calls


# A real-recovery cohort: observed VARIES per distinct input (out-A..out-D), the
# observed-variance gate's premise. A constant observed across the input-varying
# cohort would be an UNCLOSABLE false EXACT, not a CONFIRMED close.
_FOUR_VECTORS = [
    {"input_key": "A", "observed": "out-A", "predicted": "out-A", "exec_id": "e-A",
     "derived_from": True},
    {"input_key": "B", "observed": "out-B", "predicted": "out-B", "exec_id": "e-B"},
    {"input_key": "C", "observed": "out-C", "predicted": "out-C", "exec_id": "e-C"},
    {"input_key": "D", "observed": "out-D", "predicted": "out-D", "exec_id": "e-D"},
]


def test_drive_form_P_opaque_fallback_revives_pointer_indirect():
    # FORM P: a clo_deferred=False pointer-indirect window collapses to opaque on the
    # first run (symbolic_forwards==0); the opaque fallback diagnoses symbolic_address,
    # injects the staging interval(s), re-runs symex ONCE, forwards, and CLOSES ->
    # CONFIRMED. The first run injects nothing (no symbolic_addressing_gate step), so
    # this is the opaque-path branch, NOT the first-run feed line.
    runner, calls = _staging_gated_runner(
        parity="8/8", forwards=1,
        expr="def f(carrier):\n    return bytes(8)\n", vectors=_FOUR_VECTORS)
    res = drive(trace=_ptr_indirect_items(), case_config=PTR_CC,
                triton_runner=runner, decisions=DEC_PTR)
    assert isinstance(res, DriveResult)
    # clo.sufficient True -> NO first-run injection gate fired.
    assert not any(s["step"] == "symbolic_addressing_gate" for s in res.per_step)
    # the symex block ran TWICE (first collapse + fallback re-run).
    assert len(calls) == 2
    assert calls[0]["symbolic_staging"] == []          # first run: blind, no injection
    assert calls[1]["symbolic_staging"]                # re-run: fallback injected
    # the fallback step is recorded with observable counts (invariant 4: counts).
    retry = next(s for s in res.per_step if s["step"] == "opaque_forward_retry")
    assert retry["injected"] >= 1
    assert retry["retry_symbolic_forwards"] == 1
    assert retry["retry_parity_ok"] is True
    # the re-run closed via the SAME parity chain -> CONFIRMED.
    assert res.closed is True
    assert res.symbolic_forwards == 1


def test_drive_form_P_old_no_fallback_is_the_frontier_baseline():
    # CONTRAST (old code = no opaque fallback): a runner that NEVER forwards (the
    # pre-fallback behaviour — first pass collapses and there is no re-run) leaves the
    # window an opaque frontier. This is the causal control: the fallback re-run is
    # what gives the pointer-indirect staging load its forward chance.
    def no_forward_runner(ctx):
        # always collapse, regardless of injection (simulates the un-resolvable case
        # AND, for the first pass, the old single-run world).
        return {"propagated": True, "expr_source": "", "gold_parity": "0/8",
                "symbolic_forwards": 0}

    res = drive(trace=_ptr_indirect_items(), case_config=PTR_CC,
                triton_runner=no_forward_runner, decisions=DEC_PTR)
    assert isinstance(res, DriveResult)
    assert res.closed is False                         # opaque frontier
    assert res.emitted_F is None


def test_drive_form_Q_already_forwarded_window_is_not_re_run():
    # FORM Q (invariant 7 / non-redundant): a clo_deferred=True window whose FIRST run
    # already FORWARDED (symbolic_forwards > 0) but BLOCKs on parity must NOT trigger
    # the opaque fallback — it is a real frontier already tried, not an un-tried
    # collapse. symex runs exactly ONCE; no opaque_forward_retry step.
    cc = replace(SYM_EA_CC, parity_min=1)
    calls = []

    def runner_forwarded_but_blocks(ctx):
        calls.append(1)
        # forwarded on the first (clo_deferred) run, but a tautological 1/1 BLOCKs.
        return {"propagated": True, "gold_parity": "1/1", "symbolic_forwards": 2,
                "expr_source": "def f(carrier):\n    return bytes(8)\n"}

    res = drive(trace=_sym_ea_items(), case_config=cc,
                triton_runner=runner_forwarded_but_blocks, decisions=DEC_SYM_MEM)
    assert isinstance(res, DriveResult)
    assert len(calls) == 1                             # symex ran ONCE, no re-run
    assert not any(s["step"] == "opaque_forward_retry" for s in res.per_step)
    assert res.closed is False                         # stays a real frontier
    assert res.symbolic_forwards == 2                  # the first-run count is kept


def test_drive_form_R_fallback_still_opaque_yields_frontier_not_false_close():
    # FORM R (degrade still yields a verdict — gates NOT relaxed): the opaque fallback
    # injects + re-runs (forwards on the re-run) but a tautological 1/1 parity still
    # BLOCKs -> NOT closed -> stays a frontier, carrying BOTH forward counts as
    # evidence. The fallback opens the door; the parity gate still decides the close.
    runner, calls = _staging_gated_runner(
        parity="1/1", forwards=1,
        expr="def f(carrier):\n    return bytes(8)\n")   # no independent vectors
    cc = replace(PTR_CC, parity_min=1)
    res = drive(trace=_ptr_indirect_items(), case_config=cc,
                triton_runner=runner, decisions=DEC_PTR)
    assert isinstance(res, DriveResult)
    assert len(calls) == 2                             # fallback DID re-run
    retry = next(s for s in res.per_step if s["step"] == "opaque_forward_retry")
    assert retry["retry_symbolic_forwards"] == 1       # forwarded on the re-run ...
    assert retry["retry_parity_ok"] is False           # ... but parity BLOCKed
    assert res.closed is False                         # frontier, not a false close
    assert res.parity_report and res.parity_report["verdict"] == "BLOCK"


def test_drive_form_S_non_symbolic_address_window_no_fallback_invariant_7():
    # FORM S (invariant 7): a clo_deferred=False window whose diagnosis is NOT
    # symbolic_address (no input-derived / pointer-chain EA) must NOT enter the opaque
    # fallback even when it collapses opaque. drive is byte-for-byte the pre-fallback
    # path: symex runs once, no opaque_forward_retry step. Here the load EAs are all
    # backed concrete bases (no pointer chain, no symbolic EA) -> known_addr /
    # inconclusive, never symbolic_address.
    items = [
        ins(0, 0x1000, "ldr w0, [x16]", reads={"x16": 0x9000}, writes={"w0": 0x55},
            mem=[MemOp("r", 0x9000, 0x55, 4)]),
        ins(1, 0x1004, "mul w0, w0, w1", reads={"w0": 0, "w1": 0}, writes={"w0": 0}),
    ]
    # x16 is a BACKED concrete base (NOT a symbolic input, NOT pointer-derived) ->
    # closure backed -> clo_deferred False; diagnose -> known_addr/inconclusive.
    cc = replace(CC, concrete_backing=build_concrete_backing(reg_values={"x16": 0x9000}),
                 symbolic_regs=("x0", "x1"), reg_file=("x0", "x1", "x16"))
    dec = {**BOTH, "mem_input_symbolize_vs_back": {0x9000: "back"}}
    calls = []

    def collapse_runner(ctx):
        calls.append(1)
        return {"propagated": True, "expr_source": "", "gold_parity": "0/8",
                "symbolic_forwards": 0}

    res = drive(trace=items, case_config=cc, triton_runner=collapse_runner,
                decisions=dec)
    assert isinstance(res, DriveResult)
    assert len(calls) == 1                             # symex ran ONCE (no fallback)
    assert not any(s["step"] == "opaque_forward_retry" for s in res.per_step)
    assert not any(s["step"] == "symbolic_addressing_gate" for s in res.per_step)
    assert res.symbolic_forwards == 0


def _ptr_indirect_with_store_items(store_addr):
    # FORM T backbone: the pointer-indirect window of FORM P PLUS a staging STORE at
    # [x21] that lands the symbol at store_addr. The store's EA base (x21) is what the
    # SELF-DERIVED pointer-chain narrows the store side on (no caller chain). clo of
    # the ldr-[x16] resolves x16 back to backed x20 -> clo_deferred False -> the
    # opaque fallback branch (fb_diag) is the path under test.
    return [
        ins(0, 0x1000, "str x8, [x21]", reads={"x8": 0x41, "x21": store_addr},
            mem=[MemOp("w", store_addr, 0x41, 8)]),
        ins(1, 0x1004, "ldr x16, [x20]", reads={"x20": 0x8000}, writes={"x16": 0x9000},
            mem=[MemOp("r", 0x8000, 0x9000, 8)]),
        ins(2, 0x1008, "ldr w0, [x16]", reads={"x16": 0x9000}, writes={"w0": 0x55},
            mem=[MemOp("r", 0x9000, 0x55, 4)]),
        ins(3, 0x100C, "mul w0, w0, w1", reads={"w0": 0, "w1": 0}, writes={"w0": 0}),
    ]


def test_drive_form_T_self_derives_store_side_no_caller_chain():
    # FORM T — 坎3 CORE SELF-PROOF: the caller passes NO pointer_chain, yet drive
    # self-derives the staging shape from its own diagnosis and narrows the STORE
    # side. The window has a staging store [x21] (its closure is un-backed → the
    # clo_deferred first-run injection runs the SAME derivation as the fallback). The
    # injected set carries (store_addr, 8) resolved from the SELF-DERIVED
    # store_base_regs={x21} — pre-坎3 pointer_chain永远 None ⇒ store side empty ⇒ only
    # the backbone load landing. The caller never touches pointer_chain.
    store_addr = 0xA000
    items = _ptr_indirect_with_store_items(store_addr)
    cc = replace(
        PTR_CC, reg_file=("x0", "x1", "x16", "x20", "x21"),
        symbolic_regs=("x0", "x1", "x16"))
    runner, calls = _staging_gated_runner(
        parity="8/8", forwards=1,
        expr="def f(carrier):\n    return bytes(8)\n", vectors=_FOUR_VECTORS)
    res = drive(trace=items, case_config=cc, triton_runner=runner,
                decisions=DEC_PTR)                     # NO pointer_chain passed
    assert isinstance(res, DriveResult)
    gate = next(s for s in res.per_step if s["step"] == "symbolic_addressing_gate")
    assert gate["staging_verdict"] == "symbolic_address"
    injected = {tuple(iv) for iv in calls[0]["symbolic_staging"]}
    assert (0x9000, 4) in injected                     # backbone (load landing)
    assert (store_addr, 8) in injected                 # SELF-DERIVED store-side narrow


def test_drive_form_T_opaque_fallback_self_derives_store_side():
    # FORM T (fallback path): the SAME self-derivation through the OPAQUE FALLBACK
    # (fb_diag) branch — both the store leg AND the pointer slot are backed, so the
    # static closure is sufficient (clo_deferred False, no first-run gate); the window
    # collapses opaque on the first run, and the fallback re-diagnoses, SELF-DERIVES
    # the shape, and narrows the store side on the RE-RUN. Caller passes no chain.
    store_addr = 0xA000
    items = _ptr_indirect_with_store_items(store_addr)
    cc = replace(
        PTR_CC, reg_file=("x0", "x1", "x16", "x20", "x21"),
        symbolic_regs=("x0", "x1", "x16"),
        concrete_backing=build_concrete_backing(
            reg_values={"x20": 0x8000, "x21": store_addr}))
    runner, calls = _staging_gated_runner(
        parity="8/8", forwards=1,
        expr="def f(carrier):\n    return bytes(8)\n", vectors=_FOUR_VECTORS)
    res = drive(trace=items, case_config=cc, triton_runner=runner,
                decisions=DEC_PTR)                     # NO pointer_chain passed
    assert isinstance(res, DriveResult)
    assert len(calls) == 2                             # collapse + fallback re-run
    assert not any(s["step"] == "symbolic_addressing_gate" for s in res.per_step)
    assert any(s["step"] == "opaque_forward_retry" for s in res.per_step)
    injected = {tuple(iv) for iv in calls[1]["symbolic_staging"]}
    assert (0x9000, 4) in injected                     # backbone (load landing)
    assert (store_addr, 8) in injected                 # SELF-DERIVED store-side narrow


def test_drive_form_T_explicit_pointer_chain_override_wins():
    # FORM T override: an explicit caller pointer_chain OVERRIDES the self-derivation.
    # The override names the store base x21 at a NARROWER store_size=4 → the store
    # landing is resolved at the override's width (store_addr, 4), not the trace op's
    # 8, proving the caller-supplied shape takes precedence over the auto-derived one.
    store_addr = 0xA000
    items = _ptr_indirect_with_store_items(store_addr)
    cc = replace(
        PTR_CC, reg_file=("x0", "x1", "x16", "x20", "x21"),
        symbolic_regs=("x0", "x1", "x16"))
    override = PointerChainSpec(store_base_regs=("x21",), load_base_regs=("x16",),
                                store_size=4)
    runner, calls = _staging_gated_runner(
        parity="8/8", forwards=1,
        expr="def f(carrier):\n    return bytes(8)\n", vectors=_FOUR_VECTORS)
    res = drive(trace=items, case_config=cc, triton_runner=runner,
                decisions=DEC_PTR, pointer_chain=override)
    assert isinstance(res, DriveResult)
    injected = {tuple(iv) for iv in calls[0]["symbolic_staging"]}
    assert (store_addr, 4) in injected                 # override's store_size width
    assert (store_addr, 8) not in injected             # not the auto-derived width
