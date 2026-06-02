"""spec #3 — pre-flight parity / self-check wiring lint.

Pins the GENERAL behaviour (an extensible invariant registry; every finding
carries level+code+detail+fix; degenerate→surfaced; adding an invariant = one
registry entry) and the two TC2 proof-points (SEED_SHAPE_MISMATCH;
OBSERVED_SEMANTICS), plus the NON-BLOCKING drive() hook surfacing into per_step
and the lint-OK byte-for-byte regression.

Synthetic shapes only — no case coordinates / handler numbers / target names
beyond the inert smoke fixture shared with the drive tests.
"""

from __future__ import annotations

from dataclasses import replace

from engine.setup_symex import (
    CaseConfig,
    DriveResult,
    INVARIANTS,
    Invariant,
    LintFinding,
    LintReport,
    build_concrete_backing,
    drive,
    lint_case_config,
    lint_parity_inputs,
    register_invariant,
)
from engine.setup_symex._lint import LintContext
from engine.types import Instruction


# --------------------------------------------------------------------------- #
# Shared inert fixtures (mirrors test_setup_symex_drive so the regression test
# compares against that path's behaviour).
# --------------------------------------------------------------------------- #

def _ins(idx, pc, mnem, *, reads=None, writes=None):
    return Instruction(idx=idx, pc=pc, bytes_=b"\x00\x00\x00\x00", mnemonic=mnem,
                       regs_read=reads or {}, regs_write=writes or {}, mem=())


def _items():
    return [
        _ins(0, 0x1000, "ldr w0, [x16]", reads={"x16": 0x9000}),
        _ins(1, 0x1004, "mul w0, w0, w1", reads={"w0": 0, "w1": 0}, writes={"w0": 0}),
    ]


CC = CaseConfig(
    target="libEncryptor.so", input_hash="ab12", run_id="run-1",
    seed_hint_addr=0x100, sink_hint_addr=0x200, entry_pc=0x0FFF,
    window=(0x1000, 0x10FF), reg_file=("x0", "x1", "x16"),
    inputs=("carrier",), parity_min=8, symbolic_regs=("x0", "x1"),
    concrete_backing=build_concrete_backing(reg_values={"x16": 0x9000}),
    task="lint_smoke",
)


def _runner(_ctx):
    return {"propagated": True, "gold_parity": "8/8",
            "expr_source": "def f(carrier):\n    return bytes(8)\n"}


BOTH = {"alias_vs_compute": "compute", "which_static": []}


# A WELL-WIRED set of runner-side descriptors: one seed per declared input,
# sink from this window's exit, observed = window-local sink, mask matches width.
def _clean_specs(declared=("carrier",)):
    return dict(
        declared_inputs=declared,
        sink_spec={"source": "window_exit",
                   "seed_values": {n: 0x11223344 for n in declared},
                   "mask": 0xFFFFFFFF, "reg_width": 4, "form": "reg"},
        observed_spec={"source": "window_local_sink"},
    )


# --------------------------------------------------------------------------- #
# General contract: report shape + every finding carries level/code/detail/fix.
# --------------------------------------------------------------------------- #

def test_clean_config_is_ok():
    rep = lint_parity_inputs(CC, **_clean_specs())
    assert isinstance(rep, LintReport)
    assert rep.max_level == "OK" and rep.ok and not rep.has_error
    assert rep.findings == ()
    assert rep.to_dict()["max_level"] == "OK"


def test_every_finding_carries_level_code_detail_fix():
    # Trip several invariants at once and assert the universal shape.
    rep = lint_parity_inputs(
        CC,
        declared_inputs=("carrier",),
        sink_spec={"source": "downstream_call",
                   "seed_values": {"other": 1},
                   "mask": 0xFFFF, "reg_width": 4, "form": "reg"},
        observed_spec={"source": "top_level_output"})
    assert rep.findings
    for f in rep.findings:
        assert f.level in ("ERROR", "WARN", "INFO")
        assert f.code and f.detail and f.fix
        assert isinstance(f, LintFinding)


# --------------------------------------------------------------------------- #
# TC2 proof-point (a): the two misconfigs trip the right ERROR + fix text.
# --------------------------------------------------------------------------- #

def test_tc2_phase_D_seed_shape_mismatch():
    # phase_D shape: the self-check was fed a seed for the WRONG input set.
    specs = _clean_specs()
    specs["sink_spec"] = {**specs["sink_spec"],
                          "seed_values": {"wrong_input": 0x11223344}}
    rep = lint_parity_inputs(CC, **specs)
    errs = [f for f in rep.findings if f.code == "SEED_SHAPE_MISMATCH"
            and f.level == "ERROR"]
    assert len(errs) == 1
    assert rep.has_error and rep.max_level == "ERROR"
    e = errs[0]
    assert "carrier" in e.detail and "wrong_input" in e.detail
    assert "declared inputs" in e.fix and "case_config.inputs" in e.fix


def test_tc2_phase_C_observed_semantics():
    # phase_C shape: parity took the wrong observed semantics (top-level output).
    specs = _clean_specs()
    specs["observed_spec"] = {"source": "top_level_output"}
    rep = lint_parity_inputs(CC, **specs)
    errs = [f for f in rep.findings if f.code == "OBSERVED_SEMANTICS"
            and f.level == "ERROR"]
    assert len(errs) == 1
    e = errs[0]
    assert "top_level_output" in e.detail and "window's local sink" in e.detail
    assert "window_local_sink" in e.fix


# --------------------------------------------------------------------------- #
# The other two seed invariants.
# --------------------------------------------------------------------------- #

def test_sink_source_not_window_exit_errors():
    specs = _clean_specs()
    specs["sink_spec"] = {**specs["sink_spec"], "source": "global_buffer"}
    rep = lint_parity_inputs(CC, **specs)
    errs = rep.by_code("SINK_SOURCE")
    assert len(errs) == 1 and errs[0].level == "ERROR"
    assert "global_buffer" in errs[0].detail
    assert "window_exit" in errs[0].fix


def test_sink_mask_width_mismatch_errors():
    specs = _clean_specs()
    # 16-bit mask declared against a 4-byte register.
    specs["sink_spec"] = {**specs["sink_spec"], "mask": 0xFFFF, "reg_width": 4}
    rep = lint_parity_inputs(CC, **specs)
    errs = rep.by_code("SINK_MASK_WIDTH")
    assert len(errs) == 1 and errs[0].level == "ERROR"
    assert "0xffff" in errs[0].detail and "4 byte" in errs[0].detail


def test_seed_width_overflow_warns():
    # carrier declared 1 byte but seeded with a 4-byte value → WARN (truncation).
    specs = _clean_specs()
    specs["declared_inputs"] = {"carrier": 1}
    rep = lint_parity_inputs(CC, **specs)
    warns = [f for f in rep.by_code("SEED_SHAPE_MISMATCH") if f.level == "WARN"]
    assert len(warns) == 1
    assert rep.max_level == "WARN" and not rep.has_error


# --------------------------------------------------------------------------- #
# A8 check 4: degenerate (missing descriptor) → surfaced INFO/WARN, never a
# silent clean pass.
# --------------------------------------------------------------------------- #

def test_missing_descriptors_surface_info_not_silent_ok():
    rep = lint_parity_inputs(CC)  # no sink_spec / observed_spec at all
    assert rep.max_level == "INFO" and not rep.ok and not rep.has_error
    codes = {f.code for f in rep.findings}
    assert {"SEED_SHAPE_MISMATCH", "SINK_SOURCE",
            "OBSERVED_SEMANTICS", "SINK_MASK_WIDTH"} <= codes
    assert all(f.level == "INFO" for f in rep.findings)


def test_missing_source_field_warns_not_clean():
    specs = _clean_specs()
    specs["sink_spec"] = {k: v for k, v in specs["sink_spec"].items()
                          if k != "source"}
    rep = lint_parity_inputs(CC, **specs)
    warns = rep.by_code("SINK_SOURCE")
    assert len(warns) == 1 and warns[0].level == "WARN"


# --------------------------------------------------------------------------- #
# lint_case_config convenience wrapper: folds a bare seed_values into sink_spec.
# --------------------------------------------------------------------------- #

def test_lint_case_config_folds_seed_values():
    rep = lint_case_config(
        CC,
        seed_values={"carrier": 0x11223344},
        sink_spec={"source": "window_exit", "mask": 0xFFFFFFFF, "reg_width": 4},
        observed_spec={"source": "window_local_sink"})
    assert rep.max_level == "OK"
    # and the mismatch is caught through the wrapper too.
    bad = lint_case_config(
        CC, seed_values={"nope": 1},
        sink_spec={"source": "window_exit"},
        observed_spec={"source": "window_local_sink"})
    assert bad.by_code("SEED_SHAPE_MISMATCH")[0].level == "ERROR"


# --------------------------------------------------------------------------- #
# (c) GENERALITY: a NEW invariant added via the registry fires with no other
# code change.
# --------------------------------------------------------------------------- #

def test_new_invariant_via_registry_fires(monkeypatch):
    # An author adds a wiring invariant as ONE registry entry.
    def _inv_entry_pc_set(ctx: LintContext):
        cc = ctx.case_config
        if cc is not None and getattr(cc, "entry_pc", None) == 0:
            yield LintFinding(
                "ERROR", "ENTRY_PC_UNSET",
                "entry_pc is 0 — the window has no symbolic entry anchor.",
                "set case_config.entry_pc to the window's real entry PC.")

    new_inv = Invariant("ENTRY_PC_UNSET", _inv_entry_pc_set, "entry_pc must be set")

    # Snapshot + restore the global registry so the test is isolated.
    saved = list(INVARIANTS)
    try:
        register_invariant(new_inv)
        assert any(i.code == "ENTRY_PC_UNSET" for i in INVARIANTS)
        bad_cc = replace(CC, entry_pc=0)
        rep = lint_parity_inputs(bad_cc, **_clean_specs())
        fired = rep.by_code("ENTRY_PC_UNSET")
        assert len(fired) == 1 and fired[0].level == "ERROR"
        # a good config does not trip the new invariant.
        ok = lint_parity_inputs(CC, **_clean_specs())
        assert not ok.by_code("ENTRY_PC_UNSET") and ok.max_level == "OK"
    finally:
        INVARIANTS[:] = saved


def test_a_buggy_invariant_is_surfaced_not_crashed():
    def _boom(ctx: LintContext):
        raise RuntimeError("invariant defect")
        yield  # pragma: no cover

    rep = lint_parity_inputs(
        CC, **_clean_specs(),
        invariants=[Invariant("BOOM", _boom, "always raises")])
    assert rep.max_level == "WARN"
    assert rep.by_code("BOOM")[0].level == "WARN"
    assert "UNVERIFIED" in rep.by_code("BOOM")[0].fix


# --------------------------------------------------------------------------- #
# drive() hook: NON-BLOCKING; loud findings ride per_step; lint-OK is silent.
# --------------------------------------------------------------------------- #

def test_drive_hook_silent_when_lint_clean():
    # The smoke CC has no runner descriptors at drive time → INFO-only report →
    # the lint step appends NOTHING (no parity_wiring_lint entry in per_step).
    res = drive(trace=_items(), case_config=CC, triton_runner=_runner,
                decisions=BOTH)
    assert isinstance(res, DriveResult)
    steps = [s.get("step") for s in res.per_step]
    assert "parity_wiring_lint" not in steps
    # locate_boundary still the first recorded step (hook is non-disruptive).
    assert steps[0] == "locate_boundary"


def test_drive_byte_for_byte_unchanged_vs_no_lint(monkeypatch):
    # The lint-OK path must leave drive()'s per_step byte-for-byte unchanged.
    # Run once normally, then with the lint forced to OK, and compare per_step.
    res_normal = drive(trace=_items(), case_config=CC, triton_runner=_runner,
                       decisions=BOTH)

    import engine.setup_symex._driver as drv
    monkeypatch.setattr(drv, "lint_case_config",
                        lambda *_a, **_k: LintReport(findings=()))
    res_forced_ok = drive(trace=_items(), case_config=CC, triton_runner=_runner,
                          decisions=BOTH)

    assert list(res_normal.per_step) == list(res_forced_ok.per_step)


def test_drive_hook_surfaces_loud_finding_in_per_step(monkeypatch):
    # Force a loud (ERROR) lint report and assert drive surfaces it in per_step
    # WITHOUT aborting the run (non-blocking).
    loud = LintReport(findings=(
        LintFinding("ERROR", "OBSERVED_SEMANTICS",
                    "observed is the top-level output, not the window sink.",
                    "wire observed to the window-local sink."),))
    import engine.setup_symex._driver as drv
    monkeypatch.setattr(drv, "lint_case_config", lambda *_a, **_k: loud)
    res = drive(trace=_items(), case_config=CC, triton_runner=_runner,
                decisions=BOTH)
    assert isinstance(res, DriveResult)  # NON-BLOCKING — run still completes
    lint_steps = [s for s in res.per_step if s.get("step") == "parity_wiring_lint"]
    assert len(lint_steps) == 1
    assert lint_steps[0]["max_level"] == "ERROR"
    assert lint_steps[0]["findings"][0]["code"] == "OBSERVED_SEMANTICS"
    assert lint_steps[0]["findings"][0]["fix"]
