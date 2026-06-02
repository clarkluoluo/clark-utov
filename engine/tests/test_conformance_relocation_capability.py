"""Conformance robustness: A relocation-aware anchor reconcile + B C3 capability-aware.

Origin: the reference case (libEncryptor) — runner metadata reports algo_entry_pc
0x40007d88 but unidbg rebased the module so the trace's first_pc is 0x12007d88
(a clean page-aligned base delta). Pre-fix C4 FAILed on that difference and C3
FAILed downstream (observe hook installed at the wrong base → empty observations).
"""

from __future__ import annotations

from pathlib import Path

from engine.conformance import (
    OBSERVE_CAPABILITY,
    CheckId,
    CheckResult,
    Relocation,
    _c3_observe_point,
    _c4_trace_integrity,
    _peek_trace_bounds,
    detect_relocation,
    rebase_meta,
    rebase_pc,
    run_conformance,
)
from engine.runner_client import ObservedState, ObservePoint, RerunResult, RunnerAdapter
from engine.types import Instruction, TargetMeta

# The reference case anchors: static metadata vs unidbg-rebased trace.
_META_ENTRY = 0x40007D88
_META_EXIT = 0x40008000
_RUN_ENTRY = 0x12007D88
_RUN_EXIT = 0x12008000
_BASE_DELTA = _RUN_ENTRY - _META_ENTRY            # -0x2E000000 (module loaded lower)


def _meta(entry=_META_ENTRY, exit_=_META_EXIT, capabilities=()):
    return TargetMeta(
        target_name="libEncryptor.so", arch="arm64",
        algo_entry_pc=entry, algo_exit_pc=exit_,
        input_length=None, output_length=8, capabilities=tuple(capabilities))


def _ins(idx, pc, mnem="nop"):
    return Instruction(idx=idx, pc=pc, bytes_=b"\x00\x00\x00\x00", mnemonic=mnem,
                       regs_read={}, regs_write={}, mem=())


class _Reader:
    """Re-iterable trace reader (fresh generator each iter — like the real ones)."""

    def __init__(self, pcs_mnems):
        self._rows = list(pcs_mnems)

    def __iter__(self):
        for i, (pc, mnem) in enumerate(self._rows):
            yield _ins(i, pc, mnem)


def _relocated_reader():
    # 12 steps so C5's >= 10-line baseline is satisfied if reused; first at the
    # rebased entry, last a ret at the rebased exit.
    rows = [(_RUN_ENTRY + 4 * i, "nop") for i in range(11)]
    rows.append((_RUN_EXIT, "ret"))
    return _Reader(rows)


# --- A. detect_relocation -----------------------------------------------------

def test_detect_relocation_clean_rebase():
    reloc = detect_relocation(_meta(), _RUN_ENTRY, _RUN_EXIT)
    assert reloc is not None
    assert reloc.base_delta == _BASE_DELTA
    assert reloc.entry_from == _META_ENTRY and reloc.entry_to == _RUN_ENTRY
    assert reloc.exit_from == _META_EXIT and reloc.exit_to == _RUN_EXIT
    # the LOUD record carries the magnitude + from/to
    d = reloc.to_dict()
    assert d["entry"] == {"from": "0x40007d88", "to": "0x12007d88"}


def test_detect_relocation_idempotent_no_shift():
    # delta 0 → no relocation, behaviour unchanged.
    assert detect_relocation(_meta(), _META_ENTRY, _META_EXIT) is None


def test_detect_relocation_rejects_disagreeing_deltas():
    # entry shifts by the base delta but exit does NOT → not a clean rebase →
    # None (so C4 still FAILs — a real wrong anchor is never reconciled away).
    assert detect_relocation(_meta(), _RUN_ENTRY, _META_EXIT + 0x40) is None


def test_detect_relocation_rejects_unaligned_delta():
    # Same delta for both, but it is NOT page-aligned (page offset shifts) →
    # not a clean module rebase → None.
    bad = 0x12007D90  # entry off by 0x2E000008 — low 12 bits differ
    bad_exit = _META_EXIT + (bad - _META_ENTRY)
    assert detect_relocation(_meta(), bad, bad_exit) is None


def test_rebase_helpers_are_idempotent_without_reloc():
    m = _meta()
    assert rebase_meta(m, None) is m
    assert rebase_pc(0x1234, None) == 0x1234
    reloc = detect_relocation(m, _RUN_ENTRY, _RUN_EXIT)
    assert rebase_pc(_META_ENTRY, reloc) == _RUN_ENTRY


# --- A. C4 trace integrity under relocation -----------------------------------

def test_c4_passes_relocation_reconciled():
    bounds = _peek_trace_bounds(_relocated_reader())
    reloc = detect_relocation(_meta(), bounds.first_pc, bounds.last_pc)
    eff = rebase_meta(_meta(), reloc)
    rec = _c4_trace_integrity(bounds, eff, reloc=reloc)
    assert rec.result == CheckResult.PASS
    assert rec.detail["relocation_detected"]["base_delta"] == f"0x{_BASE_DELTA:x}"
    assert "relocation-reconciled" in rec.detail["note"]


def test_c4_no_reloc_pass_detail_unchanged():
    # Regression: a non-relocated matching trace yields the exact same PASS keys
    # as before (no relocation_detected / note noise).
    reader = _Reader([(_META_ENTRY, "nop"), (_META_EXIT, "ret")])
    bounds = _peek_trace_bounds(reader)
    rec = _c4_trace_integrity(bounds, _meta(), reloc=None)
    assert rec.result == CheckResult.PASS
    assert set(rec.detail) == {"first_pc", "last_pc", "last_mnem", "instr_count"}


def test_c4_wrong_anchor_still_fails():
    # first_pc differs and it is NOT a clean rebase → FAIL (detect returns None).
    reader = _Reader([(0x5000, "nop"), (0x6000, "add")])
    bounds = _peek_trace_bounds(reader)
    reloc = detect_relocation(_meta(), bounds.first_pc, bounds.last_pc)
    assert reloc is None
    rec = _c4_trace_integrity(bounds, _meta(), reloc=reloc)
    assert rec.result == CheckResult.FAIL


# --- B. C3 capability-aware ---------------------------------------------------

class _ObserveRunner(RunnerAdapter):
    """Live-shaped runner. Observes only at ``live_entry`` (proves rebase). The
    declared capability set and whether it returns observations are configurable
    to exercise every C3 branch."""

    def __init__(self, *, live_entry, capabilities=(), emit_obs=True,
                 meta_entry=_META_ENTRY, meta_exit=_META_EXIT, trace_path=None):
        self._live_entry = live_entry
        self._caps = tuple(capabilities)
        self._emit = emit_obs
        self._meta_entry = meta_entry
        self._meta_exit = meta_exit
        self._trace_path = trace_path

    def metadata(self):
        return _meta(self._meta_entry, self._meta_exit, self._caps)

    def rerun(self, input_bytes, observe_points=None):
        output = bytes((b ^ 0xAB) for b in input_bytes)   # deterministic + input-sensitive
        obs = ()
        if self._emit and observe_points:
            obs = tuple(
                ObservedState(pc=op.pc, when=op.when, regs={"x0": 0x1234}, mem={})
                for op in observe_points if op.pc == self._live_entry
            )
        return RerunResult(output=output, observations=obs)

    def get_trace(self, input_bytes, start, end):
        return str(self._trace_path)


def test_c3_no_capability_declared_degrades_not_fail():
    # PC reconciled (Part A), observe still empty, runner declares NO observe
    # capability → SKIP + verifier_degraded (a capability gap, not a failure).
    runner = _ObserveRunner(live_entry=_RUN_ENTRY, capabilities=(), emit_obs=False)
    eff = rebase_meta(_meta(), detect_relocation(_meta(), _RUN_ENTRY, _RUN_EXIT))
    rec = _c3_observe_point(runner, b"\x00" * 8, eff,
                            reloc=detect_relocation(_meta(), _RUN_ENTRY, _RUN_EXIT))
    assert rec.result == CheckResult.SKIP
    assert rec.detail["verifier_degraded"] is True
    assert rec.detail["declared"] is False
    assert rec.detail["capability"] == OBSERVE_CAPABILITY
    assert "relocation-reconciled" in rec.detail["diagnosis"]


def test_c3_declared_but_empty_fails():
    # Runner DECLARES observe capability yet returns empty → real broken hook FAIL.
    runner = _ObserveRunner(live_entry=_RUN_ENTRY, capabilities=(OBSERVE_CAPABILITY,),
                            emit_obs=False)
    rec = _c3_observe_point(runner, b"\x00" * 8, _meta(capabilities=(OBSERVE_CAPABILITY,)))
    assert rec.result == CheckResult.FAIL
    assert rec.detail["declared"] is True


def test_c3_passes_when_observe_hits_reconciled_pc():
    # Observe only fires at the rebased entry — proves the reconcile made C3 hit.
    runner = _ObserveRunner(live_entry=_RUN_ENTRY, emit_obs=True)
    reloc = detect_relocation(_meta(), _RUN_ENTRY, _RUN_EXIT)
    eff = rebase_meta(_meta(), reloc)
    rec = _c3_observe_point(runner, b"\x00" * 8, eff, reloc=reloc)
    assert rec.result == CheckResult.PASS
    assert rec.detail["relocation_reconciled"] is True
    assert rec.detail["regs_captured"] == 1


def test_c3_without_reconcile_would_miss_then_pass_with():
    # Same runner, but if we DON'T rebase, observe is at the wrong (metadata) PC →
    # empty → not the capability's fault. With rebase it hits. This is the exact
    # The reference case regression the reconcile fixes.
    runner = _ObserveRunner(live_entry=_RUN_ENTRY, capabilities=(OBSERVE_CAPABILITY,),
                            emit_obs=True)
    cap_meta = _meta(capabilities=(OBSERVE_CAPABILITY,))
    miss = _c3_observe_point(runner, b"\x00" * 8, cap_meta)   # un-rebased → wrong PC → empty → FAIL
    assert miss.result == CheckResult.FAIL
    reloc = detect_relocation(cap_meta, _RUN_ENTRY, _RUN_EXIT)
    hit = _c3_observe_point(runner, b"\x00" * 8, rebase_meta(cap_meta, reloc), reloc=reloc)
    assert hit.result == CheckResult.PASS


# --- end-to-end run_conformance ----------------------------------------------

def _trace_file(tmp_path: Path) -> Path:
    p = tmp_path / "trace.txt"
    p.write_text("\n".join(f"line{i}" for i in range(20)))   # >= 10 lines for C5
    return p


def test_run_conformance_relocation_end_to_end(tmp_path):
    runner = _ObserveRunner(live_entry=_RUN_ENTRY, capabilities=(OBSERVE_CAPABILITY,),
                            emit_obs=True, trace_path=_trace_file(tmp_path))
    report = run_conformance(runner, _relocated_reader(), probe_input=b"\x01\x02\x03\x04")
    assert report.mode == "live"
    # relocation surfaced at the verdict level (LOUD, not silent).
    assert report.relocation is not None
    assert report.relocation["base_delta"] == f"0x{_BASE_DELTA:x}"
    by = {c.check: c for c in report.checks}
    assert by[CheckId.C4].result == CheckResult.PASS          # reconciled, not raw match
    assert by[CheckId.C3].result == CheckResult.PASS          # observe hit at rebased PC
    assert report.overall == CheckResult.PASS


def test_run_conformance_capability_gap_degrades_not_fail(tmp_path):
    # Live runner, no observe capability, empty observations → C3 SKIP+degraded;
    # the gate does NOT fail on a declared capability gap.
    runner = _ObserveRunner(live_entry=_RUN_ENTRY, capabilities=(), emit_obs=False,
                            trace_path=_trace_file(tmp_path))
    report = run_conformance(runner, _relocated_reader(), probe_input=b"\x01\x02\x03\x04")
    by = {c.check: c for c in report.checks}
    assert by[CheckId.C3].result == CheckResult.SKIP
    assert report.verifier_degraded is True
    assert report.overall != CheckResult.FAIL                 # capability gap ≠ failure


# --- C3 MEM round-trip + C7 generic capability round-trip --------------------

from engine.conformance import (
    MEM_CAPTURE_CAPABILITY,
    _c3_observe_point,
    _c7_capability_round_trip,
    _mem_round_trip_probe,
)

_SP = 0xBFFFF700   # a valid stack address the runner reports in regs


class _MemRunner(RunnerAdapter):
    """Live-shaped runner that observes regs (incl. sp) and optionally echoes a
    requested mem range. ``mem_ok`` toggles whether the mem-capture wire works."""

    def __init__(self, *, capabilities=(), mem_ok=True, entry=_META_ENTRY):
        self._caps = tuple(capabilities)
        self._mem_ok = mem_ok
        self._entry = entry

    def metadata(self):
        return _meta(self._entry, _META_EXIT, self._caps)

    def rerun(self, input_bytes, observe_points=None):
        output = bytes((b ^ 0xAB) for b in input_bytes)
        obs = []
        for op in observe_points or ():
            mem = {}
            if "mem" in op.capture and self._mem_ok:
                # echo the requested ranges back (the round-trip the gate demands)
                for (addr, size) in op.mem:
                    mem[addr] = bytes(size)
            obs.append(ObservedState(pc=op.pc, when=op.when,
                                     regs={"sp": _SP, "x0": 0x1234}, mem=mem))
        return RerunResult(output=output, observations=tuple(obs))

    def get_trace(self, input_bytes, start, end):
        return ""


def test_mem_round_trip_probe_ok_when_echoed():
    runner = _MemRunner(capabilities=(MEM_CAPTURE_CAPABILITY,), mem_ok=True)
    ok, detail, status = _mem_round_trip_probe(runner, _meta())
    assert ok is True and status == "ok"
    assert detail["probe_addr"] == f"0x{_SP:x}"


def test_mem_round_trip_probe_broken_when_not_echoed():
    # Declares it works but returns no mem bytes → broken wire (the Bug1 class).
    runner = _MemRunner(capabilities=(MEM_CAPTURE_CAPABILITY,), mem_ok=False)
    ok, detail, status = _mem_round_trip_probe(runner, _meta())
    assert ok is False and status == "broken"
    assert "mem-capture wire is broken" in detail["diagnosis"]


def test_c3_passes_regs_and_mem_round_trip():
    runner = _MemRunner(capabilities=(OBSERVE_CAPABILITY, MEM_CAPTURE_CAPABILITY),
                        mem_ok=True)
    rec = _c3_observe_point(runner, b"\x00" * 8,
                            _meta(capabilities=(OBSERVE_CAPABILITY, MEM_CAPTURE_CAPABILITY)))
    assert rec.result == CheckResult.PASS
    assert rec.detail["mem_round_trip"]["status"] == "ok"


def test_c3_fails_when_mem_declared_but_broken():
    runner = _MemRunner(capabilities=(OBSERVE_CAPABILITY, MEM_CAPTURE_CAPABILITY),
                        mem_ok=False)
    rec = _c3_observe_point(runner, b"\x00" * 8,
                            _meta(capabilities=(OBSERVE_CAPABILITY, MEM_CAPTURE_CAPABILITY)))
    assert rec.result == CheckResult.FAIL
    assert rec.detail["mem_round_trip"]["status"] == "broken"


def test_c3_passes_when_mem_not_declared():
    # No mem-capture declared → MEM round-trip degrades, regs alone keep C3 PASS.
    runner = _MemRunner(capabilities=(OBSERVE_CAPABILITY,), mem_ok=False)
    rec = _c3_observe_point(runner, b"\x00" * 8, _meta(capabilities=(OBSERVE_CAPABILITY,)))
    assert rec.result == CheckResult.PASS
    assert rec.detail["mem_round_trip"]["declared"] is False
    assert rec.detail["mem_round_trip"]["status"] == "degraded"


def test_c7_round_trips_every_declared_capability():
    runner = _MemRunner(capabilities=(OBSERVE_CAPABILITY, MEM_CAPTURE_CAPABILITY),
                        mem_ok=True)
    recs = _c7_capability_round_trip(runner, _meta(
        capabilities=(OBSERVE_CAPABILITY, MEM_CAPTURE_CAPABILITY)))
    caps = {r.detail["capability"]: r for r in recs}
    assert set(caps) == {OBSERVE_CAPABILITY, MEM_CAPTURE_CAPABILITY}
    assert all(r.result == CheckResult.PASS for r in recs)


def test_c7_loud_fail_declared_but_not_wired():
    # The recurring class: capability DECLARED yet round-trip returns nothing.
    runner = _MemRunner(capabilities=(MEM_CAPTURE_CAPABILITY,), mem_ok=False)
    recs = _c7_capability_round_trip(runner, _meta(capabilities=(MEM_CAPTURE_CAPABILITY,)))
    assert len(recs) == 1
    assert recs[0].result == CheckResult.FAIL
    assert recs[0].detail["capability"] == MEM_CAPTURE_CAPABILITY


def test_c7_undeclared_but_working_surfaces_not_silent():
    # spec MEM-cap: a capability the runner does NOT declare but that ACTUALLY
    # round-trips must NOT be silently dropped — C7 surfaces a LOUD
    # capability_undeclared_but_working PASS (the criterion is the probe, not the
    # declaration). mem_ok=True + mem_capture undeclared.
    runner = _MemRunner(capabilities=(OBSERVE_CAPABILITY,), mem_ok=True)
    recs = _c7_capability_round_trip(runner, _meta(capabilities=(OBSERVE_CAPABILITY,)))
    caps = {r.detail["capability"]: r for r in recs}
    assert set(caps) == {OBSERVE_CAPABILITY, MEM_CAPTURE_CAPABILITY}
    mem = caps[MEM_CAPTURE_CAPABILITY]
    assert mem.result == CheckResult.PASS                 # treated as available
    assert mem.detail["declared"] is False
    assert mem.detail["inconsistency"] == "capability_undeclared_but_working"
    assert "ACTUALLY ROUND-TRIPS" in mem.detail["warn"]


def test_c7_undeclared_and_unavailable_not_asserted():
    # Undeclared AND does not round-trip → genuinely not this runner's capability;
    # C7 makes no assertion (no FAIL against an undeclared cap, no silent PASS).
    runner = _MemRunner(capabilities=(OBSERVE_CAPABILITY,), mem_ok=False)
    recs = _c7_capability_round_trip(runner, _meta(capabilities=(OBSERVE_CAPABILITY,)))
    caps = {r.detail["capability"] for r in recs}
    assert caps == {OBSERVE_CAPABILITY}                # mem_capture not asserted
    assert all(r.result == CheckResult.PASS for r in recs)


def test_c3_undeclared_but_working_runs_mem_with_loud_record():
    # spec MEM-cap CORE: MEM observe ACTUALLY works but mem_capture is undeclared.
    # The runner used to silently SKIP C3-mem; now it RUNS the round-trip and
    # surfaces a LOUD capability_undeclared_but_working record (never silent SKIP).
    runner = _MemRunner(capabilities=(OBSERVE_CAPABILITY,), mem_ok=True)
    rec = _c3_observe_point(runner, b"\x00" * 8, _meta(capabilities=(OBSERVE_CAPABILITY,)))
    assert rec.result == CheckResult.PASS
    mr = rec.detail["mem_round_trip"]
    assert mr["declared"] is False
    assert mr["status"] == "ok"                         # round-trip actually ran
    assert mr["capability_state"] == "undeclared_working"
    assert mr["inconsistency"] == "capability_undeclared_but_working"
    assert mr["mem_skip_reason"] == "undeclared_working"
    assert "ACTUALLY ROUND-TRIPS" in mr["warn"]


def test_c3_undeclared_and_unavailable_degrades_with_reason():
    # Undeclared AND the probe does not round-trip → genuine capability gap →
    # degrade WITH a surfaced WARN + mem_skip_reason=probe_unavailable, never a
    # silent empty (spec MEM-cap "truly unavailable").
    runner = _MemRunner(capabilities=(OBSERVE_CAPABILITY,), mem_ok=False)
    rec = _c3_observe_point(runner, b"\x00" * 8, _meta(capabilities=(OBSERVE_CAPABILITY,)))
    assert rec.result == CheckResult.PASS               # degrade, not FAIL
    mr = rec.detail["mem_round_trip"]
    assert mr["declared"] is False
    assert mr["status"] == "degraded"
    assert mr["mem_skip_reason"] == "probe_unavailable"
    assert mr["capability_state"] == "undeclared_unavailable"
    assert "warn" in mr
    assert rec.detail.get("verifier_degraded") is True


def test_c7_in_full_run_fails_gate_on_broken_capability(tmp_path):
    # End-to-end: a declared-but-broken mem-capture must FAIL the whole gate.
    runner = _MemRunner(capabilities=(OBSERVE_CAPABILITY, MEM_CAPTURE_CAPABILITY),
                        mem_ok=False, entry=_META_ENTRY)

    class _R(_MemRunner):
        def get_trace(self, input_bytes, start, end):
            return str(_trace_file(tmp_path))

    runner = _R(capabilities=(OBSERVE_CAPABILITY, MEM_CAPTURE_CAPABILITY), mem_ok=False)
    # Non-relocated trace so C4 passes on the metadata anchors.
    reader = _Reader([(_META_ENTRY + 4 * i, "nop") for i in range(11)] + [(_META_EXIT, "ret")])
    report = run_conformance(runner, reader, probe_input=b"\x01\x02\x03\x04")
    by_results = [c for c in report.checks if c.check == CheckId.C7]
    assert any(c.result == CheckResult.FAIL for c in by_results)
    assert report.overall == CheckResult.FAIL
