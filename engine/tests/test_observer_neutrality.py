"""C6 observer-neutrality check — Tier-1 detection for the "observer perturbs
the observed path" class (color-oracle vs guard-init-hook regression).

The engine does not toggle the observer flag; it runs two regimes (baseline /
instrumented) and reports divergence. These tests drive it with stub adapters.
"""

from __future__ import annotations

from engine.conformance import (
    CheckId,
    CheckResult,
    check_observer_neutrality,
    check_observer_set_matrix,
)
from engine.runner_client import RerunResult, RunnerAdapter


class _StubRunner(RunnerAdapter):
    def __init__(self, *, output: bytes | None = None,
                 raise_exc: Exception | None = None, file_mode: bool = False):
        self._output = output
        self._raise = raise_exc
        self._file = file_mode

    def metadata(self):
        return None  # C6 never reads metadata

    def rerun(self, input_bytes, observe_points=None):
        if self._file:
            raise NotImplementedError
        if self._raise is not None:
            raise self._raise
        return RerunResult(output=self._output or b"")


PROBE = b"\x01\x02\x03\x04"


def test_neutral_observer_passes():
    off = _StubRunner(output=b"PASS-129")
    on = _StubRunner(output=b"PASS-129")
    rec = check_observer_neutrality(off, on, PROBE)
    assert rec.check is CheckId.C6
    assert rec.result is CheckResult.PASS
    assert rec.detail["output_len"] == 8


def test_instrumented_fault_while_baseline_succeeds_is_the_color_oracle_case():
    off = _StubRunner(output=b"PASS[A-bridge] len=129")
    on = _StubRunner(raise_exc=RuntimeError("WRITE_UNMAPPED @0x61924 -> NULL"))
    rec = check_observer_neutrality(off, on, PROBE)
    assert rec.result is CheckResult.FAIL
    assert rec.detail["faulting_regime"] == "observers_on"
    assert rec.detail["clean_regime"] == "observers_off"
    assert "WRITE_UNMAPPED" in rec.detail["error"]
    assert "perturbs" in rec.detail["diagnosis"]


def test_baseline_fault_while_instrumented_succeeds_also_fails():
    # symmetric: whichever side diverges, the perturbation is flagged
    off = _StubRunner(raise_exc=RuntimeError("boom"))
    on = _StubRunner(output=b"ok")
    rec = check_observer_neutrality(off, on, PROBE)
    assert rec.result is CheckResult.FAIL
    assert rec.detail["faulting_regime"] == "observers_off"


def test_diverging_output_fails_with_first_diff_offset():
    off = _StubRunner(output=b"AAAA-good")
    on = _StubRunner(output=b"AAAB-good")
    rec = check_observer_neutrality(off, on, PROBE)
    assert rec.result is CheckResult.FAIL
    assert rec.detail["first_diff_offset"] == 3
    assert rec.detail["baseline_sha16"] != rec.detail["instrumented_sha16"]


def test_both_file_mode_skips():
    off = _StubRunner(file_mode=True)
    on = _StubRunner(file_mode=True)
    rec = check_observer_neutrality(off, on, PROBE)
    assert rec.result is CheckResult.SKIP


def test_one_file_mode_skips():
    off = _StubRunner(output=b"x")
    on = _StubRunner(file_mode=True)
    rec = check_observer_neutrality(off, on, PROBE)
    assert rec.result is CheckResult.SKIP
    assert "one regime" in rec.detail["reason"]


def test_custom_regime_labels_surface_in_detail():
    off = _StubRunner(output=b"a")
    on = _StubRunner(output=b"b")
    rec = check_observer_neutrality(
        off, on, PROBE,
        label_baseline="clean", label_instrumented="guard_evidence",
    )
    assert rec.detail["baseline"] == "clean"
    assert rec.detail["instrumented"] == "guard_evidence"


def test_matrix_validates_a_set_of_named_observer_sets_at_once():
    baseline = _StubRunner(output=b"PASS-129")
    regimes = {
        "guard_evidence": _StubRunner(
            raise_exc=RuntimeError("WRITE_UNMAPPED @0x61924 -> NULL")),
        "minimal_probe": _StubRunner(output=b"PASS-129"),   # neutral
        "timing_taps":   _StubRunner(output=b"PASS-130"),   # perturbs (len diff)
    }
    recs = check_observer_set_matrix(baseline, regimes, PROBE)
    by_label = {r.detail["instrumented"]: r for r in recs}
    assert by_label["guard_evidence"].result is CheckResult.FAIL
    assert by_label["minimal_probe"].result is CheckResult.PASS
    assert by_label["timing_taps"].result is CheckResult.FAIL
    assert all(r.check is CheckId.C6 for r in recs)
