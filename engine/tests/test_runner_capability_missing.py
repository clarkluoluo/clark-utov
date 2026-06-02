"""Regression test for BR-2 §13 / §14 (new bug discovered after BR-2):

The conformance gate's File-mode SKIP path was triggered only by Python-side
`NotImplementedError`. A `SubprocessRunnerAdapter` paired with a runner that
throws Java `UnsupportedOperationException` from `rerun` would surface the
error as `RuntimeError("runner error: …")` — outside the narrow `except
NotImplementedError` in `run_conformance` and in every individual check —
crashing the gate before C1/C2/C3/C5 could SKIP as documented.

Fix: `SubprocessRunnerAdapter._call` now translates capability-missing wire
errors (JSON-RPC -32601, or messages matching `unsupportedoperationexception`
/ `not implemented` / `unsupported operation` / etc.) into NotImplementedError.

This test pins the helper and the round-trip behavior.
"""

from __future__ import annotations

import pytest

from engine.runner_client import _is_capability_missing_error


@pytest.mark.parametrize("code,message,expected", [
    # JSON-RPC standard: -32601 is "Method not found"
    (-32601, "Method not found", True),
    (-32601, "",                 True),

    # Java-runner default exception name
    (-32603, "UnsupportedOperationException: rerun", True),
    (None,   "java.lang.UnsupportedOperationException", True),
    (1,      "uncaught UnsupportedOperationException at line 42", True),

    # English variants the matchers recognize
    (-32603, "not implemented",   True),
    (-32603, "NotImplemented",    True),
    (-32603, "unsupported operation",  True),
    (-32603, "method not supported",   True),
    (-32603, "not supported in file mode", True),
    (None,   "runner is in File mode",     True),

    # Real runtime errors must NOT be misclassified
    (-32000, "trace file path /tmp/x.txt does not exist", False),
    (-32603, "NullPointerException at runner/Main.java:42", False),
    (-1,     "out of memory",                              False),
    (None,   "",                                           False),
])
def test_capability_missing_classifier(code, message, expected):
    assert _is_capability_missing_error(code, message) is expected


def test_subprocess_adapter_translates_to_NotImplementedError(monkeypatch):
    """End-to-end: a stubbed SubprocessRunnerAdapter whose underlying process
    returns a `code=-32603 message=UnsupportedOperationException` error must
    surface as NotImplementedError from `.rerun()`, NOT RuntimeError. This is
    exactly what the conformance gate's auto-detect catches."""
    import io
    import json
    from engine.runner_client import SubprocessRunnerAdapter

    # Build an adapter that skips real process spawn — we synthesize stdin/out.
    class _FakeProc:
        stdin  = io.StringIO()
        stdout = io.StringIO()
        def poll(self): return None
    adapter = SubprocessRunnerAdapter.__new__(SubprocessRunnerAdapter)
    adapter._proc = _FakeProc()
    adapter._next_id = 0
    adapter._stderr_lines = []

    # Pre-load the fake stdout with an UnsupportedOperationException error.
    adapter._proc.stdout = io.StringIO(json.dumps({
        "id": 1,
        "error": {
            "code": -32603,
            "message": "UnsupportedOperationException: rerun not implemented",
        },
    }) + "\n")

    with pytest.raises(NotImplementedError, match="rerun"):
        adapter._call("rerun", {"input_hex": "00"})

    # And confirm: a non-capability-missing error stays as RuntimeError.
    adapter._next_id = 0
    adapter._proc.stdout = io.StringIO(json.dumps({
        "id": 1,
        "error": {"code": -32000, "message": "NullPointerException at Main.java:42"},
    }) + "\n")
    with pytest.raises(RuntimeError, match="NullPointerException"):
        adapter._call("rerun", {"input_hex": "00"})


def test_run_conformance_auto_detects_file_mode_from_capability_error():
    """The wire-translation makes `run_conformance` correctly mark mode='file'
    and SKIP C1/C2/C3/C5 — instead of crashing — when the runner throws
    UnsupportedOperationException for rerun."""
    from engine.conformance import run_conformance, CheckId, CheckResult
    from engine.runner_client import RunnerAdapter
    from engine.types import TargetMeta

    class _UnsupportedRunner(RunnerAdapter):
        """Live-shaped adapter (subclass of RunnerAdapter so the mode-detect
        doesn't short-circuit on NullRunnerAdapter), but rerun + get_trace
        translate to NotImplementedError — exactly what
        SubprocessRunnerAdapter._call now produces from
        UnsupportedOperationException wire errors."""

        def metadata(self):
            return TargetMeta(
                target_name="t", arch="arm64",
                algo_entry_pc=0x1000, algo_exit_pc=0x1010,
                input_length=None, output_length=4,
            )

        def rerun(self, input_bytes, observe_points=None):
            raise NotImplementedError("runner does not implement 'rerun'")

        def get_trace(self, input_bytes, start, end):
            raise NotImplementedError("runner does not implement 'get_trace'")

    class _DummyReader:
        def __iter__(self):
            from engine.types import Instruction
            yield Instruction(idx=0, pc=0x1000, bytes_=b"\x00\x00\x00\x00",
                              mnemonic="ret",
                              regs_read={}, regs_write={}, mem=())

    runner = _UnsupportedRunner()
    report = run_conformance(runner, _DummyReader(), probe_input=b"\x00" * 8)
    # Mode auto-detected as File
    assert report.mode == "file"
    # C1/C2/C3/C5 SKIP as documented in PLAN §17
    by_id = {c.check: c for c in report.checks}
    for cid in (CheckId.C1, CheckId.C2, CheckId.C3, CheckId.C5):
        assert by_id[cid].result == CheckResult.SKIP, \
            f"{cid.value} should SKIP in File mode (got {by_id[cid].result.value})"
    # C4 runs (trace_reader was supplied) and either PASS or FAIL — not SKIP.
    assert by_id[CheckId.C4].result != CheckResult.SKIP
