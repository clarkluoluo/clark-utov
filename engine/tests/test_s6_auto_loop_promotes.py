"""Regression test for BUG_REPORT-2 §2.

`s6_hypothesis.run()` must return the verdicts list so agent_mode.s6_auto_loop
can promote each `verdict == pass` hyp to a finding — otherwise verifier work
is silently thrown away and `findings.sqlite` stays empty.

We exercise the layer that actually had the bug: run `s6_hypothesis.run()`
with a stubbed always-pass LLM backend + a real Verifier on a hand-built input
state, then walk the returned `verdicts` and promote_to_finding each pass.
The finding table count must equal the pass count.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from engine.core import Core, CoreConfig
from engine.llm_client import Hypothesis, LLMBackend, LLMClient
from engine.runner_client import NullRunnerAdapter
from engine.stages.s6_hypothesis import StuckContext
from engine.stages.s6_hypothesis import run as s6_run
from engine.types import Instruction, TargetMeta


class _AlwaysPassBackend(LLMBackend):
    """Returns a single handler_semantic hyp that the verifier will accept
    given the synthetic trace below."""

    def generate_hypotheses(self, system_prompt, user_context, schema, n):
        return [Hypothesis(
            kind="handler_semantic", subject="h@eor",
            payload={"op": "XOR", "dst": "x4", "src": ["x1", "x2"]},
            confidence=0.9, rationale="stub: always XOR",
        )]


class _ReaderTwoInstr:
    """One eor instruction; regs_read/write chosen so that x1 ^ x2 == x4."""
    def __iter__(self):
        # x1=0xa, x2=0xc, x4=0x6  →  0xa ^ 0xc = 0x6, verdict=pass
        yield Instruction(
            idx=0, pc=0x1000, bytes_=b"\x00\x00\x00\x00",
            mnemonic="eor x4, x1, x2",
            regs_read={"x1": 0xA, "x2": 0xC},
            regs_write={"x4": 0x6},
            mem=(),
        )
        yield Instruction(
            idx=1, pc=0x1004, bytes_=b"\x00\x00\x00\x00",
            mnemonic="ret", regs_read={}, regs_write={}, mem=(),
        )


def _build_core() -> Core:
    tm = TargetMeta(
        target_name="synthetic-eor", arch="arm64",
        algo_entry_pc=0x1000, algo_exit_pc=0x1004,
        input_length=None, output_length=8,
    )
    work_root = Path(tempfile.mkdtemp(prefix="utov-test-promote-"))
    cfg = CoreConfig(
        work_root=work_root, target_meta=tm, input_hash="testhash",
        driver_mode="agent", new_run=True,
    )
    return Core(cfg, _ReaderTwoInstr(), NullRunnerAdapter(tm), skip_conformance=True)


def test_s6_run_returns_verdicts_and_promote_pass_count_matches():
    core = _build_core()
    sc = StuckContext(
        parent_hyp_id=None,
        kind_hint="handler_semantic",
        summary="synthetic stuck point at idx=0",
        snippet="eor x4, x1, x2",
        instr_idx=0,
    )
    in_state  = dict(core._items[0].regs_read)
    out_state = dict(core._items[0].regs_write)

    llm = LLMClient(backend=_AlwaysPassBackend())
    summary = s6_run({
        "work":          core.work,
        "verifier":      core.verifier,
        "llm":           llm,
        "stuck_context": sc,
        "input_state":   in_state,
        "expected_output_state": out_state,
        "n":             1,
    })
    # The big regression: `verdicts` is exposed so the caller can promote.
    assert "verdicts" in summary
    assert summary["candidates"] == 1
    assert summary["passed"] == 1
    assert summary["failed"] == 0
    assert summary["pending"] == 0

    # Mimic agent_mode.s6_auto_loop's promotion loop.
    promoted = 0
    for v in summary["verdicts"]:
        if v["verdict"] == "pass":
            core.promote_to_finding(int(v["hyp_id"]),
                                     verifier_strategy="handler_semantic")
            promoted += 1
    assert promoted == summary["passed"]

    # And findings.sqlite must actually have a row — this is what the bug
    # silently broke before BR-2 §2 + the s6_run verdicts return.
    findings = list(core.get_findings()) if hasattr(core, "get_findings") else None
    if findings is not None:
        # New API path
        assert len(findings) == promoted
    else:
        # Fallback: query the findings db file directly
        import sqlite3
        conn = sqlite3.connect(core.work.root / "findings.sqlite")
        try:
            n = conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0]
        finally:
            conn.close()
        assert n == promoted


def test_s6_run_inconclusive_is_counted_separately():
    """BR-2 §5: aggregate must distinguish inconclusive from
    failed/pending/passed. We force INCONCLUSIVE by handing the verifier a
    claim whose src registers aren't in input_state."""
    core = _build_core()
    sc = StuckContext(
        parent_hyp_id=None,
        kind_hint="handler_semantic",
        summary="stuck",
        snippet="eor x4, x1, x2",
        instr_idx=0,
    )

    class _BadBackend(LLMBackend):
        def generate_hypotheses(self, *_args, **_kwargs):
            return [Hypothesis(
                kind="handler_semantic", subject="bad@xor",
                payload={"op": "XOR", "dst": "x99", "src": ["x77", "x78"]},
                confidence=0.5, rationale="reg names not in state",
            )]

    llm = LLMClient(backend=_BadBackend())
    summary = s6_run({
        "work":          core.work,
        "verifier":      core.verifier,
        "llm":           llm,
        "stuck_context": sc,
        "input_state":   {"x1": 0xA, "x2": 0xC},
        "expected_output_state": {"x4": 0x6},
        "n":             1,
    })
    assert summary["candidates"] == 1
    assert summary["inconclusive"] == 1
    assert summary["passed"] == 0
    assert summary["failed"] == 0


def test_agent_mode_dispatch_error_carries_traceback():
    """BR-2 §6: JSON-RPC error.data should include traceback so callers can
    locate the failure site without grepping source."""
    import io
    from engine.orchestrators.agent_mode import _write_err

    out = io.StringIO()
    _write_err(out, 1, -32000, "boom",
               data={"traceback": 'File "x.py", line 1, in <module>'})
    import json
    msg = json.loads(out.getvalue().strip())
    assert msg["error"]["code"] == -32000
    assert "boom" in msg["error"]["message"]
    assert "traceback" in msg["error"]["data"]
