"""End-to-end test of intervention + audit + rerun_from_stage on a minimal
synthetic trace. Validates D-031 plumbing without spawning the Java runner.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from engine.core import Core, CoreConfig
from engine.runner_client import NullRunnerAdapter
from engine.types import Instruction, TargetMeta


class _SyntheticReader:
    """Minimal 5-instruction trace ending in 'ret'."""
    def __iter__(self):
        for i, (pc, mnem) in enumerate([
            (0x1000, "mov x0, #1"),
            (0x1004, "mov x1, #2"),
            (0x1008, "add x2, x0, x1"),
            (0x100c, "str x2, [sp]"),
            (0x1010, "ret"),
        ]):
            yield Instruction(
                idx=i, pc=pc, bytes_=b"\x00\x00\x00\x00",
                mnemonic=mnem,
                regs_read={}, regs_write={"x0": i} if i == 0 else {},
                mem=(),
            )


def _build_core() -> Core:
    tm = TargetMeta(
        target_name="synthetic", arch="arm64",
        algo_entry_pc=0x1000, algo_exit_pc=0x1010,
        input_length=None, output_length=4,
    )
    work_root = Path(tempfile.mkdtemp(prefix="utov-test-"))
    cfg = CoreConfig(
        work_root=work_root, target_meta=tm, input_hash="testhash",
        driver_mode="script", new_run=True,
    )
    return Core(cfg, _SyntheticReader(), NullRunnerAdapter(tm), skip_conformance=True)


def test_override_verdict_is_audited():
    core = _build_core()
    hid = core.submit_hypothesis(
        kind="handler_semantic", subject="h@0x1008",
        payload={"op": "ADD", "dst": "x2", "src": ["x0", "x1"]},
        confidence=0.5, source="agent",
    )
    # Override without verifier
    core.override_verdict(hid, "pass", reason="trace shows add x2=x0+x1")
    interventions = core.list_interventions(limit=10)
    assert any(r["action"] == "override_verdict" and r["target_id"] == str(hid)
               for r in interventions), f"override not audited: {interventions}"
    # Verify status changed
    hyps = core.get_hypotheses()
    assert any(h.id == hid and h.status == "passed" for h in hyps)


def test_rerun_from_stage_cascades():
    core = _build_core()
    # Run S1 deterministically so stage_state has s1=done
    core.run_stage("s1")
    state_before = core.work.read_stage_state()
    assert "s1" in state_before

    # Inject a hyp tagged with created_in_stage='s1b' and a finding from it
    hid = core.submit_hypothesis(
        kind="algo_signature", subject="SHA256.h0",
        payload={"fingerprint": "SHA256.h0"}, confidence=0.65,
        source="plugin",
    )
    # We don't have created_in_stage path via submit_hypothesis; patch via SQL.
    from engine.store import open_hypotheses_db
    conn = open_hypotheses_db(core.work)
    try:
        conn.execute("UPDATE hypotheses SET created_in_stage='s1b' WHERE id=?", (hid,))
        conn.execute("UPDATE hypotheses SET status='passed' WHERE id=?", (hid,))
        conn.commit()
    finally:
        conn.close()

    fid = core.promote_to_finding(hid, verifier_strategy="manual", stage="s1b")
    assert fid is not None

    # Rerun from s1b
    result = core.rerun_from_stage("s1b", reason="test cascade")
    assert "s1b" in result["cascade_stages"]
    assert result["findings_deleted"] >= 1
    assert result["hyps_abandoned_count"] >= 1

    # State after: s1 still done, s1b not
    state_after = core.work.read_stage_state()
    assert "s1" in state_after
    assert "s1b" not in state_after

    # Audit recorded
    interv = core.list_interventions(action="rerun_from_stage")
    assert len(interv) >= 1
