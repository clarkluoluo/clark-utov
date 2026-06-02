"""BR-4 §C: `propose_and_verify` must be cleanly splittable into
`generate_hypotheses_only` (LLM call, parallelizable) and
`ingest_hypotheses_and_verify` (tree/promote/verify, must stay serial).

We pin the contract:
  - generate_hypotheses_only returns (n_used, [Hypothesis, ...])
  - ingest_hypotheses_and_verify accepts that list and returns verdicts list
  - composing both reproduces propose_and_verify exactly (no regression)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from engine.core import Core, CoreConfig
from engine.discipline import DisciplineState
from engine.hyp_tree import HypTree
from engine.llm_client import Hypothesis, LLMBackend, LLMClient
from engine.runner_client import NullRunnerAdapter
from engine.stages.s6_hypothesis import (
    StuckContext,
    generate_hypotheses_only,
    ingest_hypotheses_and_verify,
    propose_and_verify,
)
from engine.store import open_hypotheses_db
from engine.types import Instruction, TargetMeta


class _PassBackend(LLMBackend):
    def generate_hypotheses(self, *_args, **_kwargs):
        return [Hypothesis(
            kind="handler_semantic", subject="h@eor",
            payload={"op": "XOR", "dst": "x4", "src": ["x1", "x2"]},
            confidence=0.9, rationale="always XOR",
        )]


def _build_core():
    tm = TargetMeta(target_name="syn", arch="arm64",
                     algo_entry_pc=0x1000, algo_exit_pc=0x1004,
                     input_length=None, output_length=8)
    instr = Instruction(idx=0, pc=0x1000, bytes_=b"\x00\x00\x00\x00",
                         mnemonic="eor x4, x1, x2",
                         regs_read={"x1": 0xA, "x2": 0xC},
                         regs_write={"x4": 0x6}, mem=())
    ret = Instruction(idx=1, pc=0x1004, bytes_=b"\x00\x00\x00\x00",
                     mnemonic="ret", regs_read={}, regs_write={}, mem=())
    work_root = Path(tempfile.mkdtemp(prefix="utov-test-s6split-"))
    cfg = CoreConfig(work_root=work_root, target_meta=tm,
                     input_hash="testhash", driver_mode="script", new_run=True)

    class _R:
        def __iter__(self): return iter([instr, ret])

    return Core(cfg, _R(), NullRunnerAdapter(tm), skip_conformance=True)


def test_generate_then_ingest_matches_propose_and_verify():
    """Splitting propose_and_verify into two halves yields the same verdicts
    list as the unsplit call. This is the C-path invariant — concurrent
    callers prefetch then drain; sequential calls compose."""
    core = _build_core()
    sc = StuckContext(
        parent_hyp_id=None, kind_hint="handler_semantic",
        summary="stuck@eor", snippet="eor x4, x1, x2", instr_idx=0,
    )
    in_state = dict(core._items[0].regs_read)
    out_state = dict(core._items[0].regs_write)

    llm = LLMClient(backend=_PassBackend())
    discipline = DisciplineState(target="syn", run_id=core.work.run_id)

    # Two-half path (concurrent caller would do this with prefetch in threads).
    conn = open_hypotheses_db(core.work)
    try:
        tree = HypTree(conn)
        n_used, hyps = generate_hypotheses_only(sc, tree, llm, discipline, n=1)
        assert n_used == 1
        assert len(hyps) == 1
        assert hyps[0].payload["op"] == "XOR"
        verdicts_split = ingest_hypotheses_and_verify(
            sc, hyps, tree, core.verifier,
            input_state=in_state, expected_output_state=out_state,
        )
    finally:
        conn.close()

    # Composed path (the public propose_and_verify API).
    conn2 = open_hypotheses_db(core.work)
    try:
        tree2 = HypTree(conn2)
        verdicts_full = propose_and_verify(
            sc, tree2, llm, core.verifier, discipline,
            input_state=in_state, expected_output_state=out_state, n=1,
        )
    finally:
        conn2.close()

    assert len(verdicts_split) == len(verdicts_full) == 1
    assert verdicts_split[0]["verdict"] == verdicts_full[0]["verdict"] == "pass"


def test_concurrency_flag_default_is_sequential():
    """run_full_pipeline.s6_concurrency defaults to 1, preserving the
    sequential code path for back-compat."""
    import inspect
    from engine.orchestrators.script_mode import run_full_pipeline
    sig = inspect.signature(run_full_pipeline)
    assert sig.parameters["s6_concurrency"].default == 1
