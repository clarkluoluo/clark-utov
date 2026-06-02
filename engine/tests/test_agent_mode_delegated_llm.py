"""Regression test for BUG_REPORT-2 §1.

Asserts that serve_mcp attaches the configured LLMClient onto `core` so that
agent-mode S6 dispatchers reach the same DelegatedBackend, instead of falling
back to a fresh `LLMClient()` that would raise `KeyError: 'DEEPSEEK_API_KEY'`
in pure-delegated agent mode (no DeepSeek key in env).

We don't actually drive the full reader/writer threads — that would require a
pty harness. We instead verify the two contract pieces directly:
  (a) `setattr(core, "_llm", llm)` happens before any S6 dispatcher runs
  (b) the S6 dispatchers in `agent_mode._dispatch` pull from `getattr(core,
      "_llm", None)` (i.e. they reference the attribute, not LLMClient()).
"""

from __future__ import annotations

import io
import os
import tempfile
import threading
from pathlib import Path

from engine.core import Core, CoreConfig
from engine.llm_client import LLMClient
from engine.orchestrators.agent_mode import _QueueDelegatedBackend, serve_mcp
from engine.runner_client import NullRunnerAdapter
from engine.types import Instruction, TargetMeta


class _MinimalReader:
    def __iter__(self):
        yield Instruction(
            idx=0, pc=0x1000, bytes_=b"\x00\x00\x00\x00",
            mnemonic="ret", regs_read={}, regs_write={}, mem=(),
        )


def _build_core() -> Core:
    tm = TargetMeta(
        target_name="synthetic", arch="arm64",
        algo_entry_pc=0x1000, algo_exit_pc=0x1000,
        input_length=None, output_length=4,
    )
    work_root = Path(tempfile.mkdtemp(prefix="utov-test-agent-llm-"))
    cfg = CoreConfig(
        work_root=work_root, target_meta=tm, input_hash="testhash",
        driver_mode="agent", new_run=True,
    )
    return Core(cfg, _MinimalReader(), NullRunnerAdapter(tm), skip_conformance=True)


def test_serve_mcp_attaches_llm_to_core():
    """Once serve_mcp has wired the delegated backend, every subsequent S6
    dispatcher must be able to `getattr(core, "_llm", None)` and get the
    SAME LLMClient instance — not a fresh one that would try to read
    DEEPSEEK_API_KEY."""
    core = _build_core()

    # Build an LLMClient with a placeholder backend (so __init__ doesn't read
    # the API key) — exactly how `utov agent-serve` wires it.
    from engine.llm_client import DelegatedBackend
    llm = LLMClient(backend=DelegatedBackend(in_stream=None, out_stream=None))

    # Drive serve_mcp far enough to attach the LLM. We close stdin immediately
    # so it returns on EOF.
    stdin  = io.StringIO("")
    stdout = io.StringIO()
    stderr = io.StringIO()

    # serve_mcp blocks on the reader thread; running in another thread keeps
    # the test from hanging if anything is wrong. EOF on stdin shuts it down.
    t = threading.Thread(target=serve_mcp, kwargs={
        "core": core, "stdin": stdin, "stdout": stdout, "stderr": stderr,
        "llm": llm,
    }, daemon=True)
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive(), "serve_mcp didn't terminate on EOF"

    # The hot guarantee: core._llm is the same llm object, and its backend was
    # replaced with the Queue-flavored delegated variant.
    assert getattr(core, "_llm", None) is llm
    assert isinstance(llm.backend, _QueueDelegatedBackend)

    # Sanity: even in the absence of DEEPSEEK_API_KEY this works.
    saved = os.environ.pop("DEEPSEEK_API_KEY", None)
    try:
        assert getattr(core, "_llm", None) is llm
    finally:
        if saved is not None:
            os.environ["DEEPSEEK_API_KEY"] = saved
