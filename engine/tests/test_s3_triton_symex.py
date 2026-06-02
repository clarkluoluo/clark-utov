"""Smoke + degradation tests for the optional Triton symex path.

The module must import even when Triton isn't installed (so that running
`utov pipeline` without `--symex triton` never trips a missing dep). When
Triton IS installed, we verify the executor runs end-to-end on a single
hand-encoded AArch64 instruction.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from engine.stages import s3_triton_symex
from engine.store import WorkDir
from engine.types import Instruction


def test_module_importable_without_triton():
    """is_available()/unavailable_reason() must be consistent regardless of
    whether Triton is actually on the host."""
    if s3_triton_symex.is_available():
        assert s3_triton_symex.unavailable_reason() is None
    else:
        assert s3_triton_symex.unavailable_reason() is not None
        assert isinstance(s3_triton_symex.unavailable_reason(), str)


def test_env_mode_default_is_concrete(monkeypatch):
    monkeypatch.delenv("UTOV_SYMEX_MODE", raising=False)
    assert s3_triton_symex.env_mode() == "concrete"
    monkeypatch.setenv("UTOV_SYMEX_MODE", "triton")
    assert s3_triton_symex.env_mode() == "triton"
    monkeypatch.setenv("UTOV_SYMEX_MODE", "TRITON")
    assert s3_triton_symex.env_mode() == "triton"


def test_run_symex_raises_when_unavailable():
    """When Triton isn't loaded, run_symex must raise RuntimeError rather
    than silently returning an empty result."""
    if s3_triton_symex.is_available():
        pytest.skip("Triton is installed — this test only runs without it")
    work_root = Path(tempfile.mkdtemp(prefix="utov-test-triton-unavail-"))
    work = WorkDir(root=work_root, target="t", input_hash="h", new_run=True)
    with pytest.raises(RuntimeError, match="Triton unavailable"):
        s3_triton_symex.run_symex([], work)


@pytest.mark.skipif(not s3_triton_symex.is_available(),
                     reason="Triton bindings not installed")
def test_triton_executes_single_eor():  # pragma: no cover — only when Triton is present
    """eor x4, x1, x2 — AArch64 encoding: 0xCA020024 (little-endian bytes)."""
    instr = Instruction(
        idx=0, pc=0x1000, bytes_=bytes([0x24, 0x00, 0x02, 0xCA]),
        mnemonic="eor x4, x1, x2",
        regs_read={"x1": 0xA, "x2": 0xC},
        regs_write={"x4": 0x6},
        mem=(),
    )
    work_root = Path(tempfile.mkdtemp(prefix="utov-test-triton-ok-"))
    work = WorkDir(root=work_root, target="t", input_hash="h", new_run=True)
    summary = s3_triton_symex.run_symex([instr], work)
    assert summary["nodes"] == 1
    assert summary["symex"] == "triton"
    # The output JSONL must contain the x4 expression.
    out = Path(summary["out"]).read_text().strip().splitlines()
    assert len(out) == 1
    import json
    row = json.loads(out[0])
    # When decoded successfully, reg_exprs["x4"] is some non-empty AST string.
    if not summary["decode_failed"]:
        assert "x4" in row.get("reg_exprs", {})
        assert isinstance(row["reg_exprs"]["x4"], str)
