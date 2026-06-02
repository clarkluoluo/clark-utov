"""BR-4 §F: ternary handler_semantic ops (Ch / Maj / Parity).

These are the SHA-2 / SHA-1 round-function primitives. Adding them lets the
verifier mechanically accept claims like
   `{op: CH, dst: x4, src: [x1, x2, x3]}` ↔ `(x1 & x2) ^ (~x1 & x3)`
which previously fell through to INCONCLUSIVE because `_BIN_OPS` only knows
2-operand ops.
"""

from __future__ import annotations

import pytest

from engine.runner_client import NullRunnerAdapter
from engine.types import TargetMeta
from engine.verifier import Verdict, Verifier


def _ver() -> Verifier:
    return Verifier(NullRunnerAdapter(TargetMeta(
        target_name="t", arch="arm64",
        algo_entry_pc=0x0, algo_exit_pc=0x0,
        input_length=None, output_length=4,
    )))


def test_ch_passes_on_correct_values():
    """Ch(x, y, z) = (x & y) ^ (~x & z)"""
    # x=0xF0F0, y=0xFF00, z=0x00FF
    # (0xF0F0 & 0xFF00) ^ (~0xF0F0 & 0x00FF) = 0xF000 ^ 0x000F = 0xF00F
    x, y, z = 0xF0F0, 0xFF00, 0x00FF
    expected = (x & y) ^ ((~x) & z) & 0xFFFFFFFFFFFFFFFF
    res = _ver().check_handler_semantic(
        {"x1": x, "x2": y, "x3": z},
        {"op": "CH", "dst": "x4", "src": ["x1", "x2", "x3"]},
        {"x4": expected},
    )
    assert res.verdict == Verdict.PASS


def test_maj_passes_on_correct_values():
    """Maj(x, y, z) = (x&y) ^ (x&z) ^ (y&z)"""
    x, y, z = 0xAAAA, 0xCCCC, 0xF0F0
    expected = (x & y) ^ (x & z) ^ (y & z)
    res = _ver().check_handler_semantic(
        {"a": x, "b": y, "c": z},
        {"op": "MAJ", "dst": "out", "src": ["a", "b", "c"]},
        {"out": expected},
    )
    assert res.verdict == Verdict.PASS


def test_parity_passes_on_correct_values():
    """Parity(x, y, z) = x ^ y ^ z — SHA-1 rounds 20-39, 60-79"""
    x, y, z = 0xDEAD, 0xBEEF, 0xCAFE
    expected = x ^ y ^ z
    res = _ver().check_handler_semantic(
        {"a": x, "b": y, "c": z},
        {"op": "PARITY", "dst": "out", "src": ["a", "b", "c"]},
        {"out": expected},
    )
    assert res.verdict == Verdict.PASS


def test_ch_fails_on_wrong_expected():
    res = _ver().check_handler_semantic(
        {"a": 0xF0F0, "b": 0xFF00, "c": 0x00FF},
        {"op": "CH", "dst": "out", "src": ["a", "b", "c"]},
        {"out": 0xDEADBEEF},   # not Ch(F0F0, FF00, 00FF)
    )
    assert res.verdict == Verdict.FAIL


def test_ch_inconclusive_on_missing_input_reg():
    res = _ver().check_handler_semantic(
        {"a": 0x1, "b": 0x2},   # `c` missing
        {"op": "CH", "dst": "out", "src": ["a", "b", "c"]},
        {"out": 0x0},
    )
    assert res.verdict == Verdict.INCONCLUSIVE


@pytest.mark.parametrize("op", ["CH", "MAJ", "PARITY"])
def test_ternary_op_lowercase_accepted(op):
    """Mnemonic-case insensitivity — capstone yields lowercase, claims often uppercase."""
    res = _ver().check_handler_semantic(
        {"a": 0x1, "b": 0x2, "c": 0x4},
        {"op": op.lower(), "dst": "out", "src": ["a", "b", "c"]},
        {"out": ({"CH":   (1 & 2) ^ ((~1) & 4) & 0xFFFFFFFFFFFFFFFF,
                  "MAJ":  (1 & 2) ^ (1 & 4) ^ (2 & 4),
                  "PARITY": 1 ^ 2 ^ 4}[op])},
    )
    assert res.verdict == Verdict.PASS
