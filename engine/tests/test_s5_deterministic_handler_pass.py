"""Regression test for BR-4 §1: S5 deterministic handler-semantic pass.

`Core.verify_and_promote_handler_binops` must:
  1. PASS reg-reg-reg `eor w?, w?, w?` whose regs_write matches `src1 ^ src2`
  2. SKIP extended-register forms (`add x8, x9, w10, sxtw`) — verifier doesn't
     have the shift/extend amount in the heuristic payload, safer to skip
  3. SKIP sp-relative arithmetic (`sub sp, sp, #0x60`) — not a binop shape
  4. Dedupe by PC — the same instruction repeated in a loop produces ONE
     finding, not N
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from engine.core import Core, CoreConfig
from engine.runner_client import NullRunnerAdapter
from engine.types import Instruction, TargetMeta


def _build_core(instrs) -> Core:
    tm = TargetMeta(
        target_name="syn-handlers", arch="arm64",
        algo_entry_pc=instrs[0].pc, algo_exit_pc=instrs[-1].pc,
        input_length=None, output_length=4,
    )
    work_root = Path(tempfile.mkdtemp(prefix="utov-test-s5verify-"))
    cfg = CoreConfig(
        work_root=work_root, target_meta=tm, input_hash="testhash",
        driver_mode="script", new_run=True,
    )

    class _R:
        def __init__(self, xs): self.xs = xs
        def __iter__(self): return iter(self.xs)

    return Core(cfg, _R(instrs), NullRunnerAdapter(tm), skip_conformance=True)


def test_eor_reg_reg_reg_promotes():
    """x4 = x1 ^ x2 — verifier confirms, one finding promoted."""
    eor = Instruction(
        idx=0, pc=0x1000, bytes_=b"\x00\x00\x00\x00",
        mnemonic="eor x4, x1, x2",
        regs_read={"x1": 0xA, "x2": 0xC},
        regs_write={"x4": 0x6},  # 0xA ^ 0xC = 0x6
        mem=(),
    )
    ret = Instruction(idx=1, pc=0x1004, bytes_=b"\x00\x00\x00\x00",
                     mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([eor, ret])
    r = core.verify_and_promote_handler_binops()
    assert r["stage"] == "s5-verify"
    assert r["checked"] == 1
    assert r["passed"] == 1
    assert r["failed"] == 0
    assert r["inconclusive"] == 0
    assert r["promoted"] == 1


def test_extended_register_form_is_skipped():
    """`add x8, x9, w10, sxtw` has a sxtw tail — heuristic payload doesn't
    carry it, so safer to skip than to false-pass/fail."""
    add = Instruction(
        idx=0, pc=0x1000, bytes_=b"\x00\x00\x00\x00",
        mnemonic="add x8, x9, w10, sxtw",
        regs_read={"x9": 0x10, "w10": 0xFFFFFFFF},
        regs_write={"x8": 0x0F},  # depends on sxtw — verifier wouldn't be right
        mem=(),
    )
    ret = Instruction(idx=1, pc=0x1004, bytes_=b"\x00\x00\x00\x00",
                     mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([add, ret])
    r = core.verify_and_promote_handler_binops()
    assert r["checked"] == 0     # not even attempted
    assert r["promoted"] == 0


def test_sp_relative_arithmetic_is_skipped():
    """`sub sp, sp, #0x60` is prologue/frame setup — not a binop shape
    verifier should be asked to confirm."""
    sub = Instruction(
        idx=0, pc=0x1000, bytes_=b"\x00\x00\x00\x00",
        mnemonic="sub sp, sp, #0x60",
        regs_read={"sp": 0x100}, regs_write={"sp": 0xA0}, mem=(),
    )
    ret = Instruction(idx=1, pc=0x1004, bytes_=b"\x00\x00\x00\x00",
                     mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([sub, ret])
    r = core.verify_and_promote_handler_binops()
    assert r["checked"] == 0
    assert r["promoted"] == 0


def test_same_pc_dedupes_to_one_finding():
    """A loop body's `eor` appears at the same PC across iterations.
    We promote one finding per PC, not N."""
    # 3 traces of the same eor instruction at the same PC, all with matching
    # regs_read/regs_write semantics. Should produce ONE finding.
    instrs = [
        Instruction(
            idx=i, pc=0x1000, bytes_=b"\x00\x00\x00\x00",
            mnemonic="eor w4, w1, w2",
            regs_read={"w1": 0xA + i, "w2": 0xC + i},
            regs_write={"w4": (0xA + i) ^ (0xC + i)},
            mem=(),
        )
        for i in range(3)
    ]
    instrs.append(Instruction(idx=3, pc=0x1004, bytes_=b"\x00\x00\x00\x00",
                              mnemonic="ret", regs_read={}, regs_write={}, mem=()))
    core = _build_core(instrs)
    r = core.verify_and_promote_handler_binops()
    assert r["checked"] == 1, \
        f"after dedup, only PC=0x1000 should be checked; got checked={r['checked']}"
    assert r["passed"] == 1
    assert r["promoted"] == 1
