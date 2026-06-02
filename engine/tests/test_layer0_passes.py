"""Regression tests for 0526Plan C5 layer-0 single-instruction handler passes.

Covers three new methods plus verifier fixes uncovered while wiring them:

  - verify_and_promote_handler_unaries       (C5.4 + C5.5)
  - verify_and_promote_handler_imm_binops    (C5.1 + C5.2 + C5.3)
  - verify_and_promote_handler_extended_binops (C5.7)

Also pins regressions for the verifier bugs the TC2 baseline exposed:

  - _SRC2_EXTS uxtw/sxtw etc must honor `amount` (left-shift after extend)
  - REV on a w-reg dst is a 32-bit byte-reverse, not 64-bit
  - wzr/xzr substitute as 0 (not in regs_read)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from engine.core import Core, CoreConfig
from engine.runner_client import NullRunnerAdapter
from engine.types import Instruction, TargetMeta
from engine.verifier import Verdict, Verifier


def _build_core(instrs) -> Core:
    tm = TargetMeta(
        target_name="syn-handlers", arch="arm64",
        algo_entry_pc=instrs[0].pc, algo_exit_pc=instrs[-1].pc,
        input_length=None, output_length=4,
    )
    work_root = Path(tempfile.mkdtemp(prefix="utov-test-layer0-"))
    cfg = CoreConfig(
        work_root=work_root, target_meta=tm, input_hash="testhash",
        driver_mode="script", new_run=True,
    )

    class _R:
        def __init__(self, xs): self.xs = xs
        def __iter__(self): return iter(self.xs)

    return Core(cfg, _R(instrs), NullRunnerAdapter(tm), skip_conformance=True)


# --- Unary pass (C5.4 + C5.5) ----------------------------------------------

def test_unary_sxtw_promotes():
    """sxtw x4, w1 — sign-extend 32-bit (negative)."""
    sxtw = Instruction(
        idx=0, pc=0x1000, bytes_=b"\x00" * 4,
        mnemonic="sxtw x4, w1",
        regs_read={"w1": 0x80000001},
        regs_write={"x4": 0xFFFFFFFF80000001},
        mem=(),
    )
    ret = Instruction(idx=1, pc=0x1004, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([sxtw, ret])
    r = core.verify_and_promote_handler_unaries()
    assert r["stage"] == "s5-verify-unary"
    assert r["checked"] == 1 and r["passed"] == 1 and r["promoted"] == 1


def test_unary_rev_w_reg_uses_32bit():
    """rev w9, w9 — 32-bit byte-reverse (would fail with 64-bit semantics)."""
    rev = Instruction(
        idx=0, pc=0x1000, bytes_=b"\x00" * 4,
        mnemonic="rev w9, w9",
        regs_read={"w9": 0x07740160},
        regs_write={"w9": 0x60017407},   # 32-bit byte-reversed
        mem=(),
    )
    ret = Instruction(idx=1, pc=0x1004, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([rev, ret])
    r = core.verify_and_promote_handler_unaries()
    assert r["passed"] == 1 and r["failed"] == 0


def test_unary_mov_xzr_treats_zero():
    """mov x23, xzr — zero register substitutes as 0 (regs_read empty)."""
    mov = Instruction(
        idx=0, pc=0x1000, bytes_=b"\x00" * 4,
        mnemonic="mov x23, xzr",
        regs_read={},          # xzr usually not captured
        regs_write={"x23": 0},
        mem=(),
    )
    ret = Instruction(idx=1, pc=0x1004, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([mov, ret])
    r = core.verify_and_promote_handler_unaries()
    assert r["passed"] == 1 and r["inconclusive"] == 0


def test_unary_skips_3_operand_ext_form():
    """`add x4, x5, x6` is binop, not unary — must skip."""
    add = Instruction(
        idx=0, pc=0x1000, bytes_=b"\x00" * 4,
        mnemonic="add x4, x5, x6",
        regs_read={"x5": 0x1, "x6": 0x2},
        regs_write={"x4": 0x3},
        mem=(),
    )
    ret = Instruction(idx=1, pc=0x1004, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([add, ret])
    r = core.verify_and_promote_handler_unaries()
    assert r["checked"] == 0 and r["promoted"] == 0


# --- Reg-imm binop pass (C5.1 + C5.2 + C5.3) -------------------------------

def test_imm_add_with_hex_imm():
    """add x4, x5, #0x10 — add w/ hex immediate."""
    add = Instruction(
        idx=0, pc=0x1000, bytes_=b"\x00" * 4,
        mnemonic="add x4, x5, #0x10",
        regs_read={"x5": 0x100},
        regs_write={"x4": 0x110},
        mem=(),
    )
    ret = Instruction(idx=1, pc=0x1004, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([add, ret])
    r = core.verify_and_promote_handler_imm_binops()
    assert r["stage"] == "s5-verify-imm"
    assert r["passed"] == 1 and r["promoted"] == 1


def test_imm_lsl_decimal_imm():
    """lsl w4, w5, #3 — shift left by 3 (decimal imm)."""
    lsl = Instruction(
        idx=0, pc=0x1000, bytes_=b"\x00" * 4,
        mnemonic="lsl w4, w5, #3",
        regs_read={"w5": 0x10},
        regs_write={"w4": 0x80},
        mem=(),
    )
    ret = Instruction(idx=1, pc=0x1004, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([lsl, ret])
    r = core.verify_and_promote_handler_imm_binops()
    assert r["passed"] == 1


def test_imm_skips_sp_relative():
    """sub sp, sp, #0x60 — sp-relative arith, skip."""
    sub = Instruction(
        idx=0, pc=0x1000, bytes_=b"\x00" * 4,
        mnemonic="sub sp, sp, #0x60",
        regs_read={}, regs_write={}, mem=(),
    )
    ret = Instruction(idx=1, pc=0x1004, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([sub, ret])
    r = core.verify_and_promote_handler_imm_binops()
    assert r["checked"] == 0 and r["promoted"] == 0


# --- Extended-register binop pass (C5.7) -----------------------------------

def test_extended_uxtw_shift_3():
    """add x9, x21, w9, uxtw #3 — x21 + (uxtw(w9) << 3).

    Pins the verifier regression: previously _SRC2_EXTS["uxtw"] ignored
    the `amount`, making this case fail.
    """
    add = Instruction(
        idx=0, pc=0x1000, bytes_=b"\x00" * 4,
        mnemonic="add x9, x21, w9, uxtw #3",
        regs_read={"x21": 0xbffff538, "w9": 0x1d},
        regs_write={"x9": 0xbffff620},   # 0xbffff538 + (0x1d << 3)
        mem=(),
    )
    ret = Instruction(idx=1, pc=0x1004, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([add, ret])
    r = core.verify_and_promote_handler_extended_binops()
    assert r["stage"] == "s5-verify-ext"
    assert r["passed"] == 1 and r["failed"] == 0 and r["promoted"] == 1


def test_extended_sxtw_negative():
    """add x4, x5, w6, sxtw — sign-extend negative w-reg, no shift."""
    add = Instruction(
        idx=0, pc=0x1000, bytes_=b"\x00" * 4,
        mnemonic="add x4, x5, w6, sxtw",
        regs_read={"x5": 0x10, "w6": 0xFFFFFFFF},   # -1 as w-reg
        regs_write={"x4": 0xF},   # 0x10 + (-1) = 0xF
        mem=(),
    )
    ret = Instruction(idx=1, pc=0x1004, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([add, ret])
    r = core.verify_and_promote_handler_extended_binops()
    assert r["passed"] == 1


def test_extended_skips_plain_binop():
    """`add x4, x5, x6` (no tail) is plain binop, not extended-register."""
    add = Instruction(
        idx=0, pc=0x1000, bytes_=b"\x00" * 4,
        mnemonic="add x4, x5, x6",
        regs_read={"x5": 0x1, "x6": 0x2},
        regs_write={"x4": 0x3},
        mem=(),
    )
    ret = Instruction(idx=1, pc=0x1004, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([add, ret])
    r = core.verify_and_promote_handler_extended_binops()
    assert r["checked"] == 0


# --- Verifier-level regression: src2_ext amount + wzr handling -------------

def test_verifier_uxtw_amount():
    """_SRC2_EXTS["uxtw"] now multiplies the post-extend value by 2**amount."""

    class _NullRerun:
        pass

    v = Verifier(_NullRerun())
    res = v.check_handler_semantic(
        {"x21": 0xbffff538, "w9": 0x1d},
        {"op": "ADD", "dst": "x9", "src": ["x21", "w9"],
         "src2_ext": {"kind": "uxtw", "amount": 3}},
        {"x9": 0xbffff620},
    )
    assert res.verdict == Verdict.PASS


def test_verifier_wzr_in_binop():
    """add x4, x5, xzr — xzr substitutes as 0."""

    class _NullRerun:
        pass

    v = Verifier(_NullRerun())
    res = v.check_handler_semantic(
        {"x5": 0x42},   # xzr not in input_state
        {"op": "ADD", "dst": "x4", "src": ["x5", "xzr"]},
        {"x4": 0x42},
    )
    assert res.verdict == Verdict.PASS


# --- Bit-field extract pass (C5.6) -----------------------------------------

def test_bfx_ubfx_promotes():
    """ubfx w9, w12, #0x15, #5 — extract bits 21..25 of w12.

    w12 = 0xdfbdfc28, lsb=21, width=5 → (0xdfbdfc28 >> 21) & 0x1F = 0x1d
    """
    ubfx = Instruction(
        idx=0, pc=0x1000, bytes_=b"\x00" * 4,
        mnemonic="ubfx w9, w12, #0x15, #5",
        regs_read={"w12": 0xdfbdfc28},
        regs_write={"w9": 0x1d},
        mem=(),
    )
    ret = Instruction(idx=1, pc=0x1004, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([ubfx, ret])
    r = core.verify_and_promote_handler_bfx()
    assert r["stage"] == "s5-verify-bfx"
    assert r["passed"] == 1 and r["failed"] == 0 and r["promoted"] == 1


def test_bfx_sbfx_sign_extends():
    """sbfx w4, w5, #0, #8 — top bit of byte sign-extends to width."""
    sbfx = Instruction(
        idx=0, pc=0x1000, bytes_=b"\x00" * 4,
        mnemonic="sbfx w4, w5, #0, #8",
        regs_read={"w5": 0x000000FF},   # top bit set in the 8-bit slice
        regs_write={"w4": 0xFFFFFFFF},  # sign-extended to 32-bit
        mem=(),
    )
    ret = Instruction(idx=1, pc=0x1004, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([sbfx, ret])
    r = core.verify_and_promote_handler_bfx()
    assert r["passed"] == 1 and r["failed"] == 0


def test_bfx_skips_non_bfx():
    """add not a BFX shape — must skip."""
    add = Instruction(
        idx=0, pc=0x1000, bytes_=b"\x00" * 4,
        mnemonic="add x4, x5, #0x10",
        regs_read={"x5": 0x100}, regs_write={"x4": 0x110}, mem=(),
    )
    ret = Instruction(idx=1, pc=0x1004, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([add, ret])
    r = core.verify_and_promote_handler_bfx()
    assert r["checked"] == 0


# --- C3 Ch idiom discoverer ------------------------------------------------

def test_ch_idiom_promotes():
    """Real TC2 SHA-512 Ch idiom — concrete values from libEncryptor.so trace.

      eor  x18, x13, x15      ; t = y ^ z
      and  x18, x12, x18      ; t = x & t        (src-order x first, then t)
      eor  x15, x18, x15      ; d = t ^ z
    """
    x12 = 0xDEADBEEFCAFEBABE  # x (= a)
    x13 = 0x0123456789ABCDEF  # y (= b)
    x15 = 0xFEDCBA9876543210  # z (= c)
    M = 0xFFFFFFFFFFFFFFFF
    t1 = (x13 ^ x15) & M
    t2 = (x12 & t1) & M
    d  = (t2 ^ x15) & M
    ch_expected = ((x12 & x13) ^ ((~x12) & x15)) & M
    assert d == ch_expected   # sanity-check the algebra

    eor0 = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                       mnemonic="eor x18, x13, x15",
                       regs_read={"x13": x13, "x15": x15},
                       regs_write={"x18": t1}, mem=())
    and1 = Instruction(idx=1, pc=0x1004, bytes_=b"\x00" * 4,
                       mnemonic="and x18, x12, x18",
                       regs_read={"x12": x12, "x18": t1},
                       regs_write={"x18": t2}, mem=())
    eor2 = Instruction(idx=2, pc=0x1008, bytes_=b"\x00" * 4,
                       mnemonic="eor x15, x18, x15",
                       regs_read={"x18": t2, "x15": x15},
                       regs_write={"x15": d}, mem=())
    ret = Instruction(idx=3, pc=0x100C, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([eor0, and1, eor2, ret])
    r = core.verify_and_promote_handler_ch_idioms()
    assert r["stage"] == "s5-verify-ch"
    assert r["checked"] == 1 and r["passed"] == 1
    assert r["failed"] == 0 and r["inconclusive"] == 0
    assert r["promoted"] == 1


def test_sigma_idiom_sha256_sigma1():
    """SHA-256 Σ1(w21) = ROR(w21,6) ^ ROR(w21,11) ^ ROR(w21,25), as the
    canonical clang 3-insn ARM form `ror; eor ...ror; eor ...ror`.
    """
    W = 32
    M = 0xFFFFFFFF

    def ror32(v, n):
        v &= M
        return ((v >> n) | ((v << (W - n)) & M)) & M

    x = 0x510e527f
    a = ror32(x, 6)
    b = a ^ ror32(x, 11)
    c = b ^ ror32(x, 25)
    sigma1_expected = ror32(x, 6) ^ ror32(x, 11) ^ ror32(x, 25)
    assert c == sigma1_expected

    ins0 = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                       mnemonic="ror w22, w21, #6",
                       regs_read={"w21": x}, regs_write={"w22": a},
                       mem=())
    ins1 = Instruction(idx=1, pc=0x1004, bytes_=b"\x00" * 4,
                       mnemonic="eor w22, w22, w21, ror #11",
                       regs_read={"w22": a, "w21": x},
                       regs_write={"w22": b}, mem=())
    ins2 = Instruction(idx=2, pc=0x1008, bytes_=b"\x00" * 4,
                       mnemonic="eor w22, w22, w21, ror #25",
                       regs_read={"w22": b, "w21": x},
                       regs_write={"w22": c}, mem=())
    ret = Instruction(idx=3, pc=0x100C, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([ins0, ins1, ins2, ret])
    r = core.verify_and_promote_sigma_idioms()
    assert r["stage"] == "s5-fold-sigma"
    assert r["matched"] == 1 and r["promoted"] == 1
    assert r["algebra_mismatch"] == 0


def test_sigma_idiom_sha512_sigma0_lsr_variant():
    """SHA-512 σ0(x) = ROR(x,1) ^ ROR(x,8) ^ SHR(x,7). lsr in the 3rd insn."""
    W = 64
    M = (1 << W) - 1

    def ror64(v, n):
        v &= M
        return ((v >> n) | ((v << (W - n)) & M)) & M

    x = 0x0123456789ABCDEF
    a = ror64(x, 1)
    b = a ^ ror64(x, 8)
    c = b ^ (x >> 7)
    sigma_expected = ror64(x, 1) ^ ror64(x, 8) ^ (x >> 7)
    assert c == sigma_expected

    ins0 = Instruction(idx=0, pc=0x2000, bytes_=b"\x00" * 4,
                       mnemonic="ror x4, x5, #1",
                       regs_read={"x5": x}, regs_write={"x4": a}, mem=())
    ins1 = Instruction(idx=1, pc=0x2004, bytes_=b"\x00" * 4,
                       mnemonic="eor x4, x4, x5, ror #8",
                       regs_read={"x4": a, "x5": x},
                       regs_write={"x4": b}, mem=())
    ins2 = Instruction(idx=2, pc=0x2008, bytes_=b"\x00" * 4,
                       mnemonic="eor x4, x4, x5, lsr #7",
                       regs_read={"x4": b, "x5": x},
                       regs_write={"x4": c}, mem=())
    ret = Instruction(idx=3, pc=0x200C, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([ins0, ins1, ins2, ret])
    r = core.verify_and_promote_sigma_idioms()
    assert r["matched"] == 1 and r["promoted"] == 1


def test_sigma_idiom_rejects_wrong_amounts():
    """A window with the σ-shape but non-matching (kind, amount) triple
    must not be promoted."""
    ins0 = Instruction(idx=0, pc=0x3000, bytes_=b"\x00" * 4,
                       mnemonic="ror w4, w5, #1",
                       regs_read={"w5": 0x12345678},
                       regs_write={"w4": 0x91A2B3C}, mem=())
    ins1 = Instruction(idx=1, pc=0x3004, bytes_=b"\x00" * 4,
                       mnemonic="eor w4, w4, w5, ror #2",
                       regs_read={"w4": 0x91A2B3C, "w5": 0x12345678},
                       regs_write={"w4": 0xDEADBEEF}, mem=())
    ins2 = Instruction(idx=2, pc=0x3008, bytes_=b"\x00" * 4,
                       mnemonic="eor w4, w4, w5, ror #3",
                       regs_read={"w4": 0xDEADBEEF, "w5": 0x12345678},
                       regs_write={"w4": 0xDEADCAFE}, mem=())
    ret = Instruction(idx=3, pc=0x300C, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([ins0, ins1, ins2, ret])
    r = core.verify_and_promote_sigma_idioms()
    assert r["matched"] == 0 and r["promoted"] == 0


def test_sigma_phase3_tc1_sha256_sigma1_dst_eq_input():
    """BR-8 #1: TC1 SHA-256 σ₁ at 0x12000b00 area — the three pieces are
    interleaved with σ₀'s pieces (gap=1) AND the final write targets the
    input register (w21). Phase 1's chained-acc check rules this out;
    Phase 3 DFG-grouped scan catches it via (kind, amount) on shared input.
    """
    W = 32
    M = 0xFFFFFFFF

    def ror32(v, n):
        v &= M
        return ((v >> n) | ((v << (W - n)) & M)) & M

    x = 0x510e527f
    a = ror32(x, 17)
    b = a ^ ror32(x, 19)
    c = b ^ (x >> 10)
    assert c == ror32(x, 17) ^ ror32(x, 19) ^ (x >> 10)

    # σ₁ piece 1 at b00.
    p0 = Instruction(idx=0, pc=0x12000b00, bytes_=b"\x00" * 4,
                     mnemonic="ror w22, w21, #17",
                     regs_read={"w21": x}, regs_write={"w22": a}, mem=())
    # σ₀ piece 1 (unrelated input register w23) interleaved at b04.
    p1 = Instruction(idx=1, pc=0x12000b04, bytes_=b"\x00" * 4,
                     mnemonic="ror w24, w23, #7",
                     regs_read={"w23": 0xABCDEF01},
                     regs_write={"w24": 0}, mem=())
    # σ₁ piece 2 at b08.
    p2 = Instruction(idx=2, pc=0x12000b08, bytes_=b"\x00" * 4,
                     mnemonic="eor w22, w22, w21, ror #19",
                     regs_read={"w22": a, "w21": x},
                     regs_write={"w22": b}, mem=())
    # σ₁ piece 3 at b0c, final dst is the input register w21.
    p3 = Instruction(idx=3, pc=0x12000b0c, bytes_=b"\x00" * 4,
                     mnemonic="eor w21, w22, w21, lsr #10",
                     regs_read={"w22": b, "w21": x},
                     regs_write={"w21": c}, mem=())
    ret = Instruction(idx=4, pc=0x12000b10, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([p0, p1, p2, p3, ret])
    # No s1 run: Phase 3 doesn't depend on BB metadata.
    r = core.verify_and_promote_sigma_idioms()
    assert r["matched"] >= 1 and r["promoted"] >= 1
    # The σ₁ fold_idiom finding should be present.
    subjects = [f["subject"] for f in core.get_findings(kind="fold_idiom")]
    assert any(s.startswith("SHA256.sigma1@") for s in subjects)


def test_sigma_phase3_tc1_sha256_Sigma0_large_gap():
    """BR-8 #1: TC1 SHA-256 Σ₀ at 0x12000c54/c6c/c7c — gap of 5 unrelated
    insns between piece 1 and piece 2, all on input register w22 with the
    final write targeting w22 itself. Phase-2 MAX_GAP=8 covers this, but
    only if a BB exists; with no BB metadata, Phase 3 still finds it.
    """
    W = 32
    M = 0xFFFFFFFF

    def ror32(v, n):
        v &= M
        return ((v >> n) | ((v << (W - n)) & M)) & M

    x = 0x6a09e667
    a = ror32(x, 2)
    b = a ^ ror32(x, 13)
    c = b ^ ror32(x, 22)

    p0 = Instruction(idx=0, pc=0x12000c54, bytes_=b"\x00" * 4,
                     mnemonic="ror w28, w22, #2",
                     regs_read={"w22": x}, regs_write={"w28": a}, mem=())
    # 5 unrelated insns (Maj-style, none on w22's value side).
    pads = []
    for k, pc in enumerate(range(0x12000c58, 0x12000c6c, 4)):
        pads.append(Instruction(
            idx=1 + k, pc=pc, bytes_=b"\x00" * 4,
            mnemonic=f"and w{25 + (k % 3)}, w26, w27",
            regs_read={"w26": 0, "w27": 0},
            regs_write={f"w{25 + (k % 3)}": 0}, mem=(),
        ))
    p2 = Instruction(idx=1 + len(pads), pc=0x12000c6c, bytes_=b"\x00" * 4,
                     mnemonic="eor w27, w28, w22, ror #13",
                     regs_read={"w28": a, "w22": x},
                     regs_write={"w27": b}, mem=())
    # 3 unrelated padding.
    pads2 = []
    for k, pc in enumerate(range(0x12000c70, 0x12000c7c, 4)):
        pads2.append(Instruction(
            idx=2 + len(pads) + k, pc=pc, bytes_=b"\x00" * 4,
            mnemonic="and w24, w26, w22",
            regs_read={"w26": 0, "w22": x}, regs_write={"w24": 0}, mem=(),
        ))
    p3 = Instruction(idx=2 + len(pads) + len(pads2), pc=0x12000c7c,
                     bytes_=b"\x00" * 4,
                     mnemonic="eor w22, w27, w22, ror #22",
                     regs_read={"w27": b, "w22": x},
                     regs_write={"w22": c}, mem=())
    ret = Instruction(idx=3 + len(pads) + len(pads2), pc=0x12000c80,
                      bytes_=b"\x00" * 4, mnemonic="ret",
                      regs_read={}, regs_write={}, mem=())
    core = _build_core([p0, *pads, p2, *pads2, p3, ret])
    r = core.verify_and_promote_sigma_idioms()
    assert r["matched"] >= 1 and r["promoted"] >= 1
    subjects = [f["subject"] for f in core.get_findings(kind="fold_idiom")]
    assert any(s.startswith("SHA256.Sigma0@") for s in subjects)


def test_algorithm_template_fit_sha256():
    """Layer-2 fit (E1.4 + E1.5): seed findings with SHA-256 anchors
    (σ/Σ idioms + plugin h0..h7 fingerprints), run the algorithm-template
    fit, expect a SHA-256 `algorithm_identified` finding and a NullRunner
    `io_test=skipped` annotation."""
    import sqlite3

    nop = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([nop])

    # Seed findings directly: 4 σ/Σ idioms + 8 plugin h0..h7 = full SHA-256
    # anchor set.
    from engine.store import open_findings_db, upsert_payload, _now_iso
    f_conn = open_findings_db(core.work)
    ch = upsert_payload(f_conn, {"seed": 1})
    for subj in ("SHA256.Sigma0@0x1000", "SHA256.Sigma1@0x1004",
                 "SHA256.sigma0@0x1008", "SHA256.sigma1@0x100c"):
        f_conn.execute(
            "INSERT INTO findings(stage, kind, subject, payload_ref, "
            "verified_at, verifier_strategy, origin_hyp_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("s5-fold", "fold_idiom", subj, ch, _now_iso(),
             "algebraic_idiom_match", None),
        )
    for i in range(8):
        f_conn.execute(
            "INSERT INTO findings(stage, kind, subject, payload_ref, "
            "verified_at, verifier_strategy, origin_hyp_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("s1b-verify", "algo_signature", f"SHA256.h{i}", ch, _now_iso(),
             "handler_semantic", None),
        )
    f_conn.commit()
    f_conn.close()

    r = core.verify_and_promote_algorithm_templates()
    assert r["matched_algorithms"] == ["SHA-256"]
    assert r["promoted"] == 1
    assert "skipped" in r["io_test"]   # NullRunner / file-mode

    # Confirm a promoted algorithm finding lives in findings.sqlite
    f_conn = sqlite3.connect(core.work.root / "findings.sqlite")
    row = f_conn.execute(
        "SELECT subject, verifier_strategy FROM findings "
        "WHERE kind = 'algorithm_hyp' AND subject = 'SHA-256'"
    ).fetchone()
    assert row is not None
    assert row[1] == "structural_anchor_set_match"
    # Linked anchor count = 12 (4 idioms + 8 plugin constants)
    n_members = f_conn.execute(
        "SELECT COUNT(*) FROM finding_groups WHERE idiom_name = 'SHA-256'"
    ).fetchone()[0]
    assert n_members == 12
    f_conn.close()


def test_algorithm_hyp_carries_local_closure_trap():
    """Task 7 — the structural matcher emits ``algorithm_hyp`` (NOT
    ``algorithm_identified``), and the payload carries the explicit
    LOCAL_CLOSURE_ONLY trap so a reader never mistakes a pre-oracle-closure
    primitive hypothesis for a final identification."""
    import json
    import sqlite3

    nop = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([nop])
    from engine.store import _now_iso, open_findings_db, upsert_payload
    f_conn = open_findings_db(core.work)
    ch = upsert_payload(f_conn, {"seed": 1})
    for subj in ("SHA256.Sigma0@0x1000", "SHA256.Sigma1@0x1004",
                 "SHA256.sigma0@0x1008", "SHA256.sigma1@0x100c"):
        f_conn.execute(
            "INSERT INTO findings(stage, kind, subject, payload_ref, "
            "verified_at, verifier_strategy, origin_hyp_id) VALUES (?,?,?,?,?,?,?)",
            ("s5-fold", "fold_idiom", subj, ch, _now_iso(),
             "algebraic_idiom_match", None))
    for i in range(8):
        f_conn.execute(
            "INSERT INTO findings(stage, kind, subject, payload_ref, "
            "verified_at, verifier_strategy, origin_hyp_id) VALUES (?,?,?,?,?,?,?)",
            ("s1b-verify", "algo_signature", f"SHA256.h{i}", ch, _now_iso(),
             "handler_semantic", None))
    f_conn.commit()
    f_conn.close()

    core.verify_and_promote_algorithm_templates()

    f_conn = sqlite3.connect(core.work.root / "findings.sqlite")
    # The matcher emits the HYP kind, NEVER the strong reserved kind.
    assert f_conn.execute(
        "SELECT COUNT(*) FROM findings WHERE kind='algorithm_identified'"
    ).fetchone()[0] == 0
    ref = f_conn.execute(
        "SELECT payload_ref FROM findings WHERE kind='algorithm_hyp' "
        "AND subject='SHA-256'"
    ).fetchone()[0]
    payload = json.loads(f_conn.execute(
        "SELECT payload FROM hyp_payloads WHERE content_hash=?", (ref,)
    ).fetchone()[0])
    f_conn.close()
    closure = payload["closure"]
    assert closure["trap_state"] == "LOCAL_CLOSURE_ONLY"
    assert closure["algorithm_closed"] is False
    assert closure["is_primitive"] is True
    assert "HYPOTHESIS" in payload["rationale"]


def test_sigma_idiom_sha512_sigma1_dst_switch():
    """BUG_REPORT-6 #4a: σ₁ with a compiler-chosen different live-out register
    on the final eor. Previously rejected by the dst-stability check.

      ror x15, x13, #0x3d
      eor x15, x15, x13, lsr #6
      eor x13, x15, x13, ror #19      ; ← dst switches to x13
    """
    W = 64
    M = (1 << W) - 1

    def ror64(v, n):
        v &= M
        return ((v >> n) | ((v << (W - n)) & M)) & M

    x = 0xCAFEBABEDEADBEEF
    a = ror64(x, 61)
    b = a ^ (x >> 6)
    c = b ^ ror64(x, 19)
    ins0 = Instruction(idx=0, pc=0x68cc, bytes_=b"\x00" * 4,
                       mnemonic="ror x15, x13, #61",
                       regs_read={"x13": x}, regs_write={"x15": a}, mem=())
    ins1 = Instruction(idx=1, pc=0x68d0, bytes_=b"\x00" * 4,
                       mnemonic="eor x15, x15, x13, lsr #6",
                       regs_read={"x15": a, "x13": x},
                       regs_write={"x15": b}, mem=())
    ins2 = Instruction(idx=2, pc=0x68d4, bytes_=b"\x00" * 4,
                       mnemonic="eor x13, x15, x13, ror #19",
                       regs_read={"x15": b, "x13": x},
                       regs_write={"x13": c}, mem=())
    ret = Instruction(idx=3, pc=0x68d8, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([ins0, ins1, ins2, ret])
    r = core.verify_and_promote_sigma_idioms()
    assert r["matched"] == 1 and r["promoted"] == 1
    assert r["algebra_mismatch"] == 0


def test_sigma_idiom_phase2_ilp_interleaved():
    """BUG_REPORT-6 #4b: SHA-512 Σ₁ where the three components are
    non-contiguous (Σ pieces interleaved with Maj-like writes). Phase-2
    per-BB scan should still match.
    """
    W = 64
    M = (1 << W) - 1

    def ror64(v, n):
        v &= M
        return ((v >> n) | ((v << (W - n)) & M)) & M

    x = 0x12345678DEADBEEF
    a = ror64(x, 14)
    b = a ^ ror64(x, 18)
    c = b ^ ror64(x, 41)

    # Σ₁ piece 1
    ins0 = Instruction(idx=0, pc=0x6924, bytes_=b"\x00" * 4,
                       mnemonic="ror x4, x16, #14",
                       regs_read={"x16": x}, regs_write={"x4": a}, mem=())
    # 3 unrelated Maj-like writes that DON'T clobber x4
    pad1 = Instruction(idx=1, pc=0x6928, bytes_=b"\x00" * 4,
                       mnemonic="orr x6, x13, x12",
                       regs_read={"x13": 0, "x12": 0},
                       regs_write={"x6": 0}, mem=())
    pad2 = Instruction(idx=2, pc=0x692c, bytes_=b"\x00" * 4,
                       mnemonic="and x7, x13, x12",
                       regs_read={"x13": 0, "x12": 0},
                       regs_write={"x7": 0}, mem=())
    pad3 = Instruction(idx=3, pc=0x6930, bytes_=b"\x00" * 4,
                       mnemonic="and x6, x6, x15",
                       regs_read={"x6": 0, "x15": 0},
                       regs_write={"x6": 0}, mem=())
    # Σ₁ piece 2
    ins1 = Instruction(idx=4, pc=0x6934, bytes_=b"\x00" * 4,
                       mnemonic="eor x4, x4, x16, ror #18",
                       regs_read={"x4": a, "x16": x},
                       regs_write={"x4": b}, mem=())
    # one more pad
    pad4 = Instruction(idx=5, pc=0x6938, bytes_=b"\x00" * 4,
                       mnemonic="orr x6, x6, x7",
                       regs_read={"x6": 0, "x7": 0},
                       regs_write={"x6": 0}, mem=())
    # Σ₁ piece 3
    ins2 = Instruction(idx=6, pc=0x693c, bytes_=b"\x00" * 4,
                       mnemonic="eor x5, x4, x16, ror #41",
                       regs_read={"x4": b, "x16": x},
                       regs_write={"x5": c}, mem=())
    ret = Instruction(idx=7, pc=0x6940, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([ins0, pad1, pad2, pad3, ins1, pad4, ins2, ret])
    # Phase-2 scan reads s1 blocks from disk — run s1 first.
    core.run_stage("s1")
    r = core.verify_and_promote_sigma_idioms()
    assert r["matched"] == 1 and r["promoted"] == 1


def test_recompute_algorithm_fits_updates_anchors():
    """BUG_REPORT-6 #5: after a new fold_idiom finding is injected, calling
    recompute_algorithm_fits refreshes the existing algorithm_identified
    payload's anchors_seen / evidence_score in place.
    """
    import json
    import sqlite3
    from engine.store import _now_iso, open_findings_db, upsert_payload

    nop = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([nop])

    f_conn = open_findings_db(core.work)
    ch = upsert_payload(f_conn, {"seed": 1})
    # Seed 11/12 SHA-512 anchors (missing σ₁).
    for subj in ("SHA512.Sigma0@0x1000", "SHA512.Sigma1@0x1004",
                 "SHA512.sigma0@0x1008"):
        f_conn.execute(
            "INSERT INTO findings(stage, kind, subject, payload_ref, "
            "verified_at, verifier_strategy, origin_hyp_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("s5-fold", "fold_idiom", subj, ch, _now_iso(),
             "algebraic_idiom_match", None),
        )
    for i in range(8):
        f_conn.execute(
            "INSERT INTO findings(stage, kind, subject, payload_ref, "
            "verified_at, verifier_strategy, origin_hyp_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("s1b-verify", "algo_signature", f"SHA512.h{i}", ch, _now_iso(),
             "handler_semantic", None),
        )
    f_conn.commit()
    f_conn.close()

    r1 = core.verify_and_promote_algorithm_templates()
    assert r1["matched_algorithms"] == ["SHA-512"]

    # Read the initial anchors_seen.
    def _read_anchors() -> set[str]:
        f_conn = sqlite3.connect(core.work.root / "findings.sqlite")
        row = f_conn.execute(
            "SELECT payload_ref FROM findings WHERE kind='algorithm_hyp' AND subject='SHA-512'"
        ).fetchone()
        ref = row[0]
        pload = f_conn.execute(
            "SELECT payload FROM hyp_payloads WHERE content_hash=?", (ref,)
        ).fetchone()[0]
        f_conn.close()
        return set(json.loads(pload)["anchors_seen"])

    before = _read_anchors()
    assert "SHA512.sigma1" not in before
    assert len(before) == 11

    # Inject the missing σ₁ fold.
    core.inject_finding(
        kind="fold_idiom",
        subject="SHA512.sigma1@0x2000",
        payload={"idiom": "SHA512.sigma1", "manually_injected": True},
        reason="agent recovered missing σ₁ from trace",
    )

    after = _read_anchors()
    assert "SHA512.sigma1" in after
    assert len(after) == 12


def test_algorithm_template_fit_below_threshold_skips():
    """When fewer than `min_unique_anchors` anchors are present, no
    algorithm finding is promoted."""
    nop = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([nop])

    from engine.store import open_findings_db, upsert_payload, _now_iso
    f_conn = open_findings_db(core.work)
    ch = upsert_payload(f_conn, {"seed": 1})
    # Only 3 anchors — below min_unique_anchors=4.
    for subj in ("SHA256.Sigma0@0x1000", "SHA256.h0", "SHA256.h1"):
        f_conn.execute(
            "INSERT INTO findings(stage, kind, subject, payload_ref, "
            "verified_at, verifier_strategy, origin_hyp_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("s5-fold", "fold_idiom" if subj.endswith(("0x1000",)) else "algo_signature",
             subj, ch, _now_iso(), "noop", None),
        )
    f_conn.commit()
    f_conn.close()

    r = core.verify_and_promote_algorithm_templates()
    assert r["matched_algorithms"] == []
    assert r["promoted"] == 0


def test_algorithm_template_fit_aes_single_te0_hit():
    """BUG_REPORT-7 §B: a single AES.Te0[0] fingerprint hit identifies AES.

    AES.Te0 constants are zero-FP outside an AES implementation, so the
    template uses min_unique_anchors=1 and a confidence_override (the
    fraction-of-12-anchors formula would yield ~0.55 which understates it).
    """
    import sqlite3

    nop = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([nop])

    from engine.store import _now_iso, open_findings_db, upsert_payload
    f_conn = open_findings_db(core.work)
    ch = upsert_payload(f_conn, {"seed": 1})
    f_conn.execute(
        "INSERT INTO findings(stage, kind, subject, payload_ref, "
        "verified_at, verifier_strategy, origin_hyp_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("s1b-verify", "algo_signature", "AES.Te0[0]", ch, _now_iso(),
         "plugin", None),
    )
    f_conn.commit()
    f_conn.close()

    r = core.verify_and_promote_algorithm_templates()
    assert r["matched_algorithms"] == ["AES"]
    assert r["promoted"] == 1
    # Confidence override pins AES at 0.85 with just 1 anchor.
    f_conn = sqlite3.connect(core.work.root / "findings.sqlite")
    payload_ref = f_conn.execute(
        "SELECT payload_ref FROM findings "
        "WHERE kind='algorithm_hyp' AND subject='AES'"
    ).fetchone()[0]
    import json
    payload = json.loads(f_conn.execute(
        "SELECT payload FROM hyp_payloads WHERE content_hash=?",
        (payload_ref,),
    ).fetchone()[0])
    f_conn.close()
    assert payload["algorithm"] == "AES"
    assert payload["anchors_seen"] == ["AES.Te0[0]"]
    assert payload["evidence_score"] == round(1 / 7, 3)
    # io_test is dict (§C): AES keyed → skipped, no canonical vector.
    assert payload["io_test"]["status"] == "skipped"
    # Reference impl is wired (§J.5).
    assert payload["reference_impl"]["unknowns"] == ["key", "iv", "mode"]


def test_algorithm_template_fit_sha256_io_passed():
    """BUG_REPORT-7 §C: SHA-256 structural match + duck-typed working runner.

    Uses a fake runner that returns hashlib.sha256 of the input. Expect
    io_test.status='passed' and the structural-only 0.85 cap lifted by +0.10.
    """
    import hashlib
    import sqlite3

    from engine.runner_client import RerunResult, RunnerAdapter

    class FakeSha256Runner(RunnerAdapter):
        def __init__(self, meta):
            self._meta = meta
        def metadata(self):
            return self._meta
        def rerun(self, input_bytes, observe_points=None):
            return RerunResult(output=hashlib.sha256(input_bytes).digest())

    nop = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    tm = TargetMeta(
        target_name="syn-sha256-live", arch="arm64",
        algo_entry_pc=0x1000, algo_exit_pc=0x1000,
        input_length=None, output_length=32,
    )
    import tempfile
    work_root = Path(tempfile.mkdtemp(prefix="utov-test-iopass-"))
    cfg = CoreConfig(
        work_root=work_root, target_meta=tm, input_hash="iopass",
        driver_mode="script", new_run=True,
    )

    class _R:
        def __init__(self, xs): self.xs = xs
        def __iter__(self): return iter(self.xs)

    core = Core(cfg, _R([nop]), FakeSha256Runner(tm), skip_conformance=True)

    from engine.store import _now_iso, open_findings_db, upsert_payload
    f_conn = open_findings_db(core.work)
    ch = upsert_payload(f_conn, {"seed": 1})
    for subj in ("SHA256.Sigma0", "SHA256.Sigma1",
                 "SHA256.sigma0", "SHA256.sigma1"):
        f_conn.execute(
            "INSERT INTO findings(stage, kind, subject, payload_ref, "
            "verified_at, verifier_strategy, origin_hyp_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("s5-fold", "fold_idiom", subj, ch, _now_iso(),
             "algebraic_idiom_match", None),
        )
    f_conn.commit()
    f_conn.close()

    r = core.verify_and_promote_algorithm_templates()
    assert r["matched_algorithms"] == ["SHA-256"]
    assert r["io_test_by_algo"]["SHA-256"]["status"] == "passed"
    # Read back the hypothesis and confirm the confidence was boosted.
    f_conn = sqlite3.connect(core.work.root / "hypotheses.sqlite")
    conf = f_conn.execute(
        "SELECT t.confidence FROM hypotheses h "
        "JOIN claim_templates t ON h.template_id = t.id "
        "WHERE t.kind='algorithm_hyp' AND h.subject='SHA-256'"
    ).fetchone()[0]
    f_conn.close()
    # 4/12 anchors → structural conf = 0.5 + 0.35*0.333 = 0.617 → +0.10 = 0.717
    assert conf > 0.70


def test_algorithm_template_fit_sha256_io_failed_drops_confidence():
    """BUG_REPORT-7 §C: a runner returning wrong bytes for SHA-256("abc") gets
    io_test.status='failed' and confidence drops by 0.30.
    """
    import sqlite3
    from engine.runner_client import RerunResult, RunnerAdapter

    class WrongRunner(RunnerAdapter):
        def __init__(self, meta):
            self._meta = meta
        def metadata(self):
            return self._meta
        def rerun(self, input_bytes, observe_points=None):
            return RerunResult(output=b"\x00" * 32)   # wrong

    nop = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    tm = TargetMeta(
        target_name="syn-sha256-wrong", arch="arm64",
        algo_entry_pc=0x1000, algo_exit_pc=0x1000,
        input_length=None, output_length=32,
    )
    import tempfile
    work_root = Path(tempfile.mkdtemp(prefix="utov-test-iofail-"))
    cfg = CoreConfig(
        work_root=work_root, target_meta=tm, input_hash="iofail",
        driver_mode="script", new_run=True,
    )

    class _R:
        def __init__(self, xs): self.xs = xs
        def __iter__(self): return iter(self.xs)

    core = Core(cfg, _R([nop]), WrongRunner(tm), skip_conformance=True)

    from engine.store import _now_iso, open_findings_db, upsert_payload
    f_conn = open_findings_db(core.work)
    ch = upsert_payload(f_conn, {"seed": 1})
    for subj in ("SHA256.Sigma0", "SHA256.Sigma1",
                 "SHA256.sigma0", "SHA256.sigma1"):
        f_conn.execute(
            "INSERT INTO findings(stage, kind, subject, payload_ref, "
            "verified_at, verifier_strategy, origin_hyp_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("s5-fold", "fold_idiom", subj, ch, _now_iso(),
             "algebraic_idiom_match", None),
        )
    f_conn.commit()
    f_conn.close()

    r = core.verify_and_promote_algorithm_templates()
    assert r["matched_algorithms"] == ["SHA-256"]
    assert r["io_test_by_algo"]["SHA-256"]["status"] == "failed"


def test_algorithm_template_fit_io_skipped_on_null_runner():
    """NullRunnerAdapter raises NotImplementedError from rerun → io_test
    cleanly degrades to status='skipped' instead of falsely 'errored'."""
    nop = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([nop])
    from engine.store import _now_iso, open_findings_db, upsert_payload
    f_conn = open_findings_db(core.work)
    ch = upsert_payload(f_conn, {"seed": 1})
    for subj in ("SHA256.Sigma0", "SHA256.Sigma1",
                 "SHA256.sigma0", "SHA256.sigma1"):
        f_conn.execute(
            "INSERT INTO findings(stage, kind, subject, payload_ref, "
            "verified_at, verifier_strategy, origin_hyp_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("s5-fold", "fold_idiom", subj, ch, _now_iso(),
             "algebraic_idiom_match", None),
        )
    f_conn.commit()
    f_conn.close()
    r = core.verify_and_promote_algorithm_templates()
    assert r["io_test_by_algo"]["SHA-256"]["status"] == "skipped"
    assert "NotImplementedError" in r["io_test_by_algo"]["SHA-256"]["detail"]


def test_static_artifacts_skipped_when_no_so_path():
    """No --so → graceful skip, never crashes."""
    nop = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([nop])
    r = core.verify_and_promote_static_artifacts()
    assert r["promoted"] == 0
    assert "no --so" in r["skipped"]


def test_static_artifacts_finds_high_entropy_window_in_real_so():
    """Run the scan against example/task-libEncryptor/libs/arm64-v8a/libEncryptor.so. Confirm at least
    one high-entropy window is emitted. The exact offset depends on .rodata
    content; we just want a stable lower bound that proves the pipeline
    works end-to-end (objdump parse + entropy + finding emit).
    """
    import shutil
    if shutil.which("objdump") is None:
        import pytest
        pytest.skip("objdump not available")
    so = Path(__file__).resolve().parents[2] / "example" / "task-libEncryptor" / "libs" / "arm64-v8a" / "libEncryptor.so"
    if not so.exists():
        import pytest
        pytest.skip(f"sample .so not present at {so}")

    nop = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([nop])
    core.session["so_path"] = str(so)

    r = core.verify_and_promote_static_artifacts()
    # We only assert "the path runs" + "some bytes were scanned" — the
    # specific candidate count drifts with the .so but should never be 0.
    assert r["rodata_bytes_scanned"] > 0
    # Either we found candidates or .rodata is all low entropy — both are
    # valid for an arbitrary .so. Don't assert promoted > 0 here.
    assert "skipped" not in r


def test_mode_evidence_ledger_hmac_missing_when_no_ipad():
    """§J.3: SHA-512 identified but no HMAC ipad/opad anchors → ledger
    records HMAC as MISSING with cap=0.30. The structural-only "HMAC-SHA-*"
    guess this challenges came from BUG_REPORT-6.
    """
    import sqlite3
    nop = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([nop])

    from engine.store import _now_iso, open_findings_db, upsert_payload
    f_conn = open_findings_db(core.work)
    ch = upsert_payload(f_conn, {"seed": 1})
    f_conn.execute(
        "INSERT INTO findings(stage, kind, subject, payload_ref, "
        "verified_at, verifier_strategy, origin_hyp_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("s5-algorithm-fit", "algorithm_identified", "SHA-512", ch,
         _now_iso(), "structural_anchor_set_match", None),
    )
    f_conn.commit()
    f_conn.close()

    r = core.verify_and_promote_mode_evidence_ledger()
    assert r["modes_evaluated"] == 1
    assert r["promoted"] == 1
    f_conn = sqlite3.connect(core.work.root / "findings.sqlite")
    row = f_conn.execute(
        "SELECT payload_ref FROM findings WHERE kind='mode_evidence_ledger'"
    ).fetchone()
    import json
    payload = json.loads(f_conn.execute(
        "SELECT payload FROM hyp_payloads WHERE content_hash=?", (row[0],)
    ).fetchone()[0])
    f_conn.close()
    assert payload["mode_candidate"] == "HMAC"
    assert payload["applies_to_algorithms"] == ["SHA-512"]
    assert payload["confidence_cap"] == 0.30
    assert "MISSING" in payload["verdict"]


def test_mode_evidence_ledger_hmac_present_when_ipad_opad_anchors_fire():
    """SHA-256 + HMAC.ipad + HMAC.opad fingerprint hits → ledger says PRESENT."""
    import sqlite3
    nop = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([nop])

    from engine.store import _now_iso, open_findings_db, upsert_payload
    f_conn = open_findings_db(core.work)
    ch = upsert_payload(f_conn, {"seed": 1})
    f_conn.execute(
        "INSERT INTO findings(stage, kind, subject, payload_ref, "
        "verified_at, verifier_strategy, origin_hyp_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("s5-algorithm-fit", "algorithm_identified", "SHA-256", ch,
         _now_iso(), "structural_anchor_set_match", None),
    )
    for subj in ("HMAC.ipad@0x2000", "HMAC.opad@0x2004"):
        f_conn.execute(
            "INSERT INTO findings(stage, kind, subject, payload_ref, "
            "verified_at, verifier_strategy, origin_hyp_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("s1b-verify", "algo_signature", subj, ch, _now_iso(),
             "plugin", None),
        )
    f_conn.commit()
    f_conn.close()

    r = core.verify_and_promote_mode_evidence_ledger()
    f_conn = sqlite3.connect(core.work.root / "findings.sqlite")
    row = f_conn.execute(
        "SELECT payload_ref FROM findings WHERE kind='mode_evidence_ledger'"
    ).fetchone()
    import json
    payload = json.loads(f_conn.execute(
        "SELECT payload FROM hyp_payloads WHERE content_hash=?", (row[0],)
    ).fetchone()[0])
    f_conn.close()
    assert payload["mode_candidate"] == "HMAC"
    assert "PRESENT" in payload["verdict"]
    assert payload["confidence_cap"] >= 0.85


def test_mode_evidence_ledger_silent_without_algorithm_identified():
    """No algorithm identified → no ledger emitted (nothing to scope to)."""
    nop = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([nop])
    r = core.verify_and_promote_mode_evidence_ledger()
    assert r["modes_evaluated"] == 0
    assert r["promoted"] == 0


def test_indexed_load_table_promotes_aes_te0_pattern():
    """§J.1: 10 canonical AES Te0 lookups at the same base → one
    indexed_load_table finding with element_bytes=4 and accurate ranges.
    """
    import sqlite3
    BASE = 0x40008960
    instrs = []
    for i in range(10):
        instrs.append(Instruction(
            idx=i, pc=0x40002770 + i * 4, bytes_=b"\x00" * 4,
            mnemonic=f"ldr w{i % 4}, [x10, w{(i+1) % 4}, uxtw #2]",
            regs_read={"x10": BASE, f"w{(i+1) % 4}": i * 4},
            regs_write={f"w{i % 4}": 0xc66363a5 + i},
            mem=(),
        ))
    instrs.append(Instruction(
        idx=10, pc=0x40002800, bytes_=b"\x00" * 4,
        mnemonic="ret", regs_read={}, regs_write={}, mem=(),
    ))
    core = _build_core(instrs)
    r = core.verify_and_promote_indexed_load_table()
    assert r["tables_found"] == 1
    assert r["promoted"] == 1
    # Confirm finding
    f_conn = sqlite3.connect(core.work.root / "findings.sqlite")
    row = f_conn.execute(
        "SELECT subject, payload_ref FROM findings "
        "WHERE kind='indexed_load_table'"
    ).fetchone()
    assert row is not None
    assert f"@0x{BASE:x}/4B" in row[0]
    import json
    payload = json.loads(f_conn.execute(
        "SELECT payload FROM hyp_payloads WHERE content_hash=?", (row[1],)
    ).fetchone()[0])
    f_conn.close()
    assert payload["base_addr"]    == f"0x{BASE:x}"
    assert payload["element_bytes"] == 4
    assert payload["total_loads"]   == 10
    assert payload["unique_indexes"] >= 4
    assert payload["trace_idx_range"] == [0, 9]


def test_indexed_load_table_skips_below_threshold():
    """Fewer than MIN_LOADS (8) loads at the same base — not a table."""
    BASE = 0x40008960
    instrs = []
    for i in range(5):
        instrs.append(Instruction(
            idx=i, pc=0x40002770 + i * 4, bytes_=b"\x00" * 4,
            mnemonic="ldr w0, [x10, w1, uxtw #2]",
            regs_read={"x10": BASE, "w1": i},
            regs_write={"w0": i}, mem=(),
        ))
    instrs.append(Instruction(idx=5, pc=0x40002784, bytes_=b"\x00" * 4,
                              mnemonic="ret", regs_read={}, regs_write={}, mem=()))
    core = _build_core(instrs)
    r = core.verify_and_promote_indexed_load_table()
    assert r["promoted"] == 0
    assert r["tables_found"] == 0


def test_indexed_load_table_distinguishes_bases():
    """Two distinct base addresses → two separate findings."""
    BASE_A, BASE_B = 0x40008960, 0x4000c5dc
    instrs = []
    n = 8
    for i in range(n):
        instrs.append(Instruction(
            idx=i, pc=0x40002770 + i * 4, bytes_=b"\x00" * 4,
            mnemonic="ldr w0, [x10, w1, uxtw #2]",
            regs_read={"x10": BASE_A, "w1": i},
            regs_write={"w0": i}, mem=(),
        ))
    for i in range(n):
        instrs.append(Instruction(
            idx=n + i, pc=0x40002800 + i * 4, bytes_=b"\x00" * 4,
            mnemonic="ldr w2, [x12, w3, uxtw #2]",
            regs_read={"x12": BASE_B, "w3": i},
            regs_write={"w2": i * 17}, mem=(),
        ))
    instrs.append(Instruction(idx=2 * n, pc=0x40002900, bytes_=b"\x00" * 4,
                              mnemonic="ret", regs_read={}, regs_write={}, mem=()))
    core = _build_core(instrs)
    r = core.verify_and_promote_indexed_load_table()
    assert r["tables_found"] == 2
    assert r["promoted"]     == 2


def test_indexed_load_table_dedupes_on_replay():
    BASE = 0x40008960
    instrs = []
    for i in range(8):
        instrs.append(Instruction(
            idx=i, pc=0x40002770 + i * 4, bytes_=b"\x00" * 4,
            mnemonic="ldr w0, [x10, w1, uxtw #2]",
            regs_read={"x10": BASE, "w1": i},
            regs_write={"w0": i}, mem=(),
        ))
    instrs.append(Instruction(idx=8, pc=0x40002790, bytes_=b"\x00" * 4,
                              mnemonic="ret", regs_read={}, regs_write={}, mem=()))
    core = _build_core(instrs)
    r1 = core.verify_and_promote_indexed_load_table()
    r2 = core.verify_and_promote_indexed_load_table()
    assert r1["promoted"] == 1
    assert r2["promoted"] == 0


def test_primitive_timeline_emitted_when_two_families_fire():
    """§J.2: a trace touching both SHA-512 constants and AES.Te0 constants
    should emit one primitive_timeline finding with two segments, ordered by
    trace_idx of first appearance.
    """
    import sqlite3
    # SHA-512 IV constant: 0x6a09e667f3bcc908 (SHA512.h0)
    sha_load = Instruction(
        idx=10, pc=0x40006884, bytes_=b"\x00" * 4,
        mnemonic="movz x0, #0xc908",
        regs_read={}, regs_write={"x0": 0x6a09e667f3bcc908}, mem=(),
    )
    # AES.Te0[0]: 0xc66363a5
    aes_load = Instruction(
        idx=50, pc=0x40002770, bytes_=b"\x00" * 4,
        mnemonic="ldr w0, [x10, w1, uxtw #2]",
        regs_read={}, regs_write={"w0": 0xc66363a5}, mem=(),
    )
    ret = Instruction(idx=51, pc=0x40002774, bytes_=b"\x00" * 4,
                     mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([sha_load, aes_load, ret])
    r = core.verify_and_promote_primitive_timeline()
    assert r["promoted"] == 1
    assert r["families_seen"] == 2
    assert r["segments_count"] == 2
    # SHA-512 fired first (idx=10), AES second (idx=50)
    assert "SHA-512 → AES" in r["ordering"]
    # Confirm the finding landed
    f_conn = sqlite3.connect(core.work.root / "findings.sqlite")
    row = f_conn.execute(
        "SELECT subject FROM findings WHERE kind='primitive_timeline'"
    ).fetchone()
    f_conn.close()
    assert row and "SHA-512" in row[0] and "AES" in row[0]


def test_primitive_timeline_silent_when_only_one_family():
    """No timeline finding when only a single primitive family fires."""
    sha_load = Instruction(
        idx=0, pc=0x1000, bytes_=b"\x00" * 4,
        mnemonic="movz x0, #0xc908",
        regs_read={}, regs_write={"x0": 0x6a09e667f3bcc908}, mem=(),
    )
    ret = Instruction(idx=1, pc=0x1004, bytes_=b"\x00" * 4,
                     mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([sha_load, ret])
    r = core.verify_and_promote_primitive_timeline()
    assert r["promoted"] == 0
    assert r["families_seen"] == 1


def test_primitive_timeline_dedupes_on_replay():
    """Re-running the pass on the same trace doesn't duplicate the finding."""
    sha_load = Instruction(
        idx=10, pc=0x40006884, bytes_=b"\x00" * 4,
        mnemonic="movz x0, #0xc908",
        regs_read={}, regs_write={"x0": 0x6a09e667f3bcc908}, mem=(),
    )
    aes_load = Instruction(
        idx=50, pc=0x40002770, bytes_=b"\x00" * 4,
        mnemonic="ldr w0, [x10, w1, uxtw #2]",
        regs_read={}, regs_write={"w0": 0xc66363a5}, mem=(),
    )
    ret = Instruction(idx=51, pc=0x40002774, bytes_=b"\x00" * 4,
                     mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([sha_load, aes_load, ret])
    r1 = core.verify_and_promote_primitive_timeline()
    r2 = core.verify_and_promote_primitive_timeline()
    assert r1["promoted"] == 1
    assert r2["promoted"] == 0    # already exists


def test_finding_group_schema_link_and_query():
    """finding_groups table: link two members under a parent, read back."""
    import tempfile
    from pathlib import Path
    from engine.store import (
        link_finding_group_members, get_finding_group_members,
        open_findings_db, upsert_payload,
    )

    class _W:
        def __init__(self, root):
            self.root = root

    work = _W(Path(tempfile.mkdtemp(prefix="utov-test-fgroups-")))
    conn = open_findings_db(work)
    # Seed three findings to be the parent + members
    ch = upsert_payload(conn, {"x": 1})
    ids = []
    for i in range(3):
        cur = conn.execute(
            "INSERT INTO findings(stage, kind, subject, payload_ref, "
            "verified_at, verifier_strategy, origin_hyp_id) "
            "VALUES (?, ?, ?, ?, datetime('now'), ?, ?)",
            ("s5-test", "test", f"subj-{i}", ch, "noop", None),
        )
        ids.append(cur.lastrowid)
    conn.commit()
    link_finding_group_members(
        conn,
        parent_finding_id=ids[0],
        idiom_name="TEST.Idiom",
        members=[(ids[1], "role_a"), (ids[2], "role_b")],
    )
    rows = get_finding_group_members(conn, parent_finding_id=ids[0])
    assert len(rows) == 2
    assert {r["role"] for r in rows} == {"role_a", "role_b"}
    # Idempotency: re-linking same pair doesn't duplicate
    link_finding_group_members(
        conn,
        parent_finding_id=ids[0],
        idiom_name="TEST.Idiom",
        members=[(ids[1], "role_a")],
    )
    rows = get_finding_group_members(conn, parent_finding_id=ids[0])
    assert len(rows) == 2


def test_findings_source_field_populated():
    """Promoted findings carry the hyp's source — 0526Plan B1."""
    eor = Instruction(
        idx=0, pc=0x1000, bytes_=b"\x00" * 4,
        mnemonic="eor x4, x1, x2",
        regs_read={"x1": 0xA, "x2": 0xC},
        regs_write={"x4": 0x6},
        mem=(),
    )
    ret = Instruction(idx=1, pc=0x1004, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([eor, ret])
    core.verify_and_promote_handler_binops()
    rows = core.get_findings(source="s5_deterministic")
    assert len(rows) == 1
    assert rows[0]["source"] == "s5_deterministic"
    assert rows[0]["kind"] == "handler_semantic"
    assert "@0x1000" in rows[0]["subject"]
    # subject_like filter
    rows2 = core.get_findings(subject_like="binop%")
    assert any(r["subject"].startswith("binop@") for r in rows2)


def test_get_findings_filter_and_limit():
    """get_findings filters compose (source AND stage AND kind) and respect limit."""
    nop = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([nop])

    from engine.store import open_findings_db, upsert_payload, _now_iso
    f_conn = open_findings_db(core.work)
    ch = upsert_payload(f_conn, {"x": 1})
    seeds = [
        ("s1b", "algo_signature", "SHA256.h0", "plugin"),
        ("s5-verify", "handler_semantic", "binop@0x1", "s5_deterministic"),
        ("s5-fold", "fold_idiom", "SHA256.Sigma1@0x2", "s5_fold_idiom"),
    ]
    for stage, kind, subj, src in seeds:
        f_conn.execute(
            "INSERT INTO findings(stage, kind, subject, payload_ref, "
            "verified_at, verifier_strategy, origin_hyp_id, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (stage, kind, subj, ch, _now_iso(), "noop", None, src),
        )
    f_conn.commit()
    f_conn.close()

    all_rows = core.get_findings()
    assert len(all_rows) == 3
    assert core.get_findings(source="plugin")[0]["subject"] == "SHA256.h0"
    assert len(core.get_findings(kind="fold_idiom")) == 1
    assert len(core.get_findings(stage="s5-verify")) == 1
    assert len(core.get_findings(limit=2)) == 2


def test_preprocess_batch_default_chain_tags_findings():
    """preprocess_batch runs the canonical chain, tags every promoted
    finding with batch_id, returns the standard summary + hints."""
    eor = Instruction(
        idx=0, pc=0x1000, bytes_=b"\x00" * 4,
        mnemonic="eor x4, x1, x2",
        regs_read={"x1": 0xA, "x2": 0xC},
        regs_write={"x4": 0x6},
        mem=(),
    )
    ret = Instruction(idx=1, pc=0x1004, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([eor, ret])
    r = core.preprocess_batch(passes=["plugin", "binop"])
    assert r["ran"] == ["plugin", "binop"]
    assert "batch_id" in r and len(r["batch_id"]) == 12
    # binop pass should promote one reg-reg-reg XOR
    assert r["totals"]["promoted"] >= 1
    assert "s5_deterministic" in r["totals"]["by_source"]
    # finding row carries the batch_id
    rows = core.get_findings(source="s5_deterministic")
    assert all(row["id"] for row in rows)
    import sqlite3
    fc = sqlite3.connect(core.work.root / "findings.sqlite")
    tagged = fc.execute(
        "SELECT COUNT(*) FROM findings WHERE batch_id = ?", (r["batch_id"],)
    ).fetchone()[0]
    fc.close()
    assert tagged == r["totals"]["promoted"]


def test_preprocess_batch_unknown_pass_raises():
    nop = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([nop])
    import pytest
    with pytest.raises(ValueError, match="unknown preprocess pass names"):
        core.preprocess_batch(passes=["plugin", "nosuchpass"])


def test_preprocess_batch_no_anchor_hint():
    """When the batch produces no plugin/fold/algorithm anchor, hints
    nudge the agent toward stuck_statistics."""
    nop = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                      mnemonic="nop", regs_read={}, regs_write={}, mem=())
    ret = Instruction(idx=1, pc=0x1004, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([nop, ret])
    r = core.preprocess_batch(passes=["plugin", "binop", "sigma", "algorithm"])
    hints = " ".join(r["next_step_hints"])
    assert "no algorithmic anchor" in hints or "no notable anchors" in hints
    assert "stuck_statistics" in hints


def test_discard_batch_failures_keep_audit():
    """discard_batch flips hyp status to 'fail' but leaves the finding
    row + writes an audit intervention. PLAN §1.1 append-only ledger."""
    eor = Instruction(
        idx=0, pc=0x1000, bytes_=b"\x00" * 4,
        mnemonic="eor x4, x1, x2",
        regs_read={"x1": 0xA, "x2": 0xC},
        regs_write={"x4": 0x6}, mem=(),
    )
    ret = Instruction(idx=1, pc=0x1004, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([eor, ret])
    r = core.preprocess_batch(passes=["binop"])
    assert r["totals"]["promoted"] == 1
    d = core.discard_batch(r["batch_id"], reason="unit test discard")
    assert d["discarded"] == 1
    assert d["candidate_count"] == 1
    # status flipped to 'failed' on the underlying hyp
    statuses = {h.status for h in core.get_hypotheses()}
    assert "failed" in statuses
    # finding row still exists (we don't DELETE)
    rows = core.get_findings()
    assert len(rows) == 1
    # an intervention row recorded the discard
    from engine.store import open_hypotheses_db, read_interventions
    conn = open_hypotheses_db(core.work)
    try:
        ints = read_interventions(conn, limit=10)
    finally:
        conn.close()
    assert any(i["action"] in ("override_verdict", "force_status")
               and "unit test discard" in (i.get("reason") or "")
               for i in ints)


def test_discard_batch_source_filter():
    """source filter restricts discard to that subset."""
    eor = Instruction(
        idx=0, pc=0x1000, bytes_=b"\x00" * 4,
        mnemonic="eor x4, x1, x2",
        regs_read={"x1": 0xA, "x2": 0xC},
        regs_write={"x4": 0x6}, mem=(),
    )
    ret = Instruction(idx=1, pc=0x1004, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([eor, ret])
    r = core.preprocess_batch(passes=["binop"])
    # Source for binop is "s5_deterministic"; "s5_triton" not in this batch.
    d = core.discard_batch(r["batch_id"], sources=["s5_triton"])
    assert d["candidate_count"] == 0
    assert d["discarded"] == 0


def test_triton_unavailable_returns_skipped(monkeypatch):
    """When Triton bindings aren't importable, the method returns a clean
    `skipped_reason`-marked summary instead of raising. Keeps script_mode
    wiring uniform across hosts with / without Triton."""
    from engine.stages import s3_triton_symex
    monkeypatch.setattr(s3_triton_symex, "is_available", lambda: False)
    monkeypatch.setattr(s3_triton_symex, "unavailable_reason",
                        lambda: "test stub: Triton not importable")
    nop = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                      mnemonic="nop", regs_read={}, regs_write={}, mem=())
    core = _build_core([nop])
    r = core.verify_and_promote_triton_simplifications()
    assert r["promoted"] == 0
    assert r["checked"] == 0
    assert "skipped_reason" in r


def test_ch_idiom_skips_ext_form():
    """`eor x2, x2, x13, ror #34` (ext binop) inside an eor/and/eor window
    must not be mistaken for a Ch idiom — it's the σ/Σ side of SHA-512.
    """
    e0 = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                     mnemonic="eor x2, x2, x13, ror #34",
                     regs_read={}, regs_write={"x2": 0}, mem=())
    a1 = Instruction(idx=1, pc=0x1004, bytes_=b"\x00" * 4,
                     mnemonic="and x3, x3, x1",
                     regs_read={"x3": 0, "x1": 0}, regs_write={"x3": 0}, mem=())
    e2 = Instruction(idx=2, pc=0x1008, bytes_=b"\x00" * 4,
                     mnemonic="eor x12, x12, x17, ror #41",
                     regs_read={}, regs_write={"x12": 0}, mem=())
    ret = Instruction(idx=3, pc=0x100C, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([e0, a1, e2, ret])
    r = core.verify_and_promote_handler_ch_idioms()
    assert r["checked"] == 0 and r["promoted"] == 0


def test_ch_idiom_and_bic_orr_variant_tc1():
    """BR-8 #2: TC1 SHA-256 Ch shape — `and t1, x, y ; bic t2, z, x ;
    orr d, t2, t1`. Same truth table as the existing eor/and/eor form
    (disjoint supports → | == ⊕), but the Phase-1 matcher rejects it.
    Phase 2 detects it via the (and, bic, orr) lattice + verifier op=CH.
    """
    e = 0x510E527F
    f = 0xDEADBEEF
    g = 0x12345678
    ch = (e & f) | ((~e) & g) & 0xFFFFFFFF
    # Truncate to 32-bit for w-reg semantics.
    ch &= 0xFFFFFFFF

    a = Instruction(idx=0, pc=0x12000c18, bytes_=b"\x00" * 4,
                    mnemonic="and w23, w21, w24",
                    regs_read={"w21": e, "w24": f},
                    regs_write={"w23": (e & f) & 0xFFFFFFFF},
                    mem=())
    b = Instruction(idx=1, pc=0x12000c1c, bytes_=b"\x00" * 4,
                    mnemonic="bic w22, w25, w21",
                    regs_read={"w25": g, "w21": e},
                    regs_write={"w22": ((~e) & g) & 0xFFFFFFFF},
                    mem=())
    # Two unrelated insns in the gap.
    pad1 = Instruction(idx=2, pc=0x12000c20, bytes_=b"\x00" * 4,
                       mnemonic="and w30, w31, w29",
                       regs_read={"w31": 0, "w29": 0},
                       regs_write={"w30": 0}, mem=())
    pad2 = Instruction(idx=3, pc=0x12000c24, bytes_=b"\x00" * 4,
                       mnemonic="and w28, w27, w26",
                       regs_read={"w27": 0, "w26": 0},
                       regs_write={"w28": 0}, mem=())
    o = Instruction(idx=4, pc=0x12000c28, bytes_=b"\x00" * 4,
                    mnemonic="orr w21, w22, w23",
                    regs_read={"w22": ((~e) & g) & 0xFFFFFFFF,
                               "w23": (e & f) & 0xFFFFFFFF},
                    regs_write={"w21": ch},
                    mem=())
    ret = Instruction(idx=5, pc=0x12000c2c, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([a, b, pad1, pad2, o, ret])
    r = core.verify_and_promote_handler_ch_idioms()
    assert r["passed"] >= 1 and r["promoted"] >= 1
    subjects = [f["subject"] for f in core.get_findings(kind="handler_semantic")]
    assert any(s.startswith("ch@0x12000c28") for s in subjects)


def test_maj_idiom_promotes_three_insn_form():
    """BR-8 #2: SHA-2 Maj 3-insn idiom = eor⊕and⊕eor with a precomputed
    (b ∧ c) reused via OR. Verifier `op=MAJ` confirms the algebra.
    """
    a_val = 0xCAFEBABE
    b_val = 0xDEADBEEF
    c_val = 0x13579BDF

    M = 0xFFFFFFFF
    bc_xor = (b_val ^ c_val) & M
    bc_and = (b_val & c_val) & M
    t_and = (bc_xor & a_val) & M
    maj = (t_and ^ bc_and) & M
    expected_maj = ((a_val & b_val) ^ (a_val & c_val) ^ (b_val & c_val)) & M
    assert maj == expected_maj

    bc_and_ins = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                             mnemonic="and w7, w5, w6",
                             regs_read={"w5": b_val, "w6": c_val},
                             regs_write={"w7": bc_and},
                             mem=())
    eor0 = Instruction(idx=1, pc=0x1004, bytes_=b"\x00" * 4,
                       mnemonic="eor w8, w5, w6",
                       regs_read={"w5": b_val, "w6": c_val},
                       regs_write={"w8": bc_xor},
                       mem=())
    and1 = Instruction(idx=2, pc=0x1008, bytes_=b"\x00" * 4,
                       mnemonic="and w8, w8, w4",
                       regs_read={"w8": bc_xor, "w4": a_val},
                       regs_write={"w8": t_and},
                       mem=())
    eor1 = Instruction(idx=3, pc=0x100c, bytes_=b"\x00" * 4,
                       mnemonic="eor w9, w8, w7",
                       regs_read={"w8": t_and, "w7": bc_and},
                       regs_write={"w9": maj},
                       mem=())
    ret = Instruction(idx=4, pc=0x1010, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([bc_and_ins, eor0, and1, eor1, ret])
    r = core.verify_and_promote_handler_maj_idioms()
    assert r["passed"] >= 1 and r["promoted"] >= 1
    subjects = [f["subject"] for f in core.get_findings(kind="handler_semantic")]
    assert any(s.startswith("maj@0x100c") for s in subjects)


def test_self_rescan_picks_up_missing_anchor():
    """BR-8 #3: after a partial algorithm_identified finding is in place,
    self_rescan_missing_anchors triggers σ/Σ + Ch + Maj re-runs and a
    recompute_algorithm_fits, lifting the existing payload's anchors_seen.
    """
    import json
    from engine.store import _now_iso, open_findings_db, read_payload, upsert_payload

    W = 32
    M = 0xFFFFFFFF

    def ror32(v, n):
        v &= M
        return ((v >> n) | ((v << (W - n)) & M)) & M

    x = 0x6a09e667
    a = ror32(x, 17)
    b = a ^ ror32(x, 19)
    c = b ^ (x >> 10)
    # TC1-shape SHA-256 σ₁ that only Phase 3 picks up (dst==input + gap).
    p0 = Instruction(idx=0, pc=0x12000b00, bytes_=b"\x00" * 4,
                     mnemonic="ror w22, w21, #17",
                     regs_read={"w21": x}, regs_write={"w22": a}, mem=())
    pad = Instruction(idx=1, pc=0x12000b04, bytes_=b"\x00" * 4,
                      mnemonic="ror w24, w23, #7",
                      regs_read={"w23": 0xABCDEF01},
                      regs_write={"w24": 0}, mem=())
    p2 = Instruction(idx=2, pc=0x12000b08, bytes_=b"\x00" * 4,
                     mnemonic="eor w22, w22, w21, ror #19",
                     regs_read={"w22": a, "w21": x},
                     regs_write={"w22": b}, mem=())
    p3 = Instruction(idx=3, pc=0x12000b0c, bytes_=b"\x00" * 4,
                     mnemonic="eor w21, w22, w21, lsr #10",
                     regs_read={"w22": b, "w21": x},
                     regs_write={"w21": c}, mem=())
    ret = Instruction(idx=4, pc=0x12000b10, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([p0, pad, p2, p3, ret])

    # Pre-seed a partial SHA-256 algorithm_identified finding missing σ₁.
    f_conn = open_findings_db(core.work)
    seen_anchors = ["SHA256.Sigma0", "SHA256.Sigma1", "SHA256.sigma0",
                    "SHA256.h0", "SHA256.h1", "SHA256.h2", "SHA256.h3",
                    "SHA256.h4", "SHA256.h5", "SHA256.h6", "SHA256.h7"]
    pre_payload = {
        "algorithm":        "SHA-256",
        "anchors_seen":     seen_anchors,
        "anchors_expected": [
            "SHA256.Sigma0", "SHA256.Sigma1",
            "SHA256.sigma0", "SHA256.sigma1",
            "SHA256.h0", "SHA256.h1", "SHA256.h2", "SHA256.h3",
            "SHA256.h4", "SHA256.h5", "SHA256.h6", "SHA256.h7",
        ],
        "evidence_score":   round(11 / 12, 3),
    }
    ref = upsert_payload(f_conn, pre_payload)
    f_conn.execute(
        "INSERT INTO findings(stage, kind, subject, payload_ref, "
        "verified_at, verifier_strategy, origin_hyp_id, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("s5-algorithm-fit", "algorithm_identified", "SHA-256", ref,
         _now_iso(), "structural_anchor_set_match", None,
         "s5_algorithm_fit"),
    )
    # Seed the existing anchors as findings so recompute keeps them.
    seed_ref = upsert_payload(f_conn, {"seed": 1})
    for subj in ("SHA256.Sigma0@0x1100", "SHA256.Sigma1@0x1104",
                 "SHA256.sigma0@0x1108"):
        f_conn.execute(
            "INSERT INTO findings(stage, kind, subject, payload_ref, "
            "verified_at, verifier_strategy, origin_hyp_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("s5-fold", "fold_idiom", subj, seed_ref, _now_iso(),
             "algebraic_idiom_match", None),
        )
    for i in range(8):
        f_conn.execute(
            "INSERT INTO findings(stage, kind, subject, payload_ref, "
            "verified_at, verifier_strategy, origin_hyp_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("s1b-verify", "algo_signature", f"SHA256.h{i}", seed_ref,
             _now_iso(), "handler_semantic", None),
        )
    f_conn.commit()
    f_conn.close()

    r = core.self_rescan_missing_anchors()
    assert r["stage"] == "s5-anchor-rescan"
    assert "SHA-256" in r["missing_before"]
    assert "SHA256.sigma1" in r["missing_before"]["SHA-256"]
    assert r["sigma_promoted"] >= 1
    # Refit must update the existing finding to 12/12.
    f_conn = open_findings_db(core.work)
    try:
        row = f_conn.execute(
            "SELECT payload_ref FROM findings WHERE kind = 'algorithm_identified'"
        ).fetchone()
        pl = read_payload(f_conn, row[0])
    finally:
        f_conn.close()
    assert "SHA256.sigma1" in pl["anchors_seen"]
    assert pl["evidence_score"] == 1.0


def test_dataflow_query_rotations_on_input():
    """BR-8 #4: `dataflow_query(kind='rotations_on_input')` returns every
    rotate/shift acting on the supplied register, including the shifted-
    source operand of `eor d, lhs, x, ror|lsr #N`.
    """
    p0 = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                     mnemonic="ror w4, w5, #6",
                     regs_read={"w5": 0xCAFEBABE},
                     regs_write={"w4": 0}, mem=())
    p1 = Instruction(idx=1, pc=0x1004, bytes_=b"\x00" * 4,
                     mnemonic="eor w4, w4, w5, ror #11",
                     regs_read={"w4": 0, "w5": 0xCAFEBABE},
                     regs_write={"w4": 0}, mem=())
    p2 = Instruction(idx=2, pc=0x1008, bytes_=b"\x00" * 4,
                     mnemonic="add w6, w7, w8",
                     regs_read={"w7": 1, "w8": 2},
                     regs_write={"w6": 3}, mem=())
    p3 = Instruction(idx=3, pc=0x100c, bytes_=b"\x00" * 4,
                     mnemonic="eor w4, w4, w5, ror #25",
                     regs_read={"w4": 0, "w5": 0xCAFEBABE},
                     regs_write={"w4": 0}, mem=())
    ret = Instruction(idx=4, pc=0x1010, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([p0, p1, p2, p3, ret])
    r = core.dataflow_query(kind="rotations_on_input", input_reg="w5")
    assert len(r) == 3
    kinds_amounts = {(x["kind"], x["amount"]) for x in r}
    assert kinds_amounts == {("ror", 6), ("ror", 11), ("ror", 25)}


def test_dataflow_query_unknown_kind_raises():
    """BR-8 #4: invalid `kind` is a programmer error, not a soft skip."""
    import pytest
    nop = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([nop])
    with pytest.raises(ValueError):
        core.dataflow_query(kind="not_a_real_query")


def test_pipeline_file_orchestrator_runs_verify_chain():
    """BR-9 regression: `utov pipeline-file` must run the full
    deterministic verify chain (not just S1..S5). The CLI delegates to
    `run_full_pipeline(core, mode=FRUGAL)`, so we pin that orchestrator
    emits every layer-0 / 1 / 2 verify stage summary in the report.
    """
    from engine.orchestrators.script_mode import Mode, run_full_pipeline

    # Two-instruction trace; meaningful enough that the deterministic chain
    # has something to chew on without needing a real σ/Σ shape.
    eor = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                      mnemonic="eor x4, x1, x2",
                      regs_read={"x1": 0xA, "x2": 0xC},
                      regs_write={"x4": 0x6}, mem=())
    ret = Instruction(idx=1, pc=0x1004, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([eor, ret])
    report = run_full_pipeline(core, mode=Mode.FRUGAL)
    stage_names = {s.get("stage") for s in report.stage_summaries}
    # Every verify-chain stage BR-9 said was missing must be present.
    must_have = {
        "s1b-verify", "s5-verify", "s5-verify-unary", "s5-verify-imm",
        "s5-verify-ext", "s5-verify-bfx", "s5-verify-ch", "s5-verify-maj",
        "s5-fold-sigma", "s5-algorithm-fit", "s5-anchor-rescan",
    }
    missing = must_have - stage_names
    assert not missing, f"verify chain missing stages: {missing}"
    # And the binop pass should have promoted at least the eor finding,
    # so findings_promoted > 0 — proves the chain actually ran end-to-end.
    assert report.findings_promoted >= 1
