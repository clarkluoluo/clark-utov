"""Level-2 set-up symex runner — the escape-hatch framework.

Pins the behaviour that makes the Level-2 runner NOT whack-a-mole: an
un-modeled instruction is a precise BLOCK + checkpoint (never force-concretized,
never silently skipped); a hand-filled semantics entry is cached and lets the
run continue; the runner conforms to the drive `triton_runner` protocol. The
Triton bulk decoder is exercised by a gated smoke (skipped when Triton absent) —
the framework itself is covered with a deterministic fake decoder.
"""

from __future__ import annotations

import pytest

from engine.setup_symex_runner import (
    InstructionSemantics,
    RunnerResult,
    SemanticsApplyError,
    SemanticsParseError,
    SemanticsTable,
    TritonStepDecoder,
    UnmodeledInstruction,
    build_level2_runner,
    opcode_hex,
    parse_sexpr,
    run_window,
    triton_available,
    triton_unavailable_reason,
    validate_sexpr,
)
from engine.types import Instruction, MemOp


def ins(idx, pc, mnem, *, code=b"\x00\x00\x00\x00", reads=None, writes=None):
    return Instruction(idx=idx, pc=pc, bytes_=code, mnemonic=mnem,
                       regs_read=reads or {}, regs_write=writes or {}, mem=())


class FakeDecoder:
    """Deterministic stand-in for Triton: models the mnemonic heads it is told to,
    rejects the rest (so the escape hatch is exercised), never concretizes."""

    def __init__(self, modeled):
        self.modeled = set(modeled)
        self.expr = ""
        self.hatched = []

    def reset(self, entry):
        regs = entry.get("symbolic_regs") or entry.get("reg_file") or ()
        self.expr = "seed(" + ",".join(regs) + ")"
        self.hatched = []

    def step(self, instr):
        head = instr.mnemonic.split()[0]
        if head in self.modeled:
            self.expr = f"{head}({self.expr})"
            return True
        return False

    def apply_semantics(self, instr, sem):
        self.hatched.append(sem.opcode_hex)
        self.expr = f"hatch[{sem.mnemonic}]({self.expr})"

    def expression(self):
        return self.expr


WIN = (0x1000, 0x10FF)
ENTRY = {"symbolic_regs": ["x0", "x1"]}


# --- module-level / availability --------------------------------------------

def test_module_importable_and_availability_consistent():
    if triton_available():
        assert triton_unavailable_reason() is None
    else:
        assert isinstance(triton_unavailable_reason(), str)


def test_opcode_hex_is_the_raw_bytes():
    assert opcode_hex(ins(0, 0x1000, "mul", code=b"\x11\x22\x33\x44")) == "11223344"


# --- semantics table (the persistent escape-hatch cache) --------------------

def test_semantics_table_lookup_exact_then_mnemonic_fallback():
    sem = InstructionSemantics(opcode_hex="aabbccdd", mnemonic="mul",
                               effects=(("x0", "(bvmul x1 x2)"),))
    t = SemanticsTable([sem])
    assert t.lookup("aabbccdd") is sem                       # exact opcode wins
    assert t.lookup("ffffffff", "mul w0, w1, w2") is sem     # mnemonic family
    assert t.lookup("ffffffff", "and") is None


def test_semantics_table_persist_roundtrip(tmp_path):
    p = tmp_path / "sem.json"
    t = SemanticsTable(path=p)
    t.register(InstructionSemantics("aabbccdd", "mul", (("x0", "(bvmul x1 x2)"),)))
    assert p.exists()
    t2 = SemanticsTable.load(p)
    assert "aabbccdd" in t2 and t2.lookup("aabbccdd").mnemonic == "mul"


# --- run_window: the escape-hatch core --------------------------------------

def test_run_window_all_modeled_yields_expression():
    items = [ins(0, 0x1000, "add", writes={"x0": 0}),
             ins(1, 0x1004, "eor", writes={"x0": 0})]
    res = run_window(FakeDecoder({"add", "eor"}), SemanticsTable(), items,
                     window=WIN, entry=ENTRY)
    assert isinstance(res, RunnerResult)
    assert not res.blocked and res.unmodeled is None
    assert res.steps == 2 and res.modeled == 2 and res.escape_hatch_hits == 0
    assert res.expr_source.startswith("eor(add(seed(x0,x1")


def test_run_window_unmodeled_blocks_precisely_never_concretizes():
    # 'mul' isn't modeled and isn't in the table -> BLOCK at that exact step.
    items = [ins(0, 0x1000, "add", writes={"x0": 0}),
             ins(7, 0x1008, "mul w0, w0, w1", code=b"\x53\x7c\x01\x9b", writes={"x0": 0}),
             ins(8, 0x100c, "eor", writes={"x0": 0})]
    res = run_window(FakeDecoder({"add", "eor"}), SemanticsTable(), items,
                     window=WIN, entry=ENTRY)
    assert res.blocked
    u = res.unmodeled
    assert isinstance(u, UnmodeledInstruction)
    assert u.mnemonic == "mul w0, w0, w1" and u.idx == 7 and u.opcode_hex == "537c019b"
    assert res.expr_source == ""              # nothing emitted on a block
    assert res.modeled == 1 and res.steps == 2  # stopped AT the mul, didn't run eor
    assert "not force-concretized" in u.question.lower() or "not modeled" in u.question


def test_run_window_escape_hatch_lets_it_continue():
    items = [ins(0, 0x1000, "add", writes={"x0": 0}),
             ins(7, 0x1008, "mul w0, w0, w1", code=b"\x53\x7c\x01\x9b", writes={"x0": 0}),
             ins(8, 0x100c, "eor", writes={"x0": 0})]
    # The agent hand-fills the mul's semantics; cached in the table.
    table = SemanticsTable([InstructionSemantics(
        "537c019b", "mul", (("x0", "(bvmul x0 x1)"),))])
    res = run_window(FakeDecoder({"add", "eor"}), table, items, window=WIN, entry=ENTRY)
    assert not res.blocked
    assert res.escape_hatch_hits == 1 and res.modeled == 3
    assert "hatch[mul]" in res.expr_source


def test_run_window_skips_outside_window():
    items = [ins(0, 0x0fff, "add", writes={"x0": 0}),   # below window
             ins(1, 0x1000, "eor", writes={"x0": 0}),   # in window
             ins(2, 0x2000, "mul", writes={"x0": 0})]   # above window (unmodeled, but skipped)
    res = run_window(FakeDecoder({"eor"}), SemanticsTable(), items, window=WIN, entry=ENTRY)
    assert not res.blocked and res.steps == 1 and res.modeled == 1


# --- ① trace-guided: branch skipping + idx-range segment --------------------

@pytest.mark.parametrize("mnem,expected", [
    ("b.hi #0x100c", True), ("b", True), ("bl func", True), ("cbz x0, .L1", True),
    ("ret", True), ("tbnz w0, #3, .L", True),
    ("add x8, x0, x1", False), ("mul w0, w0, w1", False), ("ldr x0, [x1]", False),
])
def test_is_control_flow(mnem, expected):
    from engine.setup_symex_runner import is_control_flow
    assert is_control_flow(mnem) is expected


def test_run_window_skips_branches_trace_guided():
    # The b.hi is NOT handed to the decoder — control flow follows the trace order.
    items = [ins(0, 0x1000, "add", writes={"x0": 0}),
             ins(1, 0x1004, "b.hi #0x100c"),            # branch: skipped, not stepped
             ins(2, 0x1008, "eor", writes={"x0": 0})]
    res = run_window(FakeDecoder({"add", "eor"}), SemanticsTable(), items,
                     window=WIN, entry=ENTRY)
    assert not res.blocked
    assert res.branches_skipped == 1 and res.steps == 2 and res.modeled == 2
    assert res.expr_source.startswith("eor(add(seed")    # only the data steps


def test_run_window_idx_segment_excludes_wrong_pc_occurrence():
    # idx-range segment follows the executed order; a later occurrence of an
    # in-band pc (idx outside the segment) must NOT be pulled in (the b.hi trap).
    items = [ins(61, 0x9530, "add", writes={"x0": 0}),
             ins(62, 0x9534, "b.hi #0x9540"),           # branch in segment: skipped
             ins(63, 0x9538, "eor", writes={"x0": 0}),
             ins(64, 0x953c, "mul", writes={"x0": 0})]  # pc in band, idx OUT of (61,63)
    res = run_window(FakeDecoder({"add", "eor", "mul"}), SemanticsTable(), items,
                     window=(61, 63), entry=ENTRY, window_kind="idx")
    assert not res.blocked
    assert res.branches_skipped == 1 and res.steps == 2     # add + eor only
    assert "mul" not in res.expr_source                     # idx64 excluded


# --- build_level2_runner: drive `triton_runner` protocol --------------------

def _ctx(items):
    return {"entry": ENTRY, "mode": "backward_alias", "window": list(WIN),
            "items": items, "decisions": {}}


def test_level2_runner_surfaces_unmodeled_as_block():
    items = [ins(7, 0x1008, "mul w0, w0, w1", code=b"\x53\x7c\x01\x9b", writes={"x0": 0})]
    runner = build_level2_runner(decoder=FakeDecoder({"add"}), table=SemanticsTable())
    out = runner(_ctx(items))
    assert out["propagated"] is False and out["expr_source"] == ""
    assert out["unmodeled"]["mnemonic"] == "mul w0, w0, w1"
    assert out["unmodeled"]["kind"] == "setup_symex_unmodeled_insn"


def test_level2_runner_returns_expression_and_gold_parity():
    items = [ins(0, 0x1000, "add", writes={"x0": 0})]

    def gold(expr, ctx):
        # the agent's oracle: here a stub that reports a clean multi-vector pass
        assert "add(seed" in expr
        return {"gold_parity": "8/8",
                "parity_vectors": [{"input_key": f"g{i}", "observed": f"o{i}",
                                    "predicted": f"o{i}", "exec_id": f"r{i}"}
                                   for i in range(8)]}

    runner = build_level2_runner(decoder=FakeDecoder({"add"}), gold=gold)
    out = runner(_ctx(items))
    assert out["propagated"] is True and out["gold_parity"] == "8/8"
    assert len(out["parity_vectors"]) == 8


def test_level2_runner_without_gold_has_no_parity():
    items = [ins(0, 0x1000, "add", writes={"x0": 0})]
    out = build_level2_runner(decoder=FakeDecoder({"add"}))(_ctx(items))
    assert out["propagated"] is True and out["expr_source"]
    assert "gold_parity" not in out               # framework never fabricates parity


# --- semantics DSL (pure parser/validator, no Triton) -----------------------

def test_parse_sexpr_structure():
    assert parse_sexpr("x0") == "x0"
    assert parse_sexpr("(bvmul x0 x1)") == ["bvmul", "x0", "x1"]
    assert parse_sexpr("(bvadd (bvand x2 (bv 255 64)) x3)") == \
        ["bvadd", ["bvand", "x2", ["bv", 255, 64]], "x3"]
    assert parse_sexpr("(bv 0x1f 32)") == ["bv", 31, 32]   # hex immediate


@pytest.mark.parametrize("bad", ["", "(", ")", "()", "(bvadd x0", "(bvadd x0 x1) x2"])
def test_parse_sexpr_rejects_malformed(bad):
    with pytest.raises(SemanticsParseError):
        parse_sexpr(bad)


def test_validate_sexpr_accepts_wellformed():
    for ok in ["(bvmul x0 x1)", "(extract 31 0 x0)", "(zx 32 (extract 31 0 x0))",
               "(concat x0 x1)", "(bvnot x5)", "(bvadd (bv 1 64) x0)"]:
        validate_sexpr(parse_sexpr(ok))            # must not raise


@pytest.mark.parametrize("bad", [
    "5",                       # bare immediate — width must be explicit
    "(bvadd x0)",              # wrong arity
    "(bv 1)",                  # bv needs value + size
    "(unknownop x0 x1)",       # unknown op
    "(bvadd x0 5)",            # bare immediate operand
])
def test_validate_sexpr_rejects(bad):
    with pytest.raises(SemanticsParseError):
        validate_sexpr(parse_sexpr(bad))


class RaisingDecoder(FakeDecoder):
    def apply_semantics(self, instr, sem):
        raise SemanticsApplyError(f"boom: {sem.opcode_hex}")


def test_level2_runner_surfaces_semantics_error():
    # A malformed / un-injectable hand fill is surfaced precisely, not emitted.
    items = [ins(7, 0x1008, "mul w0, w0, w1", code=b"\x53\x7c\x01\x9b", writes={"x0": 0})]
    table = SemanticsTable([InstructionSemantics("537c019b", "mul", (("x0", "(bogus)"),))])
    out = build_level2_runner(decoder=RaisingDecoder({"add"}), table=table)(_ctx(items))
    assert out["propagated"] is False and "semantics_error" in out


# --- Triton bulk decoder + real semantics injection (gated) ------------------

def test_triton_decoder_requires_triton():
    if triton_available():
        dec = TritonStepDecoder(output_reg="x0")
        dec.reset({"symbolic_regs": ["x0", "x1"]})
        # MOV W0, #1  (AArch64) — a trivially decodable opcode; just must not raise.
        dec.step(ins(0, 0x1000, "mov w0, #1", code=b"\x20\x00\x80\x52", writes={"x0": 1}))
        assert isinstance(dec.expression(), str)
    else:
        with pytest.raises(RuntimeError):
            TritonStepDecoder()


requires_triton = pytest.mark.skipif(
    not triton_available(), reason="Triton bindings not installed on host")


@requires_triton
def test_triton_apply_semantics_injects_into_symbolic_state():
    # ① the real injection: a hand-filled (bvadd x0 x1) must DRIVE x2's symbolic
    # value — not a stub, not a concretize. Verify behaviorally by evaluating.
    dec = TritonStepDecoder(output_reg="x2")
    dec.reset({"symbolic_regs": ["x0", "x1"]})
    ctx = dec._ctx
    ctx.setConcreteRegisterValue(ctx.getRegister("x0"), 7)
    ctx.setConcreteRegisterValue(ctx.getRegister("x1"), 5)
    dec.apply_semantics(
        ins(0, 0x1000, "add"),
        InstructionSemantics("00000000", "add", (("x2", "(bvadd x0 x1)"),)))
    assert ctx.getSymbolicRegisterValue(ctx.getRegister("x2")) == 12
    assert dec.expression()                            # x2's recovered AST, non-empty


@requires_triton
def test_triton_apply_semantics_nested_expression():
    dec = TritonStepDecoder(output_reg="x2")
    dec.reset({"symbolic_regs": ["x0", "x1"]})
    ctx = dec._ctx
    ctx.setConcreteRegisterValue(ctx.getRegister("x0"), 0x1FF)   # 511
    ctx.setConcreteRegisterValue(ctx.getRegister("x1"), 5)
    # (x0 & 0xff) + x1 = 255 + 5 = 260
    dec.apply_semantics(
        ins(0, 0x1000, "and+add"),
        InstructionSemantics("00000001", "udf",
                             (("x2", "(bvadd (bvand x0 (bv 255 64)) x1)"),)))
    assert ctx.getSymbolicRegisterValue(ctx.getRegister("x2")) == 260


@requires_triton
def test_triton_apply_semantics_bad_register_raises():
    dec = TritonStepDecoder()
    dec.reset({"symbolic_regs": ["x0"]})
    with pytest.raises(SemanticsApplyError):
        dec.apply_semantics(
            ins(0, 0x1000, "x"),
            InstructionSemantics("00000002", "x", (("x0", "(bvadd x0 znot_a_reg)"),)))


# --- handler10 fixture: full-seed + idx-segment + branch-skip propagation ----
#
# A handler10-shaped segment (window idx 61–113 + a b.hi side path). The recovered
# symbolic x8 must propagate from the FULL input along the trace path and evaluate
# to the truth 0xfb9881b1 — not the "exit all concrete, x8 == 0" of a one-register
# seed, and not the wrong value a pc-band would give by pulling in a later pc.
_ADD_X8_X0_X1 = b"\x08\x00\x01\x8b"      # add x8, x0, x1
_ADD_X8_X8_X3 = b"\x08\x01\x03\x8b"      # add x8, x8, x3
_B_HI = b"\x48\x00\x00\x54"              # b.hi <off>

_H10_ITEMS = [
    Instruction(idx=61, pc=0x9530, bytes_=_ADD_X8_X0_X1, mnemonic="add x8, x0, x1",
                regs_read={"x0": 0, "x1": 0}, regs_write={"x8": 0}, mem=()),
    Instruction(idx=62, pc=0x9534, bytes_=_B_HI, mnemonic="b.hi #0x9540",
                regs_read={}, regs_write={}, mem=()),
    Instruction(idx=63, pc=0x9538, bytes_=_ADD_X8_X8_X3, mnemonic="add x8, x8, x3",
                regs_read={"x8": 0, "x3": 0}, regs_write={"x8": 0}, mem=()),
    # a LATER occurrence of an in-band pc, OUTSIDE the idx segment (61–63):
    Instruction(idx=64, pc=0x953c, bytes_=_ADD_X8_X8_X3, mnemonic="add x8, x8, x3",
                regs_read={"x8": 0, "x3": 0}, regs_write={"x8": 0}, mem=()),
]
_H10_ENTRY = {"symbolic_regs": ["x0", "x1", "x3"],
              "concrete_regs": {"x0": 0xFB000000, "x1": 0x00988000, "x3": 0x000001B1}}
_H10_TRUTH = 0xFB9881B1


def _h10_x8(dec):
    ctx = dec._ctx
    return ctx.getSymbolicRegisterValue(ctx.getRegister("x8")), \
        ctx.getRegisterAst(ctx.getRegister("x8")).isSymbolized()


@requires_triton
def test_handler10_full_seed_idx_segment_propagates_to_truth():
    dec = TritonStepDecoder(output_reg="x8")
    res = run_window(dec, SemanticsTable(), _H10_ITEMS,
                     window=(61, 63), entry=_H10_ENTRY, window_kind="idx")
    assert not res.blocked and res.branches_skipped == 1     # b.hi taken from trace
    assert dec.seeded_regs == ("x0", "x1", "x3") and dec.unseeded_regs == ()
    value, symbolic = _h10_x8(dec)
    assert value == _H10_TRUTH                               # x0+x1+x3, full input
    assert symbolic                                          # a function of the inputs


@requires_triton
def test_handler10_one_register_seed_gives_concrete_zero():
    # ② before-fix: seed nothing → inputs default to 0 → x8 == 0 ("symbolic=0").
    dec = TritonStepDecoder(output_reg="x8")
    run_window(dec, SemanticsTable(), _H10_ITEMS,
               window=(61, 63), entry={}, window_kind="idx")
    value, _ = _h10_x8(dec)
    assert value == 0


@requires_triton
def test_handler10_pc_band_pulls_in_wrong_occurrence():
    # ① before-fix: a pc-band grabs idx64 too → x8 = truth + x3 (wrong). The idx
    # segment is what excludes it.
    dec = TritonStepDecoder(output_reg="x8")
    run_window(dec, SemanticsTable(), _H10_ITEMS,
               window=(0x9530, 0x9540), entry=_H10_ENTRY, window_kind="pc")
    value, _ = _h10_x8(dec)
    assert value == (_H10_TRUTH + 0x1B1) & 0xFFFFFFFFFFFFFFFF      # idx64 leaked in


@requires_triton
def test_handler10_runner_surfaces_seed_and_branch_info():
    runner = build_level2_runner(decoder=TritonStepDecoder(output_reg="x8"))
    out = runner({"entry": _H10_ENTRY, "window": [61, 63], "window_kind": "idx",
                  "items": _H10_ITEMS, "decisions": {}})
    assert out["propagated"] is True
    assert out["seeded_regs"] == ["x0", "x1", "x3"] and out["unseeded_regs"] == []
    assert out["branches_skipped"] == 1


# --- concolic memory shadow: un-symbolized reads take the trace's real value --
#
# F0 type10: window_kind fixed the segment (53 items, mem_live_in surfaced) yet
# emitted_F was still "0". Root cause: forward Triton collapsed at
# `idx107 mul w8,w9,w8` because `idx106 ldr w9,[x24,#0xa8]` read Triton's
# uninitialised 0 (oracle truth 0x388793e9) — that slot is intra-handler state
# (a window-internal value, not a symbolic input). Registers had a concrete
# shadow; memory did not. These fixtures pin the memory half of the shadow.
def _insm(idx, pc, code, mnem, *, reads=None, writes=None, mem=()):
    return Instruction(idx=idx, pc=pc, bytes_=code, mnemonic=mnem,
                       regs_read=reads or {}, regs_write=writes or {}, mem=tuple(mem))


# x24 base chosen so x24 + 0xa8 == 0x13065f48 (the diagnosed load address): the
# concrete base makes Triton compute the SAME effective address the trace records,
# so the shadow seeded at the trace addr is the address the ldr actually reads.
_X24 = 0x13065EA0
_MEM_ADDR = _X24 + 0xA8                    # 0x13065f48
_MEM_TRUTH = 0x388793E9                    # the slot's ground-truth value
_LDR_W9 = b"\x09\xab\x40\xb9"              # ldr w9, [x24, #0xa8]
_MUL_W8 = b"\x28\x7d\x08\x1b"              # mul w8, w9, w8
_STR_W0 = b"\x00\xab\x00\xb9"              # str w0, [x24, #0xa8]
_LDR_W10 = b"\x0a\xaf\x40\xb9"             # ldr w10, [x24, #0xac]

_T10_ITEMS = [
    _insm(106, 0x13065F44, _LDR_W9, "ldr w9, [x24, #0xa8]",
          reads={"x24": _X24}, writes={"x9": _MEM_TRUTH},
          mem=[MemOp("r", _MEM_ADDR, _MEM_TRUTH, 4)]),
    _insm(107, 0x13065F48, _MUL_W8, "mul w8, w9, w8",
          reads={"w9": _MEM_TRUTH, "w8": 5}, writes={"w8": (_MEM_TRUTH * 5) & 0xFFFFFFFF}),
]
# entry: w8 is the symbolic input (shadow 5); x24 is the pinned base (concrete).
_T10_ENTRY = {"symbolic_regs": ["x8"], "concrete_regs": {"x24": _X24, "x8": 5}}

# Register-trace form of the SAME scenario: the F0 reality — the ldr carries NO
# MemOp (mem[] is sparse), so the loaded value lives ONLY in regs_write["w9"].
# This is what the per-step mem shadow could not see (shadowed_mem_reads stayed 0)
# and what register reconciliation recovers. NO hand-filled MemOp anywhere.
_T10_REGTRACE_ITEMS = [
    _insm(106, 0x13065F44, _LDR_W9, "ldr w9, [x24, #0xa8]",
          reads={"x24": _X24}, writes={"w9": _MEM_TRUTH}, mem=()),
    _insm(107, 0x13065F48, _MUL_W8, "mul w8, w9, w8",
          reads={"w9": _MEM_TRUTH, "w8": 5}, writes={"w8": (_MEM_TRUTH * 5) & 0xFFFFFFFF}),
]


class _NoConcolicShadow(TritonStepDecoder):
    """Both off-switches: no mem shadow, no register reconciliation — the
    pre-fix behaviour (an un-seeded load reads Triton's 0)."""

    def _shadow_concrete_reads(self, ins):
        return

    def _reconcile_concrete_regs(self, ins):
        return


@requires_triton
def test_concolic_mem_shadow_keeps_mul_alive():
    # WITH the shadow: idx106 reads 0x388793e9 (not 0), idx107 mul does NOT
    # collapse — w8 stays a symbolic function of the input, value != 0.
    dec = TritonStepDecoder(output_reg="x8")
    res = run_window(dec, SemanticsTable(), _T10_ITEMS,
                     window=(106, 107), entry=_T10_ENTRY, window_kind="idx")
    assert not res.blocked
    ctx = dec._ctx
    w8 = ctx.getRegister("x8")
    assert ctx.getRegisterAst(w8).isSymbolized()                       # F is a fn of input
    assert ctx.getConcreteRegisterValue(w8) == (_MEM_TRUTH * 5) & 0xFFFFFFFF
    assert ctx.getConcreteRegisterValue(w8) != 0
    assert dec.shadowed_mem_reads == 1                                 # the one un-symbolized read
    assert res.expr_source                                            # non-empty recovered expr


@requires_triton
def test_without_shadow_mul_collapses_to_zero():
    # CONTRAST: disable BOTH shadows → idx106 reads Triton's 0 → mul → 0 (emit "0").
    dec = _NoConcolicShadow(output_reg="x8")
    run_window(dec, SemanticsTable(), _T10_ITEMS,
               window=(106, 107), entry=_T10_ENTRY, window_kind="idx")
    ctx = dec._ctx
    assert ctx.getConcreteRegisterValue(ctx.getRegister("x8")) == 0    # the collapse


@requires_triton
def test_register_reconciliation_keeps_mul_alive_on_regtrace():
    # THE headline: a register-trace (empty mem[]) — the mem shadow has no data
    # source (shadowed_mem_reads==0) yet reconciliation recovers w9 from
    # regs_write, so idx107 mul does NOT collapse. No hand-filled MemOp.
    dec = TritonStepDecoder(output_reg="x8")
    res = run_window(dec, SemanticsTable(), _T10_REGTRACE_ITEMS,
                     window=(106, 107), entry=_T10_ENTRY, window_kind="idx")
    assert not res.blocked
    ctx = dec._ctx
    w8 = ctx.getRegister("x8")
    assert ctx.getRegisterAst(w8).isSymbolized()                       # F is a fn of input
    assert ctx.getConcreteRegisterValue(w8) == (_MEM_TRUTH * 5) & 0xFFFFFFFF
    assert dec.shadowed_mem_reads == 0                                 # mem[] was empty
    assert dec.shadowed_reg_writes >= 1                                # w9 recovered from regs_write
    assert res.expr_source


@requires_triton
def test_regtrace_collapses_without_reconciliation():
    # CONTRAST on the reg-trace form: with both off-switches, w9 has no source
    # (empty mem[]) → reads 0 → mul collapses. Proves reconciliation is load-bearing.
    dec = _NoConcolicShadow(output_reg="x8")
    run_window(dec, SemanticsTable(), _T10_REGTRACE_ITEMS,
               window=(106, 107), entry=_T10_ENTRY, window_kind="idx")
    ctx = dec._ctx
    assert ctx.getConcreteRegisterValue(ctx.getRegister("x8")) == 0


@requires_triton
def test_register_reconciliation_preserves_input_tainted_writes():
    # No over-concretize: a register Triton kept SYMBOLIC (an input-dependent
    # result) is LEFT alone — reconciliation must not clobber it with the trace's
    # scalar, or the input path is lost. Here w8 = w0 (symbolic) stays symbolic
    # even though regs_write carries a concrete scalar for it.
    add_w8_w0 = b"\x08\x00\x00\x0b"            # add w8, w0, w0
    items = [
        _insm(0, 0x1000, add_w8_w0, "add w8, w0, w0",
              reads={"w0": 7}, writes={"w8": 14}),   # trace scalar 14, but w8 is symbolic
    ]
    dec = TritonStepDecoder(output_reg="x8")
    run_window(dec, SemanticsTable(), items, window=(0, 0),
               entry={"symbolic_regs": ["x0"], "concrete_regs": {"x0": 7}},
               window_kind="idx")
    ctx = dec._ctx
    w8 = ctx.getRegister("x8")
    assert ctx.getRegisterAst(w8).isSymbolized()        # input-tainted → preserved
    assert dec.shadowed_reg_writes == 0                 # nothing reconciled (all symbolic)


# --- generality: the shadow is self-adaptive across trace FORMS, not curve-fit -
# The data source follows the trace's shape — mem[].val when present, regs_write
# otherwise, both when both, and a load with NEITHER is surfaced (never silent).
# Zero case-specific knowledge: the engine keys only off "non-symbolic + a trace
# value exists", with a load-with-no-source counter for observability.

@requires_triton
def test_shadow_mixed_trace_forms_each_field_drives_its_own_path():
    # Form C (mixed): idx0 carries a MemOp (memory path), idx1 has an EMPTY mem[]
    # but a regs_write value (register path), idx2 is the symbolic mul. Each field
    # drives its own shadow; the input-tainted w8 is preserved; nothing runs blind.
    v9, v10 = _MEM_TRUTH, 0x11223344
    a, b = _X24 + 0xA8, _X24 + 0xAC
    items = [
        _insm(0, 0x1000, _LDR_W9, "ldr w9, [x24, #0xa8]",
              reads={"x24": _X24}, writes={"w9": v9},
              mem=[MemOp("r", a, v9, 4)]),                       # B-form step: mem[] present
        _insm(1, 0x1004, _LDR_W10, "ldr w10, [x24, #0xac]",
              reads={"x24": _X24}, writes={"w10": v10}, mem=()),  # A-form step: reg only
        _insm(2, 0x1008, _MUL_W8, "mul w8, w9, w8",
              reads={"w9": v9, "w8": 5}, writes={"w8": (v9 * 5) & 0xFFFFFFFF}),
    ]
    dec = TritonStepDecoder(output_reg="x8")
    run_window(dec, SemanticsTable(), items, window=(0, 2),
               entry={"symbolic_regs": ["x8"], "concrete_regs": {"x24": _X24, "x8": 5}},
               window_kind="idx")
    ctx = dec._ctx
    assert ctx.getConcreteRegisterValue(ctx.getRegister("x9")) == v9     # via mem path
    assert ctx.getConcreteRegisterValue(ctx.getRegister("x10")) == v10   # via reg path
    assert ctx.getRegisterAst(ctx.getRegister("x8")).isSymbolized()      # input preserved
    assert dec.shadowed_mem_reads == 1          # only idx0 had a MemOp
    assert dec.shadowed_reg_writes >= 2         # idx0 w9 + idx1 w10 (idx2 w8 symbolic → skipped)
    assert dec.unshadowed_steps == 0            # every load had a source


@requires_triton
def test_blind_load_with_no_trace_source_is_surfaced_not_silent():
    # Form D: a load with NEITHER a mem[] value NOR a regs_write entry — no trace
    # ground-truth exists, so it runs on Triton's 0. This MUST be counted (the F0
    # false-green was exactly this going unnoticed), never silently produce "0".
    items = [
        _insm(0, 0x13065F44, _LDR_W9, "ldr w9, [x24, #0xa8]",
              reads={"x24": _X24}, writes={}, mem=()),          # no source at all
        _insm(1, 0x13065F48, _MUL_W8, "mul w8, w9, w8",
              reads={"w9": 0, "w8": 5}, writes={"w8": 0}),
    ]
    dec = TritonStepDecoder(output_reg="x8")
    run_window(dec, SemanticsTable(), items, window=(0, 1), entry=_T10_ENTRY,
               window_kind="idx")
    assert dec.unshadowed_steps == 1            # the blind load is visible
    assert dec.shadowed_mem_reads == 0 and dec.shadowed_reg_writes == 0
    # and it did collapse (the honest consequence) — the counter is the warning.
    assert dec._ctx.getConcreteRegisterValue(dec._ctx.getRegister("x8")) == 0


@requires_triton
def test_clean_forms_report_zero_unshadowed_steps():
    # Regression guard: the A (reg) and B (mem) forms each have a source for their
    # load, so unshadowed_steps stays 0 — the counter only fires on a real blind load.
    for items in (_T10_ITEMS, _T10_REGTRACE_ITEMS):
        dec = TritonStepDecoder(output_reg="x8")
        run_window(dec, SemanticsTable(), items, window=(106, 107),
                   entry=_T10_ENTRY, window_kind="idx")
        assert dec.unshadowed_steps == 0


@requires_triton
def test_level2_runner_surfaces_shadow_counters():
    # All three concolic counters are surfaced for the agent to read.
    runner = build_level2_runner(decoder=TritonStepDecoder(output_reg="x8"))
    out = runner({"entry": _T10_ENTRY, "window": [106, 107], "window_kind": "idx",
                  "items": _T10_REGTRACE_ITEMS, "decisions": {}})
    assert out["propagated"] is True
    assert out["shadowed_mem_reads"] == 0
    assert out["shadowed_reg_writes"] >= 1
    assert out["unshadowed_steps"] == 0


@requires_triton
def test_concolic_shadow_does_not_overconcretize_input_derived_mem():
    # A symbolic store then a load of the SAME address stays symbolic (input-
    # derived → F is still a function of the input); a never-written address is
    # trace-concrete. The shadow must distinguish the two via isMemorySymbolized.
    a = _X24 + 0xA8
    b = _X24 + 0xAC
    items = [
        _insm(0, 0x1000, _STR_W0, "str w0, [x24, #0xa8]",
              reads={"x0": 0x11, "x24": _X24}, mem=[MemOp("w", a, 0x11, 4)]),
        _insm(1, 0x1004, _LDR_W9, "ldr w9, [x24, #0xa8]",
              reads={"x24": _X24}, writes={"x9": 0x11}, mem=[MemOp("r", a, 0x11, 4)]),
        _insm(2, 0x1008, _LDR_W10, "ldr w10, [x24, #0xac]",
              reads={"x24": _X24}, writes={"w10": 0xCAFEBABE}, mem=[MemOp("r", b, 0xCAFEBABE, 4)]),
    ]
    dec = TritonStepDecoder(output_reg="x9")
    run_window(dec, SemanticsTable(), items,
               window=(0, 2), entry={"symbolic_regs": ["x0"],
                                     "concrete_regs": {"x24": _X24, "x0": 0x11}},
               window_kind="idx")
    ctx = dec._ctx
    # input-derived (symbolic store) → still symbolic; the shadow LEFT it alone.
    assert ctx.getRegisterAst(ctx.getRegister("x9")).isSymbolized()
    # never-written slot → trace concrete value, not symbolic, not 0.
    w10 = ctx.getRegister("x10")
    assert not ctx.getRegisterAst(w10).isSymbolized()
    assert ctx.getConcreteRegisterValue(w10) == 0xCAFEBABE
    assert dec.shadowed_mem_reads == 1                  # only slot b was shadowed


@requires_triton
def test_level2_runner_surfaces_shadowed_mem_read_count():
    runner = build_level2_runner(decoder=TritonStepDecoder(output_reg="x8"))
    out = runner({"entry": _T10_ENTRY, "window": [106, 107], "window_kind": "idx",
                  "items": _T10_ITEMS, "decisions": {}})
    assert out["propagated"] is True
    assert out["shadowed_mem_reads"] == 1


@requires_triton
def test_concrete_backing_mem_seeded_upfront_from_entry():
    # The backing flow internalized: concrete_mem (region bytes) handed in the
    # entry dict are upfront-seeded so a load with an EMPTY trace mem[] (reg-trace
    # runner) still reads real bytes, not Triton's 0. Here idx0's ldr carries NO
    # MemOp, so only the upfront concrete_mem seed can supply the value.
    le_truth = (_MEM_TRUTH).to_bytes(4, "little")
    entry = {
        "symbolic_regs": ["x8"],
        "concrete_regs": {"x24": _X24, "x8": 5},
        "concrete_mem": [{"addr": _MEM_ADDR, "size": 4, "data_hex": le_truth.hex()}],
    }
    items = [
        _insm(0, 0x13065F44, _LDR_W9, "ldr w9, [x24, #0xa8]",
              reads={"x24": _X24}, writes={"w9": 0}, mem=()),       # empty mem[]
        _insm(1, 0x13065F48, _MUL_W8, "mul w8, w9, w8",
              reads={"w9": 0, "w8": 5}, writes={"w8": 0}),
    ]

    # Isolate the UPFRONT-mem path: disable register reconciliation so only the
    # concrete_mem seed can supply w9 (regs_write["w9"] is the placeholder 0 here).
    class _NoRegReconcile(TritonStepDecoder):
        def _reconcile_concrete_regs(self, ins):
            return

    dec = _NoRegReconcile(output_reg="x8")
    run_window(dec, SemanticsTable(), items, window=(0, 1), entry=entry, window_kind="idx")
    ctx = dec._ctx
    w8 = ctx.getRegister("x8")
    assert ctx.getConcreteRegisterValue(w8) == (_MEM_TRUTH * 5) & 0xFFFFFFFF
    assert ctx.getRegisterAst(w8).isSymbolized()


@requires_triton
def test_level2_escape_hatch_end_to_end_with_triton():
    # The whole loop with the REAL decoder: Triton models 'mov', a chosen opcode is
    # forced into the long tail (Triton's blind spot), the table's hand-filled
    # semantics is INJECTED into the live symbolic state, and the run completes.
    class TritonForceUnmodeled(TritonStepDecoder):
        def step(self, instr):
            if opcode_hex(instr) == "1f2003d5":        # treat NOP as un-modeled here
                return False
            return super().step(instr)

    items = [
        ins(0, 0x1000, "mov x0, #3", code=b"\x60\x00\x80\xd2", writes={"x0": 3}),
        ins(1, 0x1004, "nop", code=b"\x1f\x20\x03\xd5", writes={"x2": 0}),
    ]
    table = SemanticsTable([InstructionSemantics(
        "1f2003d5", "nop", (("x2", "(bvadd x0 (bv 4 64))"),))])    # x2 = x0 + 4
    runner = build_level2_runner(
        table=table, decoder=TritonForceUnmodeled(output_reg="x2"))
    out = runner({"entry": {"symbolic_regs": ["x0"]}, "window": [0x1000, 0x10FF],
                  "items": items, "decisions": {}})
    assert out["propagated"] is True and out["escape_hatch_hits"] == 1
    assert out["expr_source"]                          # x2 recovered via the hatch


# --- handler11 fixture: external memory input symbolized via symbolic_mem -----
#
# h11's real input enters through ldr (carrier byte), invisible to register seed.
# Without symbolizing that memory the load reads concrete 0 → exit collapses to 0
# (symbolic=0). symbolic_mem symbolizes it → the value propagates along the trace.
_LDR_X0_X9 = b"\x20\x01\x40\xf9"        # ldr x0, [x9]
_H11_BASE = 0x90000
_H11_TRUTH = 0xFB9881B1


def _h11_items():
    return [Instruction(idx=0, pc=0x1000, bytes_=_LDR_X0_X9, mnemonic="ldr x0, [x9]",
                        regs_read={"x9": _H11_BASE}, regs_write={"x0": 0}, mem=())]


@requires_triton
def test_handler11_external_mem_input_symbolized_propagates_to_truth():
    dec = TritonStepDecoder(output_reg="x0")
    entry = {"symbolic_regs": [], "concrete_regs": {"x9": _H11_BASE},
             "symbolic_mem": [{"addr": _H11_BASE, "size": 8, "value": _H11_TRUTH}]}
    res = run_window(dec, SemanticsTable(), _h11_items(),
                     window=(0x1000, 0x10FF), entry=entry)
    assert not res.blocked and dec.seeded_mem == ((_H11_BASE, 8),)
    ctx = dec._ctx
    assert ctx.getSymbolicRegisterValue(ctx.getRegister("x0")) == _H11_TRUTH
    assert ctx.getRegisterAst(ctx.getRegister("x0")).isSymbolized()


@requires_triton
def test_handler11_unsymbolized_mem_input_is_concrete_zero():
    # before-fix: no symbolic_mem → the load reads concrete 0 → x0 == 0, not symbolic.
    dec = TritonStepDecoder(output_reg="x0")
    entry = {"symbolic_regs": [], "concrete_regs": {"x9": _H11_BASE}}
    run_window(dec, SemanticsTable(), _h11_items(), window=(0x1000, 0x10FF), entry=entry)
    ctx = dec._ctx
    assert ctx.getSymbolicRegisterValue(ctx.getRegister("x0")) == 0
    assert not ctx.getRegisterAst(ctx.getRegister("x0")).isSymbolized()


# --- opaque-staging Phase 1 / 2(i): symbolic store→load forwarding ------------
#
# dev-symbolic-input-through-opaque-staging.md / dev-opaque-staging-phases-impl-spec.md.
# A symbolic value stored to a staging slot must FORWARD to a later load of that
# slot — the load reads back the SYMBOLIC value, not the trace's concrete byte, so
# F stays a function of the input. The runner's `_symbolic_staging` interval set
# (seeded in reset() from symbolic_mem + resolved chain landings, grown in step()
# on each symbolic store) makes `_shadow_concrete_reads` SKIP injecting a concrete
# value over a read that hits a symbolic staging interval. Synthetic, zero
# case-specific (every address is a fixture constant exercising the mechanism).
_OS_STAGING = 0x10020
_OS_STR_X8 = b"\x48\x01\x00\xf9"      # str x8, [x10]
_OS_LDR_X9 = b"\x49\x01\x40\xf9"      # ldr x9, [x10]


@requires_triton
def test_opaque_staging_symbolic_store_forwards_to_later_load():
    # Phase 1: x8 (symbolic input) is stored to [x10] (concrete base) then loaded
    # back into x9. x9 must stay symbolic (forwarded), and the store landing is
    # recorded in _symbolic_staging. The load carries a trace mem[] read op, so
    # WITHOUT the skip _shadow_concrete_reads would clobber it — the interval keeps it.
    items = [
        _insm(0, 0x1000, _OS_STR_X8, "str x8, [x10]",
              reads={"x8": 0x41, "x10": _OS_STAGING},
              mem=[MemOp("w", _OS_STAGING, 0x41, 8)]),
        _insm(1, 0x1004, _OS_LDR_X9, "ldr x9, [x10]",
              reads={"x10": _OS_STAGING}, writes={"x9": 0x41},
              mem=[MemOp("r", _OS_STAGING, 0x41, 8)]),
    ]
    dec = TritonStepDecoder(output_reg="x9")
    entry = {"symbolic_regs": ["x8"], "concrete_regs": {"x10": _OS_STAGING}}
    run_window(dec, SemanticsTable(), items, window=(0, 1), entry=entry,
               window_kind="idx")
    ctx = dec._ctx
    assert ctx.getRegisterAst(ctx.getRegister("x9")).isSymbolized()   # forwarded
    assert (_OS_STAGING, 8) in dec._symbolic_staging                  # recorded
    assert dec.unshadowed_steps == 0                                  # the load had a source


@requires_triton
def test_opaque_staging_phase2i_symbolic_address_forwards_by_op_addr():
    # Phase 2(i): the load EA register x10 is itself input-derived (symbolic). The
    # forwarding is keyed on the trace's concrete op.addr (address concolic), so the
    # symbolic store still forwards its value (value symbolic). x9 stays a function
    # of the input even though the address is symbolic.
    add_x10 = b"\x4a\x01\x0b\x8b"     # add x10, x10, x11  (x10 derived in-window)
    items = [
        _insm(0, 0x1000, add_x10, "add x10, x10, x11",
              reads={"x10": 0, "x11": _OS_STAGING}, writes={"x10": _OS_STAGING}),
        _insm(1, 0x1004, _OS_STR_X8, "str x8, [x10]",
              reads={"x8": 0x41, "x10": _OS_STAGING},
              mem=[MemOp("w", _OS_STAGING, 0x41, 8)]),
        _insm(2, 0x1008, _OS_LDR_X9, "ldr x9, [x10]",
              reads={"x10": _OS_STAGING}, writes={"x9": 0x41},
              mem=[MemOp("r", _OS_STAGING, 0x41, 8)]),
    ]
    dec = TritonStepDecoder(output_reg="x9")
    # x8 symbolic input; x10/x11 carry concrete shadows so Triton resolves the real
    # EA (concolic), x8's symbol is what must reach x9.
    entry = {"symbolic_regs": ["x8"],
             "concrete_regs": {"x10": 0, "x11": _OS_STAGING}}
    run_window(dec, SemanticsTable(), items, window=(0, 2), entry=entry,
               window_kind="idx")
    ctx = dec._ctx
    assert ctx.getRegisterAst(ctx.getRegister("x9")).isSymbolized()   # value symbolic
    assert (_OS_STAGING, 8) in dec._symbolic_staging


class _NoSymbolicStaging(TritonStepDecoder):
    """Disable the opaque-staging interval skip (and the store recording) — the
    pre-Phase-1 behaviour. Forwarding then relies ONLY on single-point
    isMemorySymbolized, which misses a staging slot Triton symbolized at a
    different concrete address (symbolic store EA with no resolvable shadow)."""

    def _intersects_symbolic_staging(self, addr, size):
        return False

    def _record_symbolic_stores(self, ins):
        return


@requires_triton
def test_opaque_staging_contrast_symbolic_store_ea_collapses_without_skip():
    # The genuine forwarding break: the store EA (x10) is symbolic with NO concrete
    # shadow, so Triton lands the symbolic bytes at a WRONG concrete address; at the
    # load's trace op.addr isMemorySymbolized is False → the read is concretized →
    # taint dies. The _symbolic_staging interval (seeded from the trace landing via
    # symbolic_staging entry) is what skips that concretize and keeps x9 symbolic.
    items = [
        _insm(0, 0x1000, _OS_LDR_X9, "ldr x9, [x10]",
              reads={"x10": _OS_STAGING}, writes={"x9": 0x41},
              mem=[MemOp("r", _OS_STAGING, 0x41, 8)]),
    ]
    # Seed the staging slot symbolic at the TRACE landing + record the interval.
    entry = {"symbolic_regs": [], "concrete_regs": {"x10": _OS_STAGING},
             "symbolic_mem": [{"addr": _OS_STAGING, "size": 8, "value": 0x41}],
             "symbolic_staging": [[_OS_STAGING, 8]]}

    dec = TritonStepDecoder(output_reg="x9")
    run_window(dec, SemanticsTable(), items, window=(0, 0), entry=entry,
               window_kind="idx")
    assert dec._ctx.getRegisterAst(dec._ctx.getRegister("x9")).isSymbolized()

    # CONTRAST: with the skip disabled, the same load is concretized (taint dropped)
    # IF the slot weren't already symbolic via symbolic_mem. Prove the skip path is
    # consulted: the interval set is empty under the subclass (recording disabled).
    bare = _NoSymbolicStaging(output_reg="x9")
    run_window(bare, SemanticsTable(), items, window=(0, 0),
               entry={"symbolic_regs": [], "concrete_regs": {"x10": _OS_STAGING}},
               window_kind="idx")
    # no symbolic_mem, skip disabled → the load is shadowed concrete, x9 not symbolic
    assert not bare._ctx.getRegisterAst(bare._ctx.getRegister("x9")).isSymbolized()
    assert bare.shadowed_mem_reads == 1                  # concretized (the collapse)


@requires_triton
def test_opaque_staging_empty_set_is_baseline_behaviour():
    # Invariant 7: with no symbolic staging (no symbolic_mem, no symbolic_staging, no
    # in-window symbolic store), _symbolic_staging stays empty and a plain concrete
    # load is shadowed exactly as before — the green baseline does not move.
    items = [
        _insm(0, 0x1000, _OS_LDR_X9, "ldr x9, [x10]",
              reads={"x10": _OS_STAGING}, writes={"x9": 0xCAFE},
              mem=[MemOp("r", _OS_STAGING, 0xCAFE, 8)]),
    ]
    dec = TritonStepDecoder(output_reg="x9")
    run_window(dec, SemanticsTable(), items, window=(0, 0),
               entry={"symbolic_regs": [], "concrete_regs": {"x10": _OS_STAGING}},
               window_kind="idx")
    ctx = dec._ctx
    assert dec._symbolic_staging == set()                # nothing recorded
    assert not ctx.getRegisterAst(ctx.getRegister("x9")).isSymbolized()
    assert ctx.getConcreteRegisterValue(ctx.getRegister("x9")) == 0xCAFE  # trace value
    assert dec.shadowed_mem_reads == 1                   # shadowed exactly as before


@requires_triton
def test_opaque_staging_unshadowed_zero_for_forwarded_load():
    # Regression metric: a forwarded staging load (symbolic store then load) has a
    # source, so unshadowed_steps stays 0 (the symbolic byte its store wrote).
    items = [
        _insm(0, 0x1000, _OS_STR_X8, "str x8, [x10]",
              reads={"x8": 0x41, "x10": _OS_STAGING},
              mem=[MemOp("w", _OS_STAGING, 0x41, 8)]),
        _insm(1, 0x1004, _OS_LDR_X9, "ldr x9, [x10]",
              reads={"x10": _OS_STAGING}, writes={"x9": 0x41},
              mem=[MemOp("r", _OS_STAGING, 0x41, 8)]),
    ]
    dec = TritonStepDecoder(output_reg="x9")
    run_window(dec, SemanticsTable(), items, window=(0, 1),
               entry={"symbolic_regs": ["x8"], "concrete_regs": {"x10": _OS_STAGING}},
               window_kind="idx")
    assert dec.unshadowed_steps == 0


# --- opaque-staging Phase 2(i) "+ record-a-line": forward COUNT observability ---
#
# dev-opaque-staging-phases-impl-spec.md (## Phase 2(i) 续). Forwarding is the third
# destination a non-symbolized load can take (the other two are shadowed_mem_reads /
# unshadowed_steps). The forward itself was silent; now _shadow_concrete_reads counts
# each load it forwards (symbolic_forwards) and samples the site (pc, addr, size).
# Purely observational — never feeds any close/parity/G4/seed gate. Synthetic, zero
# case-specific (every address/count is a fixture constant exercising the mechanism).


@requires_triton
def test_symbolic_forwards_counts_hits_and_records_sites():
    # Form M: an INJECTED staging interval (Phase 2(i): the slot is symbolic but
    # Triton symbolized it at a different concrete address, so isMemorySymbolized is
    # False at the load's trace op.addr) that the load EA hits → _shadow_concrete_reads
    # forwards (skips the concretize) and counts it; the site (pc, addr, size) is
    # recorded exactly. shadowed_mem_reads stays 0 (the read was NOT concretized).
    items = [
        _insm(0, 0x1004, _OS_LDR_X9, "ldr x9, [x10]",
              reads={"x10": _OS_STAGING}, writes={"x9": 0x41},
              mem=[MemOp("r", _OS_STAGING, 0x41, 8)]),
    ]
    dec = TritonStepDecoder(output_reg="x9")
    entry = {"symbolic_regs": [], "concrete_regs": {"x10": _OS_STAGING},
             "symbolic_staging": [[_OS_STAGING, 8]]}
    run_window(dec, SemanticsTable(), items, window=(0, 0), entry=entry,
               window_kind="idx")
    assert dec.symbolic_forwards == 1                       # the load forwarded
    assert dec.symbolic_forward_sites == [(0x1004, _OS_STAGING, 8)]
    assert dec.shadowed_mem_reads == 0                      # NOT concretized


@requires_triton
def test_symbolic_forwards_surfaced_in_runner_result():
    # The count + capped sample reach the runner result (same exit as the other
    # concolic counters) so drive / the agent can read "forwarded M loads".
    items = [
        _insm(0, 0x1004, _OS_LDR_X9, "ldr x9, [x10]",
              reads={"x10": _OS_STAGING}, writes={"x9": 0x41},
              mem=[MemOp("r", _OS_STAGING, 0x41, 8)]),
    ]
    runner = build_level2_runner(decoder=TritonStepDecoder(output_reg="x9"))
    out = runner({"entry": {"symbolic_regs": [],
                            "concrete_regs": {"x10": _OS_STAGING},
                            "symbolic_staging": [[_OS_STAGING, 8]]},
                  "window": [0, 0], "window_kind": "idx",
                  "items": items, "decisions": {}})
    assert out["symbolic_forwards"] == 1
    assert out["symbolic_forward_sites"] == [[0x1004, _OS_STAGING, 8]]


@requires_triton
def test_symbolic_forwards_injected_but_not_hit_is_observable():
    # Form N (tc4 half-wired signal): an interval is injected but the runtime load EA
    # lands OUTSIDE it → the staging set is non-empty yet symbolic_forwards stays 0.
    # "injected > 0, forwarded == 0" is the directly-observable wired-but-never-hit
    # state (visible without inferring it from closed == 0).
    other = _OS_STAGING + 0x1000                            # injected elsewhere
    items = [
        _insm(0, 0x1004, _OS_LDR_X9, "ldr x9, [x10]",
              reads={"x10": _OS_STAGING}, writes={"x9": 0xCAFE},
              mem=[MemOp("r", _OS_STAGING, 0xCAFE, 8)]),
    ]
    dec = TritonStepDecoder(output_reg="x9")
    entry = {"symbolic_regs": [], "concrete_regs": {"x10": _OS_STAGING},
             "symbolic_staging": [[other, 8]]}             # does not cover the load
    run_window(dec, SemanticsTable(), items, window=(0, 0), entry=entry,
               window_kind="idx")
    assert dec._symbolic_staging == {(other, 8)}            # injected > 0
    assert dec.symbolic_forwards == 0                       # but never hit
    assert dec.symbolic_forward_sites == []
    assert dec.shadowed_mem_reads == 1                      # load ran the concrete path


@requires_triton
def test_symbolic_forwards_empty_staging_stays_zero_invariant7():
    # Form O (invariant 7): with no symbolic staging the hit point is never reached →
    # symbolic_forwards == 0, sites empty, and the load is shadowed exactly as before
    # (byte-for-byte baseline; the two new fields are the only addition, both inert).
    items = [
        _insm(0, 0x1004, _OS_LDR_X9, "ldr x9, [x10]",
              reads={"x10": _OS_STAGING}, writes={"x9": 0xCAFE},
              mem=[MemOp("r", _OS_STAGING, 0xCAFE, 8)]),
    ]
    dec = TritonStepDecoder(output_reg="x9")
    run_window(dec, SemanticsTable(), items, window=(0, 0),
               entry={"symbolic_regs": [], "concrete_regs": {"x10": _OS_STAGING}},
               window_kind="idx")
    assert dec._symbolic_staging == set()
    assert dec.symbolic_forwards == 0
    assert dec.symbolic_forward_sites == []
    assert dec.shadowed_mem_reads == 1                      # shadowed as before


@requires_triton
def test_symbolic_forward_sites_capped_count_keeps_counting():
    # Invariant 4: when hits exceed SYMBOLIC_FORWARD_SITE_CAP the exact count keeps
    # rising while the sample list is bounded at the cap. Build CAP+10 distinct
    # forwarding loads over one wide staging interval.
    from engine.setup_symex_runner import SYMBOLIC_FORWARD_SITE_CAP
    n_hits = SYMBOLIC_FORWARD_SITE_CAP + 10
    base = _OS_STAGING
    items = []
    for i in range(n_hits):
        addr = base + i * 8
        items.append(
            _insm(i, 0x2000 + i * 4, _OS_LDR_X9, "ldr x9, [x10]",
                  reads={"x10": addr}, writes={"x9": 0x41},
                  mem=[MemOp("r", addr, 0x41, 8)]))
    dec = TritonStepDecoder(output_reg="x9")
    # One interval spanning every load's slot; each read hits it and forwards.
    span = n_hits * 8
    entry = {"symbolic_regs": [], "concrete_regs": {"x10": base},
             "symbolic_staging": [[base, span]]}
    run_window(dec, SemanticsTable(), items, window=(0, n_hits - 1), entry=entry,
               window_kind="idx")
    assert dec.symbolic_forwards == n_hits                          # exact count
    assert len(dec.symbolic_forward_sites) == SYMBOLIC_FORWARD_SITE_CAP  # capped
