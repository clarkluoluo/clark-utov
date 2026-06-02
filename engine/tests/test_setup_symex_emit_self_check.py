"""G4 emit self-check — the recovered F must reproduce its own trace.

Concolic symex runs ON a trace, so the recovered F evaluated on the trace's own
concrete seed values must equal the trace's concrete sink at the window exit. If
not, the symex is unsound and the F is wrong (the real F0 handler56 case: symex
exit x8=69 while the trace's exit was 0xfb9881b1, emitted silently). This pins
the necessary-not-sufficient gate that catches that BEFORE emit.

Synthetic shapes only — no F0 coordinates / handler numbers / case names.
"""

from __future__ import annotations

from dataclasses import replace

from engine.setup_symex import (
    CaseConfig,
    DriveResult,
    build_concrete_backing,
    check_emit_self_consistency,
    drive,
)
from engine.types import Instruction


# --------------------------------------------------------------------------- #
# Unit: check_emit_self_consistency over the four A8 shapes.
# --------------------------------------------------------------------------- #

def test_self_check_block_when_F_disagrees_with_its_own_trace():
    # Replicate the h56 SHAPE (not its coordinates): F emits a constant 69 but the
    # trace's window-exit sink was something else entirely.
    rep = check_emit_self_consistency(
        expr_source="def f(carrier):\n    return 69\n",
        inputs=("carrier",),
        seed_values={"carrier": 0x11223344},
        trace_sink=0xFB9881B1,
        sink_mask=0xFFFFFFFF)
    assert rep.status == "BLOCK"
    assert rep.f_on_trace == hex(69) and rep.trace_sink == hex(0xFB9881B1)
    assert "UNSOUND" in rep.note


def test_self_check_pass_when_F_reproduces_trace():
    # A real symbolic transform that agrees with the trace at the trace's seed.
    rep = check_emit_self_consistency(
        expr_source="def f(a, b):\n    return (a ^ b) & 0xffffffff\n",
        inputs=("a", "b"),
        seed_values={"a": 0xDEADBEEF, "b": 0x0F0F0F0F},
        trace_sink=(0xDEADBEEF ^ 0x0F0F0F0F),
        sink_mask=0xFFFFFFFF)
    assert rep.status == "PASS"
    assert "necessary, not sufficient" in rep.note


def test_self_check_constant_emit_no_symbolic_seed():
    # A8: emitted is a constant (no symbolic seed) — compared directly to the
    # trace sink. Matching constant only passes the self-check (seed/parity gates
    # still judge the degenerate-ness separately).
    ok = check_emit_self_consistency(
        expr_source="def f():\n    return 0x1234\n", inputs=(),
        seed_values={}, trace_sink=0x1234)
    assert ok.status == "PASS"
    bad = check_emit_self_consistency(
        expr_source="def f():\n    return 0x1234\n", inputs=(),
        seed_values={}, trace_sink=0x9999)
    assert bad.status == "BLOCK"


def test_self_check_inconclusive_when_no_trace_facts():
    # No trace sink → cannot self-check (surfaced, never silent PASS).
    rep = check_emit_self_consistency(
        expr_source="def f(a):\n    return a\n", inputs=("a",),
        seed_values={"a": 1}, trace_sink=None)
    assert rep.status == "INCONCLUSIVE"
    # No seed values → likewise inconclusive.
    rep2 = check_emit_self_consistency(
        expr_source="def f(a):\n    return a\n", inputs=("a",),
        seed_values=None, trace_sink=0x10)
    assert rep2.status == "INCONCLUSIVE"


def test_self_check_inconclusive_when_F_references_off_trace_quantity():
    # A8: emitted references a quantity not on the trace / does not evaluate →
    # INCONCLUSIVE, not a silent pass.
    rep = check_emit_self_consistency(
        expr_source="def f(a):\n    return a + missing_global\n", inputs=("a",),
        seed_values={"a": 1}, trace_sink=0x10)
    assert rep.status == "INCONCLUSIVE"
    assert "did not evaluate" in rep.note


def test_self_check_mem_sink_compared_as_bytes():
    # A8: sink is a multi-byte / memory region — compared in its real (bytes) form,
    # not a single masked reg.
    good = check_emit_self_consistency(
        expr_source="def f(a):\n    return a.to_bytes(4, 'little')\n", inputs=("a",),
        seed_values={"a": 0x04030201}, trace_sink=b"\x01\x02\x03\x04",
        sink_form="mem")
    assert good.status == "PASS" and good.sink_form == "mem"
    bad = check_emit_self_consistency(
        expr_source="def f(a):\n    return a.to_bytes(4, 'little')\n", inputs=("a",),
        seed_values={"a": 0x99999999}, trace_sink=b"\x01\x02\x03\x04",
        sink_form="mem")
    assert bad.status == "BLOCK"


def test_self_check_bare_expression_form():
    # Emit form that is a bare expression rather than a def f(...).
    rep = check_emit_self_consistency(
        expr_source="(a + b) & 0xff", inputs=("a", "b"),
        seed_values={"a": 0x40, "b": 0x02}, trace_sink=0x42, sink_mask=0xFF)
    assert rep.status == "PASS"


# --------------------------------------------------------------------------- #
# Drive-level: a runner whose emitted F disagrees with the trace it ran on must
# NOT close — the self-check BLOCKs ahead of the parity gate.
# --------------------------------------------------------------------------- #

def _ins(idx, pc, mnem, *, reads=None, writes=None, mem=()):
    return Instruction(idx=idx, pc=pc, bytes_=b"\x00\x00\x00\x00", mnemonic=mnem,
                       regs_read=reads or {}, regs_write=writes or {}, mem=tuple(mem))


def _items():
    return [
        _ins(0, 0x1000, "ldr w0, [x16]", reads={"x16": 0x9000}),
        _ins(1, 0x1004, "mul w0, w0, w1", reads={"w0": 0, "w1": 0}, writes={"w0": 0}),
    ]


_CC = CaseConfig(
    target="synthetic.so", input_hash="ab12", run_id="run-1",
    seed_hint_addr=0x100, sink_hint_addr=0x200, entry_pc=0x0FFF,
    window=(0x1000, 0x10FF), reg_file=("x0", "x1", "x16"),
    inputs=("carrier",), parity_min=8, symbolic_regs=("x0", "x1"),
    concrete_backing=build_concrete_backing(reg_values={"x16": 0x9000}),
    task="self_check_smoke")

_BOTH = {"alias_vs_compute": "compute", "which_static": []}


def test_drive_blocks_when_emitted_F_fails_self_check():
    def runner(_ctx):
        return {
            "propagated": True, "gold_parity": "8/8",
            # F emits a constant that does NOT match the trace's exit sink.
            "expr_source": "def f(carrier):\n    return 69\n",
            # observed VARIES per distinct input (real recovery's output is
            # input-dependent); a constant observed would be an UNCLOSABLE false EXACT.
            "parity_vectors": [
                {"input_key": f"v{i}", "observed": f"o{i}", "predicted": f"o{i}",
                 "exec_id": f"e{i}"} for i in range(3)],
            "trace_self_check": {
                "seed_values": {"carrier": 0x11223344},
                "sink_value": 0xFB9881B1, "sink_mask": 0xFFFFFFFF},
        }

    res = drive(trace=_items(), case_config=_CC, triton_runner=runner, decisions=_BOTH)
    assert isinstance(res, DriveResult)
    assert res.closed is False
    assert res.self_check is not None and res.self_check["status"] == "BLOCK"
    assert "UNSOUND" in res.note
    sc = next(s for s in res.per_step if s["step"] == "emit_self_check")
    assert sc["status"] == "BLOCK"


def test_drive_closes_when_emitted_F_reproduces_trace():
    def runner(_ctx):
        return {
            "propagated": True, "gold_parity": "8/8",
            "expr_source": "def f(carrier):\n    return (carrier ^ 0x5a5a5a5a) & 0xffffffff\n",
            # observed VARIES per distinct input (real recovery's output is
            # input-dependent); a constant observed would be an UNCLOSABLE false EXACT.
            "parity_vectors": [
                {"input_key": f"v{i}", "observed": f"o{i}", "predicted": f"o{i}",
                 "exec_id": f"e{i}"} for i in range(3)],
            "trace_self_check": {
                "seed_values": {"carrier": 0x11223344},
                "sink_value": (0x11223344 ^ 0x5A5A5A5A), "sink_mask": 0xFFFFFFFF},
        }

    res = drive(trace=_items(), case_config=_CC, triton_runner=runner, decisions=_BOTH)
    assert res.closed is True
    assert res.self_check["status"] == "PASS"


def test_drive_surfaces_inconclusive_without_blocking_close():
    # A runner that supplies no trace_self_check facts: the self-check is
    # INCONCLUSIVE (legacy / green-baseline path) — surfaced, never silently
    # claimed PASS, but it does not by itself flip an otherwise-passing close.
    def runner(_ctx):
        return {
            "propagated": True, "gold_parity": "8/8",
            "expr_source": "def f(carrier):\n    return carrier & 0xffffffff\n",
            # observed VARIES per distinct input (real recovery's output is
            # input-dependent); a constant observed would be an UNCLOSABLE false EXACT.
            "parity_vectors": [
                {"input_key": f"v{i}", "observed": f"o{i}", "predicted": f"o{i}",
                 "exec_id": f"e{i}"} for i in range(3)],
        }

    res = drive(trace=_items(), case_config=_CC, triton_runner=runner, decisions=_BOTH)
    assert res.closed is True
    assert res.self_check["status"] == "INCONCLUSIVE"
    assert "INCONCLUSIVE" in res.note
