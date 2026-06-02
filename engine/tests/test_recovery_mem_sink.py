"""Issue 7 — generic mem-write / window-output recovery (spec_f0_mem_write_window_sink.md).

A recovery window whose OUTPUT is a memory write (a store), not a register. The
runner reads the store's symbolic bytes after the window (mem sink), the self-check
compares BYTES, parity is bytewise across vectors, and the degenerate ends are
structured (MEM_SINK_UNPLACEABLE for an unplaceable store; the existing seed-
independence exclusion for an input-invariant store) — NEVER a silent register/
constant fallback.

Four regression fixtures (spec §"Regression fixtures"):
  1. mem-output recovery → bytes emitted, self_check PASS, parity EXACT, CLOSED;
  2. reg path unchanged → byte-for-byte the current x8 behaviour (regression guard);
  3. unplaceable sink → MEM_SINK_UNPLACEABLE with the structured needed[] list;
  4. input-invariant store → seed-independence exclusion (UNCLOSABLE-class), NOT a
     constant-via-x8 / a new ad-hoc verdict.

Synthetic shapes only — no F0 coordinates / handler numbers / case names.
"""

from __future__ import annotations

import pytest

from engine.cvd import CvdState, VStatus
from engine.cvd_recovery import (
    MEM_SINK_UNPLACEABLE_NEEDED,
    RECOVER_WINDOW,
    TERMINAL_MEM_SINK_UNPLACEABLE,
    RecoverWindowVerifier,
    derive_mem_sink_interval,
)
from engine.cvd import Candidate
from engine.setup_symex import CaseConfig, build_concrete_backing
from engine.setup_symex_runner import (
    SemanticsTable,
    TritonStepDecoder,
    build_level2_runner,
    run_window,
    triton_available,
)
from engine.types import Instruction, MemOp

requires_triton = pytest.mark.skipif(
    not triton_available(), reason="Triton bindings not installed on host")

# A staging slot the store lands in (fixture constant — exercises the mechanism).
_SINK = 0x20000
_STR_X0_X1 = b"\x20\x00\x00\xf9"      # str x0, [x1]


def _store_items(*, value=0x04030201, symbolic=True):
    """A one-store window: ``str x0, [x1]`` writing ``value`` to ``_SINK``."""
    return [Instruction(
        idx=0, pc=0x1000, bytes_=_STR_X0_X1, mnemonic="str x0, [x1]",
        regs_read={"x0": value, "x1": _SINK}, regs_write={},
        mem=(MemOp("w", _SINK, value, 8),))]


# --------------------------------------------------------------------------- #
# ① runner mem-sink expression (Triton): expression() returns the store BYTES.
# --------------------------------------------------------------------------- #

@requires_triton
def test_runner_mem_sink_emits_store_bytes():
    # The sink store writes F(x0) = x0; expression() reads [sink, sink+4) symbolic
    # bytes and emits a byte-list (NOT an x8 register value).
    dec = TritonStepDecoder(output_mem={"sink_addr": _SINK, "sink_size": 4,
                                         "sink_idx": 0})
    res = run_window(dec, SemanticsTable(), _store_items(),
                     window=(0, 0), window_kind="idx",
                     entry={"symbolic_regs": ["x0"],
                            "concrete_regs": {"x0": 0x04030201, "x1": _SINK}})
    assert not res.blocked
    assert res.expr_source == "bytes([1, 2, 3, 4])"     # little-endian store bytes
    assert dec.mem_sink_unreadable is None


@requires_triton
def test_runner_mem_sink_input_invariant_store_is_surfaced_not_emitted():
    # A constant store (no symbolic input reaches it) → NO byte became symbolic →
    # expression() returns "" and records the input-invariant reason (never a silent
    # constant byte-list emit).
    dec = TritonStepDecoder(output_mem={"sink_addr": _SINK, "sink_size": 4,
                                        "sink_idx": 0})
    res = run_window(dec, SemanticsTable(), _store_items(),
                     window=(0, 0), window_kind="idx",
                     entry={"symbolic_regs": [],          # nothing symbolic
                            "concrete_regs": {"x0": 0x04030201, "x1": _SINK}})
    assert res.expr_source == ""
    assert "input-invariant" in (dec.mem_sink_unreadable or "")


@requires_triton
def test_level2_runner_surfaces_mem_sink_unreadable():
    # The structured reason rides the runner result (same exit as the seed counters)
    # so drive / the recovery layer reads it (never a silent fallback).
    runner = build_level2_runner(decoder=TritonStepDecoder())
    out = runner({"entry": {"symbolic_regs": [],
                            "concrete_regs": {"x0": 0x04030201, "x1": _SINK}},
                  "window": [0, 0], "window_kind": "idx",
                  "items": _store_items(), "decisions": {},
                  "output_mem": {"sink_addr": _SINK, "sink_size": 4, "sink_idx": 0}})
    assert "input-invariant" in out["mem_sink_unreadable"]


@requires_triton
def test_runner_reg_path_unchanged_when_no_output_mem():
    # PRESERVE: with no output_mem the decoder is byte-for-byte the register path —
    # expression() reads the output register, not memory bytes.
    add = b"\x08\x00\x01\x8b"                  # add x8, x0, x1
    items = [Instruction(idx=0, pc=0x1000, bytes_=add, mnemonic="add x8, x0, x1",
                         regs_read={"x0": 0, "x1": 0}, regs_write={"x8": 0}, mem=())]
    dec = TritonStepDecoder(output_reg="x8")   # register sink, output_mem is None
    run_window(dec, SemanticsTable(), items, window=(0, 0), window_kind="idx",
               entry={"symbolic_regs": ["x0", "x1"],
                      "concrete_regs": {"x0": 7, "x1": 5}})
    assert dec.output_mem is None
    assert dec.mem_sink_unreadable is None
    assert dec.expression() == str(12)          # x0 + x1, a register value (not bytes)


# --------------------------------------------------------------------------- #
# derive_mem_sink_interval — the EA derivation (caller need not fill addr+size).
# --------------------------------------------------------------------------- #

def test_derive_interval_from_trace_mem_op():
    # No addr+size in the descriptor → derive from the WRITE mem op at sink_idx.
    iv, why = derive_mem_sink_interval(_store_items(), {"sink_idx": 0})
    assert iv == (_SINK, 8) and why is None
    # a caller sink_size narrows the recorded store width.
    iv2, _ = derive_mem_sink_interval(_store_items(), {"sink_idx": 0, "sink_size": 4})
    assert iv2 == (_SINK, 4)


def test_derive_interval_explicit_addr_size_hex():
    iv, why = derive_mem_sink_interval(
        _store_items(), {"sink_idx": 0, "sink_addr": "0x20000", "sink_size": 4})
    assert iv == (_SINK, 4) and why is None


def test_derive_interval_unplaceable_when_no_trace_mem_op():
    # A store instruction in the trace but with NO write mem op (EA not decoded).
    items = [Instruction(idx=0, pc=0x1000, bytes_=_STR_X0_X1, mnemonic="str x0, [x1]",
                         regs_read={}, regs_write={}, mem=())]
    iv, why = derive_mem_sink_interval(items, {"sink_idx": 0})
    assert iv is None and "no trace write mem op" in why


# --------------------------------------------------------------------------- #
# ③ verifier-level — the four spec regression fixtures end to end.
# --------------------------------------------------------------------------- #

_MEM_CC = CaseConfig(
    target="synthetic.so", input_hash="ab12", run_id="run-mem",
    seed_hint_addr=0x100, sink_hint_addr=_SINK, entry_pc=0x0FFF,
    window=(0, 0), window_kind="idx", reg_file=("x0", "x1"),
    inputs=("carrier",), parity_min=8, symbolic_regs=("x0",),
    concrete_backing=build_concrete_backing(reg_values={"x1": _SINK}),
    task="recover_mem_window")

_BOTH = {"alias_vs_compute": "compute", "which_static": []}


def _mem_cand(descriptor):
    return Candidate(RECOVER_WINDOW, locus=0, signal="mem_sink_rep",
                     entry_reason="mem-write output store",
                     payload={"window": [0, 0], "window_kind": "idx",
                              "mem_sink": descriptor})


def test_fixture1_mem_output_recovery_closes_with_bytes():
    # Fixture 1: the sink store writes 4 bytes as F(carrier); the runner emits those
    # bytes, self_check PASS, parity EXACT (bytewise), terminal CLOSED, expr returns
    # bytes. The fake runner is mem-aware: ctx carries output_mem (derived addr+size),
    # so it returns a bytes-shaped F + a mem-form trace_self_check + bytewise vectors.
    def runner(ctx):
        assert ctx.get("output_mem") == {"sink_addr": _SINK, "sink_size": 4,
                                         "sink_idx": 0}
        # observed/predicted are byte-derived (the store's bytes), distinct per input.
        obs = [b"\x01\x02\x03\x04", b"\x05\x06\x07\x08", b"\x09\x0a\x0b\x0c"]
        return {
            "propagated": True, "gold_parity": "8/8",
            "expr_source": ("def f(carrier):\n"
                            "    return carrier.to_bytes(4, 'little')\n"),
            "parity_vectors": [
                {"input_key": f"v{i}", "observed": str(o), "predicted": str(o),
                 "exec_id": f"e{i}"} for i, o in enumerate(obs)],
            "trace_self_check": {
                "seed_values": {"carrier": 0x04030201},
                "sink_value": b"\x01\x02\x03\x04", "sink_form": "mem"},
        }

    v = RecoverWindowVerifier(base_config=_MEM_CC, triton_runner=runner,
                              decisions=_BOTH)
    out = v.verify(_mem_cand({"sink_form": "mem", "sink_idx": 0, "sink_size": 4}),
                   CvdState(_store_items(), b"\x01\x02\x03\x04"))
    assert out.status is VStatus.CONFIRMED
    assert out.evidence["parity"] == "8/8"
    assert out.evidence["self_check"] == "PASS"
    assert out.evidence["sink_form"] == "mem"        # compared as BYTES, not an x8 value
    assert "bytes" in out.evidence["emitted_F"]      # F returns bytes, not an x8 value


def test_fixture2_reg_path_unchanged_no_mem_sink():
    # Fixture 2 (regression guard): no mem_sink descriptor → the register path is
    # byte-for-byte the current x8 behaviour (the ctx carries NO output_mem; the
    # self-check is reg-form). A CONFIRMED close exactly as a reg recovery does today.
    def runner(ctx):
        assert ctx.get("output_mem") is None         # reg path: never mem-wired
        return {"propagated": True, "gold_parity": "8/8",
                "expr_source": "def f(carrier):\n    return (carrier ^ 0x5a) & 0xff\n",
                "parity_vectors": [{"input_key": f"v{i}", "observed": f"o{i}",
                                    "predicted": f"o{i}", "exec_id": f"e{i}"}
                                   for i in range(3)],
                "trace_self_check": {"seed_values": {"carrier": 0x10},
                                     "sink_value": (0x10 ^ 0x5A) & 0xFF,
                                     "sink_mask": 0xFF}}      # reg form (no sink_form)

    v = RecoverWindowVerifier(base_config=_MEM_CC, triton_runner=runner,
                              decisions=_BOTH)
    out = v.verify(
        Candidate(RECOVER_WINDOW, locus=0, signal="reg_rep", entry_reason="reg",
                  payload={"window": [0, 0], "window_kind": "idx"}),
        CvdState(_store_items(), b"\x00"))
    assert out.status is VStatus.CONFIRMED
    assert out.evidence["self_check"] == "PASS"
    assert out.evidence["sink_form"] == "reg"        # register path, unchanged


def test_fixture3_unplaceable_sink_is_structured_terminal():
    # Fixture 3: symbolic addr, no trace mem op at sink_idx, no pinned addr/size →
    # MEM_SINK_UNPLACEABLE with the VERBATIM needed[] list; no silent fallback. The
    # runner is never even reached (the EA cannot be derived before drive).
    def runner(_ctx):                                # must NOT be reached
        raise AssertionError("runner reached on an unplaceable sink")

    # A store instruction in the trace but with NO write mem op (EA not decoded).
    items = [Instruction(idx=0, pc=0x1000, bytes_=_STR_X0_X1, mnemonic="str x0, [x1]",
                         regs_read={"x1": _SINK}, regs_write={}, mem=())]
    v = RecoverWindowVerifier(base_config=_MEM_CC, triton_runner=runner,
                              decisions=_BOTH)
    out = v.verify(_mem_cand({"sink_form": "mem", "sink_idx": 0}),
                   CvdState(items, b"\x00"))
    assert out.status is VStatus.TERMINAL
    assert out.terminal_kind == TERMINAL_MEM_SINK_UNPLACEABLE
    assert out.evidence["needed"] == MEM_SINK_UNPLACEABLE_NEEDED
    assert out.evidence["needed"] == ["trace mem op", "pc-local regs", "EA decode",
                                      "memory backing"]
    assert out.evidence["sink_form"] == "mem"


def test_fixture4_input_invariant_store_routes_to_seed_independence():
    # Fixture 4: the store is constant across vectors (seed/driver-independent). The
    # runner surfaces mem_sink_unreadable="...input-invariant..."; the verifier routes
    # it to the EXISTING seed-independence exclusion (a seed_invariant TERMINAL), NOT
    # MEM_SINK_UNPLACEABLE and NOT a constant-via-x8.
    def runner(ctx):
        assert ctx.get("output_mem") == {"sink_addr": _SINK, "sink_size": 4,
                                         "sink_idx": 0}
        return {
            "propagated": True, "gold_parity": "0/8",
            "expr_source": "",                       # nothing emitted (no symbolic byte)
            "mem_sink_unreadable": (
                "the sink interval [0x20000, 0x20004) is input-invariant (no byte "
                "became symbolic after the window) — the store is seed/driver-"
                "independent, not a recovery target"),
        }

    v = RecoverWindowVerifier(base_config=_MEM_CC, triton_runner=runner,
                              decisions=_BOTH)
    out = v.verify(_mem_cand({"sink_form": "mem", "sink_idx": 0, "sink_size": 4}),
                   CvdState(_store_items(), b"\x00"))
    assert out.status is VStatus.TERMINAL
    assert out.terminal_kind == "seed_invariant"          # NOT MEM_SINK_UNPLACEABLE
    assert out.terminal_kind != TERMINAL_MEM_SINK_UNPLACEABLE
    assert "input-invariant" in out.evidence["mem_sink_seed_independent"]
