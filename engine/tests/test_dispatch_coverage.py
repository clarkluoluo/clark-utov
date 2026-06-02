"""Dispatch preflight coverage map — see all the work before running.

Pins that the coverage primitive classifies the call sequence into types and, once
per type, surfaces the full I/O gap list (reg + mem live-in, un-modeled opcodes,
state-carrier outputs) — so h12…h100 are covered by ~N type solves, not discovered
one invocation at a time. Target-agnostic: the agent supplies the classification.
"""

from __future__ import annotations

import pytest

from engine.dispatch_coverage import (
    CoverageMap,
    HandlerInvocation,
    preflight_dispatch_coverage,
    triton_decode_probe,
)
from engine.export_stamp import is_utov_export
from engine.setup_symex_runner import triton_available
from engine.types import Instruction, MemOp


def insm(idx, pc, mnem, *, reads=None, writes=None, mem=(), code=b"\x00\x00\x00\x00"):
    return Instruction(idx=idx, pc=pc, bytes_=code, mnemonic=mnem,
                       regs_read=reads or {}, regs_write=writes or {}, mem=tuple(mem))


# A 2-type VM program: t0 = pure-register handler, t1 = memory-input handler
# (h11-shaped: an external ldr). Each runs twice → h-reuse, not new types.
_LDR_T1 = b"\xaa\xbb\xcc\xdd"          # t1's distinguishing opcode (probe rejects it)


def _trace():
    return [
        insm(0, 0x100, "add x8, x0, x1", reads={"x0": 0, "x1": 0}, writes={"x8": 0}),
        insm(1, 0x104, "mov x9, x8", reads={"x8": 0}, writes={"x9": 0}),
        insm(2, 0x200, "ldr x10, [x16]", reads={"x16": 0x9000}, writes={"x10": 0},
             mem=[MemOp("r", 0x9000, 0, 8)], code=_LDR_T1),
        insm(3, 0x204, "add x8, x10, x9", reads={"x10": 0, "x9": 0}, writes={"x8": 0}),
        insm(4, 0x100, "add x8, x0, x1", reads={"x0": 0, "x1": 0}, writes={"x8": 0}),
        insm(5, 0x104, "mov x9, x8", reads={"x8": 0}, writes={"x9": 0}),
        insm(6, 0x200, "ldr x10, [x16]", reads={"x16": 0x9000}, writes={"x10": 0},
             mem=[MemOp("r", 0x9000, 0, 8)], code=_LDR_T1),
        insm(7, 0x204, "add x8, x10, x9", reads={"x10": 0, "x9": 0}, writes={"x8": 0}),
        insm(8, 0x300, "str x8, [x20]", reads={"x8": 0}, mem=[MemOp("w", 0xB000, 0, 8)]),
    ]


_INVS = [
    HandlerInvocation("t0", 0, 1), HandlerInvocation("t1", 2, 3),
    HandlerInvocation("t0", 4, 5), HandlerInvocation("t1", 6, 7),
]


def _fake_probe(code: bytes) -> bool:
    return code != _LDR_T1            # t1's opcode is the only un-modeled one


def _cov():
    return preflight_dispatch_coverage(_trace(), invocations=_INVS, decode_probe=_fake_probe)


def test_classifies_sequence_and_counts_occurrences():
    cov = _cov()
    assert isinstance(cov, CoverageMap)
    assert cov.n_types == 2 and len(cov.sequence) == 4
    assert cov.sequence == ("t0", "t1", "t0", "t1")
    by_id = {t.type_id: t for t in cov.types}
    assert by_id["t0"].occurrences == 2 and by_id["t1"].occurrences == 2
    assert by_id["t1"].representative == (2, 3)     # the FIRST t1 invocation


def test_per_type_io_signature():
    by_id = {t.type_id: t for t in _cov().types}
    t0, t1 = by_id["t0"], by_id["t1"]
    # t0: pure register inputs, no memory input.
    assert set(t0.reg_live_in) == {"x0", "x1"} and t0.mem_live_in == ()
    # t1: the external memory input is listed UP FRONT (not discovered by running).
    assert [m.addr for m in t1.mem_live_in] == [0x9000]
    assert "x16" in t1.reg_live_in            # the load base is a live-in too
    # outputs = state carrier threaded to the next handler.
    assert "x8" in t0.outputs


def test_unmodeled_opcodes_from_probe():
    by_id = {t.type_id: t for t in _cov().types}
    assert by_id["t1"].unmodeled_opcodes == ("aabbccdd",) and by_id["t1"].decode_probed
    assert by_id["t0"].unmodeled_opcodes == ()


def test_no_probe_leaves_unmodeled_empty_and_flagged():
    cov = preflight_dispatch_coverage(_trace(), invocations=_INVS)   # no probe
    assert all(t.unmodeled_opcodes == () and t.decode_probed is False for t in cov.types)


def test_markdown_and_stamped_export():
    cov = _cov()
    md = cov.to_markdown()
    assert "Dispatch coverage map" in md and "t0" in md and "t1" in md
    assert "0x9000" in md and "aabbccdd" in md
    stamped = cov.to_stamped_markdown(
        source="utov/cvd_ledger.sqlite",
        exec_identity={"target": "libEncryptor.so", "run_id": "r1"},
        ts="2026-05-31T12:00:00Z")
    assert is_utov_export(stamped) and "Dispatch coverage map" in stamped


def test_to_dict_shape():
    d = _cov().to_dict()
    assert d["kind"] == "dispatch_coverage_map" and d["n_types"] == 2
    assert d["n_invocations"] == 4
    assert d["types"][0]["kind"] == "dispatch_type_coverage"


@pytest.mark.skipif(not triton_available(), reason="Triton bindings not installed")
def test_triton_decode_probe_models_real_opcode_rejects_garbage():
    probe = triton_decode_probe()
    assert probe(b"\x20\x00\x80\x52") is True       # movz w0, #1 — decodable
    assert probe(b"\xff\xff\xff\xff") is False      # not a valid AArch64 opcode
