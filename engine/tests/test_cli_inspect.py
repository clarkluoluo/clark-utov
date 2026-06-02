"""0527 BUG_REPORT-7 §J.7: `utov inspect` trace-query subcommand."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from engine.cli import main


def _write_jsonl(tmp: Path, instructions: list[dict]) -> Path:
    p = tmp / "trace.jsonl"
    with p.open("w") as f:
        for ins in instructions:
            f.write(json.dumps(ins) + "\n")
    return p


def _ins(idx: int, pc: int, mnem: str, *, reads=None, writes=None) -> dict:
    return {
        "idx":        idx,
        "pc":         f"0x{pc:x}",
        "bytes":      "00000000",
        "mnemonic":   mnem,
        "regs_read":  {k: f"0x{v:x}" for k, v in (reads or {}).items()},
        "regs_write": {k: f"0x{v:x}" for k, v in (writes or {}).items()},
        "mem":        [],
    }


def test_inspect_fingerprint_filter():
    """--fingerprint AES.Te0[0] finds the canonical 0xc66363a5 hit only."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        trace = _write_jsonl(tmp, [
            _ins(0, 0x40002770, "ldr w0, [x10]",
                 reads={"x10": 0x40008960},
                 writes={"w0": 0xc66363a5}),         # ← AES.Te0[0]
            _ins(1, 0x40002774, "add x1, x0, #1",
                 reads={"x0": 0xc66363a5},
                 writes={"x1": 0xc66363a6}),
            _ins(2, 0x40002778, "ret"),
        ])
        runner = CliRunner()
        r = runner.invoke(main, [
            "inspect", "--trace", str(trace),
            "--fingerprint", "AES.Te0[0]", "--json",
        ])
        assert r.exit_code == 0, r.output
        out = json.loads(r.output)
        assert len(out) == 1
        assert out[0]["idx"] == 0


def test_inspect_pc_range_filter():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        trace = _write_jsonl(tmp, [
            _ins(0, 0x40001000, "ret"),
            _ins(1, 0x40002770, "ldr w0, [x10]"),
            _ins(2, 0x40002780, "add x1, x0, #1"),
            _ins(3, 0x40003000, "ret"),
        ])
        runner = CliRunner()
        r = runner.invoke(main, [
            "inspect", "--trace", str(trace),
            "--pc-range", "0x40002770..0x400028a0", "--json",
        ])
        assert r.exit_code == 0, r.output
        out = json.loads(r.output)
        assert [o["idx"] for o in out] == [1, 2]


def test_inspect_reg_value_filter():
    """--reg-value 0x40008960 catches any instruction that reads or writes it."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        trace = _write_jsonl(tmp, [
            _ins(0, 0x1000, "movz x10, #0x8960", writes={"x10": 0x40008960}),
            _ins(1, 0x1004, "ldr w0, [x10]",
                 reads={"x10": 0x40008960},
                 writes={"w0": 0xc66363a5}),
            _ins(2, 0x1008, "add x0, x0, #1",
                 reads={"x0": 0xc66363a5},
                 writes={"x0": 0xc66363a6}),
        ])
        runner = CliRunner()
        r = runner.invoke(main, [
            "inspect", "--trace", str(trace),
            "--reg-value", "0x40008960", "--json",
        ])
        assert r.exit_code == 0, r.output
        out = json.loads(r.output)
        assert [o["idx"] for o in out] == [0, 1]


def test_inspect_anchor_near_window():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        trace = _write_jsonl(tmp, [
            _ins(i, 0x1000 + i * 4, f"nop {i}") for i in range(20)
        ])
        runner = CliRunner()
        r = runner.invoke(main, [
            "inspect", "--trace", str(trace),
            "--anchor-near", "10", "--window", "2", "--json",
        ])
        assert r.exit_code == 0, r.output
        out = json.loads(r.output)
        assert [o["idx"] for o in out] == [8, 9, 10, 11, 12]


def test_inspect_context_emits_surrounding_lines():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        trace = _write_jsonl(tmp, [
            _ins(0, 0x1000, "nop"),
            _ins(1, 0x1004, "nop"),
            _ins(2, 0x1008, "ldr w0, [x10]", writes={"w0": 0xc66363a5}),
            _ins(3, 0x100c, "nop"),
            _ins(4, 0x1010, "nop"),
        ])
        runner = CliRunner()
        r = runner.invoke(main, [
            "inspect", "--trace", str(trace),
            "--fingerprint", "AES.Te0[0]", "--context", "1", "--json",
        ])
        assert r.exit_code == 0, r.output
        out = json.loads(r.output)
        assert [o["idx"] for o in out] == [1, 2, 3]


def test_inspect_unknown_fingerprint_errors_helpfully():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        trace = _write_jsonl(tmp, [_ins(0, 0x1000, "ret")])
        runner = CliRunner()
        r = runner.invoke(main, [
            "inspect", "--trace", str(trace),
            "--fingerprint", "NotARealName",
        ])
        assert r.exit_code != 0
        assert "unknown fingerprint name" in r.output
