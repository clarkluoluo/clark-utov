"""Observation readers: parse a runner's calltrace log + register-pointed hook
dumps into the engine's canonical shapes (CallEvent / MemSnapshot). Generic —
no target address / tag / PC baked in; the snapshots feed the existing #1/#3.
"""

from __future__ import annotations

import inspect
import json
import re

from engine import obs_readers
from engine.obs_readers import (
    indirect_call_targets,
    parse_calltrace,
    parse_hook_snapshots,
)
from engine.oracle_sink import SinkVerdict, validate_sink
from engine.types import Instruction, MemSnapshot

EXPECTED = bytes([0x34, 0x15, 0x5f, 0xe9])


# --- calltrace reader -------------------------------------------------------

def test_parse_calltrace_basic_and_skips_noise():
    text = (
        "# a comment\n"
        "\n"
        "BL\t0xb3844\t0x7273c\t0x10\t0x20\n"
        "BLR\t0x70a90\t0x72ecc\t0xaa\n"
        "B\t0x71000\t-\n"
        "garbage line without tabs\n"
    )
    events = parse_calltrace(text)
    assert [e.kind for e in events] == ["BL", "BLR", "B"]
    assert events[0].pc == 0xb3844 and events[0].target == 0x7273c
    assert events[0].cols == ("0x10", "0x20")
    assert events[2].target is None              # "-" -> None


def test_indirect_call_targets():
    text = (
        "BL\t0xb3844\t0x7273c\n"          # direct -> not indirect
        "BLR\t0x70a90\t0x72ecc\n"         # indirect, resolved
        "BR\t0x70b00\t0x75cd8\n"          # indirect, resolved
        "BLR\t0x70c00\t-\n"               # indirect, unresolved -> skipped
    )
    assert indirect_call_targets(parse_calltrace(text)) == [(0x70a90, 0x72ecc),
                                                            (0x70b00, 0x75cd8)]


# --- hook-dump -> MemSnapshot ----------------------------------------------

def test_parse_hook_snapshots_register_pointed():
    rows = [
        {"tag": "x25", "pc_rva": "0x70a90", "x0": "0x1000", "mem_x0": "34 15 5f e9"},
        {"tag": "stk", "pc_rva": "0x70b00", "sp": "0xbef00", "mem_sp_0x10": "deadbeef"},
        {"tag": "bad", "pc_rva": "0x1", "x9": "0x2000", "mem_x9": "skip"},   # not hex
        {"tag": "noreg", "mem_x3": "aabb"},                                  # no x3 -> skip
    ]
    text = "\n".join(json.dumps(r) for r in rows)
    snaps = parse_hook_snapshots(text)
    by_addr = {s.addr: s for s in snaps}
    assert by_addr[0x1000].data == EXPECTED           # mem_x0 at x0
    assert by_addr[0x1000].label == "x25"
    assert by_addr[0xbef10].data == bytes.fromhex("deadbeef")  # sp + 0x10 offset
    assert 0x2000 not in by_addr                      # "skip" value dropped
    # the no-register row produced nothing
    assert all(s.label != "noreg" for s in snaps)


def test_hook_snapshots_feed_sink_validator():
    # the canonical MemSnapshots the reader produces are consumed by #1 directly.
    row = {"tag": "out", "pc_rva": "0x72b18", "x0": "0x72b18", "mem_x0": "34155fe9"}
    snaps = parse_hook_snapshots(json.dumps(row))
    trace = [Instruction(idx=0, pc=0x70000, bytes_=b"\x00\x00\x00\x00",
                         mnemonic="nop", regs_read={}, regs_write={}, mem=())]
    sv = validate_sink(trace, EXPECTED, snapshots=snaps)
    assert sv.verdict is SinkVerdict.SINK_CONFIRMED
    assert sv.located_via == "snapshot"
    assert sv.base == 0x72b18


def test_no_hardcoded_address_in_module():
    big = re.findall(r"0x[0-9a-fA-F]{4,}", inspect.getsource(obs_readers))
    assert big == [], f"unexpected hardcoded address literal(s): {big}"
