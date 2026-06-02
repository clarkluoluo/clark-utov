"""phase_discovery — locate a producing phase across various data sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from engine.phase_discovery import (
    InMemoryDataSource,
    PhaseDiscoveryConfig,
    PhaseDiscoveryResult,
    WriterHit,
    discover_phase,
    discover_phases_in_params,
    render_phase_discovery_alert,
)


# ---------------------------------------------------------------------------
# Minimal Instruction/MemOp doubles — avoids dragging engine.types in if a
# user wants to feed plain dicts later.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Mem:
    rw: str
    addr: int
    val: int = 0
    size: int = 1


@dataclass(frozen=True)
class _Ins:
    idx: int
    pc: int
    mnemonic: str = ""
    mem: tuple[_Mem, ...] = field(default_factory=tuple)


def _trace_window(pc_lo: int, pc_hi: int) -> list[_Ins]:
    return [
        _Ins(idx=i, pc=pc_lo + i * 4, mnemonic="nop")
        for i in range((pc_hi - pc_lo) // 4)
    ]


# ---------------------------------------------------------------------------
# In-window producer → no boundary.
# ---------------------------------------------------------------------------


def test_in_window_writer_returns_no_boundary():
    insns = _trace_window(0x40001000, 0x40001020)
    # Add a direct writer at idx 3.
    insns[3] = _Ins(
        idx=3, pc=0x40001000 + 3 * 4, mnemonic="str",
        mem=(_Mem(rw="w", addr=0xbabe0000, val=0xdeadbeef, size=4),),
    )
    src = InMemoryDataSource(instructions=insns)
    r = discover_phase(0xbabe0000, src, value_name="template", value_size=4,
                       cfg=PhaseDiscoveryConfig())
    assert r.boundary is None
    assert r.crosses_out is False
    assert "in_window" in r.reason


# ---------------------------------------------------------------------------
# Out-of-window producer (no writer in any source) → memregion anchor.
# ---------------------------------------------------------------------------


def test_no_writer_anywhere_returns_memregion_boundary():
    insns = _trace_window(0x40001000, 0x40001020)
    src = InMemoryDataSource(instructions=insns)
    r = discover_phase(0xbabe0000, src, value_name="template", value_size=32,
                       cfg=PhaseDiscoveryConfig())
    assert r.crosses_out is True
    assert r.boundary is not None
    b = r.boundary
    assert b.region is not None
    assert b.region[0] == 0xbabe0000
    assert b.region[1] >= 32
    assert b.anchor is not None
    assert b.anchor.anchor_type == "memregion_first_access"
    assert r.reason == "no_writer_in_any_source"


# ---------------------------------------------------------------------------
# Out-of-window writer hit via the ledger probe — boundary picks addr_first_exec.
# ---------------------------------------------------------------------------


def test_out_of_window_writer_via_ledger_returns_pc_anchor():
    insns = _trace_window(0x40001000, 0x40001020)
    out_of_window_pc = 0x40000800
    probe_called: dict[str, int] = {"n": 0}

    def probe(addr: int, size: int) -> WriterHit | None:
        probe_called["n"] += 1
        if addr == 0xbabe0000:
            return WriterHit(
                pc=out_of_window_pc, addr=addr, size=4, value=0x01,
                idx=-1,
            )
        return None

    src = InMemoryDataSource(instructions=insns, ledger_probe=probe)
    r = discover_phase(0xbabe0000, src, value_name="template", value_size=4,
                       cfg=PhaseDiscoveryConfig())
    assert probe_called["n"] >= 1
    assert r.crosses_out is True
    assert r.boundary is not None
    assert r.boundary.anchor is not None
    assert r.boundary.anchor.anchor_type == "addr_first_exec"
    assert r.boundary.anchor.params["pc"] == out_of_window_pc
    assert "out_of_window_writer" in r.reason


# ---------------------------------------------------------------------------
# Copy-chain walking — value written by an in-window ldr from a region the
# trace never wrote → boundary describes the *source* region.
# ---------------------------------------------------------------------------


def test_copy_chain_walks_to_unwritten_source():
    # Instruction 0..4 are nops. Inst 5 is a `ldr` that *writes* to
    # dst=0xc000_0000 from src=0xa000_0000. The trace never wrote to
    # 0xa000_0000, so discovery should produce a boundary describing
    # the source region (the producing phase).
    insns: list[_Ins] = []
    for i in range(5):
        insns.append(_Ins(idx=i, pc=0x40001000 + i * 4, mnemonic="nop"))
    insns.append(_Ins(
        idx=5, pc=0x40001014, mnemonic="ldr x0, [x1]",
        mem=(
            _Mem(rw="r", addr=0xa0000000, val=0xfeed, size=4),
            _Mem(rw="w", addr=0xc0000000, val=0xfeed, size=4),
        ),
    ))
    src = InMemoryDataSource(instructions=insns)
    r = discover_phase(0xc0000000, src, value_name="x", value_size=4,
                       cfg=PhaseDiscoveryConfig())
    assert r.crosses_out is True
    assert r.boundary is not None
    # The boundary should describe the source region, not the dest.
    assert r.boundary.region is not None
    assert r.boundary.region[0] == 0xa0000000
    # Walk produced at least one ``copy`` edge.
    assert any(e.kind == "copy" for e in r.chain)


# ---------------------------------------------------------------------------
# Disabled → result with no boundary.
# ---------------------------------------------------------------------------


def test_disabled_returns_empty_result():
    src = InMemoryDataSource(instructions=_trace_window(0x4000, 0x4020))
    r = discover_phase(0xbabe0000, src,
                       cfg=PhaseDiscoveryConfig(enabled=False))
    assert r.boundary is None
    assert r.crosses_out is False
    assert r.reason == "phase_discovery disabled"


# ---------------------------------------------------------------------------
# discover_phases_in_params — walks the dict for landing_address records.
# ---------------------------------------------------------------------------


def test_discover_phases_in_params_picks_up_landing_addr():
    insns = _trace_window(0x40001000, 0x40001020)
    src = InMemoryDataSource(instructions=insns)
    params = {
        "report": {
            "values": [
                {"value_name": "template", "source": "hook",
                 "landing_address": 0xbabe0000, "size": 32},
                # second value lives at an in-window-written addr →
                # should be filtered out as not crossing
            ],
        },
    }
    # Add an in-window writer for one of them so it's filtered out.
    insns[2] = _Ins(
        idx=2, pc=0x40001008, mnemonic="str",
        mem=(_Mem(rw="w", addr=0xbeef0000, val=1, size=4),),
    )
    params["report"]["values"].append(
        {"value_name": "key", "source": "hook",
         "landing_address": 0xbeef0000, "size": 4}
    )
    results = discover_phases_in_params(params, src, cfg=PhaseDiscoveryConfig())
    names = [r.value_addr for r in results]
    assert 0xbabe0000 in names
    assert 0xbeef0000 not in names  # in-window writer → no boundary


def test_render_alert_excludes_in_window_results():
    res = [
        PhaseDiscoveryResult(
            value_addr=0xbabe,
            boundary=None,
            crosses_out=False,
            reason="in_window",
        ),
    ]
    assert render_phase_discovery_alert(res) is None


def test_render_alert_present_for_crossings(tmp_path):
    src = InMemoryDataSource(instructions=_trace_window(0x4000, 0x4020))
    r = discover_phase(0xbabe0000, src, value_name="template",
                       value_size=32, cfg=PhaseDiscoveryConfig())
    line = render_phase_discovery_alert([r])
    assert line is not None
    assert "PHASE-DISCOVERY" in line
    assert "0xbabe0000" in line
