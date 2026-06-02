"""DisciplineWrapper integration for the phase gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from engine.discipline_wrapper import DisciplineWrapper
from engine.methodology import MethodologyConfig
from engine.phase_discovery import (
    InMemoryDataSource,
    PhaseDiscoveryConfig,
    WriterHit,
)
from engine.phase_instrument import PhaseInstrumentConfig


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


# ---------------------------------------------------------------------------
# Source provider that drives discovery off a synthesized trace window.
# ---------------------------------------------------------------------------


def _make_wrapper_with_window(
    window: list[_Ins],
    *,
    ledger_probe=None,
) -> DisciplineWrapper:
    def provider(core, method, params):
        return InMemoryDataSource(
            instructions=window, ledger_probe=ledger_probe,
        )

    return DisciplineWrapper(
        config=MethodologyConfig.from_env(),
        phase_discovery_config=PhaseDiscoveryConfig(),
        phase_instrument_config=PhaseInstrumentConfig(),
        phase_discovery_source_provider=provider,
    )


# ---------------------------------------------------------------------------
# Envelope carries phase_discovery + phase_instrument_suggestions when a
# crossings-out value is in the params dict.
# ---------------------------------------------------------------------------


def test_envelope_surfaces_phase_discovery_and_instrument():
    window = [_Ins(idx=i, pc=0x40010000 + i * 4, mnemonic="nop")
              for i in range(8)]
    # ledger probe stands in for an out-of-window writer at pc=0x4000_5200.
    def probe(addr, size):
        if addr == 0xbeef0000:
            return WriterHit(pc=0x40005200, addr=addr, size=4, value=1, idx=-1)
        return None

    w = _make_wrapper_with_window(window, ledger_probe=probe)
    params = {
        "report": {
            "values": [
                {"value_name": "callee_key", "source": "hook",
                 "landing_address": 0xbeef0000, "size": 4},
            ],
        },
    }

    def dispatch(method, p):
        return {"ok": True}

    result, env = w.step("submit_report", params, dispatch)
    assert result == {"ok": True}
    assert env.phase_discovery, "expected phase_discovery on envelope"
    assert env.phase_instrument_suggestions, \
        "expected phase_instrument_suggestions on envelope"
    pd = env.phase_discovery[0]
    assert pd["crosses_out"] is True
    assert pd["boundary"]["anchor"]["anchor_type"] == "addr_first_exec"
    sugg = env.phase_instrument_suggestions[0]
    assert sugg["spec"]["anchor"]["anchor_type"] == "addr_first_exec"
    assert sugg["spec"]["granularity"] == "full_instruction"
    # alert line emitted too
    assert any("PHASE-DISCOVERY" in a for a in env.alerts)
    assert any("PHASE-INSTRUMENT" in a for a in env.alerts)


def test_envelope_skips_when_no_source_provider():
    # No provider plugged — wrapper should not auto-run discovery.
    w = DisciplineWrapper(
        config=MethodologyConfig.from_env(),
        phase_discovery_config=PhaseDiscoveryConfig(),
        phase_instrument_config=PhaseInstrumentConfig(),
        phase_discovery_source_provider=None,
    )
    params = {
        "report": {
            "values": [
                {"value_name": "k", "source": "hook",
                 "landing_address": 0xbabe0000, "size": 32},
            ],
        },
    }
    _, env = w.step("submit_report", params, lambda m, p: {"ok": True})
    assert env.phase_discovery == []
    assert env.phase_instrument_suggestions == []


def test_envelope_skips_when_value_in_window():
    window = [
        _Ins(
            idx=0, pc=0x40010000, mnemonic="str",
            mem=(_Mem(rw="w", addr=0xbabe0000, val=0xdead, size=4),),
        ),
        _Ins(idx=1, pc=0x40010004, mnemonic="nop"),
    ]
    w = _make_wrapper_with_window(window)
    params = {
        "report": {
            "values": [
                {"value_name": "k", "source": "hook",
                 "landing_address": 0xbabe0000, "size": 4},
            ],
        },
    }
    _, env = w.step("submit_report", params, lambda m, p: {"ok": True})
    # In-window writer → no crossing-out, no instrument suggestion.
    assert env.phase_discovery == []
    assert env.phase_instrument_suggestions == []


def test_phase_discovery_disabled_env_toggle():
    window = [_Ins(idx=0, pc=0x4000, mnemonic="nop")]
    w = DisciplineWrapper(
        config=MethodologyConfig.from_env(),
        phase_discovery_config=PhaseDiscoveryConfig(enabled=False),
        phase_instrument_config=PhaseInstrumentConfig(),
        phase_discovery_source_provider=lambda c, m, p: InMemoryDataSource(
            instructions=window,
        ),
    )
    params = {
        "report": {
            "values": [
                {"value_name": "k", "source": "hook",
                 "landing_address": 0xbabe0000, "size": 4},
            ],
        },
    }
    _, env = w.step("submit_report", params, lambda m, p: {"ok": True})
    assert env.phase_discovery == []
    assert env.phase_instrument_suggestions == []


def test_phase_instrument_disabled_env_toggle():
    window = [_Ins(idx=0, pc=0x4000, mnemonic="nop")]
    w = DisciplineWrapper(
        config=MethodologyConfig.from_env(),
        phase_discovery_config=PhaseDiscoveryConfig(),
        phase_instrument_config=PhaseInstrumentConfig(enabled=False),
        phase_discovery_source_provider=lambda c, m, p: InMemoryDataSource(
            instructions=window,
        ),
    )
    params = {
        "report": {
            "values": [
                {"value_name": "k", "source": "hook",
                 "landing_address": 0xbabe0000, "size": 4},
            ],
        },
    }
    _, env = w.step("submit_report", params, lambda m, p: {"ok": True})
    # Discovery still runs.
    assert env.phase_discovery, "discovery should still fire"
    # Instrument suggestions should be empty due to toggle.
    assert env.phase_instrument_suggestions == []


def test_envelope_to_dict_includes_new_fields():
    window = [_Ins(idx=0, pc=0x40010000, mnemonic="nop")]
    w = _make_wrapper_with_window(
        window,
        ledger_probe=lambda a, s: (
            WriterHit(pc=0x40005000, addr=a, size=4, value=1, idx=-1)
            if a == 0xbabe0000 else None
        ),
    )
    _, env = w.step(
        "submit_report",
        {"report": {"values": [
            {"value_name": "k", "source": "hook",
             "landing_address": 0xbabe0000, "size": 4},
        ]}},
        lambda m, p: {"ok": True},
    )
    d = env.to_dict()
    assert "phase_discovery" in d
    assert "phase_instrument_suggestions" in d
