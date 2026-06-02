"""DisciplineWrapper integration for block_cause routing.

The contract under test:

  1. When a router is configured AND ``hide_raw_phase_outputs=True``
     (the default), the envelope contains a ``block_cause`` sibling
     and DOES NOT contain ``phase_discovery`` / ``phase_instrument_suggestions``.
     This is the L1-only-routes-conclusion path.

  2. When ``UTOV_PHASE_DEBUG=1`` (or equivalently
     ``hide_raw_phase_outputs=False``), all three siblings appear so
     a developer can verify the routing decision against the raw
     intermediates.

  3. When no router is wired, the wrapper keeps the legacy
     raw-surfacing behavior so opt-in is non-breaking.

  4. Regression: ``phase_discovery.crosses_out=True`` MUST NOT
     surface to the agent as a phase_discovery sibling when a
     router is configured — otherwise the agent receives the class-1
     gap as a decision prompt (the very job-chain misalignment this
     module fixes).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from engine.block_cause import (
    BlockCauseConfig,
    BlockCauseRouter,
    StaticCapabilityOracle,
)
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


def _wrapper(
    *,
    router: BlockCauseRouter | None,
    block_cause_config: BlockCauseConfig | None = None,
) -> DisciplineWrapper:
    window = [_Ins(idx=i, pc=0x40010000 + i * 4) for i in range(4)]

    def provider(core, method, params):
        return InMemoryDataSource(
            instructions=window,
            ledger_probe=lambda a, s: None,   # always treat as crossing out
        )

    return DisciplineWrapper(
        config=MethodologyConfig.from_env(),
        phase_discovery_config=PhaseDiscoveryConfig(),
        phase_instrument_config=PhaseInstrumentConfig(),
        phase_discovery_source_provider=provider,
        block_cause_config=block_cause_config,
        block_cause_router=router,
    )


def _params_with_observed_value() -> dict:
    return {
        "report": {
            "values": [
                {"value_name": "template", "source": "hook",
                 "landing_address": 0xbabe0000, "size": 32},
            ],
        },
    }


# ---------------------------------------------------------------------------
# Default: router active, hide raw — only block_cause surfaces.
# ---------------------------------------------------------------------------


def test_router_active_hides_raw_phase_outputs_by_default():
    router = BlockCauseRouter(
        oracle=StaticCapabilityOracle(static=frozenset({"memregion_watch"})),
    )
    w = _wrapper(router=router)
    _, env = w.step(
        "submit_report", _params_with_observed_value(),
        lambda m, p: {"ok": True},
    )
    assert env.block_cause, "expected block_cause on envelope"
    assert env.phase_discovery == []
    assert env.phase_instrument_suggestions == []
    # block_cause carries the routing conclusion, not the raw discovery.
    bc0 = env.block_cause[0]
    assert bc0["classification"]["class"] == "collection_gap"
    assert bc0["action"] == "auto_collect"
    assert bc0["rerun_request"] is not None


# ---------------------------------------------------------------------------
# Phase debug toggle — all three siblings appear.
# ---------------------------------------------------------------------------


def test_phase_debug_toggle_restores_raw_siblings():
    cfg = BlockCauseConfig(hide_raw_phase_outputs=False)
    router = BlockCauseRouter(
        oracle=StaticCapabilityOracle(static=frozenset({"memregion_watch"})),
    )
    w = _wrapper(router=router, block_cause_config=cfg)
    _, env = w.step(
        "submit_report", _params_with_observed_value(),
        lambda m, p: {"ok": True},
    )
    assert env.block_cause
    # Raw siblings restored.
    assert env.phase_discovery, "phase_discovery should appear in debug mode"
    assert env.phase_instrument_suggestions, \
        "phase_instrument_suggestions should appear in debug mode"


# ---------------------------------------------------------------------------
# No router → legacy raw-surfacing behavior preserved.
# ---------------------------------------------------------------------------


def test_no_router_keeps_legacy_raw_surfacing():
    w = _wrapper(router=None)
    _, env = w.step(
        "submit_report", _params_with_observed_value(),
        lambda m, p: {"ok": True},
    )
    # No block_cause when router not wired.
    assert env.block_cause == []
    # Legacy: raw siblings DO appear.
    assert env.phase_discovery
    assert env.phase_instrument_suggestions


# ---------------------------------------------------------------------------
# block_cause disabled toggle: router exists but routing skipped.
# ---------------------------------------------------------------------------


def test_block_cause_disabled_toggle_skips_routing():
    cfg = BlockCauseConfig(enabled=False)
    router = BlockCauseRouter(
        oracle=StaticCapabilityOracle(static=frozenset({"memregion_watch"})),
    )
    w = _wrapper(router=router, block_cause_config=cfg)
    _, env = w.step(
        "submit_report", _params_with_observed_value(),
        lambda m, p: {"ok": True},
    )
    # No routing → no block_cause; and since hide_raw guard is tied
    # to (router_active AND enabled), raw siblings come back.
    assert env.block_cause == []
    assert env.phase_discovery
    assert env.phase_instrument_suggestions


# ---------------------------------------------------------------------------
# Regression: crosses_out NEVER surfaces directly when router is active.
# This is the core job-chain-alignment guarantee.
# ---------------------------------------------------------------------------


def test_regression_class1_never_surfaces_as_phase_discovery_to_agent():
    router = BlockCauseRouter(
        oracle=StaticCapabilityOracle(static=frozenset()),    # capability missing
    )
    w = _wrapper(router=router)
    _, env = w.step(
        "submit_report", _params_with_observed_value(),
        lambda m, p: {"ok": True},
    )
    # The gap IS in block_cause (as a backlog entry, owned by clark).
    assert env.block_cause
    assert env.block_cause[0]["action"] == "register_backlog"
    # And it is NOT in phase_discovery / phase_instrument_suggestions —
    # the agent never sees a "should we collect?" prompt.
    assert env.phase_discovery == []
    assert env.phase_instrument_suggestions == []


def test_envelope_to_dict_includes_block_cause():
    router = BlockCauseRouter(
        oracle=StaticCapabilityOracle(static=frozenset({"memregion_watch"})),
    )
    w = _wrapper(router=router)
    _, env = w.step(
        "submit_report", _params_with_observed_value(),
        lambda m, p: {"ok": True},
    )
    d = env.to_dict()
    assert "block_cause" in d
    assert "phase_discovery" not in d
    assert "phase_instrument_suggestions" not in d
