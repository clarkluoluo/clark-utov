"""phase_instrument — spec construction + auto-suggestion."""

from __future__ import annotations

import pytest

from engine.phase import (
    ANCHOR_ADDR_FIRST_EXEC,
    ANCHOR_FUNC_ENTRY,
    ANCHOR_MEMREGION_FIRST_ACCESS,
    Anchor,
    PhaseBoundary,
)
from engine.phase_discovery import PhaseDiscoveryResult
from engine.phase_instrument import (
    GRAN_FULL,
    GRAN_SPARSE_SAMPLE,
    KNOWN_GRANULARITIES,
    PhaseInstrumentConfig,
    PhaseInstrumentResult,
    PhaseInstrumentSpec,
    render_phase_instrument_alert,
    request_phase_instrument,
    suggest_instrument_for_boundary,
    suggest_instruments_for_results,
)


# ---------------------------------------------------------------------------
# Spec construction
# ---------------------------------------------------------------------------


def test_request_spec_with_func_entry_anchor():
    spec = request_phase_instrument(
        phase_name="initChain",
        anchor=Anchor(
            anchor_type=ANCHOR_FUNC_ENTRY,
            params={"pc": 0x40001000},
            label="initChain entry",
        ),
        granularity=GRAN_FULL,
        max_steps=5000,
        cfg=PhaseInstrumentConfig(),
    )
    assert spec.phase_name == "initChain"
    assert spec.granularity == "full_instruction"
    assert spec.anchor.anchor_type == "func_entry"
    assert spec.max_steps == 5000
    d = spec.to_dict()
    assert d["kind"] == "phase_instrument"
    assert d["anchor"]["anchor_type"] == "func_entry"


def test_request_spec_rejects_unknown_granularity():
    with pytest.raises(ValueError, match="unknown granularity"):
        PhaseInstrumentSpec(
            phase_name="x",
            anchor=Anchor(
                anchor_type=ANCHOR_FUNC_ENTRY,
                params={"pc": 0x1000},
            ),
            granularity="nonsense",
        )


def test_disabled_request_raises():
    with pytest.raises(RuntimeError, match="disabled"):
        request_phase_instrument(
            phase_name="x",
            anchor=Anchor(anchor_type=ANCHOR_FUNC_ENTRY, params={"pc": 0x1}),
            cfg=PhaseInstrumentConfig(enabled=False),
        )


def test_three_v1_anchors_all_accepted():
    cfg = PhaseInstrumentConfig()
    for at, params in [
        (ANCHOR_FUNC_ENTRY,             {"pc": 0x1000}),
        (ANCHOR_ADDR_FIRST_EXEC,        {"pc": 0x2000}),
        (ANCHOR_MEMREGION_FIRST_ACCESS, {"base": 0xa000, "length": 32, "access": "w"}),
    ]:
        spec = request_phase_instrument(
            phase_name=f"phase_{at}",
            anchor=Anchor(anchor_type=at, params=params),
            cfg=cfg,
        )
        assert spec.anchor.anchor_type == at


# ---------------------------------------------------------------------------
# Auto-suggestion from boundary
# ---------------------------------------------------------------------------


def test_suggest_uses_boundary_anchor_when_present():
    anchor = Anchor(
        anchor_type=ANCHOR_ADDR_FIRST_EXEC,
        params={"pc": 0x40001000},
        label="initChain",
    )
    boundary = PhaseBoundary(name="initChain", anchor=anchor,
                              pc_range=(0x40001000, 0x40001100))
    s = suggest_instrument_for_boundary(boundary, cfg=PhaseInstrumentConfig())
    assert s is not None
    assert s.spec is not None
    assert s.spec.anchor.anchor_type == "addr_first_exec"
    assert s.spec.granularity == "full_instruction"


def test_suggest_picks_addr_first_exec_when_pc_range_only():
    boundary = PhaseBoundary(
        name="callee_phase",
        pc_range=(0x40005000, 0x40005400),
    )
    s = suggest_instrument_for_boundary(boundary, cfg=PhaseInstrumentConfig())
    assert s is not None
    assert s.spec is not None
    assert s.spec.anchor.anchor_type == "addr_first_exec"
    assert s.spec.anchor.params["pc"] == 0x40005000


def test_suggest_picks_memregion_when_only_region():
    boundary = PhaseBoundary(
        name="region_phase",
        region=(0xbabe0000, 32),
    )
    s = suggest_instrument_for_boundary(boundary, cfg=PhaseInstrumentConfig())
    assert s is not None
    assert s.spec is not None
    assert s.spec.anchor.anchor_type == "memregion_first_access"
    assert s.spec.anchor.params["base"] == 0xbabe0000
    assert s.spec.anchor.params["length"] == 32
    # region also propagates to the regions list (so runner records
    # every read/write inside it regardless of granularity).
    assert (0xbabe0000, 32) in s.spec.regions


def test_suggest_advisory_only_when_no_anchor_possible():
    boundary = PhaseBoundary(name="unknown_phase")
    s = suggest_instrument_for_boundary(boundary, cfg=PhaseInstrumentConfig())
    assert s is not None
    assert s.spec is None
    assert "unresolvable" in s.advisory


def test_disabled_suggest_returns_none():
    boundary = PhaseBoundary(name="x", region=(0xa000, 4))
    assert suggest_instrument_for_boundary(
        boundary, cfg=PhaseInstrumentConfig(enabled=False),
    ) is None


def test_suggest_instruments_for_results_skips_in_window():
    results = [
        PhaseDiscoveryResult(value_addr=0x1, boundary=None, crosses_out=False),
        PhaseDiscoveryResult(
            value_addr=0xbabe,
            boundary=PhaseBoundary(name="p", region=(0xbabe, 32)),
            crosses_out=True,
        ),
    ]
    sugg = suggest_instruments_for_results(results, cfg=PhaseInstrumentConfig())
    assert len(sugg) == 1
    assert sugg[0].phase_name == "p"


def test_render_alert_includes_anchor_and_granularity():
    boundary = PhaseBoundary(name="initChain", region=(0xbabe, 32))
    s = suggest_instrument_for_boundary(boundary, cfg=PhaseInstrumentConfig())
    line = render_phase_instrument_alert([s])
    assert line is not None
    assert "initChain" in line
    assert "memregion_first_access" in line
    assert "full_instruction" in line


def test_result_to_dict_round_trip(tmp_path):
    spec = request_phase_instrument(
        phase_name="p",
        anchor=Anchor(anchor_type=ANCHOR_FUNC_ENTRY, params={"pc": 0x1000}),
        cfg=PhaseInstrumentConfig(),
    )
    r = PhaseInstrumentResult(
        spec=spec,
        jsonl_path=tmp_path / "p.trace.jsonl",
        sidecar_path=tmp_path / "p.sidecar.json",
        anchor_hit_idx=12,
        captured_steps=345,
        truncated=False,
    )
    d = r.to_dict()
    assert d["captured_steps"] == 345
    assert d["spec"]["kind"] == "phase_instrument"
