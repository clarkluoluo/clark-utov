"""Shared-type tests for engine.phase."""

from __future__ import annotations

import pytest

from engine.phase import (
    ANCHOR_ADDR_FIRST_EXEC,
    ANCHOR_FUNC_ENTRY,
    ANCHOR_MEMREGION_FIRST_ACCESS,
    Anchor,
    EntryState,
    KNOWN_ANCHOR_TYPES,
    PhaseBoundary,
    V1_ANCHOR_TYPES,
    boundaries_overlap,
    boundary_covers_addr,
    boundary_covers_pc,
    read_sidecar,
    register_anchor_type,
    write_sidecar,
)


def test_v1_anchor_types_are_exactly_three():
    assert V1_ANCHOR_TYPES == frozenset({
        ANCHOR_FUNC_ENTRY,
        ANCHOR_ADDR_FIRST_EXEC,
        ANCHOR_MEMREGION_FIRST_ACCESS,
    })


def test_anchor_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown anchor_type"):
        Anchor(anchor_type="xyzzy", params={})


def test_register_anchor_type_extends_registry():
    register_anchor_type("test_only_anchor")
    try:
        assert "test_only_anchor" in KNOWN_ANCHOR_TYPES
        a = Anchor(anchor_type="test_only_anchor", params={"x": 1})
        assert a.anchor_type == "test_only_anchor"
    finally:
        KNOWN_ANCHOR_TYPES.discard("test_only_anchor")


def test_anchor_to_dict_round_trip():
    a = Anchor(
        anchor_type=ANCHOR_FUNC_ENTRY,
        params={"pc": 0x40001000},
        label="initChain entry",
    )
    d = a.to_dict()
    assert d["anchor_type"] == "func_entry"
    assert d["params"]["pc"] == 0x40001000
    assert d["label"] == "initChain entry"


def test_phase_boundary_to_dict_emits_hex():
    b = PhaseBoundary(
        name="producer_of_template",
        pc_range=(0x40001000, 0x40001100),
        region=(0xbabe0000, 32),
    )
    d = b.to_dict()
    assert d["pc_range_hex"] == ["0x40001000", "0x40001100"]
    assert d["region_hex"] == ["0xbabe0000", 32]
    assert d["anchor"] is None


def test_boundary_covers_pc_and_addr():
    b = PhaseBoundary(
        name="p",
        pc_range=(0x1000, 0x1100),
        region=(0xa000, 16),
    )
    assert boundary_covers_pc(b, 0x1000)
    assert boundary_covers_pc(b, 0x10FF)
    assert not boundary_covers_pc(b, 0x1100)
    assert boundary_covers_addr(b, 0xa00f)
    assert not boundary_covers_addr(b, 0xa010)


def test_boundaries_overlap_detects_pc_and_idx_overlap():
    a = PhaseBoundary(name="a", entry_idx=10, exit_idx=20,
                      pc_range=(0x1000, 0x1100))
    b = PhaseBoundary(name="b", entry_idx=15, exit_idx=25,
                      pc_range=(0x1100, 0x1200))
    # idx overlaps (15 in [10,20)), pc does NOT touch.
    assert boundaries_overlap(a, b)
    c = PhaseBoundary(name="c", entry_idx=30, exit_idx=40,
                      pc_range=(0x1100, 0x1200))
    assert not boundaries_overlap(a, c)


def test_sidecar_round_trip(tmp_path):
    boundary = PhaseBoundary(
        name="initChain",
        pc_range=(0x40001000, 0x40001100),
        region=(0xbabe0000, 32),
        anchor=Anchor(
            anchor_type=ANCHOR_ADDR_FIRST_EXEC,
            params={"pc": 0x40001000},
            label="initChain first-exec",
        ),
        note="produced by phase_discovery",
    )
    state = EntryState(
        regs={"x0": 0x40001000, "sp": 0xbffff700},
        mem={0xbabe0000: b"\xde\xad\xbe\xef"},
    )
    path = tmp_path / "phase.sidecar.json"
    write_sidecar(path, boundary=boundary, entry_state=state,
                  source="phase_replay_fixture",
                  extra={"note": "test"})
    b2, s2, extra = read_sidecar(path)
    assert b2.name == "initChain"
    assert b2.pc_range == (0x40001000, 0x40001100)
    assert b2.region == (0xbabe0000, 32)
    assert b2.anchor is not None
    assert b2.anchor.anchor_type == "addr_first_exec"
    assert s2.regs["x0"] == 0x40001000
    assert s2.mem[0xbabe0000] == b"\xde\xad\xbe\xef"
    assert extra["note"] == "test"
