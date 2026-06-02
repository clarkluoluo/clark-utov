"""phase_replay — ReplayableUnit construction + JsonlTraceReader compat."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from engine.phase import (
    ANCHOR_FUNC_ENTRY,
    Anchor,
    EntryState,
    PhaseBoundary,
    ReplayableUnit,
)
from engine.phase_instrument import (
    PhaseInstrumentConfig,
    PhaseInstrumentResult,
    request_phase_instrument,
)
from engine.phase_replay import (
    load_replayable_unit,
    make_replayable_unit,
    open_replayable_unit,
    synthesize_unit_from_instructions,
)
from engine.runner_client import JsonlTraceReader
from engine.types import Instruction, MemOp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sample_instructions() -> list[Instruction]:
    return [
        Instruction(
            idx=0,
            pc=0x40001000,
            bytes_=bytes.fromhex("ff8301d1"),
            mnemonic="sub sp, sp, #0x60",
            regs_read={"sp": 0xbffff700},
            regs_write={"sp": 0xbffff6a0},
            mem=(),
        ),
        Instruction(
            idx=1,
            pc=0x40001004,
            bytes_=bytes.fromhex("e003a8d2"),
            mnemonic="mov x0, #0x4001f",
            regs_read={},
            regs_write={"x0": 0x4001f},
            mem=(),
        ),
        Instruction(
            idx=2,
            pc=0x40001008,
            bytes_=bytes.fromhex("000040b9"),
            mnemonic="ldr w0, [x0]",
            regs_read={"x0": 0xbabe0000},
            regs_write={"x0": 0xdeadbeef},
            mem=(
                MemOp(rw="r", addr=0xbabe0000, val=0xdeadbeef, size=4),
            ),
        ),
    ]


def _sample_boundary() -> PhaseBoundary:
    return PhaseBoundary(
        name="initChain",
        pc_range=(0x40001000, 0x40001100),
        region=(0xbabe0000, 32),
        anchor=Anchor(
            anchor_type=ANCHOR_FUNC_ENTRY,
            params={"pc": 0x40001000},
            label="initChain entry",
        ),
        note="located by phase_discovery",
    )


def _sample_entry_state() -> EntryState:
    return EntryState(
        regs={"x0": 0x40001000, "sp": 0xbffff700},
        mem={0xbabe0000: b"\x00" * 32},
    )


# ---------------------------------------------------------------------------
# Synthesize from in-memory instructions.
# ---------------------------------------------------------------------------


def test_synthesize_unit_emits_jsonl_and_sidecar(tmp_path):
    insns = _sample_instructions()
    unit = synthesize_unit_from_instructions(
        insns,
        boundary=_sample_boundary(),
        entry_state=_sample_entry_state(),
        out_dir=tmp_path,
    )
    assert unit.jsonl_path.exists()
    assert unit.sidecar_path.exists()
    assert unit.source == "phase_replay_fixture"


def test_synthesized_unit_reads_back_via_jsonl_trace_reader(tmp_path):
    insns = _sample_instructions()
    unit = synthesize_unit_from_instructions(
        insns,
        boundary=_sample_boundary(),
        entry_state=_sample_entry_state(),
        out_dir=tmp_path,
    )
    reader = open_replayable_unit(unit)
    assert isinstance(reader, JsonlTraceReader)
    read_back = list(reader)
    assert len(read_back) == 3
    assert read_back[0].pc == 0x40001000
    assert read_back[2].mnemonic == "ldr w0, [x0]"
    # Mem ops survive the round trip.
    assert read_back[2].mem[0].addr == 0xbabe0000
    assert read_back[2].mem[0].rw == "r"


def test_unit_iterable_matches_a_main_trace(tmp_path):
    """Acceptance: a ReplayableUnit's iterator yields the same shape
    of records as a JsonlTraceReader on a regular trace, so any
    stage code that takes a trace iterator works without branching
    on source. This test exercises the contract directly."""
    insns = _sample_instructions()
    unit = synthesize_unit_from_instructions(
        insns,
        boundary=_sample_boundary(),
        entry_state=_sample_entry_state(),
        out_dir=tmp_path,
    )

    # Now write a *separate* JSONL file with the same shape, treat it
    # as a "main trace", and check the reader yields equivalent values.
    main_jsonl = tmp_path / "main.trace.jsonl"
    import json as _json
    with main_jsonl.open("w", encoding="utf-8") as f:
        for ins in insns:
            f.write(_json.dumps({
                "idx": ins.idx,
                "pc": f"0x{ins.pc:x}",
                "bytes": ins.bytes_.hex(),
                "mnemonic": ins.mnemonic,
                "regs_read": {k: f"0x{v:x}" for k, v in ins.regs_read.items()},
                "regs_write": {k: f"0x{v:x}" for k, v in ins.regs_write.items()},
                "mem": [
                    {"rw": m.rw, "addr": f"0x{m.addr:x}",
                     "val": f"0x{m.val:x}", "size": m.size}
                    for m in ins.mem
                ],
            }) + "\n")

    main_reader = JsonlTraceReader(main_jsonl)
    main_list = list(main_reader)
    unit_list = list(open_replayable_unit(unit))
    assert len(main_list) == len(unit_list)
    for a, b in zip(main_list, unit_list):
        assert a.idx == b.idx
        assert a.pc == b.pc
        assert a.mnemonic == b.mnemonic
        assert a.regs_read == b.regs_read
        assert a.regs_write == b.regs_write
        assert a.mem == b.mem


# ---------------------------------------------------------------------------
# make_replayable_unit (from a real PhaseInstrumentResult)
# ---------------------------------------------------------------------------


def test_make_replayable_unit_from_result(tmp_path):
    insns = _sample_instructions()
    spec = request_phase_instrument(
        phase_name="initChain",
        anchor=Anchor(
            anchor_type=ANCHOR_FUNC_ENTRY,
            params={"pc": 0x40001000},
        ),
        cfg=PhaseInstrumentConfig(),
    )
    # The "runner" writes the JSONL itself in real life. Here we
    # synthesize one first to mimic that.
    pre = synthesize_unit_from_instructions(
        insns,
        boundary=_sample_boundary(),
        entry_state=_sample_entry_state(),
        out_dir=tmp_path,
    )
    result = PhaseInstrumentResult(
        spec=spec,
        jsonl_path=pre.jsonl_path,
        sidecar_path=tmp_path / "rewritten.sidecar.json",
        anchor_hit_idx=42,
        captured_steps=3,
        truncated=False,
    )
    unit = make_replayable_unit(
        result, _sample_entry_state(), _sample_boundary(),
    )
    assert unit.jsonl_path.exists()
    assert unit.sidecar_path.exists()


def test_make_replayable_unit_fails_when_jsonl_missing(tmp_path):
    spec = request_phase_instrument(
        phase_name="x",
        anchor=Anchor(anchor_type=ANCHOR_FUNC_ENTRY, params={"pc": 0x1}),
        cfg=PhaseInstrumentConfig(),
    )
    result = PhaseInstrumentResult(
        spec=spec,
        jsonl_path=tmp_path / "missing.jsonl",
        sidecar_path=tmp_path / "missing.sidecar.json",
        anchor_hit_idx=0,
        captured_steps=0,
    )
    with pytest.raises(FileNotFoundError):
        make_replayable_unit(result, _sample_entry_state(), _sample_boundary())


def test_load_replayable_unit_round_trip(tmp_path):
    unit = synthesize_unit_from_instructions(
        _sample_instructions(),
        boundary=_sample_boundary(),
        entry_state=_sample_entry_state(),
        out_dir=tmp_path,
    )
    reloaded = load_replayable_unit(unit.jsonl_path, unit.sidecar_path)
    assert reloaded.boundary.name == "initChain"
    assert reloaded.boundary.region == (0xbabe0000, 32)
    assert reloaded.entry_state.regs["sp"] == 0xbffff700
