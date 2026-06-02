"""End-to-end acceptance for the three-primitive phase capability.

Two synthesized scenarios — both follow the same pipeline:

    key value (landing addr)
      → phase_discovery (locate phase)
      → phase_instrument suggestion (anchor + granularity)
      → phase_replay (wrap captured instructions as ReplayableUnit)
      → JsonlTraceReader (same as main trace) → main pipeline

Scenario A. Key value computed inside a *callee*. The main trace
window covers the caller; the callee runs at PCs outside the
window. Discovery should locate the callee's pc_range via the
out-of-window writer.

Scenario B. Key value computed during *library load*. There's no
in-window writer at all — the value is already in memory by the
time the main trace begins. Discovery should produce a memregion
anchor describing the value's landing address.

Both scenarios MUST end with a ReplayableUnit whose JSONL is
consumable by :class:`engine.runner_client.JsonlTraceReader` —
the main-pipeline interface — without any per-phase branching.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from engine.phase import (
    ANCHOR_ADDR_FIRST_EXEC,
    ANCHOR_MEMREGION_FIRST_ACCESS,
    Anchor,
    EntryState,
    PhaseBoundary,
)
from engine.phase_discovery import (
    InMemoryDataSource,
    PhaseDiscoveryConfig,
    WriterHit,
    discover_phase,
)
from engine.phase_instrument import (
    PhaseInstrumentConfig,
    suggest_instrument_for_boundary,
)
from engine.phase_replay import (
    open_replayable_unit,
    synthesize_unit_from_instructions,
)
from engine.runner_client import JsonlTraceReader
from engine.types import Instruction, MemOp


# ---------------------------------------------------------------------------
# Scenario A — key value computed inside a callee outside the main window.
# ---------------------------------------------------------------------------


def test_scenario_callee_internal_value(tmp_path):
    # Main trace window: [0x4001_0000, 0x4001_0100). The trace only
    # *uses* the key value; the producer ran at PC 0x4000_5200,
    # outside the window. We model this by routing latest_writer to
    # an out-of-window writer via the ledger probe.

    main_window = [
        Instruction(
            idx=i, pc=0x40010000 + i * 4, bytes_=b"\x00" * 4,
            mnemonic="use", regs_read={}, regs_write={}, mem=(),
        )
        for i in range(0x40)  # ~ 64 instructions
    ]
    out_of_window_writer_pc = 0x40005200

    def probe(addr: int, size: int) -> WriterHit | None:
        if addr == 0xbeefcafe:
            return WriterHit(
                pc=out_of_window_writer_pc,
                addr=addr, size=4, value=0x4242, idx=-1,
            )
        return None

    src = InMemoryDataSource(instructions=main_window, ledger_probe=probe)
    discovery = discover_phase(
        0xbeefcafe, src,
        value_name="callee_key", value_size=4,
        phase_name="callee_phase",
        cfg=PhaseDiscoveryConfig(),
    )
    assert discovery.crosses_out is True
    assert discovery.boundary is not None
    assert discovery.boundary.anchor is not None
    assert discovery.boundary.anchor.anchor_type == ANCHOR_ADDR_FIRST_EXEC
    assert discovery.boundary.anchor.params["pc"] == out_of_window_writer_pc

    # Instrument auto-suggestion: full granularity, anchored at the
    # callee's first-execute PC.
    sugg = suggest_instrument_for_boundary(
        discovery.boundary, cfg=PhaseInstrumentConfig(),
    )
    assert sugg is not None
    assert sugg.spec is not None
    assert sugg.spec.granularity == "full_instruction"
    assert sugg.spec.anchor.anchor_type == ANCHOR_ADDR_FIRST_EXEC

    # Replay: synthesize a captured trace for the callee phase. This
    # stands in for the runner-side fulfilment of the spec.
    callee_insns = [
        Instruction(
            idx=0, pc=out_of_window_writer_pc, bytes_=b"\xff\xff\xff\xff",
            mnemonic="mov w0, #0x4242",
            regs_read={}, regs_write={"w0": 0x4242}, mem=(),
        ),
        Instruction(
            idx=1, pc=out_of_window_writer_pc + 4, bytes_=b"\xff\xff\xff\xff",
            mnemonic="str w0, [x1]",
            regs_read={"w0": 0x4242, "x1": 0xbeefcafe},
            regs_write={},
            mem=(MemOp(rw="w", addr=0xbeefcafe, val=0x4242, size=4),),
        ),
    ]
    unit = synthesize_unit_from_instructions(
        callee_insns,
        boundary=discovery.boundary,
        entry_state=EntryState(
            regs={"x1": 0xbeefcafe, "lr": 0x40010100},
            mem={},
        ),
        out_dir=tmp_path,
    )

    # Acceptance: the unit's JSONL is consumable by the MAIN pipeline
    # reader. Same class, same iteration shape.
    reader = open_replayable_unit(unit)
    assert isinstance(reader, JsonlTraceReader)
    rows = list(reader)
    assert len(rows) == 2
    # The producer write is present in the captured stream — main
    # pipeline can now slice/symex it.
    write_rows = [
        r for r in rows
        if any(m.rw == "w" and m.addr == 0xbeefcafe for m in r.mem)
    ]
    assert len(write_rows) == 1
    assert write_rows[0].pc == out_of_window_writer_pc + 4


# ---------------------------------------------------------------------------
# Scenario B — key value computed during library load (no writer anywhere).
# ---------------------------------------------------------------------------


def test_scenario_libload_period_value(tmp_path):
    # Main trace window: only later execution; the value at landing
    # address 0xbabe_0000 was already written during library load.
    # No writer is reachable from any data source.
    main_window = [
        Instruction(
            idx=i, pc=0x40010000 + i * 4, bytes_=b"\x00" * 4,
            mnemonic="use", regs_read={}, regs_write={}, mem=(),
        )
        for i in range(0x10)
    ]
    src = InMemoryDataSource(instructions=main_window)  # no ledger probe

    discovery = discover_phase(
        0xbabe0000, src,
        value_name="libload_blob", value_size=32,
        phase_name="libload_phase",
        cfg=PhaseDiscoveryConfig(),
    )
    assert discovery.crosses_out is True
    assert discovery.boundary is not None
    # No-writer-anywhere ⇒ memregion anchor on the value's address.
    assert discovery.boundary.anchor is not None
    assert discovery.boundary.anchor.anchor_type == ANCHOR_MEMREGION_FIRST_ACCESS
    assert discovery.boundary.anchor.params["base"] == 0xbabe0000
    assert discovery.boundary.anchor.params["length"] >= 32

    sugg = suggest_instrument_for_boundary(
        discovery.boundary, cfg=PhaseInstrumentConfig(),
    )
    assert sugg is not None
    assert sugg.spec is not None
    assert sugg.spec.anchor.anchor_type == ANCHOR_MEMREGION_FIRST_ACCESS
    # Region propagated so the runner captures every read/write of
    # the value at any granularity.
    assert (0xbabe0000, 32) in sugg.spec.regions

    # Replay: synthesize a libload-period trace that builds the blob.
    libload_insns: list[Instruction] = []
    for i in range(8):  # 8 × 4-byte stores → 32 bytes
        libload_insns.append(Instruction(
            idx=i, pc=0x40000400 + i * 4, bytes_=b"\xaa" * 4,
            mnemonic="str w0, [x1]",
            regs_read={"x1": 0xbabe0000 + i * 4, "w0": 0x11223344},
            regs_write={},
            mem=(MemOp(rw="w", addr=0xbabe0000 + i * 4,
                       val=0x11223344, size=4),),
        ))
    unit = synthesize_unit_from_instructions(
        libload_insns,
        boundary=discovery.boundary,
        entry_state=EntryState(regs={"sp": 0xbffff700}, mem={}),
        out_dir=tmp_path,
    )

    reader = open_replayable_unit(unit)
    assert isinstance(reader, JsonlTraceReader)
    rows = list(reader)
    writes_to_region = [
        m for r in rows for m in r.mem
        if m.rw == "w" and 0xbabe0000 <= m.addr < 0xbabe0000 + 32
    ]
    assert len(writes_to_region) == 8


# ---------------------------------------------------------------------------
# Anti-target-leakage — capability code must NOT reference any
# target-specific names. Sanity-check the implementation modules.
# ---------------------------------------------------------------------------


def test_capability_code_is_target_agnostic():
    """The three primitive modules must not name reference-target-specific
    symbols (the first validation target should be a *consumer* of
    the capability, not baked in). The forbidden tuple keeps the historical
    target tokens as a denylist so they can never leak back into the
    capability layer."""
    import inspect
    from engine import phase, phase_discovery, phase_instrument, phase_replay
    forbidden = ("reference_target", "template32", "12309240", "scratch21")
    for mod in (phase, phase_discovery, phase_instrument, phase_replay):
        src = inspect.getsource(mod).lower()
        for word in forbidden:
            assert word not in src, (
                f"target-specific symbol {word!r} found in {mod.__name__}; "
                f"capability must stay general"
            )
