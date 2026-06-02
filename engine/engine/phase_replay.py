"""Phase replay — package a captured phase as a ReplayableUnit.

A :class:`engine.phase.ReplayableUnit` is the engineering crystal
of the design's core abstraction — **phase is a trace source, not a
special-cased object**. The unit consists of:

  * a JSONL trace file in the *exact* schema the main pipeline
    already consumes (contracts/runner_interface.md §2.1), and
  * a sidecar JSON file carrying entry register state, memory
    snapshot, and phase boundary metadata.

Consumers feed the unit into the existing pipeline:

  * :class:`engine.runner_client.JsonlTraceReader` reads the JSONL.
    No new reader class is needed — the unit is just a path-pair.
  * Stages that need entry state pull from the sidecar via
    :func:`engine.phase.read_sidecar`. The sidecar is *optional*
    consumer information; stages that only need the instruction
    stream ignore it entirely.

This module supplies the constructor (:func:`make_replayable_unit`),
the inverse opener (:func:`open_replayable_unit`), and a
fixture-only synthesizer (:func:`synthesize_unit_from_instructions`)
that lets tests build a unit from in-memory instruction records
without needing a runner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .phase import (
    EntryState,
    PhaseBoundary,
    ReplayableUnit,
    read_sidecar,
    write_sidecar,
)
from .phase_instrument import PhaseInstrumentResult, PhaseInstrumentSpec
from .runner_client import JsonlTraceReader


# ---------------------------------------------------------------------------
# Construction from a runner result.
# ---------------------------------------------------------------------------


def make_replayable_unit(
    result: PhaseInstrumentResult,
    entry_state: EntryState,
    boundary: PhaseBoundary,
    *,
    source: str = "phase_replay",
) -> ReplayableUnit:
    """Wrap a runner-supplied :class:`PhaseInstrumentResult` plus
    its entry state into a :class:`ReplayableUnit`.

    The runner writes the JSONL trace itself; this function does not
    rewrite it. It does (re)write the sidecar from the supplied
    ``entry_state`` + ``boundary`` so the unit is self-describing —
    runners that omit the sidecar (older shims) still produce a
    usable unit through this path.

    The JSONL file's existence is checked but not its contents — the
    main pipeline will fail later with a clearer error if the schema
    is wrong. We do not duplicate parser logic here.
    """
    jsonl = Path(result.jsonl_path)
    if not jsonl.exists():
        raise FileNotFoundError(f"runner-reported JSONL trace missing: {jsonl}")
    sidecar = Path(result.sidecar_path)
    write_sidecar(
        sidecar,
        boundary=boundary,
        entry_state=entry_state,
        source=source,
        extra={
            "anchor_hit_idx": result.anchor_hit_idx,
            "captured_steps": result.captured_steps,
            "truncated":      result.truncated,
            "instrument_spec": result.spec.to_dict(),
        },
    )
    return ReplayableUnit(
        jsonl_path=jsonl,
        sidecar_path=sidecar,
        boundary=boundary,
        entry_state=entry_state,
        source=source,
    )


# ---------------------------------------------------------------------------
# Opener — main-pipeline-compatible iterator over the instruction stream.
# ---------------------------------------------------------------------------


def open_replayable_unit(unit: ReplayableUnit) -> JsonlTraceReader:
    """Return a :class:`JsonlTraceReader` over the unit's JSONL.

    The deliberate point: callers in the main pipeline get a reader
    of *exactly* the same type they use for the main trace. No
    new abstraction, no per-source branching. Stage code:

        reader = open_replayable_unit(unit)  # or JsonlTraceReader(main_path)
        for ins in reader:
            ...
    """
    return JsonlTraceReader(unit.jsonl_path)


def load_replayable_unit(
    jsonl_path: str | Path,
    sidecar_path: str | Path,
) -> ReplayableUnit:
    """Load a unit from disk paths. The inverse of
    :func:`make_replayable_unit` — reads back what was written."""
    jsonl = Path(jsonl_path)
    if not jsonl.exists():
        raise FileNotFoundError(f"JSONL trace missing: {jsonl}")
    sidecar = Path(sidecar_path)
    if not sidecar.exists():
        raise FileNotFoundError(f"sidecar missing: {sidecar}")
    boundary, entry_state, extra = read_sidecar(sidecar)
    source = extra.get("source") or "phase_replay"
    # ``source`` actually lives in the sidecar payload root, not in
    # extra; tolerate both shapes.
    if not isinstance(source, str):
        source = "phase_replay"
    raw = json.loads(sidecar.read_text(encoding="utf-8"))
    if isinstance(raw.get("source"), str):
        source = raw["source"]
    return ReplayableUnit(
        jsonl_path=jsonl,
        sidecar_path=sidecar,
        boundary=boundary,
        entry_state=entry_state,
        source=source,
    )


# ---------------------------------------------------------------------------
# Fixture synthesizer — tests / file-mode runs.
# ---------------------------------------------------------------------------


def synthesize_unit_from_instructions(
    instructions: Iterable[Any],
    *,
    boundary: PhaseBoundary,
    entry_state: EntryState,
    out_dir: str | Path,
    spec: PhaseInstrumentSpec | None = None,
    source: str = "phase_replay_fixture",
) -> ReplayableUnit:
    """Build a unit from in-memory instruction records — used by
    tests and by file-mode runs that synthesize a phase from a
    captured trace slice.

    ``instructions`` must yield records compatible with the JSONL
    schema (contracts §2.1). Each must have at least: ``idx``,
    ``pc`` (int), ``bytes_`` (bytes) or ``bytes`` (hex str),
    ``mnemonic``, ``regs_read``, ``regs_write``. ``mem`` items use
    the same shape as :class:`engine.types.MemOp` (``rw``, ``addr``,
    ``val``, ``size``).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / f"{boundary.name or 'phase'}.trace.jsonl"
    sidecar = out / f"{boundary.name or 'phase'}.sidecar.json"

    records: list[dict[str, Any]] = []
    for ins in instructions:
        records.append(_instruction_to_jsonl_record(ins))
    with jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    extra: dict[str, Any] = {
        "captured_steps": len(records),
        "synthesized":    True,
    }
    if spec is not None:
        extra["instrument_spec"] = spec.to_dict()
    write_sidecar(
        sidecar,
        boundary=boundary,
        entry_state=entry_state,
        source=source,
        extra=extra,
    )
    return ReplayableUnit(
        jsonl_path=jsonl,
        sidecar_path=sidecar,
        boundary=boundary,
        entry_state=entry_state,
        source=source,
    )


def _instruction_to_jsonl_record(ins: Any) -> dict[str, Any]:
    bytes_attr = getattr(ins, "bytes_", None)
    if bytes_attr is None:
        # tolerate field named ``bytes`` (kwarg conflict; rare in
        # our types but cheap to accept)
        bytes_attr = getattr(ins, "bytes", b"")
    if isinstance(bytes_attr, (bytes, bytearray)):
        bytes_hex = bytes(bytes_attr).hex()
    else:
        bytes_hex = str(bytes_attr)
    mem_records: list[dict[str, Any]] = []
    for m in getattr(ins, "mem", ()) or ():
        mem_records.append({
            "rw":   getattr(m, "rw"),
            "addr": f"0x{int(getattr(m, 'addr')):x}",
            "val":  f"0x{int(getattr(m, 'val', 0)):x}",
            "size": int(getattr(m, "size", 1)),
        })
    return {
        "idx":     int(getattr(ins, "idx")),
        "pc":      f"0x{int(getattr(ins, 'pc')):x}",
        "bytes":   bytes_hex,
        "mnemonic": getattr(ins, "mnemonic", ""),
        "regs_read":  {
            k: f"0x{int(v):x}"
            for k, v in (getattr(ins, "regs_read", {}) or {}).items()
        },
        "regs_write": {
            k: f"0x{int(v):x}"
            for k, v in (getattr(ins, "regs_write", {}) or {}).items()
        },
        "mem": mem_records,
    }


__all__ = [
    "make_replayable_unit",
    "open_replayable_unit",
    "load_replayable_unit",
    "synthesize_unit_from_instructions",
]
