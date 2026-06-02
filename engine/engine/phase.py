"""Phase-scoped observation — shared types.

Background. Trace consumption today is phase-agnostic. When key data
for the main algorithm is *computed* in a phase outside the main
observation window (a callee, library-load period, callback, delayed
init, child thread, etc.), the main trace only sees the *use* of
that data, not its construction. Restoring the key data requires:

  (1) **locating** the producing phase (:mod:`engine.phase_discovery`),
  (2) **instrumenting** that phase at the necessary resolution
      (:mod:`engine.phase_instrument`),
  (3) **wrapping** the captured execution into a unit that the
      existing main pipeline can consume *without per-phase
      branching* (:mod:`engine.phase_replay`).

Core abstraction — **phase is a source of trace, not a special-cased
object**. A :class:`ReplayableUnit` is exactly one JSONL trace
(consumable by :class:`engine.runner_client.JsonlTraceReader`) plus a
sidecar carrying entry register state, memory snapshot, and phase
metadata. The main pipeline stages do not branch on phase source.

This module supplies the shared types only.

Anchor type registry. v1 ships three anchor kinds that can be
described purely in terms of PC / addresses and therefore drive
fixtures end-to-end without a live runner. Future kinds
(``libload_done``, ``thread_start``) are runtime/OS events and will
land alongside their runner-side signals; until then a caller can
extend the registry but the in-tree primitives only emit the v1 set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Anchor types — extensible enum.
# ---------------------------------------------------------------------------


# Function-entry anchor — fire on the first execution at a PC that
# matches a function's entry. Parameters: ``pc`` (int).
ANCHOR_FUNC_ENTRY = "func_entry"

# Address-first-execute anchor — fire on the first time *any* of a
# set of PCs is executed. Useful for "library entry first reached",
# "constructor first reached", "delayed-init thunk first reached".
# Parameters: ``pc`` (int) or ``pc_any_of`` (list[int]).
ANCHOR_ADDR_FIRST_EXEC = "addr_first_exec"

# Memory-region-first-access anchor — fire on the first access (read
# or write, configurable) to any address in [base, base+length). The
# bridge from :mod:`engine.phase_discovery` (which often locates a
# producing phase by "the source address of this load has no in-window
# writer") to :mod:`engine.phase_instrument` — discovery emits a
# region, instrument hangs an anchor on that region.
# Parameters: ``base`` (int), ``length`` (int), ``access`` ("r"|"w"|"rw").
ANCHOR_MEMREGION_FIRST_ACCESS = "memregion_first_access"


# v1 anchors. All three describable purely by PC/address, no runtime
# signals, no OS events — runnable from synthesized JSONL fixtures.
V1_ANCHOR_TYPES: frozenset[str] = frozenset({
    ANCHOR_FUNC_ENTRY,
    ANCHOR_ADDR_FIRST_EXEC,
    ANCHOR_MEMREGION_FIRST_ACCESS,
})


# Registry — mutable so a future runner-side patch can register
# ``libload_done`` / ``thread_start`` without touching this module's
# v1 surface. Callers should treat membership in
# :data:`KNOWN_ANCHOR_TYPES` as the source of truth, not a hard-coded
# list.
KNOWN_ANCHOR_TYPES: set[str] = set(V1_ANCHOR_TYPES)


def register_anchor_type(name: str) -> None:
    """Add ``name`` to the known anchor types. Idempotent. Callers
    extending the registry are responsible for wiring runner-side
    fulfilment of the new kind."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"anchor type name must be a non-empty str, got {name!r}")
    KNOWN_ANCHOR_TYPES.add(name)


# ---------------------------------------------------------------------------
# Anchor.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Anchor:
    """An instrumentation anchor — when to start capturing."""

    anchor_type: str            # one of KNOWN_ANCHOR_TYPES
    params: dict[str, Any]      # type-specific (pc / base / length / access)
    label: str = ""             # human-readable, for alerts/logs

    def __post_init__(self) -> None:
        if self.anchor_type not in KNOWN_ANCHOR_TYPES:
            raise ValueError(
                f"unknown anchor_type {self.anchor_type!r}; "
                f"known={sorted(KNOWN_ANCHOR_TYPES)}; "
                f"use register_anchor_type() to extend"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_type": self.anchor_type,
            "params":      dict(self.params),
            "label":       self.label,
        }


# ---------------------------------------------------------------------------
# Phase boundary.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PhaseBoundary:
    """A located producing phase — output of phase_discovery, input
    of phase_instrument / phase_replay.

    Carries whichever of (instruction-index window, pc range, memory
    region) the discovery step was able to resolve. None means the
    discovery did not pin that dimension; the consumer decides how to
    fall back. Discovery always sets at least one of these so the
    boundary is actionable.
    """

    name: str                                 # caller-supplied phase tag
    entry_idx: int | None = None              # main-trace idx of phase entry
    exit_idx:  int | None = None              # main-trace idx of phase exit
    pc_range:  tuple[int, int] | None = None  # [lo, hi) on the producing thread
    region:    tuple[int, int] | None = None  # [base, length) memory region
    anchor:    Anchor | None = None           # suggested anchor for instrument
    note:      str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name":      self.name,
            "entry_idx": self.entry_idx,
            "exit_idx":  self.exit_idx,
            "pc_range":  list(self.pc_range) if self.pc_range else None,
            "pc_range_hex": (
                [f"0x{self.pc_range[0]:x}", f"0x{self.pc_range[1]:x}"]
                if self.pc_range else None
            ),
            "region":    list(self.region) if self.region else None,
            "region_hex": (
                [f"0x{self.region[0]:x}", self.region[1]]
                if self.region else None
            ),
            "anchor":    self.anchor.to_dict() if self.anchor else None,
            "note":      self.note,
        }


# ---------------------------------------------------------------------------
# ReplayableUnit — phase-as-trace-source.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EntryState:
    """Register + memory state at the anchor hit moment. Sufficient
    to seed a symbolic execution of the unit without context from the
    larger trace."""

    regs:     dict[str, int]
    mem:      dict[int, bytes] = field(default_factory=dict)
    # ``mem`` keys are base addresses, values are concrete bytes the
    # captor read at that address. The captor is free to consolidate
    # adjacent regions before recording. Consumers must not assume
    # any particular chunking.

    def to_dict(self) -> dict[str, Any]:
        return {
            "regs": {k: f"0x{v:x}" for k, v in self.regs.items()},
            "mem": {
                f"0x{addr:x}": bytes_.hex()
                for addr, bytes_ in self.mem.items()
            },
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EntryState":
        regs = {
            k: int(v, 16) if isinstance(v, str) else int(v)
            for k, v in (raw.get("regs") or {}).items()
        }
        mem: dict[int, bytes] = {}
        for k, v in (raw.get("mem") or {}).items():
            addr = int(k, 16) if isinstance(k, str) else int(k)
            data = bytes.fromhex(v) if isinstance(v, str) else bytes(v)
            mem[addr] = data
        return cls(regs=regs, mem=mem)


@dataclass(frozen=True, slots=True)
class ReplayableUnit:
    """A self-contained executable phase captured for replay.

    Identity invariant with the main trace — the instruction stream
    is **the same JSONL format** the main pipeline already reads via
    :class:`engine.runner_client.JsonlTraceReader`. Phase-specific
    metadata (boundary, entry state, memory snapshot) rides in a
    sidecar JSON file. Main-pipeline stages take the unit as a
    resource descriptor; they do not branch on ``source``.
    """

    jsonl_path:    Path
    sidecar_path:  Path
    boundary:      PhaseBoundary
    entry_state:   EntryState
    source:        str = "phase_replay"
    # ``source`` is metadata, not control flow. Stages that care about
    # provenance (e.g. ledger archival) may surface it; stages that
    # care only about the instruction stream should never read it.

    def to_dict(self) -> dict[str, Any]:
        return {
            "jsonl_path":   str(self.jsonl_path),
            "sidecar_path": str(self.sidecar_path),
            "boundary":     self.boundary.to_dict(),
            "entry_state":  self.entry_state.to_dict(),
            "source":       self.source,
        }


# ---------------------------------------------------------------------------
# Sidecar IO — pure JSON.
# ---------------------------------------------------------------------------


def write_sidecar(
    path: str | Path,
    *,
    boundary: PhaseBoundary,
    entry_state: EntryState,
    source: str = "phase_replay",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the sidecar JSON next to a JSONL trace file. The
    structure is stable and consumed by :func:`read_sidecar`."""
    p = Path(path)
    payload: dict[str, Any] = {
        "kind":        "replayable_unit_sidecar",
        "version":     1,
        "source":      source,
        "boundary":    boundary.to_dict(),
        "entry_state": entry_state.to_dict(),
    }
    if extra:
        payload["extra"] = dict(extra)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def read_sidecar(path: str | Path) -> tuple[PhaseBoundary, EntryState, dict[str, Any]]:
    """Inverse of :func:`write_sidecar`. Returns
    ``(boundary, entry_state, extra)``."""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if raw.get("kind") != "replayable_unit_sidecar":
        raise ValueError(
            f"sidecar at {p} has wrong kind={raw.get('kind')!r}"
        )
    b = raw.get("boundary") or {}
    pc_range = b.get("pc_range")
    region   = b.get("region")
    anchor_raw = b.get("anchor")
    anchor = None
    if anchor_raw:
        anchor = Anchor(
            anchor_type=anchor_raw["anchor_type"],
            params=dict(anchor_raw.get("params") or {}),
            label=anchor_raw.get("label", ""),
        )
    boundary = PhaseBoundary(
        name=b.get("name", ""),
        entry_idx=b.get("entry_idx"),
        exit_idx=b.get("exit_idx"),
        pc_range=tuple(pc_range) if pc_range else None,
        region=tuple(region) if region else None,
        anchor=anchor,
        note=b.get("note", ""),
    )
    entry_state = EntryState.from_dict(raw.get("entry_state") or {})
    extra = dict(raw.get("extra") or {})
    return boundary, entry_state, extra


# ---------------------------------------------------------------------------
# Convenience predicates.
# ---------------------------------------------------------------------------


def boundary_covers_pc(boundary: PhaseBoundary, pc: int) -> bool:
    """True iff ``pc`` lies inside ``boundary.pc_range``. None range
    means "unknown" — caller decides how to treat it; this helper
    returns False to avoid false positives."""
    if boundary.pc_range is None:
        return False
    lo, hi = boundary.pc_range
    return lo <= pc < hi


def boundary_covers_addr(boundary: PhaseBoundary, addr: int) -> bool:
    """True iff ``addr`` lies inside ``boundary.region``."""
    if boundary.region is None:
        return False
    base, length = boundary.region
    return base <= addr < base + length


def boundaries_overlap(a: PhaseBoundary, b: PhaseBoundary) -> bool:
    """True iff ``a`` and ``b`` overlap on any resolved dimension.
    Used by phase_discovery to dedupe redundant boundaries."""
    if a.entry_idx is not None and b.entry_idx is not None \
       and a.exit_idx is not None and b.exit_idx is not None:
        if not (a.exit_idx <= b.entry_idx or b.exit_idx <= a.entry_idx):
            return True
    if a.pc_range and b.pc_range:
        if not (a.pc_range[1] <= b.pc_range[0] or b.pc_range[1] <= a.pc_range[0]):
            return True
    if a.region and b.region:
        a_end = a.region[0] + a.region[1]
        b_end = b.region[0] + b.region[1]
        if not (a_end <= b.region[0] or b_end <= a.region[0]):
            return True
    return False


__all__ = [
    "ANCHOR_FUNC_ENTRY",
    "ANCHOR_ADDR_FIRST_EXEC",
    "ANCHOR_MEMREGION_FIRST_ACCESS",
    "V1_ANCHOR_TYPES",
    "KNOWN_ANCHOR_TYPES",
    "register_anchor_type",
    "Anchor",
    "PhaseBoundary",
    "EntryState",
    "ReplayableUnit",
    "write_sidecar",
    "read_sidecar",
    "boundary_covers_pc",
    "boundary_covers_addr",
    "boundaries_overlap",
]
