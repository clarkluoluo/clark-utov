"""setup_symex.boundary section (split from the monolithic module)."""
from __future__ import annotations


import enum
import os
import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..dataflow import classify_semop
from ..types import Instruction, MemSnapshot
from ..watch_first_write import (
    WatchFirstWriteConfig,
    WatchFirstWriteSpec,
    request_watch_first_write,
)
from ._config import SetupSymexConfig, _require_enabled


class BoundaryRole(str, enum.Enum):
    SEED = "seed"   # the symbolic INPUT end (where SymVars enter)
    SINK = "sink"   # the OUTPUT end (where the recovered value materializes)


# How a boundary end's concrete address was established. ``ASSUMED`` is the
# anti-pattern the case hit three times — it is rejected by ``bind_boundary``.
LOCATED_WATCH = "watch_first_write"   # found the real writer/reader via watchpoint
LOCATED_DFG = "dfg"                   # found via reg_deps / mem_deps backtrace
LOCATED_SINK_VALIDATION = "sink_validation"  # oracle_sink located the real sink
LOCATED_ASSUMED = "assumed"           # hand-typed / guessed — FORBIDDEN

_PROVENANCE_LOCATORS = frozenset(
    {LOCATED_WATCH, LOCATED_DFG, LOCATED_SINK_VALIDATION}
)


@dataclass(frozen=True, slots=True)
class BoundaryEnd:
    """One end of the symbolic boundary, with its provenance receipt.

    ``located_via`` MUST be one of the provenance locators — binding to a
    hand-typed address (``assumed``) is the structural error this contract
    exists to stop, so :func:`bind_boundary` rejects it."""

    role:        BoundaryRole
    addr:        int
    located_via: str
    value_name:  str = ""
    note:        str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "role":        self.role.value,
            "addr":        self.addr,
            "addr_hex":    f"0x{self.addr:x}",
            "located_via": self.located_via,
            "value_name":  self.value_name,
            "note":        self.note,
        }


class BoundaryNotProvenanceLocated(ValueError):
    """The boundary end was bound to an assumed / un-located address."""


def bind_boundary(end: BoundaryEnd) -> BoundaryEnd:
    """Guardrail: accept a boundary end ONLY if it was provenance-located.

    This is the contract's teeth. The case bound the sink to a carrier (should
    have been the in-window landing), bound an indirection to the pointer
    (should have been the pointed bytes), and bound the input to an assumed
    address (the input was already in a register before the window). Every one
    was an ``assumed`` bind. The primitive refuses them so the agent is pushed
    back to ``locate_boundary``."""
    if end.located_via not in _PROVENANCE_LOCATORS:
        raise BoundaryNotProvenanceLocated(
            f"boundary {end.role.value} at 0x{end.addr:x} is located_via="
            f"{end.located_via!r}; a set-up symex boundary must be located via "
            f"provenance ({sorted(_PROVENANCE_LOCATORS)}), never a hand-typed "
            f"address — call locate_boundary() and bind to the real read/write point"
        )
    return end


@dataclass(frozen=True, slots=True)
class BoundaryPlan:
    """The runner-fulfillable plan that LOCATES both boundary ends.

    Carries watchpoint specs (seed + sink) the runner installs to find the real
    read/write points. The agent does not type addresses; it dispatches these
    specs, then binds the returned PCs/addresses via :func:`bind_boundary`."""

    seed_watch: WatchFirstWriteSpec
    sink_watch: WatchFirstWriteSpec
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_watch": self.seed_watch.to_dict(),
            "sink_watch": self.sink_watch.to_dict(),
            "note":       self.note,
            "kind":       "setup_symex_boundary_plan",
        }


def locate_boundary(
    *,
    seed_hint_addr: int,
    sink_hint_addr: int,
    seed_name: str = "symex_seed",
    sink_name: str = "symex_sink",
    cfg: SetupSymexConfig | None = None,
) -> BoundaryPlan:
    """Build the provenance plan that finds the REAL seed and sink addresses.

    ``*_hint_addr`` are starting points (e.g. a carrier address, the buffer the
    runner first reported) — NOT the addresses symex will bind to. The returned
    watchpoints find who really writes the sink and who really feeds the seed;
    the agent binds those located points via :func:`bind_boundary`.

    Wires :func:`engine.watch_first_write.request_watch_first_write` for both
    ends. Pair with :func:`engine.oracle_sink.validate_sink` when the expected
    output bytes are known (that locator is ``sink_validation``)."""
    cfg = cfg or SetupSymexConfig.from_env()
    _require_enabled(cfg)
    wcfg = WatchFirstWriteConfig(watch_width_bytes=cfg.watch_width_bytes)
    seed_watch = request_watch_first_write(
        seed_hint_addr, seed_name,
        reason="set-up symex: locate the REAL feeder of the symbolic seed "
               "(the input is often already in a register before the window)",
        cfg=wcfg,
    )
    sink_watch = request_watch_first_write(
        sink_hint_addr, sink_name,
        reason="set-up symex: locate the REAL writer of the sink (bind the "
               "in-window landing, not the carrier / pointer)",
        cfg=wcfg,
    )
    return BoundaryPlan(
        seed_watch=seed_watch,
        sink_watch=sink_watch,
        note="locate both ends via provenance before binding; do NOT type "
             "addresses. Sink bytes known? also run oracle_sink.validate_sink.",
    )


# ---------------------------------------------------------------------------
# Contract 2 — entry-state completeness (full reg_file + pointed buffers).
# ---------------------------------------------------------------------------


