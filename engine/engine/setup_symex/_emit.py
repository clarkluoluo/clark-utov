"""setup_symex.emit section (split from the monolithic module)."""
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


@dataclass(frozen=True, slots=True)
class EmitIntent:
    """A request to emit the recovered expression as plain Python.

    The emitted artifact is what phase_5 / the oracle parity-checks. This is an
    intent carrier — the actual AST simplification + codegen is the consumer's
    (or a backend's) job; the primitive records what to emit and the parity
    contract it must satisfy."""

    mode:        SymexMode
    expr_source: str            # the symbolic expression / alias map to render
    inputs:      tuple[str, ...]
    parity_min:  int = 8        # gold inputs the emitted fn must match (hard gate)
    note:        str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode":        self.mode.value,
            "expr_source": self.expr_source,
            "inputs":      list(self.inputs),
            "parity_min":  self.parity_min,
            "note":        self.note,
            "kind":        "setup_symex_emit",
        }


def emit_python(
    *,
    mode: SymexMode,
    expr_source: str,
    inputs: Sequence[str],
    parity_min: int = 8,
) -> EmitIntent:
    """Stage the emit-to-Python step with its parity gate.

    ``parity_min`` is a HARD gate, not advisory: the emitted function must match
    the live oracle on at least this many inputs or the recovery is not closed
    (a stub that passes a critic but is gold 0/8 is a false close)."""
    if not str(expr_source).strip():
        raise ValueError("emit_python needs a non-empty expr_source to render")
    return EmitIntent(
        mode=mode,
        expr_source=str(expr_source),
        inputs=tuple(inputs),
        parity_min=int(parity_min),
        note="emit plain Python; parity_min is a hard gate (no critic-only close)",
    )


# ---------------------------------------------------------------------------
# Per-handler / window parity — MULTI-VECTOR (cross-run) gate.
#
# A single 1/1 parity ≈ verifying a transform with the very trace it was derived
# from — a tautology that proves nothing, yet the old gate stamped it EXACT. The
# VMP cipher case devirt'd handler10 with an INCOMPLETE transform (stopped at
# idx107, dropped the idx107→idx113 x8 update); it passed parity 1/1 and was
# marked exact CLOSED, and the wrong handler only surfaced rounds later via
# compose + gold 0/8 + boundary diff. This gate refuses to call a transform
# EXACT until it matches on >= N INDEPENDENT cross-run vectors, each checked
# against its OWN execution's observed output (determinism — never the deriving
# trace, never another run's output). Fewer than N independent passes, or a
# mixed-execution vector → BLOCK + report, not exact.
# ---------------------------------------------------------------------------


