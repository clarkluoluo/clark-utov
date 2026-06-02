"""setup_symex.mode section (split from the monolithic module)."""
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
from ._config import SetupSymexConfig


class SymexMode(str, enum.Enum):
    FORWARD = "forward"                 # per-instruction symbolic propagation
    BACKWARD_ALIAS = "backward_alias"   # materialization-reverse alias + diff


@dataclass(frozen=True, slots=True)
class OpacitySignals:
    """The measurable signals that decide path opacity for :func:`pick_mode`.

    ``sym_propagated`` — did forward symbolic propagation actually reach the
    sink? (False is the precondition for switching.) The rest measure WHY the
    path resists forward symex: an indirect-branch dispatch, a huge slice, or a
    high rate of concrete overwrites killing the symbols."""

    sym_propagated:          bool
    slice_steps:             int = 0
    indirect_branch_density: float = 0.0   # indirect branches / total steps
    concrete_overwrite_rate: float = 0.0   # sym regs overwritten concrete / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "sym_propagated":          self.sym_propagated,
            "slice_steps":             self.slice_steps,
            "indirect_branch_density": round(self.indirect_branch_density, 4),
            "concrete_overwrite_rate": round(self.concrete_overwrite_rate, 4),
        }


@dataclass(frozen=True, slots=True)
class ModeDecision:
    mode:    SymexMode
    reason:  str
    signals: OpacitySignals

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode":    self.mode.value,
            "reason":  self.reason,
            "signals": self.signals.to_dict(),
            "kind":    "setup_symex_mode",
        }


def estimate_opacity(
    items: Iterable[Instruction],
    *,
    sym_propagated: bool,
) -> OpacitySignals:
    """Derive :class:`OpacitySignals` from a trace slice.

    Uses :func:`engine.dataflow.classify_semop` to count indirect-branch density
    (the dispatch signature). ``concrete_overwrite_rate`` is not derivable from
    mnemonics alone, so it is left at 0 here; callers who track symbol liveness
    pass it into :func:`pick_mode` via a hand-built :class:`OpacitySignals`."""
    items = list(items)
    n = len(items)
    indirect = 0
    for ins in items:
        first = ins.mnemonic.split(None, 1)
        mnem = first[0] if first else ""
        if mnem in ("br", "blr"):       # register-indirect branch = dispatch tell
            indirect += 1
    density = (indirect / n) if n else 0.0
    return OpacitySignals(
        sym_propagated=sym_propagated,
        slice_steps=n,
        indirect_branch_density=density,
    )


def pick_mode(
    signals: OpacitySignals,
    *,
    cfg: SetupSymexConfig | None = None,
) -> ModeDecision:
    """Encode the forward→backward switch criterion.

    Rule (from the case): if forward symbolic propagation did NOT reach the sink
    AND the path is opaque (dispatch / huge / high concrete-overwrite), do not
    keep pushing forward — switch to backward alias materialization. A
    transparent path, or one where symbols already propagated, stays forward."""
    cfg = cfg or SetupSymexConfig.from_env()
    if signals.sym_propagated:
        return ModeDecision(
            mode=SymexMode.FORWARD, signals=signals,
            reason="symbols propagated to the sink — forward symex is working; "
                   "stay forward",
        )
    opaque_reasons = []
    if signals.indirect_branch_density >= cfg.dispatch_density:
        opaque_reasons.append(
            f"indirect-branch density {signals.indirect_branch_density:.2f} "
            f">= {cfg.dispatch_density} (dispatch)"
        )
    if signals.concrete_overwrite_rate >= cfg.concrete_overwrite_rate:
        opaque_reasons.append(
            f"concrete-overwrite rate {signals.concrete_overwrite_rate:.2f} "
            f">= {cfg.concrete_overwrite_rate} (symbols killed)"
        )
    if signals.slice_steps >= cfg.huge_slice_steps:
        opaque_reasons.append(
            f"slice {signals.slice_steps} steps >= {cfg.huge_slice_steps} (huge)"
        )
    if opaque_reasons:
        return ModeDecision(
            mode=SymexMode.BACKWARD_ALIAS, signals=signals,
            reason="forward symex did not reach the sink and the path is opaque ["
                   + "; ".join(opaque_reasons)
                   + "] — the sink is a materialization of the seed's alias; "
                   "backtrace it instead of forcing forward",
        )
    return ModeDecision(
        mode=SymexMode.FORWARD, signals=signals,
        reason="forward did not reach the sink yet but the path is not opaque — "
               "keep trying forward (no opacity signal crossed threshold)",
    )


# ---------------------------------------------------------------------------
# Emit — AST-simplify to Python (the parity-checkable artifact).
# ---------------------------------------------------------------------------


