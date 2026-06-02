"""setup_symex.config section (split from the monolithic module)."""
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


@dataclass(slots=True)
class SetupSymexConfig:
    enabled: bool = True
    # Width (bytes) of the watchpoints locate_boundary requests for the seed /
    # sink ends. 8 is the platform-friendly default the runner may override.
    watch_width_bytes: int = 8
    # Opacity thresholds for pick_mode (see OpacitySignals / estimate_opacity).
    # A slice is "opaque" when EITHER the indirect-branch density crosses
    # ``dispatch_density`` OR the concrete-overwrite rate crosses
    # ``concrete_overwrite_rate``, OR the slice is simply huge.
    dispatch_density: float = 0.15
    concrete_overwrite_rate: float = 0.50
    huge_slice_steps: int = 50_000
    # Per-handler / window parity floor: the emitted transform must match on at
    # least this many INDEPENDENT cross-run vectors before it can be stamped
    # EXACT. 1 is a tautology (≈ verifying the transform with the trace it was
    # derived from); fewer than this floor → BLOCK, never exact.
    parity_min_vectors: int = 3

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "SetupSymexConfig":
        src = env if env is not None else os.environ
        cfg = cls()
        flag = (src.get("UTOV_SETUP_SYMEX") or "").strip().lower()
        if flag in ("off", "0", "false", "no"):
            cfg.enabled = False
        w = src.get("UTOV_SETUP_SYMEX_WATCH_WIDTH")
        if w is not None:
            try:
                cfg.watch_width_bytes = int(w)
            except ValueError:
                pass
        v = src.get("UTOV_SETUP_SYMEX_PARITY_VECTORS")
        if v is not None:
            try:
                cfg.parity_min_vectors = max(1, int(v))
            except ValueError:
                pass
        return cfg


class SetupSymexDisabled(RuntimeError):
    """Raised when the primitive is invoked while UTOV_SETUP_SYMEX is off."""


def _require_enabled(cfg: SetupSymexConfig) -> None:
    if not cfg.enabled:
        raise SetupSymexDisabled(
            "UTOV_SETUP_SYMEX disabled — set-up symex primitive is unavailable"
        )


# ---------------------------------------------------------------------------
# Contract 1 — boundary binding via provenance (not assumed addresses).
# ---------------------------------------------------------------------------


