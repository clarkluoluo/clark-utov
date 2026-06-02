"""Unified trace observability profile (item ③ of the trace-sufficiency package).

THE single ground-self-check every analysis primitive consults before drawing a
conclusion: "does this trace/window actually carry the dimension my analysis keys
off?" Before this module, the coverage self-check lived in TWO independent places
— ``cohort_diff``'s ``observability_rate`` and ``opaque_staging``'s
``min_regs_write_coverage`` — each computing its own coverage. Per the A8 hard
requirement, both now consume :func:`assess_trace_observability` (or its kernel
:func:`dimension_coverage`); there is no third parallel computation, and the two
keep their existing verdict behaviour byte-for-byte (only the coverage SOURCE is
unified).

Deterministic, zero-LLM. The three dimensions:
  * ``regs_write`` — fraction of window steps with a non-empty regs_write. The DFG
    producer chain (build_dfg / EA backtrace) keys off this; low here means the
    producer graph is largely blind.
  * ``regs_read``  — fraction with a non-empty regs_read.
  * ``mem``        — fraction with at least one mem op (the memory dimension; empty
    when memory lives in an un-merged sidecar — item ① fixes that).

Plus ``sink_captured``: given a sink address/window, did the trace actually OBSERVE
the output (a write or a snapshot covering it)? ``None`` when no sink was supplied.
``False`` = the output is NOT observed → "output not captured, needs re-capture"
(utov self-check; the actual re-capture is the harness's job, not ours).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .types import Instruction, MemSnapshot

__all__ = [
    "DimReadiness",
    "TraceObservability",
    "DEFAULT_THRESHOLDS",
    "has_regs_write",
    "has_write_dim",
    "dimension_coverage",
    "assess_trace_observability",
]


# --- shared per-instruction dimension predicates (the unified kernel) --------
# Both cohort_diff and opaque_staging compute coverage by counting instructions
# that satisfy one of these predicates over their (own) region. Centralising the
# predicate here is the single-source unification: there is no second definition
# of "does this step carry the diffed dimension" anywhere else.

def has_regs_write(ins: "Instruction") -> bool:
    """The regs_write dimension is present on this step (opaque_staging's DFG /
    EA backtrace keys off regs_write producers)."""
    return bool(ins.regs_write)


def has_write_dim(ins: "Instruction") -> bool:
    """The cohort-diff diffed dimension is present: a non-empty regs_write OR a
    memory WRITE. Empty here = a blind position for the value diff."""
    if ins.regs_write:
        return True
    for op in ins.mem:
        if op.rw == "w":
            return True
    return False

# Default sufficiency thresholds per dimension (parameterised — no case constant).
# 0.05 mirrors the existing cohort_diff.min_observability / opaque_staging
# .min_regs_write_coverage defaults, so the unified source reproduces both.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "regs_write": 0.05,
    "regs_read":  0.05,
    "mem":        0.05,
}


@dataclass(frozen=True, slots=True)
class DimReadiness:
    """Whether one dimension is observable enough to trust an analysis that keys
    off it. ``sufficient=False`` carries a precise ``reason`` (wrong dimension /
    incomplete data / output not observed)."""

    dimension:  str
    rate:       float
    sufficient: bool
    reason:     str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"dimension": self.dimension, "rate": round(self.rate, 4),
                "sufficient": self.sufficient, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class TraceObservability:
    """The single observability profile for a trace/window.

    ``overall_sufficient_for(dimension)`` is the gate every analysis calls; the
    per-dimension ``DimReadiness`` gives the precise downgrade reason."""

    n_items:         int
    window:          tuple[int, int] | None
    regs_write_rate: float
    regs_read_rate:  float
    mem_event_rate:  float
    sink_captured:   bool | None
    dims:            tuple[DimReadiness, ...]

    def _dim(self, dimension: str) -> DimReadiness | None:
        for d in self.dims:
            if d.dimension == dimension:
                return d
        return None

    def overall_sufficient_for(self, dimension: str) -> bool:
        d = self._dim(dimension)
        return bool(d and d.sufficient)

    def reason_for(self, dimension: str) -> str:
        d = self._dim(dimension)
        return d.reason if d else f"unknown dimension {dimension!r}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind":            "trace_observability",
            "n_items":         self.n_items,
            "window":          (list(self.window) if self.window is not None else None),
            "regs_write_rate": round(self.regs_write_rate, 4),
            "regs_read_rate":  round(self.regs_read_rate, 4),
            "mem_event_rate":  round(self.mem_event_rate, 4),
            "sink_captured":   self.sink_captured,
            "dims":            [d.to_dict() for d in self.dims],
        }


def _window_items(items: Sequence[Instruction],
                  window: tuple[int, int] | None, window_is_idx: bool) -> list[Instruction]:
    if window is None:
        return list(items)
    lo, hi = int(window[0]), int(window[1])
    if lo > hi:
        lo, hi = hi, lo
    key = (lambda ins: ins.idx) if window_is_idx else (lambda ins: ins.pc)
    return [ins for ins in items if lo <= key(ins) <= hi]


def dimension_coverage(
    items: Sequence[Instruction],
    *,
    window: tuple[int, int] | None = None,
    window_is_idx: bool = True,
) -> tuple[int, float, float, float]:
    """The SHARED coverage kernel both cohort_diff and opaque_staging consume.

    Returns ``(n_window_items, regs_write_rate, regs_read_rate, mem_event_rate)``
    where each rate is the fraction of window instructions with a non-empty
    regs_write / regs_read / mem. ``n==0`` → all rates 0.0 (an empty window cannot
    be trusted; callers decide whether 0 items is "no window" vs "blind")."""
    win = _window_items(items, window, window_is_idx)
    n = len(win)
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    n_rw = sum(1 for ins in win if has_regs_write(ins))
    n_rr = sum(1 for ins in win if ins.regs_read)
    n_mem = sum(1 for ins in win if ins.mem)
    return n, n_rw / n, n_rr / n, n_mem / n


def _sink_captured(
    items: Sequence[Instruction],
    sink_addr: int | None,
    sink_window: tuple[int, int] | None,
    snapshots: Sequence[MemSnapshot],
) -> bool | None:
    """Did the trace OBSERVE the sink region (a write touching it, or a snapshot
    covering it)? None when no sink target was supplied."""
    if sink_addr is None and sink_window is None:
        return None
    if sink_window is not None:
        lo, hi = int(sink_window[0]), int(sink_window[1])
        if lo > hi:
            lo, hi = hi, lo
    elif sink_addr is not None:
        lo = hi = int(sink_addr)
    else:                                   # pragma: no cover - guarded above
        return None
    for ins in items:
        for op in ins.mem:
            if op.rw == "w" and op.size > 0:
                if op.addr <= hi and (op.addr + op.size - 1) >= lo:
                    return True
    for s in snapshots:
        if len(s.data) > 0 and s.addr <= hi and (s.addr + len(s.data) - 1) >= lo:
            return True
    return False


def assess_trace_observability(
    items: Sequence[Instruction],
    *,
    window: tuple[int, int] | None = None,
    window_is_idx: bool = True,
    sink_addr: int | None = None,
    sink_window: tuple[int, int] | None = None,
    snapshots: Sequence[MemSnapshot] = (),
    thresholds: Mapping[str, float] | None = None,
) -> TraceObservability:
    """Compute the unified observability profile for ``items`` (optionally a
    window). Deterministic, zero-LLM. See module docstring for the dimensions."""
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)

    n, rw_rate, rr_rate, mem_rate = dimension_coverage(
        items, window=window, window_is_idx=window_is_idx)

    sink_captured = _sink_captured(items, sink_addr, sink_window, snapshots)

    dims: list[DimReadiness] = []
    for dim, rate in (("regs_write", rw_rate), ("regs_read", rr_rate), ("mem", mem_rate)):
        thr = th.get(dim, 0.05)
        if n == 0:
            dims.append(DimReadiness(
                dim, rate, False,
                "empty window — no instructions to observe this dimension"))
        elif rate < thr:
            dims.append(DimReadiness(
                dim, rate, False,
                f"{dim} basically empty in this trace "
                f"(coverage={rate:.2%} < {thr:.2%}) — wrong dimension fed / "
                f"incomplete data; merge the missing dimension or feed a "
                f"populated trace"))
        else:
            dims.append(DimReadiness(dim, rate, True, ""))

    # Output dimension: when a sink was supplied, sink_captured drives a synthetic
    # readiness so ④ can report "output not observed → needs re-capture" uniformly.
    if sink_captured is not None:
        if sink_captured:
            dims.append(DimReadiness("sink", 1.0, True, ""))
        else:
            dims.append(DimReadiness(
                "sink", 0.0, False,
                "output not observed: no captured write/snapshot covers the sink "
                "region — needs re-capture (utov cannot fabricate the observation)"))

    return TraceObservability(
        n_items=n,
        window=(tuple(window) if window is not None else None),  # type: ignore[arg-type]
        regs_write_rate=rw_rate,
        regs_read_rate=rr_rate,
        mem_event_rate=mem_rate,
        sink_captured=sink_captured,
        dims=tuple(dims),
    )
