"""setup_symex.emit_selfcheck section (split from the monolithic module)."""
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
class EmitSelfCheckReport:
    """Did the recovered F reproduce its own trace at the window exit?

    ``status`` is one of:
      * ``"PASS"`` — F evaluated on the trace's concrete seed values equals the
        trace's concrete sink value (necessary condition met);
      * ``"BLOCK"`` — F evaluated on the trace seed != trace sink → symex unsound,
        do NOT emit;
      * ``"INCONCLUSIVE"`` — the check could not be run (no trace facts supplied,
        or F references quantities not on the trace / does not evaluate). Surfaced,
        never silently treated as PASS — the layer above decides.

    ``f_on_trace`` / ``trace_sink`` are the canonical (hex/repr) string forms that
    were compared; ``sink_form`` records whether the sink was a register or a
    memory region so multi-byte / mem sinks are compared in their real shape."""

    status:      str            # PASS | BLOCK | INCONCLUSIVE
    f_on_trace:  str | None
    trace_sink:  str | None
    sink_form:   str = "reg"    # reg | mem
    note:        str = ""

    @property
    def blocked(self) -> bool:
        return self.status == "BLOCK"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind":       "setup_symex_emit_self_check",
            "status":     self.status,
            "f_on_trace": self.f_on_trace,
            "trace_sink": self.trace_sink,
            "sink_form":  self.sink_form,
            "note":       self.note,
        }


def _normalize_sink(value: Any) -> tuple[str, int | bytes] | None:
    """Normalize a trace sink value (int / hex-str / bytes) to (repr, comparable).

    Returns None if it cannot be interpreted (→ INCONCLUSIVE upstream)."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        b = bytes(value)
        return (b.hex(), b)
    if isinstance(value, bool):  # bool is an int subclass — treat as int
        value = int(value)
    if isinstance(value, int):
        return (hex(value), value)
    if isinstance(value, str):
        s = value.strip()
        try:
            iv = int(s, 0) if s.lower().startswith(("0x", "-0x")) else int(s)
            return (hex(iv), iv)
        except ValueError:
            # hex bytes? (even-length hex string)
            try:
                b = bytes.fromhex(s)
                return (b.hex(), b)
            except ValueError:
                return None
    return None


def _eval_emitted_on_seed(
    expr_source: str, inputs: Sequence[str], seed_values: Mapping[str, Any],
) -> tuple[bool, Any, str]:
    """Evaluate the emitted artifact on the trace's concrete seed values.

    Handles both emit forms: a full ``def f(...)`` (exec + call) and a bare
    expression (eval with the seed names bound). Runs in a builtins-stripped
    namespace — the artifact is utov's own emitted code, but we still deny it the
    builtins so a malformed expr cannot reach the filesystem etc. Returns
    ``(ok, value, why)``; ``ok=False`` means the self-check is INCONCLUSIVE (could
    not evaluate / missing seed), with ``why`` carrying the reason."""
    src = str(expr_source).strip()
    if not src:
        return (False, None, "empty expr_source")
    safe_globals: dict[str, Any] = {"__builtins__": {}}
    # A small, side-effect-free toolbox commonly used in recovered transforms.
    safe_globals.update({
        "bytes": bytes, "int": int, "len": len, "range": range,
        "abs": abs, "min": min, "max": max, "pow": pow,
    })
    try:
        if "def f(" in src:
            local_ns: dict[str, Any] = {}
            exec(src, safe_globals, local_ns)  # noqa: S102 — utov's own emitted code
            fn = local_ns.get("f") or safe_globals.get("f")
            if not callable(fn):
                return (False, None, "expr_source defines no callable f()")
            missing = [n for n in inputs if n not in seed_values]
            if missing:
                return (False, None,
                        f"no trace seed value for input(s) {missing}")
            args = [seed_values[n] for n in inputs]
            return (True, fn(*args), "")
        # bare expression: bind the seed names (and the declared inputs) directly
        ns = dict(seed_values)
        missing = [n for n in inputs if n not in ns]
        if missing and any(n in src for n in missing):
            return (False, None, f"no trace seed value for input(s) {missing}")
        return (True, eval(src, safe_globals, ns), "")  # noqa: S307 — utov's own expr
    except Exception as exc:  # eval/exec failure → INCONCLUSIVE, never silent pass
        return (False, None, f"emitted F did not evaluate on the trace seed: {exc!r}")


def check_emit_self_consistency(
    *,
    expr_source: str,
    inputs: Sequence[str],
    seed_values: Mapping[str, Any] | None,
    trace_sink: Any,
    sink_mask: int | None = None,
    sink_form: str = "reg",
) -> EmitSelfCheckReport:
    """Decide whether the recovered F reproduces its own trace at the window exit.

    * ``seed_values`` — the trace's concrete value for each emit input, observed at
      the window entry (the runner reads them off the concolic shadow / the trace).
      These are *trace facts*, not symex computation — reliable even when the
      propagation that produced ``expr_source`` is buggy.
    * ``trace_sink`` — the trace's concrete sink value at the window exit (the
      ground truth F must reproduce). int / hex-str / bytes.
    * ``sink_mask`` — width mask for an integer sink (e.g. ``0xffffffff`` for a
      32-bit reg); ``None`` compares raw / for mem (bytes) sinks.

    No trace facts, or an F that does not evaluate / references off-trace
    quantities → INCONCLUSIVE (surfaced for the layer above), NEVER a silent PASS.
    """
    norm = _normalize_sink(trace_sink)
    if seed_values is None or norm is None:
        return EmitSelfCheckReport(
            status="INCONCLUSIVE", f_on_trace=None,
            trace_sink=(norm[0] if norm else None), sink_form=sink_form,
            note=("cannot self-check: no trace sink value supplied"
                  if norm is None else
                  "cannot self-check: no trace seed values supplied — runner "
                  "must surface the window-entry concrete seed for each input"))
    sink_repr, sink_cmp = norm
    ok, value, why = _eval_emitted_on_seed(expr_source, inputs, seed_values or {})
    if not ok:
        return EmitSelfCheckReport(
            status="INCONCLUSIVE", f_on_trace=None, trace_sink=sink_repr,
            sink_form=sink_form,
            note=f"cannot self-check: {why}")
    # Bring the evaluated F output into the sink's comparable shape.
    if isinstance(value, (bytes, bytearray)):
        f_cmp: int | bytes = bytes(value)
        f_repr = bytes(value).hex()
    elif isinstance(value, bool):
        f_cmp = int(value)
        f_repr = hex(int(value))
    elif isinstance(value, int):
        f_cmp = (value & sink_mask) if sink_mask is not None else value
        f_repr = hex(f_cmp)
    else:
        return EmitSelfCheckReport(
            status="INCONCLUSIVE", f_on_trace=repr(value), trace_sink=sink_repr,
            sink_form=sink_form,
            note=f"cannot self-check: F produced non-numeric/bytes {type(value).__name__}")
    # Mask the trace sink too when it is an integer being compared to an integer.
    if isinstance(sink_cmp, int) and isinstance(f_cmp, int) and sink_mask is not None:
        sink_cmp = sink_cmp & sink_mask
        sink_repr = hex(sink_cmp)
    if f_cmp == sink_cmp:
        return EmitSelfCheckReport(
            status="PASS", f_on_trace=f_repr, trace_sink=sink_repr,
            sink_form=sink_form,
            note="recovered F reproduces its own trace at the window exit "
                 "(necessary, not sufficient — parity still gates generality)")
    return EmitSelfCheckReport(
        status="BLOCK", f_on_trace=f_repr, trace_sink=sink_repr, sink_form=sink_form,
        note=(f"recovered F evaluated on its own trace seed = {f_repr}, but the "
              f"trace's window-exit sink = {sink_repr} → symex is UNSOUND "
              "(shadow / reconcile / sink leak); the emitted F does not even "
              "reproduce the trace it was derived from. NOT emitting."))


# ---------------------------------------------------------------------------
# The guard-railed template — strings the contracts into an ordered plan
# with EXPLICIT agent checkpoints (judgments are surfaced, not auto-decided).
# ---------------------------------------------------------------------------


