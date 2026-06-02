"""Watch-first-write primitive + auto-suggestion gate.

Background — the reference target's Round 5 retro: when a value's bytes were
known *at* a landing address but the algorithm that produced them
was not, the agent manually proposed "install a memory watchpoint
on that address and trace the first write". The decision was right
but ad-hoc. This module turns the decision into a primitive and
adds the auto-suggestion.

Two pieces:

  1. :func:`request_watch_first_write` — primitive call. Given a
     concrete address, build a :class:`WatchFirstWriteSpec` that the
     runner side can fulfil. The runner is responsible for the
     memory-watchpoint mechanics; this module owns the contract.
     Returning a ``spec`` (rather than a result) keeps the engine
     side pure and runner-agnostic.

  2. :func:`maybe_suggest_watch` — auto-suggestion. Given a value
     record (typically produced by :mod:`engine.value_provenance`),
     fires when:
       - provenance is ``observed`` (parity-only is not enough), AND
       - ``closed_form`` is NOT verified (no recompute or recompute
         fails), AND
       - the record carries a concrete ``landing_address``.
     The suggestion can either be *advisory* (attached to the
     envelope as ``recommended_followup``) or *triggered* (a
     ``WatchFirstWriteSpec`` immediately added to the envelope so
     the orchestrator can dispatch the watch on the next step).

This is the M3-style auto-followup for "I know the value lands here
but I don't know who put it there".

Independent toggle: ``UTOV_WATCH_FIRST_WRITE=off|0|false|no``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


# Sources whose values are eligible for the watch suggestion. (We
# never suggest watching a closed_form value — we already know how
# it was made.)
DEFAULT_WATCH_ELIGIBLE_SOURCES: tuple[str, ...] = (
    "hook", "dump", "io", "snapshot", "memcpy_capture", "memory_watch",
)


@dataclass(slots=True)
class WatchFirstWriteConfig:
    enabled: bool = True
    # When True, ``maybe_suggest_watch`` emits a triggerable
    # :class:`WatchFirstWriteSpec` alongside the advisory note;
    # otherwise it only emits the advisory and lets the agent decide.
    # Default True ("half-automatic" per the spec).
    auto_trigger: bool = True
    eligible_sources: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_WATCH_ELIGIBLE_SOURCES,
    )
    # Maximum number of bytes the watchpoint should capture per first
    # write. 8 is the platform-friendly default; the runner is free to
    # ignore it if its watchpoint granularity is fixed.
    watch_width_bytes: int = 8

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "WatchFirstWriteConfig":
        src = env if env is not None else os.environ
        cfg = cls()
        flag = (src.get("UTOV_WATCH_FIRST_WRITE") or "").strip().lower()
        if flag in ("off", "0", "false", "no"):
            cfg.enabled = False
        auto = (src.get("UTOV_WATCH_FIRST_WRITE_AUTO_TRIGGER") or "").strip().lower()
        if auto in ("off", "0", "false", "no"):
            cfg.auto_trigger = False
        w = src.get("UTOV_WATCH_FIRST_WRITE_WIDTH")
        if w is not None:
            try:
                cfg.watch_width_bytes = int(w)
            except ValueError:
                pass
        return cfg


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


# Watch directive kinds. ``first_write`` is the original watch-first-write
# semantics (arm a memory watchpoint, catch the FIRST write — no PC gate). The
# two point-watch kinds are PC-gated single-point captures: arm at a specific
# ``pc``; when execution reaches it, resolve ``[base_reg + offset]`` from the
# LIVE register value and capture ``width_bytes`` once, in the given direction.
WATCH_KIND_FIRST_WRITE = "first_write"
WATCH_KIND_READ = "read"
WATCH_KIND_WRITE = "write"
_POINT_WATCH_KINDS = (WATCH_KIND_READ, WATCH_KIND_WRITE)


@dataclass(frozen=True, slots=True)
class WatchFirstWriteSpec:
    """The contract a runner must fulfil. Engine-side users build one
    of these and let the runner do the actual watchpoint install.

    Two mutually exclusive addressing modes:

      * **concrete** (``base_reg is None``) — watch the fixed address
        ``addr``. This is the original behaviour and is byte-for-byte
        unchanged: callers that never touch the reg-relative fields get
        exactly the dict they got before.
      * **reg-relative** (``base_reg is not None``) — watch ``[base_reg
        + offset]``, resolved by the runner AT ARM TIME. Use this when
        the buffer address is not stable across runs (heap / dynamic
        allocation): a concrete address captured on one run is useless
        on the next, but a register-relative expression re-resolves.
        ``addr`` then carries the address observed on the trace that
        derived the spec (diagnostic only — the runner re-resolves from
        the live register, it does NOT trust ``addr``).

    Orthogonal to addressing: the **trigger** (``kind`` + ``pc``):

      * ``kind=first_write`` (``pc is None``, default) — the original
        watch-first-write trigger: arm a memory watchpoint, resume, catch
        the FIRST write. No PC gate. The runner picks the trigger moment.
      * **point-watch** (``kind`` in {``read``,``write``}, ``pc`` set) —
        a PC-conditional single-point capture: arm at ``pc``; when
        execution reaches it, resolve ``[base_reg + offset]`` from the
        LIVE register value at that instant and capture ``width_bytes``
        once, in the ``kind`` direction (read or write). This is the
        precise, low-noise alternative to a wide concrete-range sweep —
        it captures exactly the one access of interest, so it does not
        flood the runner's record cap. Point-watch is meaningful only in
        reg-relative mode (the whole point is "address per live register
        at this PC"); a concrete point-watch is rejected at build time."""

    addr:        int
    width_bytes: int
    value_name:  str           # the value this watch is tracking
    reason:      str           # human-readable purpose
    base_reg:    str | None = None   # reg-relative mode: the pointer register
    offset:      int = 0             # reg-relative mode: byte offset off base_reg
    pc:          int | None = None   # point-watch: arm at this PC (None = first_write)
    kind:        str = WATCH_KIND_FIRST_WRITE  # first_write | read | write

    @property
    def is_reg_relative(self) -> bool:
        return self.base_reg is not None

    @property
    def is_point_watch(self) -> bool:
        """True when this is a PC-gated single-point capture (arm at ``pc``,
        resolve ``[base_reg+offset]`` from the live register, capture once)."""
        return self.pc is not None and self.kind in _POINT_WATCH_KINDS

    @property
    def addr_expr(self) -> str:
        """Human-readable watch target: ``[xN + off]`` (reg-relative) or
        ``0x...`` (concrete)."""
        if self.base_reg is None:
            return f"0x{self.addr:x}"
        if self.offset == 0:
            return f"[{self.base_reg}]"
        sign = "+" if self.offset >= 0 else "-"
        return f"[{self.base_reg} {sign} 0x{abs(self.offset):x}]"

    def to_dict(self) -> dict[str, Any]:
        d = {
            "addr":        self.addr,
            "addr_hex":    f"0x{self.addr:x}",
            "width_bytes": self.width_bytes,
            "value_name":  self.value_name,
            "reason":      self.reason,
            "kind":        "watch_first_write",
        }
        # Invariant 7: the concrete first-write path (base_reg=None, pc=None)
        # emits EXACTLY the original dict — no new keys — so today's behaviour is
        # byte-identical. The reg-relative addressing fields appear ONLY when this
        # is a reg-relative spec; the point-watch trigger fields ONLY when armed
        # at a PC.
        if self.is_reg_relative:
            # Reg-relative target: the runner resolves [base_reg + offset]
            # at arm time. ``addr`` above is the observed (diagnostic-only)
            # address from the deriving run, not a target to watch directly.
            d["addressing"] = "reg_relative"
            d["addr_expr"] = self.addr_expr
            d["base_reg"] = self.base_reg
            d["offset"] = self.offset
        if self.is_point_watch:
            # PC-gated single-point capture. Overrides the coarse
            # ``kind=watch_first_write`` with the precise trigger contract so
            # the runner arms at ``pc`` and captures one access in ``watch_kind``
            # direction, instead of catching the first write.
            d["kind"] = "point_watch"
            d["watch_kind"] = self.kind            # read | write
            d["pc"] = self.pc
            d["pc_hex"] = f"0x{self.pc:x}"
        return d


@dataclass(frozen=True, slots=True)
class WatchFirstWriteResult:
    """The shape the runner sends back after fulfilling a spec.
    Engine-side this is a passive carrier; we do not synthesise
    results, only consume them."""

    spec:           WatchFirstWriteSpec
    first_write_pc: int | None
    source_bytes:   bytes | None
    note:           str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec":           self.spec.to_dict(),
            "first_write_pc": self.first_write_pc,
            "first_write_pc_hex":
                f"0x{self.first_write_pc:x}" if self.first_write_pc is not None else None,
            "source_bytes":   self.source_bytes.hex() if self.source_bytes else None,
            "note":           self.note,
        }


@dataclass(frozen=True, slots=True)
class WatchSuggestion:
    """Advisory + (optional) triggered watch attached to the envelope.

    The wrapper writes one of these onto the envelope's ``watch_suggestion``
    field; the agent / orchestrator decides whether to call
    :func:`request_watch_first_write` on the spec immediately, or
    treat it as a note."""

    value_name: str
    spec:       WatchFirstWriteSpec | None   # None when auto_trigger=False
    advisory:   str

    def to_dict(self) -> dict[str, Any]:
        return {
            "value_name": self.value_name,
            "spec":       self.spec.to_dict() if self.spec else None,
            "advisory":   self.advisory,
        }


# ---------------------------------------------------------------------------
# Primitive entry point
# ---------------------------------------------------------------------------


def request_watch_first_write(
    addr: int,
    value_name: str,
    *,
    reason: str = "trace producer of observed value",
    cfg: WatchFirstWriteConfig | None = None,
    base_reg: str | None = None,
    offset: int = 0,
    width_bytes: int | None = None,
    pc: int | None = None,
    kind: str = WATCH_KIND_FIRST_WRITE,
) -> WatchFirstWriteSpec:
    """Build a runner-fulfillable watchpoint spec.

    The runner side is contractually expected to:
      * install a memory watchpoint at the target for ``width_bytes``,
      * resume the target,
      * at the first write, capture the writing PC and the bytes,
      * return a :class:`WatchFirstWriteResult`.

    Two addressing modes (mutually exclusive):
      * **concrete** (``base_reg is None``, default) — watch the fixed
        ``addr``. Unchanged from the original contract.
      * **reg-relative** (``base_reg`` given) — watch ``[base_reg +
        offset]``, which the runner resolves at arm time. Use when the
        target buffer address is not stable across runs. ``addr`` is then
        the address observed on the deriving run (diagnostic only).

    Two triggers (orthogonal to addressing, via ``kind`` + ``pc``):
      * ``kind=first_write`` (``pc=None``, default) — the original
        first-write trigger. Unchanged.
      * **point-watch** (``kind`` in {``read``,``write``}, ``pc`` set) —
        arm at ``pc``; on reaching it, resolve ``[base_reg+offset]`` from
        the live register and capture ``width_bytes`` once in the ``kind``
        direction. Requires reg-relative mode (see :func:`request_point_watch`,
        the preferred entry for point-watches).

    The engine is deliberately runner-agnostic. Tests and callers
    treat this function as the public API; runner shims wire the
    spec through their own watchpoint plumbing.
    """
    if not isinstance(addr, int) or addr < 0:
        raise ValueError(f"addr must be a non-negative int, got {addr!r}")
    if base_reg is not None:
        if not isinstance(base_reg, str) or not base_reg.strip():
            raise ValueError(f"base_reg must be a non-empty register name, got {base_reg!r}")
        if not isinstance(offset, int):
            raise ValueError(f"offset must be an int, got {offset!r}")
    if kind not in (WATCH_KIND_FIRST_WRITE, WATCH_KIND_READ, WATCH_KIND_WRITE):
        raise ValueError(
            f"kind must be one of first_write|read|write, got {kind!r}")
    if pc is not None:
        if not isinstance(pc, int) or pc < 0:
            raise ValueError(f"pc must be a non-negative int, got {pc!r}")
        if kind == WATCH_KIND_FIRST_WRITE:
            raise ValueError(
                "a PC-gated point-watch needs kind=read|write (kind=first_write "
                "has no PC gate — it catches the first write)")
        if base_reg is None:
            # A concrete-address point-watch is meaningless: the whole point of a
            # PC gate is "compute the address from the LIVE register at this PC".
            # If the address were stable you would not need a PC gate at all.
            raise ValueError(
                "a point-watch (pc set) must be reg-relative (give base_reg): the "
                "PC gate exists to resolve [base_reg+offset] from the live register "
                "at that instant; a concrete point-watch has no purpose")
    elif kind in (WATCH_KIND_READ, WATCH_KIND_WRITE):
        raise ValueError(
            f"kind={kind} is a point-watch direction but no pc was given — a "
            "point-watch must be armed at a PC")
    cfg = cfg or WatchFirstWriteConfig.from_env()
    if not cfg.enabled:
        raise RuntimeError("UTOV_WATCH_FIRST_WRITE disabled — primitive is unavailable")
    return WatchFirstWriteSpec(
        addr=addr,
        width_bytes=cfg.watch_width_bytes if width_bytes is None else int(width_bytes),
        value_name=value_name,
        reason=reason,
        base_reg=base_reg,
        offset=offset,
        pc=pc,
        kind=kind,
    )


def request_point_watch(
    pc: int,
    base_reg: str,
    offset: int,
    width_bytes: int,
    value_name: str,
    *,
    kind: str = WATCH_KIND_READ,
    addr: int = 0,
    reason: str = "PC-gated reg-relative single-point capture",
    cfg: WatchFirstWriteConfig | None = None,
) -> WatchFirstWriteSpec:
    """Build a PC-conditional reg-relative single-point watch spec.

    Semantics (the runner contract): arm at ``pc``; when execution reaches that
    instruction, resolve ``[base_reg + offset]`` from the LIVE register value at
    that instant and capture ``width_bytes`` once, in the ``kind`` direction
    (``read`` or ``write``). This is the precise, low-noise alternative to a wide
    concrete-range sweep — it captures exactly the one access of interest and so
    never floods the runner's record cap.

    This reuses :func:`request_watch_first_write`'s arm-time reg-relative
    resolution (it is the SAME spec / SAME runner mechanism) — it only adds the
    PC gate + capture direction. ``addr`` is diagnostic-only (the address
    observed on the deriving run, if any); the runner re-resolves from the live
    register and does NOT trust it.
    """
    if kind not in (WATCH_KIND_READ, WATCH_KIND_WRITE):
        raise ValueError(
            f"point-watch kind must be read|write, got {kind!r}")
    return request_watch_first_write(
        addr, value_name, reason=reason, cfg=cfg,
        base_reg=base_reg, offset=offset, width_bytes=width_bytes,
        pc=pc, kind=kind,
    )


# ---------------------------------------------------------------------------
# Auto-suggestion gate
# ---------------------------------------------------------------------------


def maybe_suggest_watch(
    record: dict[str, Any],
    *,
    cfg: WatchFirstWriteConfig | None = None,
) -> WatchSuggestion | None:
    """Inspect a value record. Return a :class:`WatchSuggestion` when
    the conditions for the M3-style auto-suggestion are met; ``None``
    otherwise.

    Conditions (all must hold):
      * ``cfg.enabled``
      * source is in ``cfg.eligible_sources`` (observed bytes), OR
        the record's ``provenance`` already says ``observed`` /
        ``hybrid``
      * a verified closed-form recompute is NOT available
      * ``landing_address`` is a concrete int (the address the
        observed bytes live at)
    """
    cfg = cfg or WatchFirstWriteConfig.from_env()
    if not cfg.enabled or not isinstance(record, dict):
        return None

    source = str(record.get("source") or "").strip().lower()
    provenance = str(record.get("provenance") or "").strip().lower()
    fn_present = bool(record.get("recompute_fn_present"))
    fn_matches = bool(record.get("recompute_matches_measured"))
    addr = record.get("landing_address")
    if addr is None:
        addr = record.get("addr")
    name = str(record.get("value_name") or record.get("name") or "<unnamed>")

    if not isinstance(addr, int):
        return None
    if fn_present and fn_matches:
        # already closed_form — no need to chase the writer.
        return None
    is_observed = (
        source in cfg.eligible_sources
        or provenance in ("observed", "hybrid")
    )
    if not is_observed:
        return None

    advisory = (
        f"value {name} is provenance=observed at addr=0x{addr:x} with no "
        f"verified closed-form recompute — recommend `watch_first_write(0x{addr:x})` "
        f"to capture the producing PC and source bytes (first writer = "
        f"the algorithm we still don't know)."
    )
    spec = (
        request_watch_first_write(addr, name, reason=advisory, cfg=cfg)
        if cfg.auto_trigger else None
    )
    return WatchSuggestion(value_name=name, spec=spec, advisory=advisory)


def suggest_watches_in_params(
    params: dict[str, Any] | None,
    *,
    cfg: WatchFirstWriteConfig | None = None,
) -> list[WatchSuggestion]:
    """Walk ``params`` looking for value records and accumulate
    suggestions. The wrapper attaches the resulting list to the
    envelope so agents see them on the same call that emitted the
    observed value."""
    cfg = cfg or WatchFirstWriteConfig.from_env()
    if not cfg.enabled or params is None:
        return []
    out: list[WatchSuggestion] = []
    _walk(params, cfg, out)
    return out


def _walk(
    node: Any,
    cfg: WatchFirstWriteConfig,
    out: list[WatchSuggestion],
    *,
    depth: int = 5,
) -> None:
    if depth <= 0 or node is None:
        return
    if isinstance(node, dict):
        looks_like_value = (
            ("value_name" in node or "name" in node)
            and (
                "source" in node
                or "provenance" in node
                or "landing_address" in node
                or "addr" in node
            )
        )
        if looks_like_value:
            s = maybe_suggest_watch(node, cfg=cfg)
            if s is not None:
                out.append(s)
        for v in node.values():
            _walk(v, cfg, out, depth=depth - 1)
    elif isinstance(node, list):
        for v in node:
            _walk(v, cfg, out, depth=depth - 1)


def render_watch_suggestion_alert(suggestions: list[WatchSuggestion]) -> str | None:
    if not suggestions:
        return None
    parts = []
    for s in suggestions:
        if s.spec is not None:
            parts.append(
                f"{s.value_name}@0x{s.spec.addr:x} (auto-trigger ready)"
            )
        else:
            parts.append(f"{s.value_name} (advisory only)")
    return "[WATCH-FIRST-WRITE] suggested: " + "; ".join(parts)
