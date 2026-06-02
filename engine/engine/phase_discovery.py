"""Phase discovery — locate a producing phase for a key value.

Given a key value (typically identified by its landing address in
memory), walk backward through the available data sources looking
for the producer:

  * if the producer PC lies inside the current trace window, the value
    is *not* phase-isolated — discovery returns ``None`` (no phase to
    promote);
  * if the producer is the result of a memcpy/load whose own source
    address has no in-window writer, *or* the producer PC sits outside
    the current window, the producing computation lives in a phase
    we don't have full visibility on — discovery returns a
    :class:`engine.phase.PhaseBoundary` describing what we know
    (region, pc range if we have it) and a suggested anchor
    (memregion_first_access by default).

Pluggable data sources. ``discover_phase`` takes a
:class:`DiscoveryDataSource`. The default implementation
(:class:`InMemoryDataSource`) walks the loaded trace window plus an
optional ledger snapshot — option 1 from the design discussion. The
interface is shaped so that follow-up implementations
(:class:`RunnerProbingDataSource`, :class:`ProvenanceTagDataSource`)
can plug in without touching the discovery algorithm. The auto-gate
on the wrapper instantiates the default; agents calling the RPC
directly can hand in a richer one.

Independent toggle: ``UTOV_PHASE_DISCOVERY=off|0|false|no``.
"""

from __future__ import annotations

import os
import abc
from dataclasses import dataclass, field
from typing import Any, Iterable

from .phase import (
    Anchor,
    PhaseBoundary,
    ANCHOR_MEMREGION_FIRST_ACCESS,
    ANCHOR_ADDR_FIRST_EXEC,
)


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PhaseDiscoveryConfig:
    enabled: bool = True
    # Default memory-region width to mark around a key value when the
    # caller doesn't supply one. 32 bytes is a common materialization
    # block size; callers should override when they know better.
    default_region_width: int = 32
    # Maximum chain depth — how many producer hops to walk before
    # giving up. Trace windows are big, this avoids quadratic blowup
    # if a value is reached by a long chain of memcpys.
    max_chain_depth: int = 8

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "PhaseDiscoveryConfig":
        src = env if env is not None else os.environ
        cfg = cls()
        flag = (src.get("UTOV_PHASE_DISCOVERY") or "").strip().lower()
        if flag in ("off", "0", "false", "no"):
            cfg.enabled = False
        w = src.get("UTOV_PHASE_DISCOVERY_REGION_WIDTH")
        if w is not None:
            try:
                cfg.default_region_width = int(w)
            except ValueError:
                pass
        d = src.get("UTOV_PHASE_DISCOVERY_MAX_CHAIN")
        if d is not None:
            try:
                cfg.max_chain_depth = int(d)
            except ValueError:
                pass
        return cfg


# ---------------------------------------------------------------------------
# Data-source interface — pluggable.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WriterHit:
    """A producer record: the trace step that wrote ``size`` bytes at
    ``addr``. ``pc`` is the writer's PC. ``idx`` is the trace index
    (None if the data source can't pin one)."""

    pc:    int
    addr:  int
    size:  int
    value: int | None = None
    idx:   int | None = None


@dataclass(frozen=True, slots=True)
class ProducerEdge:
    """Edge in the producer chain. Either:
      * ``writer`` — the direct writer at ``dst_addr`` (terminates chain), OR
      * ``copy``  — the load/memcpy that brought bytes from ``src_addr``
        to ``dst_addr``; chain continues from ``src_addr``.
    """

    kind:     str             # "writer" | "copy"
    dst_addr: int
    src_addr: int | None = None
    writer:   WriterHit | None = None
    note:     str = ""


class DiscoveryDataSource(abc.ABC):
    """Read-only access to the producer information available for the
    current call. Implementations may consult the loaded trace, a
    sqlite ledger, the runner's RPC, value-provenance tags, etc.

    The contract is intentionally narrow so a default in-memory
    impl can plug, an RPC-probing impl can plug, and a hybrid impl
    can compose both — none of them should need to know about the
    discovery walker's loop.
    """

    @abc.abstractmethod
    def window_pc_range(self) -> tuple[int, int] | None:
        """Return ``(lo_pc, hi_pc)`` of the currently loaded trace
        window, or ``None`` if no window is loaded. Used to classify
        producer PCs as in-window vs out-of-window."""

    @abc.abstractmethod
    def window_idx_range(self) -> tuple[int, int] | None:
        """Return ``(lo_idx, hi_idx)`` of the currently loaded trace
        window, or ``None``."""

    @abc.abstractmethod
    def latest_writer(self, addr: int, *, size: int = 1) -> WriterHit | None:
        """Return the most recent writer that touched ``addr``, or
        ``None`` if no writer to that address exists in the source.

        ``size`` is a hint — the impl may return any writer that
        overlaps ``[addr, addr+size)``; the caller uses the writer's
        ``addr`` / ``size`` for further chain walking."""

    @abc.abstractmethod
    def producer_edge(self, addr: int, *, size: int = 1) -> ProducerEdge | None:
        """Return the most recent producer edge for ``addr``. Either:
          * a ``writer`` edge whose writer PC is in-window, OR
          * a ``copy`` edge whose source address has a (recursive)
            in-window writer.

        ``None`` means "no information at all" — the address was
        never written to in any data source the impl knows about."""


# ---------------------------------------------------------------------------
# Default impl — in-memory walk over an instruction iterable.
# ---------------------------------------------------------------------------


# Mnemonics we treat as "copy" semantics (transfers bytes between
# memory locations or between a register and memory). Conservative:
# the walker only needs to know whether a write at ``dst_addr`` was
# *sourced* from another address; if we don't recognise the
# mnemonic we treat the writer as terminal (a direct writer), which
# is the safer answer for discovery (no false phase-isolation claim).
_COPY_MNEMS: frozenset[str] = frozenset({
    "ldr", "ldrb", "ldrh", "ldrsh", "ldrsw", "ldp",
    "ldur", "ldurh", "ldurb",
    "str", "strb", "strh", "stp", "stur", "sturh", "sturb",
})


@dataclass(slots=True)
class InMemoryDataSource(DiscoveryDataSource):
    """Walks an in-memory iterable of :class:`engine.types.Instruction`-shaped
    records. The exact class isn't required — anything with
    ``idx``, ``pc``, ``mnemonic``, ``mem`` (iterable of MemOp-like
    objects with ``rw``, ``addr``, ``size``, ``val``) works. That
    keeps this module easy to test with synthetic traces.
    """

    instructions: list[Any] = field(default_factory=list)
    # Optional ledger probe — callable (addr, size) -> WriterHit | None.
    # When provided, used as a fallback after the in-memory walk
    # exhausts; allows option-3 ("trace + ledger") to plug without
    # subclassing.
    ledger_probe: Any = None

    def window_pc_range(self) -> tuple[int, int] | None:
        if not self.instructions:
            return None
        pcs = [int(getattr(i, "pc")) for i in self.instructions]
        return (min(pcs), max(pcs) + 1)

    def window_idx_range(self) -> tuple[int, int] | None:
        if not self.instructions:
            return None
        idxs = [int(getattr(i, "idx")) for i in self.instructions]
        return (min(idxs), max(idxs) + 1)

    def latest_writer(self, addr: int, *, size: int = 1) -> WriterHit | None:
        addr_lo = addr
        addr_hi = addr + max(size, 1)
        for ins in reversed(self.instructions):
            for m in getattr(ins, "mem", ()) or ():
                if getattr(m, "rw", "") != "w":
                    continue
                m_addr = int(getattr(m, "addr"))
                m_size = int(getattr(m, "size", 1))
                if m_addr < addr_hi and (m_addr + m_size) > addr_lo:
                    return WriterHit(
                        pc=int(getattr(ins, "pc")),
                        addr=m_addr,
                        size=m_size,
                        value=int(getattr(m, "val", 0)) if hasattr(m, "val") else None,
                        idx=int(getattr(ins, "idx", -1)),
                    )
        if callable(self.ledger_probe):
            try:
                w = self.ledger_probe(addr, size)
                if isinstance(w, WriterHit):
                    return w
            except Exception:
                pass
        return None

    def producer_edge(self, addr: int, *, size: int = 1) -> ProducerEdge | None:
        writer = self.latest_writer(addr, size=size)
        if writer is None:
            return None
        # Try to classify: was the write sourced from another address?
        # We look at the writer's instruction record and check whether
        # it also had a read mem-op (typical of a load-store pattern).
        for ins in self.instructions:
            if int(getattr(ins, "idx", -1)) != (writer.idx or -1):
                continue
            mnem = (getattr(ins, "mnemonic", "") or "").split(maxsplit=1)
            head = (mnem[0] if mnem else "").lower()
            reads = [
                m for m in (getattr(ins, "mem", ()) or ())
                if getattr(m, "rw", "") == "r"
            ]
            if head in _COPY_MNEMS and reads:
                src = reads[0]
                return ProducerEdge(
                    kind="copy",
                    dst_addr=writer.addr,
                    src_addr=int(getattr(src, "addr")),
                    writer=writer,
                    note=f"{head} src=0x{int(getattr(src, 'addr')):x}",
                )
            break
        return ProducerEdge(kind="writer", dst_addr=writer.addr, writer=writer)


# ---------------------------------------------------------------------------
# Result.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PhaseDiscoveryResult:
    """Output of :func:`discover_phase`.

    ``boundary`` is non-None iff the discovery decided the producing
    computation lives outside the current trace window. ``chain`` is
    the walk that led to that decision (or to the in-window
    termination) — surfaced for explainability.
    """

    value_addr: int
    boundary:   PhaseBoundary | None
    crosses_out: bool
    chain:      tuple[ProducerEdge, ...] = ()
    reason:     str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "value_addr":     self.value_addr,
            "value_addr_hex": f"0x{self.value_addr:x}",
            "boundary":       self.boundary.to_dict() if self.boundary else None,
            "crosses_out":    self.crosses_out,
            "chain": [
                {
                    "kind":     e.kind,
                    "dst_addr": e.dst_addr,
                    "dst_addr_hex": f"0x{e.dst_addr:x}",
                    "src_addr": e.src_addr,
                    "src_addr_hex":
                        f"0x{e.src_addr:x}" if e.src_addr is not None else None,
                    "writer_pc": e.writer.pc if e.writer else None,
                    "writer_pc_hex":
                        f"0x{e.writer.pc:x}" if e.writer else None,
                    "note": e.note,
                }
                for e in self.chain
            ],
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Discovery entry point.
# ---------------------------------------------------------------------------


def discover_phase(
    value_addr: int,
    source: DiscoveryDataSource,
    *,
    value_name: str = "",
    phase_name: str = "",
    value_size: int = 1,
    cfg: PhaseDiscoveryConfig | None = None,
) -> PhaseDiscoveryResult:
    """Locate the phase that produced ``value_addr``.

    Algorithm.
      1. Ask the data source for the producer edge at ``value_addr``.
      2. If none → boundary is unknown; mark ``crosses_out=True``
         with a memregion anchor on the value's address (best we can
         do; the discovery saw no writer at all, which is the
         strongest "crosses out" signal).
      3. If a ``writer`` edge whose PC is in-window → in-window
         producer; no boundary.
      4. If a ``writer`` edge whose PC is out-of-window → boundary
         with pc_range hint + region.
      5. If a ``copy`` edge — recursive walk on ``src_addr``, bounded
         by ``cfg.max_chain_depth``. The chain terminates as in (2)–(4).
    """
    cfg = cfg or PhaseDiscoveryConfig.from_env()
    if not cfg.enabled:
        return PhaseDiscoveryResult(
            value_addr=value_addr,
            boundary=None,
            crosses_out=False,
            reason="phase_discovery disabled",
        )

    window_pc = source.window_pc_range()

    chain: list[ProducerEdge] = []
    cursor_addr = value_addr
    cursor_size = max(value_size, 1)

    for hop in range(cfg.max_chain_depth):
        edge = source.producer_edge(cursor_addr, size=cursor_size)
        if edge is None:
            # No writer at all → strongest crosses-out signal. Use
            # ``cursor_addr`` (the upstream end of the chain) so a
            # memcpy chain points the runner at the original producer
            # region, not the in-window destination buffer.
            region = (cursor_addr, max(cfg.default_region_width, cursor_size))
            anchor = Anchor(
                anchor_type=ANCHOR_MEMREGION_FIRST_ACCESS,
                params={"base": region[0], "length": region[1], "access": "w"},
                label=f"first writer to {value_name or hex(cursor_addr)}",
            )
            return PhaseDiscoveryResult(
                value_addr=value_addr,
                boundary=PhaseBoundary(
                    name=phase_name or f"producer_of_{value_name or hex(value_addr)}",
                    region=region,
                    anchor=anchor,
                    note="no in-window writer found; producing phase unobserved",
                ),
                crosses_out=True,
                chain=tuple(chain),
                reason="no_writer_in_any_source",
            )

        chain.append(edge)
        if edge.kind == "writer":
            assert edge.writer is not None
            pc = edge.writer.pc
            in_window = (
                window_pc is not None and window_pc[0] <= pc < window_pc[1]
            )
            if in_window:
                return PhaseDiscoveryResult(
                    value_addr=value_addr,
                    boundary=None,
                    crosses_out=False,
                    chain=tuple(chain),
                    reason=f"in_window_writer pc=0x{pc:x}",
                )
            # Out-of-window writer — phase isolated.
            region = (value_addr, max(cfg.default_region_width, cursor_size))
            anchor = Anchor(
                anchor_type=ANCHOR_ADDR_FIRST_EXEC,
                params={"pc": pc},
                label=f"first execute at out-of-window writer 0x{pc:x}",
            )
            return PhaseDiscoveryResult(
                value_addr=value_addr,
                boundary=PhaseBoundary(
                    name=phase_name or f"producer_of_{value_name or hex(value_addr)}",
                    pc_range=(pc, pc + 4),  # one instruction wide; widens via instrument
                    region=region,
                    anchor=anchor,
                    note=f"out-of-window writer at pc=0x{pc:x}",
                ),
                crosses_out=True,
                chain=tuple(chain),
                reason=f"out_of_window_writer pc=0x{pc:x}",
            )

        # kind == "copy" — walk to the source address.
        if edge.src_addr is None:
            # malformed copy edge; fall back to memregion anchor.
            region = (value_addr, max(cfg.default_region_width, cursor_size))
            anchor = Anchor(
                anchor_type=ANCHOR_MEMREGION_FIRST_ACCESS,
                params={"base": region[0], "length": region[1], "access": "w"},
                label=f"first writer to {value_name or hex(value_addr)}",
            )
            return PhaseDiscoveryResult(
                value_addr=value_addr,
                boundary=PhaseBoundary(
                    name=phase_name or f"producer_of_{value_name or hex(value_addr)}",
                    region=region,
                    anchor=anchor,
                    note="copy edge with no src_addr; phase boundary unresolved",
                ),
                crosses_out=True,
                chain=tuple(chain),
                reason="malformed_copy_edge",
            )
        cursor_addr = edge.src_addr
        cursor_size = edge.writer.size if edge.writer else cursor_size

    # Hit max chain depth — treat as crosses-out at the final cursor.
    region = (cursor_addr, max(cfg.default_region_width, cursor_size))
    anchor = Anchor(
        anchor_type=ANCHOR_MEMREGION_FIRST_ACCESS,
        params={"base": region[0], "length": region[1], "access": "w"},
        label=f"first writer to ultimate-source 0x{cursor_addr:x}",
    )
    return PhaseDiscoveryResult(
        value_addr=value_addr,
        boundary=PhaseBoundary(
            name=phase_name or f"producer_of_{value_name or hex(value_addr)}",
            region=region,
            anchor=anchor,
            note=f"chain depth {cfg.max_chain_depth} reached; phase boundary partial",
        ),
        crosses_out=True,
        chain=tuple(chain),
        reason="max_chain_depth",
    )


# ---------------------------------------------------------------------------
# Wrapper-friendly auto-discovery on a params dict.
# ---------------------------------------------------------------------------


def discover_phases_in_params(
    params: dict[str, Any] | None,
    source: DiscoveryDataSource | None,
    *,
    cfg: PhaseDiscoveryConfig | None = None,
) -> list[PhaseDiscoveryResult]:
    """Walk a params dict for value records carrying a
    ``landing_address`` and run :func:`discover_phase` on each.
    No-op when the data source is unavailable — the wrapper supplies
    a source only when the current call has a loaded trace window.
    """
    cfg = cfg or PhaseDiscoveryConfig.from_env()
    if not cfg.enabled or params is None or source is None:
        return []
    out: list[PhaseDiscoveryResult] = []
    _walk_for_discovery(params, source, cfg, out)
    return out


def _walk_for_discovery(
    node: Any,
    source: DiscoveryDataSource,
    cfg: PhaseDiscoveryConfig,
    out: list[PhaseDiscoveryResult],
    *,
    depth: int = 5,
) -> None:
    if depth <= 0 or node is None:
        return
    if isinstance(node, dict):
        addr = node.get("landing_address")
        if addr is None:
            addr = node.get("addr")
        if isinstance(addr, int):
            name = str(node.get("value_name") or node.get("name") or "")
            size = int(node.get("size") or 1) if isinstance(node.get("size"), (int, str)) else 1
            res = discover_phase(
                addr, source,
                value_name=name,
                value_size=size,
                cfg=cfg,
            )
            if res.crosses_out:
                out.append(res)
        for v in node.values():
            _walk_for_discovery(v, source, cfg, out, depth=depth - 1)
    elif isinstance(node, list):
        for v in node:
            _walk_for_discovery(v, source, cfg, out, depth=depth - 1)


# ---------------------------------------------------------------------------
# Alerts.
# ---------------------------------------------------------------------------


def render_phase_discovery_alert(results: Iterable[PhaseDiscoveryResult]) -> str | None:
    crossings = [r for r in results if r.crosses_out and r.boundary is not None]
    if not crossings:
        return None
    parts = []
    for r in crossings:
        b = r.boundary
        assert b is not None
        anc = b.anchor.anchor_type if b.anchor else "?"
        parts.append(
            f"value@0x{r.value_addr:x} → phase {b.name} ({r.reason}, anchor={anc})"
        )
    return "[PHASE-DISCOVERY] producing phase outside window: " + "; ".join(parts)


__all__ = [
    "PhaseDiscoveryConfig",
    "DiscoveryDataSource",
    "InMemoryDataSource",
    "WriterHit",
    "ProducerEdge",
    "PhaseDiscoveryResult",
    "discover_phase",
    "discover_phases_in_params",
    "render_phase_discovery_alert",
]
