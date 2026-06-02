"""Observation readers — normalise runner capture formats into canonical shapes.

Two agent-written tools (toolsFromTestAgent/) re-implemented format parsing inline
with case-specific addresses baked in. The reusable kernel is just the parsing:
turn a runner's calltrace log + register-pointed memory dumps into the engine's
canonical shapes, which the existing primitives already consume generically —
#1 sink-validator / #3 provenance over MemSnapshot, indirect-call targets feeding
the OPAQUE_CALLEE terminal. No target address, tag, or PC is hardcoded here.

  - read_calltrace / parse_calltrace -> list[CallEvent] from a tab-separated
    `kind \\t pc \\t target \\t <cols...>` calltrace; indirect_call_targets()
    pulls the resolved (pc, target) of every BLR/BR/BLX.
  - read_hook_snapshots / parse_hook_snapshots -> list[MemSnapshot] from a JSONL
    register-pointed memory dump: a `mem_<reg>` field is the bytes at the address
    in the row's `<reg>` register (with an optional `_0x<off>` suffix).

These are READERS (the runner/agent still does the capture, per the runner
contract); they only canonicalise a captured artifact so utov can consume it
without each agent re-writing the parse.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .trace_merge import MemEvent
from .types import MemOp, MemSnapshot

_INDIRECT_CALLS = frozenset({"blr", "br", "blx"})


@dataclass(frozen=True)
class CallEvent:
    kind: str                       # BL / BLR / B / ... (as in the log)
    pc: int
    target: int | None              # resolved target, or None ("-"/"")
    cols: tuple[str, ...] = ()       # remaining columns verbatim (regs/mem; format-specific)
    raw: str = ""

    @property
    def is_indirect_call(self) -> bool:
        return self.kind.lower() in _INDIRECT_CALLS

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "pc": f"0x{self.pc:x}",
                "target": None if self.target is None else f"0x{self.target:x}",
                "cols": list(self.cols)}


def _to_int(tok: str) -> int | None:
    tok = tok.strip()
    if not tok or tok == "-":
        return None
    try:
        return int(tok, 16)
    except ValueError:
        return None


def parse_calltrace(text: str) -> list[CallEvent]:
    """Parse a tab-separated calltrace: `kind \\t pc_hex \\t target_hex \\t ...`.
    Comment (#) and blank lines are skipped; malformed lines are skipped."""
    events: list[CallEvent] = []
    for ln in text.splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        parts = ln.split("\t")
        if len(parts) < 2:
            continue
        pc = _to_int(parts[1])
        if pc is None:
            continue
        target = _to_int(parts[2]) if len(parts) > 2 else None
        events.append(CallEvent(kind=parts[0].strip(), pc=pc, target=target,
                                cols=tuple(parts[3:]), raw=ln))
    return events


def read_calltrace(path: str | Path) -> list[CallEvent]:
    return parse_calltrace(Path(path).read_text(encoding="utf-8", errors="replace"))


def indirect_call_targets(events: Iterable[CallEvent]) -> list[tuple[int, int]]:
    """The (call-site pc, resolved target) of every indirect call (BLR/BR/BLX)
    whose target is known — feeds #3's OPAQUE_CALLEE callee naming generically."""
    out: list[tuple[int, int]] = []
    for e in events:
        if e.is_indirect_call and e.target is not None:
            out.append((e.pc, e.target))
    return out


# --- hook-dump -> MemSnapshot -----------------------------------------------

_MEM_FIELD = re.compile(r"^mem_(?P<reg>[a-zA-Z]+\d*|sp|lr|fp)(?:_0x(?P<off>[0-9a-fA-F]+))?$")


def _resolve_snapshot(field_name: str, value: str, row: dict) -> MemSnapshot | None:
    m = _MEM_FIELD.match(field_name)
    if not m or not isinstance(value, str):
        return None
    hexs = value.replace(" ", "")
    if len(hexs) < 2 or len(hexs) % 2 or not re.fullmatch(r"[0-9a-fA-F]+", hexs):
        return None   # "skip"/"err"/odd-length -> not a byte dump
    reg = m.group("reg")
    base = _to_int(str(row.get(reg, "")))
    if base is None:
        return None   # cannot resolve the address without the register's value
    off = int(m.group("off"), 16) if m.group("off") else 0
    return MemSnapshot(addr=base + off, data=bytes.fromhex(hexs),
                       label=str(row.get("tag", "")),
                       source=f"hook@{row.get('pc_rva', row.get('pc', '?'))}")


def parse_hook_snapshots(text: str) -> list[MemSnapshot]:
    """Parse a JSONL register-pointed memory dump into MemSnapshots. Each
    `mem_<reg>[_0x<off>]` field is the bytes at `row[<reg>] + off`. Rows/fields
    that don't resolve to a concrete (addr, bytes) are skipped (not an error)."""
    snaps: list[MemSnapshot] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            row = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        for k, v in row.items():
            s = _resolve_snapshot(k, v, row)
            if s is not None:
                snaps.append(s)
    return snaps


def read_hook_snapshots(path: str | Path) -> list[MemSnapshot]:
    return parse_hook_snapshots(Path(path).read_text(encoding="utf-8", errors="replace"))


# --- memory-event sidecar (_mem.jsonl) -> MemEvent ---------------------------
#
# A separate per-step memory sidecar: one JSON object per memory access, carrying
# a locating key (idx preferred, pc fallback) so it can be merged back onto the
# main Instruction stream by :func:`engine.trace_merge.merge_trace_sources`. This
# is the READER (canonicalise a captured artifact); the engine never parses a
# runner-proprietary format — the field names are the canonical ones below. Rows
# that lack the required (rw, addr, val, size) or a locating key are skipped (not
# an error), so a malformed line never crashes the merge.


def parse_mem_events(text: str) -> list[MemEvent]:
    """Parse a JSONL memory sidecar into canonical :class:`MemEvent`s.

    Each row: ``{"idx"?:int, "pc"?:hexstr, "rw":"r"|"w", "addr":hexstr,
    "val":hexstr|int, "size":int}``. ``idx`` is the preferred locating key; ``pc``
    (hex) the fallback. A row missing rw/addr/size or BOTH locating keys is
    skipped (not an error)."""
    events: list[MemEvent] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            row = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        rw = row.get("rw")
        if rw not in ("r", "w"):
            continue
        addr = _to_int(str(row.get("addr", "")))
        size = row.get("size")
        if addr is None or not isinstance(size, int):
            continue
        raw_val = row.get("val", 0)
        val = raw_val if isinstance(raw_val, int) else (_to_int(str(raw_val)) or 0)
        idx = row.get("idx")
        idx = idx if isinstance(idx, int) else None
        pc = _to_int(str(row["pc"])) if "pc" in row else None
        if idx is None and pc is None:
            continue   # no locating key — cannot align
        events.append(MemEvent(
            op=MemOp(rw=rw, addr=addr, val=val, size=size), idx=idx, pc=pc))
    return events


def read_mem_events(path: str | Path) -> list[MemEvent]:
    return parse_mem_events(Path(path).read_text(encoding="utf-8", errors="replace"))


__all__ = [
    "CallEvent", "parse_calltrace", "read_calltrace", "indirect_call_targets",
    "parse_hook_snapshots", "read_hook_snapshots",
    "parse_mem_events", "read_mem_events",
]
