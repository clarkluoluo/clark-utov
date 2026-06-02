"""Dispatch preflight coverage map — see all the work before running.

A VMP dispatch loop calls ~6 handler TYPES (keyed by the dispatch opcode, e.g.
``w21 & 0x3f``) many times. Solving invocation-by-invocation is whack-a-mole:
each new call may hit a fresh gap (an external memory input, an un-modeled
opcode), discovered only by running into it — one conversation round per gap.

This primitive classifies the call sequence into types and computes, ONCE per
type, the full I/O signature: register + memory live-in (the input gaps, via the
:mod:`setup_symex` live-in derivation), the opcodes the bulk decoder can't model
(the Level-2 escape-hatch gaps), and the outputs (the state carrier threaded to
the next handler). The whole gap list is then visible UP FRONT — solve each of
the ~6 types to EXACT, compose along the sequence, and N invocations are covered
by ~6 solves.

Target-agnostic boundary: the agent supplies the segmentation + classification
(how to read the dispatch opcode, where each handler body is) as
:class:`HandlerInvocation`\\ s, and the bulk-decode probe. The case-specific decode
(``w21 & 0x3f``) and concrete addresses never enter this primitive — they live in
the agent's config / the fixture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from .export_stamp import build_export_header
from .setup_symex import (
    MemLiveIn,
    derive_window_mem_live_in,
    derive_window_symbolic_regs,
)
from .types import Instruction

# A probe that tells whether the bulk decoder (Triton) can model an opcode.
# Injected so the primitive stays pure / testable; True = modeled.
DecodeProbe = Callable[[bytes], bool]


@dataclass(frozen=True, slots=True)
class HandlerInvocation:
    """One handler invocation in the dispatch trace, already classified.

    ``type_id`` is the dispatch-opcode key the agent decoded (target-specific —
    e.g. ``w21 & 0x3f``); ``idx_lo``/``idx_hi`` bound the handler body in
    EXECUTION ORDER (trace idx, so a recurring pc / branch side-path can't pull in
    the wrong occurrence — the same idx-segment rule the Level-2 runner uses)."""

    type_id: Any
    idx_lo:  int
    idx_hi:  int


@dataclass(frozen=True, slots=True)
class TypeCoverage:
    """The once-per-type I/O signature — the gap list for a handler type."""

    type_id:           Any
    occurrences:       int
    representative:    tuple[int, int]       # the rep invocation's idx window
    reg_live_in:       tuple[str, ...]       # register inputs (A: derive_window_symbolic_regs)
    mem_live_in:       tuple[MemLiveIn, ...]  # memory inputs (A: derive_window_mem_live_in)
    unmodeled_opcodes: tuple[str, ...]       # opcodes the bulk decoder can't model
    outputs:           tuple[str, ...]       # regs written here + read later = state carrier
    decode_probed:     bool                  # was a decode probe supplied?

    def to_dict(self) -> dict[str, Any]:
        return {
            "type_id":           self.type_id,
            "occurrences":       self.occurrences,
            "representative":    list(self.representative),
            "reg_live_in":       list(self.reg_live_in),
            "mem_live_in":       [m.to_dict() for m in self.mem_live_in],
            "unmodeled_opcodes": list(self.unmodeled_opcodes),
            "outputs":           list(self.outputs),
            "decode_probed":     self.decode_probed,
            "kind":              "dispatch_type_coverage",
        }


@dataclass(frozen=True, slots=True)
class CoverageMap:
    """The whole VM program's coverage: the call sequence + each type's gaps."""

    types:    tuple[TypeCoverage, ...]
    sequence: tuple[Any, ...]                # [type_id, …] — the VM "program"

    @property
    def n_types(self) -> int:
        return len(self.types)

    def to_dict(self) -> dict[str, Any]:
        return {
            "types":      [t.to_dict() for t in self.types],
            "sequence":   list(self.sequence),
            "n_types":    self.n_types,
            "n_invocations": len(self.sequence),
            "kind":       "dispatch_coverage_map",
        }

    def to_markdown(self) -> str:
        """A human/agent table — one row per type, the gap list at a glance."""
        lines = [
            "# Dispatch coverage map",
            "",
            f"{len(self.sequence)} invocation(s) → {self.n_types} handler type(s). "
            "Solve each type once, compose along the sequence.",
            "",
            "| type | × | reg inputs | mem inputs (symbolize/back?) | unmodeled | outputs |",
            "|---|---|---|---|---|---|",
        ]
        for t in self.types:
            mem = ", ".join(f"0x{m.addr:x}+{m.size}@idx{m.src_idx}" for m in t.mem_live_in) or "—"
            lines.append(
                f"| {t.type_id} | {t.occurrences} | {', '.join(t.reg_live_in) or '—'} "
                f"| {mem} | {', '.join(t.unmodeled_opcodes) or '—'} "
                f"| {', '.join(t.outputs) or '—'} |")
        lines.append("")
        lines.append("sequence: " + " → ".join(str(s) for s in self.sequence))
        return "\n".join(lines) + "\n"

    def to_stamped_markdown(
        self, *, source: str, exec_identity: dict[str, Any], ts: str,
        from_entries: Iterable[str] = (),
    ) -> str:
        """The coverage table with the authoritative ``utov-export`` stamp header."""
        header = build_export_header(
            source=source, exported_by="preflight_dispatch_coverage",
            exec_identity=exec_identity, from_entries=from_entries, ts=ts)
        return header + self.to_markdown()


def _window_outputs(items: Sequence[Instruction], lo: int, hi: int) -> tuple[str, ...]:
    """Registers WRITTEN in the idx window AND read after it = the state carrier
    threaded to the next handler (live-out of the window)."""
    written: set[str] = set()
    read_after: set[str] = set()
    for ins in items:
        if lo <= ins.idx <= hi:
            written |= set(ins.regs_write.keys())
        elif ins.idx > hi:
            read_after |= set(ins.regs_read.keys())
    return tuple(sorted(written & read_after))


def preflight_dispatch_coverage(
    trace: Iterable[Instruction],
    *,
    invocations: Sequence[HandlerInvocation],
    reg_file: Sequence[str] | None = None,
    decode_probe: DecodeProbe | None = None,
) -> CoverageMap:
    """Build the dispatch coverage map from a classified call sequence.

    ``invocations`` is the agent-supplied, already-classified call sequence (each
    a ``type_id`` + idx window). Grouped by ``type_id``; the FIRST invocation of a
    type is its representative. For each type the I/O signature is computed
    mechanically — register + memory live-in (the input gaps), the opcodes
    ``decode_probe`` rejects (the un-modeled gaps; left empty when no probe is
    supplied, flagged via ``decode_probed``), and the live-out outputs. Returns a
    :class:`CoverageMap` whose ``sequence`` is the whole VM "program"."""
    items = list(trace)
    groups: dict[Any, list[HandlerInvocation]] = {}
    sequence: list[Any] = []
    order: list[Any] = []
    for inv in invocations:
        if inv.type_id not in groups:
            groups[inv.type_id] = []
            order.append(inv.type_id)
        groups[inv.type_id].append(inv)
        sequence.append(inv.type_id)

    types: list[TypeCoverage] = []
    for type_id in order:
        invs = groups[type_id]
        rep = invs[0]
        win = (rep.idx_lo, rep.idx_hi)
        reg_li, _ = derive_window_symbolic_regs(
            items, window=win, reg_file=reg_file, window_is_idx=True)
        mem_li, _ = derive_window_mem_live_in(items, window=win, window_is_idx=True)
        win_items = [ins for ins in items if rep.idx_lo <= ins.idx <= rep.idx_hi]
        unmodeled: tuple[str, ...] = ()
        if decode_probe is not None:
            seen: list[str] = []
            for ins in win_items:
                if not decode_probe(bytes(ins.bytes_)):
                    h = bytes(ins.bytes_).hex()
                    if h not in seen:
                        seen.append(h)
            unmodeled = tuple(seen)
        types.append(TypeCoverage(
            type_id=type_id,
            occurrences=len(invs),
            representative=win,
            reg_live_in=reg_li,
            mem_live_in=mem_li,
            unmodeled_opcodes=unmodeled,
            outputs=_window_outputs(items, rep.idx_lo, rep.idx_hi),
            decode_probed=decode_probe is not None))
    return CoverageMap(types=tuple(types), sequence=tuple(sequence))


def triton_decode_probe() -> DecodeProbe:
    """A :data:`DecodeProbe` backed by Triton: True when the bulk decoder models
    the opcode. Raises if Triton is unavailable (honest — the caller can pass a
    different probe, or omit it to skip the un-modeled-opcode column)."""
    from .setup_symex_runner import triton_available, triton_unavailable_reason
    if not triton_available():
        raise RuntimeError(
            f"triton_decode_probe needs Triton: {triton_unavailable_reason()}")
    from triton import ARCH, MODE, TritonContext  # type: ignore
    from triton import Instruction as TritonInstr  # type: ignore

    def _probe(code: bytes) -> bool:
        ctx = TritonContext()
        ctx.setArchitecture(ARCH.AARCH64)
        ctx.setMode(MODE.ALIGNED_MEMORY, True)
        try:
            t = TritonInstr()
            t.setAddress(0x1000)
            t.setOpcode(code)
            ctx.processing(t)
            return True
        except Exception:
            return False

    return _probe


__all__ = [
    "DecodeProbe",
    "HandlerInvocation",
    "TypeCoverage",
    "CoverageMap",
    "preflight_dispatch_coverage",
    "triton_decode_probe",
]
