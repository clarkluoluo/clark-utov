"""Recapture-target orchestration — auto-produce a precise re-collection spec
when the *target output buffer* was never observed.

Background (F0 end-to-end compose, 2026-06-01): recovery verifies four
intermediate transform windows EXACT, yet cannot connect to the final
``cipher_body``. The raw/cipher buffer's *contents* never entered the trace,
and its address is not stable across runs (heap / dynamic allocation). utov
today only reports ``NEEDS_OBSERVATION`` / ``OUTPUT_NOT_OBSERVABLE`` — it tells
the harness *that* the buffer was not observed, but not *which* concrete target
to re-collect. The agent then hand-searches across multiple rounds.

This module closes the "primitive in, not wired into one shot" gap. All three
primitives already exist:

  * :mod:`engine.dataflow`         — ``regflow_forward`` / ``producer_backward``
                                     pointer-follow (which reg held which value).
  * :mod:`engine.oracle_sink`      — ``validate_sink`` locates where the target
                                     bytes land on THIS run (even partially).
  * :mod:`engine.watch_first_write`— ``request_watch_first_write`` builds the
                                     runner contract (now reg-relative capable).

What was missing was the ORCHESTRATION:

    NEEDS_OBSERVATION on the target output
        -> locate the target buffer base on the deriving run (validate_sink)
        -> pointer-follow from the nearest observed pointer register
           (dataflow) to that buffer base
        -> derive ``[base_reg + offset]``
        -> emit a reg-relative WatchFirstWriteSpec (covering the target full
           length) + a ``recapture_directive`` for gap_map / terminal evidence.

Because the buffer address is not stable across runs, the produced watch spec
is **register-relative** (``[xN + off]``), never a bare constant address. The
register and offset are derived from REAL dataflow + the buffer's observed
location on this run — never fabricated (invariant 8). When no observed pointer
register reaches the buffer, the orchestration degrades EXPLICITLY (it does not
silently skip — A8④).

Zero case-fit: no F0 address, no handler name, no fixed register is baked in.
The pointer register, offset, and target length all come from the caller's
inputs + the trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .dataflow import producer_backward
from .oracle_sink import SinkValidation, SinkVerdict, validate_sink
from .types import Instruction
from .watch_first_write import (
    WATCH_KIND_READ,
    WATCH_KIND_WRITE,
    WatchFirstWriteSpec,
    request_point_watch,
    request_watch_first_write,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class RecaptureTargetStatus:
    """String status codes for a derive attempt (kept as plain strings so the
    directive serializes cleanly into gap_map / terminal JSON)."""

    DERIVED = "DERIVED"                  # reg-relative watch spec produced
    BUFFER_NOT_LOCATED = "BUFFER_NOT_LOCATED"   # target bytes nowhere on this run
    NO_POINTER = "NO_POINTER"            # buffer located, but no reg points at it
    INSUFFICIENT_COVERAGE = "INSUFFICIENT_COVERAGE"   # partial match too weak to anchor


# Coverage / confidence gate (dev-closure-evidence-layering-trap-state-spec, task 4).
# A partial-match buffer anchor is trustworthy only when the partial match is STRONG
# enough relative to the TARGET — a 3/65 (≈5%) match almost certainly anchored the
# WRONG buffer and would send the agent to re-collect the wrong address (worse than
# silence: confidently wrong). The threshold is a RATIO (match_count / target_length),
# NOT an absolute count, so it is target/形态-agnostic (A7: "换标的还成立吗"). The
# default 0.25 means "at least a quarter of the target's bytes already match here";
# callers override via ``min_partial_coverage``. The default is NOT derived from any
# case number (not 3/65, not 3/8) — it is a round "non-trivial fraction" floor that
# sits well above the 3/65 (≈5%) noise case the spec calls out and well below a
# genuinely-partial buffer (a third / half already matching). A SINK_CONFIRMED /
# WRONG_SINK base (a full located region, not a partial) bypasses this gate entirely
# — it is not a partial anchor.
_DEFAULT_MIN_PARTIAL_COVERAGE = 0.25


@dataclass(frozen=True)
class PointerHit:
    """A register whose OBSERVED value reaches the target buffer base.
    ``offset = buffer_base - reg_value`` (>= 0; the buffer is at or after the
    pointer)."""
    idx: int                 # trace idx where the register held this value
    reg: str
    reg_value: int
    offset: int
    mnemonic: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
            "reg": self.reg,
            "reg_value": f"0x{self.reg_value:x}",
            "offset": self.offset,
            "mnemonic": self.mnemonic,
        }


@dataclass(frozen=True)
class RecaptureDirective:
    """The auto-produced re-collection instruction for the harness, plus the
    derivation evidence. Written into gap_map / terminal evidence so a human or
    a narrow agent sees "re-collect THIS" without hand-searching.

    On the degraded paths (``BUFFER_NOT_LOCATED`` / ``NO_POINTER``) ``spec`` is
    None and ``detail`` carries the explicit cannot-derive reason."""

    status: str
    value_name: str
    target_length: int
    spec: WatchFirstWriteSpec | None = None
    buffer_base: int | None = None          # base located on the deriving run
    pointer: PointerHit | None = None       # the observed pointer that reaches it
    detail: str = ""
    candidates_considered: int = 0          # observed pointer regs inspected
    coverage: dict[str, Any] | None = None  # task 4: partial-match coverage facts
    # How the produced spec captures the target:
    #   "point_watch"     — precise PC-gated single-point capture (clean, no
    #                        noise, will not flood the runner's record cap);
    #   "reg_relative_range" — the wide reg-relative range fallback (the only
    #                        option when the arm PC is not cleanly known). This
    #                        is NOISY and MAY HIT the runner's record cap — it is
    #                        NOT clean provenance (see ``capture_risk``).
    capture_mode: str | None = None
    # Explicit risk annotation for the wide-range fallback (A8④ / spec ②: never
    # silently degrade). None on the clean point-watch path.
    capture_risk: str | None = None

    @property
    def derived(self) -> bool:
        return self.status == RecaptureTargetStatus.DERIVED

    @property
    def is_point_watch(self) -> bool:
        return self.capture_mode == "point_watch"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": "recapture_directive",
            "status": self.status,
            "value_name": self.value_name,
            "target_length": self.target_length,
            "spec": self.spec.to_dict() if self.spec is not None else None,
            "buffer_base": None if self.buffer_base is None else f"0x{self.buffer_base:x}",
            "pointer": self.pointer.to_dict() if self.pointer is not None else None,
            "candidates_considered": self.candidates_considered,
            "detail": self.detail,
        }
        if self.coverage is not None:
            out["coverage"] = dict(self.coverage)
        if self.capture_mode is not None:
            out["capture_mode"] = self.capture_mode
        if self.capture_risk is not None:
            out["capture_risk"] = self.capture_risk
        return out


# ---------------------------------------------------------------------------
# Step 1: locate the target buffer base on the deriving run
# ---------------------------------------------------------------------------


def _locate_buffer_base(
    items: list[Instruction],
    target_value: bytes,
    *,
    sink_validation: SinkValidation | None,
    snapshots: Any | None,
) -> tuple[int | None, str, dict[str, Any]]:
    """Find where the target bytes land on THIS run — even when not fully
    captured. Reuses :func:`oracle_sink.validate_sink` (no new locating logic):

      * a confirmed/located base (SINK_CONFIRMED / WRONG_SINK) is used directly;
      * otherwise the OUTPUT_NOT_OBSERVABLE ``longest_partial`` base is the best
        anchor (the region where the most target bytes already match — that is
        exactly the buffer that is only partially observed).

    Returns ``(base_or_None, how, partial_info)``. ``partial_info`` carries the
    coverage facts (``match_count`` / ``length`` / ``coverage``) of a partial anchor
    so the caller can apply the task-4 coverage gate; ``{}`` for a fully-located base
    (not a partial → no gate) or no match."""
    sv = sink_validation
    if sv is None:
        sv = validate_sink(items, target_value, snapshots=snapshots)

    if sv.verdict in (SinkVerdict.SINK_CONFIRMED, SinkVerdict.WRONG_SINK) and sv.base is not None:
        return sv.base, f"validate_sink:{sv.verdict.value.lower()}", {}

    # OUTPUT_NOT_OBSERVABLE: anchor on the partially-matching region, if any.
    partial = sv.longest_partial or {}
    base_hex = partial.get("base")
    if base_hex and partial.get("match_count", 0) > 0:
        match_count = int(partial.get("match_count", 0))
        length = int(partial.get("length", len(target_value)) or len(target_value))
        coverage = (match_count / length) if length else 0.0
        return int(base_hex, 16), "validate_sink:longest_partial", {
            "match_count": match_count,
            "length": length,
            "coverage": round(coverage, 4),
        }

    return None, "no-region-matches-target", {}


# ---------------------------------------------------------------------------
# Step 2: pointer-follow — nearest observed pointer register reaching the base
# ---------------------------------------------------------------------------


def _nearest_pointer_to_base(
    items: list[Instruction],
    buffer_base: int,
    *,
    sink_idx: int,
    max_offset: int,
    max_back: int,
) -> tuple[PointerHit | None, int]:
    """Find the OBSERVED pointer register nearest the sink boundary whose value
    reaches ``buffer_base`` within ``[0, max_offset]``.

    Reuses :func:`dataflow.producer_backward`: for each candidate offset (0 ==
    the buffer base itself is in a register, growing outward), ask "who most
    recently wrote ``buffer_base - offset`` into any register before the sink".
    The hit with the SMALLEST offset whose producer is the MOST RECENT wins —
    i.e. the nearest observed pointer to the buffer. ``offset = buffer_base -
    reg_value``.

    Returns ``(hit_or_None, candidates_considered)``.

    No new locating logic: the per-value backward search is the existing
    primitive; this only sweeps offsets and picks the nearest pointer."""
    best: PointerHit | None = None
    considered = 0
    for offset in range(0, max_offset + 1):
        cand_value = buffer_base - offset
        if cand_value < 0:
            break
        hit = producer_backward(items, cand_value, sink_idx, max_back=max_back)
        if hit is None:
            continue
        considered += 1
        ph = PointerHit(
            idx=hit.idx, reg=hit.reg, reg_value=hit.value,
            offset=offset, mnemonic=hit.mnemonic,
        )
        # Prefer the smallest offset; among equal-quality, the most recent
        # producer (highest idx) — that is the nearest observed pointer.
        if best is None or ph.offset < best.offset or (
            ph.offset == best.offset and ph.idx > best.idx
        ):
            best = ph
    return best, considered


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def derive_recapture_directive(
    items: list[Instruction],
    target_value: bytes,
    value_name: str,
    *,
    sink_idx: int | None = None,
    sink_validation: SinkValidation | None = None,
    snapshots: Any | None = None,
    max_offset: int = 0x10000,
    max_back: int = 100_000,
    reason: str | None = None,
    min_partial_coverage: float = _DEFAULT_MIN_PARTIAL_COVERAGE,
    point_watch_pc: int | None = None,
    point_watch_kind: str = WATCH_KIND_READ,
) -> RecaptureDirective:
    """Auto-produce a recapture directive for an UNOBSERVED target output buffer.

    Call this when sink validation / end-to-end provenance returns
    NEEDS_OBSERVATION (the target's bytes were not captured). Given the known
    target value (oracle output) + the trace's register/pointer observations,
    it:

      1. locates the target buffer base on the deriving run (validate_sink);
      2. pointer-follows from the nearest observed pointer register to that base
         (dataflow.producer_backward) and derives ``[base_reg + offset]``;
      3. emits a reg-relative :class:`WatchFirstWriteSpec` covering the target
         full length + a :class:`RecaptureDirective` for gap_map / terminal.

    **Two capture modes (spec ①②):**

      * **point-watch (preferred, clean)** — when the arm PC is KNOWN (caller
        passes ``point_watch_pc``) AND a pointer register + offset are derived
        from the trace, the directive captures exactly the one PC-gated
        reg-relative access (``capture_mode="point_watch"``). No noise, never
        floods the runner's record cap.
      * **wide reg-relative range (fallback, NOISY)** — when the arm PC is NOT
        known, the engine CANNOT cleanly produce a point-watch (it must not
        fabricate a PC — see the module note). It falls back to a wide
        reg-relative watch over the full target length and EXPLICITLY marks the
        capture as noisy + possibly cap-hitting via ``capture_risk``
        (``capture_mode="reg_relative_range"``). This is NOT clean provenance —
        the annotation is mandatory (A8④: never silently degrade).

    ``point_watch_kind`` is ``read`` or ``write`` (the capture direction at the
    armed PC); it only applies to the point-watch path.

    **Why the engine does NOT derive the arm PC itself:** the pointer-follow
    finds where the base register was *set up* (a ``mov``/``add``), which is not
    the PC of the future *access* of ``[base_reg+offset]``; and for an unobserved
    READ target the access instruction is, by definition, not in the trace. The
    arm PC is therefore a caller-supplied fact, never invented here.

    Degradation is EXPLICIT (A8④, invariant 8): if the buffer is nowhere on this
    run, or no observed pointer reaches it, the directive carries a cannot-derive
    status + detail and ``spec=None`` — never a fabricated address.

    Generic: the register, offset, length, and arm PC come from the trace +
    caller — no target-specific value is baked in.
    """
    if point_watch_pc is not None:
        if not isinstance(point_watch_pc, int) or point_watch_pc < 0:
            raise ValueError(
                f"point_watch_pc must be a non-negative int, got {point_watch_pc!r}")
        if point_watch_kind not in (WATCH_KIND_READ, WATCH_KIND_WRITE):
            raise ValueError(
                f"point_watch_kind must be read|write, got {point_watch_kind!r}")
    items = list(items)
    target_value = bytes(target_value)
    if not target_value:
        raise ValueError("target_value must be non-empty")
    target_length = len(target_value)
    if sink_idx is None:
        sink_idx = len(items)   # whole trace is "before" the (unobserved) sink

    # --- Step 1: where do the target bytes land on this run? ---
    buffer_base, how, partial_info = _locate_buffer_base(
        items, target_value, sink_validation=sink_validation, snapshots=snapshots)
    if buffer_base is None:
        return RecaptureDirective(
            status=RecaptureTargetStatus.BUFFER_NOT_LOCATED,
            value_name=value_name, target_length=target_length,
            detail=(
                "cannot derive recapture target: the target bytes do not appear "
                "(even partially) in any observed region of this run — there is no "
                "buffer base to anchor a pointer-follow on. Widen the trace window "
                "so the target buffer is at least partially captured, then re-derive."
            ),
        )

    # --- Step 1b: coverage / confidence gate (task 4) ---
    # A PARTIAL anchor (longest_partial) is trustworthy only when the match is strong
    # enough relative to the target. A 3/65 (≈5%) partial almost certainly anchored
    # the WRONG buffer →派 a confidently-wrong reg-relative watch. Below the ratio
    # threshold: DO NOT派 watch; surface an explicit INSUFFICIENT_COVERAGE state with
    # the match_count/total so the consumer knows the evidence is too thin to anchor
    # (A8④ — degradation is explicit, never a self-confident wrong directive). A fully
    # located base (SINK_CONFIRMED / WRONG_SINK → partial_info empty) is NOT gated.
    if partial_info:
        coverage = float(partial_info.get("coverage", 0.0))
        if coverage < min_partial_coverage:
            return RecaptureDirective(
                status=RecaptureTargetStatus.INSUFFICIENT_COVERAGE,
                value_name=value_name, target_length=target_length,
                buffer_base=buffer_base, coverage=dict(partial_info),
                detail=(
                    "cannot derive a trustworthy recapture target: the best partial "
                    f"match covers only {partial_info.get('match_count')}/"
                    f"{partial_info.get('length')} target bytes "
                    f"(coverage {coverage:.1%} < {min_partial_coverage:.0%} floor) — "
                    f"the anchor at 0x{buffer_base:x} ({how}) is too weak to trust "
                    "(it likely points at the WRONG buffer). Do NOT install a "
                    "reg-relative watch on it; widen the trace so MORE of the target "
                    "buffer is captured (raising the partial-match coverage), then "
                    "re-derive. Lower ``min_partial_coverage`` only if you can confirm "
                    "the partial region truly is the target buffer."
                ),
            )

    # --- Step 2: nearest observed pointer register reaching that base ---
    pointer, considered = _nearest_pointer_to_base(
        items, buffer_base, sink_idx=sink_idx,
        max_offset=max_offset, max_back=max_back)
    if pointer is None:
        return RecaptureDirective(
            status=RecaptureTargetStatus.NO_POINTER,
            value_name=value_name, target_length=target_length,
            buffer_base=buffer_base, candidates_considered=considered,
            coverage=dict(partial_info) if partial_info else None,
            detail=(
                "cannot derive recapture target: no observed pointer reaches the "
                f"buffer (located at 0x{buffer_base:x} via {how}). No register in "
                "the trace held the buffer base (or a base within "
                f"0x{max_offset:x} bytes of it) before the sink boundary, so a "
                "register-relative watch cannot be anchored. Widen the trace to "
                "capture the pointer setup, or snapshot the buffer directly."
            ),
        )

    # --- Step 3: build the watch spec + directive ---
    # Preferred (clean) path: a PC-gated single-point watch — but ONLY when the
    # caller supplied the arm PC. The engine never invents it (the pointer-follow
    # gives the pointer-SETUP PC, not the access PC; for an unobserved read the
    # access PC is not in the trace at all). With (pc, base_reg, offset, width)
    # all known, capture exactly one access — no noise, no record-cap flood.
    if point_watch_pc is not None:
        why = reason or (
            f"target output {value_name} ({target_length}B) was not observed; "
            f"point-watch at pc=0x{point_watch_pc:x}: on reaching it, capture the "
            f"{point_watch_kind} of [{pointer.reg} + 0x{pointer.offset:x}] "
            f"({target_length}B) resolved from the LIVE register "
            f"(reg={pointer.reg}=0x{pointer.reg_value:x} at idx {pointer.idx}, "
            f"buffer base 0x{buffer_base:x} via {how}). Single-point — clean, no "
            f"wide-range noise, will not hit the runner's record cap."
        )
        spec = request_point_watch(
            point_watch_pc, pointer.reg, pointer.offset, target_length,
            value_name, kind=point_watch_kind, addr=buffer_base, reason=why,
        )
        return RecaptureDirective(
            status=RecaptureTargetStatus.DERIVED,
            value_name=value_name, target_length=target_length,
            spec=spec, buffer_base=buffer_base, pointer=pointer,
            candidates_considered=considered, detail=why,
            coverage=dict(partial_info) if partial_info else None,
            capture_mode="point_watch", capture_risk=None,
        )

    # Fallback (NOISY): arm PC unknown → cannot cleanly produce a point-watch.
    # Emit a wide reg-relative watch over the full target length and EXPLICITLY
    # mark the noise/cap risk (A8④ / spec ②: never silently degrade). This is NOT
    # clean provenance — downstream must treat it accordingly.
    risk = (
        "WIDE reg-relative range capture (no arm PC known) — this sweeps the whole "
        f"[{pointer.reg} + 0x{pointer.offset:x}] region every time it is touched, so "
        "it is NOISY (unrelated reads/writes mixed in) and MAY HIT the runner's "
        "record cap (e.g. X25_REGREL_CONCRETE_WRITE_MAX), in which case the ledger is "
        "truncated and is NOT clean provenance. To get a clean single-point capture, "
        "supply the access PC via ``point_watch_pc`` (then a point-watch is produced)."
    )
    why = reason or (
        f"target output {value_name} ({target_length}B) was not observed; its "
        f"buffer lands at [{pointer.reg} + 0x{pointer.offset:x}] on this run "
        f"(reg={pointer.reg}=0x{pointer.reg_value:x} at idx {pointer.idx}, "
        f"buffer base 0x{buffer_base:x} via {how}). Address is not stable across "
        f"runs — re-collect register-relative. " + risk
    )
    spec = request_watch_first_write(
        buffer_base, value_name,
        reason=why, base_reg=pointer.reg, offset=pointer.offset,
        width_bytes=target_length,
    )
    return RecaptureDirective(
        status=RecaptureTargetStatus.DERIVED,
        value_name=value_name, target_length=target_length,
        spec=spec, buffer_base=buffer_base, pointer=pointer,
        candidates_considered=considered, detail=why,
        coverage=dict(partial_info) if partial_info else None,
        capture_mode="reg_relative_range", capture_risk=risk,
    )


__all__ = [
    "RecaptureTargetStatus",
    "PointerHit",
    "RecaptureDirective",
    "derive_recapture_directive",
]
