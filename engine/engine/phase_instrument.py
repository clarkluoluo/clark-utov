"""Phase instrumentation — anchor + granularity spec.

Solves the general "target phase can't be reached from main task
timing" problem: the observer install timing is decoupled from the
main task and bound to an :class:`engine.phase.Anchor`. The capture
granularity is configurable from sparse sampling up to full
instruction-level trace.

  * :class:`PhaseInstrumentSpec` is the runner-facing contract — when
    to hook, how much to capture, where to land the output. Built by
    :func:`request_phase_instrument` from a :class:`PhaseBoundary` (the
    output of :mod:`engine.phase_discovery`) or directly by a caller
    that already knows the anchor.

  * :class:`PhaseInstrumentResult` is the runner-side return shape:
    pointer to the captured JSONL trace + sidecar with entry state
    and memory snapshot. Engine-side this is a passive carrier; the
    engine does not synthesise results, only consumes them. Tests in
    this repo drive end-to-end via synthesized fixtures that emit a
    result of this exact shape.

  * Auto-suggestion gate. When :mod:`engine.phase_discovery` finds a
    boundary on a given call, :func:`suggest_instrument_for_boundary`
    builds the matching spec at full instruction-level granularity
    (the current defect being addressed is "granularity too coarse",
    so the default for an auto-suggestion is aggressive — explicit
    callers can downgrade).

Independent toggle: ``UTOV_PHASE_INSTRUMENT=off|0|false|no``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .phase import (
    Anchor,
    PhaseBoundary,
    ANCHOR_FUNC_ENTRY,
    ANCHOR_ADDR_FIRST_EXEC,
    ANCHOR_MEMREGION_FIRST_ACCESS,
    KNOWN_ANCHOR_TYPES,
)


# Granularity ladder. Strings are wire-stable; runners compare them
# directly. Ordered loosely from cheapest to most detailed.
GRAN_SPARSE_SAMPLE  = "sparse_sample"      # caller-specified PC list only
GRAN_PC_BAND        = "pc_band"            # every step inside a PC range, no mem/reg deltas
GRAN_REG_DELTA      = "reg_delta"          # every step, register diffs only
GRAN_FULL           = "full_instruction"   # every step + full regs_read/write + mem ops

KNOWN_GRANULARITIES: frozenset[str] = frozenset({
    GRAN_SPARSE_SAMPLE,
    GRAN_PC_BAND,
    GRAN_REG_DELTA,
    GRAN_FULL,
})


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PhaseInstrumentConfig:
    enabled: bool = True
    # Granularity used by auto-suggestion. The current defect is
    # "granularity too coarse" — explicit callers can request less.
    default_auto_granularity: str = GRAN_FULL
    # Cap on captured steps. Runners that hit the cap MUST surface
    # ``truncated=True`` in the result so the engine knows the unit
    # is partial.
    max_steps: int = 100_000

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "PhaseInstrumentConfig":
        src = env if env is not None else os.environ
        cfg = cls()
        flag = (src.get("UTOV_PHASE_INSTRUMENT") or "").strip().lower()
        if flag in ("off", "0", "false", "no"):
            cfg.enabled = False
        g = (src.get("UTOV_PHASE_INSTRUMENT_GRANULARITY") or "").strip()
        if g and g in KNOWN_GRANULARITIES:
            cfg.default_auto_granularity = g
        m = src.get("UTOV_PHASE_INSTRUMENT_MAX_STEPS")
        if m is not None:
            try:
                cfg.max_steps = int(m)
            except ValueError:
                pass
        return cfg


# ---------------------------------------------------------------------------
# Spec — engine→runner contract.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PhaseInstrumentSpec:
    """The contract a runner must fulfil to capture a phase.

    Fields.
      ``phase_name``  human-readable tag (also used as filename prefix).
      ``anchor``      when to start capturing.
      ``granularity`` one of :data:`KNOWN_GRANULARITIES`.
      ``stop``        optional stop condition. A dict with one of:
                      ``{"after_steps": int}``,
                      ``{"on_return_from": <pc>}``,
                      ``{"on_exit_pc": <pc>}``,
                      ``{"on_region_quiet": [base, length, n_steps]}``.
                      Wire-stable; runners may extend.
      ``regions``     optional list of ``(base, length)`` memory
                      regions whose every read/write must be captured
                      in addition to the granularity rules.
      ``max_steps``   hard cap; runner must mark result truncated if
                      hit before stop condition fires.
      ``label``       free-form, for logs only.
    """

    phase_name:   str
    anchor:       Anchor
    granularity:  str
    stop:         dict[str, Any] | None = None
    regions:      tuple[tuple[int, int], ...] = ()
    max_steps:    int = 100_000
    label:        str = ""

    def __post_init__(self) -> None:
        if self.granularity not in KNOWN_GRANULARITIES:
            raise ValueError(
                f"unknown granularity {self.granularity!r}; "
                f"known={sorted(KNOWN_GRANULARITIES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind":        "phase_instrument",
            "phase_name":  self.phase_name,
            "anchor":      self.anchor.to_dict(),
            "granularity": self.granularity,
            "stop":        dict(self.stop) if self.stop else None,
            "regions":     [list(r) for r in self.regions],
            "max_steps":   self.max_steps,
            "label":       self.label,
        }


# ---------------------------------------------------------------------------
# Result — runner→engine.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PhaseInstrumentResult:
    """The shape the runner returns after fulfilling a spec.

    Both paths land here:
      * a live runner that actually hooked the anchor and streamed,
      * a fixture-mode synthesizer used in tests / file-mode runs.

    ``jsonl_path`` is the path to a JSONL trace file using the same
    schema as the main trace (contracts/runner_interface.md §2.1).
    ``sidecar_path`` is the JSON file emitted via
    :func:`engine.phase.write_sidecar` carrying the entry state +
    memory snapshot for the anchor hit moment.

    Both paths are passed by reference (the engine reads them
    lazily) so the result is cheap to ship over JSON-RPC.
    """

    spec:           PhaseInstrumentSpec
    jsonl_path:     Path
    sidecar_path:   Path
    anchor_hit_idx: int          # main-trace idx where the anchor fired
    captured_steps: int
    truncated:      bool = False
    note:           str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec":           self.spec.to_dict(),
            "jsonl_path":     str(self.jsonl_path),
            "sidecar_path":   str(self.sidecar_path),
            "anchor_hit_idx": self.anchor_hit_idx,
            "captured_steps": self.captured_steps,
            "truncated":      self.truncated,
            "note":           self.note,
        }


# ---------------------------------------------------------------------------
# Suggestion.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PhaseInstrumentSuggestion:
    """An auto-suggested spec attached to the discipline envelope.
    Mirrors :class:`engine.watch_first_write.WatchSuggestion` —
    advisory + (optional) triggerable spec."""

    phase_name: str
    spec:       PhaseInstrumentSpec | None
    advisory:   str

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_name": self.phase_name,
            "spec":       self.spec.to_dict() if self.spec else None,
            "advisory":   self.advisory,
        }


# ---------------------------------------------------------------------------
# Primitive entry point + auto-suggestion.
# ---------------------------------------------------------------------------


def request_phase_instrument(
    *,
    phase_name: str,
    anchor: Anchor,
    granularity: str = GRAN_FULL,
    stop: dict[str, Any] | None = None,
    regions: Iterable[tuple[int, int]] = (),
    max_steps: int | None = None,
    label: str = "",
    cfg: PhaseInstrumentConfig | None = None,
) -> PhaseInstrumentSpec:
    """Build a runner-fulfillable phase-instrument spec.

    The runner side is contractually expected to:
      * install the anchor (per ``anchor.anchor_type``),
      * begin capturing per ``granularity`` once the anchor fires,
      * write a JSONL trace + sidecar covering the captured steps,
      * return a :class:`PhaseInstrumentResult`.
    """
    cfg = cfg or PhaseInstrumentConfig.from_env()
    if not cfg.enabled:
        raise RuntimeError("UTOV_PHASE_INSTRUMENT disabled — primitive is unavailable")
    if anchor.anchor_type not in KNOWN_ANCHOR_TYPES:
        raise ValueError(
            f"unknown anchor_type {anchor.anchor_type!r}; "
            f"known={sorted(KNOWN_ANCHOR_TYPES)}"
        )
    cap = max_steps if max_steps is not None else cfg.max_steps
    return PhaseInstrumentSpec(
        phase_name=phase_name,
        anchor=anchor,
        granularity=granularity,
        stop=dict(stop) if stop else None,
        regions=tuple((int(b), int(n)) for b, n in regions),
        max_steps=cap,
        label=label,
    )


def suggest_instrument_for_boundary(
    boundary: PhaseBoundary,
    *,
    granularity: str | None = None,
    cfg: PhaseInstrumentConfig | None = None,
) -> PhaseInstrumentSuggestion | None:
    """Build an auto-suggestion spec for a discovered phase boundary.

    Auto-suggestion chooses the *most informative* anchor available:

      * if the boundary supplies an anchor (discovery already chose
        one), reuse it;
      * else if the boundary has a pc_range, use ``addr_first_exec``
        on the range lower bound;
      * else if the boundary has a region, use
        ``memregion_first_access`` on the region;
      * else (unknown phase) → no spec, only an advisory note.

    Granularity defaults to ``full_instruction`` so the captured
    unit is rich enough for the main pipeline.
    """
    cfg = cfg or PhaseInstrumentConfig.from_env()
    if not cfg.enabled:
        return None
    g = granularity or cfg.default_auto_granularity

    anchor: Anchor | None = boundary.anchor
    if anchor is None:
        if boundary.pc_range is not None:
            anchor = Anchor(
                anchor_type=ANCHOR_ADDR_FIRST_EXEC,
                params={"pc": boundary.pc_range[0]},
                label=f"first-exec at pc_range start of {boundary.name}",
            )
        elif boundary.region is not None:
            anchor = Anchor(
                anchor_type=ANCHOR_MEMREGION_FIRST_ACCESS,
                params={
                    "base":   boundary.region[0],
                    "length": boundary.region[1],
                    "access": "w",
                },
                label=f"first-write to region of {boundary.name}",
            )

    if anchor is None:
        return PhaseInstrumentSuggestion(
            phase_name=boundary.name,
            spec=None,
            advisory=(
                f"phase {boundary.name} located but anchor unresolvable "
                f"(no pc_range or region); manual anchor required."
            ),
        )

    regions: list[tuple[int, int]] = []
    if boundary.region is not None:
        regions.append(boundary.region)

    spec = request_phase_instrument(
        phase_name=boundary.name,
        anchor=anchor,
        granularity=g,
        regions=regions,
        label=f"auto-suggest for {boundary.name}",
        cfg=cfg,
    )
    advisory = (
        f"phase {boundary.name} producing value lives outside current "
        f"window — recommend phase_instrument(anchor={anchor.anchor_type}, "
        f"granularity={g}) to capture full producer execution. "
        f"Result can be fed to main pipeline as a ReplayableUnit."
    )
    return PhaseInstrumentSuggestion(
        phase_name=boundary.name,
        spec=spec,
        advisory=advisory,
    )


def suggest_instruments_for_results(
    discovery_results: Iterable[Any],   # phase_discovery.PhaseDiscoveryResult
    *,
    cfg: PhaseInstrumentConfig | None = None,
) -> list[PhaseInstrumentSuggestion]:
    """Convert a batch of discovery results into instrument
    suggestions. Skips results without a boundary (no phase to
    instrument)."""
    cfg = cfg or PhaseInstrumentConfig.from_env()
    if not cfg.enabled:
        return []
    out: list[PhaseInstrumentSuggestion] = []
    for r in discovery_results:
        boundary = getattr(r, "boundary", None)
        if boundary is None:
            continue
        s = suggest_instrument_for_boundary(boundary, cfg=cfg)
        if s is not None:
            out.append(s)
    return out


def render_phase_instrument_alert(
    suggestions: Iterable[PhaseInstrumentSuggestion],
) -> str | None:
    items = list(suggestions)
    if not items:
        return None
    parts = []
    for s in items:
        if s.spec is not None:
            parts.append(
                f"{s.phase_name} (anchor={s.spec.anchor.anchor_type}, "
                f"granularity={s.spec.granularity})"
            )
        else:
            parts.append(f"{s.phase_name} (advisory only)")
    return "[PHASE-INSTRUMENT] suggested: " + "; ".join(parts)


__all__ = [
    "GRAN_SPARSE_SAMPLE",
    "GRAN_PC_BAND",
    "GRAN_REG_DELTA",
    "GRAN_FULL",
    "KNOWN_GRANULARITIES",
    "PhaseInstrumentConfig",
    "PhaseInstrumentSpec",
    "PhaseInstrumentResult",
    "PhaseInstrumentSuggestion",
    "request_phase_instrument",
    "suggest_instrument_for_boundary",
    "suggest_instruments_for_results",
    "render_phase_instrument_alert",
]
