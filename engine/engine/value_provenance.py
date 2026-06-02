"""Value-provenance state machine.

Background — the reference target's Round 5 retro: the agent kept conflating two
different things under the word "recovered":

  * a value whose **bytes** were captured (hook / dump / I/O snapshot)
    — call it ``observed``; we know *what* but not *how*; running the
    algorithm forward from scratch is not yet possible.
  * a value whose **closed-form function** is available and
    ``f(input) == measured`` for every test vector — call it
    ``closed_form``; we know both *what* and *how*.

The first does not imply the second. observation_parity (two hook
points yielding the same bytes) is consistent with both an observed
constant *and* a real recompute, so parity alone cannot promote a
value to ``closed_form``. The retro reached this distinction by hand;
this module turns it into a primitive.

Rule (extends M1 — :mod:`engine.m1_success_audit`):

  - source ∈ {hook, dump, io, snapshot}  AND closed-form recompute
    NOT verified ⇒ provenance=``observed``; evidence_class is capped
    at **B**.
  - source ∈ {formula, closed_form} AND recompute function is
    declared AND ``recompute_matches_measured=True`` ⇒
    provenance=``closed_form``; evidence_class A allowed.
  - explicit hybrid (e.g. structure from formula, constants from
    dump) ⇒ provenance=``hybrid``; cap at **B** until both halves
    are closed.
  - anything else ⇒ provenance=``unknown``; cap at **C**.

This complements :mod:`engine.m1_success_audit` — M1 catches "a
dimension was held constant in the test"; this module catches "the
value was *observed* and never reconstructed". Both downgrade the
evidence_class ceiling; both are independently toggleable.

Independent toggle: ``UTOV_VALUE_PROVENANCE=off|0|false|no``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


# Source labels treated as "observation" — bytes were captured, the
# generating algorithm was NOT executed.
DEFAULT_OBSERVED_SOURCES: tuple[str, ...] = (
    "hook", "dump", "io", "snapshot", "memcpy_capture", "memory_watch",
)

# Source labels that count as a closed-form *origin*. Still requires
# the recompute predicate to actually match the measurements.
DEFAULT_CLOSED_FORM_SOURCES: tuple[str, ...] = (
    "formula", "closed_form", "ir", "decompiler", "triton_symbolic",
    "algorithm_template",
)


@dataclass(slots=True)
class ValueProvenanceConfig:
    enabled: bool = True
    observed_sources:    tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_OBSERVED_SOURCES,
    )
    closed_form_sources: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_CLOSED_FORM_SOURCES,
    )
    # When True, observation_parity=True alone never raises the
    # ceiling above B — the retro lesson made flesh.
    parity_does_not_imply_closed_form: bool = True

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "ValueProvenanceConfig":
        src = env if env is not None else os.environ
        cfg = cls()
        flag = (src.get("UTOV_VALUE_PROVENANCE") or "").strip().lower()
        if flag in ("off", "0", "false", "no"):
            cfg.enabled = False
        return cfg


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


# Provenance vocabulary. Single source of truth — agents and tests
# should import this rather than spelling the strings out.
PROVENANCE_OBSERVED:    str = "observed"
PROVENANCE_CLOSED_FORM: str = "closed_form"
PROVENANCE_HYBRID:      str = "hybrid"
PROVENANCE_UNKNOWN:     str = "unknown"


@dataclass(frozen=True, slots=True)
class ValueProvenanceResult:
    """One tagged value. ``ceiling`` is the highest evidence_class the
    value is *eligible* for given its provenance — independent of what
    the caller claimed."""

    value_name: str
    provenance: str            # observed | closed_form | hybrid | unknown
    ceiling: str               # 'A' | 'B' | 'C'
    requested_class: str | None
    downgraded: bool           # True iff requested_class > ceiling
    final_class: str           # min(requested, ceiling)
    reasons: tuple[str, ...]
    parity_disclaimer: str | None  # set when parity alone was offered as proof

    def to_dict(self) -> dict[str, Any]:
        return {
            "value_name":        self.value_name,
            "provenance":        self.provenance,
            "ceiling":           self.ceiling,
            "requested_class":   self.requested_class,
            "downgraded":        self.downgraded,
            "final_class":       self.final_class,
            "reasons":           list(self.reasons),
            "parity_disclaimer": self.parity_disclaimer,
        }


# ---------------------------------------------------------------------------
# Tagging
# ---------------------------------------------------------------------------


def _class_order(cls: str) -> int:
    return {"A": 3, "B": 2, "C": 1}.get(cls, 0)


def _min_class(a: str, b: str) -> str:
    return a if _class_order(a) <= _class_order(b) else b


def tag_value(
    record: dict[str, Any],
    *,
    cfg: ValueProvenanceConfig | None = None,
) -> ValueProvenanceResult:
    """Tag one value record.

    Recognised fields (all optional except ``value_name`` / ``source``):
      - ``value_name``                  (string)
      - ``source``                      (string, see *_sources lists)
      - ``recompute_fn_present``        (bool)
      - ``recompute_matches_measured``  (bool)
      - ``observation_parity``          (bool — captured-twice equality)
      - ``hybrid``                      (bool — explicit hybrid claim)
      - ``evidence_class``              (string — claimed class)
    """
    cfg = cfg or ValueProvenanceConfig.from_env()
    name = str(record.get("value_name") or record.get("name") or "<unnamed>")
    source = str(record.get("source") or "").strip().lower()
    fn_present = bool(record.get("recompute_fn_present"))
    fn_matches = bool(record.get("recompute_matches_measured"))
    parity     = bool(record.get("observation_parity"))
    is_hybrid  = bool(record.get("hybrid"))
    requested  = record.get("evidence_class")
    requested_s = str(requested).strip().upper() if requested else None

    reasons: list[str] = []
    parity_disclaimer: str | None = None

    # If module is off, pass through: provenance=unknown, ceiling=A.
    if not cfg.enabled:
        final = requested_s or "A"
        return ValueProvenanceResult(
            value_name=name,
            provenance=PROVENANCE_UNKNOWN,
            ceiling="A",
            requested_class=requested_s,
            downgraded=False,
            final_class=final,
            reasons=("UTOV_VALUE_PROVENANCE disabled",),
            parity_disclaimer=None,
        )

    is_obs_src    = source in cfg.observed_sources
    is_closed_src = source in cfg.closed_form_sources

    if is_hybrid:
        provenance = PROVENANCE_HYBRID
        ceiling = "B"
        reasons.append("explicit hybrid claim — cap at B until every component is closed")
    elif is_closed_src and fn_present and fn_matches:
        provenance = PROVENANCE_CLOSED_FORM
        ceiling = "A"
        reasons.append(f"source={source} with verified recompute → closed_form")
    elif is_obs_src:
        provenance = PROVENANCE_OBSERVED
        ceiling = "B"
        reasons.append(
            f"source={source} captures bytes, not algorithm → observed; "
            f"evidence_class A blocked until a closed-form recompute is supplied"
        )
        if parity and cfg.parity_does_not_imply_closed_form:
            parity_disclaimer = (
                "observation_parity=True confirms the BYTES match across "
                "captures — it does NOT recover the closed-form. Two hook "
                "points reading the same constant are equally parity-consistent."
            )
            reasons.append("parity present but does not imply closed_form (M1+ rule)")
    elif is_closed_src and not (fn_present and fn_matches):
        provenance = PROVENANCE_OBSERVED
        ceiling = "B"
        reasons.append(
            f"source={source} declared closed-form but recompute "
            f"unverified (fn_present={fn_present}, "
            f"matches_measured={fn_matches}) → degraded to observed"
        )
    else:
        provenance = PROVENANCE_UNKNOWN
        ceiling = "C"
        reasons.append(f"source={source or '(missing)'} is unrecognised — cap at C")

    if requested_s is None:
        final = ceiling
        downgraded = False
    else:
        final = _min_class(requested_s, ceiling)
        downgraded = final != requested_s
        if downgraded:
            reasons.append(
                f"requested evidence_class={requested_s} capped to {ceiling} by provenance={provenance}"
            )

    return ValueProvenanceResult(
        value_name=name,
        provenance=provenance,
        ceiling=ceiling,
        requested_class=requested_s,
        downgraded=downgraded,
        final_class=final,
        reasons=tuple(reasons),
        parity_disclaimer=parity_disclaimer,
    )


def tag_values_in_params(
    params: dict[str, Any] | None,
    *,
    cfg: ValueProvenanceConfig | None = None,
) -> list[ValueProvenanceResult]:
    """Walk ``params`` for value records and return one
    :class:`ValueProvenanceResult` per record found. A "value record"
    is any dict carrying both ``value_name`` (or ``name``) and
    ``source``. Mutates each found record in place to attach the
    resulting ``final_class`` / ``provenance`` so downstream code
    cannot keep using the un-capped claim."""
    cfg = cfg or ValueProvenanceConfig.from_env()
    out: list[ValueProvenanceResult] = []
    if not cfg.enabled or params is None:
        return out
    _walk(params, cfg, out)
    return out


def _walk(
    node: Any,
    cfg: ValueProvenanceConfig,
    out: list[ValueProvenanceResult],
    *,
    depth: int = 5,
) -> None:
    if depth <= 0 or node is None:
        return
    if isinstance(node, dict):
        if ("value_name" in node or "name" in node) and "source" in node:
            res = tag_value(node, cfg=cfg)
            out.append(res)
            node["provenance"]  = res.provenance
            node["final_class"] = res.final_class
            if res.downgraded:
                node["evidence_class"] = res.final_class
        for v in node.values():
            _walk(v, cfg, out, depth=depth - 1)
    elif isinstance(node, list):
        for v in node:
            _walk(v, cfg, out, depth=depth - 1)


def render_provenance_alert(results: list[ValueProvenanceResult]) -> str | None:
    """Format one envelope alert line summarising the tagged values.
    Returns ``None`` if nothing was downgraded and no parity warning
    needs surfacing (i.e. silent pass)."""
    if not results:
        return None
    downgraded = [r for r in results if r.downgraded]
    parity     = [r for r in results if r.parity_disclaimer]
    if not downgraded and not parity:
        return None
    parts: list[str] = []
    for r in downgraded:
        parts.append(
            f"{r.value_name}: {r.requested_class}→{r.final_class} "
            f"({r.provenance})"
        )
    if parity:
        parts.append(
            "parity-only: " + ", ".join(p.value_name for p in parity)
        )
    return f"[VALUE-PROVENANCE] {'; '.join(parts)}"
