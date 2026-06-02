"""Constant-provenance classifier.

Decides what kind of value we're looking at — a true hard-coded
constant, a deterministic function of a fixed input (typically
appkey), a session-level derivative whose value rerolls each
session, or a per-input variable that doesn't deserve a constant
claim at all. Drives the evidence_class ceiling and downstream
routing.

Background. The reference target prefix and template both fell into the
session-level trap because nothing automatically distinguished
"the bytes are the same every time we look at them in this
session" from "the bytes are computed once per session and reused".
Manual M1 audit rescued the analysis. This module solidifies the
distinction as a deterministic capability so the agent doesn't have
to re-derive it case-by-case.

Two orthogonal probes — neither alone is sufficient:

  * **rerun variability** (M3 generalised across four axes).
    Control-variable rerun N times and record whether the value
    changes when the *same session* / *new session* / *new appkey*
    / *new per-input* is varied. The pattern of change/no-change
    locates the category.

  * **producer dataflow** (covers the rerun probe's blindspot).
    Inspect what inputs the producer instructions read. If the
    rerun looked stable but the producer reads ``time``,
    ``random``, or a ``session_token``, the test environment
    locked the entropy source and the value is session-level
    despite appearing constant in reruns.

Crossing the two probes:

  * dataflow says reads session/time/random → category is
    SESSION_LEVEL_DERIVED, period. Rerun stability is the lie.
  * dataflow says reads only static/appkey, reruns say new_appkey
    changes it → APPKEY_FIXED_FUNCTION.
  * dataflow says reads only static, reruns say nothing ever
    changes → HARDCODED_FIXED.
  * reruns say new_per_input changes it → PER_INPUT_VARIABLE
    (treat as variable; no constant claim possible).

Evidence-class ceiling per category (M1 link):

  | Category                  | Ceiling | Scope        |
  |---------------------------|---------|--------------|
  | HARDCODED_FIXED           | A       | universal    |
  | APPKEY_FIXED_FUNCTION     | A       | per_appkey   |
  | SESSION_LEVEL_DERIVED     | B       | per_session  |
  | PER_INPUT_VARIABLE        | —       | per_input    |
  | UNDETERMINED              | B       | unspecified  |

Recommended actions for the downstream router:

  HARDCODED_FIXED        → auto_pin (close the node)
  APPKEY_FIXED_FUNCTION  → mark_dual_path
                           ("recover f(appkey) → A, else pin
                           current as B with per_appkey scope")
  SESSION_LEVEL_DERIVED  → escalate_usage_decision
                           (cap at B, surface per_session scope to
                           user for usage-intent decision)
  PER_INPUT_VARIABLE     → treat_as_variable
  UNDETERMINED           → request_more_observations
                           (specify which axes are missing)

Relationship to existing gates. ``engine.value_provenance`` caps
hook/dump observed values at evidence_class B. That's a coarser
rule — it doesn't distinguish session-level vs. truly fixed within
the observed class. This module subsumes M3 (per-input axis) and
the M1 "dimension-variability=0 means untested" check as the
unified multi-dim provenance test. Existing gates are not refactored
to use this — they keep their narrower contracts; this module
ships as the recommended classifier for anything that needs the
finer-grained verdict.

Independent toggle: ``UTOV_CONSTANT_PROVENANCE=off|0|false|no``.
"""

from __future__ import annotations

import enum
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Enums.
# ---------------------------------------------------------------------------


class SourceCategory(enum.Enum):
    HARDCODED_FIXED         = "hardcoded_fixed"
    APPKEY_FIXED_FUNCTION   = "appkey_fixed_function"
    SESSION_LEVEL_DERIVED   = "session_level_derived"
    PER_INPUT_VARIABLE      = "per_input_variable"
    UNDETERMINED            = "undetermined"


class RerunDimension(enum.Enum):
    """The four canonical control axes the rerun probe sweeps."""

    SAME_SESSION   = "same_session"
    NEW_SESSION    = "new_session"
    NEW_APPKEY     = "new_appkey"
    NEW_PER_INPUT  = "new_per_input"


class DataflowReadKind(enum.Enum):
    """What kind of source the producer instructions read."""

    STATIC         = "static"
    APPKEY         = "appkey"
    TIME           = "time"
    RANDOM         = "random"
    SESSION_TOKEN  = "session_token"
    INPUT          = "input"
    ALLOCATOR      = "allocator"
    UNKNOWN        = "unknown"


# Read kinds that imply session-level entropy, regardless of what
# the rerun probe saw. If the producer reads any of these, the value
# is session-level even if reruns looked stable.
SESSION_ENTROPY_READS: frozenset[DataflowReadKind] = frozenset({
    DataflowReadKind.TIME,
    DataflowReadKind.RANDOM,
    DataflowReadKind.SESSION_TOKEN,
})


# Read kinds that are *noise* for provenance purposes — they materialise
# the destination register (a malloc / mmap side-effect) but say nothing
# about the *concrete bytes* written into that register. The judgement
# criterion is the materialised dataflow link to the concrete-bytes
# producer, not the allocator write. (§19.9 base #8 / v0.4.0 B5 —
# one-target field evidence pending a second-target confirmation.)
NOISE_READS: frozenset[DataflowReadKind] = frozenset({
    DataflowReadKind.ALLOCATOR,
})


class RoutingAction(enum.Enum):
    AUTO_PIN                    = "auto_pin"
    MARK_DUAL_PATH              = "mark_dual_path"
    ESCALATE_USAGE_DECISION     = "escalate_usage_decision"
    TREAT_AS_VARIABLE           = "treat_as_variable"
    REQUEST_MORE_OBSERVATIONS   = "request_more_observations"


# Mapping category → (ceiling, scope, action).
_CATEGORY_TABLE: dict[SourceCategory, tuple[str, str, RoutingAction]] = {
    SourceCategory.HARDCODED_FIXED:       ("A", "universal",     RoutingAction.AUTO_PIN),
    SourceCategory.APPKEY_FIXED_FUNCTION: ("A", "per_appkey",    RoutingAction.MARK_DUAL_PATH),
    SourceCategory.SESSION_LEVEL_DERIVED: ("B", "per_session",   RoutingAction.ESCALATE_USAGE_DECISION),
    SourceCategory.PER_INPUT_VARIABLE:    ("",  "per_input",     RoutingAction.TREAT_AS_VARIABLE),
    SourceCategory.UNDETERMINED:          ("B", "unspecified",   RoutingAction.REQUEST_MORE_OBSERVATIONS),
}


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ConstantProvenanceConfig:
    enabled: bool = True
    # Minimum samples per dimension before we trust a "stable"
    # verdict. Two observations on an axis isn't enough to claim
    # universal stability; we require at least this many.
    min_samples_per_axis: int = 2

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "ConstantProvenanceConfig":
        src = env if env is not None else os.environ
        cfg = cls()
        flag = (src.get("UTOV_CONSTANT_PROVENANCE") or "").strip().lower()
        if flag in ("off", "0", "false", "no"):
            cfg.enabled = False
        n = src.get("UTOV_CONSTANT_PROVENANCE_MIN_SAMPLES")
        if n is not None:
            try:
                cfg.min_samples_per_axis = max(1, int(n))
            except ValueError:
                pass
        return cfg


# ---------------------------------------------------------------------------
# Probe inputs.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RerunObservation:
    """One observed value of the same logical value, varied on one
    dimension. ``dimension`` says which axis was varied for *this*
    observation relative to the baseline; ``value_hex`` is the
    captured bytes for comparison."""

    dimension:  RerunDimension
    value_hex:  str           # hex string for stable equality
    note:       str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "value_hex": self.value_hex,
            "note":      self.note,
        }


@dataclass(frozen=True, slots=True)
class DataflowSummary:
    """Producer's dataflow at a glance — what kinds of inputs the
    producing instructions read. Aggregated upstream from s3_dfg /
    a rerun probe; this module only consumes the summary.

    Allocator side-effects (a ``malloc`` / ``mmap`` materialising the
    destination register) are *not* a source of the concrete bytes —
    the bytes still come from whatever populates the buffer after
    allocation. The methods below consult :meth:`meaningful_reads`,
    which filters :data:`NOISE_READS` (currently
    ``DataflowReadKind.ALLOCATOR``); ``producer_reads`` itself stays
    unmodified so callers can still see the raw observation if they
    want it.
    """

    producer_reads: tuple[DataflowReadKind, ...]
    note: str = ""

    def meaningful_reads(self) -> tuple[DataflowReadKind, ...]:
        return tuple(r for r in self.producer_reads if r not in NOISE_READS)

    def reads_session_entropy(self) -> bool:
        return any(r in SESSION_ENTROPY_READS for r in self.meaningful_reads())

    def reads_only_static_or_appkey(self) -> bool:
        reads = self.meaningful_reads()
        if not reads:
            return False
        allowed = {DataflowReadKind.STATIC, DataflowReadKind.APPKEY}
        return all(r in allowed for r in reads)

    def reads_appkey(self) -> bool:
        return DataflowReadKind.APPKEY in self.meaningful_reads()

    def reads_input(self) -> bool:
        return DataflowReadKind.INPUT in self.meaningful_reads()

    def reads_allocator(self) -> bool:
        """Did the producer touch an allocator (informational)?
        Allocator reads are filtered from the other ``reads_*`` checks
        but kept on the raw ``producer_reads`` for diagnostic display.
        """
        return DataflowReadKind.ALLOCATOR in self.producer_reads

    def to_dict(self) -> dict[str, Any]:
        return {
            "producer_reads": [r.value for r in self.producer_reads],
            "note":           self.note,
        }


# ---------------------------------------------------------------------------
# Probe one — rerun variability.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RerunAnalysis:
    """Per-axis result of the rerun probe. ``stable=None`` means
    not enough samples on that axis to decide."""

    same_session_stable:  bool | None
    new_session_stable:   bool | None
    new_appkey_stable:    bool | None
    new_per_input_stable: bool | None

    def axes_with_evidence(self) -> int:
        return sum(1 for v in (
            self.same_session_stable, self.new_session_stable,
            self.new_appkey_stable,   self.new_per_input_stable,
        ) if v is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "same_session_stable":  self.same_session_stable,
            "new_session_stable":   self.new_session_stable,
            "new_appkey_stable":    self.new_appkey_stable,
            "new_per_input_stable": self.new_per_input_stable,
            "axes_with_evidence":   self.axes_with_evidence(),
        }


def analyse_reruns(
    observations: Iterable[RerunObservation],
    *,
    cfg: ConstantProvenanceConfig | None = None,
) -> RerunAnalysis:
    """Probe-one analysis. Groups observations by dimension and
    decides per-axis stability."""
    cfg = cfg or ConstantProvenanceConfig.from_env()
    by_axis: dict[RerunDimension, list[str]] = defaultdict(list)
    for o in observations:
        by_axis[o.dimension].append(o.value_hex)

    def _axis_stable(axis: RerunDimension) -> bool | None:
        values = by_axis.get(axis, [])
        # SAME_SESSION needs at least 2 samples; the cross-axis
        # axes (NEW_SESSION/APPKEY/PER_INPUT) need at least one
        # sample whose value we can compare against the same_session
        # baseline. If we only have one sample on those axes and no
        # same_session baseline either, we can't decide.
        if axis == RerunDimension.SAME_SESSION:
            if len(values) < cfg.min_samples_per_axis:
                return None
            return len(set(values)) == 1
        # Other axes: compare against same_session baseline if
        # present, otherwise against the first value seen.
        baseline_pool = by_axis.get(RerunDimension.SAME_SESSION) or values
        if not values or not baseline_pool:
            return None
        baseline = baseline_pool[0]
        return all(v == baseline for v in values)

    return RerunAnalysis(
        same_session_stable=_axis_stable(RerunDimension.SAME_SESSION),
        new_session_stable=_axis_stable(RerunDimension.NEW_SESSION),
        new_appkey_stable=_axis_stable(RerunDimension.NEW_APPKEY),
        new_per_input_stable=_axis_stable(RerunDimension.NEW_PER_INPUT),
    )


# ---------------------------------------------------------------------------
# Probe two — dataflow.
# ---------------------------------------------------------------------------


def classify_from_dataflow_only(
    dataflow: DataflowSummary | None,
) -> SourceCategory:
    """Probe-two-only verdict — uses the producer's dataflow alone.

    Returns the *strongest* claim we can make purely from dataflow
    (no rerun evidence). The cross-classifier may upgrade /
    downgrade based on rerun results.

    Allocator-only dataflow returns ``UNDETERMINED`` — the allocator
    write doesn't tell us where the concrete bytes came from (§19.9
    base #8). Callers should supply the materialised concrete-bytes
    producer as a second read kind to get a real verdict.
    """
    if dataflow is None or not dataflow.producer_reads:
        return SourceCategory.UNDETERMINED
    if not dataflow.meaningful_reads():
        # Producer recorded only allocator-style noise; no signal about
        # the actual byte source.
        return SourceCategory.UNDETERMINED
    if dataflow.reads_session_entropy():
        return SourceCategory.SESSION_LEVEL_DERIVED
    if dataflow.reads_input():
        # Producer reads per-input bytes — can still be a
        # function of input, not necessarily a hard-coded constant.
        return SourceCategory.PER_INPUT_VARIABLE
    if dataflow.reads_appkey():
        return SourceCategory.APPKEY_FIXED_FUNCTION
    if dataflow.reads_only_static_or_appkey():
        # static-only, no appkey involvement.
        return SourceCategory.HARDCODED_FIXED
    return SourceCategory.UNDETERMINED


# ---------------------------------------------------------------------------
# Cross-classifier.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConstantProvenanceResult:
    """Combined verdict for one logical value."""

    value_name:              str
    category:                SourceCategory
    evidence_class_ceiling:  str        # "" / "A" / "B"
    scope:                   str
    recommended_action:      RoutingAction
    rerun_analysis:          RerunAnalysis | None
    dataflow:                DataflowSummary | None
    signals:                 tuple[str, ...] = ()
    reasoning:               str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "value_name":             self.value_name,
            "category":               self.category.value,
            "evidence_class_ceiling": self.evidence_class_ceiling,
            "scope":                  self.scope,
            "recommended_action":     self.recommended_action.value,
            "rerun_analysis":         self.rerun_analysis.to_dict() if self.rerun_analysis else None,
            "dataflow":               self.dataflow.to_dict() if self.dataflow else None,
            "signals":                list(self.signals),
            "reasoning":              self.reasoning,
        }


def classify_value(
    value_name: str,
    *,
    rerun_observations: Iterable[RerunObservation] = (),
    dataflow: DataflowSummary | None = None,
    cfg: ConstantProvenanceConfig | None = None,
) -> ConstantProvenanceResult:
    """Run both probes, cross them, and produce the verdict."""
    cfg = cfg or ConstantProvenanceConfig.from_env()
    if not cfg.enabled:
        return ConstantProvenanceResult(
            value_name=value_name,
            category=SourceCategory.UNDETERMINED,
            evidence_class_ceiling="",
            scope="",
            recommended_action=RoutingAction.REQUEST_MORE_OBSERVATIONS,
            rerun_analysis=None,
            dataflow=dataflow,
            reasoning="constant_provenance disabled",
        )

    rerun_list = list(rerun_observations)
    rerun = analyse_reruns(rerun_list, cfg=cfg) if rerun_list else None
    signals: list[str] = []
    reasoning_parts: list[str] = []

    # Probe 2 — dataflow override path (covers entropy-locked
    # blindspot of probe 1).
    if dataflow is not None and dataflow.reads_session_entropy():
        signals.append("dataflow.reads_session_entropy=true")
        reasoning_parts.append(
            "producer reads time / random / session_token — "
            "session-level regardless of rerun stability."
        )
        return _result(
            value_name, SourceCategory.SESSION_LEVEL_DERIVED,
            rerun, dataflow, signals, " ".join(reasoning_parts),
        )

    # Probe 1 — rerun pattern.
    if rerun is not None:
        # PER_INPUT: changes when input changes (rest of axes
        # irrelevant once we see input-dependence).
        if rerun.new_per_input_stable is False:
            signals.append("rerun.new_per_input_changed")
            reasoning_parts.append(
                "value changes when per-input is varied — not a constant."
            )
            return _result(
                value_name, SourceCategory.PER_INPUT_VARIABLE,
                rerun, dataflow, signals, " ".join(reasoning_parts),
            )
        # SESSION_LEVEL: changes across sessions, not per-input.
        if rerun.new_session_stable is False:
            signals.append("rerun.new_session_changed")
            reasoning_parts.append(
                "value changes across sessions (input held) — session-level."
            )
            return _result(
                value_name, SourceCategory.SESSION_LEVEL_DERIVED,
                rerun, dataflow, signals, " ".join(reasoning_parts),
            )
        # APPKEY_FIXED_FUNCTION: stable within appkey, changes with
        # appkey.
        if rerun.new_appkey_stable is False and rerun.new_session_stable is not False:
            signals.append("rerun.new_appkey_changed")
            reasoning_parts.append(
                "value stable across sessions and inputs but changes with appkey — "
                "appkey-keyed function."
            )
            return _result(
                value_name, SourceCategory.APPKEY_FIXED_FUNCTION,
                rerun, dataflow, signals, " ".join(reasoning_parts),
            )
        # All axes stable → HARDCODED_FIXED, but only if we have
        # enough axes of evidence. Without enough axes we fall
        # through to UNDETERMINED.
        all_stable = (
            rerun.same_session_stable is True
            and rerun.new_session_stable is True
            and rerun.new_appkey_stable is True
        )
        if all_stable:
            signals.append("rerun.all_axes_stable")
            # If dataflow corroborates static-only, we can promote.
            if dataflow is not None and dataflow.reads_only_static_or_appkey():
                if dataflow.reads_appkey():
                    reasoning_parts.append(
                        "all reruns stable across sessions and appkey but producer "
                        "reads appkey — treat as appkey-fixed function with "
                        "current appkey constant."
                    )
                    return _result(
                        value_name, SourceCategory.APPKEY_FIXED_FUNCTION,
                        rerun, dataflow, signals, " ".join(reasoning_parts),
                    )
                signals.append("dataflow.static_only")
                reasoning_parts.append(
                    "all reruns stable and dataflow shows producer reads only "
                    "static sources — hard-coded constant."
                )
                return _result(
                    value_name, SourceCategory.HARDCODED_FIXED,
                    rerun, dataflow, signals, " ".join(reasoning_parts),
                )
            if dataflow is None:
                # No dataflow corroboration. Reruns alone can be
                # fooled by an entropy-locked environment — we
                # downgrade to UNDETERMINED with a note so the
                # caller knows to add a dataflow probe.
                reasoning_parts.append(
                    "all reruns stable but no dataflow evidence — "
                    "cannot rule out entropy-locked environment hiding "
                    "session-level reads. Request producer-dataflow probe."
                )
                return _result(
                    value_name, SourceCategory.UNDETERMINED,
                    rerun, dataflow, signals, " ".join(reasoning_parts),
                )
            # dataflow present but not strictly static — fall through.

    # Dataflow-only fallback when reruns are absent or
    # inconclusive.
    if dataflow is not None:
        df_cat = classify_from_dataflow_only(dataflow)
        if df_cat is not SourceCategory.UNDETERMINED:
            signals.append(f"dataflow_only:{df_cat.value}")
            reasoning_parts.append(
                "rerun probe inconclusive; verdict derived from producer dataflow alone."
            )
            return _result(
                value_name, df_cat,
                rerun, dataflow, signals, " ".join(reasoning_parts),
            )

    reasoning_parts.append(
        "insufficient observations — need rerun samples on multiple "
        "axes and a producer-dataflow probe."
    )
    return _result(
        value_name, SourceCategory.UNDETERMINED,
        rerun, dataflow, signals, " ".join(reasoning_parts),
    )


def _result(
    value_name: str,
    category: SourceCategory,
    rerun: RerunAnalysis | None,
    dataflow: DataflowSummary | None,
    signals: list[str],
    reasoning: str,
) -> ConstantProvenanceResult:
    ceiling, scope, action = _CATEGORY_TABLE[category]
    return ConstantProvenanceResult(
        value_name=value_name,
        category=category,
        evidence_class_ceiling=ceiling,
        scope=scope,
        recommended_action=action,
        rerun_analysis=rerun,
        dataflow=dataflow,
        signals=tuple(signals),
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Wrapper-friendly walk over params.
# ---------------------------------------------------------------------------


def classify_values_in_params(
    params: dict[str, Any] | None,
    *,
    cfg: ConstantProvenanceConfig | None = None,
) -> list[ConstantProvenanceResult]:
    """Walk a params dict for value records that carry
    ``rerun_observations`` and/or ``producer_dataflow`` shapes.

    A value record is classified when it has at least one of those
    keys; records with neither are silently skipped (the caller is
    free to attach the data later in the pipeline).
    """
    cfg = cfg or ConstantProvenanceConfig.from_env()
    if not cfg.enabled or params is None:
        return []
    out: list[ConstantProvenanceResult] = []
    _walk_for_classify(params, cfg, out)
    return out


def _walk_for_classify(
    node: Any,
    cfg: ConstantProvenanceConfig,
    out: list[ConstantProvenanceResult],
    *,
    depth: int = 6,
) -> None:
    if depth <= 0 or node is None:
        return
    if isinstance(node, dict):
        has_value_name = ("value_name" in node) or ("name" in node)
        has_provenance_data = (
            "rerun_observations" in node
            or "producer_dataflow" in node
        )
        if has_value_name and has_provenance_data:
            name = str(node.get("value_name") or node.get("name") or "<unnamed>")
            reruns = _parse_rerun_observations(node.get("rerun_observations"))
            dataflow = _parse_dataflow(node.get("producer_dataflow"))
            out.append(classify_value(
                name, rerun_observations=reruns,
                dataflow=dataflow, cfg=cfg,
            ))
        for v in node.values():
            _walk_for_classify(v, cfg, out, depth=depth - 1)
    elif isinstance(node, list):
        for v in node:
            _walk_for_classify(v, cfg, out, depth=depth - 1)


def _parse_rerun_observations(raw: Any) -> list[RerunObservation]:
    if not isinstance(raw, list):
        return []
    parsed: list[RerunObservation] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        dim_raw = entry.get("dimension")
        if not isinstance(dim_raw, str):
            continue
        try:
            dim = RerunDimension(dim_raw)
        except ValueError:
            continue
        val = entry.get("value_hex") or entry.get("value")
        if not isinstance(val, str):
            continue
        parsed.append(RerunObservation(
            dimension=dim,
            value_hex=val,
            note=str(entry.get("note", "")),
        ))
    return parsed


def _parse_dataflow(raw: Any) -> DataflowSummary | None:
    if not isinstance(raw, dict):
        return None
    reads_raw = raw.get("producer_reads") or raw.get("reads")
    if not isinstance(reads_raw, list):
        return None
    reads: list[DataflowReadKind] = []
    for r in reads_raw:
        if not isinstance(r, str):
            continue
        try:
            reads.append(DataflowReadKind(r))
        except ValueError:
            reads.append(DataflowReadKind.UNKNOWN)
    return DataflowSummary(
        producer_reads=tuple(reads),
        note=str(raw.get("note", "")),
    )


# ---------------------------------------------------------------------------
# Alerts.
# ---------------------------------------------------------------------------


def render_constant_provenance_alert(
    results: Iterable[ConstantProvenanceResult],
) -> str | None:
    items = [r for r in results if r.category is not SourceCategory.UNDETERMINED]
    if not items:
        # Surface UNDETERMINED only when there's at least one — it's
        # an actionable "add more probes" signal.
        undetermined = [r for r in results if r.category is SourceCategory.UNDETERMINED]
        if not undetermined:
            return None
        names = ", ".join(r.value_name for r in undetermined)
        return f"[CONST-PROV] undetermined (need more axes/dataflow): {names}"
    parts = []
    for r in items:
        scope = f" scope={r.scope}" if r.scope and r.scope != "universal" else ""
        ceiling = f" ≤{r.evidence_class_ceiling}" if r.evidence_class_ceiling else ""
        parts.append(f"{r.value_name}: {r.category.value}{ceiling}{scope}")
    return "[CONST-PROV] " + "; ".join(parts)


__all__ = [
    "ConstantProvenanceConfig",
    "ConstantProvenanceResult",
    "DataflowReadKind",
    "DataflowSummary",
    "NOISE_READS",
    "RerunAnalysis",
    "RerunDimension",
    "RerunObservation",
    "RoutingAction",
    "SESSION_ENTROPY_READS",
    "SourceCategory",
    "analyse_reruns",
    "classify_from_dataflow_only",
    "classify_value",
    "classify_values_in_params",
    "render_constant_provenance_alert",
]
