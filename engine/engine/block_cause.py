"""Block-cause auto-classification + routing.

Problem this module exists to fix. When a node cannot be derived
from the loaded trace, three things were getting confused:

  * L1 mechanical work (discovery / instrument / replay / pipeline /
    structural-diff) was leaking out of the py layer and being
    handed to the agent to do by hand,
  * the agent was then escalating a class-1 collection gap ("we
    don't have the bytes") to the user as a three-way choice,
  * the user got prompts that should have been auto-resolved.

Job-chain alignment. Block causes split cleanly into three classes
and clark owns the first two:

  * **collection_gap** (class 1) — the trace doesn't cover the
    producing computation. Signals: ``phase_discovery.crosses_out``,
    zero mem-writes at the value's landing address, discovery chain
    terminates at a read pole, materialisation outside trace shadow.
    Auto-actions:
      - runner has the matching collection capability → emit an
        auto-collect spec, dispatch through the runner, queue a
        rerun request. The agent is never asked.
      - runner lacks the capability → append a backlog entry to
        ``<run_dir>/capability_backlog.jsonl`` and mark the node
        ``capability_gap_pending_collection``. Still never asks.
  * **recognition_gap / strategy_gap** (class 2) — data is
    collected but the pattern is unrecognised (→ L2) or a strategy
    switch is needed (→ L3). Routes through the existing
    methodology escalation paths, not the user.
  * **true_boundary** (class 3) — collected, parsed, symbolised,
    and still can't derive. Only this class escalates to the user,
    and only with clark-prepared :class:`DecisionElements` (what's
    still missing, the cost of collecting it, and a success
    probability estimate).

Independent toggle: ``UTOV_BLOCK_CAUSE=off|0|false|no``.

Auto-rerun fulfilment is deferred. The router emits a
:class:`RerunRequest` dispatch interface; the actual rerun loop
(runner-side capture → Core.run_pipeline replay) lands in the same
milestone as :class:`engine.phase_instrument.PhaseInstrumentSpec`'s
runner fulfilment. Until then, class-1 auto-collect cases write a
backlog entry instead of dispatching (capability_present path stays
testable through synthesized oracles).
"""

from __future__ import annotations

import enum
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, TYPE_CHECKING

from .phase import (
    ANCHOR_ADDR_FIRST_EXEC,
    ANCHOR_FUNC_ENTRY,
    ANCHOR_MEMREGION_FIRST_ACCESS,
    Anchor,
    PhaseBoundary,
)
from .phase_discovery import PhaseDiscoveryResult
from .phase_instrument import (
    PhaseInstrumentConfig,
    PhaseInstrumentSpec,
    suggest_instrument_for_boundary,
)

if TYPE_CHECKING:
    # Avoid a hard import — block_cause is imported by tests / wrappers
    # that don't always load the profile layer. The profile-driven
    # routing is opt-in (via the `routing_table` field); when absent,
    # block_cause works exactly as before.
    from engine.profile.routing_runtime import RoutingTable


# ---------------------------------------------------------------------------
# Anchor → capability name mapping.
# ---------------------------------------------------------------------------


# Anchor types describe *where* to hook; the matching capability
# name describes *what runner ability* is needed to fulfil the hook.
# Kept in a small registry so future anchor kinds (libload_done,
# thread_start) can land alongside their capability names.
ANCHOR_TO_CAPABILITY: dict[str, str] = {
    ANCHOR_FUNC_ENTRY:             "func_entry_hook",
    ANCHOR_ADDR_FIRST_EXEC:        "pc_first_exec_hook",
    ANCHOR_MEMREGION_FIRST_ACCESS: "memregion_watch",
}


def capability_for_anchor(anchor: Anchor | None) -> str:
    """Return the capability name needed to fulfil ``anchor``.
    Falls back to ``"unknown_capability"`` if the anchor is None or
    its type isn't in the registry."""
    if anchor is None:
        return "unknown_capability"
    return ANCHOR_TO_CAPABILITY.get(anchor.anchor_type, f"anchor:{anchor.anchor_type}")


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BlockCauseConfig:
    enabled: bool = True
    # Path the backlog writer appends to. ``None`` means "no
    # backlog persistence" — the router still produces backlog
    # entries in-memory so callers can inspect them, but nothing
    # lands on disk. Wrapper callers should set this to
    # ``<run_dir>/capability_backlog.jsonl``.
    backlog_path: Path | None = None
    # When True, the wrapper hides the raw phase_discovery /
    # phase_instrument_suggestions envelope siblings (their content
    # is fully redundant with block_cause for routing). Debug-only
    # callers flip UTOV_PHASE_DEBUG=1 to surface them again.
    hide_raw_phase_outputs: bool = True

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "BlockCauseConfig":
        src = env if env is not None else os.environ
        cfg = cls()
        flag = (src.get("UTOV_BLOCK_CAUSE") or "").strip().lower()
        if flag in ("off", "0", "false", "no"):
            cfg.enabled = False
        debug = (src.get("UTOV_PHASE_DEBUG") or "").strip().lower()
        if debug in ("1", "on", "true", "yes"):
            cfg.hide_raw_phase_outputs = False
        return cfg


# ---------------------------------------------------------------------------
# Class enum + signal evidence.
# ---------------------------------------------------------------------------


class BlockCauseClass(enum.Enum):
    """Cause classes — the routing fan-out keys."""

    COLLECTION_GAP   = "collection_gap"
    RECOGNITION_GAP  = "recognition_gap"
    STRATEGY_GAP     = "strategy_gap"
    TRUE_BOUNDARY    = "true_boundary"


class GapKind(enum.Enum):
    """Gap-kind realignment (v0.4.0 B7 / §19.9 base #5).

    Field experience (tc3) surfaced three distinct gap *kinds* that each
    demand a different routing action:

      * **capability_gap** — add runner ability (currently mapped from
        :attr:`BlockCauseClass.COLLECTION_GAP`).
      * **boundary_limit** — expand observation domain or accept the
        observed bound (currently mapped from
        :attr:`BlockCauseClass.TRUE_BOUNDARY`; only the user can authorise
        a wider observation domain or accept the boundary).
      * **analysis_incomplete** — escalate to a heavier analysis method
        (currently mapped from
        :attr:`BlockCauseClass.RECOGNITION_GAP` and
        :attr:`BlockCauseClass.STRATEGY_GAP`).

    The 4-way :class:`BlockCauseClass` enum is the *internal* fan-out
    key (the classifier produces one of those four), while the 3-way
    :class:`GapKind` is the *routing* vocabulary domain profiles use
    when they want to express policy by gap kind rather than by
    fine-grained class.  Both vocabularies share a single routing
    table on the active profile: a profile may declare either name and
    the router resolves it transparently.
    """

    CAPABILITY_GAP        = "capability_gap"
    BOUNDARY_LIMIT        = "boundary_limit"
    ANALYSIS_INCOMPLETE   = "analysis_incomplete"


_BLOCK_CAUSE_TO_GAP_KIND: dict[BlockCauseClass, GapKind] = {
    BlockCauseClass.COLLECTION_GAP:  GapKind.CAPABILITY_GAP,
    BlockCauseClass.RECOGNITION_GAP: GapKind.ANALYSIS_INCOMPLETE,
    BlockCauseClass.STRATEGY_GAP:    GapKind.ANALYSIS_INCOMPLETE,
    BlockCauseClass.TRUE_BOUNDARY:   GapKind.BOUNDARY_LIMIT,
}


def gap_kind_for(cls: BlockCauseClass) -> GapKind:
    """Resolve a 4-way :class:`BlockCauseClass` to its 3-way
    :class:`GapKind` equivalent."""
    return _BLOCK_CAUSE_TO_GAP_KIND[cls]


class RoutingAction(enum.Enum):
    """What clark does about the classified cause."""

    AUTO_COLLECT      = "auto_collect"        # runner has capability
    REGISTER_BACKLOG  = "register_backlog"    # capability missing
    ESCALATE_L2       = "escalate_l2"         # recognition gap
    ESCALATE_L3       = "escalate_l3"         # strategy gap
    ESCALATE_USER     = "escalate_user"       # true boundary


@dataclass(frozen=True, slots=True)
class BlockCauseSignal:
    """One observation that contributed to a class verdict."""

    name:     str
    evidence: str
    source:   str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "evidence": self.evidence, "source": self.source}


# ---------------------------------------------------------------------------
# NodeContext — the broader "what stages have already resolved this node"
# picture the classifier uses to distinguish class 1/2/3.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NodeContext:
    """Caller-supplied context for the classifier.

    Callers populate the booleans based on whether each stage has
    successfully processed the node. The classifier uses them to
    decide which level the node is stuck at:

      * ``data_collected=False``           → class-1 collection_gap
      * ``data_collected, ¬pattern_recognised`` → class-2 recognition_gap
      * ``pattern_recognised, ¬strategy_resolved`` → class-2 strategy_gap
      * all True, but caller still failed → class-3 true_boundary

    The default factory is "nothing resolved yet" so a caller that
    only has a phase_discovery result and no further context still
    gets a sensible class-1 verdict.
    """

    node_id:              str = ""
    data_collected:       bool = False
    pattern_recognised:   bool = False
    symbolised:           bool = False
    strategy_resolved:    bool = False
    # Caller's narrative for why the node is stuck — used as
    # explanatory text on the envelope.
    failure_summary:      str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id":             self.node_id,
            "data_collected":      self.data_collected,
            "pattern_recognised":  self.pattern_recognised,
            "symbolised":          self.symbolised,
            "strategy_resolved":   self.strategy_resolved,
            "failure_summary":     self.failure_summary,
        }


# ---------------------------------------------------------------------------
# Classification result.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BlockCauseClassification:
    cls:          BlockCauseClass
    signals:      tuple[BlockCauseSignal, ...] = ()
    reasoning:    str = ""
    node_context: NodeContext | None = None

    @property
    def gap_kind(self) -> GapKind:
        return gap_kind_for(self.cls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class":        self.cls.value,
            "gap_kind":     self.gap_kind.value,
            "signals":      [s.to_dict() for s in self.signals],
            "reasoning":    self.reasoning,
            "node_context": self.node_context.to_dict() if self.node_context else None,
        }


# ---------------------------------------------------------------------------
# Backlog entry + writer.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BacklogEntry:
    """One row in ``capability_backlog.jsonl``.

    Fields chosen for cross-run aggregation: a developer running
    ``cat work/*/capability_backlog.jsonl`` should be able to see
    what capabilities are missing across all targets, which nodes
    hit them, and roughly when.
    """

    gap_kind:           str          # e.g. "needs_collection_capability"
    missing_capability: str          # e.g. "memregion_watch"
    node_id:            str
    trigger_evidence:   dict[str, Any]
    suggested_spec:     dict[str, Any] | None = None
    timestamp:          float = 0.0
    run_dir:            str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap_kind":           self.gap_kind,
            "missing_capability": self.missing_capability,
            "node_id":            self.node_id,
            "trigger_evidence":   dict(self.trigger_evidence),
            "suggested_spec":     dict(self.suggested_spec) if self.suggested_spec else None,
            "timestamp":          self.timestamp,
            "run_dir":            self.run_dir,
        }


class BacklogWriter:
    """JSONL append-only sink for :class:`BacklogEntry` records."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path is not None else None

    def append(self, entry: BacklogEntry) -> BacklogEntry:
        if self.path is None:
            return entry
        # Make sure the parent exists; caller is responsible for
        # passing a path under <run_dir>.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")
        return entry

    def read_all(self) -> list[BacklogEntry]:
        if self.path is None or not self.path.exists():
            return []
        out: list[BacklogEntry] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                out.append(BacklogEntry(
                    gap_kind=raw["gap_kind"],
                    missing_capability=raw["missing_capability"],
                    node_id=raw["node_id"],
                    trigger_evidence=dict(raw.get("trigger_evidence") or {}),
                    suggested_spec=dict(raw["suggested_spec"]) if raw.get("suggested_spec") else None,
                    timestamp=float(raw.get("timestamp", 0.0)),
                    run_dir=str(raw.get("run_dir", "")),
                ))
        return out


# ---------------------------------------------------------------------------
# Decision elements for class-3 user escalation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DecisionElements:
    """The package clark hands to the user on class-3 (true boundary)
    escalation. Never empty for class-3.

    ``missing``   — what's still unresolved despite full L1 work
    ``补 cost``    — estimate of effort to collect what's missing
                    (runner cycles / wall time / LLM tokens — free
                    text, clark fills what it can estimate)
    ``success_probability`` — float in [0,1] with rationale; this
                    is the agent's calibrated guess based on what
                    L1 has already shown.
    ``options``   — distinct directions the user might pick. Each
                    is a short label + rationale. May be empty when
                    clark sees only one path.
    """

    missing:             tuple[str, ...]
    cost_estimate:       str = ""
    success_probability: float | None = None
    probability_basis:   str = ""
    options:             tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "missing":             list(self.missing),
            "cost_estimate":       self.cost_estimate,
            "success_probability": self.success_probability,
            "probability_basis":   self.probability_basis,
            "options":             [{"label": l, "rationale": r} for l, r in self.options],
        }


# ---------------------------------------------------------------------------
# Capability oracle protocol.
# ---------------------------------------------------------------------------


class CapabilityOracle(Protocol):
    """Anything that answers 'does the runner have capability X'."""

    def has(self, capability: str) -> bool: ...


@dataclass(slots=True)
class StaticCapabilityOracle:
    """In-memory oracle wrapping a static capability set. Useful
    for tests and as the default when no adapter is in play.
    ``metadata_override`` if present *wins* over ``static`` — that
    way a runner can opt-into capabilities it implements at
    project level without code changes on the engine side."""

    static:            frozenset[str] = frozenset()
    metadata_override: frozenset[str] | None = None

    def has(self, capability: str) -> bool:
        if self.metadata_override is not None and capability in self.metadata_override:
            return True
        return capability in self.static


def oracle_from_adapter(
    adapter: Any,
    *,
    metadata: Any | None = None,
) -> StaticCapabilityOracle:
    """Build an oracle from a RunnerAdapter-shaped object.

    The adapter is expected to expose a ``CAPABILITIES`` class attr
    (``frozenset[str]``) — the engine side's static declaration of
    what it knows the adapter implements. If ``metadata`` is given
    and carries a ``capabilities`` field (list[str]), the runner's
    runtime declaration overrides the static set."""
    static = getattr(adapter, "CAPABILITIES", None) or frozenset()
    if not isinstance(static, (set, frozenset)):
        static = frozenset(static)
    static = frozenset(static)

    override: frozenset[str] | None = None
    caps_raw = None
    if metadata is not None:
        caps_raw = getattr(metadata, "capabilities", None)
        if caps_raw is None and isinstance(metadata, dict):
            caps_raw = metadata.get("capabilities")
    if caps_raw:
        override = frozenset(str(c) for c in caps_raw)
    return StaticCapabilityOracle(static=static, metadata_override=override)


# ---------------------------------------------------------------------------
# Rerun request — dispatch interface for the deferred fulfilment.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RerunRequest:
    """Emitted by the router on class-1 auto-collect path. Signals
    'once the collection completes, feed the captured unit back to
    the pipeline starting at this entry'.

    Runtime fulfilment lands in the same milestone as runner-side
    :class:`engine.phase_instrument.PhaseInstrumentSpec` execution.
    Until then, the request is informational — callers receive it
    on the router result so they can plumb it later without
    re-shaping the contract.
    """

    instrument_spec:   PhaseInstrumentSpec
    triggered_by_node: str
    reason:            str

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument_spec":   self.instrument_spec.to_dict(),
            "triggered_by_node": self.triggered_by_node,
            "reason":            self.reason,
            "kind":              "rerun_request",
        }


# ---------------------------------------------------------------------------
# Router result.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RouterResult:
    """Output of :meth:`BlockCauseRouter.route` for one classified
    block. The action determines which of the optional fields is
    populated:

      AUTO_COLLECT     → ``rerun_request``
      REGISTER_BACKLOG → ``backlog_entry`` (written to disk if
                         writer.path is set)
      ESCALATE_L2 / L3 → ``escalation_hint`` (free-text)
      ESCALATE_USER    → ``decision_elements`` (always present;
                         user contact only happens with full
                         L1-prepared context)
    """

    classification:    BlockCauseClassification
    action:            RoutingAction
    rerun_request:     RerunRequest | None = None
    backlog_entry:     BacklogEntry | None = None
    escalation_hint:   str = ""
    decision_elements: DecisionElements | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification":    self.classification.to_dict(),
            "action":            self.action.value,
            "rerun_request":     self.rerun_request.to_dict() if self.rerun_request else None,
            "backlog_entry":     self.backlog_entry.to_dict() if self.backlog_entry else None,
            "escalation_hint":   self.escalation_hint,
            "decision_elements": self.decision_elements.to_dict() if self.decision_elements else None,
        }


# ---------------------------------------------------------------------------
# Classifier.
# ---------------------------------------------------------------------------


def classify(
    discovery: PhaseDiscoveryResult | None = None,
    *,
    node_context: NodeContext | None = None,
) -> BlockCauseClassification:
    """Classify the cause of a node-resolution failure.

    Inputs.
      ``discovery``    — the phase_discovery result for the node's
                         key value, if available. When ``crosses_out``
                         is True and the node hasn't collected data
                         yet, that pins class-1 immediately.
      ``node_context`` — broader caller context. When the data is
                         already collected, the discovery result is
                         irrelevant for class selection; the
                         remaining gap is L2 or L3 or true.

    The classifier never invents context — callers must pass
    ``node_context`` if they want class-2/3 verdicts. With only a
    ``discovery`` argument, the verdict is either class-1 (when
    discovery says crosses_out) or class-3 (when discovery shows
    the producer is fully in-window).
    """
    signals: list[BlockCauseSignal] = []
    ctx = node_context or NodeContext()

    # Phase 1. Collection gap takes precedence — if the trace doesn't
    # cover the producing computation, every higher-class
    # classification is moot.
    if discovery is not None and discovery.crosses_out:
        signals.append(BlockCauseSignal(
            name="phase_discovery_crosses_out",
            evidence=discovery.reason or "phase_discovery flagged source outside window",
            source="engine.phase_discovery",
        ))
        # Distinguish "no writer in any source" (strongest signal)
        # vs "out-of-window writer found via ledger".
        if "no_writer_in_any_source" in discovery.reason:
            signals.append(BlockCauseSignal(
                name="zero_writers_at_address",
                evidence=f"value_addr=0x{discovery.value_addr:x}",
                source="engine.phase_discovery",
            ))
        return BlockCauseClassification(
            cls=BlockCauseClass.COLLECTION_GAP,
            signals=tuple(signals),
            reasoning=(
                "phase_discovery reports the producing computation lives "
                "outside the loaded trace window; clark owns the auto-collect "
                "decision, not the agent."
            ),
            node_context=node_context,
        )

    # When caller gave node_context, the discovery flag wasn't set
    # but the higher-class structure may still be tagged.
    if not ctx.data_collected:
        # No discovery + no collection — caller signalled the data
        # isn't here. Treat as collection_gap with weaker evidence.
        signals.append(BlockCauseSignal(
            name="node_context.data_collected=False",
            evidence=ctx.failure_summary or "caller marked node data uncollected",
            source="caller",
        ))
        return BlockCauseClassification(
            cls=BlockCauseClass.COLLECTION_GAP,
            signals=tuple(signals),
            reasoning="node context indicates data is not yet collected.",
            node_context=node_context,
        )

    if not ctx.pattern_recognised:
        signals.append(BlockCauseSignal(
            name="node_context.pattern_recognised=False",
            evidence=ctx.failure_summary or "caller marked pattern unrecognised",
            source="caller",
        ))
        return BlockCauseClassification(
            cls=BlockCauseClass.RECOGNITION_GAP,
            signals=tuple(signals),
            reasoning="data is collected but layer-1/2 recognition has not converged.",
            node_context=node_context,
        )

    if not ctx.strategy_resolved:
        signals.append(BlockCauseSignal(
            name="node_context.strategy_resolved=False",
            evidence=ctx.failure_summary or "caller marked strategy unresolved",
            source="caller",
        ))
        return BlockCauseClassification(
            cls=BlockCauseClass.STRATEGY_GAP,
            signals=tuple(signals),
            reasoning="data is recognised but strategy-layer reasoning has not converged.",
            node_context=node_context,
        )

    # All L1 channels exhausted.
    signals.append(BlockCauseSignal(
        name="all_lower_layers_exhausted",
        evidence=(
            ctx.failure_summary
            or "data collected, pattern recognised, symbolised, strategy resolved"
        ),
        source="caller",
    ))
    return BlockCauseClassification(
        cls=BlockCauseClass.TRUE_BOUNDARY,
        signals=tuple(signals),
        reasoning=(
            "L1 / L2 / L3 channels have all run; the remaining ambiguity "
            "is a direction-decision worth surfacing to the user, with "
            "clark-prepared decision elements."
        ),
        node_context=node_context,
    )


# ---------------------------------------------------------------------------
# Router.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BlockCauseRouter:
    """Auto-router. Composes the classifier, capability oracle, and
    backlog writer into one entry point used by the wrapper.

    ``routing_table`` (optional) provides a profile-driven cause →
    action lookup (v0.3.0 profile layer). When set, the router asks
    the profile for the primary action per cause; when unset, falls
    back to the legacy hardcoded enum mapping below. Existing call
    sites that haven't migrated yet keep working unchanged.
    """

    oracle:         CapabilityOracle | None = None
    backlog_writer: BacklogWriter | None = None
    instrument_cfg: PhaseInstrumentConfig | None = None
    run_dir:        str = ""
    routing_table:  "RoutingTable | None" = None

    # collection_gap has a capability-aware fork (auto_collect vs
    # register_backlog) handled in _route_collection_gap; the
    # remaining three causes go through _action_for with their legacy
    # default supplied inline at the call site.

    def _action_for(
        self,
        cause: BlockCauseClass,
        legacy: RoutingAction,
    ) -> RoutingAction:
        """Resolve cause → action. Consult the profile's RoutingTable
        when present; fall back to ``legacy`` otherwise. If the profile
        declares an action id the RoutingAction enum doesn't recognise,
        fall back to ``legacy`` rather than crashing — the profile is
        the policy layer, the enum is the executor's vocabulary.

        Profile authors may state routing by the fine-grained
        :class:`BlockCauseClass` value (``recognition_gap`` →
        ``escalate_l2``) or by the broader :class:`GapKind`
        (``analysis_incomplete`` → ``escalate_l3``). When the
        fine-grained name is absent from the table, the router falls
        through to the gap-kind name before defaulting (v0.4.0 B7 /
        §19.9 base #5).
        """
        if self.routing_table is None:
            return legacy
        action_id = self.routing_table.primary_action(cause.value)
        if action_id is None:
            # Fall back to the 3-way gap-kind name before defaulting.
            action_id = self.routing_table.primary_action(
                gap_kind_for(cause).value
            )
        if action_id is None:
            return legacy
        try:
            return RoutingAction(action_id)
        except ValueError:
            return legacy

    def route(
        self,
        discovery: PhaseDiscoveryResult | None = None,
        *,
        node_context: NodeContext | None = None,
    ) -> RouterResult:
        cls_result = classify(discovery, node_context=node_context)

        if cls_result.cls is BlockCauseClass.COLLECTION_GAP:
            return self._route_collection_gap(cls_result, discovery)
        if cls_result.cls is BlockCauseClass.RECOGNITION_GAP:
            return RouterResult(
                classification=cls_result,
                action=self._action_for(
                    BlockCauseClass.RECOGNITION_GAP, RoutingAction.ESCALATE_L2
                ),
                escalation_hint=(
                    "data is collected — escalate to layer-2 recognition "
                    "(template fit / handler classification) before the agent."
                ),
            )
        if cls_result.cls is BlockCauseClass.STRATEGY_GAP:
            return RouterResult(
                classification=cls_result,
                action=self._action_for(
                    BlockCauseClass.STRATEGY_GAP, RoutingAction.ESCALATE_L3
                ),
                escalation_hint=(
                    "pattern is recognised — escalate to layer-3 strategy "
                    "(paradigm switch / cross-node reasoning) via the agent."
                ),
            )
        # TRUE_BOUNDARY
        return RouterResult(
            classification=cls_result,
            action=self._action_for(
                BlockCauseClass.TRUE_BOUNDARY, RoutingAction.ESCALATE_USER
            ),
            decision_elements=self._build_decision_elements(cls_result, discovery),
        )

    def _route_collection_gap(
        self,
        cls_result: BlockCauseClassification,
        discovery: PhaseDiscoveryResult | None,
    ) -> RouterResult:
        # Need a boundary + anchor to know what capability is required.
        boundary: PhaseBoundary | None = (
            discovery.boundary if discovery is not None else None
        )
        anchor: Anchor | None = boundary.anchor if boundary else None
        capability_name = capability_for_anchor(anchor)
        node_id = (
            (cls_result.node_context.node_id if cls_result.node_context else "")
            or (f"value@0x{discovery.value_addr:x}" if discovery is not None else "")
            or "<unknown_node>"
        )

        sugg = (
            suggest_instrument_for_boundary(boundary, cfg=self.instrument_cfg)
            if boundary is not None else None
        )
        spec: PhaseInstrumentSpec | None = sugg.spec if sugg else None

        capability_present = (
            self.oracle is not None and self.oracle.has(capability_name)
        )

        if capability_present and spec is not None:
            return RouterResult(
                classification=cls_result,
                action=RoutingAction.AUTO_COLLECT,
                rerun_request=RerunRequest(
                    instrument_spec=spec,
                    triggered_by_node=node_id,
                    reason=(
                        f"class-1 collection_gap; runner has {capability_name}; "
                        f"auto-trigger collection and queue rerun on the captured unit."
                    ),
                ),
            )

        # capability missing → backlog. We still attach the suggested
        # spec so a developer reading the backlog knows what shape
        # the runner would need to fulfil.
        evidence: dict[str, Any] = {}
        if discovery is not None:
            evidence["value_addr_hex"] = f"0x{discovery.value_addr:x}"
            evidence["reason"]         = discovery.reason
            if boundary is not None:
                evidence["boundary"] = boundary.to_dict()
        if cls_result.node_context is not None:
            evidence["failure_summary"] = cls_result.node_context.failure_summary
        entry = BacklogEntry(
            gap_kind="needs_collection_capability",
            missing_capability=capability_name,
            node_id=node_id,
            trigger_evidence=evidence,
            suggested_spec=spec.to_dict() if spec is not None else None,
            timestamp=time.time(),
            run_dir=self.run_dir,
        )
        if self.backlog_writer is not None:
            self.backlog_writer.append(entry)
        return RouterResult(
            classification=cls_result,
            action=RoutingAction.REGISTER_BACKLOG,
            backlog_entry=entry,
        )

    def _build_decision_elements(
        self,
        cls_result: BlockCauseClassification,
        discovery: PhaseDiscoveryResult | None,
    ) -> DecisionElements:
        ctx = cls_result.node_context
        missing: list[str] = []
        if ctx and ctx.failure_summary:
            missing.append(ctx.failure_summary)
        if discovery is not None and discovery.boundary is not None:
            b = discovery.boundary
            if b.region is not None:
                missing.append(f"closed-form recompute for region 0x{b.region[0]:x}+{b.region[1]}")
            if b.pc_range is not None:
                missing.append(f"semantics for pc_range [0x{b.pc_range[0]:x}, 0x{b.pc_range[1]:x})")
        # Cost / probability are estimates clark only fills when it
        # has signal. For now the router exposes the slot; the
        # wrapper or a richer caller (e.g. with verifier-history) is
        # responsible for filling in numbers.
        return DecisionElements(
            missing=tuple(missing),
            cost_estimate="unspecified — caller did not supply estimate",
            success_probability=None,
            probability_basis="caller did not supply probability basis",
            options=(),
        )


# ---------------------------------------------------------------------------
# Wrapper-side helper: batch-route a list of discovery results.
# ---------------------------------------------------------------------------


def route_discovery_batch(
    router: BlockCauseRouter,
    results: Iterable[PhaseDiscoveryResult],
) -> list[RouterResult]:
    """Run the router on each discovery result. Used by the
    discipline wrapper to consolidate the L1 routing in one place."""
    return [router.route(r) for r in results]


def render_block_cause_alert(results: Iterable[RouterResult]) -> str | None:
    items = list(results)
    if not items:
        return None
    parts = []
    for r in items:
        node = (
            r.classification.node_context.node_id
            if r.classification.node_context else ""
        )
        if not node and r.backlog_entry is not None:
            node = r.backlog_entry.node_id
        if not node and r.rerun_request is not None:
            node = r.rerun_request.triggered_by_node
        parts.append(
            f"{r.classification.cls.value} → {r.action.value}"
            + (f" [{node}]" if node else "")
        )
    return "[BLOCK-CAUSE] " + "; ".join(parts)


__all__ = [
    "ANCHOR_TO_CAPABILITY",
    "BacklogEntry",
    "BacklogWriter",
    "BlockCauseClass",
    "BlockCauseClassification",
    "BlockCauseConfig",
    "BlockCauseRouter",
    "BlockCauseSignal",
    "CapabilityOracle",
    "DecisionElements",
    "NodeContext",
    "RerunRequest",
    "RouterResult",
    "RoutingAction",
    "StaticCapabilityOracle",
    "capability_for_anchor",
    "classify",
    "oracle_from_adapter",
    "render_block_cause_alert",
    "route_discovery_batch",
]
