"""Discipline wrapper — turns the methodology catalog into runtime hooks
around the agent_mode RPC dispatch.

The wrapper does four things, in order, on every tool call:

  1. **Pre-check (interception).** Some methods carry an inherent
     risk (verifier bypass count, un-ledgered data reference, forbidden
     keyword in `reason`). If a violation fires, return a refusal
     envelope WITHOUT calling the underlying dispatch.
  2. **Dispatch the original handler.** Stays unchanged.
  3. **Context-sensitive prompts.** Read the result + method + params,
     attach reverse questions (evidence_class on verdict, evidence_class
     on high-rate success, etc.).
  4. **Footer + (every Nth step) full card.** Always present.

The wrapper writes its output as a sibling of ``result`` in the JSON-RPC
envelope, NOT inside ``result`` — so existing parsers keep working.

Killable: ``UTOV_METHODOLOGY=off`` (or pass ``MethodologyConfig(enabled=
False)``) short-circuits everything; envelope is unchanged.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable

from .m1_success_audit import (
    M1AuditConfig,
    SuccessAuditResult,
    apply_audit_to_params,
    audit_success_claim,
    render_audit_alert,
)
from .m3_bypass_block import (
    BypassBlockDetector,
    BypassDetection,
    M3BypassConfig,
    extract_attempt as m3_extract_attempt,
    render_bypass_alert,
)
from .length_chain_check import (
    LengthChainConfig,
    check_chains_in_params,
    render_length_chain_alert,
)
from .value_provenance import (
    ValueProvenanceConfig,
    render_provenance_alert,
    tag_values_in_params,
)
from .watch_first_write import (
    WatchFirstWriteConfig,
    render_watch_suggestion_alert,
    suggest_watches_in_params,
)
from .phase_discovery import (
    DiscoveryDataSource,
    PhaseDiscoveryConfig,
    discover_phases_in_params,
    render_phase_discovery_alert,
)
from .phase_instrument import (
    PhaseInstrumentConfig,
    render_phase_instrument_alert,
    suggest_instruments_for_results,
)
from .block_cause import (
    BlockCauseConfig,
    BlockCauseRouter,
    RouterResult,
    render_block_cause_alert,
    route_discovery_batch,
)
from .constant_provenance import (
    ConstantProvenanceConfig,
    classify_values_in_params,
    render_constant_provenance_alert,
)
from .methodology import (
    MethodologyConfig,
    MethodologyState,
    render_alert,
    render_footer,
    render_periodic_card,
    render_prompt,
)


# ---------------------------------------------------------------------------
# Wrapper output
# ---------------------------------------------------------------------------


@dataclass
class DisciplineEnvelope:
    """The wrapper's add-on payload. ``intercepted`` is True iff the
    underlying dispatch was NOT called (refusal). The caller renders
    this as a sibling of ``result`` in the JSON-RPC response, or
    inserts ``error`` when ``intercepted=True``.
    """
    footer: str
    card: str | None = None
    prompts: list[str] = field(default_factory=list)
    alerts:  list[str] = field(default_factory=list)
    intercepted: bool = False
    intercepted_reason: str | None = None
    # M1 success-audit block. Populated when the audit gate fires —
    # i.e. the call carried a positive target_success / archival_allowed
    # claim. Shape matches ``SuccessAuditResult.to_dict``.
    m1_audit: dict[str, Any] | None = None
    # M3 bypass-block detection. Populated when this call either
    # triggered the bypass threshold for a block OR was refused as a
    # follow-up observation on an already-confirmed bypass block.
    # Shape matches ``BypassDetection.to_dict``.
    m3_bypass: dict[str, Any] | None = None
    # Value-provenance tags produced this call. List of
    # ``ValueProvenanceResult.to_dict()``.
    value_provenance: list[dict[str, Any]] = field(default_factory=list)
    # Watch-first-write suggestions for observed values whose
    # producer is still unknown. List of ``WatchSuggestion.to_dict()``.
    watch_suggestions: list[dict[str, Any]] = field(default_factory=list)
    # Length-chain consistency reports. List of
    # ``LengthChainResult.to_dict()``.
    length_chain: list[dict[str, Any]] = field(default_factory=list)
    # Phase discovery — value records whose producing phase was
    # located outside the current trace window. List of
    # ``PhaseDiscoveryResult.to_dict()`` (only the crossing-out cases;
    # in-window producers are not surfaced).
    phase_discovery: list[dict[str, Any]] = field(default_factory=list)
    # Phase instrument suggestions emitted in response to phase
    # discovery. List of ``PhaseInstrumentSuggestion.to_dict()``.
    phase_instrument_suggestions: list[dict[str, Any]] = field(default_factory=list)
    # Block-cause routing — the L1 routing conclusion for each
    # crossing-out phase_discovery result. List of
    # ``RouterResult.to_dict()``. This is the *authoritative* output
    # for downstream consumers; ``phase_discovery`` /
    # ``phase_instrument_suggestions`` are L1 intermediates and
    # remain hidden unless UTOV_PHASE_DEBUG=1.
    block_cause: list[dict[str, Any]] = field(default_factory=list)
    # Constant-provenance verdicts — one entry per value record on
    # the call that carried rerun_observations and/or
    # producer_dataflow. List of ``ConstantProvenanceResult.to_dict()``.
    constant_provenance: list[dict[str, Any]] = field(default_factory=list)
    # v0.3.0 profile layer (PLAN §19) — when the wrapper is
    # constructed with a profile, every envelope advertises which
    # profile the engine is running under so agents / loggers /
    # downstream tools can branch on it. Shape: ``{name, chain}``.
    # ``None`` (legacy callers) → key omitted from ``to_dict``.
    profile: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"footer": self.footer}
        if self.card is not None:
            out["card"] = self.card
        if self.prompts:
            out["prompts"] = list(self.prompts)
        if self.alerts:
            out["alerts"] = list(self.alerts)
        if self.intercepted:
            out["intercepted"] = True
            out["intercepted_reason"] = self.intercepted_reason
        if self.m1_audit is not None:
            out["m1_audit"] = self.m1_audit
        if self.m3_bypass is not None:
            out["m3_bypass"] = self.m3_bypass
        if self.value_provenance:
            out["value_provenance"] = list(self.value_provenance)
        if self.watch_suggestions:
            out["watch_suggestions"] = list(self.watch_suggestions)
        if self.length_chain:
            out["length_chain"] = list(self.length_chain)
        if self.phase_discovery:
            out["phase_discovery"] = list(self.phase_discovery)
        if self.phase_instrument_suggestions:
            out["phase_instrument_suggestions"] = list(self.phase_instrument_suggestions)
        if self.block_cause:
            out["block_cause"] = list(self.block_cause)
        if self.constant_provenance:
            out["constant_provenance"] = list(self.constant_provenance)
        if self.profile is not None:
            out["profile"] = dict(self.profile)
        return out


# ---------------------------------------------------------------------------
# Core wrapper
# ---------------------------------------------------------------------------


class DisciplineWrapper:
    """Per-session state machine + dispatch decorator."""

    def __init__(
        self,
        *,
        core: Any | None = None,
        config: MethodologyConfig | None = None,
        state:  MethodologyState  | None = None,
        m1_audit_config: M1AuditConfig | None = None,
        m3_bypass_config: M3BypassConfig | None = None,
        m3_bypass_detector: BypassBlockDetector | None = None,
        phase_discovery_config: PhaseDiscoveryConfig | None = None,
        phase_instrument_config: PhaseInstrumentConfig | None = None,
        phase_discovery_source_provider: Callable[
            [Any, str, dict[str, Any]], DiscoveryDataSource | None
        ] | None = None,
        block_cause_config: BlockCauseConfig | None = None,
        block_cause_router: BlockCauseRouter | None = None,
        constant_provenance_config: ConstantProvenanceConfig | None = None,
        profile: Any | None = None,
    ):
        self.core   = core
        self.config = config or MethodologyConfig.from_env()
        self.state  = state  or MethodologyState()
        # M1 success-audit gate. Independent toggle so we can debug
        # the gate in isolation (UTOV_M1_AUDIT=off) without flipping
        # the larger UTOV_METHODOLOGY switch.
        self.m1_audit_config = m1_audit_config or M1AuditConfig.from_env()
        # Last audit produced by _maybe_intercept; consumed during
        # envelope assembly to surface the audit block + alert to the
        # agent. Reset per-call.
        self._latest_m1_audit: SuccessAuditResult | None = None
        # M3 bypass-block gate. Stateful: tracks cross-call evidence
        # so that ≥N distinct observation methods all reporting
        # variability=0 on the same block flips it to suspected_bypass
        # and refuses any follow-up observation attempt. Independent
        # toggle: UTOV_M3_BYPASS=off.
        cfg3 = m3_bypass_config or M3BypassConfig.from_env()
        self.m3_bypass_detector = m3_bypass_detector or BypassBlockDetector(cfg3)
        self._latest_m3_detection: BypassDetection | None = None
        # Three M1/M3-style primitives — each independently toggleable.
        # Value-source state machine: hook/dump-sourced values are
        # auto-capped at evidence_class B until a closed-form
        # recompute is verified.
        self.value_provenance_config = ValueProvenanceConfig.from_env()
        self._latest_value_provenance: list[Any] = []
        # Watch-first-write auto-suggestion: observed values with a
        # concrete landing address get an M3-style follow-up note
        # offering a memory watchpoint on the producing PC.
        self.watch_first_write_config = WatchFirstWriteConfig.from_env()
        self._latest_watch_suggestions: list[Any] = []
        # Length-chain consistency: adjacent nodes on a declared
        # transformation chain must have an explainable length
        # relation; otherwise the edge is flagged unexplained.
        self.length_chain_config = LengthChainConfig.from_env()
        self._latest_length_chain: list[Any] = []
        # Phase observation — three primitive set: discovery /
        # instrument / replay. Discovery is the only one that runs
        # automatically per-call; instrument is auto-suggested off
        # discovery output; replay is invoked explicitly by callers
        # consuming runner-side results. UTOV_PHASE_DISCOVERY /
        # UTOV_PHASE_INSTRUMENT toggle the auto behaviors
        # independently.
        self.phase_discovery_config = (
            phase_discovery_config or PhaseDiscoveryConfig.from_env()
        )
        self.phase_instrument_config = (
            phase_instrument_config or PhaseInstrumentConfig.from_env()
        )
        # Optional factory: given (core, method, params), return a
        # DiscoveryDataSource — or None to skip auto-discovery on this
        # call. Callers wire this when their core has a way to
        # surface the loaded trace window (e.g. a Core object holding
        # a recent ``read_trace_window`` slice). Default None means
        # discovery is RPC-only, not auto.
        self.phase_discovery_source_provider = phase_discovery_source_provider
        self._latest_phase_discovery: list[Any] = []
        self._latest_phase_instrument_suggestions: list[Any] = []
        # Block-cause router. When set, consumes
        # ``phase_discovery`` results and produces the L1 routing
        # conclusion that *is* surfaced to the envelope; the raw
        # phase outputs are kept on the wrapper but hidden from the
        # envelope unless UTOV_PHASE_DEBUG=1 (cfg.hide_raw_phase_outputs
        # = False). When the router is None, the wrapper falls back
        # to the legacy behavior of surfacing raw phase results —
        # this preserves explicit-router-opt-in for callers that
        # haven't migrated.
        self.block_cause_config = block_cause_config or BlockCauseConfig.from_env()
        self.block_cause_router = block_cause_router
        self._latest_block_cause: list[RouterResult] = []
        # Constant-provenance — deterministic classifier for value
        # records carrying rerun_observations / producer_dataflow.
        # Fires every call (no source-provider gate); the values
        # opt-in by including the relevant fields on the record.
        # Independent toggle: UTOV_CONSTANT_PROVENANCE.
        self.constant_provenance_config = (
            constant_provenance_config or ConstantProvenanceConfig.from_env()
        )
        self._latest_constant_provenance: list[Any] = []
        # v0.3.0 profile layer (PLAN §19) — when a MergedProfile is
        # supplied, every envelope advertises its name + chain.
        # The wrapper's existing dispatch logic does NOT consume the
        # profile yet; the migration of M1/M3/etc. lookups to go
        # through ``profile.probes`` is a future step. Step 8's job is
        # to make the wire reachable.
        self.profile = profile

    # -- public entry point -------------------------------------------------

    def step(
        self,
        method: str,
        params: dict[str, Any] | None,
        dispatch_fn: Callable[[str, dict[str, Any]], Any],
    ) -> tuple[Any, DisciplineEnvelope]:
        """Run one tool call through the discipline wrapper.

        Returns ``(result, envelope)`` where ``result`` is whatever the
        original ``dispatch_fn`` returned (or ``None`` when intercepted).
        """
        params = params or {}
        if not self.config.enabled:
            result = dispatch_fn(method, params)
            return result, DisciplineEnvelope(footer="")

        self.state.step_count += 1
        self.state.steps_since_checkpoint += 1
        self.state.push_op(method, self.config.recent_ops_window)

        # 1. pre-check / interception ------------------------------------
        intercept_reason = self._maybe_intercept(method, params)
        if intercept_reason is not None:
            env = self._build_envelope(method, params, result=None,
                                       extra_alerts=[intercept_reason])
            env.intercepted = True
            env.intercepted_reason = intercept_reason
            return None, env

        # 2. dispatch ----------------------------------------------------
        try:
            result = dispatch_fn(method, params)
        except Exception:
            self.state.push_failure(method, self.config.recent_ops_window)
            env = self._build_envelope(method, params, result=None)
            env.alerts.extend(self._post_dispatch_alerts(method, params, None))
            # Re-raise after building the envelope so the serve loop can
            # encode it and the next-step state is still updated.
            self._maybe_book_periodic_card(env)
            raise DisciplineRaise(env) from None

        # checkpoint methods reset failure streaks
        if method in ("checkpoint", "rerun_from_stage"):
            self.state.reset_failures()
            if method == "rerun_from_stage":
                self.state.backtrack_count += 1

        # 2b. M3 bypass-block evidence aggregation --------------------
        # If this dispatch was an M3 variability check, feed the
        # (block_id, observation_method, failed) tuple to the
        # detector. A return value means the threshold was crossed
        # *on this call* and the block has been flipped to bypass.
        self._record_m3_observation(method, params, result)

        # 3. context-sensitive prompts + 4. footer/card ------------------
        env = self._build_envelope(method, params, result=result)
        return result, env

    # -- M3 bypass-block aggregation --------------------------------------

    def _record_m3_observation(
        self,
        method: str,
        params: dict[str, Any],
        result: Any,
    ) -> None:
        attempt = m3_extract_attempt(
            method, params, result, cfg=self.m3_bypass_detector.cfg,
        )
        if attempt is None:
            return
        block_id, observation_method, failed = attempt
        if failed:
            detection = self.m3_bypass_detector.record_attempt(
                block_id, observation_method,
                failed=True,
                note=f"method={method}",
            )
            if detection is not None:
                self._latest_m3_detection = detection
        else:
            # A successful variability observation supersedes prior
            # records for (block, method) but doesn't clear other
            # methods' failure history — the threshold is on
            # *distinct* methods, so a passing observation on one
            # method is still informative.
            self.m3_bypass_detector.record_attempt(
                block_id, observation_method,
                failed=False,
                note=f"method={method}",
            )

    # -- intercept ---------------------------------------------------------

    def _maybe_intercept(
        self,
        method: str,
        params: dict[str, Any],
    ) -> str | None:
        # Reset per-call audit slots.
        self._latest_m1_audit = None
        self._latest_m3_detection = None
        self._latest_value_provenance = []
        self._latest_watch_suggestions = []
        self._latest_length_chain = []
        self._latest_phase_discovery = []
        self._latest_phase_instrument_suggestions = []
        self._latest_block_cause = []
        self._latest_constant_provenance = []

        # (a0) M3 bypass-block follow-up refusal. If the params name a
        # block already flagged suspected_bypass and the method is on
        # the observation whitelist, refuse before dispatch — this is
        # the "stop changing posture on a dead block" rule.
        m3_cfg = self.m3_bypass_detector.cfg
        if (
            m3_cfg.enabled
            and method in m3_cfg.intercept_methods
            and isinstance(params, dict)
        ):
            block_id = params.get("block_id")
            if not isinstance(block_id, str):
                # Tolerate nested {report:{block_id:...}} shape.
                from .m3_bypass_block import _first
                block_id = _first(params, "block_id")
            if isinstance(block_id, str) and self.m3_bypass_detector.is_known_bypass(block_id):
                obs = params.get("observation_method") if isinstance(params, dict) else None
                if not isinstance(obs, str):
                    obs = method
                followup = self.m3_bypass_detector.intercept_followup(block_id, obs)
                if followup is not None:
                    self._latest_m3_detection = followup
                    return followup.intercepted_reason or render_bypass_alert(followup)

        # (a) un-ledgered data reference. Payloads / params that carry
        # explicit experiment provenance must NOT enter promotion paths.
        if method in ("promote_to_finding", "inject_finding",
                      "submit_hypothesis", "override_verdict"):
            blob = _flatten_strings(params)
            if _looks_like_unpromoted(params, blob):
                reason = params.get("reason") if isinstance(params, dict) else None
                if not (isinstance(reason, str) and "--allow-unpromoted" in reason):
                    return render_alert("unledgered_reference", method=method)

        # (b) forbidden keyword in reason. Match before dispatch so the
        # agent never gets the side-effect.
        reason = params.get("reason") if isinstance(params, dict) else None
        if isinstance(reason, str):
            for kw in self.config.forbidden_keywords:
                if kw in reason:
                    return render_alert("forbidden_keyword", keyword=kw)

        # (b1) Value-source tagging. Walks params for value records
        # and caps the evidence_class on any hook/dump-sourced value
        # whose closed-form recompute is unverified.
        if isinstance(params, dict):
            self._latest_value_provenance = tag_values_in_params(
                params, cfg=self.value_provenance_config,
            )

            # (b2) Watch-first-write auto-suggestion. Operates on the
            # records *after* provenance tagging so the observed/
            # closed_form decision is already made.
            self._latest_watch_suggestions = suggest_watches_in_params(
                params, cfg=self.watch_first_write_config,
            )

            # (b3) Length-chain consistency on any declared
            # `length_chain` lists in params.
            self._latest_length_chain = check_chains_in_params(
                params, cfg=self.length_chain_config,
            )

            # (b3a) Constant-provenance — classify any value record
            # carrying rerun_observations and/or producer_dataflow.
            # Generalises M3 (per-input axis) and the M1 dimension-
            # variability check; auto-sets evidence_class ceiling
            # and recommended_action for downstream routing.
            self._latest_constant_provenance = classify_values_in_params(
                params, cfg=self.constant_provenance_config,
            )

            # (b4) Phase discovery — only runs when a source provider
            # is wired AND it produces a source for this call. Pure
            # opt-in: by default, discovery is reachable as an
            # explicit RPC but the wrapper itself does not auto-walk
            # producer chains. Producers can plug a source provider
            # to enable auto-discovery on value records carrying a
            # landing_address.
            if (
                self.phase_discovery_config.enabled
                and self.phase_discovery_source_provider is not None
            ):
                try:
                    src = self.phase_discovery_source_provider(
                        self.core, method, params,
                    )
                except Exception:
                    src = None
                if src is not None:
                    self._latest_phase_discovery = discover_phases_in_params(
                        params, src, cfg=self.phase_discovery_config,
                    )
                    if self._latest_phase_discovery:
                        # (b5) Phase instrument auto-suggestion — only
                        # fires off discovery results. Independent
                        # toggle: UTOV_PHASE_INSTRUMENT.
                        self._latest_phase_instrument_suggestions = (
                            suggest_instruments_for_results(
                                self._latest_phase_discovery,
                                cfg=self.phase_instrument_config,
                            )
                        )
                        # (b6) Block-cause routing. Classify each
                        # crossing-out result and route it (auto-
                        # collect / backlog / L2-L3 / user). The L1
                        # routing conclusion is what surfaces on
                        # the envelope — the raw phase outputs are
                        # hidden by default (UTOV_PHASE_DEBUG=1 to
                        # unhide). Routing only runs when an explicit
                        # router is configured; without it the
                        # legacy raw-surfacing path stays for
                        # backwards compatibility.
                        if (
                            self.block_cause_config.enabled
                            and self.block_cause_router is not None
                        ):
                            self._latest_block_cause = route_discovery_batch(
                                self.block_cause_router,
                                self._latest_phase_discovery,
                            )

        # (c) M1 success-audit gate. Fires when params declare a
        # positive target_success / archival_allowed, OR the method is
        # a known archival surface with a claim somewhere in its body.
        audit = audit_success_claim(
            method, params, cfg=self.m1_audit_config,
        )
        if audit is not None:
            self._latest_m1_audit = audit
            if audit.action == "reject":
                # Annotate so downstream audit trails can see *why*,
                # then refuse — the wrapper turns this into a JSON-RPC
                # error and the underlying dispatch never runs.
                apply_audit_to_params(params, audit)
                return audit.intercepted_reason or render_audit_alert(audit)
            if audit.action == "downgrade":
                # Rewrite params before dispatch so the archival path
                # sees target_success=False / strong_partial. Dispatch
                # still proceeds; the agent learns of the downgrade
                # via the envelope alert below.
                apply_audit_to_params(params, audit)
            else:  # allow
                apply_audit_to_params(params, audit)

        return None

    # -- post-dispatch alerts ---------------------------------------------

    def _post_dispatch_alerts(
        self,
        method: str,
        params: dict[str, Any],
        result: Any,
    ) -> list[str]:
        alerts: list[str] = []

        # verifier-bypass counter
        if method in self.config.bypass_methods:
            self.state.bypass_count += 1
            if self.state.bypass_count >= self.config.bypass_alert_threshold:
                alerts.append(render_alert(
                    "verifier_bypass", count=self.state.bypass_count,
                ))

        # too long since last checkpoint AND failures accumulated
        if (self.state.steps_since_checkpoint
                >= self.config.steps_since_checkpoint_warn
            and self.state.failures_since_checkpoint
                >= self.config.failure_streak_threshold):
            alerts.append(render_alert(
                "no_recent_checkpoint",
                steps=self.state.steps_since_checkpoint,
                failures=self.state.failures_since_checkpoint,
            ))

        return alerts

    # -- prompts -----------------------------------------------------------

    def _build_prompts(
        self,
        method: str,
        params: dict[str, Any],
        result: Any,
    ) -> list[str]:
        prompts: list[str] = []

        # verdict-producing op
        if method in self.config.verdict_methods:
            prompts.append(render_prompt("produced_verdict"))

        # numeric result with a high success rate
        if method in self.config.numeric_result_methods and isinstance(result, dict):
            if _contains_high_rate(result, floor=self.config.high_success_floor):
                prompts.append(render_prompt("high_number_success"))

        # contradiction surfaced in the result (invariants module wrote
        # `invariants_failed`, or result carries `contradicts_finding_id`)
        contradict_id = _first_contradiction_id(result)
        if contradict_id is not None:
            prompts.append(render_prompt(
                "contradicts_finding", finding_id=contradict_id,
            ))

        # repeated same-type failures
        if _recent_streak_same_type(
            self.state.recent_failures,
            threshold=self.config.failure_streak_threshold,
        ):
            prompts.append(render_prompt("repeated_failures"))

        # multi-candidate detection — explicit param key the
        # surrounding pipeline can opt into.
        if isinstance(params, dict) and params.get("multi_candidate"):
            prompts.append(render_prompt("multi_candidate"))

        # method outside utov whitelist — `static_tool` invocations
        # whose tool name isn't an in-utov capability.
        if method == "static_tool":
            tool = (params or {}).get("tool") or ""
            if _looks_non_utov(tool):
                prompts.append(render_prompt("non_utov_path"))

        return prompts

    # -- envelope assembly -------------------------------------------------

    def _build_envelope(
        self,
        method: str,
        params: dict[str, Any],
        *,
        result: Any,
        extra_alerts: list[str] | None = None,
    ) -> DisciplineEnvelope:
        prompts = self._build_prompts(method, params, result)
        alerts  = list(extra_alerts or [])
        if result is not None:
            alerts.extend(self._post_dispatch_alerts(method, params, result))
        # Surface the M1 audit outcome (A allow / B downgrade / C reject)
        # on every envelope where the gate fired.
        audit = self._latest_m1_audit
        env = DisciplineEnvelope(
            footer=render_footer(method),
            prompts=prompts,
            alerts=alerts,
        )
        if self.profile is not None:
            env.profile = {
                "name":  self.profile.name,
                "chain": list(self.profile.chain),
            }
        if audit is not None:
            audit_line = render_audit_alert(audit)
            if audit_line and audit_line not in env.alerts:
                env.alerts.append(audit_line)
            env.m1_audit = audit.to_dict()
        detection = self._latest_m3_detection
        if detection is not None:
            alert_line = render_bypass_alert(detection)
            if alert_line and alert_line not in env.alerts:
                env.alerts.append(alert_line)
            env.m3_bypass = detection.to_dict()
        # Value provenance — tag downgrades / parity disclaimers.
        if self._latest_value_provenance:
            env.value_provenance = [
                r.to_dict() for r in self._latest_value_provenance
            ]
            line = render_provenance_alert(self._latest_value_provenance)
            if line and line not in env.alerts:
                env.alerts.append(line)
        # Watch-first-write — emit suggestions/specs on observed values.
        if self._latest_watch_suggestions:
            env.watch_suggestions = [
                s.to_dict() for s in self._latest_watch_suggestions
            ]
            line = render_watch_suggestion_alert(self._latest_watch_suggestions)
            if line and line not in env.alerts:
                env.alerts.append(line)
        # Length-chain — flag unexplained edges.
        if self._latest_length_chain:
            env.length_chain = [
                r.to_dict() for r in self._latest_length_chain
            ]
            line = render_length_chain_alert(self._latest_length_chain)
            if line and line not in env.alerts:
                env.alerts.append(line)
        # Block-cause routing is the L1 routing conclusion and the
        # *authoritative* envelope surface for phase-related work.
        # When a router is configured, the raw phase outputs are
        # treated as intermediates and hidden by default — they're
        # restored alongside block_cause when UTOV_PHASE_DEBUG=1
        # (cfg.hide_raw_phase_outputs=False).
        router_active = (
            self.block_cause_router is not None
            and self.block_cause_config.enabled
        )
        hide_raw = (
            router_active
            and self.block_cause_config.hide_raw_phase_outputs
        )

        # Phase discovery / instrument — surface as siblings unless
        # the router is hiding them.
        if self._latest_phase_discovery and not hide_raw:
            env.phase_discovery = [
                r.to_dict() for r in self._latest_phase_discovery
            ]
            line = render_phase_discovery_alert(self._latest_phase_discovery)
            if line and line not in env.alerts:
                env.alerts.append(line)
        if self._latest_phase_instrument_suggestions and not hide_raw:
            env.phase_instrument_suggestions = [
                s.to_dict() for s in self._latest_phase_instrument_suggestions
            ]
            line = render_phase_instrument_alert(
                self._latest_phase_instrument_suggestions,
            )
            if line and line not in env.alerts:
                env.alerts.append(line)
        # Block-cause routing surface — always emitted when the
        # router fired.
        if self._latest_block_cause:
            env.block_cause = [r.to_dict() for r in self._latest_block_cause]
            line = render_block_cause_alert(self._latest_block_cause)
            if line and line not in env.alerts:
                env.alerts.append(line)
        # Constant-provenance — every value record that gave the
        # classifier enough info gets a verdict on the envelope.
        if self._latest_constant_provenance:
            env.constant_provenance = [
                r.to_dict() for r in self._latest_constant_provenance
            ]
            line = render_constant_provenance_alert(self._latest_constant_provenance)
            if line and line not in env.alerts:
                env.alerts.append(line)
        self._maybe_book_periodic_card(env)
        return env

    def _maybe_book_periodic_card(self, env: DisciplineEnvelope) -> None:
        step = self.state.step_count
        last = self.state.last_periodic_card_at
        if step - last >= self.config.periodic_interval:
            env.card = render_periodic_card(
                self.state,
                step=step,
                active_findings=self._safe_count_findings(),
                open_hyps=self._safe_count_open_hyps(),
            )
            self.state.last_periodic_card_at = step

    # -- telemetry helpers (best-effort; never raise) ----------------------

    def _safe_count_findings(self) -> int:
        core = self.core
        if core is None:
            return 0
        try:
            from .store import open_findings_db
            conn = open_findings_db(core.work)
            try:
                row = conn.execute("SELECT COUNT(*) FROM findings").fetchone()
                return int(row[0] or 0) if row else 0
            finally:
                conn.close()
        except (sqlite3.Error, Exception):
            return 0

    def _safe_count_open_hyps(self) -> int:
        core = self.core
        if core is None:
            return 0
        try:
            from .store import open_hypotheses_db
            conn = open_hypotheses_db(core.work)
            try:
                row = conn.execute(
                    "SELECT COUNT(*) FROM hypotheses WHERE status NOT IN "
                    "('passed', 'failed', 'abandoned')"
                ).fetchone()
                return int(row[0] or 0) if row else 0
            finally:
                conn.close()
        except (sqlite3.Error, Exception):
            return 0


# ---------------------------------------------------------------------------
# Refusal-flow marker — lets the serve loop distinguish intercepted
# calls from regular exceptions while still propagating the envelope.
# ---------------------------------------------------------------------------


class DisciplineRaise(Exception):
    """Carries a populated envelope to the serve loop so it can emit a
    JSON-RPC error response with the methodology payload attached."""

    def __init__(self, envelope: DisciplineEnvelope):
        super().__init__(envelope.intercepted_reason or "discipline intercept")
        self.envelope = envelope


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _flatten_strings(obj: Any, *, depth: int = 4) -> str:
    """Recursive flatten of all string leaves (capped). Used for
    keyword search and provenance sniff."""
    if depth <= 0:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return " ".join(_flatten_strings(v, depth=depth - 1) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return " ".join(_flatten_strings(v, depth=depth - 1) for v in obj)
    return ""


def _looks_like_unpromoted(params: dict[str, Any], blob: str) -> bool:
    """A payload that explicitly marks itself as experiment / unpromoted
    must not enter the promotion path (M2)."""
    payload = params.get("payload") if isinstance(params, dict) else None
    if isinstance(payload, dict):
        if payload.get("ledger_status") == "experiment":
            return True
        if payload.get("provenance") == "experiment":
            return True
    src = params.get("source") if isinstance(params, dict) else None
    if isinstance(src, str) and src.startswith("experiment"):
        return True
    # textual heuristic — explicit "未入账本" or "unpromoted" / "uncommitted"
    needles = ("未入账本", "unpromoted", "ledger_status=experiment",
               "ledger_status: experiment")
    low = (blob or "").lower()
    for n in needles:
        if n.lower() in low:
            return True
    return False


def _contains_high_rate(d: dict[str, Any], *, floor: float) -> bool:
    """Walk the result dict; return True if any nested key ending in
    ``_rate`` / ``_pct`` is at-or-above ``floor`` (i.e. ≥99% by default),
    OR any pass-count >= total-count with both >= 1 (the 100% trap)."""
    def _walk(node: Any) -> bool:
        if isinstance(node, dict):
            checked = node.get("checked")
            passed  = node.get("passed")
            if isinstance(checked, int) and isinstance(passed, int) \
                    and checked >= 1 and passed >= checked:
                return True
            for k, v in node.items():
                if isinstance(k, str) and (k.endswith("_rate")
                                            or k.endswith("_pct")
                                            or k.endswith("_ratio")):
                    if isinstance(v, (int, float)) and float(v) >= floor:
                        return True
                if _walk(v):
                    return True
        elif isinstance(node, list):
            for v in node:
                if _walk(v):
                    return True
        return False
    return _walk(d)


def _first_contradiction_id(result: Any) -> int | None:
    """Pull a contradicting finding id out of the result if present —
    looks at `contradicts_finding_id`, `invalidated_by`, or
    `invariants_failed` (the M8 module's output shape)."""
    if not isinstance(result, dict):
        return None
    for key in ("contradicts_finding_id", "invalidated_by"):
        v = result.get(key)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
    inv = result.get("invariants_failed")
    if isinstance(inv, list) and inv:
        # No specific id — surface a synthetic one so the prompt still
        # fires. ``-1`` signals "see invariants_failed".
        return -1
    return None


def _recent_streak_same_type(
    failures,
    *,
    threshold: int,
) -> bool:
    """Last `threshold` recent_failures are all of the same method."""
    if len(failures) < threshold:
        return False
    tail = list(failures)[-threshold:]
    return len(set(tail)) == 1


# methods provided by utov itself; static_tool calls outside this set
# count as "non-utov path".
_UTOV_NATIVE_STATIC_TOOLS: set[str] = {
    # When the engine adds first-class capability we add it here.
    # Default set is empty so any static_tool invocation pings the prompt.
}


def _looks_non_utov(tool: str) -> bool:
    if not tool:
        return False
    return tool not in _UTOV_NATIVE_STATIC_TOOLS
