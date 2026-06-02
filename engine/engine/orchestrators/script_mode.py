"""Mode 1: script-driven unattended pipeline (PLAN §13 + §15).

End-to-end orchestration with feedback context, budget gates, and mode dials:

  1. Spin up runner via SubprocessRunnerAdapter (or any RunnerAdapter)
  2. Conformance gate (PLAN §17). FAIL → bail.
  3. Run S1 → S1.5 → S2 → S3 → S4 → S5 in order. Feedback context is wired:
     stages read/write ctx['session'].
  4. Inspect S5 output for stuck points; if Mode allows LLM, run S6 loop.
  5. CostMeter monitors every LLM call; Tracker emits progress events.
     A Budget breach raises BudgetExceeded → orchestrator stops cleanly,
     returns partial results + snapshot.
  6. Mode = "frugal" defaults to skipping S6 entirely (only confidence-
     boosting plugins). Mode = "aggressive" runs the full hypothesis loop.

D-019: this file imports only `engine.core` types + LLMClient. Never
directly from engine.stages.*.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..core import Core
from ..cost import Budget, BudgetExceeded, CostMeter
from ..discipline import DisciplineState
from ..hyp_tree import HypTree
from ..llm_client import LLMClient
from ..progress import EventKind, Tracker
from ..stages.s6_hypothesis import (
    StuckContext,
    generate_hypotheses_only,
    ingest_hypotheses_and_verify,
    propose_and_verify,
)
from ..store import open_hypotheses_db


class Mode(str, Enum):
    FRUGAL     = "frugal"      # only deterministic stages + insertions; no LLM
    AGGRESSIVE = "aggressive"  # full pipeline incl. S6 LLM loop


@dataclass
class NextAction:
    """A concrete next move the agent / user should consider after this run.
    Severity: info < warning < blocker."""
    kind: str
    severity: str
    reason: str
    suggested_command: str | None = None
    # BR-4 §E: which budget axis tripped (input_tokens/output_tokens/
    # total_tokens/usd/calls/wall_seconds). Set only for kind="raise_budget".
    # Lets agent UIs render the right "raise this knob" affordance without
    # parsing `reason`.
    breached_budget: str | None = None


@dataclass
class PipelineRunReport:
    stage_summaries: list[dict[str, Any]]
    hypothesis_count: int
    findings_promoted: int
    paused: bool = False
    pause_reason: str | None = None
    cost: dict[str, Any] = field(default_factory=dict)
    progress: dict[str, Any] = field(default_factory=dict)
    next_actions: list[NextAction] = field(default_factory=list)


def _find_stuck_points(core: Core, max_points: int | None = None) -> list[StuckContext]:
    """Scan S5 output for instructions that survived the slice but were neither
    constant-folded nor matched any InsSub pattern. Those are candidates for
    LLM-led identification.

    max_points: if set, cap the list at this many. Default None = no cap; the
    natural backstop is Budget — once an LLM-token / USD / call ceiling is hit,
    `propose_and_verify` raises BudgetExceeded and the loop stops.
    """
    s5_path = core.work.root / "stage_outputs" / "s5_simplified.jsonl"
    if not s5_path.exists():
        return []
    stuck: list[StuckContext] = []
    with s5_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("kind") != "instr":
                continue
            if "constant" in row or "mov_immediate" in row or row.get("part_of_inssub"):
                continue
            stuck.append(StuckContext(
                parent_hyp_id=None,
                kind_hint="handler_semantic",
                summary=f"unrecognized instruction at idx {row['idx']} pc={row['pc']}",
                snippet=row.get("mnemonic", ""),
                instr_idx=row["idx"],
            ))
            if max_points is not None and len(stuck) >= max_points:
                break
    return stuck


def run_full_pipeline(
    core: Core,
    *,
    llm: LLMClient | None = None,
    mode: Mode | str = Mode.FRUGAL,
    budget: Budget | None = None,
    tracker: Tracker | None = None,
    max_stuck_points: int | None = None,
    emit_events_to=None,           # file-like; if set, every event written as NDJSON
    s6_concurrency: int = 1,       # BR-4 §C: parallel LLM prefetch workers (>=2 = on)
) -> PipelineRunReport:
    """End-to-end orchestration with cost & budget awareness.

    Args:
        mode: FRUGAL skips S6; AGGRESSIVE runs the LLM hypothesis loop.
        budget: limits on tokens / USD / wall time / calls. BudgetExceeded
                during S6 → graceful stop, partial result returned.
        tracker: optional Tracker for the caller to subscribe to progress
                 events. If None, a tracker is still created internally but
                 only used for proactive pause / pacing.
    """
    if isinstance(mode, str):
        mode = Mode(mode)

    meter = CostMeter(budget or Budget())
    tr = tracker or Tracker(meter)

    # Optional: stream every event to a caller-supplied file (typically stderr).
    if emit_events_to is not None:
        def _emit(evt):
            try:
                emit_events_to.write(json.dumps({
                    "type": "event", "kind": evt.kind.value,
                    "timestamp": evt.timestamp, "detail": evt.detail,
                }) + "\n")
                emit_events_to.flush()
            except Exception:
                pass
        tr.on_any(_emit)

    paused = False
    pause_reason: str | None = None
    breached_axis: str | None = None   # BR-4 §E: which budget axis tripped

    # --- deterministic chain (no LLM, no cost) ---
    stage_summaries: list[dict[str, Any]] = []
    for s in ("s1", "s1b", "s2", "s3", "s4", "s5"):
        tr.emit(EventKind.STAGE_START, stage=s)
        summary = core.run_stage(s)
        stage_summaries.append(summary)
        tr.emit(EventKind.STAGE_DONE, stage=s, summary=summary)
    core.checkpoint()
    tr.emit(EventKind.SAFE_INTERRUPT_POINT, after="deterministic_chain")

    findings_promoted = 0

    # --- Plugin-finding pass: mechanically verify S1.5 fingerprint hits
    # against the trace and promote those whose magic value really appears
    # in regs_write. No LLM, runs in both frugal and aggressive modes. This
    # is the baseline that produces findings without burning API budget.
    plugin_verify = core.verify_and_promote_plugin_findings()
    findings_promoted += plugin_verify.get("promoted", 0)
    stage_summaries.append({"stage": "s1b-verify", **plugin_verify})
    tr.emit(EventKind.STAGE_DONE, stage="s1b-verify", summary=plugin_verify)
    for _ in range(plugin_verify.get("promoted", 0)):
        tr.emit(EventKind.FINDING_ADDED, hyp_id=None)

    # BR-4 §1: deterministic handler-semantic pass. Walk the trace for plain
    # reg-reg-reg binops verifier._BIN_OPS already knows how to check; auto-
    # promote each PASS. Runs in both frugal and aggressive — symmetric to
    # the s1b-verify pass above, and recovers ~46% of S6's stuck-point load
    # without any LLM (see BR-4 §1 numbers for TC2 VMP baseline).
    handler_verify = core.verify_and_promote_handler_binops()
    findings_promoted += handler_verify.get("promoted", 0)
    stage_summaries.append(handler_verify)
    tr.emit(EventKind.STAGE_DONE, stage="s5-verify", summary=handler_verify)
    for _ in range(handler_verify.get("promoted", 0)):
        tr.emit(EventKind.FINDING_ADDED, hyp_id=None)

    # 0526Plan C5: layer-0 single-instruction handler passes — three more
    # discoverers that find ARM unary / reg-imm / shifted-extended forms
    # and run check_handler_semantic. Each promotes the same handler_semantic
    # finding kind with source=s5_deterministic; together with the binop
    # pass they cover all five verifier shapes (TC2 added 0526: 135 unary +
    # 312 imm_binop + 123 ext_binop unique-PC findings).
    for layer0_method, stage_name in (
        ("verify_and_promote_handler_unaries",        "s5-verify-unary"),
        ("verify_and_promote_handler_imm_binops",     "s5-verify-imm"),
        ("verify_and_promote_handler_extended_binops", "s5-verify-ext"),
        ("verify_and_promote_handler_bfx",            "s5-verify-bfx"),
        ("verify_and_promote_handler_ch_idioms",      "s5-verify-ch"),
        ("verify_and_promote_handler_maj_idioms",     "s5-verify-maj"),
        ("verify_and_promote_triton_simplifications", "s5-verify-triton"),
        ("verify_and_promote_sigma_idioms",           "s5-fold-sigma"),
        ("verify_and_promote_indexed_load_table",     "s5-indexed-load"),
        ("verify_and_promote_algorithm_templates",    "s5-algorithm-fit"),
        ("self_rescan_missing_anchors",               "s5-anchor-rescan"),
        ("verify_and_promote_mode_evidence_ledger",   "s5-mode-ledger"),
        ("verify_and_promote_primitive_timeline",     "s5-primitive-timeline"),
        ("verify_and_promote_static_artifacts",       "s5-static-artifacts"),
    ):
        result = getattr(core, layer0_method)()
        findings_promoted += result.get("promoted", 0)
        stage_summaries.append(result)
        tr.emit(EventKind.STAGE_DONE, stage=stage_name, summary=result)
        for _ in range(result.get("promoted", 0)):
            tr.emit(EventKind.FINDING_ADDED, hyp_id=None)

    # --- S6 LLM hypothesis loop (aggressive only) ---
    if mode == Mode.AGGRESSIVE:
        stuck_points = _find_stuck_points(core, max_points=max_stuck_points)
        if stuck_points:
            try:
                _llm = llm or LLMClient()
                _llm.attach_meter(meter)
            except Exception as e:
                core.pause(
                    reason=f"S6 needed but LLM unavailable: {type(e).__name__}",
                    hint={"stuck_count": len(stuck_points)},
                )
                hyps = core.get_hypotheses()
                return _make_report(
                    stage_summaries, hyps, findings_promoted, meter, tr,
                    paused=True, pause_reason="LLM unavailable",
                    mode=mode, verifier_degraded=_is_verifier_degraded(core),
                )

            conn = open_hypotheses_db(core.work)
            tree = HypTree(conn)
            discipline = DisciplineState(
                target=core.work.target_dir.name,
                run_id=core.work.run_id,
                tracker=tr,    # so FULL_REMINDER re-injections show as events
            )
            # BR-4 §C: only the prefetch (LLM call) is parallelizable.
            # SQLite tree mutations + verifier + promote stay serial. The
            # concurrent path is gated by `s6_concurrency > 1` AND the
            # backend being a DirectBackend — DelegatedBackend (agent-mode
            # stdio) is fundamentally sequential per the wire protocol.
            from ..llm_client import DirectBackend
            use_concurrent = (
                s6_concurrency > 1
                and isinstance(_llm.backend, DirectBackend)
            )

            def _prep(sp):
                in_state = out_state = None
                if sp.instr_idx is not None and 0 <= sp.instr_idx < len(core._items):
                    ins = core._items[sp.instr_idx]
                    in_state = dict(ins.regs_read)
                    out_state = dict(ins.regs_write)
                return in_state, out_state

            def _drain_one(sp, hyps, in_state, out_state):
                """Serial: ingest a single stuck point's hyps + emit events."""
                nonlocal findings_promoted
                verdicts = ingest_hypotheses_and_verify(
                    sp, hyps, tree, core.verifier,
                    input_state=in_state,
                    expected_output_state=out_state,
                )
                for v in verdicts:
                    tr.emit(EventKind.HYP_OPENED, hyp_id=v["hyp_id"])
                    tr.emit(EventKind.HYP_VERIFIED,
                            hyp_id=v["hyp_id"], verdict=v["verdict"])
                    if v["verdict"] == "pass":
                        core.promote_to_finding(
                            v["hyp_id"], verifier_strategy="handler_semantic"
                        )
                        tr.emit(EventKind.FINDING_ADDED, hyp_id=v["hyp_id"])
                        findings_promoted += 1
                    tr.emit(EventKind.HYP_CLOSED, hyp_id=v["hyp_id"])

            try:
                if use_concurrent:
                    from concurrent.futures import (
                        FIRST_COMPLETED, ThreadPoolExecutor, wait,
                    )
                    pending_futures: dict = {}
                    with ThreadPoolExecutor(max_workers=s6_concurrency,
                                            thread_name_prefix="utov-s6") as ex:
                        it = iter(stuck_points)
                        for sp in it:
                            in_state, out_state = _prep(sp)
                            fut = ex.submit(generate_hypotheses_only,
                                            sp, tree, _llm, discipline, None)
                            pending_futures[fut] = (sp, in_state, out_state)
                            if len(pending_futures) >= s6_concurrency:
                                break
                        while pending_futures and not paused:
                            done, _ = wait(pending_futures,
                                           return_when=FIRST_COMPLETED)
                            for fut in done:
                                sp, in_state, out_state = pending_futures.pop(fut)
                                try:
                                    _n, hyps = fut.result()
                                except BudgetExceeded as e:
                                    paused = True
                                    pause_reason = f"budget exceeded: {e}"
                                    breached_axis = getattr(e, "axis", None)
                                    core.pause(reason=pause_reason)
                                    for f in pending_futures:
                                        f.cancel()
                                    pending_futures.clear()
                                    break
                                _drain_one(sp, hyps, in_state, out_state)
                            if paused:
                                break
                            for sp in it:
                                in_state, out_state = _prep(sp)
                                fut = ex.submit(generate_hypotheses_only,
                                                sp, tree, _llm, discipline, None)
                                pending_futures[fut] = (sp, in_state, out_state)
                                if len(pending_futures) >= s6_concurrency:
                                    break
                else:
                    # Sequential — original behavior, unchanged.
                    for i, sp in enumerate(stuck_points):
                        eta_left = len(stuck_points) - i
                        snap = tr.snapshot(eta_closures=eta_left)
                        if snap.pacing == "stalled":
                            cont = tr.request_pause("pacing stalled",
                                                    eta_closures=eta_left)
                            if not cont:
                                paused = True
                                pause_reason = "pacing stalled (cost-effectiveness dropped)"
                                break
                        in_state, out_state = _prep(sp)
                        try:
                            verdicts = propose_and_verify(
                                sp, tree, _llm, core.verifier, discipline,
                                input_state=in_state,
                                expected_output_state=out_state,
                            )
                        except BudgetExceeded as e:
                            paused = True
                            pause_reason = f"budget exceeded: {e}"
                            breached_axis = getattr(e, "axis", None)
                            core.pause(reason=pause_reason)
                            break
                        for v in verdicts:
                            tr.emit(EventKind.HYP_OPENED, hyp_id=v["hyp_id"])
                            tr.emit(EventKind.HYP_VERIFIED,
                                    hyp_id=v["hyp_id"], verdict=v["verdict"])
                            if v["verdict"] == "pass":
                                core.promote_to_finding(
                                    v["hyp_id"], verifier_strategy="handler_semantic"
                                )
                                tr.emit(EventKind.FINDING_ADDED, hyp_id=v["hyp_id"])
                                findings_promoted += 1
                            tr.emit(EventKind.HYP_CLOSED, hyp_id=v["hyp_id"])
            finally:
                conn.close()

    core.checkpoint()
    tr.emit(EventKind.SAFE_INTERRUPT_POINT, after="pipeline_complete")
    hyps = core.get_hypotheses()

    # Emit agent-decision hints based on what just happened.
    if mode == Mode.FRUGAL and any(h.kind == "algo_signature" for h in hyps):
        tr.emit(EventKind.ASK_USER_BLUE_TEAM,
                reason="fingerprint hits present; aggressive run could verify them")
    if _is_verifier_degraded(core):
        tr.emit(EventKind.ASK_USER_DEGRADED_RESULT,
                reason="runner File mode; findings carry unverified-completeness caveat")
    s1b = next((s for s in stage_summaries if s.get("stage") == "s1b"), None)
    if s1b and s1b.get("fingerprint_hits", 0) == 0:
        tr.emit(EventKind.ASK_USER_NO_FINGERPRINT,
                reason="zero fingerprint hits; VMP likely chopped constants or target isn't crypto")

    # FEATURE-REQUEST-1 auto-emit: drop `<run_dir>/pseudocode.md` whenever
    # the run ended with an algorithm finding (``algorithm_hyp`` — the matcher's
    # pre-oracle-closure hypothesis — or the reserved strong ``algorithm_identified``;
    # task 7). Best-effort; never raises (the emitter helper swallows everything).
    if any(h.kind in ("algorithm_hyp", "algorithm_identified") for h in hyps):
        from ..emitter import emit_to_run_dir
        emit_to_run_dir(core.work.root)

    tr.emit(EventKind.PIPELINE_DONE, findings=findings_promoted)
    return _make_report(stage_summaries, hyps, findings_promoted, meter, tr,
                        paused=paused, pause_reason=pause_reason,
                        mode=mode, verifier_degraded=_is_verifier_degraded(core),
                        breached_axis=breached_axis)


def _is_verifier_degraded(core: Core) -> bool:
    """Read conformance_report.json to know whether we're in File mode."""
    import json
    p = core.work.root / "conformance_report.json"
    if not p.exists():
        return False
    try:
        return bool(json.loads(p.read_text()).get("verifier_degraded", False))
    except Exception:
        return False


def _make_report(
    stage_summaries: list[dict[str, Any]],
    hyps: list,
    findings_promoted: int,
    meter: CostMeter,
    tracker: Tracker,
    *,
    paused: bool,
    pause_reason: str | None,
    mode: Mode = Mode.FRUGAL,
    verifier_degraded: bool = False,
    breached_axis: str | None = None,
) -> PipelineRunReport:
    snap = tracker.snapshot()
    next_actions = _compute_next_actions(
        stage_summaries, hyps, findings_promoted, snap,
        paused=paused, pause_reason=pause_reason,
        mode=mode, verifier_degraded=verifier_degraded,
        breached_axis=breached_axis,
    )
    return PipelineRunReport(
        stage_summaries=stage_summaries,
        hypothesis_count=len(hyps),
        findings_promoted=findings_promoted,
        paused=paused,
        pause_reason=pause_reason,
        cost=snap.cost.__dict__,
        progress={
            "closures": snap.closures,
            "pending": snap.pending,
            "findings": snap.findings,
            "closure_rate_per_min": snap.closure_rate_per_min,
            "pacing": snap.pacing,
        },
        next_actions=next_actions,
    )


_AXIS_TO_FLAG = {
    "input_tokens":  "--raise-budget-tokens",
    "output_tokens": "--raise-budget-tokens",
    "total_tokens":  "--raise-budget-tokens",
    "usd":           "--raise-budget-usd",
    "calls":         "--raise-budget-tokens",   # no explicit calls flag yet
    "wall_seconds":  "--raise-budget-seconds",
}


def _compute_next_actions(
    stage_summaries: list[dict[str, Any]],
    hyps: list,
    findings_promoted: int,
    snap: Any,
    *,
    paused: bool,
    pause_reason: str | None,
    mode: Mode,
    verifier_degraded: bool,
    breached_axis: str | None = None,
) -> list[NextAction]:
    """Look at the run's outputs and propose concrete next moves for the agent."""
    out: list[NextAction] = []

    if paused:
        if pause_reason and "budget" in pause_reason.lower():
            # BR-4 §E: when we know which axis tripped, recommend raising
            # exactly that knob instead of the generic --raise-budget-usd.
            flag = _AXIS_TO_FLAG.get(breached_axis or "", "--raise-budget-usd")
            out.append(NextAction(
                kind="raise_budget",
                severity="blocker",
                reason=f"run halted: {pause_reason}; "
                       f"raise --budget-* and call `utov resume` to continue from this snapshot",
                suggested_command=f"utov resume <work-dir> {flag} <new-value>",
                breached_budget=breached_axis,
            ))
        elif pause_reason and "stalled" in pause_reason.lower():
            out.append(NextAction(
                kind="rerun_with_different_inputs",
                severity="warning",
                reason="pacing stalled — cost-effectiveness dropped. "
                       "Consider broader input coverage or switching back to frugal.",
            ))
        else:
            out.append(NextAction(
                kind="investigate_pause",
                severity="warning",
                reason=f"paused: {pause_reason or 'unknown reason'}",
            ))

    # S1.5: no fingerprint hits — VMP likely, or wrong target
    s1b = next((s for s in stage_summaries if s.get("stage") == "s1b"), None)
    if s1b and s1b.get("fingerprint_hits", 0) == 0:
        out.append(NextAction(
            kind="ask_user.no_fingerprint",
            severity="warning",
            reason="S1.5 found no crypto fingerprints. Either the target doesn't "
                   "use known crypto, or VMP chopped constants below scanner granularity. "
                   "Recommend running §6 handler-semantic recovery (P3 work).",
        ))

    if verifier_degraded:
        out.append(NextAction(
            kind="ask_user.degraded_result",
            severity="warning",
            reason="Runner is File mode; verifier ran without rerun support. "
                   "Findings carry an unverified-completeness caveat. "
                   "Mention this in the final deliverable.",
        ))

    if mode == Mode.FRUGAL and any(
        h.kind == "algo_signature" and h.status == "pending" for h in hyps
    ):
        out.append(NextAction(
            kind="rerun_aggressive",
            severity="info",
            reason="Some algo-signature hypotheses remain pending after the "
                   "deterministic plugin-verify pass (e.g. anchors outside the "
                   "captured trace). Aggressive mode adds S6 LLM coverage.",
            suggested_command=(
                "utov pipeline ... --mode aggressive "
                "--budget-usd 0.20 --budget-tokens 200000"
            ),
        ))

    if findings_promoted == 0 and any(
        h.kind == "algo_signature" and (h.confidence or 0) >= 0.8 for h in hyps
    ):
        out.append(NextAction(
            kind="ask_user.blue_team_needed",
            severity="info",
            reason="High-confidence algorithm signature(s) present but no finding "
                   "promoted yet. Consider blue-team review before downstream use.",
        ))

    return out
