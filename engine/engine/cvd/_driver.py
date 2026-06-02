"""CVD driver loop: PLACE, run_cvd/resume, the dispatch loop, and export."""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..export_stamp import export_stamped_json, filename_verdict_mismatch
from ..types import MemSnapshot
from ._model import (
    BASE_VALUE, Candidate, CvdBudget, CvdOutcome, CvdResult,
    CvdState, ExtensionRequest, Registry, Verdict, Verifier,
    VStatus, _truncated_traceback,
)
from ._registry import default_registry, _roi

_log = logging.getLogger(__name__)


# --- widen + dispatch helpers -----------------------------------------------

_SUCCESS_KINDS = ("CONTINUOUS_BUFFER", "STREAMING")


def _can_widen(state: CvdState) -> bool:
    return state.obs_scope == 0 and (bool(state.snapshots) or state.window is not None)


def _widen(state: CvdState) -> CvdState:
    return CvdState(state.items, state.expected, state.snapshots,
                    window=None, obs_scope=state.obs_scope + 1)


_GEN_DIAG_CACHE: dict[type, bool] = {}


def _generate_accepts_diag(g: "CandidateGenerator") -> bool:
    """True iff this generator's ``generate`` declares a ``diag`` parameter.

    Lets a generator opt into the run's records list (progress / truncation events)
    without changing the base ``CandidateGenerator.generate(self, state)`` contract —
    a generator that does not declare ``diag`` is called exactly as before (inv 7).
    Cached per class (signature introspection is not free)."""
    cls = type(g)
    hit = _GEN_DIAG_CACHE.get(cls)
    if hit is None:
        import inspect
        try:
            params = inspect.signature(g.generate).parameters
            hit = "diag" in params
        except (TypeError, ValueError):
            hit = False
        _GEN_DIAG_CACHE[cls] = hit
    return hit


def _generate(state: CvdState, registry: Registry, disabled: set | None = None,
              diag: list | None = None) -> list[Candidate]:
    disabled = disabled or set()
    cands: list[Candidate] = []
    for g in registry.generators:
        if g.name in disabled:
            continue
        try:
            # A generator may opt into progress/truncation logging by declaring a
            # ``diag`` parameter (e.g. the recovery window generator records its
            # backtrace progress + a GENERATION_BUDGET_EXHAUSTED truncation here).
            # Generators without it keep the bare signature, byte-for-byte (inv 7).
            if diag is not None and _generate_accepts_diag(g):
                cands += g.generate(state, diag=diag)
            else:
                cands += g.generate(state)
        except Exception as e:   # a broken generator must not derail the train
            if diag is not None:  # …but never silently swallow the diagnostic
                diag.append({"event": "GENERATOR_ERROR",
                             "tool": f"{g.name}@{g.version}",
                             "error": f"{type(e).__name__}: {e}",
                             "error_detail": _truncated_traceback()})
    seen: set[tuple[str, int]] = set()
    uniq: list[Candidate] = []
    for c in cands:
        key = (c.kind, c.locus)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


# --- PLACE contract + evidence intake + credibility (CVD_ADDENDUM) ----------

class PlacementError(ValueError):
    """PLACE refused: a mandatory precondition (§A) is missing."""


@dataclass
class AgentSubmission:
    """Evidence intake (§A backfill): an existing agent finding/hypothesis to
    ingest as a candidate. The agent submits EVIDENCE, never a confidence number —
    CVD computes the credibility (§B)."""
    locus: int
    provenance: str = "agent_hypothesis"   # agent_finding|agent_hypothesis|agent_evidence
    evidence: list = field(default_factory=list)
    kind: str = "sink"
    note: str = ""


def _observed_byte_at(state: CvdState, addr: int):
    val = None
    for ins in state.scoped_items():
        for op in ins.mem:
            if op.rw == "w" and op.size > 0 and op.addr <= addr < op.addr + op.size:
                val = (op.val >> (8 * (addr - op.addr))) & 0xFF
    for s in state.scoped_snapshots():
        if s.addr <= addr < s.addr + len(s.data):
            val = bytes(s.data)[addr - s.addr]
    return val


def _region_matches_expected(state: CvdState, locus: int, length: int) -> bool:
    return all(_observed_byte_at(state, locus + off) == state.expected[off]
               for off in range(length))


def evaluate_credibility(locus: int, evidence: list, state: CvdState) -> float:
    """§B: credibility is EVALUATED by us, not submitted. Rises with (1) quantity
    of independent evidence and (2) fit — corroboration by our verifiable signals
    (decisive: oracle-reconstruct at the locus). A bare assumption → lowest."""
    items = evidence or []
    keys = {json.dumps(e, sort_keys=True, default=str) for e in items}
    quantity = math.log1p(len(keys))
    fit = 0.0
    if _region_matches_expected(state, locus, len(state.expected)):
        fit += 2.0   # our decisive signal (oracle-reconstruct) corroborates
    corrob = sum(1 for e in items if isinstance(e, dict)
                 and (e.get("oracle_match") or e.get("verifiable")))
    if keys:
        fit += corrob / len(keys)
    return round(quantity + fit, 3)


def place(items, expected, *, snapshots=None, window=None, has_runner: bool = True,
          submissions: Iterable | None = None,
          obs_scope: int = 0) -> tuple[CvdState, list[Candidate]]:
    """PLACE (CVD_ADDENDUM §A): enforce mandatory preconditions, ingest evidence.

    Mandatory (refuse without): a working deployed runner (anti-debug/unpack/deploy
    is the agent's job), a goal oracle (expected), and a trace. Evidence intake:
    each AgentSubmission becomes a candidate carrying its provenance + a credibility
    WE computed (§B) — bootstrapping from what the agent already knows.
    """
    items = list(items)
    expected = bytes(expected)
    if not has_runner:
        raise PlacementError(
            "PLACE requires a working, deployed black-box runner — anti-debug / "
            "unpacking / deployment is the agent's job, not CVD's")
    if not expected:
        raise PlacementError("PLACE requires a goal oracle / expected bytes")
    if not items:
        raise PlacementError("PLACE requires a trace (or runner-ability to capture one)")
    state = CvdState(items, expected, list(snapshots or []), window=window,
                     obs_scope=obs_scope)
    ingested: list[Candidate] = []
    for sub in (submissions or []):
        cred = evaluate_credibility(sub.locus, sub.evidence, state)
        ingested.append(Candidate(
            sub.kind, sub.locus, signal="agent_submission",
            entry_reason=f"{sub.provenance} @0x{sub.locus:x} "
                         f"({len(sub.evidence)} evidence) {sub.note}".strip(),
            base_value=BASE_VALUE.get("agent_submission", 1.0),
            provenance=sub.provenance, credibility=cred))
    return state, ingested


# --- driver loop (CVD_DESIGN §3, CVD_PLUS_DESIGN §3 dispatch) ----------------

def run_cvd(items, expected, *, snapshots=None, window=None, budget=None,
            registry: Registry | None = None, has_runner: bool = True,
            submissions: Iterable | None = None, policy=None,
            artifacts: Iterable | None = None,
            obs_scope: int = 0,
            collect_extensions: bool = False) -> CvdResult:
    from ..cvd_mount import default_policy, build_manifest
    registry = registry or default_registry()
    policy = policy or default_policy()
    budget = budget or CvdBudget(policy.max_candidates, policy.max_widen)
    # PLACE: enforce the §A contract and ingest agent evidence as candidates.
    # ``obs_scope`` defaults to 0 (the widen ladder lifts it on stall, as before);
    # a caller that has already-in-execution snapshots (G1: SAME execution) may open
    # scope at PLACE so scoped_snapshots() exposes them to the verify path without
    # waiting for a stall-driven WIDEN — see run_recovery(snapshots=…).
    state, ingested = place(items, expected, snapshots=snapshots, window=window,
                            has_runner=has_runner, submissions=submissions,
                            obs_scope=obs_scope)
    # T-intake (§3): run default-ON Readers over supplied artifacts, self-gating on
    # detect(); fold the canonical outputs into the state before the loop.
    records: list[dict] = []
    for art in (artifacts or []):
        for r in registry.readers:
            if r.name in policy.disabled_tools or not r.detect(art):
                continue
            rr = r.read(art)
            state.snapshots.extend(rr.snapshots)
            state.call_events.extend(rr.call_events)
            if rr.items:
                state.items.extend(rr.items)
            records.append({"event": "READ", "reader": f"{r.name}@{r.version}",
                            "artifact": art.kind, "snapshots": len(rr.snapshots),
                            "call_events": len(rr.call_events)})
    manifest = build_manifest(state, registry, policy)   # §6 pre-drive announcement
    records.append({"event": "MANIFEST", **{k: manifest[k] for k in ("mounted", "rationale")}})
    generated = _generate(state, registry, policy.disabled_tools, diag=records)
    frontier = list(ingested) + _dedup_spawn(generated, list(ingested))
    return _drive(state, frontier, {}, records, registry, budget, policy, manifest,
                  stall=0, widen_steps=0, tried=0, confirms=0,
                  collect=collect_extensions)


def resume(checkpoint: dict, items, *, registry: Registry | None = None,
           budget: CvdBudget | None = None,
           collect_extensions: bool = False) -> CvdResult:
    """Continue a paused run from a serialized checkpoint with an (extended)
    Registry — the unchanged prefix is not re-verified. The MountPolicy is
    restored from the checkpoint so an agent's edit at RETURN persists (§7)."""
    from ..cvd_mount import MountPolicy, build_manifest, default_policy
    registry = registry or default_registry()
    policy = (MountPolicy.from_dict(checkpoint["policy"])
              if checkpoint.get("policy") else default_policy())
    budget = budget or CvdBudget(policy.max_candidates, policy.max_widen)
    snaps = [MemSnapshot(addr=d["addr"], data=bytes.fromhex(d["data"]),
                         label=d.get("label", ""), source=d.get("source", "snapshot"))
             for d in checkpoint.get("snapshots", [])]
    state = CvdState(list(items), bytes.fromhex(checkpoint["expected"]), snaps,
                     window=(tuple(checkpoint["window"]) if checkpoint.get("window") else None),
                     obs_scope=checkpoint.get("obs_scope", 0))
    frontier = [Candidate.from_dict(d) for d in checkpoint.get("frontier", [])]
    history = dict(checkpoint.get("history", {}))
    records = list(checkpoint.get("log", []))
    manifest = build_manifest(state, registry, policy)
    return _drive(state, frontier, history, records, registry, budget, policy, manifest,
                  stall=0, widen_steps=0, tried=0, confirms=0,
                  collect=collect_extensions)


def _checkpoint(state: CvdState, frontier: list[Candidate], history: dict,
                records: list[dict], policy) -> dict:
    return {
        "expected": state.expected.hex(),
        "window": list(state.window) if state.window else None,
        "obs_scope": state.obs_scope,
        "frontier": [c.to_dict() for c in frontier],
        "history": dict(history),
        "log": records,
        "snapshots": [{"addr": s.addr, "data": bytes(s.data).hex(),
                       "label": s.label, "source": s.source} for s in state.snapshots],
        "policy": policy.to_dict() if policy is not None else None,
    }


def _drive(state, frontier, history, records, registry, budget, policy, manifest,
           *, stall, widen_steps, tried, confirms, collect: bool = False) -> CvdResult:
    from ..cvd_mount import stall_pressure, estimate_t2, heavy_over_budget
    disabled = policy.disabled_tools
    # collect mode (gap①): one run enumerates the WHOLE gap map. Single-candidate
    # gaps (no-verifier / a verify capability_request / a PENDING agent-judgment)
    # are accumulated and the candidate skipped, instead of returning at the first
    # one. The run ends only at the global terminal (frontier empty + widen done).
    ext_requests: list[dict] = []
    pending_judgments: list[dict] = []
    confirmed_list: list[dict] = []

    def rec(event: str, **kw):
        records.append({"event": event, **kw})

    def done(res: CvdResult) -> CvdResult:
        res.manifest = manifest
        # Always surface what was gathered (empty in non-collect runs) so the gap
        # map travels with every outcome, never silently dropped.
        if collect:
            res.extension_requests = ext_requests
            res.pending_judgments = pending_judgments
            res.confirmed = confirmed_list
        return res

    def collected_result() -> CvdResult:
        """The global terminal for a collect-mode run: the whole gap map at once."""
        verdict = ("clean" if not ext_requests and not pending_judgments
                   else "gaps_collected")
        return done(CvdResult(
            CvdOutcome.COLLECTED, verdict=verdict, log=records,
            checkpoint=_checkpoint(state, frontier, history, records, policy)))

    rec("GENERATE", scope=state.obs_scope, n=len(frontier))
    level_jumped = False

    while True:
        if not frontier:
            if widen_steps < budget.max_widen and _can_widen(state):
                state = _widen(state); widen_steps += 1
                frontier = _generate(state, registry, disabled, diag=records)
                rec("WIDEN", obs_scope=state.obs_scope, n=len(frontier))
                continue
            for tc in registry.terminals:
                if tc.name in disabled:
                    continue
                try:
                    t = tc.classify(state)
                except Exception as e:
                    t = None
                    rec("TERMINAL_ERROR", tool=f"{tc.name}@{tc.version}",
                        error=f"{type(e).__name__}: {e}",
                        error_detail=_truncated_traceback())
                if t is not None:
                    rec("TERMINAL", verdict=t.kind, tool=f"{tc.name}@{tc.version}")
                    if collect:
                        # A globally-claimed terminal (e.g. the whole locus is opaque
                        # staging) belongs IN the gap map, not a short-circuit return.
                        if not t.success and t.capability_request:
                            ext_requests.append({
                                "missing_kind": "capability", "scope": "global",
                                "terminal_kind": t.kind,
                                "capability_request": t.capability_request,
                                # Req2 G3: uniform machine-readable block-reason key.
                                "block_why": (t.capability_request
                                              or f"terminal: {t.kind}"),
                                "why": "frontier exhausted; TerminalClassifier claimed "
                                       "the whole locus as a terminal frontier"})
                        return collected_result()
                    outcome = CvdOutcome.SUCCESS if t.success else CvdOutcome.TERMINAL
                    return done(CvdResult(
                        outcome, verdict=t.kind, sink_base=t.sink_base,
                        provenance=t.evidence or None, log=records,
                        capability_request=t.capability_request,
                        # Req2 G3: block_why only on a BLOCKED (non-success) terminal.
                        block_why=(None if t.success else
                                   (t.capability_request or f"terminal: {t.kind}"))))
            if collect:
                # frontier empty + widen exhausted + no terminal claimed → the
                # global terminal for a collect run: hand back the whole gap map.
                return collected_result()
            er = ExtensionRequest(
                missing_kind="terminal",
                why="dead-end state but no TerminalClassifier claimed it",
                suggestion="register a TerminalClassifier or an observation source",
                where={"obs_scope": state.obs_scope})
            rec("EXTENSION_REQUEST", **er.to_dict())
            return done(CvdResult(CvdOutcome.EXTENSION_REQUEST, extension_request=er.to_dict(),
                                  log=records, checkpoint=_checkpoint(state, frontier, history, records, policy)))

        for c in frontier:
            c.score = _roi(c, history, stall)
        # signal-triggered escalation rules (§5 / §11.2): a fired rule preempts its
        # candidate to the front and applies its escalation.
        for rule in registry.rules:
            if rule.name in disabled:
                continue
            for c in frontier:
                try:
                    fired = rule.trigger(c, state, history)
                except Exception as e:
                    fired = False
                    rec("RULE_ERROR", rule=f"{rule.name}@{rule.version}", phase="trigger",
                        error=f"{type(e).__name__}: {e}",
                        error_detail=_truncated_traceback())
                if fired:
                    c.score = float("inf")
                    try:
                        action = rule.escalate(c, state)
                    except Exception as e:
                        action = {"action": "error",
                                  "error": f"{type(e).__name__}: {e}",
                                  "error_detail": _truncated_traceback()}
                    rec("ESCALATE", rule=f"{rule.name}@{rule.version}",
                        candidate=c.to_dict(), action=action)
                    break
        frontier.sort(key=lambda c: -c.score)

        # stall_pressure level-jump (§5 / §11.4) — the heaviest lever. Suppressed
        # in collect mode: a collect run is a full enumeration pass over the
        # frontier (list every gap), not a stall-driven escalation to a heavy tier;
        # the heavy machinery's own EXTENSION_REQUEST/BUDGET_PAUSE would otherwise
        # short-circuit the enumeration before the gap map is complete.
        if not level_jumped and not collect:
            sp = stall_pressure(tried=tried, frontier_size=len(frontier),
                                confirms=confirms, policy=policy)
            if sp > policy.stall_theta:
                level_jumped = True
                rec("LEVEL_JUMP", stall_pressure=round(sp, 3), theta=policy.stall_theta)
                keep = max(1, len(frontier) // 2)            # prune the low-ROI tail
                pruned = len(frontier) - keep
                frontier = frontier[:keep]
                rec("PRUNE", dropped=pruned, kept=keep)
                armed = [v for v in registry.verifiers
                         if v.name in policy.heavy_tools and v.name not in disabled]
                if not (policy.heavy_armed and armed):
                    er = ExtensionRequest(
                        missing_kind="verifier",
                        why=f"stall_pressure>{policy.stall_theta}: heavy (T2) tier needed "
                            f"but not built/armed ({sorted(policy.heavy_tools)})",
                        suggestion="register a heavy Verifier (Triton symex / vmtrace / "
                                   "re-capture) and arm it in the policy",
                        where={"stall_pressure": round(sp, 3)})
                    rec("EXTENSION_REQUEST", **er.to_dict())
                    return done(CvdResult(CvdOutcome.EXTENSION_REQUEST,
                                          extension_request=er.to_dict(), log=records,
                                          checkpoint=_checkpoint(state, frontier, history, records, policy)))
                est = estimate_t2(state, armed[0].name)
                if heavy_over_budget(est, policy):
                    rec("BUDGET_PAUSE", tool=armed[0].name, estimate=est)
                    return done(CvdResult(CvdOutcome.BUDGET_PAUSE, verdict="budget_pause",
                                          capability_request=f"approve heavy tool {armed[0].name} "
                                                             f"(est {est}) or raise budget",
                                          budget_estimate=est, log=records,
                                          checkpoint=_checkpoint(state, frontier, history, records, policy)))
                # within budget -> mount the heavy probe for the armed verifier
                frontier.insert(0, Candidate("heavy_probe", 0, "heavy_mount",
                                             f"mount {armed[0].name} (est {est})",
                                             base_value=10.0))
                rec("MOUNT_T2", tool=armed[0].name, estimate=est)

        c = frontier.pop(0)
        history[c.signal] = history.get(c.signal, 0) + 1
        tried += 1
        if tried > budget.max_candidates:
            frontier.insert(0, c)
            rec("BUDGET_EXHAUSTED", tried=tried)
            return done(CvdResult(CvdOutcome.BUDGET_EXHAUSTED, verdict="budget",
                                  capability_request="raise max_candidates or narrow the goal",
                                  log=records,
                                  checkpoint=_checkpoint(state, frontier, history, records, policy)))

        verifiers = [v for v in registry.verifiers
                     if v.name not in disabled and _safe_applies(v, c, state, diag=records)]
        if not verifiers:
            er = ExtensionRequest(
                missing_kind="verifier",
                why=f"no Verifier applies to candidate kind {c.kind!r}",
                suggestion=f"register a Verifier whose applies() accepts kind {c.kind!r}",
                where={"candidate": c.to_dict()})
            rec("EXTENSION_REQUEST", **er.to_dict())
            if collect:
                # single-candidate gap: record it, skip this candidate, keep going.
                ext_requests.append(er.to_dict())
                continue
            return done(CvdResult(CvdOutcome.EXTENSION_REQUEST, extension_request=er.to_dict(),
                                  log=records, checkpoint=_checkpoint(state, frontier, history, records, policy)))

        v = min(verifiers, key=lambda v: _safe_cost(v, c, state, diag=records))
        tool = f"{v.name}@{v.version}"
        try:
            verdict = v.verify(c, state)
        except Exception as e:   # governance §7: a bad tool eliminates its candidate
            tb = _truncated_traceback()
            verdict = Verdict(
                VStatus.ELIMINATED,
                reason=f"tool_error:{type(e).__name__}: {e}",
                evidence={"error_detail": tb, "tool": tool},
                error_detail=tb)

        if verdict.status is VStatus.CONFIRMED:
            stall = 0
            confirms += 1
            rec("CONFIRMED", tool=tool, candidate=c.to_dict(),
                located_via=verdict.evidence.get("located_via", ""))
            if collect:
                confirmed_list.append({"candidate": c.to_dict(),
                                       "evidence": verdict.evidence})
            frontier += _dedup_spawn(verdict.spawn, frontier)
        elif verdict.status is VStatus.PENDING:
            # NOT a capability gap — an agent-judgment checkpoint surfaced by the
            # Verifier (e.g. drive returned a DrivePause). Collect it and move on;
            # in a normal run it RETURNS to the agent to decide.
            stall += 1
            judgment = {"candidate": c.to_dict(), "reason": verdict.reason,
                        "checkpoint": verdict.evidence or None,
                        "capability_request": verdict.capability_request}
            rec("PENDING_JUDGMENT", tool=tool, candidate=c.to_dict(),
                reason=verdict.reason)
            frontier += _dedup_spawn(verdict.spawn, frontier)
            if collect:
                pending_judgments.append(judgment)
                continue
            return done(CvdResult(
                CvdOutcome.PENDING_JUDGMENT, verdict=verdict.reason or "pending_judgment",
                capability_request=verdict.capability_request, log=records,
                pending_judgments=[judgment],
                checkpoint=_checkpoint(state, frontier, history, records, policy)))
        elif verdict.status is VStatus.TERMINAL:
            rec("PROVENANCE", tool=tool, verdict=verdict.terminal_kind,
                chain_len=len(verdict.evidence.get("chain", [])))
            if verdict.success:
                if collect:
                    confirmed_list.append({"candidate": c.to_dict(),
                                           "terminal_kind": verdict.terminal_kind,
                                           "evidence": verdict.evidence})
                    continue
                return done(CvdResult(CvdOutcome.SUCCESS, verdict=verdict.terminal_kind,
                                      sink_base=verdict.located_base,
                                      provenance=verdict.evidence or None, log=records))
            if collect:
                # a per-candidate terminal frontier (opaque / un-modeled / seed-
                # invariant): record the capability it needs, skip, keep going.
                ext_requests.append({
                    "missing_kind": "capability", "scope": "candidate",
                    "candidate": c.to_dict(), "terminal_kind": verdict.terminal_kind,
                    "capability_request": verdict.capability_request,
                    # Req2 G3: ONE uniform machine-readable block-reason key across
                    # every terminal exit (recapture-loop / cvd / recovery) — a
                    # consumer reads ``block_why`` without guessing reason vs detail
                    # vs why. Falls back to the terminal kind when no reason was set.
                    "block_why": verdict.reason or f"terminal: {verdict.terminal_kind}",
                    "why": verdict.reason or f"terminal: {verdict.terminal_kind}",
                    # Carry the verifier's (already-compact, invariant-4) evidence
                    # into the gap map. For an opaque-staging terminal this is what
                    # ships the symbolic_forwards count — the "the forward WAS tried,
                    # it still collapsed" proof (坎2) — instead of dropping it.
                    "evidence": verdict.evidence or None})
                stall += 1
                continue
            if widen_steps < budget.max_widen and _can_widen(state):
                state = _widen(state); widen_steps += 1
                frontier += _generate(state, registry, disabled, diag=records)
                rec("WIDEN", obs_scope=state.obs_scope, reason="provenance not resolved")
                continue
            return done(CvdResult(CvdOutcome.TERMINAL, verdict=verdict.terminal_kind,
                                  sink_base=verdict.located_base,
                                  provenance=verdict.evidence or None, log=records,
                                  capability_request=verdict.capability_request,
                                  block_why=(verdict.reason
                                             or f"terminal: {verdict.terminal_kind}")))
        else:  # ELIMINATED
            stall += 1
            c.elim_reason = verdict.reason
            # A tool_error elimination (governance §7) carries its traceback so the
            # gap map / ER can route "which key, which line" — never a flat
            # tool_error:<type>. Only attached when present (success/normal
            # eliminations add nothing — invariant 7).
            extra = {"error_detail": verdict.error_detail} if verdict.error_detail else {}
            rec("ELIMINATED", tool=tool, candidate=c.to_dict(), reason=verdict.reason,
                **extra)
            frontier += _dedup_spawn(verdict.spawn, frontier)


def _safe_applies(v: Verifier, c: Candidate, state: CvdState,
                  diag: list | None = None) -> bool:
    try:
        return bool(v.applies(c, state))
    except Exception as e:
        if diag is not None:
            diag.append({"event": "VERIFIER_APPLIES_ERROR",
                         "tool": f"{v.name}@{v.version}",
                         "error": f"{type(e).__name__}: {e}",
                         "error_detail": _truncated_traceback()})
        return False


def _safe_cost(v: Verifier, c: Candidate, state: CvdState,
               diag: list | None = None) -> float:
    try:
        return float(v.cost(c, state))
    except Exception as e:
        if diag is not None:
            diag.append({"event": "VERIFIER_COST_ERROR",
                         "tool": f"{v.name}@{v.version}",
                         "error": f"{type(e).__name__}: {e}",
                         "error_detail": _truncated_traceback()})
        return float("inf")


def export_gap_map(
    result: CvdResult,
    path: str | Path | None = None,
    *,
    ts: str,
    exec_identity: Mapping[str, Any] | None = None,
    source: str = "cvd.run_cvd (in-memory CvdResult)",
    from_entries: Iterable[str] = (),
) -> str:
    """Project a :class:`CvdResult` to the OUT-layer stamped JSON a consumer reads.

    This is the test-agent's STANDARD output / log for a CVD run (esp. a collect
    gap map): a ``<!-- utov-export ... -->`` header (the authority discriminator)
    over ``result.to_dict()``. The consumer reads THIS — never the sqlite ledger
    (utov's internal collect layer). Pass no ledger to ``run_cvd`` and the run
    produces no ``.sqlite``/``-wal``/``-shm`` at all; the gap map travels as this
    JSON. When ``path`` is given the document is written there; the text is always
    returned. See dev-consumer-output-stamped-json-not-sqlite.md.

    FILENAME-HONESTY GATE (dev-export-filename-honesty-spec.md): when ``path`` is
    given, the caller-chosen filename is checked for a strong closure-claim token
    (confirmed / closed / identified / oracle / solved). If the name asserts a
    closure the run content does NOT support (no oracle-closed classification),
    utov does NOT rewrite the name (consumer's naming right) but refuses to lie
    SILENTLY: it WARNs loudly AND stamps an explicit ``filename_verdict_mismatch``
    field INTO the document, so a reader trusts the content verdict over the name.
    An honest name (no token, or an oracle-closed run) is untouched (regression)."""
    payload = result.to_dict()
    if path is not None:
        mismatch = filename_verdict_mismatch(str(path), payload)
        if mismatch is not None:
            payload["filename_verdict_mismatch"] = mismatch
            _log.warning(
                "EXPORT FILENAME LIES: %r asserts %s but the run is not "
                "oracle-closed (outcome=%s) — wrote filename_verdict_mismatch "
                "into the export; trust the content verdict, not the filename. "
                "Use export_stamp.safe_export_name(result.to_dict()) for an "
                "honest name.",
                Path(path).name, mismatch["filename_claim"],
                payload.get("outcome"))
    text = export_stamped_json(
        payload, source=source, exported_by="run_cvd",
        exec_identity=exec_identity or {}, ts=ts, from_entries=from_entries)
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


def run_cvd_collect_to_json(
    items, expected, *,
    work_root: str | Path,
    ts: str,
    exec_identity: Mapping[str, Any] | None = None,
    filename: str = "cvd_gap_map.json",
    cohort_load: Mapping[str, Any] | None = None,
    **run_kwargs: Any,
) -> tuple[CvdResult, Path]:
    """Consumer one-shot: run a collect CVD and ACTIVELY persist the gap map.

    Runs ``run_cvd(..., collect_extensions=True)`` then writes the stamped JSON gap
    map under ``work_root`` — ONE durable, traceable artifact on disk, not a stream
    the agent reads once and loses (that left ``tc2_cvd_run.log`` empty). No ledger
    is passed, so the run leaves no ``.sqlite``/``-wal``/``-shm``. Returns
    ``(CvdResult, path)``.

    ``cohort_load`` (optional): the cohort-load report (no-mem-sidecar WARN per
    vector). When given it is stamped onto the result's run-level ``cohort_load``
    field so it lands at the TOP of the persisted gap map regardless of window
    outcome — degradation is visible even on an opaque TERMINAL / all-CONFIRMED run
    that emits no PENDING evidence. ``None`` → omitted (no load layer ran).
    See dev-consumer-output-stamped-json-not-sqlite.md."""
    if run_kwargs.pop("collect_extensions", True) is not True:
        raise ValueError("run_cvd_collect_to_json is the collect-mode entry")
    if "ledger" in run_kwargs:
        raise ValueError(
            "consumer-facing run must not take a ledger — the consumer reads the "
            "stamped JSON (OUT layer), never the sqlite ledger (internal COLLECT)")
    res = run_cvd(items, expected, collect_extensions=True, **run_kwargs)
    if cohort_load is not None:
        res.cohort_load = dict(cohort_load)
    root = Path(work_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / filename
    export_gap_map(res, path, ts=ts, exec_identity=exec_identity)
    return res, path


def _dedup_spawn(spawn: list, frontier: list) -> list:
    out = []
    present = {(c.kind, c.locus) for c in frontier}
    for s in spawn:
        if (s.kind, s.locus) not in present:
            present.add((s.kind, s.locus))
            out.append(s)
    return out
