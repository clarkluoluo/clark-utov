"""Recapture batch loop — automate the "cluster gaps → batch rerun → feed back →
re-provenance → converge or honestly stop" cycle (Req2).

Background (the reference case libEncryptor, 2026-06-02): a sink is LOCATED and confirmed
observable, yet ``trace_provenance`` still returns ``NEEDS_OBSERVATION`` — the
producer chain reads native addresses with no captured write/snapshot. The agent
had to hand-pick ``next_watch`` entries round after round (gap 232→1375→1033→673→126),
slow and easy to stall in a local pocket. The missing piece is NOT a new analysis;
it is the *loop* around the analyses already present.

G1 — same-run / nonce honesty (clark 2026-06-02, HARD CONSTRAINT)
================================================================
The earlier draft kept a fixed ``expected_output`` and did
``accumulated.extend(new_snaps)`` — accumulating snapshots ACROSS reruns. That is a
FALSE closure when the output carries a session nonce: each rerun is a *different
execution* with a *different nonce*, so a producer chain stitched from snapshots of
DIFFERENT executions never describes one real production.

The rewrite therefore accumulates the **watch-point PLAN, never the snapshots**:

  * Across rounds we grow a set of watch points (the addr/PC ranges we still need to
    observe). That plan is the only thing that survives a round.
  * Each round does **exactly one** ``adapter.rerun(loop_input, observe_points)``
    where ``observe_points`` = [the output-region observe point] + [every point in
    the current plan]. From that ONE rerun we take:
      - ``rr.output``                → THIS round's ``expected`` (nonce-consistent);
      - ``rr.observations`` → snapshots → fed to ``trace_provenance`` as the ONLY
        snapshots for this round (NEVER carrying a prior round's snapshots).
  * The new ``next_watch`` gaps are merged into the PLAN; the next round re-runs ONCE
    with the enlarged plan and re-captures everything fresh in a single execution.

So ``rr.output``, the output-region snapshot, and every watch snapshot used in one
``trace_provenance`` call all come from the SAME rerun (one nonce). No cross-rerun
snapshot accumulation, ever.

Round 1: the plan is empty → the rerun carries only the output-region observe point
→ rr.output + the output-region snapshot → provenance → the first ``next_watch`` is
seeded into the plan.

As the plan grows the per-round observe-point count grows → it may hit the runner
record cap (contracts §3.2). When the rerun reports ``truncated`` we WARN and stamp
the round dirty — a truncated ledger is NOT complete provenance (no silent closure).

Reused (NOT re-built):
  * :func:`engine.recapture.observe_points_from_provenance` — clusters the byte-level
    plan by PC + contiguous range into a few batch MEM observe points (acceptance ①).
  * ``adapter.rerun(input, observe_points) -> RerunResult`` — the generic capture
    wire (contracts/runner_interface.md §3.2 / Bug1+A).
  * :func:`engine.runner_client.mem_snapshots_from_rerun` — folds the RerunResult mem
    captures into canonical :class:`MemSnapshot` (truncation propagated, not dropped).
  * :func:`engine.oracle_provenance.trace_provenance` — re-classify on THIS round's
    rerun output + snapshots.

Convergence (G2 — not too strict)
=================================
The gap set is ALLOWED to rise temporarily (deeper observation uncovers more real
producer reads — the healthy 232→1375→1033→673→126 shape). We only declare a stable
terminal when ① the distinct gap set fails to shrink for TWO CONSECUTIVE rounds, or
② the watch plan SPREADS across many new region groups (broad diffusion into
stack/table/heap), which signals there is no single buffer to chase.

Every terminal is an EXPLICIT structured verdict (a comfortable, shaped exit —
feedback_red_line_needs_comfortable_exit); the loop NEVER spins silently and NEVER
returns without a verdict (acceptance ③ / A8④).

Generic — no tc2 address, PC, or region is baked in.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .oracle_provenance import (
    ProvenanceResult,
    ProvenanceVerdict,
    trace_provenance,
)
from .recapture import observe_points_from_provenance
from .runner_client import ObservePoint, RerunResult, mem_snapshots_from_rerun
from .types import Instruction, MemSnapshot

_log = logging.getLogger(__name__)


# A provenance verdict that is a CLOSED (production visible) terminus — no further
# observation is needed. NEEDS_OBSERVATION with an empty next_watch is ALSO closed
# (handled explicitly below) but is not a verdict value, so it is not listed here.
_CLOSED_VERDICTS = frozenset({
    ProvenanceVerdict.CONTINUOUS_BUFFER,
    ProvenanceVerdict.STREAMING,
})

# G2: a region-group count above this, AND a sharp jump from the prior round, is read
# as broad diffusion (watch spreading across many new regions — stack/table/heap with
# no single buffer to chase). A generic structural signal, not a tc2 constant.
_DIFFUSION_REGION_FLOOR = 8
_DIFFUSION_JUMP_FACTOR = 2.0


class LoopOutcome(str, enum.Enum):
    """The explicit terminal of a recapture batch loop (never silent)."""

    CLOSED = "CLOSED"                      # provenance closed — production visible
    STALLED = "STALLED"                    # gap set stopped shrinking (stable residue)
    UNPLACEABLE = "UNPLACEABLE"            # gaps remain but none can be hung on a PC
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"  # ran out of rounds while still shrinking
    # G1 (Req2): the closing snapshot set mixes snapshots ACROSS reruns (or its
    # execution provenance cannot be proven the same single execution). A producer
    # chain stitched from snapshots of DIFFERENT executions never describes one real
    # production (a different nonce per run) → NOT a closure, a loud terminal.
    CROSS_RERUN_G1 = "CROSS_RERUN_G1"


# G3: the structured block phase a non-CLOSED terminal reports. One vocabulary across
# STALLED / UNPLACEABLE / BUDGET_EXHAUSTED (not a separate split set).
_BLOCK_PHASE = "PROVENANCE_WATCH_BATCH"

# G1 (Req2): the dedicated block phase for the same-execution violation. A distinct
# vocabulary value (NOT folded into the generic STALLED/… set) so a consumer can tell
# "the gap set would not converge" apart from "the closure illegally mixed reruns".
_BLOCK_PHASE_G1 = "PROVENANCE_WATCH_BATCH_G1"

# G1 (Req2): the machine-readable same-execution requirement statement. The exact
# wording the spec pins (so a consumer can string-match it) — used by both the loop
# terminal and any caller (e.g. cvd_recovery's re-entry closure) that asserts a
# closing snapshot set is one execution.
G1_SAME_EXECUTION_VIOLATED = "same-execution snapshot requirement violated"
G1_SAME_EXECUTION_UNPROVABLE = "same-execution snapshot requirement cannot be proven"


def assert_same_execution(
    snapshots: Iterable[MemSnapshot],
    *,
    require_proof: bool = False,
) -> dict[str, Any] | None:
    """Check a CLOSING snapshot set is one single execution (Req2 G1 guard).

    Returns ``None`` when the set is same-execution-safe (normal closure path —
    never misfires, invariant 7); returns a structured VIOLATION report when the set
    must NOT be treated as one production:

      * **cross-rerun** — the set carries >= 2 DISTINCT non-None ``execution_id``
        tokens: snapshots provably came from DIFFERENT reruns (different nonce each)
        → stitching them is a FALSE closure. Always reported (positive evidence).
      * **unprovable** — ONLY when ``require_proof=True`` and the set has snapshots
        with NO execution_id at all (provenance absent, cannot be proven one
        execution). Off by default: in the loop's normal construct, same-execution
        is guaranteed BY CONSTRUCTION (one rerun per round) and absence of a token
        is NOT a violation — so the default never misfires on today's closures.

    The report is the payload the PROVENANCE_WATCH_BATCH_G1 terminal carries: which
    snapshots crossed reruns (per-token address samples) / why it is unprovable. A
    pure read of the set — no I/O, no re-derivation. Complements B2 (which makes
    same-execution true by construction); this is the loud兜底 alarm for any path
    that did not, or could not, keep that guarantee."""
    snaps = list(snapshots)
    ids = [s.execution_id for s in snaps]
    distinct = {i for i in ids if i is not None}
    if len(distinct) >= 2:
        by_id: dict[Any, list[str]] = {}
        for s in snaps:
            if s.execution_id is None:
                continue
            by_id.setdefault(s.execution_id, []).append(f"0x{s.addr:x}")
        return {
            "violation": "cross_rerun",
            "block_why": (
                f"{G1_SAME_EXECUTION_VIOLATED}: the closing snapshot set spans "
                f"{len(distinct)} distinct reruns — a producer chain stitched from "
                "snapshots of DIFFERENT executions never describes one real "
                "production (a different nonce per run); not a closure"),
            "n_executions": len(distinct),
            "executions": [
                {"execution_id": str(k), "n_snapshots": len(v), "addr_sample": v[:8]}
                for k, v in by_id.items()
            ],
        }
    if require_proof and any(i is None for i in ids) and len(distinct) < 1:
        n_unproven = sum(1 for i in ids if i is None)
        return {
            "violation": "unprovable",
            "block_why": (
                f"{G1_SAME_EXECUTION_UNPROVABLE}: {n_unproven} of {len(ids)} closing "
                "snapshot(s) carry NO execution provenance token — cannot prove they "
                "all came from one single execution; refusing to treat as a closure"),
            "n_unproven": n_unproven,
            "n_total": len(ids),
        }
    return None


def _g1_terminal(report: dict[str, Any], *, n_rounds: int) -> dict[str, Any]:
    """The PROVENANCE_WATCH_BATCH_G1 structured terminal dict (Req2 G1 variant).

    Same shape family as :meth:`RecaptureLoopResult.terminal` (terminal/block_phase/
    block_why/reason), so a consumer reads it with the SAME keys — but the dedicated
    ``PROVENANCE_WATCH_BATCH_G1`` phase + the cross-rerun/unprovable payload name WHY
    it is not a closure (which snapshots crossed reruns / why unprovable)."""
    return {
        "terminal": "BLOCKED",
        "block_phase": _BLOCK_PHASE_G1,
        "outcome": LoopOutcome.CROSS_RERUN_G1.value,
        "block_why": report["block_why"],
        "reason": report["block_why"],
        "violation": report["violation"],
        "same_execution": report,
        "n_rounds": n_rounds,
    }

# G3: how many residual next_watch entries the block terminal samples for a human to
# see WHICH gaps remain (the full set is in ``residual``; this is a readable preview).
_NEXT_WATCH_SAMPLE_N = 8


def _gap_key(w: dict) -> tuple[Any, Any]:
    """Canonical identity of a ``next_watch`` gap: (addr, pc). Two gaps are the same
    gap iff they read the same address at the same reading PC — so the loop measures
    progress by the SET of distinct (addr, pc) it still cannot observe, not by raw
    list length (which can wobble without real progress)."""
    return (w.get("addr"), w.get("pc"))


def _gap_set(prov: ProvenanceResult) -> set[tuple[Any, Any]]:
    return {_gap_key(w) for w in prov.next_watch}


def _shape_residual(prov: ProvenanceResult) -> dict[str, Any]:
    """The residual-gap SHAPE for a non-closed terminal: which PCs / regions remain
    unobserved and why. Groups the surviving ``next_watch`` by reading PC (the hook
    site) and coalesces each PC's addresses into contiguous ``[base, size]`` runs —
    the same PC+range view the batch observe points use, so the caller sees exactly
    which batch points did NOT close. ``pc=None`` gaps (sink bytes with no reading PC
    at all — unhookable) are bucketed under the literal key ``"no_pc"``."""
    by_pc: dict[str, list[int]] = {}
    reasons: dict[str, set[str]] = {}
    for w in prov.next_watch:
        pc = w.get("pc")
        key = pc if pc is not None else "no_pc"
        by_pc.setdefault(key, []).append(int(w["addr"], 16))
        reasons.setdefault(key, set()).add(w.get("reason", ""))
    regions: list[dict[str, Any]] = []
    for key in sorted(by_pc, key=lambda k: (k == "no_pc", k)):
        addrs = sorted(set(by_pc[key]))
        runs: list[list[int]] = []
        for a in addrs:
            if runs and a == runs[-1][0] + runs[-1][1]:
                runs[-1][1] += 1
            else:
                runs.append([a, 1])
        regions.append({
            "pc": key,
            "n_addrs": len(addrs),
            "ranges": [[f"0x{base:x}", size] for base, size in runs],
            "reasons": sorted(r for r in reasons[key] if r),
        })
    return {
        "n_gaps": len(prov.next_watch),
        "n_reading_pcs": sum(1 for k in by_pc if k != "no_pc"),
        "n_unhookable": len(by_pc.get("no_pc", [])),
        "n_region_groups": len(regions),
        "regions": regions,
    }


@dataclass
class LoopRound:
    """One round of the batch loop — what was planned, run, and the resulting gap.

    The round is described by THIS round's single rerun (G1): ``expected_len`` is the
    length of THAT rerun's ``rr.output`` (its nonce), and every snapshot fed this
    round came from that same rerun."""

    round: int
    verdict: str                              # provenance verdict this round produced
    n_plan_points: int                        # batch observe points sent (plan size)
    n_gaps_before: int                        # distinct (addr,pc) gaps before this round
    n_gaps_after: int                         # distinct gaps after this round's rerun
    n_unplaceable: int                        # gaps with no reading PC (cannot hook)
    n_snapshots_fed: int                      # snapshots from THIS rerun (not cumulative)
    n_region_groups: int                      # distinct reading-PC region groups in gaps
    expected_len: int                         # len(rr.output) this round (nonce check)
    chain_n: int = 0                          # producer-chain length this round
    backtrace_truncated: bool = False         # producer backtrace hit a budget ceiling
    new_region: bool = False                  # watch spread into a NEW region group
    truncated: bool = False                   # runner hit a record cap this round
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "verdict": self.verdict,
            "n_plan_points": self.n_plan_points,
            "n_gaps_before": self.n_gaps_before,
            "n_gaps_after": self.n_gaps_after,
            "n_unplaceable": self.n_unplaceable,
            "n_snapshots_fed": self.n_snapshots_fed,
            "n_region_groups": self.n_region_groups,
            "expected_len": self.expected_len,
            "chain_n": self.chain_n,
            "backtrace_truncated": self.backtrace_truncated,
            "new_region": self.new_region,
            "truncated": self.truncated,
            "note": self.note,
        }


@dataclass
class RecaptureLoopResult:
    """The explicit, structured terminal of the recapture batch loop.

    ``outcome`` is always set (never silent). ``final`` is the last provenance result
    (its verdict is the deliverable). ``residual`` carries the gap shape on every
    non-CLOSED outcome (which PCs/regions remain, why), giving the caller a shaped,
    actionable exit instead of a bare 'still NEEDS_OBSERVATION'.

    ``snapshots`` is the snapshot set from the FINAL round's single rerun ONLY (the
    nonce-consistent set behind ``final``) — NOT a cross-rerun accumulation (G1)."""

    outcome: LoopOutcome
    final: ProvenanceResult
    rounds: list[LoopRound] = field(default_factory=list)
    snapshots: list[MemSnapshot] = field(default_factory=list)
    residual: dict[str, Any] | None = None
    detail: str = ""
    # G1 (Req2): the same-execution violation report when ``outcome`` is
    # CROSS_RERUN_G1 (which snapshots crossed reruns / why unprovable). None on every
    # other outcome (invariant 7).
    same_execution: dict[str, Any] | None = None

    @property
    def closed(self) -> bool:
        return self.outcome is LoopOutcome.CLOSED

    def terminal(self) -> dict[str, Any] | None:
        """G3: the structured block terminal for a non-CLOSED outcome, aligned to the
        STALLED/UNPLACEABLE/BUDGET vocabulary (one shape, not a separate split set).
        Every terminal carries ``block_why`` (Req2): ONE machine-readable key a
        consumer reads for the block reason, no guessing reason vs detail. ``None``
        when CLOSED."""
        if self.closed:
            return None
        # G1 (Req2): the same-execution violation is its OWN dedicated terminal
        # (PROVENANCE_WATCH_BATCH_G1) — not folded into the generic gap-convergence
        # vocabulary, so a consumer tells "did not converge" apart from "illegally
        # mixed reruns". Carries block_why like every other terminal.
        if self.outcome is LoopOutcome.CROSS_RERUN_G1:
            report = self.same_execution or {
                "violation": "cross_rerun",
                "block_why": (f"{G1_SAME_EXECUTION_VIOLATED}: closing snapshot set "
                              "is not provably one execution")}
            return _g1_terminal(report, n_rounds=len(self.rounds))
        last_round = self.rounds[-1] if self.rounds else None
        last_n = last_round.n_gaps_after if last_round else len(self.final.next_watch)
        reason = {
            LoopOutcome.STALLED: "next_watch did not converge",
            LoopOutcome.UNPLACEABLE: "remaining gaps have no reading PC to hook",
            LoopOutcome.BUDGET_EXHAUSTED: "round budget exhausted while still shrinking",
        }[self.outcome]
        # G3: the last round's producer-chain length (fall back to the final prov's
        # chain when no round was recorded) and whether its backtrace was truncated —
        # so the block report says how far the chain got and whether it was cut short.
        chain_n = last_round.chain_n if last_round else len(self.final.chain)
        backtrace_truncated = (
            last_round.backtrace_truncated if last_round
            else self.final.backtrace_truncated is not None)
        return {
            "terminal": "BLOCKED",
            "block_phase": _BLOCK_PHASE,
            "outcome": self.outcome.value,
            # Req2: the ONE uniform machine-readable block-reason key (every terminal
            # carries it; ``reason`` stays the human-read long form).
            "block_why": reason,
            "reason": reason,
            "last_next_watch_n": last_n,
            "chain_n": chain_n,
            "backtrace_truncated": backtrace_truncated,
            "rounds": [rnd.to_dict() for rnd in self.rounds],
            "latest_next_watch_sample": self.final.next_watch[:_NEXT_WATCH_SAMPLE_N],
            "suggestion": (
                "need phase-level/full memory snapshot or static table import"
                if self.outcome is not LoopOutcome.BUDGET_EXHAUSTED
                else "raise max_rounds to keep converging, or accept this local closure"
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": "recapture_loop_result",
            "outcome": self.outcome.value,
            "closed": self.closed,
            "n_rounds": len(self.rounds),
            "n_snapshots": len(self.snapshots),
            "final": self.final.to_dict(),
            "rounds": [r.to_dict() for r in self.rounds],
            "detail": self.detail,
        }
        if self.residual is not None:
            out["residual"] = self.residual
        if self.same_execution is not None:
            out["same_execution"] = self.same_execution
        term = self.terminal()
        if term is not None:
            out["terminal"] = term
        return out


def _is_closed(prov: ProvenanceResult) -> bool:
    """Closed = production visible: a buffer/stream verdict, OR a NEEDS_OBSERVATION
    whose gap set is now empty (every read producer is captured)."""
    if prov.verdict in _CLOSED_VERDICTS:
        return True
    return prov.verdict is ProvenanceVerdict.NEEDS_OBSERVATION and not prov.next_watch


def _output_observe_point(
    output_observe_pc: int | None,
    sink_base: int,
    out_len: int,
) -> ObservePoint | None:
    """The output-region MEM observe point: a snapshot of ``[sink_base, +out_len)``
    captured in the SAME rerun as the watch points (G1). Hung on
    ``output_observe_pc`` (the PC at which the sink region is observable — typically
    the sink's first-write / return site the caller already located).

    Returns ``None`` when no PC is supplied (the rerun still returns ``rr.output``;
    we just cannot also snapshot-confirm the sink region in-run). ``out_len<=0``
    (round 1, length not yet known) also yields ``None`` — round 1 still gets
    ``rr.output`` to use as its expected."""
    if output_observe_pc is None or out_len <= 0:
        return None
    return ObservePoint(
        pc=int(output_observe_pc), when="after", capture=("mem",),
        mem=((int(sink_base), int(out_len)),))


def _count_region_groups(prov: ProvenanceResult) -> int:
    """Number of distinct reading-PC groups in the current gaps (a structural proxy
    for how spread-out the watch plan is — used by the G2 diffusion guard)."""
    return len({w.get("pc") for w in prov.next_watch})


def run_recapture_loop(
    items: Iterable[Instruction],
    *,
    sink_base: int,
    adapter: Any,
    loop_input: bytes,
    output_observe_pc: int | None = None,
    output_probe_len: int = 0,
    max_rounds: int = 16,
    min_size: int = 1,
    trace_kwargs: dict[str, Any] | None = None,
) -> RecaptureLoopResult:
    """Drive the plan-accumulate → single-rerun-recapture-all → re-provenance cycle to
    a terminal, honouring the G1 same-run / no-cross-rerun-snapshot constraint.

    Core data flow (G1 — what changed from the e28ba00 draft):
      * The loop accumulates a growing **watch-point PLAN** (the set of addr/PC ranges
        still needing observation), NEVER snapshots.
      * Each round runs ``adapter.rerun(loop_input, observe_points)`` EXACTLY ONCE,
        with ``observe_points`` = [output-region point] + [every plan point]. From
        that ONE rerun:
          - ``rr.output`` is THIS round's ``expected`` (one nonce);
          - its observations → snapshots, the ONLY snapshots passed to
            ``trace_provenance`` this round (no prior round's snapshots).
      * The new ``next_watch`` gaps are merged into the PLAN; next round re-runs once
        with the enlarged plan, re-capturing everything fresh in one execution.

    Round 1: plan empty → rerun carries only the output-region point → rr.output +
    output-region snapshot → provenance → first ``next_watch`` seeds the plan.

    Convergence (G2): the distinct gap set MAY rise transiently; we declare STALLED
    only on TWO CONSECUTIVE non-shrinking rounds, or on broad region diffusion.

    Terminals (G3): CLOSED / STALLED / UNPLACEABLE / BUDGET_EXHAUSTED, each an
    explicit structured verdict carrying the residual shape (never silent / never an
    infinite loop). ``output_observe_pc`` is the PC at which the sink region is
    observable; when omitted, the loop still uses ``rr.output`` as expected but cannot
    additionally snapshot-confirm the sink region in-run (a WARN is logged once).

    ``adapter`` only needs ``rerun(input, observe_points) -> RerunResult``."""
    items = list(items)
    rounds: list[LoopRound] = []
    tkw = dict(trace_kwargs or {})

    if output_observe_pc is None:
        _log.warning(
            "run_recapture_loop: no output_observe_pc — the loop will use each "
            "rerun's rr.output as that round's expected (nonce-consistent), but "
            "cannot ALSO snapshot the sink region [0x%x,+len) in the same run to "
            "confirm it holds rr.output. Pass output_observe_pc (the sink's "
            "observable PC) for full same-run sink confirmation.", int(sink_base))

    # The accumulated PLAN: distinct (addr, pc) gaps to observe. Grows across rounds;
    # never carries snapshots. ``pc`` may be None (unhookable) — kept for shape.
    plan: dict[tuple[Any, Any], dict] = {}

    # Last round's rr.output length — sizes the next round's output-region snapshot
    # (round 1 falls back to output_probe_len, possibly 0 → no output point yet).
    last_out_len = int(output_probe_len)

    # State carried for the G2 convergence judgement.
    prev_gap_n: int | None = None          # distinct gap count of the prior round
    nonshrink_streak = 0                   # consecutive rounds the gap set did not shrink
    diffusion_streak = 0                   # consecutive non-shrink rounds that ALSO exploded groups
    prev_region_groups: int | None = None  # prior round's region-group count
    prev_gap_keys: set = set()             # prior round's gap keys (new-region detection)

    last_result: RecaptureLoopResult | None = None

    for r in range(1, max_rounds + 1):
        # 1. Build this round's observe points = output-region point + PLAN points.
        #    The plan is clustered (PC + contiguous range) into a few batch MEM points.
        plan_prov = _plan_as_provenance(plan)
        pre = observe_points_from_provenance(plan_prov, min_size=min_size)
        plan_points: list[ObservePoint] = list(pre.observe_points)
        out_point = _output_observe_point(output_observe_pc, sink_base, last_out_len)
        observe_points = ([out_point] if out_point is not None else []) + plan_points

        # 2. ONE rerun for the whole batch (output region + all plan points together).
        result = adapter.rerun(loop_input, observe_points)
        if not isinstance(result, RerunResult):
            raise TypeError(
                "adapter.rerun must return a RerunResult; got "
                f"{type(result).__name__} (the batch loop folds RerunResult.mem "
                "captures into snapshots via mem_snapshots_from_rerun)")

        # 3. THIS rerun's output is THIS round's expected (G1 nonce honesty). Its
        #    observations are the ONLY snapshots fed to trace_provenance this round —
        #    NEVER carrying a prior round's snapshots.
        expected = bytes(result.output)
        if not expected:
            # An empty output cannot be provenanced — honest stop, not a silent spin.
            detail = (
                f"round {r}: adapter.rerun returned an EMPTY output — cannot "
                "trace_provenance against an empty expected. Check the rerun wire / "
                "that loop_input actually drives a production.")
            final = last_result.final if last_result is not None else _empty_prov()
            return RecaptureLoopResult(
                outcome=LoopOutcome.STALLED, final=final, rounds=rounds,
                snapshots=[], residual=_shape_residual(final), detail=detail)
        last_out_len = len(expected)
        # G1 (Req2): every snapshot from THIS round is stamped with this round's
        # single-rerun execution token. The construct then guarantees same-execution
        # by construction (one rerun per round, snapshots never accumulate across
        # rounds — see module docstring) → the closing set carries ONE token →
        # assert_same_execution never misfires. The token is positive provenance the
        # G1 兜底 guard reads.
        exec_token = _round_exec_id(r, result)
        round_snaps = _stamp_execution(mem_snapshots_from_rerun(result), exec_token)

        prov = trace_provenance(
            items, expected, sink_base=sink_base, snapshots=round_snaps, **tkw)

        gaps_now = _gap_set(prov)
        region_groups = _count_region_groups(prov)
        new_region = bool(
            bool(gaps_now - prev_gap_keys)
            and bool(prev_gap_keys)
            and region_groups > (prev_region_groups or 0))

        rounds.append(LoopRound(
            round=r,
            verdict=prov.verdict.value,
            n_plan_points=len(plan_points),
            n_gaps_before=(prev_gap_n if prev_gap_n is not None else len(gaps_now)),
            n_gaps_after=len(gaps_now),
            n_unplaceable=len(pre.unplaceable_addrs),
            n_snapshots_fed=len(round_snaps),
            n_region_groups=region_groups,
            expected_len=len(expected),
            chain_n=len(prov.chain),
            backtrace_truncated=prov.backtrace_truncated is not None,
            new_region=new_region,
            truncated=bool(result.truncated),
        ))

        last_result = RecaptureLoopResult(
            outcome=LoopOutcome.STALLED, final=prov, rounds=rounds,
            snapshots=round_snaps)

        # 4. CLOSED — production visible (a buffer/stream verdict, or no gap left).
        if _is_closed(prov):
            # G1 (Req2) 兜底: before declaring a closure, assert the backing snapshot
            # set is ONE execution. In the normal construct this always holds (one
            # rerun per round → one token); if a caller-supplied / merged set ever
            # carried snapshots from >= 2 reruns, refuse the closure and report the
            # PROVENANCE_WATCH_BATCH_G1 terminal instead of a FALSE close (B2 makes
            # this true by construction; this is the loud兜底 alarm).
            g1 = assert_same_execution(round_snaps)
            if g1 is not None:
                return RecaptureLoopResult(
                    outcome=LoopOutcome.CROSS_RERUN_G1, final=prov, rounds=rounds,
                    snapshots=round_snaps, same_execution=g1,
                    detail=(
                        f"round {r}: provenance would close, but {g1['block_why']}. "
                        "Refusing the closure — re-capture the whole producer chain "
                        "in ONE rerun (the recapture loop's same-run plan), do not "
                        "stitch snapshots across executions."),
                )
            return RecaptureLoopResult(
                outcome=LoopOutcome.CLOSED, final=prov, rounds=rounds,
                snapshots=round_snaps,
                detail=(
                    f"provenance closed after {r} batch round(s): "
                    f"verdict={prov.verdict.value}, no remaining unobserved producer. "
                    f"All snapshots from the final round's single rerun (one nonce)."),
            )

        # 5. UNPLACEABLE — gaps remain but NONE can be hung on a reading PC, so no
        #    batch rerun can capture them. Explicit terminal with the shape.
        if not plan and not pre.observe_points and prov.next_watch and not any(
                w.get("pc") is not None for w in prov.next_watch):
            # First round produced only pc=None gaps: nothing is ever placeable.
            residual = _shape_residual(prov)
            return RecaptureLoopResult(
                outcome=LoopOutcome.UNPLACEABLE, final=prov, rounds=rounds,
                snapshots=round_snaps, residual=residual,
                detail=(
                    f"{len(prov.next_watch)} producer gap(s) remain but NONE can be "
                    f"hung on a reading PC (no observe point is placeable) — a batch "
                    f"rerun cannot capture them. Widen the trace so the reads enter "
                    f"it, or snapshot the buffer directly, then re-derive."),
            )

        # 6. Merge the new gaps into the PLAN (plan grows; snapshots never accumulate).
        for w in prov.next_watch:
            plan[_gap_key(w)] = w

        # 7. Convergence (G2). Shrink? reset the non-shrink streak. Rise/flat? only
        #    terminal on TWO CONSECUTIVE non-shrinking rounds, or broad diffusion.
        if prev_gap_n is not None:
            shrank = len(gaps_now) < prev_gap_n
            if shrank:
                nonshrink_streak = 0
            else:
                nonshrink_streak += 1

            # Diffusion (G2②): the watch plan spreading across an explosion of new
            # region groups (stack/table/heap with no single buffer to chase). A
            # SINGLE big group jump is NOT enough — a productive deep-dive rise (the
            # healthy 232→1375→… shape) also explodes the group count for one round
            # and then RECOVERS (shrinks), and must NEVER be killed. So diffusion is
            # terminal only when SUSTAINED: consecutive non-shrinking rounds that each
            # keep exploding the group count (it spreads and never recovers). A
            # transient rise breaks the streak the moment it shrinks.
            group_exploded = (
                not shrank
                and region_groups >= _DIFFUSION_REGION_FLOOR
                and prev_region_groups is not None
                and region_groups >= prev_region_groups * _DIFFUSION_JUMP_FACTOR)
            diffusion_streak = diffusion_streak + 1 if group_exploded else 0
            diffused = diffusion_streak >= 2

            if nonshrink_streak >= 2 or diffused:
                residual = _shape_residual(prov)
                why = ""
                if not round_snaps:
                    why = (" The last batch rerun yielded NO mem snapshots (the runner "
                           "captured nothing at the requested points — check the "
                           "mem-observe-point wire / whether the points are live).")
                if diffused:
                    why += (f" Watch plan DIFFUSED across {region_groups} region "
                            f"group(s) (was {prev_region_groups}) — spread into many "
                            f"regions with no single buffer to chase.")
                return RecaptureLoopResult(
                    outcome=LoopOutcome.STALLED, final=prov, rounds=rounds,
                    snapshots=round_snaps, residual=residual,
                    detail=(
                        f"batch loop STALLED after {r} round(s): the distinct "
                        f"producer-gap set did not converge "
                        f"(prev={prev_gap_n} → now={len(gaps_now)}; non-shrink "
                        f"streak={nonshrink_streak}). Residual = "
                        f"{residual['n_reading_pcs']} reading PC(s) / "
                        f"{residual['n_unhookable']} unhookable byte(s) over "
                        f"{residual['n_region_groups']} region group(s) (see "
                        f"``residual``).{why} Honest local-closure: production for "
                        f"the listed regions is still not visible."),
                )

        prev_gap_n = len(gaps_now)
        prev_region_groups = region_groups
        prev_gap_keys = gaps_now

    # Ran out of rounds. If the final round was still shrinking → BUDGET_EXHAUSTED
    # (honest: not stalled, out of budget). Otherwise the convergence guard already
    # returned; reaching here means budget hit mid-progress.
    prov = last_result.final if last_result is not None else _empty_prov()
    residual = _shape_residual(prov)
    return RecaptureLoopResult(
        outcome=LoopOutcome.BUDGET_EXHAUSTED, final=prov, rounds=rounds,
        snapshots=last_result.snapshots if last_result is not None else [],
        residual=residual,
        detail=(
            f"batch loop hit the {max_rounds}-round budget while the gap set was "
            f"still being worked (last: {rounds[-1].n_gaps_before} → "
            f"{rounds[-1].n_gaps_after}) — not stalled, out of budget. Raise "
            f"``max_rounds`` to continue converging, or accept this local closure. "
            f"Residual shape in ``residual``."),
    )


def _plan_as_provenance(plan: dict[tuple[Any, Any], dict]) -> ProvenanceResult:
    """Wrap the accumulated watch PLAN as a NEEDS_OBSERVATION ProvenanceResult so the
    existing :func:`observe_points_from_provenance` clusters it (PC + contiguous
    range) into batch observe points — reuse, not a parallel clusterer."""
    return ProvenanceResult(
        ProvenanceVerdict.NEEDS_OBSERVATION,
        next_watch=list(plan.values()),
        expected=b"\x00")


def _empty_prov() -> ProvenanceResult:
    """A placeholder NEEDS_OBSERVATION result for the (degenerate) zero-round path so
    ``final`` is never None — the verdict stays explicit."""
    return ProvenanceResult(ProvenanceVerdict.NEEDS_OBSERVATION, expected=b"\x00")


def _round_exec_id(round_no: int, result: RerunResult) -> str:
    """A single-execution provenance token for this round's ONE rerun (G1).

    One rerun = one execution = one token. The token folds the round number and the
    rerun's produced output (its nonce) so two reruns with different outputs get
    different tokens — making cross-rerun stitching DETECTABLE by
    :func:`assert_same_execution`. Deterministic / pure."""
    nonce = bytes(result.output)[:16].hex()
    return f"rerun#{round_no}:{nonce}"


def _stamp_execution(
    snaps: list[MemSnapshot], exec_id: str,
) -> list[MemSnapshot]:
    """Stamp this round's snapshots with the round's single-rerun execution token (G1).

    A snapshot that ALREADY carries an ``execution_id`` is left untouched — its
    provenance is its own, and overwriting it would MASK a snapshot that actually
    came from a different rerun (the very thing the G1 guard exists to catch). So the
    loop only stamps the freshly-captured (untokened) snapshots of THIS rerun; any
    pre-tokened snapshot keeps its token, and a set mixing two reruns then carries two
    distinct tokens → :func:`assert_same_execution` fires. By construction the loop
    feeds only this round's fresh captures, so the common path is ONE token (no
    caller obligation — feedback_construct_symmetry_not_caller_obligation)."""
    from dataclasses import replace
    return [s if s.execution_id is not None else replace(s, execution_id=exec_id)
            for s in snaps]


__all__ = [
    "LoopOutcome",
    "LoopRound",
    "RecaptureLoopResult",
    "run_recapture_loop",
    "assert_same_execution",
    "G1_SAME_EXECUTION_VIOLATED",
    "G1_SAME_EXECUTION_UNPROVABLE",
]
