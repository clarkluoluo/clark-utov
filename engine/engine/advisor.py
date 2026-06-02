"""Goal-aware advisor — a read-only unified evidence-state view + a NON-blocking
rule-based advisor (spec #6 ``spec_goal_aware_advisor.md``, clark 2026-06-03).

THIN SLICE (clark's explicit choice): no state machine, no ``PhaseRun``/gate
change. This module is two pure, read-only things:

  1. :func:`evidence_state` — composes the signals utov ALREADY produces
     (``closure_classification`` / ``cvd_ledger`` / ``authority_projection`` /
     ``progress.ProgressTracker`` snapshot) into one ``EvidenceState`` view. It
     RE-DERIVES NOTHING (A8①): it only reads the verdicts/counts/snapshots the
     caller hands it. A signal it cannot read becomes ``None`` + a ``sources``
     gap entry — never a fabricated state (A8④).

  2. :func:`advise` — a pure rule evaluator over ``(goal_spec, EvidenceState)``.
     Rules live in an EXTENSIBLE registry (:data:`RULES`); adding one is a single
     entry. Every advisory is ``blocking: false`` by construction (A8③): the
     advisor can suggest, never stop or re-route a run. No rule firing ⇒ an empty
     list, not a false "all good".

GENERAL-FIRST: ``goal_spec`` is the generic
``{goal, acceptance_criteria[], suggested_phase_order[]}`` shape (the ``TaskSpec``
concept, A8①). TC2's ``phase_E COLLECTED confirmed=0`` over-investment is only the
seeded rule's proof-point, NOT the design target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

# --------------------------------------------------------------------------- #
# The unified evidence-state view (read-only composition).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvidenceState:
    """One read-only frame of the run's evidence vs the goal.

    Every layered signal is a tri-state where a signal can be unknown:
    ``True``/``False`` when the source was readable, ``None`` when it was
    absent/unreadable (and then :attr:`sources` names the gap). Counts default to
    ``0``; ``pacing`` is the progress snapshot's pacing summary (``None`` when no
    progress source).

    :attr:`sources` is the provenance map — for each signal, the source module it
    came from, or a ``"gap: ..."`` marker when it could not be read. Nothing is
    ever fabricated: a missing source yields ``None`` + a gap, not a default
    verdict (A8④)."""

    sink_confirmed: bool | None = None
    boundary_explicit: bool | None = None
    parity_candidate_exists: bool | None = None
    recovery_confirmed_count: int = 0
    pacing: dict[str, Any] | None = None
    sources: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "evidence_state",
            "sink_confirmed": self.sink_confirmed,
            "boundary_explicit": self.boundary_explicit,
            "parity_candidate_exists": self.parity_candidate_exists,
            "recovery_confirmed_count": self.recovery_confirmed_count,
            "pacing": dict(self.pacing) if self.pacing is not None else None,
            "sources": dict(self.sources),
        }


def _read_closure(
    closure: Any, sources: dict[str, str],
) -> tuple[bool | None, bool | None, bool | None]:
    """Read sink/boundary/parity from a ``ClosureClassification``-like object.

    Reuses ``closure_classification``'s already-derived booleans
    (``output_sink_confirmed`` / ``provenance_closed`` / ``parity_exact``); it does
    NOT re-classify. ``None`` source → all three ``None`` + gap markers."""
    if closure is None:
        gap = "gap: no closure_classification supplied"
        sources["sink"] = gap
        sources["boundary"] = gap
        sources["parity"] = gap
        return None, None, None

    def _b(name: str) -> bool | None:
        if isinstance(closure, Mapping):
            v = closure.get(name)
        else:
            v = getattr(closure, name, None)
        return bool(v) if v is not None else None

    sink = _b("output_sink_confirmed")
    boundary = _b("provenance_closed")
    parity = _b("parity_exact")
    sources["sink"] = "closure_classification.output_sink_confirmed"
    sources["boundary"] = "closure_classification.provenance_closed"
    sources["parity"] = "closure_classification.parity_exact"
    return sink, boundary, parity


def _read_recovery(ledger: Any, sources: dict[str, str]) -> int:
    """Count confirmed recoveries from ledger entries (read-only).

    ``ledger`` is whatever the caller already pulled: a sequence of
    ``LedgerEntry``-like objects (we count those whose ``is_closed`` is truthy), or
    a ready integer count. ``None`` → 0 + a gap marker (an unread ledger is NOT a
    fabricated "zero confirmed", so the gap is explicit)."""
    if ledger is None:
        sources["recovery"] = "gap: no cvd_ledger supplied"
        return 0
    if isinstance(ledger, int):
        sources["recovery"] = "cvd_ledger (caller-supplied count)"
        return ledger
    count = 0
    for entry in ledger:
        closed = (entry.get("is_closed") if isinstance(entry, Mapping)
                  else getattr(entry, "is_closed", None))
        if closed:
            count += 1
    sources["recovery"] = "cvd_ledger.get_latest (closed entries)"
    return count


def _read_pacing(progress: Any, sources: dict[str, str]) -> dict[str, Any] | None:
    """Read the pacing summary from a ``ProgressSnapshot``-like object.

    Reuses ``progress.ProgressTracker``'s snapshot (``pacing`` /
    ``closure_rate_per_min``). ``None`` → ``None`` + gap marker."""
    if progress is None:
        sources["pacing"] = "gap: no progress snapshot supplied"
        return None

    def _g(name: str) -> Any:
        if isinstance(progress, Mapping):
            return progress.get(name)
        return getattr(progress, name, None)

    out: dict[str, Any] = {
        "pacing": _g("pacing"),
        "closure_rate_per_min": _g("closure_rate_per_min"),
    }
    spent = _g("spent_on_current_subline")
    if spent is not None:
        out["spent_on_current_subline"] = spent
    sources["pacing"] = "progress.ProgressTracker.snapshot"
    return out


def evidence_state(
    *,
    phase_run: Any = None,
    ledger: Any = None,
    claims: Any = None,
    progress: Any = None,
    closure: Any = None,
) -> EvidenceState:
    """Compose the unified, read-only evidence-state view from existing sources.

    All sources are optional — this is pure aggregation, NOT re-derivation (A8①):
    it reads the verdicts/counts/snapshots the caller already produced. Any source
    not supplied / not readable becomes ``None`` (or ``0`` for the count) plus an
    explicit ``sources`` gap, never a fabricated verdict (A8④).

    Args:
      phase_run: a ``PhaseRun``-like object (read-only); surfaced in ``sources`` as
        the phase-trail provenance. The thin slice reads no signal off it directly
        (the over-investment rule keys on pacing + closure), but it is recorded so
        the gap/provenance is honest when a future rule wants it.
      ledger: ``cvd_ledger`` entries already pulled (sequence of ``LedgerEntry``-
        like) or a ready count → ``recovery_confirmed_count``.
      claims: ``authority_projection.project_authority`` output (or its
        ``authoritative_claims``); recorded in ``sources`` (the current authority
        face), not collapsed into a single boolean in the thin slice.
      progress: a ``progress.ProgressTracker`` snapshot → ``pacing``.
      closure: a ``closure_classification.ClosureClassification`` (or dict) →
        ``sink_confirmed`` / ``boundary_explicit`` / ``parity_candidate_exists``.
    """
    sources: dict[str, str] = {}

    sink, boundary, parity = _read_closure(closure, sources)
    recovery = _read_recovery(ledger, sources)
    pacing = _read_pacing(progress, sources)

    if phase_run is None:
        sources["phase"] = "gap: no phase_run supplied"
    else:
        sources["phase"] = "phase_sequence.PhaseRun (trail)"

    if claims is None:
        sources["authority"] = "gap: no authority_projection supplied"
    else:
        sources["authority"] = "authority_projection.project_authority"

    return EvidenceState(
        sink_confirmed=sink,
        boundary_explicit=boundary,
        parity_candidate_exists=parity,
        recovery_confirmed_count=recovery,
        pacing=pacing,
        sources=sources,
    )


# --------------------------------------------------------------------------- #
# The advisor — pure rules over (goal_spec, EvidenceState). NEVER blocks.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Advisory:
    """A NON-blocking suggestion. ``blocking`` is ``False`` by construction — the
    advisor can suggest, never stop or re-route a run (A8③)."""

    level: str           # "SUGGEST"
    trigger: str         # the rule id that fired
    message: str
    rebalance_to: str | None = None
    blocking: bool = False   # invariant: always False (asserted in `advise`)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "trigger": self.trigger,
            "message": self.message,
            "rebalance_to": self.rebalance_to,
            "blocking": self.blocking,
        }


# A rule is a pure function (goal_spec, evidence) -> Advisory | None. ``None`` =>
# the rule did not fire (no false "all good"). The registry is the extension
# point: adding a rule = one entry here (or a runtime `register_rule`), proven by
# a test.
Rule = Callable[[Mapping[str, Any], EvidenceState], "Advisory | None"]


def _is_overinvested(evidence: EvidenceState) -> bool:
    """Pacing says the current subline is stalled / has spent a long time.

    Reads ONLY the pacing summary already produced by ``progress`` — no new
    timing. ``stalled`` pacing, OR an explicit ``spent_on_current_subline`` marked
    long, counts as over-invested. Unreadable pacing → not over-invested (we never
    fabricate an over-investment verdict)."""
    if evidence.pacing is None:
        return False
    if evidence.pacing.get("pacing") == "stalled":
        return True
    spent = evidence.pacing.get("spent_on_current_subline")
    if isinstance(spent, Mapping):
        return bool(spent.get("over_invested"))
    return False


def _far_from_acceptance(evidence: EvidenceState) -> bool:
    """No oracle closure achieved yet: nothing recovered AND parity not exact.

    ``parity_candidate_exists is True`` means the strong (oracle) criterion is
    already met → NOT far. An unknown (``None``) parity does not by itself mean
    far; far requires confirmed-recovery==0 AND parity not exactly True."""
    if evidence.recovery_confirmed_count > 0:
        return False
    return evidence.parity_candidate_exists is not True


def rule_subline_overinvested_far_from_acceptance(
    goal_spec: Mapping[str, Any], evidence: EvidenceState,
) -> "Advisory | None":
    """The seeded over-investment rule (spec proof-point TC2 phase_E confirmed=0).

    Fires a NON-blocking SUGGEST when the current subline is over-invested
    (pacing stalled / long-spent) AND the run is far from acceptance (no confirmed
    recovery and parity not exact). Suggests rebalancing toward a boundary-explicit
    Python candidate + held-out parity main line — it does NOT re-route."""
    if not (_is_overinvested(evidence) and _far_from_acceptance(evidence)):
        return None
    goal = goal_spec.get("goal") or "the goal"
    return Advisory(
        level="SUGGEST",
        trigger="subline_overinvested_far_from_acceptance",
        message=(
            f"current subline is over-invested but far from acceptance for "
            f"{goal!r} (no confirmed recovery, parity not exact) → consider going "
            f"back to the Python candidate + held-out parity main line"
        ),
        rebalance_to="boundary_explicit_candidate",
        blocking=False,
    )


# The seeded registry. Extend by appending an entry (or `register_rule`).
RULES: list[Rule] = [
    rule_subline_overinvested_far_from_acceptance,
]


def register_rule(rule: Rule) -> None:
    """Append a rule to the registry (the extension point — one call adds a rule).

    Provided so a caller can extend the advisor without editing this module; the
    seeded :data:`RULES` list is itself the static form of the same registry."""
    RULES.append(rule)


def advise(
    goal_spec: Mapping[str, Any],
    evidence: EvidenceState,
    *,
    rules: Sequence[Rule] | None = None,
) -> list[Advisory]:
    """Evaluate every rule over ``(goal_spec, evidence)`` → list of advisories.

    Pure, NON-blocking (A8③): every returned advisory has ``blocking is False``
    (asserted). No rule firing ⇒ an empty list, NOT a false "all good" (A8④).
    ``rules`` defaults to the module registry :data:`RULES`."""
    registry = list(RULES if rules is None else rules)
    out: list[Advisory] = []
    for rule in registry:
        adv = rule(goal_spec, evidence)
        if adv is None:
            continue
        assert adv.blocking is False, (
            f"advisory from {getattr(rule, '__name__', rule)!r} must be "
            f"non-blocking (advisor never blocks a run)"
        )
        out.append(adv)
    return out


__all__ = [
    "EvidenceState",
    "evidence_state",
    "Advisory",
    "Rule",
    "RULES",
    "register_rule",
    "advise",
    "rule_subline_overinvested_far_from_acceptance",
]
