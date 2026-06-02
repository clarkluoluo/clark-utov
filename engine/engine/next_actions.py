"""spec A — verdict → next_actions (declarative remedy mapping on failure results).

When utov already KNOWS a result is a known failure shape, it shouldn't just hand
back the verdict — it should attach the most relevant EXISTING helper (name + why +
minimal call) so the user stops hand-rolling what a helper already does. The
verdict/reason text is unchanged; this only attaches a machine-readable pointer.

Design (A8①, reuse don't rebuild): a small declarative registry maps
``(report_kind, verdict, reason-predicate) -> suggested helper metadata``. Reports
call :func:`suggest_next_actions` at their verdict-construction site. Adding a new
``(kind, verdict, reason)->helper`` entry needs NO per-call code (extensible).

Pure additive + advisory (A8③/④): the registry NEVER changes a verdict or a reason;
when no mapping matches it returns an empty tuple (the reason text still stands —
never a wrong/forced suggestion).

Seed mapping: ``ParityVectorReport.UNCLOSABLE`` with a distinct-collapse reason ->
``real_gold.collect_real_gold`` (repeated sampling to a distinct-output floor). This
is the TC2 ``99 distinct < 100`` proof-point.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


__all__ = [
    "NextActionRule",
    "NextActionRegistry",
    "REGISTRY",
    "register_next_action",
    "suggest_next_actions",
]


# A reason-predicate inspects a report's reason tuple (+ the report itself) and
# decides whether this rule's helper applies. Kept as a callable so a rule can match
# on a substring, a regex, a numeric threshold on report fields, or "always".
ReasonPredicate = Callable[[Sequence[str], Any], bool]


def reason_contains(substr: str) -> ReasonPredicate:
    """Predicate: any reason string contains ``substr`` (case-sensitive)."""
    def _pred(reasons: Sequence[str], _report: Any) -> bool:
        return any(substr in r for r in reasons)
    return _pred


def reason_matches(pattern: str) -> ReasonPredicate:
    """Predicate: any reason string matches the regex ``pattern``."""
    rx = re.compile(pattern)
    def _pred(reasons: Sequence[str], _report: Any) -> bool:
        return any(rx.search(r) for r in reasons)
    return _pred


def always(_reasons: Sequence[str], _report: Any) -> bool:
    """Predicate that always matches (kind+verdict alone select the helper)."""
    return True


@dataclass(frozen=True)
class NextActionRule:
    """One declarative ``(report_kind, verdict, reason-predicate) -> helper`` entry.

    ``helper`` is the dotted name of an EXISTING helper (e.g.
    ``"real_gold.collect_real_gold"``); ``why`` explains why THIS helper is the next
    action for this verdict shape; ``example`` is a minimal call snippet. The
    surfaced action dict is ``{"helper", "why", "example"}`` — exactly the spec
    contract. ``predicate`` further narrows within a (kind, verdict)."""

    report_kind: str
    verdict: str
    helper: str
    why: str
    example: str
    predicate: ReasonPredicate = always

    def action(self) -> dict[str, str]:
        return {"helper": self.helper, "why": self.why, "example": self.example}


class NextActionRegistry:
    """A declarative registry of :class:`NextActionRule`. Lookup is by
    ``(report_kind, verdict)`` then filtered by each rule's reason-predicate; every
    matching rule contributes one action. No match -> empty tuple (advisory only)."""

    def __init__(self) -> None:
        self._rules: list[NextActionRule] = []

    def register(self, rule: NextActionRule) -> None:
        self._rules.append(rule)

    def suggest(
        self,
        report_kind: str,
        verdict: str,
        reasons: Sequence[str],
        report: Any = None,
    ) -> tuple[dict[str, str], ...]:
        """Return the suggested actions for this verdict shape (possibly empty).

        Order-preserving (registration order) and de-duplicated on the action dict
        so re-registering an equivalent rule never double-surfaces."""
        out: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for rule in self._rules:
            if rule.report_kind != report_kind or rule.verdict != verdict:
                continue
            try:
                ok = rule.predicate(reasons, report)
            except Exception:
                # A misbehaving predicate must never break the verdict path
                # (advisory-only invariant) — it simply does not contribute.
                ok = False
            if not ok:
                continue
            act = rule.action()
            key = (act["helper"], act["why"], act["example"])
            if key in seen:
                continue
            seen.add(key)
            out.append(act)
        return tuple(out)


# ---------------------------------------------------------------------------
# The process-wide registry + the seed mapping.
# ---------------------------------------------------------------------------

REGISTRY = NextActionRegistry()


def register_next_action(
    *,
    report_kind: str,
    verdict: str,
    helper: str,
    why: str,
    example: str,
    predicate: ReasonPredicate = always,
) -> NextActionRule:
    """Add a ``(kind, verdict, reason)->helper`` mapping to the process registry.

    Returns the rule (handy for tests). No per-call code anywhere needs to change to
    make a newly-registered mapping fire — the verdict-construction sites already
    consult :func:`suggest_next_actions`."""
    rule = NextActionRule(
        report_kind=report_kind,
        verdict=verdict,
        helper=helper,
        why=why,
        example=example,
        predicate=predicate,
    )
    REGISTRY.register(rule)
    return rule


def suggest_next_actions(
    report_kind: str,
    verdict: str,
    reasons: Sequence[str],
    report: Any = None,
) -> tuple[dict[str, str], ...]:
    """Query the process registry — the function reports call at verdict construction."""
    return REGISTRY.suggest(report_kind, verdict, reasons, report)


# Report-kind constant for the parity report (matches ParityVectorReport.to_dict's
# ``"kind"``), kept here so producers and the registry agree on the string.
PARITY_VECTORS_KIND = "setup_symex_parity_vectors"


# Seed mapping (TC2 proof-point): a parity UNCLOSABLE whose independent side
# collapsed below the distinct floor -> collect_real_gold (drive reruns to a
# distinct-output floor, feeding real held-out vectors back to check_parity_vectors).
register_next_action(
    report_kind=PARITY_VECTORS_KIND,
    verdict="UNCLOSABLE",
    helper="real_gold.collect_real_gold",
    why=(
        "UNCLOSABLE = the INDEPENDENT side's observed collapsed below the distinct "
        "floor (a constant/near-constant gold trivially matches any predicted, "
        "whatever F is) — this is a COHORT problem, not an F problem. "
        "collect_real_gold drives runner reruns until the DISTINCT-OUTPUT floor is "
        "met, feeding real output-diverse held-out vectors back to "
        "check_parity_vectors so the independent side can carry >= min_vectors "
        "distinct outputs. Do NOT keep tuning F."
    ),
    example=(
        "collect_real_gold(adapter, observe_points, seeds, loop_input=loop_input, "
        "predict=emitted_transform, window=window, "
        "distinct_output_floor=report.min_vectors, exec_identity=trace_exec_id)"
    ),
    # Narrow within UNCLOSABLE to the distinct-collapse reason so an UNCLOSABLE that
    # ever arises from a different cause does not get a forced/wrong suggestion.
    predicate=reason_contains("distinct"),
)
