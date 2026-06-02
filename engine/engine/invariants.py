"""Parity-invariant graph (capability_request.md §P1-2, M8).

The reference target hindsight: ``hook_src_valid: 626`` (a count > 0) coexisting
with ``hook_digest_eq_sign: 0%`` (a rate = 0) is *logically* impossible
— if every hooked input produces the wrong digest, the hook is on a
constant buffer and the "valid count" cell is measuring something else
(execution count, not capture success). The system tolerated the
contradiction because metrics were counted independently.

This module gives the engine an *invariant set*: simple boolean
predicates between metrics. When a parity report or gate JSON is
finalised, the engine evaluates each invariant; a violation flips
``invariants_failed`` on the report and (per M5) triggers the mandatory
ledger_patch state — the agent cannot push the report through.

The invariants here cover the patterns seen on the reference target. Adding a new
invariant is one register call; the predicate body is a plain lambda.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class Invariant:
    """One named contradiction-detector. ``predicate`` is True when the
    invariant HOLDS (no contradiction). False means the report self-
    contradicts and the engine must flag it."""

    name: str
    description: str
    # Returns False on violation. Predicate accepts the full report dict
    # so it can read whatever fields it needs.
    predicate: Callable[[dict[str, Any]], bool]
    # Human-readable diagnosis emitted when the invariant fails.
    failure_message: str


def _get(report: dict[str, Any], key: str, default: Any = None) -> Any:
    """Tolerant key access — looks at top-level and one level under
    ``"metrics"`` so the same predicates work whether the report uses a
    flat or nested layout."""
    if key in report:
        return report[key]
    metrics = report.get("metrics")
    if isinstance(metrics, dict) and key in metrics:
        return metrics[key]
    return default


# ---------------------------------------------------------------------------
# Default invariants (the canonical M8 set)
# ---------------------------------------------------------------------------


def _inv_hook_valid_but_no_match(r: dict[str, Any]) -> bool:
    """hook_valid > 0 AND eq_sign == 0 → invariant FAILS. The hook is
    capturing *something* but it never matches sign — almost always a
    constant buffer."""
    valid = _get(r, "hook_src_valid", 0) or 0
    eq    = _get(r, "hook_digest_eq_sign", None)
    if eq is None:
        eq = _get(r, "hook_eq_sign", None)
    if eq is None:
        return True   # can't evaluate; treat as OK
    try:
        valid_n = int(valid)
        eq_n = float(eq)
    except (TypeError, ValueError):
        return True
    if valid_n > 0 and eq_n == 0.0:
        return False
    return True


def _inv_input_len_constant_but_many_vectors(r: dict[str, Any]) -> bool:
    """sm3_input_len constant across N>=3 distinct inputs ⇒ dump is not
    the real per-input block (the reference target 68B appkey trap)."""
    lens = _get(r, "sm3_input_lens_seen") or _get(r, "input_lens_seen")
    vectors = _get(r, "vectors_total", 0) or _get(r, "vectors", 0) or 0
    if not lens or not isinstance(lens, (list, tuple)):
        return True
    try:
        vectors_n = int(vectors)
    except (TypeError, ValueError):
        return True
    if vectors_n >= 3 and len(set(lens)) == 1:
        return False
    return True


def _inv_pass_rate_but_no_evidence_class(r: dict[str, Any]) -> bool:
    """Any field name ending in ``_rate`` or ``_pass_rate`` exists but
    ``numeric_claims`` is empty / missing → policy violation per M1."""
    has_rate = any(
        isinstance(k, str) and (k.endswith("_rate") or k == "pass_rate" or k.endswith("_pct"))
        for k in (list(r.keys()) + list((r.get("metrics") or {}).keys()))
    )
    if not has_rate:
        return True
    claims = r.get("numeric_claims")
    if not claims:
        return False
    return True


def _inv_target_success_without_archival_allowed(r: dict[str, Any]) -> bool:
    """M6: a single boolean ``target_success`` MUST NOT appear without
    the structured ``archival_allowed`` conjunctive verdict."""
    if "target_success" in r and "archival_allowed" not in r:
        return False
    return True


def _inv_hook_digest_unique_count_below_threshold(r: dict[str, Any]) -> bool:
    """M3 hook sanity: vectors >= 3 → hook_digest_unique_count must be
    >= min(3, vectors / 2)."""
    vectors = _get(r, "vectors_total", 0) or _get(r, "vectors", 0) or 0
    uniq    = _get(r, "hook_digest_unique_count", None)
    if uniq is None:
        return True
    try:
        v_n = int(vectors); u_n = int(uniq)
    except (TypeError, ValueError):
        return True
    if v_n < 3:
        return True
    threshold = min(3, max(1, v_n // 2))
    return u_n >= threshold


DEFAULT_INVARIANTS: tuple[Invariant, ...] = (
    Invariant(
        name="hook_valid_but_no_match",
        description=(
            "hook_src_valid>0 paired with hook_digest_eq_sign=0 means the "
            "hook captures a constant blob — observation point is wrong."
        ),
        predicate=_inv_hook_valid_but_no_match,
        failure_message=(
            "hook_src_valid > 0 but hook_digest_eq_sign == 0 — observation "
            "point is on a constant buffer; mark `observation_invalid`."
        ),
    ),
    Invariant(
        name="input_len_constant_but_many_vectors",
        description=(
            "sm3_input_len is identical across N>=3 inputs — the dump is "
            "not the per-input block."
        ),
        predicate=_inv_input_len_constant_but_many_vectors,
        failure_message=(
            "sm3_input_len constant across >=3 vectors — dump is a "
            "template/appkey buffer, not real per-input compress input."
        ),
    ),
    Invariant(
        name="pass_rate_without_evidence_class",
        description=(
            "M1: any *_rate / *_pct field requires numeric_claims with "
            "evidence_class."
        ),
        predicate=_inv_pass_rate_but_no_evidence_class,
        failure_message=(
            "report carries pass-rate fields without `numeric_claims[]` — "
            "every numeric metric must list its evidence_class (M1)."
        ),
    ),
    Invariant(
        name="target_success_without_archival_allowed",
        description=(
            "M6: a single boolean `target_success` is forbidden; verdict "
            "must be the conjunctive `archival_allowed`."
        ),
        predicate=_inv_target_success_without_archival_allowed,
        failure_message=(
            "report sets `target_success` without the conjunctive "
            "`archival_allowed` field — replace single boolean with "
            "io ∧ triton_symbolic ∧ sm3_body_binary."
        ),
    ),
    Invariant(
        name="hook_digest_unique_count_below_threshold",
        description=(
            "M3: hook_digest_unique_count must be at least min(3, vectors/2) "
            "when vectors>=3, else the hook captures a constant buffer."
        ),
        predicate=_inv_hook_digest_unique_count_below_threshold,
        failure_message=(
            "hook_digest_unique_count fell below the min(3, vectors/2) "
            "threshold — INVALID_CONSTANT_BUFFER suspect."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def check_invariants(
    report: dict[str, Any],
    *,
    invariants: tuple[Invariant, ...] = DEFAULT_INVARIANTS,
) -> list[dict[str, str]]:
    """Return the list of failing invariants. Empty list = OK.

    Each entry is ``{"name": ..., "message": ..., "description": ...}``.
    Callers (gate emitter, parity reporter) write this into
    ``invariants_failed`` and fail the gate if non-empty.
    """
    failures: list[dict[str, str]] = []
    for inv in invariants:
        try:
            ok = inv.predicate(report)
        except Exception as e:
            failures.append({
                "name":        inv.name,
                "message":     f"invariant predicate raised {type(e).__name__}: {e}",
                "description": inv.description,
            })
            continue
        if not ok:
            failures.append({
                "name":        inv.name,
                "message":     inv.failure_message,
                "description": inv.description,
            })
    return failures


def annotate_report(report: dict[str, Any]) -> dict[str, Any]:
    """Run check_invariants and inject `invariants_failed` into the
    report dict (returns the same dict for chaining). When non-empty,
    the renderer is required to prefix the section title with
    ``INVALIDATED``."""
    failures = check_invariants(report)
    report["invariants_failed"] = failures
    return report
