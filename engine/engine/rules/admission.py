"""Admission test (PLAN §14, DECISIONS D-023).

Replay a rule draft against the hypothesis ledger before registering it.

  - Positive set: prior verifier-pass instances of (kind, payload_shape).
    The draft must conclude the SAME thing as the historical pass for 100%
    of positives.
  - Negative set:
      (a) prior verifier-fail instances of this kind
      (b) instances of same kind with different payload_shape
    The draft must ABSTAIN (no conclusion) for 100% of negatives.

Any failure → REJECT. The draft is not registered. Caller can re-extract
later with more samples.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .promotion import PromotionCandidate, _payload_shape_hash


@dataclass(frozen=True)
class ReplayCase:
    label: str               # "pos" | "neg-fail" | "neg-different-shape"
    target_input: str        # "<target>/<input_hash>"
    hyp_id: int
    subject: str
    payload: dict[str, Any]
    expected_concludes: bool # True = positive set, False = negative set


@dataclass(frozen=True)
class ReplayReport:
    passed: bool
    positives_total: int
    positives_ok: int
    negatives_total: int
    negatives_ok: int
    failures: list[dict[str, Any]]  # first 10 problematic cases


def _iter_ledgers(work_root: Path) -> Iterable[Path]:
    yield from work_root.glob("*/*/runs/*/hypotheses.sqlite")


def _collect_cases(
    work_root: Path, candidate: PromotionCandidate,
) -> tuple[list[ReplayCase], list[ReplayCase]]:
    positives: list[ReplayCase] = []
    negatives: list[ReplayCase] = []
    for ledger_path in _iter_ledgers(work_root):
        try:
            run_dir = ledger_path.parent
            input_hash_dir = run_dir.parent.parent
            target_dir = input_hash_dir.parent
            ti = f"{target_dir.name}/{input_hash_dir.name}"
        except Exception:
            continue
        conn = sqlite3.connect(ledger_path)
        try:
            rows = conn.execute(
                "SELECT h.id, h.status, t.kind, h.subject, p.payload"
                " FROM hypotheses h"
                " JOIN claim_templates t ON h.template_id = t.id"
                " JOIN hyp_payloads p ON p.content_hash = t.payload_ref"
                " WHERE t.kind = ?",
                (candidate.kind,),
            ).fetchall()
        except sqlite3.OperationalError:
            continue
        finally:
            conn.close()
        for hyp_id, status, kind, subject, payload_json in rows:
            payload = json.loads(payload_json) if payload_json else {}
            shape = _payload_shape_hash(payload)
            if status == "passed" and shape == candidate.payload_shape_hash:
                positives.append(
                    ReplayCase(label="pos", target_input=ti, hyp_id=hyp_id,
                               subject=subject, payload=payload, expected_concludes=True)
                )
            elif status == "failed":
                negatives.append(
                    ReplayCase(label="neg-fail", target_input=ti, hyp_id=hyp_id,
                               subject=subject, payload=payload, expected_concludes=False)
                )
            elif shape != candidate.payload_shape_hash:
                negatives.append(
                    ReplayCase(label="neg-different-shape", target_input=ti, hyp_id=hyp_id,
                               subject=subject, payload=payload, expected_concludes=False)
                )
    return positives, negatives


def _evaluate_draft(draft: dict[str, Any], case: ReplayCase) -> bool | None:
    """Evaluate the draft's matcher against the case payload.

    Returns:
        True  — matcher fires AND would conclude something (positive prediction)
        False — matcher does NOT fire / abstain_when triggers (abstain)
        None  — cannot evaluate (treated as abstain by the caller for safety)

    The matcher in a draft is a free-text predicate. We support a few simple
    forms:
        "payload.has_key:foo"    → True if 'foo' in payload
        "payload.eq:foo=bar"     → True if payload[foo] == bar
        "*" or "" → always abstain (rejects the draft via boundary gate anyway)
    For more complex matchers, the rule fails open (abstain).
    """
    matcher = (draft.get("matcher") or "").strip()
    abstain_when = (draft.get("abstain_when") or "").strip()

    # Honor abstain_when first
    if abstain_when and _simple_predicate(abstain_when, case.payload):
        return False
    if matcher == "" or matcher == "*":
        return False
    return _simple_predicate(matcher, case.payload)


def _simple_predicate(expr: str, payload: dict[str, Any]) -> bool:
    """Very small DSL for rule predicates. Returns False on parse failure."""
    expr = expr.strip()
    if expr.startswith("payload.has_key:"):
        key = expr[len("payload.has_key:"):].strip()
        return key in payload
    if expr.startswith("payload.eq:"):
        rest = expr[len("payload.eq:"):]
        if "=" not in rest:
            return False
        k, v = rest.split("=", 1)
        k, v = k.strip(), v.strip()
        actual = payload.get(k)
        if actual is None:
            return False
        return str(actual) == v
    if expr.startswith("payload.contains:"):
        rest = expr[len("payload.contains:"):]
        if "=" not in rest:
            return False
        k, v = rest.split("=", 1)
        actual = payload.get(k.strip())
        return isinstance(actual, str) and (v.strip() in actual)
    return False


def run_replay(
    work_root: Path,
    candidate: PromotionCandidate,
    draft: dict[str, Any],
) -> ReplayReport:
    """Execute positive + negative replay; return PASS/FAIL + per-case detail."""
    positives, negatives = _collect_cases(work_root, candidate)
    failures: list[dict[str, Any]] = []
    pos_ok = 0
    for c in positives:
        got = _evaluate_draft(draft, c)
        if got is True:
            pos_ok += 1
        else:
            if len(failures) < 10:
                failures.append({"case": "positive", **{k: getattr(c, k) for k in
                    ("target_input", "hyp_id", "subject")}, "draft_concluded": got})
    neg_ok = 0
    for c in negatives:
        got = _evaluate_draft(draft, c)
        if got in (False, None):
            neg_ok += 1
        else:
            if len(failures) < 10:
                failures.append({"case": "negative", **{k: getattr(c, k) for k in
                    ("target_input", "hyp_id", "subject")}, "draft_concluded": got})
    passed = (pos_ok == len(positives) and neg_ok == len(negatives)
              and len(positives) > 0)
    return ReplayReport(
        passed=passed,
        positives_total=len(positives),
        positives_ok=pos_ok,
        negatives_total=len(negatives),
        negatives_ok=neg_ok,
        failures=failures,
    )
