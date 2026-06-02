"""Hypothesis → rule promotion pipeline (PLAN §14, DECISIONS D-022).

Three gates (ALL required to register a rule):
  1. Cross-sample threshold: same (kind, payload_shape_hash) verified pass
     in >= CROSS_SAMPLE_THRESHOLD distinct (target, input_hash) pairs.
  2. Formalizable: the LLM-extracted rule doesn't contain semantic hedges,
     and the matcher reads as a concrete predicate.
  3. Boundary declared: extracted rule has explicit applicable_when AND
     abstain_when predicates, both non-empty, no wildcards.

If all three pass → produce a Rule draft → run admission.run_replay() →
register via registry.Registry.register().
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..llm_client import LLMClient

CROSS_SAMPLE_THRESHOLD = 3

SEMANTIC_HEDGES = (
    "depending on", "depends on", "in some cases", "usually", "sometimes",
    "typically", "may or may not", "context-sensitive", "case-by-case",
    "it depends", "varies", "sort of",
)


@dataclass(frozen=True)
class PromotionCandidate:
    kind: str
    payload_shape_hash: str
    sample_targets: list[str]          # distinct (target, input_hash) pairs
    sample_hyp_ids: list[tuple[Path, int]]   # (ledger_path, hyp_id)
    pass_count: int


def _payload_shape_hash(payload: dict[str, Any]) -> str:
    """Hash the SHAPE of the payload (keys, not values) so two hyps with
    different values but same structure group together."""
    keys = sorted(payload.keys())
    return hashlib.sha1(",".join(keys).encode("utf-8")).hexdigest()[:12]


def _iter_ledgers(work_root: Path) -> Iterable[Path]:
    """Yield every hypotheses.sqlite under work_root."""
    yield from work_root.glob("*/*/runs/*/hypotheses.sqlite")


def find_promotion_candidates(work_root: Path) -> list[PromotionCandidate]:
    """Scan all hypothesis ledgers under work_root for (kind, payload_shape)
    groups that have >= CROSS_SAMPLE_THRESHOLD distinct (target, input_hash)
    verifier-PASS instances.

    Schema D-027: hyp payload lives in hyp_payloads keyed by content_hash; the
    template row gives us (kind, source, payload_ref). We compute the SHAPE
    hash from the payload's keyset (not content) for grouping across slightly
    differing values.
    """
    by_group: dict[tuple[str, str], dict[str, Any]] = {}

    for ledger_path in _iter_ledgers(work_root):
        try:
            run_dir = ledger_path.parent
            input_hash_dir = run_dir.parent.parent
            target_dir = input_hash_dir.parent
            target_name = target_dir.name
            input_hash = input_hash_dir.name
        except Exception:
            continue
        sample_key = f"{target_name}/{input_hash}"

        conn = sqlite3.connect(ledger_path)
        try:
            rows = conn.execute(
                "SELECT h.id, t.kind, p.payload"
                " FROM hypotheses h"
                " JOIN claim_templates t ON h.template_id = t.id"
                " JOIN hyp_payloads p ON p.content_hash = t.payload_ref"
                " WHERE h.status = 'passed'"
            ).fetchall()
        except sqlite3.OperationalError:
            # Old-schema DB; skip
            continue
        finally:
            conn.close()

        for hyp_id, kind, payload_json in rows:
            payload = json.loads(payload_json) if payload_json else {}
            shape = _payload_shape_hash(payload)
            key = (kind, shape)
            g = by_group.setdefault(key, {
                "samples": set(),
                "hyp_ids": [],
                "count": 0,
            })
            g["samples"].add(sample_key)
            g["hyp_ids"].append((ledger_path, hyp_id))
            g["count"] += 1

    candidates: list[PromotionCandidate] = []
    for (kind, shape), g in by_group.items():
        if len(g["samples"]) >= CROSS_SAMPLE_THRESHOLD:
            candidates.append(PromotionCandidate(
                kind=kind,
                payload_shape_hash=shape,
                sample_targets=sorted(g["samples"]),
                sample_hyp_ids=g["hyp_ids"],
                pass_count=g["count"],
            ))
    return candidates


_RULE_EXTRACT_PROMPT = (
    "You are summarizing VERIFIED FACTS into a deterministic rule. "
    "You have multiple verifier-stamped hypothesis instances of the same kind. "
    "Produce a rule that fires only when its matcher predicate evaluates true, "
    "and ABSTAINS otherwise. Do NOT invent. Do NOT hedge ('usually', 'in some "
    "cases', etc.). If you cannot give clean predicates, return abstain_when='*' "
    "and applicable_when='' (will be rejected by gate)."
)

_DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "matcher":         {"type": "string"},
        "conclusion":      {"type": "string"},
        "applicable_when": {"type": "string"},
        "abstain_when":    {"type": "string"},
        "applicability_tags": {"type": "array", "items": {"type": "string"}},
        "confidence":      {"type": "number"},
    },
    "required": ["matcher", "conclusion", "applicable_when", "abstain_when",
                 "applicability_tags", "confidence"],
}


def extract_rule_draft(
    candidate: PromotionCandidate,
    *,
    llm: LLMClient | None = None,
) -> dict[str, Any] | None:
    """Feed verifier-stamped instances to LLM, ask for {matcher, conclusion,
    applicable_when, abstain_when, applicability_tags}."""
    if llm is None:
        try:
            llm = LLMClient()
        except Exception:
            return None

    # Pull a few sample payloads.
    samples_payload: list[dict[str, Any]] = []
    for ledger_path, hyp_id in candidate.sample_hyp_ids[:8]:
        conn = sqlite3.connect(ledger_path)
        try:
            row = conn.execute(
                "SELECT h.subject, p.payload"
                " FROM hypotheses h"
                " JOIN claim_templates t ON h.template_id = t.id"
                " JOIN hyp_payloads p ON p.content_hash = t.payload_ref"
                " WHERE h.id = ?",
                (hyp_id,),
            ).fetchone()
        finally:
            conn.close()
        if row:
            samples_payload.append({"subject": row[0], "payload": json.loads(row[1])})

    user_msg = (
        f"Kind: {candidate.kind}\n"
        f"Payload shape hash: {candidate.payload_shape_hash}\n"
        f"Cross-sample evidence ({len(candidate.sample_targets)} distinct targets, "
        f"{candidate.pass_count} verifier-PASS instances):\n"
        f"{json.dumps(samples_payload, indent=2)}\n\n"
        f"Generalize this into one rule per the schema."
    )

    try:
        resp = llm.client.chat.completions.create(
            model=llm.model,
            messages=[{"role": "system", "content": _RULE_EXTRACT_PROMPT},
                      {"role": "user",   "content": user_msg
                                          + "\n\nReturn JSON matching this schema:\n"
                                          + json.dumps(_DRAFT_SCHEMA)}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        text = resp.choices[0].message.content or "{}"
        return json.loads(text)
    except Exception:
        return None


def passes_formalization_gate(draft: dict) -> bool:
    """Return True iff draft has no semantic hedges and matcher is concrete."""
    blob = " ".join(str(v) for v in draft.values()).lower()
    if any(h in blob for h in SEMANTIC_HEDGES):
        return False
    return bool(draft.get("matcher", "").strip())


def passes_boundary_gate(draft: dict) -> bool:
    """Return True iff draft has non-empty applicable_when AND abstain_when,
    no wildcards, and at least one applicability_tag."""
    aw = (draft.get("applicable_when") or "").strip()
    bw = (draft.get("abstain_when") or "").strip()
    tags = draft.get("applicability_tags") or []
    if not aw or not bw:
        return False
    if aw == "*" or bw == "*":
        return False
    if not tags:
        return False
    return True
