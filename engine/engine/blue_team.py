"""Blue-team adversarial review of high-impact findings (PLAN §12.2, D-013).

Triggered when:
  - verifier passed
  - finding is high-impact (algo_signature / handler_semantic_batch / s5_root_rewrite)
  - hyp tree depth >= 2

Spawns an LLM call with an INDEPENDENT context (fresh system prompt, no history,
fresh DisciplineState). Three outcomes:
  approve   → finding upgraded
  challenge → finding stays pending; verifier runs one more round with broader inputs
  error/timeout → fall back to approve (verifier already passed)

Blue-team cannot prune. Verifier is still the only arbiter.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .discipline import DisciplineState, wrap_messages
from .llm_client import LLMClient


class BlueTeamVerdict(str, Enum):
    APPROVE = "approve"
    CHALLENGE = "challenge"


@dataclass(frozen=True)
class BlueTeamResult:
    verdict: BlueTeamVerdict
    rationale: str
    suggested_extra_inputs: list[bytes]
    suggested_observation_points: list[dict[str, Any]]


_HIGH_IMPACT_KINDS = frozenset({
    "algo_signature",
    "handler_semantic_batch",
    "s5_root_rewrite",
})

_SYSTEM_PROMPT = (
    "You are a SKEPTICAL adversarial reviewer for a reverse-engineering finding. "
    "Another LLM (the proposer) and a deterministic verifier have agreed on this "
    "claim. Your one and only job: look for ways the verifier might have been "
    "FOOLED — e.g. coincidental input coverage, fingerprint collisions with "
    "BLAKE2s/BLAKE2b that share IVs with SHA-256/SHA-512, or pattern matches "
    "that fired on ALU coincidences. "
    "If you find a plausible attack, return 'challenge' with concrete suggestions "
    "of (a) additional inputs that would expose the flaw and (b) observation "
    "points that would catch a wrong intermediate value. "
    "Otherwise return 'approve'. "
    "You CANNOT prune the finding — only request more verification. "
    "Be specific. No hedging. No 'consider also...'."
)

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict":    {"type": "string", "enum": ["approve", "challenge"]},
        "rationale":  {"type": "string"},
        "suggested_extra_inputs":     {"type": "array", "items": {"type": "string"}},
        "suggested_observation_points": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["verdict", "rationale"],
}


def should_review(finding_kind: str, hyp_depth: int) -> bool:
    """High-impact gate (D-013)."""
    if hyp_depth < 2:
        return False
    return finding_kind in _HIGH_IMPACT_KINDS


def review(
    finding: dict[str, Any],
    trace_excerpt: str,
    related_findings: list[dict[str, Any]],
    *,
    llm: LLMClient | None = None,
) -> BlueTeamResult:
    """Run the blue-team review using an isolated LLM context.

    Falls back to APPROVE if the LLM is unreachable / returns malformed output —
    blue-team is supplementary, not a gate.
    """
    if llm is None:
        try:
            llm = LLMClient()
        except Exception:
            return BlueTeamResult(
                verdict=BlueTeamVerdict.APPROVE,
                rationale="blue-team disabled: no LLM configured",
                suggested_extra_inputs=[],
                suggested_observation_points=[],
            )

    discipline = DisciplineState(target="blue-team", run_id="independent")
    related_summary = ", ".join(
        f"{f.get('kind')}:{f.get('subject')}" for f in related_findings[:5]
    ) or "(none)"
    user_msg = (
        f"FINDING UNDER REVIEW\n"
        f"  kind:    {finding.get('kind')}\n"
        f"  subject: {finding.get('subject')}\n"
        f"  payload: {finding.get('payload')}\n"
        f"  related: {related_summary}\n\n"
        f"TRACE EXCERPT (sliced + simplified):\n{trace_excerpt}\n\n"
        f"Decide approve|challenge per the system instructions."
    )
    messages = wrap_messages(
        discipline,
        [{"role": "system", "content": _SYSTEM_PROMPT},
         {"role": "user",   "content": user_msg}],
    )
    sys_text = "\n".join(m["content"] for m in messages if m["role"] == "system")
    usr_text = "\n".join(m["content"] for m in messages if m["role"] == "user")

    # Reuse generate_hypotheses scaffolding — the schema mismatch is OK because
    # we only care about parsing JSON; we hand-extract verdict afterwards.
    try:
        import json
        resp = llm.client.chat.completions.create(
            model=llm.model,
            messages=[{"role": "system", "content": sys_text},
                      {"role": "user",   "content": usr_text
                                          + "\n\nReturn JSON matching this schema:\n"
                                          + json.dumps(_RESPONSE_SCHEMA)}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        text = resp.choices[0].message.content or "{}"
        obj = json.loads(text)
    except Exception as e:
        return BlueTeamResult(
            verdict=BlueTeamVerdict.APPROVE,
            rationale=f"blue-team unreachable, defaulting to approve: {type(e).__name__}",
            suggested_extra_inputs=[],
            suggested_observation_points=[],
        )

    verdict = BlueTeamVerdict(obj.get("verdict", "approve"))
    extra_inputs: list[bytes] = []
    for s in obj.get("suggested_extra_inputs", [])[:8]:
        try:
            extra_inputs.append(bytes.fromhex(s.removeprefix("0x")))
        except ValueError:
            continue
    obs_points = obj.get("suggested_observation_points", [])[:8]
    return BlueTeamResult(
        verdict=verdict,
        rationale=obj.get("rationale", ""),
        suggested_extra_inputs=extra_inputs,
        suggested_observation_points=obs_points,
    )
