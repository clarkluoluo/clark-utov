"""Anti-drift discipline reminders injected into LLM calls.

Per PLAN §12.3 / DECISIONS D-014:
  - Every LLM call gets a short suffix on the system prompt.
  - Every Nth call (default 20) gets the FULL PLAN §1 rule set re-injected.
  - Counter is per (target, run_id), not global.
  - Applies to the S6 hypothesis-generation LLM and to other stage-driving
    LLM calls. Does NOT apply to verifier or Triton (no LLM).
  - Blue-team has its own independent counter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SHORT_REMINDER = (
    "Discipline: hypotheses are not findings. Every claim must pass the verifier "
    "before being relied on. Return valid JSON only."
)

# PLAN §1 verbatim. Re-injected every FULL_REMINDER_EVERY calls.
FULL_REMINDER = """SYSTEM RULES (PLAN §1 — non-negotiable):
1. verifier is the only source of truth. Plugin / LLM / Triton outputs are
   ALWAYS unverified until verifier rules on them.
2. findings and hypotheses are strictly separate. Plugin output is high-prior
   but still must be verified; on failure it is demoted to a hypothesis.
3. every hypothesis is verified immediately. Never batch hypotheses for a single
   end-of-pipeline I/O check — that loses the ability to localize errors.
4. hypothesis tree is N-ary with backtracking. Multiple candidates per stuck
   point; sibling failure means backtrack, not silent prune.
5. LLM only consumes clean small inputs. Never raw trace. Pattern recognition,
   constant claims, algorithm naming only. No deterministic computation.
6. no time-axis binary-search. Reduction to relevant logic is done by backward
   data-flow slicing from the output, not by time-range bisection.
7. VMP and OLLVM are processed differently. Do not mix strategies.
"""

FULL_REMINDER_EVERY = 20


@dataclass
class DisciplineState:
    target: str
    run_id: str
    call_count: int = 0
    history: list[int] = field(default_factory=list)
    tracker: object | None = None    # optional Tracker — emits DISCIPLINE_REMINDER


def wrap_messages(state: DisciplineState, messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Append short reminder to system; every Nth call, prepend full reminder.
    When state has a tracker, also emit a DISCIPLINE_REMINDER event so external
    agents see the re-injection (PLAN §12.3 visible to driving agent)."""
    state.call_count += 1
    out = list(messages)
    if not out or out[0].get("role") != "system":
        out.insert(0, {"role": "system", "content": ""})
    out[0]["content"] = (out[0]["content"] + "\n\n" + SHORT_REMINDER).strip()

    if state.call_count % FULL_REMINDER_EVERY == 0:
        out.insert(0, {"role": "system", "content": FULL_REMINDER})
        state.history.append(state.call_count)
        if state.tracker is not None:
            try:
                from .progress import EventKind
                state.tracker.emit(    # type: ignore[attr-defined]
                    EventKind.DISCIPLINE_REMINDER,
                    call_count=state.call_count,
                    rule_text_preview=FULL_REMINDER[:200],
                )
            except Exception:
                pass
    return out
