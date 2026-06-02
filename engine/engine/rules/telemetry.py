"""Rule failure-rate monitor + auto-demote (PLAN §14, DECISIONS D-024).

Each call to a registered rule produces a verifier verdict (pass/fail/
inconclusive). We log to a rolling window and quarantine the rule when
the recent failure rate crosses threshold.

Window: WINDOW_SIZE most-recent calls per rule.
Quarantine trigger: failure rate > QUARANTINE_FAILURE_RATE in the window.

Quarantined rules: not served by Registry.match(); same kind falls back to
LLM-driven hypothesis generation until human review re-admits or revokes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

WINDOW_SIZE = 50              # D-024 default
QUARANTINE_FAILURE_RATE = 0.20


@dataclass
class RuleTelemetry:
    rule_id: str
    window: deque = field(default_factory=lambda: deque(maxlen=WINDOW_SIZE))
    # window stores booleans: True = verifier pass, False = verifier fail

    def record(self, verifier_passed: bool) -> None:
        self.window.append(verifier_passed)

    def failure_rate(self) -> float:
        if not self.window:
            return 0.0
        fails = sum(1 for x in self.window if not x)
        return fails / len(self.window)

    def should_quarantine(self) -> bool:
        return len(self.window) >= WINDOW_SIZE and self.failure_rate() > QUARANTINE_FAILURE_RATE
