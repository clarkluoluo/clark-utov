"""Exception hierarchy for agent-friendly error signaling.

When the engine raises one of these, an agent driver knows what to do:
  - RecoverableError      → retry / tweak params and retry
  - NeedsUserDecisionError → present options to the user; pause until reply
  - FatalError            → setup / contract violation; do NOT retry
  - BudgetExceeded        → soft stop; ask user about raising budget OR finalize

All inherit from UtovError so callers can catch broadly when they want.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class UtovError(RuntimeError):
    """Base for all engine-raised errors."""


class RecoverableError(UtovError):
    """A transient failure. Caller may retry with same or adjusted params."""


class FatalError(UtovError):
    """Setup / contract / data corruption. Retry won't help. Stop the run."""


@dataclass
class UserDecisionOption:
    """One option an agent / human can pick to resolve a NeedsUserDecisionError."""
    label: str                  # short verb phrase, e.g. "raise_budget"
    description: str            # one-line human-readable
    suggested_command: str | None = None  # optional CLI to run


class NeedsUserDecisionError(UtovError):
    """Pipeline halted because a non-obvious choice must be made by user/agent.
    `options` lists the concrete next moves the caller can pick from."""
    def __init__(self, message: str, *, options: list[UserDecisionOption],
                 context: dict[str, Any] | None = None):
        super().__init__(message)
        self.options = options
        self.context = context or {}
