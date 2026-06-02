"""Progress event bus + cost/progress combined snapshot (PLAN §15.1, §15.4).

Pipeline emits typed events through Tracker. SDK consumers subscribe by
event kind. Orchestrators read aggregate snapshots for "X spent / Y left"
displays. A simple rate-drop policy hook is included for proactive pauses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .cost import CostMeter, CostSnapshot


class EventKind(str, Enum):
    STAGE_START   = "stage_start"
    STAGE_DONE    = "stage_done"
    HYP_OPENED    = "hyp_opened"        # new pending node
    HYP_VERIFIED  = "hyp_verified"      # pass/fail/inconclusive
    HYP_CLOSED    = "hyp_closed"        # finalized (passed → finding or abandoned)
    FINDING_ADDED = "finding_added"
    BUDGET_WARN   = "budget_warn"
    PAUSE_REQUEST = "pause_request"     # rate-drop / impending big spend
    PIPELINE_DONE = "pipeline_done"

    # --- Agent-friendly events (PLAN §15 / agent SDK contract) ---
    # When you see one of these, present a decision to the user OR auto-decide
    # based on policy. They carry `options` in `event.detail["options"]`.
    ASK_USER_BUDGET_OVERRUN  = "ask_user.budget_overrun"
    ASK_USER_DEGRADED_RESULT = "ask_user.degraded_result"
    ASK_USER_NO_FINGERPRINT  = "ask_user.no_fingerprint"
    ASK_USER_BLUE_TEAM       = "ask_user.blue_team_needed"
    ASK_USER_BACKTRACK_LIMIT = "ask_user.backtrack_limit"
    ASK_USER_RUNNER_SLOW     = "ask_user.runner_slow"

    SAFE_INTERRUPT_POINT     = "safe_interrupt_point"   # checkpoint flushed, OK to ^C
    DISCIPLINE_REMINDER      = "discipline_reminder"    # §12.3 anti-drift, surfaced to agent


@dataclass(frozen=True)
class ProgressEvent:
    kind: EventKind
    timestamp: float           # monotonic seconds since tracker init
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressSnapshot:
    """What a UI/agent shows in one frame."""
    closures: int
    pending: int
    findings: int
    cost: CostSnapshot
    eta_closures: int | None = None
    closure_rate_per_min: float | None = None
    backtracks_per_min: float | None = None
    pacing: str = "ok"        # "ok" | "slowing" | "stalled"

    def as_human(self) -> str:
        lines = [
            f"progress: closed {self.closures}  pending {self.pending}  "
            f"findings {self.findings}  pacing: {self.pacing}",
        ]
        if self.closure_rate_per_min is not None:
            lines.append(f"  rate: {self.closure_rate_per_min:.2f} closures/min")
        if self.eta_closures is not None:
            lines.append(f"  eta: ~{self.eta_closures} more closures")
        lines.append("  " + self.cost.as_human().replace("\n", "\n  "))
        return "\n".join(lines)


class Tracker:
    """Event bus + aggregate stats. Cheap; one per Core instance."""

    def __init__(self, meter: CostMeter):
        self.meter = meter
        self._started = time.monotonic()
        self._listeners: dict[EventKind, list[Callable[[ProgressEvent], None]]] = {}
        self._wildcard: list[Callable[[ProgressEvent], None]] = []
        self.closures = 0
        self.pending = 0
        self.findings = 0
        self.backtracks = 0
        # Sliding window of closure timestamps for rate computation.
        self._closure_times: list[float] = []
        self._backtrack_times: list[float] = []
        # Pause-request listeners (PLAN §15.2): orchestrator answers continue|stop.
        self._pause_handler: Callable[[ProgressSnapshot, str], bool] | None = None

    # --- subscription ---

    def on(self, kind: EventKind, cb: Callable[[ProgressEvent], None]) -> None:
        self._listeners.setdefault(kind, []).append(cb)

    def on_any(self, cb: Callable[[ProgressEvent], None]) -> None:
        self._wildcard.append(cb)

    def set_pause_handler(self, cb: Callable[[ProgressSnapshot, str], bool]) -> None:
        """Handler returns True to continue, False to stop. PLAN §15.2."""
        self._pause_handler = cb

    # --- emit ---

    def emit(self, kind: EventKind, **detail: Any) -> ProgressEvent:
        evt = ProgressEvent(kind=kind, timestamp=time.monotonic() - self._started, detail=detail)
        now = evt.timestamp
        if kind == EventKind.HYP_OPENED:
            self.pending += 1
        elif kind == EventKind.HYP_CLOSED:
            self.pending = max(0, self.pending - 1)
            self.closures += 1
            self._closure_times.append(now)
            self.meter.record_closure()
        elif kind == EventKind.FINDING_ADDED:
            self.findings += 1
        elif kind == EventKind.HYP_VERIFIED:
            if detail.get("verdict") == "fail":
                self.backtracks += 1
                self._backtrack_times.append(now)
        for cb in self._listeners.get(kind, []):
            try:
                cb(evt)
            except Exception:
                pass
        for cb in self._wildcard:
            try:
                cb(evt)
            except Exception:
                pass
        return evt

    # --- snapshot ---

    def snapshot(self, eta_closures: int | None = None) -> ProgressSnapshot:
        now = time.monotonic() - self._started
        cost = self.meter.snapshot()
        rate = _rate_per_min(self._closure_times, now, window_s=60.0)
        btr  = _rate_per_min(self._backtrack_times, now, window_s=60.0)
        pacing = self._classify_pacing(rate, btr, cost)
        if eta_closures is not None:
            cost = self.meter.set_eta_from_remaining(
                eta_closures, uncertainty="high" if pacing != "ok" else "low",
            )
        return ProgressSnapshot(
            closures=self.closures,
            pending=self.pending,
            findings=self.findings,
            cost=cost,
            eta_closures=eta_closures,
            closure_rate_per_min=rate,
            backtracks_per_min=btr,
            pacing=pacing,
        )

    def _classify_pacing(self, closure_rate: float | None,
                         backtrack_rate: float | None,
                         cost: CostSnapshot) -> str:
        # Heuristic: "stalled" if we have nonzero spend but ~zero closures
        # for over a minute. "slowing" if backtrack rate is high vs closure
        # rate. "ok" otherwise.
        if closure_rate is None:
            # Not enough history — be optimistic for the first minute.
            return "ok" if cost.calls < 10 else "slowing"
        if closure_rate < 0.05 and cost.calls > 5:
            return "stalled"
        if backtrack_rate is not None and backtrack_rate > (closure_rate * 2):
            return "slowing"
        return "ok"

    # --- proactive pause request (PLAN §15.2) ---

    def request_pause(self, reason: str, eta_closures: int | None = None) -> bool:
        """Ask the registered handler whether to continue. Returns True if we
        should continue, False if the orchestrator should bail."""
        snap = self.snapshot(eta_closures=eta_closures)
        self.emit(EventKind.PAUSE_REQUEST, reason=reason,
                  snapshot=snap.__dict__)
        if self._pause_handler is None:
            # No handler attached — default policy: ALWAYS bail when pacing is
            # stalled, otherwise continue. Caller can override by attaching a
            # handler that prompts the user.
            return snap.pacing != "stalled"
        try:
            return bool(self._pause_handler(snap, reason))
        except Exception:
            return True  # fail open — don't accidentally stop on handler bug


def _rate_per_min(events: list[float], now: float, window_s: float = 60.0) -> float | None:
    if not events:
        return None
    cutoff = now - window_s
    recent = [t for t in events if t >= cutoff]
    if not recent:
        return None
    span = max(now - recent[0], 1e-6)
    return (len(recent) / span) * 60.0
