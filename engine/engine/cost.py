"""Cost accounting + budget gates (PLAN §15).

Tracks every LLM call's input/output tokens against per-model pricing, exposes
real-time spend + burn-rate, and trips a BudgetExceeded exception when any
user-configured limit (tokens / USD / wall time / calls) is hit.

Pricing is configurable — defaults below are best-known DeepSeek list prices
as of early 2026. If pricing drifts, the user can override via Budget(prices=...).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable


# --- pricing -----------------------------------------------------------------

@dataclass(frozen=True)
class ModelPricing:
    name: str
    usd_per_million_input: float
    usd_per_million_output: float


# Conservative ballpark — verify against your provider invoice.
DEFAULT_PRICING: dict[str, ModelPricing] = {
    # DeepSeek list prices (deepseek.com). May be lower with cache-hit discount.
    "deepseek-chat":     ModelPricing("deepseek-chat",     0.27, 1.10),
    "deepseek-reasoner": ModelPricing("deepseek-reasoner", 0.55, 2.19),
    # Self-hosted MiMo: assume free (compute is yours).
    "mimo-7b-rl":        ModelPricing("mimo-7b-rl",        0.00, 0.00),
}


def usd_cost(model: str, input_tokens: int, output_tokens: int,
             pricing: dict[str, ModelPricing] | None = None) -> float:
    pricing = pricing or DEFAULT_PRICING
    p = pricing.get(model)
    if p is None:
        # Unknown model — bill at DeepSeek-V3 rates conservatively.
        p = pricing.get("deepseek-chat") or ModelPricing(model, 0.27, 1.10)
    return (input_tokens / 1_000_000.0) * p.usd_per_million_input + \
           (output_tokens / 1_000_000.0) * p.usd_per_million_output


# --- budget + meter ----------------------------------------------------------

_BUDGET_AXES = (
    "input_tokens", "output_tokens", "total_tokens",
    "usd", "calls", "wall_seconds",
)


class BudgetExceeded(RuntimeError):
    """Raised by CostMeter.charge() when any Budget limit is exceeded.

    BR-4 §E: carries `axis` so downstream agents/UI can render which budget
    tripped without parsing the message string. One of `_BUDGET_AXES`.
    """

    def __init__(self, message: str, axis: str | None = None):
        super().__init__(message)
        self.axis = axis


@dataclass
class Budget:
    """User-set ceiling. Any None field = no limit on that axis."""
    max_input_tokens:  int   | None = None
    max_output_tokens: int   | None = None
    max_total_tokens:  int   | None = None
    max_usd:           float | None = None
    max_calls:         int   | None = None
    max_wall_seconds:  float | None = None
    pricing: dict[str, ModelPricing] = field(default_factory=lambda: dict(DEFAULT_PRICING))


@dataclass
class CostSnapshot:
    input_tokens:  int = 0
    output_tokens: int = 0
    total_tokens:  int = 0
    usd:           float = 0.0
    calls:         int = 0
    wall_seconds:  float = 0.0
    by_model:      dict[str, dict[str, float]] = field(default_factory=dict)

    # Derived (only meaningful when a "unit of progress" exists):
    closures:      int = 0
    tokens_per_closure: float | None = None    # burn rate
    usd_per_closure:    float | None = None
    eta_tokens:   int | None = None             # estimated tokens to finish
    eta_usd:      float | None = None
    eta_uncertainty: str | None = None          # "low" | "high" — honest label

    def as_human(self) -> str:
        line1 = (f"tokens={self.total_tokens:,} (in {self.input_tokens:,}, "
                 f"out {self.output_tokens:,})  calls={self.calls}  "
                 f"usd=${self.usd:.4f}  wall={self.wall_seconds:.1f}s")
        if self.closures > 0 and self.tokens_per_closure is not None:
            line2 = (f"  closures={self.closures}  "
                     f"~{int(self.tokens_per_closure):,} tok/closure  "
                     f"~${self.usd_per_closure:.4f}/closure")
            if self.eta_tokens is not None:
                line2 += f"  eta ~{self.eta_tokens:,} tok / ~${self.eta_usd:.4f}"
                if self.eta_uncertainty:
                    line2 += f" (uncertainty: {self.eta_uncertainty})"
            return line1 + "\n" + line2
        return line1


class CostMeter:
    """Thread-safe cumulative cost meter. The LLM client charges into this
    every call; orchestrators read snapshots for display + budget gating."""

    def __init__(self, budget: Budget | None = None):
        self.budget = budget or Budget()
        self._started = time.monotonic()
        self._lock = threading.Lock()
        self._input_tokens  = 0
        self._output_tokens = 0
        self._usd           = 0.0
        self._calls         = 0
        self._by_model: dict[str, dict[str, float]] = {}
        self._closures      = 0
        # When closures changed — used to compute recent rate / drop detection.
        self._closure_history: list[tuple[float, int]] = []   # (wall_s, tokens at that point)
        self._on_breach_listeners: list[Callable[[CostSnapshot, str], None]] = []
        self._on_progress_listeners: list[Callable[[CostSnapshot], None]] = []

    # --- listeners (Q4 SDK contract) ---

    def on_progress(self, cb: Callable[[CostSnapshot], None]) -> None:
        self._on_progress_listeners.append(cb)

    def on_breach(self, cb: Callable[[CostSnapshot, str], None]) -> None:
        self._on_breach_listeners.append(cb)

    # --- recording ---

    def charge(self, model: str, input_tokens: int, output_tokens: int) -> CostSnapshot:
        with self._lock:
            self._input_tokens  += input_tokens
            self._output_tokens += output_tokens
            self._calls         += 1
            spent = usd_cost(model, input_tokens, output_tokens, self.budget.pricing)
            self._usd += spent
            slot = self._by_model.setdefault(model,
                {"input_tokens": 0, "output_tokens": 0, "usd": 0.0, "calls": 0})
            slot["input_tokens"]  += input_tokens
            slot["output_tokens"] += output_tokens
            slot["usd"]           += spent
            slot["calls"]         += 1
            snap = self._snapshot_unlocked()
        # Listeners outside the lock
        for cb in self._on_progress_listeners:
            try:
                cb(snap)
            except Exception:
                pass    # listener must not crash the meter
        self._check_budget(snap)
        return snap

    def record_closure(self) -> CostSnapshot:
        """Tell the meter that one logical unit of progress (a hyp closure,
        a finding promoted, ...) has happened. Burn rate is computed against
        this counter."""
        with self._lock:
            self._closures += 1
            wall_s = time.monotonic() - self._started
            self._closure_history.append((wall_s, self._input_tokens + self._output_tokens))
            snap = self._snapshot_unlocked()
        for cb in self._on_progress_listeners:
            try:
                cb(snap)
            except Exception:
                pass
        return snap

    # --- read ---

    def snapshot(self) -> CostSnapshot:
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> CostSnapshot:
        total = self._input_tokens + self._output_tokens
        wall = time.monotonic() - self._started
        s = CostSnapshot(
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            total_tokens=total,
            usd=self._usd,
            calls=self._calls,
            wall_seconds=wall,
            by_model={k: dict(v) for k, v in self._by_model.items()},
            closures=self._closures,
        )
        if self._closures > 0:
            s.tokens_per_closure = total / self._closures
            s.usd_per_closure = self._usd / self._closures
        # ETA — only meaningful if caller supplies an estimate of remaining
        # closures; the orchestrator should set this externally when known.
        return s

    # --- budget enforcement ---

    def _check_budget(self, snap: CostSnapshot) -> None:
        b = self.budget
        reason: str | None = None
        axis:   str | None = None
        if b.max_input_tokens  is not None and snap.input_tokens  >= b.max_input_tokens:
            reason = f"max_input_tokens reached ({snap.input_tokens:,})"
            axis = "input_tokens"
        elif b.max_output_tokens is not None and snap.output_tokens >= b.max_output_tokens:
            reason = f"max_output_tokens reached ({snap.output_tokens:,})"
            axis = "output_tokens"
        elif b.max_total_tokens is not None and snap.total_tokens >= b.max_total_tokens:
            reason = f"max_total_tokens reached ({snap.total_tokens:,})"
            axis = "total_tokens"
        elif b.max_usd          is not None and snap.usd >= b.max_usd:
            reason = f"max_usd reached (${snap.usd:.4f})"
            axis = "usd"
        elif b.max_calls        is not None and snap.calls >= b.max_calls:
            reason = f"max_calls reached ({snap.calls})"
            axis = "calls"
        elif b.max_wall_seconds is not None and snap.wall_seconds >= b.max_wall_seconds:
            reason = f"max_wall_seconds reached ({snap.wall_seconds:.1f}s)"
            axis = "wall_seconds"
        if reason is None:
            return
        for cb in self._on_breach_listeners:
            try:
                cb(snap, reason)
            except Exception:
                pass
        raise BudgetExceeded(reason, axis=axis)

    # --- ETA — caller-driven, since "remaining closures" needs system context ---

    def set_eta_from_remaining(self, remaining_closures: int,
                                uncertainty: str = "high") -> CostSnapshot:
        """Compute an ETA assuming current burn rate holds. Honest about
        uncertainty — denominator changes as the run progresses."""
        with self._lock:
            snap = self._snapshot_unlocked()
            if snap.tokens_per_closure is not None and remaining_closures > 0:
                snap.eta_tokens = int(snap.tokens_per_closure * remaining_closures)
                snap.eta_usd    = snap.usd_per_closure * remaining_closures \
                                  if snap.usd_per_closure else None
                snap.eta_uncertainty = uncertainty
        return snap

    # --- rate-drop detection (PLAN §15.2 mid-run pause) ---

    def recent_closure_rate(self, window_seconds: float = 60.0) -> float | None:
        """Closures per second within the last `window_seconds`. None if not
        enough history. Used by orchestrator to trigger a pause when the rate
        approaches zero while spend keeps rising."""
        with self._lock:
            now = time.monotonic() - self._started
            cutoff = now - window_seconds
            recent = [(t, k) for (t, k) in self._closure_history if t >= cutoff]
            if len(recent) < 2:
                return None
            t0, k0 = recent[0]
            t1, k1 = recent[-1]
            dt = t1 - t0
            if dt <= 0:
                return None
            count = sum(1 for t, _ in recent)
            return count / dt
