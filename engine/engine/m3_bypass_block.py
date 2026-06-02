"""M3 bypass-block auto-detector.

Background — the reference target's retro: a candidate "SM3 compress" block was
probed under three independent observation methods (pre-compress
hook, post-compress hook, dump-then-rehash). All three rounds showed
variability=0 (the bytes never changed across distinct inputs). The
real data flow ran on a parallel path; the agent kept "changing
posture" (swapping observation methods) on the same dead block
instead of stepping out of it. The block was eventually retired by
hand.

This module turns that pattern into a framework-layer gate. Rule:

  Same candidate block, ≥ N distinct observation methods (N
  configurable, default 2), all M3 variability checks fail
  (``failed=True`` / ``unique_count`` below threshold / pass_rate=0)
  ⇒ auto-mark the block ``suspected_bypass_block`` and refuse any
  further observation attempts on it.

The criterion is **N distinct observation methods all failing** —
not a count of repeated failures with the same method. Single-method
failure is treated as a possible observation bug; only the *cross-
method* invariance flips the block to bypass.

Cross-references:

  - M-R2 (no multi-candidate stacking) — sibling rule on the OTHER
    direction: M-R2 stops "swap candidate to make it pass"; this
    rule stops "swap observation method on the same block to make
    it pass".
  - M1 dimension-coverage check (``m1_success_audit.py``) — that
    rule is the per-call counterpart: if a dimension's variance=0
    within ONE call, that's an M1 untested dimension. This module
    aggregates the same evidence ACROSS calls: when the same block
    is re-probed under different methods and *all* show 0 variance,
    the conclusion is structural — the block is bypassed.

Independent toggle: ``UTOV_M3_BYPASS=off|0|false|no`` disables the
module entirely without touching the rest of the wrapper.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


# Verification methods whose params/result carry an M3 variability
# outcome on a named block. The detector inspects these post-dispatch.
DEFAULT_M3_METHODS: tuple[str, ...] = (
    "verify_block_variability",
    "verify_hook_sanity",
    "hook_sanity",
    "verify_observation",
    "m3_variability_check",
    "verify_handler_binops",
    "verify_handler_unaries",
)


# Method whitelist that the detector pre-intercepts when the target
# block has already been marked suspected_bypass. Bigger than the
# detection set: once a block is dead we refuse any *observation*
# call on it, not only M3 verifications.
DEFAULT_OBSERVATION_METHODS: tuple[str, ...] = (
    *DEFAULT_M3_METHODS,
    "install_hook",
    "trace_block",
    "dump_block",
    "observe_block",
    "verify_block_variability",
)


@dataclass(slots=True)
class M3BypassConfig:
    """Tunable knobs for the M3 bypass-block gate. Independent of
    :class:`MethodologyConfig` and :class:`M1AuditConfig` so each gate
    can be toggled in isolation during debugging."""

    enabled: bool = True
    # Minimum number of DISTINCT observation methods that must all
    # have failed on the same block to flip it to suspected_bypass.
    # Default 2 — matches the user spec; raise to 3 for stricter
    # confidence (the reference target actually had 3).
    min_failed_observations: int = 2
    # M3 verification methods (results read post-dispatch).
    m3_methods: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_M3_METHODS,
    )
    # Method whitelist that the detector pre-intercepts once a block
    # is confirmed bypass.
    intercept_methods: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_OBSERVATION_METHODS,
    )
    # When inferring "failed" from the result, treat a unique count
    # strictly below this as a variability failure.
    min_unique_for_pass: int = 2

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "M3BypassConfig":
        src = env if env is not None else os.environ
        cfg = cls()
        flag = (src.get("UTOV_M3_BYPASS") or "").strip().lower()
        if flag in ("off", "0", "false", "no"):
            cfg.enabled = False
        for env_key, attr, cast in (
            ("UTOV_M3_BYPASS_N",                  "min_failed_observations", int),
            ("UTOV_M3_BYPASS_MIN_UNIQUE_FOR_PASS", "min_unique_for_pass",    int),
        ):
            v = src.get(env_key)
            if v is None:
                continue
            try:
                setattr(cfg, attr, cast(v))
            except ValueError:
                continue
        return cfg


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ObservationAttempt:
    """One attempt on one block under one observation method."""
    block_id: str
    observation_method: str
    failed: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id":           self.block_id,
            "observation_method": self.observation_method,
            "failed":             self.failed,
            "note":               self.note,
        }


@dataclass(frozen=True, slots=True)
class BypassDetection:
    """Structured result of a bypass detection. ``triggered=True``
    means the threshold was crossed on this attempt.

    ``recommendation`` carries the agent-facing advice (look upstream
    / parallel path).  ``intercepted_reason`` is non-None when the
    *current* attempt is being refused (either as the triggering
    attempt's downgrade, or as a follow-up attempt on an already-
    confirmed bypass block)."""
    block_id:           str
    failed_methods:     tuple[str, ...]
    suspected_bypass:   bool          # True once flipped (sticky)
    triggered:          bool          # True only on the call that flipped it
    intercepted_reason: str | None
    recommendation:     str

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id":           self.block_id,
            "failed_methods":     list(self.failed_methods),
            "suspected_bypass":   self.suspected_bypass,
            "triggered":          self.triggered,
            "intercepted_reason": self.intercepted_reason,
            "recommendation":     self.recommendation,
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def _recommendation_for(block_id: str, methods: tuple[str, ...]) -> str:
    method_list = ", ".join(methods)
    return (
        f"block {block_id} failed M3 variability under {len(methods)} "
        f"independent observation methods ({method_list}) — the data "
        f"likely does not flow through this block. STOP swapping observation "
        f"methods on this block; instead prune the 'real data flows through "
        f"{block_id}' hypothesis and re-search upstream or on a parallel path."
    )


class BypassBlockDetector:
    """Per-session state machine.

    The detector keeps:

      - ``history``: ``block_id → {observation_method → ObservationAttempt}``.
        Only the latest attempt per (block, method) is kept; repeated
        same-method retries do not inflate the failure count.
      - ``confirmed_bypass``: blocks that have already crossed the
        threshold. Once a block is in this set, ``is_known_bypass``
        returns True for it and the wrapper refuses follow-up
        observation attempts.
      - ``triggers``: the original :class:`BypassDetection` that
        flipped each block to suspected — used for intercept message
        + ledger annotation.
    """

    def __init__(self, cfg: M3BypassConfig | None = None):
        self.cfg = cfg or M3BypassConfig.from_env()
        self.history: dict[str, dict[str, ObservationAttempt]] = {}
        self.confirmed_bypass: set[str] = set()
        self.triggers: dict[str, BypassDetection] = {}

    # -- queries -----------------------------------------------------

    def is_known_bypass(self, block_id: str) -> bool:
        return self.cfg.enabled and block_id in self.confirmed_bypass

    def trigger_for(self, block_id: str) -> BypassDetection | None:
        return self.triggers.get(block_id)

    # -- writes ------------------------------------------------------

    def record_attempt(
        self,
        block_id: str,
        observation_method: str,
        *,
        failed: bool,
        note: str = "",
    ) -> BypassDetection | None:
        """Record one observation attempt and return a detection iff
        the threshold was crossed *on this call*. Returns ``None`` if
        below threshold OR the block was already confirmed (in that
        case the caller should use :meth:`intercept_followup`)."""
        if not self.cfg.enabled or not block_id or not observation_method:
            return None
        slot = self.history.setdefault(block_id, {})
        slot[observation_method] = ObservationAttempt(
            block_id=block_id,
            observation_method=observation_method,
            failed=failed,
            note=note,
        )
        if block_id in self.confirmed_bypass:
            # Sticky — already triggered. Caller should already be
            # intercepting before dispatch; record stays for audit
            # but we don't re-fire.
            return None
        failed_methods = tuple(
            sorted(m for m, a in slot.items() if a.failed)
        )
        if len(failed_methods) >= self.cfg.min_failed_observations:
            self.confirmed_bypass.add(block_id)
            detection = BypassDetection(
                block_id=block_id,
                failed_methods=failed_methods,
                suspected_bypass=True,
                triggered=True,
                intercepted_reason=(
                    f"M3 bypass-block gate flipped {block_id} to "
                    f"suspected_bypass after {len(failed_methods)} distinct "
                    f"observation methods ({', '.join(failed_methods)}) all "
                    f"reported variability=0. Subsequent observation attempts "
                    f"on this block will be refused — change the hypothesis, "
                    f"not the posture."
                ),
                recommendation=_recommendation_for(block_id, failed_methods),
            )
            self.triggers[block_id] = detection
            return detection
        return None

    def intercept_followup(
        self,
        block_id: str,
        observation_method: str,
    ) -> BypassDetection | None:
        """Return the original detection when ``block_id`` is already
        marked bypass — i.e. this attempt should be refused before
        dispatch. Returns ``None`` otherwise."""
        if not self.is_known_bypass(block_id):
            return None
        original = self.triggers.get(block_id)
        if original is None:
            return None
        # Surface a follow-up record so the caller can see what was
        # blocked, but the original detection is what carries the
        # "why" + recommendation.
        return BypassDetection(
            block_id=block_id,
            failed_methods=original.failed_methods,
            suspected_bypass=True,
            triggered=False,
            intercepted_reason=(
                f"refused {observation_method} on {block_id}: block already "
                f"marked suspected_bypass after {', '.join(original.failed_methods)} "
                f"all failed M3 variability. Switching observation method on "
                f"a dead block is the anti-pattern this gate exists for — go "
                f"upstream/parallel and re-anchor the hypothesis."
            ),
            recommendation=original.recommendation,
        )


# ---------------------------------------------------------------------------
# Input extraction
# ---------------------------------------------------------------------------


def extract_attempt(
    method: str,
    params: dict[str, Any] | None,
    result: Any,
    *,
    cfg: M3BypassConfig,
) -> tuple[str, str, bool] | None:
    """Pull ``(block_id, observation_method, failed)`` from a tool call,
    returning ``None`` if the call doesn't qualify as an M3 attempt.

    Looks in params first, then result. Variability failure is
    derived in this priority order:
      1. explicit ``failed`` boolean
      2. explicit ``variability_failed`` boolean
      3. unique_count < ``min_unique_for_pass``
      4. ``hook_digest_eq_sign`` == 0 (a hook-equality rate of zero)
      5. ``passed == False`` together with ``checked >= 1``
    """
    if not cfg.enabled:
        return None
    if method not in cfg.m3_methods:
        return None

    block_id = _first(params, "block_id") or _first(result, "block_id")
    if not isinstance(block_id, str) or not block_id:
        return None
    obs = _first(params, "observation_method") or _first(result, "observation_method")
    if not isinstance(obs, str) or not obs:
        return None

    failed = _infer_failed(params, result, cfg=cfg)
    if failed is None:
        return None
    return (block_id, obs, failed)


def _infer_failed(
    params: dict[str, Any] | None,
    result: Any,
    *,
    cfg: M3BypassConfig,
) -> bool | None:
    # 1 / 2 — explicit booleans
    for key in ("failed", "variability_failed"):
        v = _first(params, key)
        if isinstance(v, bool):
            return v
        v = _first(result, key)
        if isinstance(v, bool):
            return v

    # 3 — unique_count
    uniq = _first(params, "unique_count")
    if uniq is None:
        uniq = _first(result, "unique_count")
    if uniq is None:
        uniq = _first(params, "hook_digest_unique_count") or _first(
            result, "hook_digest_unique_count",
        )
    if isinstance(uniq, int):
        return uniq < cfg.min_unique_for_pass

    # 4 — hook_digest_eq_sign
    eq = _first(params, "hook_digest_eq_sign")
    if eq is None:
        eq = _first(result, "hook_digest_eq_sign")
    if isinstance(eq, (int, float)):
        return float(eq) == 0.0

    # 5 — passed/checked pair
    passed  = _first(params, "passed")
    if passed is None:
        passed = _first(result, "passed")
    checked = _first(params, "checked")
    if checked is None:
        checked = _first(result, "checked")
    if isinstance(passed, int) and isinstance(checked, int) and checked >= 1:
        return passed == 0

    return None


def _first(node: Any, key: str, *, depth: int = 4) -> Any:
    """Walk ``node`` and return the first value found under ``key``."""
    if depth <= 0 or node is None:
        return None
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for v in node.values():
            r = _first(v, key, depth=depth - 1)
            if r is not None:
                return r
    elif isinstance(node, list):
        for v in node:
            r = _first(v, key, depth=depth - 1)
            if r is not None:
                return r
    return None


# ---------------------------------------------------------------------------
# Envelope rendering
# ---------------------------------------------------------------------------


def render_bypass_alert(detection: BypassDetection) -> str:
    if detection.triggered:
        return (
            f"[M3-BYPASS/TRIGGERED] {detection.block_id}: "
            f"variability=0 across {len(detection.failed_methods)} methods "
            f"({', '.join(detection.failed_methods)}) — flipped to "
            f"suspected_bypass_block. {detection.recommendation}"
        )
    return (
        f"[M3-BYPASS/REFUSED] {detection.block_id}: follow-up observation "
        f"refused on confirmed bypass block. "
        f"{detection.intercepted_reason or ''}"
    )
