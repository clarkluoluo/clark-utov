"""Scope-boundary gate (v0.4.0 base / §19.9 base #3).

Statement of the principle, from the field audit:

  *scope declaration ≤ actually observed boundary; no extrapolation.*

A node may not advertise a scope wider than the evidence chain has
material for.  Specifically — a value whose observation domain only
covered (runner X, task T) cannot, without further work, support a
cross-env claim about every (runner, task) pair.  Promoting the claim
past the observed boundary is the family of bugs the field audit
crystallised under "本可下放却上抛".

The gate is mechanism: every domain needs it, and the principle is
the same regardless of what concrete scope vocabulary the domain
uses.  The scope vocabulary itself is domain — the domain profile
declares :attr:`MergedProfile.scope_order` (narrowest first) and the
gate consults the active profile to compare ranks.  Without a profile
the gate degrades to ``undetermined`` rather than guessing the
ordering.

Input shape (params, all optional):

  * ``scope_claim``     — the scope tag the caller wants to advertise
                          (e.g. ``"cross_env"``).
  * ``scope_observed``  — the widest scope the evidence chain
                          actually covered (e.g. ``"env_bound"``).
  * ``observed_boundary`` — alias for ``scope_observed``.

When either is missing the gate returns ``undetermined``; missing
evidence is the caller's job to fix, not the gate's to assume.

Independent toggle: ``UTOV_SCOPE_BOUNDARY_GATE=off|0|false|no``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True, slots=True)
class ScopeBoundaryVerdict:
    """Outcome of one scope-boundary check."""

    result: str                       # 'pass' | 'fail' | 'undetermined'
    claim: str
    observed: str
    claim_rank: Optional[int]
    observed_rank: Optional[int]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "result":        self.result,
            "claim":         self.claim,
            "observed":      self.observed,
            "claim_rank":    self.claim_rank,
            "observed_rank": self.observed_rank,
            "reason":        self.reason,
        }


@dataclass(slots=True)
class ScopeBoundaryConfig:
    enabled: bool = True

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "ScopeBoundaryConfig":
        src = env if env is not None else os.environ
        cfg = cls()
        flag = (src.get("UTOV_SCOPE_BOUNDARY_GATE") or "").strip().lower()
        if flag in ("off", "0", "false", "no"):
            cfg.enabled = False
        return cfg


def check_scope_boundary(
    params: dict[str, Any] | None,
    *,
    scope_rank: callable | None = None,
    cfg: ScopeBoundaryConfig | None = None,
) -> ScopeBoundaryVerdict | None:
    """Run the gate.

    ``scope_rank`` is a callable ``(scope_name) -> rank | None`` —
    typically :meth:`MergedProfile.scope_rank`. When ``None`` is
    supplied or returns ``None`` for either input, the gate returns
    ``undetermined`` (the profile didn't declare ordering for the
    relevant scope value, and the gate refuses to invent one).
    """
    cfg = cfg or ScopeBoundaryConfig.from_env()
    if not cfg.enabled:
        return None
    params = params or {}
    claim = _extract_scope(params, ("scope_claim", "claim_scope", "scope"))
    observed = _extract_scope(
        params, ("scope_observed", "observed_boundary", "observed_scope")
    )
    if not claim or not observed:
        return ScopeBoundaryVerdict(
            result="undetermined",
            claim=claim or "",
            observed=observed or "",
            claim_rank=None,
            observed_rank=None,
            reason="scope_claim and/or observed_boundary missing",
        )

    claim_rank = scope_rank(claim) if scope_rank else None
    observed_rank = scope_rank(observed) if scope_rank else None
    if claim_rank is None or observed_rank is None:
        return ScopeBoundaryVerdict(
            result="undetermined",
            claim=claim,
            observed=observed,
            claim_rank=claim_rank,
            observed_rank=observed_rank,
            reason=(
                "profile did not declare ordering for one or both scope "
                "values — ScopeBoundaryGate cannot compare"
            ),
        )
    if claim_rank > observed_rank:
        return ScopeBoundaryVerdict(
            result="fail",
            claim=claim,
            observed=observed,
            claim_rank=claim_rank,
            observed_rank=observed_rank,
            reason=(
                f"claim '{claim}' (rank {claim_rank}) wider than observed "
                f"boundary '{observed}' (rank {observed_rank}) — refuse to "
                f"extrapolate past the actually-observed boundary "
                f"(§19.9 base #3)"
            ),
        )
    return ScopeBoundaryVerdict(
        result="pass",
        claim=claim,
        observed=observed,
        claim_rank=claim_rank,
        observed_rank=observed_rank,
        reason=(
            f"claim '{claim}' (rank {claim_rank}) within observed boundary "
            f"'{observed}' (rank {observed_rank})"
        ),
    )


def _extract_scope(node: Any, candidates: tuple[str, ...], *, depth: int = 4) -> str:
    """Walk ``node`` for the first string value under any of
    ``candidates``."""
    if depth <= 0 or node is None:
        return ""
    if isinstance(node, dict):
        for key in candidates:
            v = node.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        for v in node.values():
            r = _extract_scope(v, candidates, depth=depth - 1)
            if r:
                return r
    elif isinstance(node, list):
        for v in node:
            r = _extract_scope(v, candidates, depth=depth - 1)
            if r:
                return r
    return ""


__all__ = [
    "ScopeBoundaryConfig",
    "ScopeBoundaryVerdict",
    "check_scope_boundary",
]
