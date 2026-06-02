"""Scope-upscale gate (v0.4.0 base / §19.9 base #1).

The strongest tc3 finding the v0.4.0 batch crystallises:

  *pinned 锁运行时量 → 观测必然无法区分.*

A value that is pinned at observation time (the runner happens to
return the same bytes every read) tells you *nothing* about what the
value would be in a different runtime context.  Observation alone
cannot distinguish a true cross-environment constant from a
runtime-locked artefact; promoting an observed-pinned value to the
widest scope (``cross_env``) requires evidence the producer doesn't
read run-time inputs at all — i.e. producer-dataflow proof.

The gate composes two probes that already exist in v0.3.0 base
mechanism:

  * :mod:`engine.value_provenance` — caps hook/dump *observed* values
    at evidence class B (or whatever the domain calls "observed
    ceiling"). Without an explicit invariance proof, observed = capped.
  * :mod:`engine.constant_provenance` — dataflow probe; classification
    ``HARDCODED_FIXED`` requires the producer reads only static
    sources.

This module wires the composition: when the caller advertises a
``cross_env`` (or otherwise *widest-rank*) scope claim, the gate
demands either a closed-form-recompute attestation OR a
``constant_provenance`` verdict whose source category is
``HARDCODED_FIXED`` / ``APPKEY_FIXED_FUNCTION`` (the two categories
with cross-env-safe ceilings).  Otherwise it fails.

Input shape (params, all optional):

  * ``scope_claim`` / ``claim_scope`` / ``scope`` — caller's claim.
  * ``value_class`` / ``value_provenance_class`` — ``"observed"`` /
    ``"closed_form"`` / etc.  Treated as the value-provenance verdict.
  * ``constant_provenance_category`` — explicit category fed by an
    upstream CP run, when present.
  * ``producer_dataflow.proof_of_invariance: bool`` — explicit
    attestation that the producer's reads do not include any runtime
    dimension (caller-supplied; the gate treats it as definitive).

Independent toggle: ``UTOV_SCOPE_UPSCALE_GATE=off|0|false|no``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional


# Scope-claim ranks the gate considers "wide enough to need dataflow
# proof".  The active profile's scope_order is the authority; the
# fallback for profile-less contexts is the canonical narrowest →
# widest tuple shipped by v0.4.0 vmp.
_DEFAULT_WIDE_THRESHOLD_RANK = 2  # = single_identity_bound under canonical order


# Constant-provenance categories that DO support cross_env without
# further evidence (producer reads only static or appkey sources).
_CROSS_ENV_SAFE_CATEGORIES: frozenset[str] = frozenset({
    "hardcoded_fixed",
    "appkey_fixed_function",
})


# Value-class labels that constitute closed-form attestation (the
# caller has already proved the value).  Match shapes commonly used
# across runner reports.
_CLOSED_FORM_LABELS: frozenset[str] = frozenset({
    "closed_form",
    "closed-form",
    "closed_form_attested",
})


@dataclass(frozen=True, slots=True)
class ScopeUpscaleVerdict:
    result: str                              # 'pass' | 'fail' | 'undetermined'
    claim: str
    value_class: str
    cp_category: str
    proof_of_invariance: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "result":              self.result,
            "claim":               self.claim,
            "value_class":         self.value_class,
            "cp_category":         self.cp_category,
            "proof_of_invariance": self.proof_of_invariance,
            "reason":              self.reason,
        }


@dataclass(slots=True)
class ScopeUpscaleConfig:
    enabled: bool = True
    # Minimum rank (in the active profile's scope_order) at which the
    # gate starts requiring dataflow proof.  Profile-less callers may
    # pass an explicit threshold; the in-profile default is "any rank
    # at or above _DEFAULT_WIDE_THRESHOLD_RANK".
    wide_threshold_rank: int = _DEFAULT_WIDE_THRESHOLD_RANK

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "ScopeUpscaleConfig":
        src = env if env is not None else os.environ
        cfg = cls()
        flag = (src.get("UTOV_SCOPE_UPSCALE_GATE") or "").strip().lower()
        if flag in ("off", "0", "false", "no"):
            cfg.enabled = False
        rank = src.get("UTOV_SCOPE_UPSCALE_WIDE_RANK")
        if rank is not None:
            try:
                cfg.wide_threshold_rank = max(0, int(rank))
            except ValueError:
                pass
        return cfg


def check_scope_upscale(
    params: dict[str, Any] | None,
    *,
    scope_rank: Optional[callable] = None,
    cfg: ScopeUpscaleConfig | None = None,
) -> ScopeUpscaleVerdict | None:
    """Run the gate.

    Returns ``None`` when the gate is disabled. Otherwise returns a
    verdict — possibly ``undetermined`` when the caller has supplied
    a claim but no provenance information.
    """
    cfg = cfg or ScopeUpscaleConfig.from_env()
    if not cfg.enabled:
        return None
    params = params or {}

    claim = _extract_first_str(
        params, ("scope_claim", "claim_scope", "scope")
    )
    if not claim:
        return ScopeUpscaleVerdict(
            result="undetermined",
            claim="",
            value_class="",
            cp_category="",
            proof_of_invariance=False,
            reason="no scope_claim supplied",
        )

    # Only fire when the claim is at the "wide" end of the ordering.
    rank = scope_rank(claim) if scope_rank else None
    if rank is None:
        # No ordering info — apply the gate only when the claim is the
        # well-known top label.
        if claim.lower() != "cross_env":
            return ScopeUpscaleVerdict(
                result="pass",
                claim=claim,
                value_class="",
                cp_category="",
                proof_of_invariance=False,
                reason="claim below the cross-env threshold; gate not engaged",
            )
    else:
        if rank < cfg.wide_threshold_rank:
            return ScopeUpscaleVerdict(
                result="pass",
                claim=claim,
                value_class="",
                cp_category="",
                proof_of_invariance=False,
                reason=(
                    f"claim rank {rank} below wide threshold "
                    f"{cfg.wide_threshold_rank}; gate not engaged"
                ),
            )

    value_class = _extract_first_str(
        params, ("value_class", "value_provenance_class", "vp_class")
    )
    cp_category = _extract_first_str(
        params,
        (
            "constant_provenance_category",
            "cp_category",
            "source_category",
        ),
    )
    proof = _extract_proof_of_invariance(params)

    # Pass conditions — any one is sufficient.
    if value_class.lower() in _CLOSED_FORM_LABELS:
        return ScopeUpscaleVerdict(
            result="pass",
            claim=claim,
            value_class=value_class,
            cp_category=cp_category,
            proof_of_invariance=proof,
            reason=(
                "value carries closed-form attestation — cross-env claim "
                "supported"
            ),
        )
    if proof:
        return ScopeUpscaleVerdict(
            result="pass",
            claim=claim,
            value_class=value_class,
            cp_category=cp_category,
            proof_of_invariance=True,
            reason=(
                "caller supplied explicit producer-dataflow proof of "
                "invariance"
            ),
        )
    if cp_category.lower() in _CROSS_ENV_SAFE_CATEGORIES:
        return ScopeUpscaleVerdict(
            result="pass",
            claim=claim,
            value_class=value_class,
            cp_category=cp_category,
            proof_of_invariance=False,
            reason=(
                f"constant_provenance category '{cp_category}' is "
                f"cross-env-safe; producer reads only static/appkey"
            ),
        )
    return ScopeUpscaleVerdict(
        result="fail",
        claim=claim,
        value_class=value_class,
        cp_category=cp_category,
        proof_of_invariance=False,
        reason=(
            f"claim '{claim}' is at the cross-env scale, but no closed-form "
            f"attestation, no cross-env-safe constant_provenance category, "
            f"and no explicit dataflow proof — observation alone cannot "
            f"distinguish a true constant from a runtime-locked artefact "
            f"(§19.9 base #1)"
        ),
    )


def _extract_first_str(node: Any, candidates: Iterable[str], *, depth: int = 4) -> str:
    if depth <= 0 or node is None:
        return ""
    if isinstance(node, dict):
        for key in candidates:
            v = node.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        for v in node.values():
            r = _extract_first_str(v, candidates, depth=depth - 1)
            if r:
                return r
    elif isinstance(node, list):
        for v in node:
            r = _extract_first_str(v, candidates, depth=depth - 1)
            if r:
                return r
    return ""


def _extract_proof_of_invariance(node: Any, *, depth: int = 4) -> bool:
    if depth <= 0 or node is None:
        return False
    if isinstance(node, dict):
        for key in ("proof_of_invariance", "dataflow_proof_of_invariance"):
            v = node.get(key)
            if isinstance(v, bool) and v:
                return True
        for v in node.values():
            if _extract_proof_of_invariance(v, depth=depth - 1):
                return True
    elif isinstance(node, list):
        for v in node:
            if _extract_proof_of_invariance(v, depth=depth - 1):
                return True
    return False


__all__ = [
    "ScopeUpscaleConfig",
    "ScopeUpscaleVerdict",
    "check_scope_upscale",
]
