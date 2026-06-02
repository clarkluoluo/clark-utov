"""Use-case fork rule (v0.4.0 V4 / §19.9 vmp #5).

Domain probe (not mechanism) for ``vmp_algorithm_extraction``.

Encodes a decision point that has been resolved by hand for the entire
v0.3.0 cycle: "do we need to handle this for the current target /
runner, or do we need to defend it across all targets?"  The fork
sets very different scope requirements:

  * **current_context_reproduction** — the caller wants the value to
    reproduce inside one (runner × task) context.  ``task_bound`` or
    ``env_bound`` is enough; the claim has no business reaching
    ``cross_env``.  Asking for a wider claim under this use case is
    over-extension.

  * **cross_context_claim** — the caller wants to defend the value
    across (runners × tasks).  ``cross_env`` is appropriate, but the
    value-provenance + dataflow attestation must back it (the same
    bar :class:`ScopeUpscaleGateProbe` enforces at the mechanism
    layer).  Reproducing only inside the current context isn't enough.

This probe is **domain** because the vocabulary
(``current_context_reproduction``, ``cross_context_claim``) lives in
VMP-algorithm-extraction; another domain may declare its own fork.
The probe itself is a stub when the caller has no ``use_case`` field
— other gates handle the bare-claim cases.

Input shape (params):

  * ``use_case`` — ``"current_context_reproduction"`` or
    ``"cross_context_claim"``.
  * ``scope_claim`` — the claim the caller wants to advertise.
  * ``producer_dataflow.proof_of_invariance: bool`` — explicit
    attestation that the producer reads no run-time dimensions.
"""

from __future__ import annotations

from typing import Any

from engine.profile.probe_runtime import (
    Probe,
    ProbeContext,
    Verdict,
    register_builtin_probe,
)


_USE_CASE_CURRENT = "current_context_reproduction"
_USE_CASE_CROSS = "cross_context_claim"


# Scope vocabulary the rule recognises (matches vmp_algorithm_extraction's
# scope_order).  Position in the tuple = rank (narrowest first).
_FALLBACK_SCOPE_ORDER = (
    "task_bound", "env_bound", "single_identity_bound", "cross_env",
)


@register_builtin_probe("use_case_fork")
class UseCaseForkProbe(Probe):
    """Domain probe: enforce the current-vs-cross-context fork.

    The probe consults :attr:`MergedProfile.scope_order` when available
    so a domain that reorders its scope vocabulary stays consistent
    with the rule.  Without a profile, the canonical v0.4.0 vmp
    ordering is used as a fallback.
    """

    name = "use_case_fork"
    mechanism = False
    inputs = ("method", "params")
    outputs = ("use_case_fork",)

    def run(self, ctx: ProbeContext) -> Verdict:
        params = ctx.params or {}
        use_case = _first_str(params, ("use_case",))
        if not use_case:
            return Verdict(probe=self.name, result="undetermined")

        scope_claim = _first_str(params, ("scope_claim", "claim_scope", "scope"))
        scope_rank = (
            getattr(ctx.profile, "scope_rank", None)
            if ctx.profile is not None else None
        )

        if use_case == _USE_CASE_CURRENT:
            verdict = self._current_context(scope_claim, scope_rank)
        elif use_case == _USE_CASE_CROSS:
            verdict = self._cross_context(scope_claim, scope_rank, params)
        else:
            verdict = (
                "undetermined",
                f"unknown use_case '{use_case}' — domain probe declined to "
                f"adjudicate; declare {_USE_CASE_CURRENT} or {_USE_CASE_CROSS}",
            )

        result, reason = verdict
        return Verdict(
            probe=self.name,
            result=result,  # type: ignore[arg-type]
            evidence={
                "use_case":    use_case,
                "scope_claim": scope_claim,
                "reason":      reason,
            },
        )

    @staticmethod
    def _current_context(
        scope_claim: str,
        scope_rank: callable | None,
    ) -> tuple[str, str]:
        """``current_context_reproduction`` — claim must stay ≤ env_bound."""
        if not scope_claim:
            return "undetermined", "use_case=current_context but no scope_claim"
        rank = _rank(scope_claim, scope_rank)
        env_bound_rank = _rank("env_bound", scope_rank)
        if rank is None or env_bound_rank is None:
            return (
                "undetermined",
                "profile did not declare ordering for the relevant scopes",
            )
        if rank > env_bound_rank:
            return (
                "fail",
                (
                    f"use_case=current_context_reproduction but scope_claim "
                    f"'{scope_claim}' (rank {rank}) exceeds env_bound "
                    f"(rank {env_bound_rank}); narrow the claim or switch to "
                    f"cross_context_claim and supply dataflow proof"
                ),
            )
        return (
            "pass",
            (
                f"use_case=current_context_reproduction; claim '{scope_claim}' "
                f"within env_bound — OK"
            ),
        )

    @staticmethod
    def _cross_context(
        scope_claim: str,
        scope_rank: callable | None,
        params: dict[str, Any],
    ) -> tuple[str, str]:
        """``cross_context_claim`` — claim must be cross_env AND backed
        by dataflow proof (or a closed-form attestation)."""
        if scope_claim and scope_claim != "cross_env":
            rank = _rank(scope_claim, scope_rank)
            cross_rank = _rank("cross_env", scope_rank)
            if rank is not None and cross_rank is not None and rank < cross_rank:
                return (
                    "fail",
                    (
                        f"use_case=cross_context_claim but scope_claim "
                        f"'{scope_claim}' is narrower than cross_env — a "
                        f"cross-context claim must advertise the widest scope "
                        f"and back it with producer-dataflow proof"
                    ),
                )

        proof = _proof_of_invariance(params)
        vp_class = _first_str(params, ("value_class", "value_provenance_class"))
        if vp_class.lower() in ("closed_form", "closed-form", "closed_form_attested"):
            return (
                "pass",
                "cross_context_claim backed by closed-form attestation",
            )
        if proof:
            return (
                "pass",
                "cross_context_claim backed by explicit dataflow proof",
            )
        cp_cat = _first_str(
            params, ("constant_provenance_category", "cp_category")
        )
        if cp_cat.lower() in ("hardcoded_fixed", "appkey_fixed_function"):
            return (
                "pass",
                (
                    f"cross_context_claim backed by constant_provenance "
                    f"category '{cp_cat}' (producer reads only static / appkey)"
                ),
            )
        return (
            "fail",
            (
                "use_case=cross_context_claim but no closed-form attestation, "
                "no cross-env-safe constant_provenance category, and no "
                "explicit producer-dataflow proof — observation alone cannot "
                "carry a cross-context claim"
            ),
        )


def _rank(scope: str, scope_rank: callable | None) -> int | None:
    if scope_rank is not None:
        v = scope_rank(scope)
        if v is not None:
            return v
    if scope in _FALLBACK_SCOPE_ORDER:
        return _FALLBACK_SCOPE_ORDER.index(scope)
    return None


def _first_str(node: Any, keys, *, depth: int = 4) -> str:
    if depth <= 0 or node is None:
        return ""
    if isinstance(node, dict):
        for k in keys:
            v = node.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        for v in node.values():
            r = _first_str(v, keys, depth=depth - 1)
            if r:
                return r
    elif isinstance(node, list):
        for v in node:
            r = _first_str(v, keys, depth=depth - 1)
            if r:
                return r
    return ""


def _proof_of_invariance(node: Any, *, depth: int = 4) -> bool:
    if depth <= 0 or node is None:
        return False
    if isinstance(node, dict):
        for k in ("proof_of_invariance", "dataflow_proof_of_invariance"):
            v = node.get(k)
            if isinstance(v, bool) and v:
                return True
        for v in node.values():
            if _proof_of_invariance(v, depth=depth - 1):
                return True
    elif isinstance(node, list):
        for v in node:
            if _proof_of_invariance(v, depth=depth - 1):
                return True
    return False
