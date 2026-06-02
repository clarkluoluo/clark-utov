"""Scope-upscale gate probe adapter (v0.4.0 B2 / §19.9 base #1).

Wraps :mod:`engine.scope_upscale_gate` as a mechanism probe so the
conjunctive gate force-includes it.  Composes
:mod:`engine.value_provenance` (observed must cap) and
:mod:`engine.constant_provenance` (dataflow override) into one
declarable check: cross-env claims demand dataflow proof, full stop.
"""

from __future__ import annotations

from engine.scope_upscale_gate import (
    ScopeUpscaleConfig,
    check_scope_upscale,
)
from engine.profile.probe_runtime import (
    Probe,
    ProbeContext,
    Verdict,
    register_builtin_probe,
)


@register_builtin_probe("scope_upscale_gate")
class ScopeUpscaleGateProbe(Probe):
    """Mechanism probe: an observation-pinned value cannot be promoted
    to cross-env scope without dataflow provenance corroboration."""

    name = "scope_upscale_gate"
    mechanism = True
    inputs = ("method", "params")
    outputs = ("scope_upscale",)

    def __init__(self, config: ScopeUpscaleConfig | None = None) -> None:
        self._config = config

    def run(self, ctx: ProbeContext) -> Verdict:
        scope_rank = None
        if ctx.profile is not None:
            scope_rank = getattr(ctx.profile, "scope_rank", None)
        verdict = check_scope_upscale(
            ctx.params,
            scope_rank=scope_rank,
            cfg=self._config,
        )
        if verdict is None:
            return Verdict(probe=self.name, result="undetermined")
        return Verdict(
            probe=self.name,
            result=verdict.result,  # type: ignore[arg-type]
            evidence={"scope_upscale": verdict.to_dict()},
        )
