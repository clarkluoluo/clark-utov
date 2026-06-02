"""Scope-boundary gate probe adapter (v0.4.0 B1 / §19.9 base #3).

Wraps :mod:`engine.scope_boundary_gate` as a mechanism probe so the
conjunctive gate force-includes it at every archival surface, just
like M1 / M3 / constant_provenance.  The probe reads the active
profile's ``scope_order`` to compare claim vs observed boundary —
domain plugs the vocabulary, base enforces "no extrapolation".
"""

from __future__ import annotations

from engine.scope_boundary_gate import (
    ScopeBoundaryConfig,
    check_scope_boundary,
)
from engine.profile.probe_runtime import (
    Probe,
    ProbeContext,
    Verdict,
    register_builtin_probe,
)


@register_builtin_probe("scope_boundary_gate")
class ScopeBoundaryGateProbe(Probe):
    """Mechanism probe: a node's claimed scope must not exceed the
    actually-observed boundary.  Every domain needs the rule; the
    scope vocabulary itself is domain semantics.
    """

    name = "scope_boundary_gate"
    mechanism = True
    inputs = ("method", "params")
    outputs = ("scope_boundary",)

    def __init__(self, config: ScopeBoundaryConfig | None = None) -> None:
        self._config = config

    def run(self, ctx: ProbeContext) -> Verdict:
        scope_rank = None
        if ctx.profile is not None:
            scope_rank = getattr(ctx.profile, "scope_rank", None)
        verdict = check_scope_boundary(
            ctx.params,
            scope_rank=scope_rank,
            cfg=self._config,
        )
        if verdict is None:
            return Verdict(probe=self.name, result="undetermined")
        return Verdict(
            probe=self.name,
            result=verdict.result,  # type: ignore[arg-type]
            evidence={"scope_boundary": verdict.to_dict()},
        )
