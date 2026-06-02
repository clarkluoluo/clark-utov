"""Length-chain-check probe adapter (PLAN §19 / IMPL_PLAN §P1.0 step 4).

**Domain probe**, not mechanism. Lives in
``vmp_algorithm_extraction.json`` because ``length_chain`` is a
declared invariant specific to the VMP / algorithm-extraction style
of node-graph analysis. Other domains (key extraction, etc.) are
free not to declare length chains at all, or to declare their own
invariant types via their own probes. The generic "declared
invariants must hold" principle stays in the gate framework (step 5);
this probe is the concrete check for one particular invariant.

Subprofiles MAY override this probe by name, or ``disable:`` it —
that's the "domain layer is open" property. Tests in
``test_vmp_profile_regression.py`` exercise both.

Mapping :class:`LengthChainResult` → :class:`Verdict`:

  * ``no length_chain in params``        → ``undetermined``
  * ``all edges explained``              → ``pass``
  * ``≥ 1 unexplained edge``             → ``fail``
"""

from __future__ import annotations

from typing import Iterable

from engine.length_chain_check import (
    LengthChainConfig,
    LengthChainResult,
    check_chains_in_params,
)
from engine.profile.probe_runtime import (
    Probe,
    ProbeContext,
    Verdict,
    register_builtin_probe,
)


def _aggregate_ok(results: Iterable[LengthChainResult]) -> bool:
    """All chains must be ok for the overall verdict to pass."""
    results = list(results)
    return bool(results) and all(r.ok for r in results)


@register_builtin_probe("length_chain_check")
class LengthChainCheckProbe(Probe):
    """Domain probe: adjacent nodes on a declared length_chain must
    satisfy an explainable length relation (equal / integer-multiple /
    hex 2:1 / base64 4:3 / explicit ratio / explicit delta / caller
    whitelist). Unexplained edges suggest the wrong intermediate
    representation was chosen for the chain.
    """

    name = "length_chain_check"
    inputs = ("params",)
    outputs = ("length_chain",)

    def __init__(self, config: LengthChainConfig | None = None) -> None:
        self._config = config

    def run(self, ctx: ProbeContext) -> Verdict:
        results = check_chains_in_params(ctx.params, cfg=self._config)
        if not results:
            return Verdict(probe=self.name, result="undetermined")

        ok = _aggregate_ok(results)
        unexplained_count = sum(len(r.unexplained_edges) for r in results)
        evidence = {
            "chains": [r.to_dict() for r in results],
            "chain_count": len(results),
            "unexplained_edge_count": unexplained_count,
        }

        return Verdict(
            probe=self.name,
            result="pass" if ok else "fail",
            evidence=evidence,
        )
