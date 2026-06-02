"""Constant-provenance probe adapter (PLAN §19 / IMPL_PLAN §P1.0 step 3).

Wraps :mod:`engine.constant_provenance` — the dataflow + rerun two-axis
classification framework is the base mechanism. The mapping from the
5-way category to a concrete evidence-class cap stays domain-driven:
the underlying module already emits an ``evidence_class_ceiling``
string per result for the VMP-algorithm-extraction domain, and the
probe propagates that as-is. Step 5's evidence-class synth will
recompute the cap through the active profile's ordering.

Mapping :class:`ConstantProvenanceResult` → :class:`Verdict`:

  * ``no value records in params``     → ``undetermined``
  * ``≥ 1 classifications produced``   → ``pass`` (the probe never
    "fails" a call — its job is classification + ceiling
    propagation, not gating)
  * ``affects_evidence_class``         → most-restrictive ceiling
    across all classifications, computed via
    :func:`engine.profile.evidence_class_synth.most_restrictive_class_id`.
    When ``ctx.profile`` is set, the helper walks the profile's
    ``evidence_classes`` ordering (step-5+); without a profile it
    falls back to alphabetic-max, preserving step-4 isolated-probe
    test behaviour.
"""

from __future__ import annotations

from engine.constant_provenance import (
    ConstantProvenanceConfig,
    classify_values_in_params,
)
from engine.profile.evidence_class_synth import most_restrictive_class_id
from engine.profile.probe_runtime import (
    EvidenceClassCap,
    Probe,
    ProbeContext,
    Verdict,
    register_builtin_probe,
)


@register_builtin_probe("constant_provenance")
class ConstantProvenanceProbe(Probe):
    """Mechanism probe: classify value records by source provenance
    using two orthogonal probes (rerun variability + producer
    dataflow). Each classification carries an evidence-class ceiling
    that caps any downstream claim about that value.

    The probe is stateless — each call classifies value records found
    in ``ctx.params`` independently.
    """

    name = "constant_provenance"
    mechanism = True
    inputs = ("method", "params")
    outputs = ("constant_provenance",)

    def __init__(self, config: ConstantProvenanceConfig | None = None) -> None:
        self._config = config

    def run(self, ctx: ProbeContext) -> Verdict:
        results = classify_values_in_params(ctx.params, cfg=self._config)
        if not results:
            return Verdict(probe=self.name, result="undetermined")

        # Profile-driven cap_mapping override (§19.9 vmp #4 / v0.4.0).
        # When the active profile declares a cap_mapping, the per-category
        # ceiling is taken from there rather than the module-hardcoded
        # _CATEGORY_TABLE. The module fallback preserves v0.3.0 behaviour
        # for callers without a profile.
        ceilings = [
            self._ceiling_for(ctx.profile, r) for r in results
        ]
        ec_tuple = getattr(ctx.profile, "evidence_classes", ()) if ctx.profile else ()
        ceiling = most_restrictive_class_id(
            ceilings,
            evidence_classes=ec_tuple,
        )

        cap: EvidenceClassCap | None = None
        if ceiling:
            cap = EvidenceClassCap(
                class_id=ceiling,
                reason=f"constant_provenance most-restrictive ceiling across "
                f"{len(results)} value record(s)",
            )

        return Verdict(
            probe=self.name,
            result="pass",
            evidence={
                "classifications": [r.to_dict() for r in results],
                "most_restrictive_ceiling": ceiling,
            },
            affects_evidence_class=cap,
        )

    @staticmethod
    def _ceiling_for(profile: object, result) -> str:
        """Resolve the evidence-class ceiling for one classification.

        Profile-declared :attr:`MergedProfile.cap_mapping` wins when
        present — the category-to-class table is domain semantics, and
        a domain that orders evidence classes differently needs the
        right ceiling without a kernel edit. Falls back to whatever the
        underlying module computed (``result.evidence_class_ceiling``)
        when the profile has no override.
        """
        if profile is not None:
            override = getattr(profile, "cap_for_category", None)
            if callable(override):
                value = override(result.category.value)
                if value is not None:
                    return value
        return result.evidence_class_ceiling
