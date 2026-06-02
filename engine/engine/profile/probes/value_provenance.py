"""Value-provenance probe adapter (PLAN §19 / IMPL_PLAN §P1.0 step 4).

Wraps :mod:`engine.value_provenance` — the principle "values backed
only by observation must be capped, never claimed as full closure"
is universal across exploration domains, so the probe lives in the
base profile with ``mechanism: true``. The specific evidence-class
ceiling (A/B/C) emitted by the underlying module is the current VMP
encoding; step 5 reframes the cap through the active profile's
ordering, at which point the cap targets become profile-driven.

Mapping :class:`ValueProvenanceResult` → :class:`Verdict`:

  * ``no value records``         → ``undetermined`` (probe doesn't
    apply)
  * ``records found``            → ``pass`` (the probe is a
    classifier; it doesn't reject calls, it just caps them)
  * ``affects_evidence_class``   → most-restrictive ceiling across
    all tagged records, computed via
    :func:`engine.profile.evidence_class_synth.most_restrictive_class_id`.
    Profile-ordered when ``ctx.profile`` is set; alphabetic-max
    fallback otherwise (same shape as ConstantProvenanceProbe).
"""

from __future__ import annotations

from engine.profile.evidence_class_synth import most_restrictive_class_id
from engine.profile.probe_runtime import (
    EvidenceClassCap,
    Probe,
    ProbeContext,
    Verdict,
    register_builtin_probe,
)
from engine.value_provenance import (
    ValueProvenanceConfig,
    tag_values_in_params,
)


@register_builtin_probe("value_provenance")
class ValueProvenanceProbe(Probe):
    """Mechanism probe: tag each value record by provenance
    (observed / closed_form / hybrid / unknown) and emit the
    eligible evidence-class ceiling. Caller's claimed
    ``evidence_class`` is independently downgraded to the ceiling
    by the underlying module's in-place rewrite — that mutation is
    part of the M1+ rule and is preserved.
    """

    name = "value_provenance"
    mechanism = True
    inputs = ("method", "params")
    outputs = ("value_provenance",)

    def __init__(self, config: ValueProvenanceConfig | None = None) -> None:
        self._config = config

    def run(self, ctx: ProbeContext) -> Verdict:
        results = tag_values_in_params(ctx.params, cfg=self._config)
        if not results:
            return Verdict(probe=self.name, result="undetermined")

        ec_tuple = getattr(ctx.profile, "evidence_classes", ()) if ctx.profile else ()
        ceiling = most_restrictive_class_id(
            (r.ceiling for r in results),
            evidence_classes=ec_tuple,
        )

        cap: EvidenceClassCap | None = None
        if ceiling:
            cap = EvidenceClassCap(
                class_id=ceiling,
                reason=f"value_provenance most-restrictive ceiling across "
                f"{len(results)} value record(s)",
            )

        return Verdict(
            probe=self.name,
            result="pass",
            evidence={
                "tagged_values": [r.to_dict() for r in results],
                "most_restrictive_ceiling": ceiling,
                "any_downgraded": any(r.downgraded for r in results),
                "any_parity_disclaimer": any(r.parity_disclaimer for r in results),
            },
            affects_evidence_class=cap,
        )
