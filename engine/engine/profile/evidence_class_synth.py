"""Evidence-class cap synthesis (PLAN §19.3).

Replaces the alphabetic-max placeholder that lived in
:mod:`engine.profile.probes.constant_provenance` /
:mod:`engine.profile.probes.value_provenance` during step 4.

The cap rule, stated structurally rather than by ASCII accident:
when multiple verdicts cap the same node, the resulting cap is the
**weakest** — i.e. the cap whose ``class_id`` sits furthest from
position 0 in the active profile's ``evidence_classes`` tuple
(``earlier = stronger``). If the profile reorders or renames the
classes (``S > A > B > C`` instead of ``A > B > C``), this helper
re-derives the ordering from the profile without any kernel code
change. That's the property §19.7 #5 asks for.

Fallback: when no profile context is available (probes constructed
without a :class:`ProbeContext.profile`, legacy callers, tests
running probes in isolation), the helper degrades to alphabetic-max
so step-4 tests and existing v0.2.0-dev behaviour continue to work.
The fallback is documented as an explicit second-class path — the
profile-driven branch is what production should hit once the
wrapper starts threading the profile through (step 8).
"""

from __future__ import annotations

from typing import Iterable, Optional

from engine.profile.probe_runtime import EvidenceClassCap, Verdict
from engine.profile.types import EvidenceClassSpec


def most_restrictive_class_id(
    class_ids: Iterable[str],
    evidence_classes: Iterable[EvidenceClassSpec] = (),
) -> Optional[str]:
    """Return the weakest (most-restrictive) class id from ``class_ids``.

    If ``evidence_classes`` is non-empty, "weakest" = the one with the
    *highest* position in the profile's ordering. If empty (or none of
    the inputs are recognised by the profile), falls back to
    alphabetic max — the step-4 heuristic that happens to be correct
    for ``A < B < C``.
    """
    ids = [c for c in class_ids if c]
    if not ids:
        return None

    order: dict[str, int] = {ec.id: i for i, ec in enumerate(evidence_classes)}
    if order:
        # Filter to IDs actually known to the profile; if none match,
        # we still need a sensible answer for the unknown IDs, so
        # fall through to the alphabetic fallback.
        known = [c for c in ids if c in order]
        if known:
            return max(known, key=lambda c: order[c])

    # Profile-less or all-unknown fallback. Documented as second-class.
    return max(ids)


def synth_node_cap(
    verdicts: Iterable[Verdict],
    evidence_classes: Iterable[EvidenceClassSpec] = (),
) -> Optional[EvidenceClassCap]:
    """Combine all ``affects_evidence_class`` caps across ``verdicts``
    into a single node-level cap. Returns ``None`` if no verdict
    carried a cap.

    The composed cap's ``reason`` summarises how many verdicts
    contributed, so the synth result is self-documenting in
    envelopes / ledgers.
    """
    caps = [v.affects_evidence_class for v in verdicts if v.affects_evidence_class is not None]
    if not caps:
        return None

    ec_tuple = tuple(evidence_classes)
    winner_id = most_restrictive_class_id(
        (cap.class_id for cap in caps),
        evidence_classes=ec_tuple,
    )
    if winner_id is None:
        return None

    contributors = [cap for cap in caps if cap.class_id == winner_id]
    reason = (
        f"synth across {len(caps)} verdict cap(s); winning class {winner_id} "
        f"contributed by {len(contributors)} verdict(s) "
        f"({', '.join(c.reason or '<unspecified>' for c in contributors)})"
    )
    return EvidenceClassCap(class_id=winner_id, reason=reason)
