"""Evidence-class taxonomy + numeric-claim guard (capability_request.md §P1-1, M1).

The reference target hindsight: numbers such as ``627 vectors`` / ``hook_src_valid:
626`` / ``98.73%`` / ``93/93`` snuck into reports as "success" evidence
without anyone recording *which layer* the number measured. ``hook==sign
0%`` and ``hook_src_valid: 626`` lived side-by-side and the contradiction
went un-noticed.

This module pins down two things:

1. **EvidenceClass** — the closed enum of "what kind of evidence is
   this number based on" (mirrors the table in
   the reference target partial archive's ``mechanism_improvements.md`` §M1/M6).
2. **NumericClaim** — the record we attach to any gate / report field
   that carries a number. Default status is ``pending_review``; only an
   independent verifier ack or a reproducer script may flip it to
   ``confirmed``. The engine refuses to render a "success" verdict that
   references an un-confirmed claim.

The module is **policy data + helpers**; emitters / gate writers import
``record_numeric_claim`` to ensure they never silently drop the
provenance fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvidenceClass(str, Enum):
    """Where the number came from. Add new values **here only** so the
    set stays auditable. Order = increasing strength (rough)."""

    EXPERIMENT          = "experiment"           # un-promoted try; M2
    TRACE_OFFLINE       = "trace_offline"        # static log row
    SYMBOLIC_SLICE      = "symbolic_slice"       # Triton / pseudocode
    IMPLEMENTATION_TEST = "implementation_test"  # python impl vs RFC vectors
    IO_ORACLE           = "io_oracle"            # black-box I/O match
    STATIC_ELF          = "static_elf"           # objdump / xxd reproducible
    BINARY_MEMORY       = "binary_memory"        # runtime byte dump


class ClaimStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    CONFIRMED      = "confirmed"
    INVALIDATED    = "invalidated"


# Patterns the engine MUST refuse to count as success evidence even when
# pass-rate is 100%. ``hook_src_valid: 626`` plus a 32-byte constant
# ``e9a86ab9...`` is the canonical reference target failure mode.
KNOWN_NEGATIVE_PATTERNS: tuple[str, ...] = (
    "e9a86ab9",   # the reference target 32B digest-export constant
)


@dataclass(frozen=True, slots=True)
class NumericClaim:
    """One number-as-evidence record. Every gate / report row carrying
    a numeric metric should embed (or reference) one of these."""

    metric: str                              # e.g. "hook_eq_sign_rate"
    value: float | int | str                 # the raw number
    layer: str                               # e.g. "binary_sm3_body"
    evidence_class: EvidenceClass
    artifact_path: str | None = None         # repro pointer
    status: ClaimStatus = ClaimStatus.PENDING_REVIEW
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric":         self.metric,
            "value":          self.value,
            "layer":          self.layer,
            "evidence_class": self.evidence_class.value,
            "artifact_path":  self.artifact_path,
            "status":         self.status.value,
            "note":           self.note,
        }


class NumericClaimError(ValueError):
    """Raised by guard helpers when a metric is misencoded."""


def record_numeric_claim(
    metric: str,
    value: float | int | str,
    *,
    layer: str,
    evidence_class: EvidenceClass | str,
    artifact_path: str | None = None,
    note: str | None = None,
) -> NumericClaim:
    """Construct a NumericClaim with input checking.

    Always returns a ``pending_review`` claim; promotion is a separate
    operation that requires a verifier ack (see ``confirm_claim``).
    """
    if not metric:
        raise NumericClaimError("metric must be non-empty")
    if not layer:
        raise NumericClaimError("layer must be non-empty (e.g. 'binary_sm3_body')")
    if isinstance(evidence_class, str):
        try:
            evidence_class = EvidenceClass(evidence_class)
        except ValueError as e:
            raise NumericClaimError(
                f"unknown evidence_class {evidence_class!r}; "
                f"choose from {[c.value for c in EvidenceClass]}"
            ) from e
    return NumericClaim(
        metric=metric, value=value, layer=layer,
        evidence_class=evidence_class,
        artifact_path=artifact_path, note=note,
    )


def confirm_claim(claim: NumericClaim, *, verifier_ack: bool) -> NumericClaim:
    """Promote pending_review → confirmed. ``verifier_ack`` must be True;
    a False call (or any non-True value) leaves the claim pending — this
    asymmetry matches M1: an agent may not promote a number without an
    explicit ack."""
    if verifier_ack is not True:
        return claim
    return NumericClaim(
        metric=claim.metric, value=claim.value, layer=claim.layer,
        evidence_class=claim.evidence_class,
        artifact_path=claim.artifact_path,
        status=ClaimStatus.CONFIRMED, note=claim.note,
    )


def invalidate_claim(claim: NumericClaim, reason: str) -> NumericClaim:
    """Set status=invalidated; reason is appended to ``note``."""
    new_note = f"{claim.note} | invalidated: {reason}" if claim.note else f"invalidated: {reason}"
    return NumericClaim(
        metric=claim.metric, value=claim.value, layer=claim.layer,
        evidence_class=claim.evidence_class,
        artifact_path=claim.artifact_path,
        status=ClaimStatus.INVALIDATED, note=new_note,
    )


# ---------------------------------------------------------------------------
# Negative-pattern detector — used by gate emitters to auto-invalidate
# a numeric claim if its dump payload matches a known constant blob.
# ---------------------------------------------------------------------------


def looks_like_known_negative(blob_hex: str) -> str | None:
    """Return the matched pattern name (lowercased) if the dump's hex
    prefix is in ``KNOWN_NEGATIVE_PATTERNS``, else None.

    Comparison is *prefix* on the lowercased hex with whitespace removed,
    so ``"e9a86ab9..."`` matches ``"e9a86ab9"``.
    """
    s = (blob_hex or "").lower().replace(" ", "")
    if not s:
        return None
    for pat in KNOWN_NEGATIVE_PATTERNS:
        if s.startswith(pat):
            return pat
    return None


@dataclass(frozen=True, slots=True)
class GateClaims:
    """Bundle of NumericClaims attached to one gate JSON file. Renderer
    can ask ``has_unconfirmed`` before writing a "pass" verdict."""

    claims: tuple[NumericClaim, ...] = field(default_factory=tuple)

    def has_unconfirmed(self) -> bool:
        return any(c.status != ClaimStatus.CONFIRMED for c in self.claims)

    def to_list(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.claims]
