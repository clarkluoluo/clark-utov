"""Closure-evidence layering + pseudo-closure trap state — the SAFETY GATE that
stops utov presenting a LOCAL / STRUCTURAL closure as an ALGORITHM closure.

> dev-closure-evidence-layering-trap-state-spec.md (F0 / the reference case color_cipher,
> 2026-06-01/02). clark: "this is not an optimisation, it's a safety gate" —防
> 把「局部闭合 / 结构闭合」误呈现成「算法闭合」.

Closure is NOT a boolean; it is three RECESSIVE, independently-judged layers. Only
the third layer may be called "algorithm closure". This module is the verdict
CLASSIFICATION layer that hangs on ALREADY-EXISTING signals (``SinkVerdict`` /
``ProvenanceVerdict`` / a parity report) — it builds NO new engine (A8①: it does
not re-compute sink/provenance/parity; it reads their verdicts and MECHANICALLY
derives the closure level + label + any trap state).

    structural_closed  — symex converged on an expression (e.g. F=7)
                         → at most ``local_formula``
    provenance_closed  — that expression is on the target output's producer chain
                         → ``candidate_formula`` (on the chain, not yet对拍)
    oracle_closed      — multi-input REAL-runner parity passes AND the independent
                         side has output variance
                         → the ONLY level that may be ``algorithm_closed_form``

Hard pre-condition (mechanical, never self-reported by a stage):

    algorithm_closed_form  ⟺  output_sink_confirmed
                             && provenance_closed
                             && parity_exact         # incl. independent-side
                                                     # observed >= min distinct

Any pre-oracle-closure algorithm result — a primitive (SHA-512 / AES / MD5 /
ChaCha / …) OR a window constant — is the SAME class of false closure: a
structural / local signal masquerading as algorithm closure. It MUST carry an
explicit trap marker (``LOCAL_CLOSURE_ONLY`` / ``PSEUDO_CLOSURE_TRAP``). The trap
states are honest, full-marks terminal states (a clear shape + next step), never a
failure or a blank — feedback_red_line_needs_comfortable_exit.

Generic (A7): the three-layer model holds for ANY recovery case; the trap judgment
is keyed on "constant-ness / primitive-ness + provenance support", never an F0 or
a SHA-512/AES specific field.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

from .oracle_provenance import ProvenanceVerdict
from .oracle_sink import SinkVerdict

# --------------------------------------------------------------------------- #
# The three closure levels + their labels.
# --------------------------------------------------------------------------- #


class ClosureLevel(str, enum.Enum):
    """The three RECESSIVE closure layers (task 0). Ordered weakest → strongest."""

    STRUCTURAL = "structural"     # symex converged on an expression in-window
    PROVENANCE = "provenance"     # the expression is on the output's producer chain
    ORACLE = "oracle"             # multi-input real-runner parity + observed variance


# The label a given level is ALLOWED to be presented as. The strong word
# "algorithm_closed_form" is reserved for ORACLE level only — the spine constraint.
LABEL_LOCAL_FORMULA     = "local_formula"           # STRUCTURAL: window-local only
LABEL_CANDIDATE_FORMULA = "candidate_formula"       # PROVENANCE: on-chain, not对拍
LABEL_ALGORITHM_CLOSED  = "algorithm_closed_form"   # ORACLE: the only true closure


# --------------------------------------------------------------------------- #
# Trap states (task 1). Honest, full-marks terminal states — not failures.
# --------------------------------------------------------------------------- #


class TrapState(str, enum.Enum):
    """Explicit closure-trap markers. ``NONE`` = no trap (either genuinely oracle-
    closed, or there is no closure claim to trap)."""

    NONE = "NONE"
    # A real (possibly non-constant) window formula exists, but it is NOT on the
    # output chain and/or NOT对拍 — local only. The mild trap.
    LOCAL_CLOSURE_ONLY = "LOCAL_CLOSURE_ONLY"
    # A PURE CONSTANT with NO output-provenance support — the strong false-closure
    # trap (F0 F=7 / the 5-handler 0xFB9881B1 collapse).
    PSEUDO_CLOSURE_TRAP = "PSEUDO_CLOSURE_TRAP"


# --------------------------------------------------------------------------- #
# Result.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ClosureClassification:
    """The mechanically-derived closure verdict for one recovery / algorithm result.

    * ``level``  — the highest closure LAYER the evidence actually supports.
    * ``label``  — what the result is ALLOWED to be called at that level.
    * ``trap``   — the explicit trap marker (NONE / LOCAL_CLOSURE_ONLY /
                   PSEUDO_CLOSURE_TRAP) — surfaced LOUDLY (A8④ / never silent).
    * ``algorithm_closed`` — True ONLY when the hard pre-condition holds.
    * the three booleans — the pre-condition inputs, echoed for transparency.
    * ``constant`` / ``provenance_supported`` — the two trap-deciding facts.
    * ``source`` — provenance coordinate of a trapped constant (task 2:
                   {window, idx_range, reg}); ``{}`` when not a constant / not given.
    """

    level: ClosureLevel
    label: str
    trap: TrapState
    algorithm_closed: bool
    output_sink_confirmed: bool
    provenance_closed: bool
    parity_exact: bool
    structural_closed: bool
    is_constant: bool
    provenance_supported: bool
    is_primitive: bool = False
    reason: str = ""
    source: dict[str, Any] = field(default_factory=dict)
    next_step: str = ""

    @property
    def is_trap(self) -> bool:
        return self.trap is not TrapState.NONE

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": "closure_classification",
            "closure_level": self.level.value,
            "label": self.label,
            "trap_state": self.trap.value,
            "is_trap": self.is_trap,
            "algorithm_closed": self.algorithm_closed,
            "output_sink_confirmed": self.output_sink_confirmed,
            "provenance_closed": self.provenance_closed,
            "parity_exact": self.parity_exact,
            "structural_closed": self.structural_closed,
            "is_constant": self.is_constant,
            "is_primitive": self.is_primitive,
            "provenance_supported": self.provenance_supported,
            "reason": self.reason,
        }
        if self.source:
            out["constant_source"] = dict(self.source)
        if self.next_step:
            out["next_step"] = self.next_step
        return out


# --------------------------------------------------------------------------- #
# Signal adapters — derive the three pre-condition booleans from EXISTING verdicts
# (A8①: no new computation; map the verdicts utov already produces).
# --------------------------------------------------------------------------- #


def sink_confirmed_from_verdict(sink_verdict: Any) -> bool:
    """``output_sink_confirmed`` from a :class:`SinkVerdict` (or its string).

    Only SINK_CONFIRMED counts — WRONG_SINK / OUTPUT_NOT_OBSERVABLE do NOT confirm
    the OUTPUT sink (WRONG_SINK located a different region; OUTPUT_NOT_OBSERVABLE
    never observed it). ``None`` → False (unknown is not confirmed)."""
    if sink_verdict is None:
        return False
    v = sink_verdict.value if isinstance(sink_verdict, SinkVerdict) else str(sink_verdict)
    return v == SinkVerdict.SINK_CONFIRMED.value


def provenance_closed_from_verdict(prov_verdict: Any, *, sink_captured: Any = None) -> bool:
    """``provenance_closed`` from a :class:`ProvenanceVerdict` (or its string).

    The expression is on the target output's producer chain when provenance is
    CONTINUOUS_BUFFER or STREAMING (the production is observable and anchored).
    NEEDS_OBSERVATION (chain breaks at an un-captured read) and OPAQUE_CALLEE
    (produced across a boundary we cannot see) do NOT close provenance. A
    ``sink_captured is False`` overrides to NOT-closed regardless of verdict (the
    output writer itself was never observed → nothing anchors)."""
    if sink_captured is False:
        return False
    if prov_verdict is None:
        return False
    v = (prov_verdict.value if isinstance(prov_verdict, ProvenanceVerdict)
         else str(prov_verdict))
    return v in (ProvenanceVerdict.CONTINUOUS_BUFFER.value,
                 ProvenanceVerdict.STREAMING.value)


def parity_exact_from_report(parity_report: Any) -> bool:
    """``parity_exact`` from a parity report (dict or ParityVectorReport-like).

    Reuses the sibling spec's gate: EXACT requires the verdict to be EXACT AND the
    independent side to carry >= min_vectors distinct observed outputs (that gate
    already lives in ``check_parity_vectors``; a report with verdict==EXACT has
    PASSED it). UNCLOSABLE / BLOCK / a missing report → False."""
    if parity_report is None:
        return False
    if isinstance(parity_report, dict):
        verdict = parity_report.get("verdict")
    else:
        verdict = getattr(parity_report, "verdict", None)
    return verdict == "EXACT"


# --------------------------------------------------------------------------- #
# The mechanical classifier (tasks 0 / 1 / 2).
# --------------------------------------------------------------------------- #


def classify_closure(
    *,
    structural_closed: bool,
    output_sink_confirmed: bool,
    provenance_closed: bool,
    parity_exact: bool,
    is_constant: bool = False,
    provenance_supported: bool | None = None,
    is_primitive: bool = False,
    constant_source: dict[str, Any] | None = None,
) -> ClosureClassification:
    """Mechanically derive the closure level + label + trap state.

    This is the SAFETY GATE (task 1): the level/label are DERIVED from the
    pre-condition booleans — no stage may self-report ``closed``. A pure constant
    (or a primitive) without output-provenance support is the strong false-closure
    trap; a real local formula off the chain / not对拍 is the mild trap; only the
    full pre-condition yields ``algorithm_closed_form``.

    Args:
      structural_closed: symex converged on an expression in-window.
      output_sink_confirmed / provenance_closed / parity_exact: the hard
        pre-condition (use the ``*_from_*`` adapters to derive these from the
        engine's existing verdicts).
      is_constant: the structural result is a PURE CONSTANT (no input reference in
        the body). A constant + no provenance support → PSEUDO_CLOSURE_TRAP.
      provenance_supported: whether the constant/expression is backed by an output-
        provenance chain. Defaults to ``provenance_closed`` when not given (task 2:
        a constant is allowed into candidate_formula ONLY with provenance support).
      is_primitive: the result is a recognised crypto primitive (SHA/AES/MD5/…).
        A pre-oracle-closure primitive is the SAME class of false closure as a
        window constant (task 1 / task 7②) — it must carry the trap marker too.
      constant_source: provenance coordinate {window, idx_range, reg} of a trapped
        constant (task 2 — mandatory for a trapped constant so the consumer can read
        "this constant came from idx[0,800] dispatch exit"). Whatever the caller
        has (any subset of keys) — "实有什么报什么" (A7).
    """
    if provenance_supported is None:
        provenance_supported = provenance_closed

    algorithm_closed = bool(
        output_sink_confirmed and provenance_closed and parity_exact)

    # --- ORACLE level: the ONLY true closure (the hard pre-condition holds). ---
    if algorithm_closed:
        return ClosureClassification(
            level=ClosureLevel.ORACLE,
            label=LABEL_ALGORITHM_CLOSED,
            trap=TrapState.NONE,
            algorithm_closed=True,
            output_sink_confirmed=output_sink_confirmed,
            provenance_closed=provenance_closed,
            parity_exact=parity_exact,
            structural_closed=structural_closed,
            is_constant=is_constant,
            provenance_supported=provenance_supported,
            is_primitive=is_primitive,
            reason=("oracle closure: output sink confirmed, provenance closed, AND "
                    "multi-input parity EXACT (independent side output-diverse) — "
                    "the only level that may be called an algorithm closed form"),
            source=dict(constant_source or {}),
            next_step="",
        )

    # --- below ORACLE: derive the trap state + the highest honest level. ---
    # Two pre-oracle-closure trap kinds (both EXPLICIT — the safety gate holds either
    # way; the distinction is severity / shape so the agent reads the right next step):
    #   * PSEUDO_CLOSURE_TRAP (验收①) — a PURE CONSTANT with NO output-provenance
    #     support: a dispatch/window constant masquerading as a formula (F0 F=7 / the
    #     0xFB9881B1 collapse). The strong false-closure trap.
    #   * LOCAL_CLOSURE_ONLY (验收③) — a recognised PRIMITIVE (SHA-512/AES/MD5/…) not
    #     yet oracle-closed: a structural fingerprint, the same CLASS of false closure
    #     but with a primitive shape (no window constant). Surfaced as the local-
    #     closure trap, not presented as a final algorithm.
    # A constant WITH provenance support is NOT trapped here — it advances to
    # candidate_formula (it is at least on the chain; task 2 验收②).
    pseudo = is_constant and not provenance_supported
    primitive_trap = is_primitive and not pseudo   # primitive but not a bare constant

    if pseudo:
        reason = ("PURE CONSTANT with no output-provenance support — a dispatch / "
                  "window constant is NOT a recovered formula; presenting it as "
                  "closure is the pseudo-closure trap (F0 F=7 / the 0xFB9881B1 "
                  "collapse)")
        next_step = ("do NOT advance this constant into the closure path; "
                     "re-anchor on the output's producer chain (recapture the "
                     "output if its writer was never observed)")
        # A pure constant with no provenance is at most structural-local (not on-chain).
        return ClosureClassification(
            level=ClosureLevel.STRUCTURAL,
            label=LABEL_LOCAL_FORMULA,
            trap=TrapState.PSEUDO_CLOSURE_TRAP,
            algorithm_closed=False,
            output_sink_confirmed=output_sink_confirmed,
            provenance_closed=provenance_closed,
            parity_exact=parity_exact,
            structural_closed=structural_closed,
            is_constant=is_constant,
            provenance_supported=provenance_supported,
            is_primitive=is_primitive,
            reason=reason,
            source=dict(constant_source or {}),
            next_step=next_step,
        )

    # --- PRIMITIVE trap (验收③) — SHA/AES/MD5/… not oracle-closed → LOCAL trap. ---
    if primitive_trap:
        return ClosureClassification(
            level=ClosureLevel.STRUCTURAL,
            label=LABEL_LOCAL_FORMULA,
            trap=TrapState.LOCAL_CLOSURE_ONLY,
            algorithm_closed=False,
            output_sink_confirmed=output_sink_confirmed,
            provenance_closed=provenance_closed,
            parity_exact=parity_exact,
            structural_closed=structural_closed,
            is_constant=is_constant,
            provenance_supported=provenance_supported,
            is_primitive=is_primitive,
            reason=("pre-oracle-closure PRIMITIVE hypothesis — a recognised primitive "
                    "is NOT a closed algorithm until oracle closure (sink_confirmed && "
                    "provenance_closed && parity_exact); same CLASS of false closure as "
                    "a window constant, surfaced as the local-closure trap, NOT a final "
                    "algorithm"),
            source=dict(constant_source or {}),
            next_step=("oracle-close it: confirm the output sink, anchor provenance, "
                       "and pass multi-input parity before calling it identified"),
        )

    # --- PROVENANCE level: on the chain, but not对拍 → candidate_formula. ---
    # (structural must hold AND provenance is closed but parity is not yet EXACT.)
    if structural_closed and provenance_closed:
        return ClosureClassification(
            level=ClosureLevel.PROVENANCE,
            label=LABEL_CANDIDATE_FORMULA,
            trap=TrapState.LOCAL_CLOSURE_ONLY,
            algorithm_closed=False,
            output_sink_confirmed=output_sink_confirmed,
            provenance_closed=provenance_closed,
            parity_exact=parity_exact,
            structural_closed=structural_closed,
            is_constant=is_constant,
            provenance_supported=provenance_supported,
            is_primitive=is_primitive,
            reason=("on the output's producer chain but NOT yet对拍 (parity not "
                    "EXACT) — a candidate formula, not a closed algorithm"),
            source=dict(constant_source or {}),
            next_step=("run multi-input real-runner parity (independent side must be "
                       "output-diverse) to promote candidate → algorithm closure"),
        )

    # --- STRUCTURAL level: a real expression, but off-chain / sink unconfirmed. ---
    if structural_closed:
        return ClosureClassification(
            level=ClosureLevel.STRUCTURAL,
            label=LABEL_LOCAL_FORMULA,
            trap=TrapState.LOCAL_CLOSURE_ONLY,
            algorithm_closed=False,
            output_sink_confirmed=output_sink_confirmed,
            provenance_closed=provenance_closed,
            parity_exact=parity_exact,
            structural_closed=structural_closed,
            is_constant=is_constant,
            provenance_supported=provenance_supported,
            is_primitive=is_primitive,
            reason=("a window-local expression exists, but the output sink is not "
                    "confirmed / its production is not anchored to the output's "
                    "producer chain — local closure only, not an algorithm closure"),
            source=dict(constant_source or {}),
            next_step=("confirm the output sink + anchor provenance (the expression "
                       "is on the output path), then对拍, before calling it closed"),
        )

    # --- nothing structural: no closure claim to make (not a trap, just open). ---
    return ClosureClassification(
        level=ClosureLevel.STRUCTURAL,
        label=LABEL_LOCAL_FORMULA,
        trap=TrapState.NONE,
        algorithm_closed=False,
        output_sink_confirmed=output_sink_confirmed,
        provenance_closed=provenance_closed,
        parity_exact=parity_exact,
        structural_closed=False,
        is_constant=is_constant,
        provenance_supported=provenance_supported,
        is_primitive=is_primitive,
        reason="no structural closure (symex did not converge on an expression)",
        source=dict(constant_source or {}),
        next_step="",
    )


__all__ = [
    "ClosureLevel",
    "TrapState",
    "ClosureClassification",
    "LABEL_LOCAL_FORMULA",
    "LABEL_CANDIDATE_FORMULA",
    "LABEL_ALGORITHM_CLOSED",
    "sink_confirmed_from_verdict",
    "provenance_closed_from_verdict",
    "parity_exact_from_report",
    "classify_closure",
]
