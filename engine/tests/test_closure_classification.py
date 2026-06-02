"""Closure-evidence layering + pseudo-closure trap state (the SAFETY GATE).

dev-closure-evidence-layering-trap-state-spec.md tasks 0/1/2. Covers the three-
layer model + the trap states across MULTIPLE forms (pure constant / off-chain
non-constant / primitive / full oracle closure) — not just F0's F=7 shape.
"""

from __future__ import annotations

from engine.closure_classification import (
    LABEL_ALGORITHM_CLOSED,
    LABEL_CANDIDATE_FORMULA,
    LABEL_LOCAL_FORMULA,
    ClosureLevel,
    TrapState,
    classify_closure,
    parity_exact_from_report,
    provenance_closed_from_verdict,
    sink_confirmed_from_verdict,
)
from engine.oracle_provenance import ProvenanceVerdict
from engine.oracle_sink import SinkVerdict


# --------------------------------------------------------------------------- #
# Signal adapters — derive the three booleans from EXISTING verdicts (A8①).
# --------------------------------------------------------------------------- #

def test_sink_confirmed_only_on_sink_confirmed():
    assert sink_confirmed_from_verdict(SinkVerdict.SINK_CONFIRMED) is True
    assert sink_confirmed_from_verdict(SinkVerdict.WRONG_SINK) is False
    assert sink_confirmed_from_verdict(SinkVerdict.OUTPUT_NOT_OBSERVABLE) is False
    assert sink_confirmed_from_verdict(None) is False
    # string form
    assert sink_confirmed_from_verdict("SINK_CONFIRMED") is True


def test_provenance_closed_on_buffer_or_streaming_not_needs_observation():
    assert provenance_closed_from_verdict(ProvenanceVerdict.CONTINUOUS_BUFFER) is True
    assert provenance_closed_from_verdict(ProvenanceVerdict.STREAMING) is True
    assert provenance_closed_from_verdict(ProvenanceVerdict.NEEDS_OBSERVATION) is False
    assert provenance_closed_from_verdict(ProvenanceVerdict.OPAQUE_CALLEE) is False
    # sink_captured False overrides even a closing verdict (writer never observed)
    assert provenance_closed_from_verdict(
        ProvenanceVerdict.CONTINUOUS_BUFFER, sink_captured=False) is False


def test_parity_exact_only_on_exact_verdict():
    assert parity_exact_from_report({"verdict": "EXACT"}) is True
    assert parity_exact_from_report({"verdict": "UNCLOSABLE"}) is False
    assert parity_exact_from_report({"verdict": "BLOCK"}) is False
    assert parity_exact_from_report(None) is False


# --------------------------------------------------------------------------- #
# Task 1 验收① — F0 phase C: F=7 constant + no provenance → PSEUDO_CLOSURE_TRAP,
# NEVER algorithm_closed_form.
# --------------------------------------------------------------------------- #

def test_f0_phase_c_constant_no_provenance_is_pseudo_trap():
    cls = classify_closure(
        structural_closed=True,            # symex got F=7
        output_sink_confirmed=False,       # validate_sink=OUTPUT_NOT_OBSERVABLE
        provenance_closed=False,           # trace_provenance=NEEDS_OBSERVATION
        parity_exact=False,
        is_constant=True,                  # F=7 is a pure constant
        provenance_supported=False,
        constant_source={"window": [0, 800], "idx_range": [0, 800],
                         "source": "dispatch_coverage"},
    )
    assert cls.trap is TrapState.PSEUDO_CLOSURE_TRAP
    assert cls.label == LABEL_LOCAL_FORMULA
    assert cls.algorithm_closed is False
    assert cls.label != LABEL_ALGORITHM_CLOSED
    # task 2: the constant's source coordinate is reported.
    d = cls.to_dict()
    assert d["constant_source"]["idx_range"] == [0, 800]
    assert d["trap_state"] == "PSEUDO_CLOSURE_TRAP"


# 验收② — non-constant expression but sink unconfirmed → LOCAL_CLOSURE_ONLY.
def test_offchain_nonconstant_is_local_closure_only():
    cls = classify_closure(
        structural_closed=True,
        output_sink_confirmed=False,
        provenance_closed=False,
        parity_exact=False,
        is_constant=False,                 # a real expr referencing inputs
    )
    assert cls.trap is TrapState.LOCAL_CLOSURE_ONLY
    assert cls.level is ClosureLevel.STRUCTURAL
    assert cls.label == LABEL_LOCAL_FORMULA
    assert cls.algorithm_closed is False


# 验收③ — SHA-512/AES primitive not oracle-closed → explicit LOCAL_CLOSURE_ONLY trap
# (primitive shape), not a final algorithm, not just a silent rename.
def test_primitive_pre_oracle_closure_carries_trap():
    cls = classify_closure(
        structural_closed=False,           # a primitive carries no window expr
        output_sink_confirmed=False,
        provenance_closed=False,
        parity_exact=False,
        is_primitive=True,
    )
    assert cls.is_trap is True
    assert cls.trap is TrapState.LOCAL_CLOSURE_ONLY
    assert cls.algorithm_closed is False
    assert "PRIMITIVE" in cls.reason.upper()


# 验收④ — three layers all satisfied → algorithm_closed_form, zero regression.
def test_full_oracle_closure_is_algorithm_closed():
    cls = classify_closure(
        structural_closed=True,
        output_sink_confirmed=True,
        provenance_closed=True,
        parity_exact=True,
        is_constant=False,
    )
    assert cls.level is ClosureLevel.ORACLE
    assert cls.label == LABEL_ALGORITHM_CLOSED
    assert cls.trap is TrapState.NONE
    assert cls.algorithm_closed is True


def test_primitive_with_full_closure_is_not_trapped():
    # A primitive that DID reach whole-case oracle closure may be called identified.
    cls = classify_closure(
        structural_closed=True,
        output_sink_confirmed=True,
        provenance_closed=True,
        parity_exact=True,
        is_primitive=True,
    )
    assert cls.algorithm_closed is True
    assert cls.trap is TrapState.NONE
    assert cls.label == LABEL_ALGORITHM_CLOSED


# --------------------------------------------------------------------------- #
# Task 2 验收② — a constant WITH provenance support is NOT mis-killed: it advances
# to candidate_formula (on the chain, not yet对拍).
# --------------------------------------------------------------------------- #

def test_constant_with_provenance_support_is_candidate_not_pseudo():
    cls = classify_closure(
        structural_closed=True,
        output_sink_confirmed=True,
        provenance_closed=True,            # on the output producer chain
        parity_exact=False,                # not对拍 yet
        is_constant=True,
        provenance_supported=True,
    )
    assert cls.trap is not TrapState.PSEUDO_CLOSURE_TRAP
    assert cls.level is ClosureLevel.PROVENANCE
    assert cls.label == LABEL_CANDIDATE_FORMULA


def test_nonconstant_onchain_not_parity_is_candidate():
    cls = classify_closure(
        structural_closed=True,
        output_sink_confirmed=True,
        provenance_closed=True,
        parity_exact=False,
        is_constant=False,
    )
    assert cls.level is ClosureLevel.PROVENANCE
    assert cls.label == LABEL_CANDIDATE_FORMULA
    assert cls.trap is TrapState.LOCAL_CLOSURE_ONLY


# nonce-only / no structural closure at all → open, NOT a trap (no claim to trap).
def test_no_structural_closure_is_open_not_trap():
    cls = classify_closure(
        structural_closed=False,
        output_sink_confirmed=False,
        provenance_closed=False,
        parity_exact=False,
        is_constant=False,
    )
    assert cls.trap is TrapState.NONE
    assert cls.algorithm_closed is False
    assert cls.structural_closed is False
