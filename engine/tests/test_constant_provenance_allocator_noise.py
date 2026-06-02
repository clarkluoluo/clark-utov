"""v0.4.0 B5 — allocator-noise filter for ``constant_provenance`` dataflow.

A producer-dataflow probe that observes a ``malloc`` / ``mmap`` side-effect
into the destination register is not seeing a *source* — the allocator
just materialises the buffer; whatever populates the concrete bytes is
the real source. v0.3.0 treated any non-static read kind as a signal
(allocator was effectively ``UNKNOWN``-like), polluting a HARDCODED-shaped
verdict toward SESSION_LEVEL when the trace happened to also touch an
allocator. This module ships :data:`NOISE_READS` and filters allocator
reads out of every classification path.

Acceptance:

  * ``producer_reads = [allocator, static]`` → still classifies as
    ``HARDCODED_FIXED`` (was poisoned to UNDETERMINED before).
  * ``producer_reads = [allocator, time]``  → still ``SESSION_LEVEL`` —
    time entropy is meaningful, allocator is filtered.
  * ``producer_reads = [allocator]`` alone → ``UNDETERMINED`` (no
    information about concrete bytes).
  * ``reads_session_entropy`` / ``reads_input`` / ``reads_appkey``
    consult the filtered view; allocator alone returns False on all.
"""

from __future__ import annotations

import pytest

from engine.constant_provenance import (
    DataflowReadKind,
    DataflowSummary,
    NOISE_READS,
    SESSION_ENTROPY_READS,
    SourceCategory,
    classify_from_dataflow_only,
    classify_value,
    RerunDimension,
    RerunObservation,
)


# ---------------------------------------------------------------------------
# DataflowSummary semantics
# ---------------------------------------------------------------------------


def test_allocator_is_in_noise_reads():
    assert DataflowReadKind.ALLOCATOR in NOISE_READS


def test_allocator_is_not_session_entropy():
    """Allocator must not be classed as a session-entropy source —
    that's what caused the pollution this filter exists to fix."""
    assert DataflowReadKind.ALLOCATOR not in SESSION_ENTROPY_READS


def test_meaningful_reads_filters_allocator():
    df = DataflowSummary(producer_reads=(
        DataflowReadKind.ALLOCATOR, DataflowReadKind.STATIC,
    ))
    assert df.meaningful_reads() == (DataflowReadKind.STATIC,)


def test_raw_producer_reads_preserves_allocator():
    """Diagnostic surface — allocator stays visible on the raw tuple
    so a reader can confirm the producer DID touch an allocator; only
    the *classification* methods skip it."""
    df = DataflowSummary(producer_reads=(
        DataflowReadKind.ALLOCATOR, DataflowReadKind.STATIC,
    ))
    assert DataflowReadKind.ALLOCATOR in df.producer_reads
    assert df.reads_allocator() is True


def test_reads_only_static_with_allocator_noise_still_true():
    df = DataflowSummary(producer_reads=(
        DataflowReadKind.ALLOCATOR, DataflowReadKind.STATIC,
    ))
    assert df.reads_only_static_or_appkey() is True


def test_reads_only_allocator_returns_false_on_meaningful_checks():
    df = DataflowSummary(producer_reads=(DataflowReadKind.ALLOCATOR,))
    assert df.meaningful_reads() == ()
    assert df.reads_session_entropy() is False
    assert df.reads_input() is False
    assert df.reads_appkey() is False
    # No meaningful reads -> "only static" returns False (the helper
    # requires at least one meaningful read to claim a category).
    assert df.reads_only_static_or_appkey() is False


# ---------------------------------------------------------------------------
# classify_from_dataflow_only behaviour
# ---------------------------------------------------------------------------


def test_classify_dataflow_allocator_plus_static_is_hardcoded_fixed():
    """Tag-1 acceptance: allocator noise must not poison a
    hardcoded-only verdict."""
    df = DataflowSummary(producer_reads=(
        DataflowReadKind.ALLOCATOR, DataflowReadKind.STATIC,
    ))
    assert classify_from_dataflow_only(df) is SourceCategory.HARDCODED_FIXED


def test_classify_dataflow_allocator_plus_time_is_session_level():
    df = DataflowSummary(producer_reads=(
        DataflowReadKind.ALLOCATOR, DataflowReadKind.TIME,
    ))
    assert classify_from_dataflow_only(df) is SourceCategory.SESSION_LEVEL_DERIVED


def test_classify_dataflow_allocator_only_is_undetermined():
    """Allocator alone tells you nothing about the concrete bytes — the
    honest verdict is UNDETERMINED, not a confidently-wrong category."""
    df = DataflowSummary(producer_reads=(DataflowReadKind.ALLOCATOR,))
    assert classify_from_dataflow_only(df) is SourceCategory.UNDETERMINED


def test_classify_dataflow_allocator_plus_input_is_per_input_variable():
    df = DataflowSummary(producer_reads=(
        DataflowReadKind.ALLOCATOR, DataflowReadKind.INPUT,
    ))
    assert classify_from_dataflow_only(df) is SourceCategory.PER_INPUT_VARIABLE


# ---------------------------------------------------------------------------
# Cross-classifier: rerun stable + (allocator + static) → still HARDCODED
# ---------------------------------------------------------------------------


def _stable_reruns():
    """Five same-bytes observations across the three relevant axes."""
    return [
        RerunObservation(dimension=RerunDimension.SAME_SESSION,  value_hex="DEAD"),
        RerunObservation(dimension=RerunDimension.SAME_SESSION,  value_hex="DEAD"),
        RerunObservation(dimension=RerunDimension.NEW_SESSION,   value_hex="DEAD"),
        RerunObservation(dimension=RerunDimension.NEW_APPKEY,    value_hex="DEAD"),
        RerunObservation(dimension=RerunDimension.NEW_PER_INPUT, value_hex="DEAD"),
    ]


def test_classify_value_stable_reruns_plus_allocator_static_is_hardcoded():
    """Regression of the v0.3.0 reference-case finding — allocator noise
    in the dataflow trace must not downgrade a stable-rerun verdict."""
    result = classify_value(
        "flag",
        rerun_observations=_stable_reruns(),
        dataflow=DataflowSummary(producer_reads=(
            DataflowReadKind.ALLOCATOR, DataflowReadKind.STATIC,
        )),
    )
    assert result.category is SourceCategory.HARDCODED_FIXED
    assert result.evidence_class_ceiling == "A"
