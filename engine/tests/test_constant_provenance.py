"""constant_provenance — classifier + dataflow override + ceiling."""

from __future__ import annotations

import pytest

from engine.constant_provenance import (
    ConstantProvenanceConfig,
    DataflowReadKind,
    DataflowSummary,
    RerunDimension,
    RerunObservation,
    RoutingAction,
    SourceCategory,
    analyse_reruns,
    classify_from_dataflow_only,
    classify_value,
    classify_values_in_params,
    render_constant_provenance_alert,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _obs(dim: RerunDimension, hex_val: str) -> RerunObservation:
    return RerunObservation(dimension=dim, value_hex=hex_val)


def _all_axes_stable(value: str = "deadbeef") -> list[RerunObservation]:
    return [
        _obs(RerunDimension.SAME_SESSION, value),
        _obs(RerunDimension.SAME_SESSION, value),
        _obs(RerunDimension.NEW_SESSION, value),
        _obs(RerunDimension.NEW_APPKEY,  value),
        _obs(RerunDimension.NEW_PER_INPUT, value),
    ]


# ---------------------------------------------------------------------------
# Probe one — rerun analysis
# ---------------------------------------------------------------------------


def test_analyse_reruns_all_stable():
    a = analyse_reruns(_all_axes_stable(), cfg=ConstantProvenanceConfig())
    assert a.same_session_stable is True
    assert a.new_session_stable is True
    assert a.new_appkey_stable is True
    assert a.new_per_input_stable is True
    assert a.axes_with_evidence() == 4


def test_analyse_reruns_session_axis_changes():
    obs = [
        _obs(RerunDimension.SAME_SESSION, "aa"),
        _obs(RerunDimension.SAME_SESSION, "aa"),
        _obs(RerunDimension.NEW_SESSION, "bb"),
    ]
    a = analyse_reruns(obs, cfg=ConstantProvenanceConfig())
    assert a.same_session_stable is True
    assert a.new_session_stable is False


def test_analyse_reruns_axis_without_samples_is_none():
    obs = [
        _obs(RerunDimension.SAME_SESSION, "aa"),
        _obs(RerunDimension.SAME_SESSION, "aa"),
    ]
    a = analyse_reruns(obs, cfg=ConstantProvenanceConfig())
    assert a.new_session_stable is None
    assert a.new_appkey_stable is None
    assert a.new_per_input_stable is None


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------


def test_hardcoded_fixed_needs_static_dataflow_corroboration():
    """All-stable reruns + static-only dataflow → HARDCODED_FIXED.
    Without dataflow, we downgrade to UNDETERMINED (entropy-lock
    blindspot)."""
    df = DataflowSummary(producer_reads=(DataflowReadKind.STATIC,))
    r = classify_value("k", rerun_observations=_all_axes_stable(),
                       dataflow=df, cfg=ConstantProvenanceConfig())
    assert r.category is SourceCategory.HARDCODED_FIXED
    assert r.evidence_class_ceiling == "A"
    assert r.scope == "universal"
    assert r.recommended_action is RoutingAction.AUTO_PIN


def test_appkey_fixed_function_from_appkey_axis_change():
    obs = [
        _obs(RerunDimension.SAME_SESSION, "aa"),
        _obs(RerunDimension.SAME_SESSION, "aa"),
        _obs(RerunDimension.NEW_SESSION,  "aa"),
        _obs(RerunDimension.NEW_APPKEY,   "bb"),
        _obs(RerunDimension.NEW_PER_INPUT, "aa"),
    ]
    r = classify_value("k", rerun_observations=obs,
                       cfg=ConstantProvenanceConfig())
    assert r.category is SourceCategory.APPKEY_FIXED_FUNCTION
    assert r.evidence_class_ceiling == "A"
    assert r.scope == "per_appkey"
    assert r.recommended_action is RoutingAction.MARK_DUAL_PATH


def test_session_level_from_session_axis_change():
    obs = [
        _obs(RerunDimension.SAME_SESSION, "aa"),
        _obs(RerunDimension.SAME_SESSION, "aa"),
        _obs(RerunDimension.NEW_SESSION,  "cc"),
        _obs(RerunDimension.NEW_PER_INPUT, "aa"),
    ]
    r = classify_value("session_key", rerun_observations=obs,
                       cfg=ConstantProvenanceConfig())
    assert r.category is SourceCategory.SESSION_LEVEL_DERIVED
    assert r.evidence_class_ceiling == "B"
    assert r.scope == "per_session"
    assert r.recommended_action is RoutingAction.ESCALATE_USAGE_DECISION


def test_per_input_variable_from_input_axis_change():
    obs = [
        _obs(RerunDimension.SAME_SESSION, "aa"),
        _obs(RerunDimension.SAME_SESSION, "aa"),
        _obs(RerunDimension.NEW_PER_INPUT, "dd"),
    ]
    r = classify_value("input_derived", rerun_observations=obs,
                       cfg=ConstantProvenanceConfig())
    assert r.category is SourceCategory.PER_INPUT_VARIABLE
    assert r.evidence_class_ceiling == ""   # no constant claim
    assert r.scope == "per_input"
    assert r.recommended_action is RoutingAction.TREAT_AS_VARIABLE


# ---------------------------------------------------------------------------
# Dataflow override — the entropy-locked blindspot.
# ---------------------------------------------------------------------------


def test_dataflow_session_entropy_overrides_stable_reruns():
    """All-stable reruns BUT producer reads time/random/session_token
    → flip to SESSION_LEVEL_DERIVED. Probe two saves us from the
    entropy-locked test environment trap."""
    df = DataflowSummary(producer_reads=(
        DataflowReadKind.STATIC, DataflowReadKind.SESSION_TOKEN,
    ))
    r = classify_value("template", rerun_observations=_all_axes_stable(),
                       dataflow=df, cfg=ConstantProvenanceConfig())
    assert r.category is SourceCategory.SESSION_LEVEL_DERIVED
    assert "dataflow.reads_session_entropy=true" in r.signals
    assert r.evidence_class_ceiling == "B"
    assert r.scope == "per_session"


def test_dataflow_only_classification_when_no_reruns():
    df = DataflowSummary(producer_reads=(DataflowReadKind.STATIC,))
    cat = classify_from_dataflow_only(df)
    assert cat is SourceCategory.HARDCODED_FIXED

    df = DataflowSummary(producer_reads=(DataflowReadKind.APPKEY,))
    cat = classify_from_dataflow_only(df)
    assert cat is SourceCategory.APPKEY_FIXED_FUNCTION

    df = DataflowSummary(producer_reads=(DataflowReadKind.TIME,))
    cat = classify_from_dataflow_only(df)
    assert cat is SourceCategory.SESSION_LEVEL_DERIVED

    df = DataflowSummary(producer_reads=(DataflowReadKind.INPUT,))
    cat = classify_from_dataflow_only(df)
    assert cat is SourceCategory.PER_INPUT_VARIABLE


def test_dataflow_only_classification_when_reruns_empty():
    df = DataflowSummary(producer_reads=(DataflowReadKind.APPKEY,))
    r = classify_value("k", rerun_observations=[], dataflow=df,
                       cfg=ConstantProvenanceConfig())
    assert r.category is SourceCategory.APPKEY_FIXED_FUNCTION


# ---------------------------------------------------------------------------
# Undetermined — the safe-default path.
# ---------------------------------------------------------------------------


def test_undetermined_when_no_evidence_anywhere():
    r = classify_value("k", cfg=ConstantProvenanceConfig())
    assert r.category is SourceCategory.UNDETERMINED
    assert r.recommended_action is RoutingAction.REQUEST_MORE_OBSERVATIONS


def test_undetermined_when_reruns_stable_but_no_dataflow():
    """Without a dataflow probe we can't rule out entropy-lock —
    UNDETERMINED is the honest answer."""
    r = classify_value("k", rerun_observations=_all_axes_stable(),
                       cfg=ConstantProvenanceConfig())
    assert r.category is SourceCategory.UNDETERMINED
    assert "no dataflow evidence" in r.reasoning


def test_disabled_returns_undetermined():
    r = classify_value("k", rerun_observations=_all_axes_stable(),
                       cfg=ConstantProvenanceConfig(enabled=False))
    assert r.category is SourceCategory.UNDETERMINED
    assert "disabled" in r.reasoning


# ---------------------------------------------------------------------------
# Params walk
# ---------------------------------------------------------------------------


def test_classify_values_in_params_picks_up_records():
    params = {
        "report": {
            "values": [
                {"value_name": "session_key",
                 "rerun_observations": [
                     {"dimension": "same_session", "value_hex": "aa"},
                     {"dimension": "same_session", "value_hex": "aa"},
                     {"dimension": "new_session",  "value_hex": "bb"},
                 ]},
                {"value_name": "no_probes_yet"},   # skipped silently
            ],
        },
    }
    results = classify_values_in_params(params,
                                        cfg=ConstantProvenanceConfig())
    assert len(results) == 1
    assert results[0].value_name == "session_key"
    assert results[0].category is SourceCategory.SESSION_LEVEL_DERIVED


def test_classify_values_in_params_parses_dataflow():
    params = {
        "values": [
            {"value_name": "template",
             "rerun_observations": [
                 {"dimension": "same_session", "value_hex": "aa"},
                 {"dimension": "same_session", "value_hex": "aa"},
                 {"dimension": "new_session",  "value_hex": "aa"},
                 {"dimension": "new_appkey",   "value_hex": "aa"},
                 {"dimension": "new_per_input", "value_hex": "aa"},
             ],
             "producer_dataflow": {
                 "producer_reads": ["static", "session_token"],
             }},
        ],
    }
    results = classify_values_in_params(params,
                                        cfg=ConstantProvenanceConfig())
    assert len(results) == 1
    # Dataflow override → SESSION_LEVEL_DERIVED despite stable reruns.
    assert results[0].category is SourceCategory.SESSION_LEVEL_DERIVED


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------


def test_render_alert_lists_categories():
    r = classify_value(
        "session_key",
        rerun_observations=[
            _obs(RerunDimension.SAME_SESSION, "aa"),
            _obs(RerunDimension.SAME_SESSION, "aa"),
            _obs(RerunDimension.NEW_SESSION, "bb"),
        ],
        cfg=ConstantProvenanceConfig(),
    )
    line = render_constant_provenance_alert([r])
    assert line is not None
    assert "session_key" in line
    assert "session_level_derived" in line
    assert "per_session" in line


def test_render_alert_surfaces_undetermined():
    r = classify_value("k", cfg=ConstantProvenanceConfig())
    line = render_constant_provenance_alert([r])
    assert line is not None
    assert "undetermined" in line


def test_render_alert_none_when_empty():
    assert render_constant_provenance_alert([]) is None
