"""v0.4.0 B7 — gap-class realignment (§19.9 base #5).

The 4-way :class:`BlockCauseClass` predates the field categorisation
that emerged from tc3: every gap landed in one of three *kinds* and
each kind demanded a different routing action.  This module ships the
3-way :class:`GapKind` enum + a deterministic 4-way → 3-way mapping +
a routing fall-through so profile authors may state policy by the
broader gap kind without giving up the fine-grained vocabulary.

Mapping:

  * COLLECTION_GAP                       → CAPABILITY_GAP
  * RECOGNITION_GAP, STRATEGY_GAP        → ANALYSIS_INCOMPLETE
  * TRUE_BOUNDARY                        → BOUNDARY_LIMIT
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.block_cause import (
    BacklogWriter,
    BlockCauseClass,
    BlockCauseClassification,
    BlockCauseRouter,
    BlockCauseSignal,
    GapKind,
    NodeContext,
    RoutingAction,
    classify,
    gap_kind_for,
)


# ---------------------------------------------------------------------------
# 4-way → 3-way mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls, expected", [
    (BlockCauseClass.COLLECTION_GAP,  GapKind.CAPABILITY_GAP),
    (BlockCauseClass.RECOGNITION_GAP, GapKind.ANALYSIS_INCOMPLETE),
    (BlockCauseClass.STRATEGY_GAP,    GapKind.ANALYSIS_INCOMPLETE),
    (BlockCauseClass.TRUE_BOUNDARY,   GapKind.BOUNDARY_LIMIT),
])
def test_gap_kind_for_each_block_cause_class(cls, expected):
    assert gap_kind_for(cls) is expected


def test_gap_kind_mapping_is_total():
    """Every BlockCauseClass must have a GapKind mapping — no silent
    misroutes."""
    for cls in BlockCauseClass:
        assert isinstance(gap_kind_for(cls), GapKind)


# ---------------------------------------------------------------------------
# Classification exposes gap_kind
# ---------------------------------------------------------------------------


def _stuck_node_ctx() -> NodeContext:
    return NodeContext(
        node_id="value@0xdeadbeef",
        data_collected=True,
        pattern_recognised=True,
        symbolised=True,
        strategy_resolved=False,
        failure_summary="symbex returned no model",
    )


def test_classification_to_dict_includes_gap_kind():
    cls = classify(None, node_context=_stuck_node_ctx())
    payload = cls.to_dict()
    assert payload["gap_kind"] == cls.gap_kind.value


def test_classification_gap_kind_property_matches_class():
    cls = BlockCauseClassification(
        cls=BlockCauseClass.RECOGNITION_GAP,
        signals=(),
    )
    assert cls.gap_kind is GapKind.ANALYSIS_INCOMPLETE


# ---------------------------------------------------------------------------
# Profile-routing fall-through: domain may declare by gap_kind
# ---------------------------------------------------------------------------


def _vmp_with_gap_kind_routing(tmp_path: Path):
    """Build a profile that routes ``analysis_incomplete`` instead of the
    fine-grained recognition_gap / strategy_gap pair."""
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "base.json").write_text(json.dumps({
        "profile": "base",
        "probes": [
            {"name": "m1_success_audit",    "mechanism": True},
            {"name": "m3_bypass_block",     "mechanism": True},
            {"name": "constant_provenance", "mechanism": True},
            {"name": "value_provenance",    "mechanism": True},
            {"name": "watch_first_write",   "mechanism": True},
        ],
    }))
    (profiles_dir / "vmp_with_gap_kind.json").write_text(json.dumps({
        "profile": "vmp_with_gap_kind",
        "inherits": "base",
        "evidence_classes": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
        "node_states": [{"name": "closed_form", "roles": ["closure_state"]}],
        "routing_rules": [
            {"cause": "capability_gap",      "actions": ["auto_collect", "register_backlog"]},
            {"cause": "analysis_incomplete", "actions": ["escalate_l3"]},
            {"cause": "boundary_limit",      "actions": ["escalate_user"]},
        ],
    }))
    return profiles_dir


def test_router_falls_through_to_gap_kind_when_class_name_absent(tmp_path):
    """Profile declares only the 3-way names; router still routes
    recognition_gap → escalate_l3 by way of gap_kind = analysis_incomplete."""
    from engine.profile.registry import ProfileRegistry
    from engine.profile.routing_runtime import RoutingTable

    reg = ProfileRegistry(_vmp_with_gap_kind_routing(tmp_path))
    profile = reg.load_chain("vmp_with_gap_kind")
    table = RoutingTable(profile)

    router = BlockCauseRouter(routing_table=table)
    # recognition_gap not declared by fine-grained name; gap_kind is.
    result = router._action_for(BlockCauseClass.RECOGNITION_GAP,
                                RoutingAction.ESCALATE_L2)
    assert result is RoutingAction.ESCALATE_L3


def test_router_fine_grained_name_still_wins_when_declared(tmp_path):
    """If a profile DOES declare the fine-grained name, that wins —
    fall-through is only a fallback."""
    from engine.profile.registry import ProfileRegistry
    from engine.profile.routing_runtime import RoutingTable

    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "base.json").write_text(json.dumps({
        "profile": "base",
        "probes": [
            {"name": "m1_success_audit",    "mechanism": True},
            {"name": "m3_bypass_block",     "mechanism": True},
            {"name": "constant_provenance", "mechanism": True},
            {"name": "value_provenance",    "mechanism": True},
            {"name": "watch_first_write",   "mechanism": True},
        ],
    }))
    (profiles_dir / "vmp_dual.json").write_text(json.dumps({
        "profile": "vmp_dual",
        "inherits": "base",
        "evidence_classes": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
        "node_states": [{"name": "closed_form", "roles": ["closure_state"]}],
        "routing_rules": [
            {"cause": "recognition_gap",     "actions": ["escalate_l2"]},
            {"cause": "analysis_incomplete", "actions": ["escalate_user"]},
        ],
    }))
    reg = ProfileRegistry(profiles_dir)
    profile = reg.load_chain("vmp_dual")
    table = RoutingTable(profile)
    router = BlockCauseRouter(routing_table=table)
    result = router._action_for(BlockCauseClass.RECOGNITION_GAP,
                                RoutingAction.ESCALATE_L2)
    assert result is RoutingAction.ESCALATE_L2


def test_router_legacy_default_when_neither_name_declared():
    """No routing_table = no override = legacy behaviour."""
    router = BlockCauseRouter()
    result = router._action_for(BlockCauseClass.RECOGNITION_GAP,
                                RoutingAction.ESCALATE_L2)
    assert result is RoutingAction.ESCALATE_L2
