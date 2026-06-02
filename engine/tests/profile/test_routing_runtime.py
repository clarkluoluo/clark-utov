"""RoutingTable — profile-driven cause → action lookup (PLAN §19.4).

Step 6 acceptance: ``block_cause`` reaches into the profile for the
action mapping. When the wrapper provides a ``RoutingTable``, the
router honours the profile's declaration; without one, it falls
back to the legacy hardcoded enum mapping (so step-2-to-5 callers
keep working). A domain profile that re-maps a cause to a different
action (e.g. ``recognition_gap → escalate_user`` for a domain with
no L2 layer) routes accordingly with zero code change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.block_cause import (
    BlockCauseClass,
    BlockCauseRouter,
    NodeContext,
    RoutingAction,
)
from engine.profile import BASE_PROFILE_NAME, ProfileRegistry, RoutingTable
from engine.profile.routing_runtime import lint_actions_against_known


VMP_PROFILE = "vmp_algorithm_extraction"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(profiles_dir: Path, name: str, body: dict) -> Path:
    path = profiles_dir / f"{name}.json"
    path.write_text(json.dumps(body))
    return path


@pytest.fixture()
def vmp_routing_table() -> RoutingTable:
    reg = ProfileRegistry()
    return RoutingTable(reg.load_chain(VMP_PROFILE))


@pytest.fixture()
def profiles_dir(tmp_path: Path) -> Path:
    d = tmp_path / "profiles"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Shipped VMP profile — routing table mirrors the legacy hardcoded mapping
# ---------------------------------------------------------------------------


def test_vmp_routing_covers_all_four_causes(vmp_routing_table):
    assert set(vmp_routing_table.causes) == {
        "collection_gap",
        "recognition_gap",
        "strategy_gap",
        "true_boundary",
    }


def test_vmp_collection_gap_lists_two_actions(vmp_routing_table):
    """collection_gap is the capability-aware fork: auto_collect when
    runner can capture, register_backlog when it can't. Profile
    declares both, the router picks at runtime based on the capability
    oracle."""
    assert vmp_routing_table.lookup("collection_gap") == (
        "auto_collect",
        "register_backlog",
    )


def test_vmp_recognition_gap_routes_to_l2(vmp_routing_table):
    assert vmp_routing_table.primary_action("recognition_gap") == "escalate_l2"


def test_vmp_strategy_gap_routes_to_l3(vmp_routing_table):
    assert vmp_routing_table.primary_action("strategy_gap") == "escalate_l3"


def test_vmp_true_boundary_routes_to_user(vmp_routing_table):
    assert vmp_routing_table.primary_action("true_boundary") == "escalate_user"


def test_unknown_cause_returns_empty(vmp_routing_table):
    """No rule for the cause → empty tuple / None. Callers fall back
    to their own default."""
    assert vmp_routing_table.lookup("imaginary_cause") == ()
    assert vmp_routing_table.primary_action("imaginary_cause") is None


# ---------------------------------------------------------------------------
# Action vocabulary lint
# ---------------------------------------------------------------------------


def test_lint_passes_when_profile_actions_match_known(vmp_routing_table):
    """All four causes' actions should map cleanly onto RoutingAction
    enum values shipped with block_cause."""
    known = {a.value for a in RoutingAction}
    assert lint_actions_against_known(vmp_routing_table, known) == []


def test_lint_flags_unknown_action_in_profile(profiles_dir):
    """A profile that names ``escalate_l4`` (no such action) gets
    flagged before the router silently falls back."""
    _write(
        profiles_dir,
        BASE_PROFILE_NAME,
        {"profile": BASE_PROFILE_NAME},
    )
    _write(
        profiles_dir,
        "weird",
        {
            "profile": "weird",
            "routing_rules": [
                {"cause": "strategy_gap", "actions": ["escalate_l4"]},
            ],
        },
    )
    reg = ProfileRegistry(profiles_dir)
    table = RoutingTable(reg.load_chain("weird"))
    known = {a.value for a in RoutingAction}
    violations = lint_actions_against_known(table, known)
    assert any("escalate_l4" in v for v in violations)


# ---------------------------------------------------------------------------
# BlockCauseRouter actually consults the routing table
# ---------------------------------------------------------------------------


def _recognition_gap_node() -> NodeContext:
    """A node where data is collected but pattern recognition has
    failed — class-2 recognition_gap."""
    return NodeContext(
        node_id="some_node",
        data_collected=True,
        pattern_recognised=False,
        failure_summary="template fit failed",
    )


def _strategy_gap_node() -> NodeContext:
    return NodeContext(
        node_id="strategy_stuck",
        data_collected=True,
        pattern_recognised=True,
        symbolised=True,
        strategy_resolved=False,
        failure_summary="paradigm switch needed",
    )


def _true_boundary_node() -> NodeContext:
    return NodeContext(
        node_id="boundary",
        data_collected=True,
        pattern_recognised=True,
        symbolised=True,
        strategy_resolved=True,
        failure_summary="cannot proceed",
    )


def test_router_without_routing_table_falls_back_to_legacy(vmp_routing_table):
    """When no routing_table is supplied (existing callers), the
    router uses its hardcoded enum mapping. This is the
    backward-compat property — step 2/3/4/5 tests keep working."""
    router = BlockCauseRouter()  # no routing_table
    result = router.route(node_context=_recognition_gap_node())
    assert result.action is RoutingAction.ESCALATE_L2


def test_router_with_vmp_routing_table_matches_legacy_behaviour(vmp_routing_table):
    """VMP profile's routing_rules are identical to the legacy
    hardcoded mapping, so behaviour is unchanged when the table is
    supplied — that's the migration's correctness check."""
    router = BlockCauseRouter(routing_table=vmp_routing_table)

    assert router.route(node_context=_recognition_gap_node()).action is RoutingAction.ESCALATE_L2
    assert router.route(node_context=_strategy_gap_node()).action is RoutingAction.ESCALATE_L3
    assert router.route(node_context=_true_boundary_node()).action is RoutingAction.ESCALATE_USER


def test_router_honours_custom_profile_remapping(profiles_dir):
    """A different domain that doesn't have an L2 layer can declare
    ``recognition_gap → escalate_user`` and the router routes
    accordingly — zero code change. This is the migration's
    why-bother: domains can re-route causes by editing the profile."""
    _write(
        profiles_dir,
        BASE_PROFILE_NAME,
        {"profile": BASE_PROFILE_NAME},
    )
    _write(
        profiles_dir,
        "no_l2_domain",
        {
            "profile": "no_l2_domain",
            "routing_rules": [
                # This domain has no small-LLM L2; recognition gaps
                # go straight to the user.
                {"cause": "recognition_gap", "actions": ["escalate_user"]},
                {"cause": "strategy_gap",    "actions": ["escalate_l3"]},
                {"cause": "true_boundary",   "actions": ["escalate_user"]},
            ],
        },
    )
    reg = ProfileRegistry(profiles_dir)
    table = RoutingTable(reg.load_chain("no_l2_domain"))
    router = BlockCauseRouter(routing_table=table)

    # Recognition gap now routes to user, not L2:
    result = router.route(node_context=_recognition_gap_node())
    assert result.action is RoutingAction.ESCALATE_USER

    # Strategy gap unchanged:
    assert (
        router.route(node_context=_strategy_gap_node()).action
        is RoutingAction.ESCALATE_L3
    )


def test_router_falls_back_to_legacy_when_profile_lacks_rule(profiles_dir):
    """A profile that doesn't declare a rule for a given cause → the
    router falls back to its hardcoded default for that cause.
    Partial migrations are safe."""
    _write(
        profiles_dir,
        BASE_PROFILE_NAME,
        {"profile": BASE_PROFILE_NAME},
    )
    _write(
        profiles_dir,
        "partial",
        {
            "profile": "partial",
            "routing_rules": [
                # Only override recognition_gap. Strategy/true_boundary
                # have no rule → legacy default kicks in.
                {"cause": "recognition_gap", "actions": ["escalate_user"]},
            ],
        },
    )
    reg = ProfileRegistry(profiles_dir)
    table = RoutingTable(reg.load_chain("partial"))
    router = BlockCauseRouter(routing_table=table)

    assert (
        router.route(node_context=_recognition_gap_node()).action
        is RoutingAction.ESCALATE_USER  # from profile
    )
    assert (
        router.route(node_context=_strategy_gap_node()).action
        is RoutingAction.ESCALATE_L3  # legacy fallback
    )


def test_router_falls_back_on_unknown_action_id(profiles_dir):
    """A profile that names an action id the RoutingAction enum
    doesn't recognise → the router falls back to legacy rather than
    crashing. ``lint_actions_against_known`` is supposed to catch
    this earlier, but the router stays robust if lint was skipped."""
    _write(
        profiles_dir,
        BASE_PROFILE_NAME,
        {"profile": BASE_PROFILE_NAME},
    )
    _write(
        profiles_dir,
        "bogus_action",
        {
            "profile": "bogus_action",
            "routing_rules": [
                {"cause": "recognition_gap", "actions": ["escalate_l99"]},
            ],
        },
    )
    reg = ProfileRegistry(profiles_dir)
    table = RoutingTable(reg.load_chain("bogus_action"))
    router = BlockCauseRouter(routing_table=table)

    # Unknown action → legacy fallback kicks in.
    assert (
        router.route(node_context=_recognition_gap_node()).action
        is RoutingAction.ESCALATE_L2
    )
