"""weird_target_x — acceptance §19.7 #4: inheritance + incremental addition.

The user-facing scenario: someone hits a target that's *mostly* like
a VMP algorithm-extraction case but has one structural quirk —
say, a custom block-dispatch table that needs its own signature
probe. They don't want to re-declare every VMP state and probe; they
want to inherit ``vmp_algorithm_extraction`` and add their increment.

Acceptance:

  * Profile loads with the full three-step chain
    ``base → vmp_algorithm_extraction → weird_target_x``.
  * Every state, probe, evidence_class, routing rule, and scope
    rule inherited from the parents is still present in the merged
    view — the child didn't accidentally erase its inheritance.
  * The child's increment is present (one new state, one new domain
    probe).
  * Base mechanism set is unchanged — adding a child profile cannot
    weaken the baseline (the same Lock A property the adversarial
    suite verifies negatively; here verified positively).
"""

from __future__ import annotations

import pytest

from engine.profile import (
    BASE_PROFILE_NAME,
    ProfileRegistry,
)


PROFILE = "weird_target_x"
VMP = "vmp_algorithm_extraction"


@pytest.fixture()
def merged():
    return ProfileRegistry().load_chain(PROFILE)


# ---------------------------------------------------------------------------
# Inheritance chain
# ---------------------------------------------------------------------------


def test_chain_is_base_then_vmp_then_weird(merged):
    assert merged.chain == (BASE_PROFILE_NAME, VMP, PROFILE)


# ---------------------------------------------------------------------------
# Inherited from vmp_algorithm_extraction — all still present
# ---------------------------------------------------------------------------


def test_all_vmp_states_carried_through(merged):
    names = {s.name for s in merged.node_states}
    expected_vmp_states = {
        "closed_form",
        "env_fixed_observed",
        "observed_stable",
        "stuck",
        "capability_gap",
    }
    assert expected_vmp_states.issubset(names)


def test_vmp_role_binding_carried_through(merged):
    """closure_state was bound to closed_form in VMP — must still
    resolve to closed_form here. (The weird target didn't bind a
    second state to closure_state; if it had, that would be a
    legitimate domain re-binding, but the increment didn't.)"""
    closed_form = next(
        (s for s in merged.node_states if s.name == "closed_form"), None
    )
    assert closed_form is not None
    assert "closure_state" in closed_form.roles


def test_vmp_evidence_classes_carried_through(merged):
    """vmp_algorithm_extraction declared evidence_classes A/B/C.
    weird_target_x didn't redeclare → inherited as-is."""
    ids = [ec.id for ec in merged.evidence_classes]
    assert ids == ["A", "B", "C"]


def test_vmp_length_chain_probe_still_registered(merged):
    """The domain probe from the parent profile survives the merge."""
    probe_names = {p.name for p in merged.probes}
    assert "length_chain_check" in probe_names


def test_vmp_routing_rules_carried_through(merged):
    """All four cause→action rules from vmp_algorithm_extraction."""
    causes = {r.cause for r in merged.routing_rules}
    assert {
        "collection_gap",
        "recognition_gap",
        "strategy_gap",
        "true_boundary",
    }.issubset(causes)


def test_vmp_scope_rule_carried_through(merged):
    rule = next(
        (s for s in merged.scope_semantics if s.when_state == "env_fixed_observed"),
        None,
    )
    assert rule is not None
    assert rule.tag_scope == "env_bound"


# ---------------------------------------------------------------------------
# Child's increment is also present
# ---------------------------------------------------------------------------


def test_child_added_new_state(merged):
    names = {s.name for s in merged.node_states}
    assert "block_dispatch_table_resolved" in names


def test_child_added_new_domain_probe(merged):
    probe_names = {p.name for p in merged.probes}
    assert "weird_block_signature_check" in probe_names
    # And it's non-mechanism — the child can't sneak in a mechanism.
    probe = next(p for p in merged.probes if p.name == "weird_block_signature_check")
    assert probe.mechanism is False


def test_child_added_new_scope_rule(merged):
    rule = next(
        (
            s for s in merged.scope_semantics
            if s.when_state == "block_dispatch_table_resolved"
        ),
        None,
    )
    assert rule is not None
    assert rule.tag_scope == "post_dispatch_table"


# ---------------------------------------------------------------------------
# Base mechanism set unchanged through the deeper chain
# ---------------------------------------------------------------------------


def test_all_five_base_mechanisms_still_locked(merged):
    expected_mechanism = {
        "m1_success_audit",
        "m3_bypass_block",
        "constant_provenance",
        "value_provenance",
        "watch_first_write",
    }
    assert expected_mechanism.issubset(merged.mechanism_probe_names)
    # And every one of them is still in the probe list as a mechanism entry.
    for name in expected_mechanism:
        probe = next(p for p in merged.probes if p.name == name)
        assert probe.mechanism is True


def test_child_did_not_dilute_mechanism_count(merged):
    """The child profile's increment didn't push, hide, or shadow any
    mechanism probe.  Count grows as new mechanism probes ship
    (v0.4.0 added scope_boundary_gate + scope_upscale_gate); the
    property is "≥ v0.3.0 baseline" plus every original name still
    flagged."""
    mechanism_probes = [p for p in merged.probes if p.mechanism]
    assert len(mechanism_probes) >= 5


# ---------------------------------------------------------------------------
# Open-layer remains open through the chain
# ---------------------------------------------------------------------------


def test_grandchild_can_still_disable_inherited_domain_probe(tmp_path):
    """A profile that inherits weird_target_x can disable the
    grandfather's length_chain_check just as cleanly as if it
    inherited vmp_algorithm_extraction directly. Open layer is
    transitive."""
    import json

    pdir = tmp_path / "profiles"
    pdir.mkdir()

    # Mirror the shipped chain in the temp dir so the registry can
    # resolve the inheritance path without touching the real profiles.
    (pdir / "base.json").write_text(json.dumps({
        "profile": "base",
        "probes": [{"name": name, "mechanism": True} for name in (
            "m1_success_audit", "m3_bypass_block", "constant_provenance",
            "value_provenance", "watch_first_write",
        )],
    }))
    (pdir / "vmp_algorithm_extraction.json").write_text(json.dumps({
        "profile": "vmp_algorithm_extraction",
        "inherits": "base",
        "node_states": [
            {"name": "closed_form", "roles": ["closure_state"]},
        ],
        "probes": [{"name": "length_chain_check"}],
    }))
    (pdir / "weird_target_x.json").write_text(json.dumps({
        "profile": "weird_target_x",
        "inherits": "vmp_algorithm_extraction",
        "probes": [{"name": "weird_block_signature_check"}],
    }))
    (pdir / "grandchild.json").write_text(json.dumps({
        "profile": "grandchild",
        "inherits": "weird_target_x",
        "disable": ["length_chain_check"],
    }))

    reg = ProfileRegistry(pdir)
    grandchild = reg.load_chain("grandchild")
    probe_names = {p.name for p in grandchild.probes}

    # Inherited domain probe disabled, even though declared two levels up:
    assert "length_chain_check" not in probe_names
    # But its sibling (added at the same parent layer) survives:
    assert "weird_block_signature_check" in probe_names
    # And mechanism set is still intact:
    assert "m1_success_audit" in grandchild.mechanism_probe_names
