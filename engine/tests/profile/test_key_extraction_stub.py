"""key_extraction stub profile — acceptance §19.7 #3.

This is the profile layer's central promise, demonstrated:
**a completely different domain, expressed in profile data alone,
makes the engine behave correctly with zero kernel source change.**

What "correct" looks like here:

  * The profile loads cleanly with the same registry / loader / lint
    code that handles VMP. No engine code branches on "is this VMP
    or not".
  * The new domain's vocabulary (``key_material_located``,
    ``key_recoverable``, ``key_verified``, …) is recognised and the
    role binding (``key_verified → closure_state``) actually wires
    up — so a mechanism rule referencing ``closure_state`` resolves
    to ``key_verified`` for this domain, just as it resolves to
    ``closed_form`` for VMP. Different state name, same mechanism.
  * Base mechanism probes (M1 / M3 / constant_provenance /
    value_provenance / watch_first_write) work unchanged against
    fixtures shaped for the new domain. They don't know or care that
    "the closure state" is now called ``key_verified``.

Practical check on "no kernel source change": this entire file +
``engine/profiles/key_extraction.json`` are the only paths the step-7
commit should add. If a step-7 reviewer sees a diff under
``engine/engine/`` they should reject — the profile layer is
supposed to remove the need for such diffs.
"""

from __future__ import annotations

import pytest

from engine.profile import (
    BASE_PROFILE_NAME,
    ConjunctiveGate,
    ProbeContext,
    ProfileRegistry,
    StateMachine,
)


PROFILE = "key_extraction"


@pytest.fixture()
def merged():
    return ProfileRegistry().load_chain(PROFILE)


@pytest.fixture()
def gate(merged) -> ConjunctiveGate:
    return ConjunctiveGate(merged)


# ---------------------------------------------------------------------------
# Profile loads cleanly via the same registry path as VMP
# ---------------------------------------------------------------------------


def test_key_extraction_profile_loads(merged):
    assert merged.name == PROFILE
    assert merged.chain == (BASE_PROFILE_NAME, PROFILE)


def test_key_extraction_inherits_base_mechanism_set(merged):
    """The five base mechanism probes are present and locked, exactly
    the same as under VMP — the domain layer adds, it doesn't touch
    the baseline."""
    expected = {
        "m1_success_audit",
        "m3_bypass_block",
        "constant_provenance",
        "value_provenance",
        "watch_first_write",
    }
    assert expected.issubset(merged.mechanism_probe_names)


def test_key_extraction_declares_its_own_state_vocabulary(merged):
    """Three brand-new states — never appeared in VMP profile. The
    registry merge accepts them without complaint."""
    names = {s.name for s in merged.node_states}
    for new_state in ("key_material_located", "key_recoverable", "key_verified"):
        assert new_state in names, f"missing new state {new_state}"


def test_key_extraction_evidence_classes_independent_of_vmp(merged):
    """The new domain redeclares its own evidence-class descriptions
    (still A/B/C identifiers, but the desc text reflects
    key-extraction semantics — that's the freedom domain profiles
    have)."""
    ids = [ec.id for ec in merged.evidence_classes]
    assert ids == ["A", "B", "C"]
    # The desc text proves it's the key-extraction declaration, not
    # the VMP one that happens to share the IDs.
    assert "verified" in merged.evidence_classes[0].desc.lower()


def test_key_extraction_has_its_own_scope_rule(merged):
    """Scope semantics is a domain field; key_extraction declares
    one specific to its state vocabulary."""
    rule = next(
        (s for s in merged.scope_semantics if s.when_state == "key_recoverable"),
        None,
    )
    assert rule is not None
    assert rule.tag_scope == "recoverable_session_only"


# ---------------------------------------------------------------------------
# Role binding — the key indirection point (§19.1)
# ---------------------------------------------------------------------------


def test_key_verified_binds_to_closure_state_role(merged):
    """The whole point of the role-binding layer: a different concrete
    state name plays the same mechanism role. Base mechanism rules
    that reference ``closure_state`` resolve to ``key_verified`` in
    this domain — and to ``closed_form`` in VMP — with zero kernel
    change."""
    key_verified = next(s for s in merged.node_states if s.name == "key_verified")
    assert "closure_state" in key_verified.roles


def test_state_machine_resolves_closure_state_to_key_verified(merged):
    """The StateMachine actually produces a ProbeContext binding that
    a mechanism probe would see when this domain's node is in the
    ``key_verified`` state — and that binding's name is
    ``key_verified``, not ``closed_form``."""
    machine = StateMachine(merged)
    bindings = machine.state_bindings_for("key_verified")
    assert "closure_state" in bindings
    assert bindings["closure_state"].name == "key_verified"


def test_state_machine_returns_no_binding_for_intermediate_states(merged):
    """Other states (located / recoverable) have no roles → mechanism
    probes querying closure_state get None back and treat the role
    as unbound (correct — the node hasn't reached closure yet)."""
    machine = StateMachine(merged)
    assert machine.state_bindings_for("key_material_located") == {}
    assert machine.state_bindings_for("key_recoverable") == {}


# ---------------------------------------------------------------------------
# Base mechanism probes work against a key-extraction-shaped fixture
# ---------------------------------------------------------------------------


def test_m1_audit_works_against_key_extraction_node(gate, merged):
    """M1's algorithm doesn't care which domain it's in — a bare
    success claim with a pinned dimension gets downgraded the same way
    whether the node lives in VMP or in key_extraction. That's the
    payoff of putting M1 in base."""
    overfit_params = {
        "report": {
            "target_success":       True,
            "archival_allowed":     True,
            "success_dependencies": ["session_id", "key_addr_offset"],
            "samples": [
                # session_id is pinned (the prefix-fixed equivalent
                # for the key-extraction domain) — M1 should flag this.
                {"session_id": "fixed_session", "key_addr_offset": i}
                for i in range(50)
            ],
            "pass_rate":  1.0,
            "scope":      "in_session",
            "closure_paths": [
                {"name": "recompute_a", "digest": "abc"},
                {"name": "recompute_b", "digest": "abc"},
            ],
        }
    }
    result = gate.evaluate(
        ProbeContext(method="promote_to_finding", params=overfit_params, profile=merged)
    )
    # M1 fires through the gate — claim is downgraded because
    # session_id is pinned.
    assert result.passed is False
    assert "m1_success_audit" in result.failing_probes


def test_value_provenance_caps_observed_in_key_domain(gate, merged):
    """observed source → cap B, regardless of domain. Confirms
    value_provenance treats key_extraction values identically to
    VMP values — observation-without-recompute is universal."""
    params = {
        "values": [
            {
                "value_name": "session_key_bytes",
                "source": "hook",
                "evidence_class": "A",  # caller's optimistic claim
            }
        ]
    }
    result = gate.evaluate(
        ProbeContext(method="record_value", params=params, profile=merged)
    )
    assert result.node_cap is not None
    assert result.node_cap.class_id == "B"


def test_routing_rules_present_for_key_extraction(merged):
    """key_extraction declares the same cause → action mapping as VMP
    by default, because the mechanism (cause classifier) is the same
    cross-domain and the default L1/L2/L3/L4 ladder applies to either
    target type. A domain that wanted different routing could declare
    its own mapping freely."""
    rules = {r.cause: r.actions for r in merged.routing_rules}
    assert "collection_gap" in rules
    assert "true_boundary" in rules
    assert rules["true_boundary"] == ("escalate_user",)
