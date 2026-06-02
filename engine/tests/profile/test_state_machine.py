"""StateMachine — role→state binding (PLAN §19.1).

Step-5 surface: only the static role-binding lookup that
:class:`ProbeContext` reads. Dynamic per-node tracking lands when
the wrapper threads ``current_state`` through call sites.
"""

from __future__ import annotations

import pytest

from engine.profile import (
    BASE_PROFILE_NAME,
    ProbeContext,
    ProfileRegistry,
    StateMachine,
)


VMP_PROFILE = "vmp_algorithm_extraction"


@pytest.fixture()
def vmp_machine() -> StateMachine:
    reg = ProfileRegistry()
    return StateMachine(reg.load_chain(VMP_PROFILE))


# ---------------------------------------------------------------------------
# Construction + lookup
# ---------------------------------------------------------------------------


def test_state_machine_holds_profile_states(vmp_machine):
    names = {s.name for s in vmp_machine.all_states}
    expected = {
        "closed_form",
        "env_fixed_observed",
        "observed_stable",
        "stuck",
        "capability_gap",
    }
    assert expected.issubset(names)


def test_state_machine_knows_role_to_state_binding(vmp_machine):
    bound = vmp_machine.states_binding_role("closure_state")
    assert bound == ("closed_form",)


def test_unknown_role_returns_empty_binding(vmp_machine):
    assert vmp_machine.states_binding_role("absolutely_not_a_real_role") == ()


def test_state_view_for_known_state(vmp_machine):
    view = vmp_machine.state_view("closed_form")
    assert view is not None
    assert view.name == "closed_form"
    assert "closure_state" in view.roles


def test_state_view_for_unknown_state_is_none(vmp_machine):
    assert vmp_machine.state_view("definitely_not_a_state") is None


# ---------------------------------------------------------------------------
# Bindings for ProbeContext.state_for_role
# ---------------------------------------------------------------------------


def test_bindings_for_closed_form_resolve_closure_state(vmp_machine):
    bindings = vmp_machine.state_bindings_for("closed_form")
    assert "closure_state" in bindings
    assert bindings["closure_state"].name == "closed_form"


def test_bindings_for_non_role_state_are_empty(vmp_machine):
    """env_fixed_observed has no roles — looking up role bindings
    for that current state yields an empty map, and ctx returns None
    for any role query."""
    bindings = vmp_machine.state_bindings_for("env_fixed_observed")
    assert bindings == {}

    ctx = ProbeContext(state_bindings=bindings)
    assert ctx.state_for_role("closure_state") is None


def test_bindings_for_unknown_state_are_empty(vmp_machine):
    """A state name not declared in the profile (typo / removed) →
    empty bindings, ctx returns None — mechanism rule treats role
    as unbound and proceeds (the mechanism conjunctive gate handles
    the "no closure evidence" case explicitly)."""
    assert vmp_machine.state_bindings_for("imaginary_state") == {}


def test_bindings_for_no_current_state_are_empty(vmp_machine):
    """``current_state_name=None`` is legal — caller hasn't set the
    node's state yet (early-pipeline scenario)."""
    assert vmp_machine.state_bindings_for(None) == {}


# ---------------------------------------------------------------------------
# ProbeContext integration — the role lookup actually works
# ---------------------------------------------------------------------------


def test_ctx_state_for_role_returns_view_when_state_plays_role(vmp_machine):
    """The whole point of the indirection layer: a mechanism rule
    written against the ``closure_state`` role gets the actual VMP
    ``closed_form`` state back, without ever spelling that name."""
    bindings = vmp_machine.state_bindings_for("closed_form")
    ctx = ProbeContext(state_bindings=bindings)

    view = ctx.state_for_role("closure_state")
    assert view is not None
    assert view.name == "closed_form"


def test_ctx_state_for_role_returns_none_when_role_unplayed(vmp_machine):
    bindings = vmp_machine.state_bindings_for("observed_stable")
    ctx = ProbeContext(state_bindings=bindings)
    assert ctx.state_for_role("closure_state") is None


# ---------------------------------------------------------------------------
# Cross-domain check (the role-binding payoff)
# ---------------------------------------------------------------------------


def test_base_profile_has_no_states_so_role_unbound():
    """A registry asked for just `base` produces a state machine with
    zero states — there's no domain to bind roles. Mechanism probes
    that reference roles will see None and must handle it (M1 etc.
    don't currently call state_for_role, so this is a future-proof
    check)."""
    reg = ProfileRegistry()
    base_machine = StateMachine(reg.load_chain(BASE_PROFILE_NAME))
    assert base_machine.all_states == ()
    assert base_machine.states_binding_role("closure_state") == ()
