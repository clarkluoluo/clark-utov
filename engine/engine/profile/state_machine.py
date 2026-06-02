"""Node state machine (PLAN §19.1 / §19.4).

The kernel-side state machine. Its single responsibility today is to
materialise the **role → state** binding that
:meth:`ProbeContext.state_for_role` reads, derived from the active
profile's ``node_states`` declarations.

Why this matters: base mechanism rules reference roles (e.g.
``closure_state``), never concrete state names (e.g. ``closed_form``).
When a probe asks ``ctx.state_for_role("closure_state")``, the state
machine answers "which state in this domain plays that role, AND is
that state the node's current state?". A new domain (key extraction)
plugs a different concrete state into the same role and the
mechanism rule keeps working without a single kernel edit.

Step-5 scope: only the static binding (role → state for a given
*current* node state). Dynamic per-node tracking + transition
validation come when the wrapper actually starts threading
``current_state`` through call sites (step 8 + onward).
"""

from __future__ import annotations

from typing import Optional

from engine.profile.probe_runtime import StateView
from engine.profile.registry import MergedProfile
from engine.profile.types import StateSpec


class StateMachine:
    """Holds the active profile's state vocabulary and role bindings.

    Constructed once per merged profile. Read-only — there is no API
    to mutate states or roles at runtime (the open-layer principle
    applies at profile *declaration*, not at runtime).
    """

    def __init__(self, profile: MergedProfile) -> None:
        self._profile = profile
        self._states_by_name: dict[str, StateSpec] = {
            state.name: state for state in profile.node_states
        }
        # role → tuple of state names that bind it (usually 0 or 1, but
        # a domain could legitimately bind one role to multiple states
        # — the machine accepts that and lets the caller decide which
        # is "current" via state_bindings_for(current_state_name)).
        self._states_by_role: dict[str, tuple[str, ...]] = {}
        for state in profile.node_states:
            for role in state.roles:
                self._states_by_role.setdefault(role, ())
                self._states_by_role[role] = self._states_by_role[role] + (state.name,)

    @property
    def profile(self) -> MergedProfile:
        return self._profile

    @property
    def all_states(self) -> tuple[StateSpec, ...]:
        return tuple(self._states_by_name.values())

    def has_state(self, name: str) -> bool:
        return name in self._states_by_name

    def states_binding_role(self, role: str) -> tuple[str, ...]:
        """Names of every node_state that declares ``role`` in its
        ``roles`` list. Empty tuple if no state binds the role."""
        return self._states_by_role.get(role, ())

    def state_bindings_for(self, current_state_name: str | None) -> dict[str, StateView]:
        """Build the ``role → StateView`` map that ProbeContext consumes.

        Given the node's current state name, return one StateView per
        role that the current state plays. If the current state is
        unknown to the profile (typo / state from a removed domain),
        return an empty map — mechanism probes will get None back
        from :meth:`ProbeContext.state_for_role` and treat the role
        as unbound.
        """
        if current_state_name is None:
            return {}
        state = self._states_by_name.get(current_state_name)
        if state is None:
            return {}
        view = StateView(name=state.name, roles=state.roles)
        return {role: view for role in state.roles}

    def state_view(self, name: str) -> Optional[StateView]:
        """Return a StateView for the named state, or None if unknown."""
        state = self._states_by_name.get(name)
        if state is None:
            return None
        return StateView(name=state.name, roles=state.roles)
