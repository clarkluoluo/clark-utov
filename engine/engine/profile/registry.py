"""Profile registry — chain assembly + mechanism lock (PLAN §19.4).

Loads profiles by name, walks ``inherits`` to assemble a chain ending
at ``base`` (force-included), then merges semantic fields with child
override. Mechanism entries from base profile are **locked**: a
subprofile that redeclares one by name or sets ``mechanism: true``
itself causes registry merge to fail with an explicit error
("Lock A" in §19.7 #7).

Surface is intentionally read-only — there is no ``set_probe`` /
``disable_mechanism`` API. Step-2+ runtime gates also force-include
the mechanism verdicts from base regardless of what shape the merged
profile presents ("Lock B"); that lives in ``gate_runtime``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from engine.profile.loader import ProfileLoadError, load_profile_file
from engine.profile.types import (
    CapMappingEntry,
    EvidenceClassSpec,
    GateSpec,
    Profile,
    ProbeSpec,
    RoutingRule,
    ScopeRule,
    StateSpec,
    TaskTemplateRef,
)


# engine/profiles/  (sibling of the engine/engine/ python package)
PROFILES_DIR: Path = Path(__file__).resolve().parents[2] / "profiles"

BASE_PROFILE_NAME = "base"


class ProfileMergeError(Exception):
    """Raised when inheritance merge violates a mechanism lock or structural rule."""


@dataclass(frozen=True)
class MergedProfile:
    """Resolved (base ∪ domain ∪ user) profile view consumed by Core / runtime.

    ``chain`` is base→leaf, useful for diagnostics. The mechanism index
    sets are filled from the base profile and let downstream runtime
    (gate / state-machine / lint) force-include mechanism entries
    regardless of any subprofile reshape attempt.
    """

    name: str
    chain: tuple[str, ...]
    evidence_classes: tuple[EvidenceClassSpec, ...]
    node_states: tuple[StateSpec, ...]
    probes: tuple[ProbeSpec, ...]
    gates: tuple[GateSpec, ...]
    routing_rules: tuple[RoutingRule, ...]
    scope_semantics: tuple[ScopeRule, ...]
    scope_order: tuple[str, ...]
    cap_mapping: tuple[CapMappingEntry, ...]
    task_templates: tuple[TaskTemplateRef, ...]
    mechanism_probe_names: frozenset[str]
    mechanism_gate_ids: frozenset[str]

    def cap_for_category(self, category: str) -> Optional[str]:
        """Return the evidence_class id this profile maps ``category`` to.

        ``None`` = profile declares no mapping for this category (caller
        may then fall back to a runtime default).
        """
        for entry in self.cap_mapping:
            if entry.category == category:
                return entry.class_id
        return None

    def scope_rank(self, scope: str) -> Optional[int]:
        """Return ``scope``'s position in :attr:`scope_order` (0 =
        narrowest) or ``None`` if the profile doesn't declare it."""
        for i, s in enumerate(self.scope_order):
            if s == scope:
                return i
        return None

    def task_template_for(self, name: str) -> Optional[dict]:
        """Return the raw spec dict for the named template, or ``None``.

        Caller materialises via :func:`engine.task.parse_task_spec`.
        """
        for entry in self.task_templates:
            if entry.name == name:
                return dict(entry.spec)  # defensive copy
        return None

    def task_template_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.task_templates)


_ROLE_PATTERN = re.compile(r"verdict\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)")


class ProfileRegistry:
    """Loads profile files from disk and resolves inheritance chains.

    Caches raw single-file parses and resolved merges. Callers ask for
    a profile by leaf name; the registry walks ``inherits`` until it
    reaches a profile already on the chain (cycle = error) or a
    profile with no ``inherits``, then force-includes the base profile
    if it isn't already on the chain.
    """

    def __init__(self, profiles_dir: Path | str | None = None) -> None:
        self._dir = Path(profiles_dir) if profiles_dir else PROFILES_DIR
        self._raw: dict[str, Profile] = {}
        self._resolved: dict[str, MergedProfile] = {}

    @property
    def profiles_dir(self) -> Path:
        return self._dir

    def load_chain(self, name: str) -> MergedProfile:
        cached = self._resolved.get(name)
        if cached is not None:
            return cached
        chain = self._collect_chain(name)
        merged = self._merge(chain)
        self._resolved[name] = merged
        return merged

    # ------------------------------------------------------------------
    # Chain assembly
    # ------------------------------------------------------------------

    def _collect_chain(self, leaf: str) -> list[Profile]:
        seen_order: list[str] = []
        cursor: Optional[str] = leaf
        chain: list[Profile] = []
        while cursor is not None:
            if cursor in seen_order:
                raise ProfileMergeError(
                    f"profile inheritance cycle: {' -> '.join(seen_order + [cursor])}"
                )
            seen_order.append(cursor)
            profile = self._load_raw(cursor)
            chain.append(profile)
            cursor = profile.inherits

        if not any(p.is_base for p in chain):
            # Force-include base — subprofiles may forget to declare inheritance.
            chain.append(self._load_raw(BASE_PROFILE_NAME))

        chain.reverse()  # base first, leaf last
        return chain

    def _load_raw(self, name: str) -> Profile:
        cached = self._raw.get(name)
        if cached is not None:
            return cached
        is_base = name == BASE_PROFILE_NAME
        path = self._dir / f"{name}.json"
        profile = load_profile_file(path, is_base=is_base)
        if profile.name != name:
            raise ProfileLoadError(
                f"profile {path}: 'profile' field is '{profile.name}' "
                f"but filename implies '{name}'"
            )
        self._raw[name] = profile
        return profile

    # ------------------------------------------------------------------
    # Merge + lock check
    # ------------------------------------------------------------------

    def _merge(self, chain: list[Profile]) -> MergedProfile:
        base = next((p for p in chain if p.is_base), None)
        mech_probes = frozenset(p.name for p in (base.probes if base else ()) if p.mechanism)
        mech_gates = frozenset(g.id for g in (base.gates if base else ()) if g.mechanism)

        # Lock A · load-time: subprofile cannot redeclare or claim mechanism
        for profile in chain:
            if profile.is_base:
                continue
            for probe in profile.probes:
                if probe.name in mech_probes:
                    raise ProfileMergeError(
                        f"profile '{profile.name}': probe '{probe.name}' is "
                        f"mechanism-locked by base — redeclaration / override forbidden"
                    )
                if probe.mechanism:
                    raise ProfileMergeError(
                        f"profile '{profile.name}': probe '{probe.name}' declares "
                        f"'mechanism: true' in a non-base profile — only base may "
                        f"declare mechanism (§19.7 #8)"
                    )
            for gate in profile.gates:
                if gate.id in mech_gates:
                    raise ProfileMergeError(
                        f"profile '{profile.name}': gate '{gate.id}' is "
                        f"mechanism-locked by base — redeclaration / override forbidden"
                    )
                if gate.mechanism:
                    raise ProfileMergeError(
                        f"profile '{profile.name}': gate '{gate.id}' declares "
                        f"'mechanism: true' in a non-base profile (§19.7 #8)"
                    )
            for target in profile.disable:
                if target in mech_probes or target in mech_gates:
                    raise ProfileMergeError(
                        f"profile '{profile.name}': cannot disable mechanism entry "
                        f"'{target}' — base mechanism is locked (§19.7 #7 Lock A)"
                    )

        # Collect non-mechanism disable targets from the chain — domain
        # probes and gates may be freely removed by subprofiles ("open
        # semantic layer"). Mechanism targets already failed above.
        disabled_names: set[str] = set()
        for profile in chain:
            if profile.is_base:
                continue
            for target in profile.disable:
                disabled_names.add(target)

        # Semantic merges
        states: dict[str, StateSpec] = {}
        probes: dict[str, ProbeSpec] = {}
        gates: dict[str, GateSpec] = {}
        evidence_classes: tuple[EvidenceClassSpec, ...] = ()
        routing: dict[str, RoutingRule] = {}
        scope: dict[str, ScopeRule] = {}
        scope_order: tuple[str, ...] = ()
        cap_mapping: dict[str, CapMappingEntry] = {}
        task_templates: dict[str, TaskTemplateRef] = {}

        for profile in chain:
            for state in profile.node_states:
                states[state.name] = state
            for probe in profile.probes:
                probes[probe.name] = probe
            for gate in profile.gates:
                gates[gate.id] = gate
            if profile.evidence_classes:
                # Per §19.4: child fully replaces evidence_classes ordering to
                # avoid ambiguity. (Explicit ``extend:`` syntax may be added
                # in a later step if real usage requires it.)
                evidence_classes = profile.evidence_classes
            for rule in profile.routing_rules:
                routing[rule.cause] = rule
            for sc_rule in profile.scope_semantics:
                scope[sc_rule.when_state] = sc_rule
            if profile.scope_order:
                # Child fully replaces ordering — same rationale as evidence_classes.
                scope_order = profile.scope_order
            for cm in profile.cap_mapping:
                cap_mapping[cm.category] = cm
            for tmpl in profile.task_templates:
                # Child overrides parent template of the same name —
                # consistent with semantic-merge rules elsewhere.
                task_templates[tmpl.name] = tmpl

        # Apply non-mechanism disables — silently drop unknown targets
        # (typo-tolerant; intent is "if this is on the chain, remove it").
        # Mechanism targets cannot enter this set (rejected above).
        for name_ in list(probes.keys()):
            if name_ in disabled_names:
                del probes[name_]
        for gate_id in list(gates.keys()):
            if gate_id in disabled_names:
                del gates[gate_id]

        # Orphan-role check — base references a role; some profile in the
        # chain must bind that role to ≥1 node_state.
        referenced_roles = self._collect_base_referenced_roles(base)
        bound_roles = {role for state in states.values() for role in state.roles}
        orphans = referenced_roles - bound_roles
        if orphans:
            raise ProfileMergeError(
                f"profile '{chain[-1].name}': base references role(s) "
                f"{sorted(orphans)} but no node_state in the merged chain binds "
                f"them (§19.7 #1)"
            )

        return MergedProfile(
            name=chain[-1].name,
            chain=tuple(p.name for p in chain),
            evidence_classes=evidence_classes,
            node_states=tuple(states.values()),
            probes=tuple(probes.values()),
            gates=tuple(gates.values()),
            routing_rules=tuple(routing.values()),
            scope_semantics=tuple(scope.values()),
            scope_order=scope_order,
            cap_mapping=tuple(cap_mapping.values()),
            task_templates=tuple(task_templates.values()),
            mechanism_probe_names=mech_probes,
            mechanism_gate_ids=mech_gates,
        )

    @staticmethod
    def _collect_base_referenced_roles(base: Optional[Profile]) -> set[str]:
        """Parse base gate rules for ``verdict(<role>)`` references."""
        if base is None:
            return set()
        roles: set[str] = set()
        for gate in base.gates:
            for match in _ROLE_PATTERN.finditer(gate.rule):
                roles.add(match.group(1))
        return roles
