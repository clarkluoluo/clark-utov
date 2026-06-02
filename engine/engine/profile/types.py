"""Profile data types — declarable judgment semantics (PLAN §19.1 / §19.2).

These are pure data containers. Behaviour (loading, merging, mechanism
lock enforcement, lint) lives in the sibling modules. All containers
are frozen so registry consumers can hand them out without worrying
about downstream mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class EvidenceClassSpec:
    """One evidence-class tier (e.g. ``A`` / ``B`` / ``C``).

    Order within the profile's ``evidence_classes`` tuple defines
    strength — earlier is stronger. The kernel never references class
    IDs by literal; cap synthesis (§19.3) walks the tuple in order.
    """

    id: str
    desc: str = ""


@dataclass(frozen=True)
class StateSpec:
    """One node-state entry plus the roles it plays.

    The state *name* (``closed_form``, ``env_fixed_observed``, …) is a
    domain enum — extensible per profile. The *roles* the state fills
    (``closure_state``, ``bypass_state``, …) are how base mechanism
    rules reach domain states without hardcoding the names. See §19.1.
    """

    name: str
    roles: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProbeSpec:
    """Declarative reference to a probe registered in the profile.

    ``mechanism=True`` may appear **only** in the base profile —
    enforced by registry merge (§19.4) and by lint (§19.7 #8). When set,
    the probe joins the mechanism baseline: subprofiles cannot
    redeclare or disable it, and the conjunctive gate force-includes
    its verdict at runtime regardless of profile shape.

    ``module`` is the dotted import path for user-provided probes
    (e.g. ``my_pkg.weird_probe``). Base mechanism probes are imported
    from inside the engine and have ``module=None``.
    """

    name: str
    mechanism: bool = False
    module: Optional[str] = None
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateSpec:
    """One gate (entry in the conjunctive closure check).

    The ``rule`` text uses ``verdict(<role>)`` to reach domain states
    via roles — base gates therefore reference roles, never concrete
    state names. ``requires_verdicts`` lists probe names whose verdicts
    must be present for the gate to evaluate.
    """

    id: str
    mechanism: bool = False
    rule: str = ""
    requires_verdicts: tuple[str, ...] = ()


@dataclass(frozen=True)
class RoutingRule:
    """One cause→actions mapping (block_cause router, §19 / PLAN §16)."""

    cause: str
    actions: tuple[str, ...]


@dataclass(frozen=True)
class ScopeRule:
    """One ``when_state → tag_scope`` rule.

    Domain-level — different domains tag scope from different states.
    """

    when_state: str
    tag_scope: str


@dataclass(frozen=True)
class TaskTemplateRef:
    """A profile-declared task template (v0.5.0 / PLAN §20.1.2 task ↔ profile).

    The profile names a reusable task spec by ``name`` and carries the
    raw spec body in ``spec``.  Materialisation happens on demand —
    callers ask the profile for the template name they want, then
    feed the returned dict through :func:`engine.task.parse_task_spec`.

    Task templates are a **domain** concept (recommended procedures
    for one intent class), never base — the lint rejects
    ``task_templates`` in base.
    """

    name: str
    spec: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CapMappingEntry:
    """One ``category → evidence_class id`` entry (v0.4.0 / §19.9 vmp #4).

    The 5-way ``constant_provenance`` source-category labels (e.g.
    ``hardcoded_fixed``, ``session_level_derived``) had their evidence-class
    ceilings hardcoded inside the kernel module. That kept the mapping
    living in two layers: the category vocabulary is mechanism, but the
    cap *value* is domain-specific (a domain that orders evidence classes
    differently needs the right ceiling without a kernel edit). Lifting
    the mapping into a domain profile section closes the gap.
    """

    category: str
    class_id: str


@dataclass(frozen=True)
class Profile:
    """One profile file's parsed form (no inheritance applied yet).

    ``is_base`` is set explicitly by the loader when the file is the
    canonical ``base.json`` — used by the registry to identify the
    mechanism baseline regardless of profile name. All other profiles
    default ``is_base=False`` and cannot opt in (lint refuses
    ``mechanism: true`` outside base).
    """

    name: str
    inherits: Optional[str] = None
    evidence_classes: tuple[EvidenceClassSpec, ...] = field(default_factory=tuple)
    node_states: tuple[StateSpec, ...] = field(default_factory=tuple)
    probes: tuple[ProbeSpec, ...] = field(default_factory=tuple)
    gates: tuple[GateSpec, ...] = field(default_factory=tuple)
    routing_rules: tuple[RoutingRule, ...] = field(default_factory=tuple)
    scope_semantics: tuple[ScopeRule, ...] = field(default_factory=tuple)
    # Domain-declared scope vocabulary ordered narrowest → widest. Empty
    # = legacy (no ordering, ScopeBoundaryGate undetermined).
    scope_order: tuple[str, ...] = field(default_factory=tuple)
    # ``constant_provenance`` category → evidence_class id mapping. Empty
    # = CP probe falls back to its module-hardcoded table.
    cap_mapping: tuple[CapMappingEntry, ...] = field(default_factory=tuple)
    # Profile-recommended reusable task templates (v0.5.0 / §20.1.2).
    # Domain-only — base lint rejects.
    task_templates: tuple[TaskTemplateRef, ...] = field(default_factory=tuple)
    # Names a subprofile asks to disable. Per §19.7 #7 Lock A, the registry
    # rejects any disable: targeting a base mechanism entry. Only legal on
    # non-base profiles (loader enforces).
    disable: tuple[str, ...] = field(default_factory=tuple)
    is_base: bool = False
