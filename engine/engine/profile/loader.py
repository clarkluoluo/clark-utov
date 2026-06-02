"""Profile file loader (PLAN §19, IMPL_PLAN §P1.0).

Single-file parse only — inheritance resolution lives in
``engine.profile.registry``. Profile files ship as JSON in v0.3.0
(zero new runtime deps); a YAML loader can be wired in later by
swapping ``json.loads`` for a YAML parser and adding ``pyyaml`` to
``[project.optional-dependencies]``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


class ProfileLoadError(Exception):
    """Raised when a profile file is missing, malformed, or self-inconsistent."""


def load_profile_file(path: str | Path, *, is_base: bool = False) -> Profile:
    """Parse one profile file into a :class:`Profile`.

    ``is_base`` is set explicitly by the registry when the file is the
    canonical ``base.json``. Callers should not flip this for
    arbitrary files — only the registry can.
    """
    path = Path(path)
    if not path.exists():
        raise ProfileLoadError(f"profile file not found: {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ProfileLoadError(f"profile {path} is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise ProfileLoadError(f"profile {path} must be a JSON object at the top level")

    name = raw.get("profile")
    if not isinstance(name, str) or not name:
        raise ProfileLoadError(f"profile {path} missing required string field 'profile'")

    inherits = raw.get("inherits")
    if inherits is not None and not isinstance(inherits, str):
        raise ProfileLoadError(f"profile {path}: 'inherits' must be a string or null")

    disable = raw.get("disable", [])
    if disable is None:
        disable = []
    if not isinstance(disable, list) or any(not isinstance(d, str) for d in disable):
        raise ProfileLoadError(
            f"profile {path}: 'disable' must be a list of strings"
        )
    if is_base and disable:
        raise ProfileLoadError(
            f"profile {path}: 'disable' is not allowed on the base profile"
        )

    return Profile(
        name=name,
        inherits=inherits,
        evidence_classes=_parse_evidence_classes(raw.get("evidence_classes", []), path),
        node_states=_parse_node_states(raw.get("node_states", []), path),
        probes=_parse_probes(raw.get("probes", []), path),
        gates=_parse_gates(raw.get("gates", []), path),
        routing_rules=_parse_routing(raw.get("routing_rules", []), path),
        scope_semantics=_parse_scope(raw.get("scope_semantics", []), path),
        scope_order=_parse_scope_order(raw.get("scope_order", []), path),
        cap_mapping=_parse_cap_mapping(raw.get("cap_mapping", []), path),
        task_templates=_parse_task_templates(raw.get("task_templates", []), path),
        disable=tuple(disable),
        is_base=is_base,
    )


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------


def _require_list(items: Any, *, section: str, path: Path) -> list:
    if items is None:
        return []
    if not isinstance(items, list):
        raise ProfileLoadError(f"profile {path}: section '{section}' must be a list")
    return items


def _parse_evidence_classes(items: Any, path: Path) -> tuple[EvidenceClassSpec, ...]:
    out: list[EvidenceClassSpec] = []
    for entry in _require_list(items, section="evidence_classes", path=path):
        if isinstance(entry, str):
            out.append(EvidenceClassSpec(id=entry))
        elif isinstance(entry, dict):
            eid = entry.get("id")
            if not isinstance(eid, str) or not eid:
                raise ProfileLoadError(f"profile {path}: evidence_class entry missing 'id'")
            out.append(EvidenceClassSpec(id=eid, desc=entry.get("desc", "")))
        else:
            raise ProfileLoadError(
                f"profile {path}: evidence_class entry must be string or object"
            )
    return tuple(out)


def _parse_node_states(items: Any, path: Path) -> tuple[StateSpec, ...]:
    out: list[StateSpec] = []
    for entry in _require_list(items, section="node_states", path=path):
        if isinstance(entry, str):
            out.append(StateSpec(name=entry))
        elif isinstance(entry, dict):
            name = entry.get("name")
            if not isinstance(name, str) or not name:
                raise ProfileLoadError(f"profile {path}: node_state entry missing 'name'")
            roles_raw = entry.get("roles", [])
            if not isinstance(roles_raw, list) or any(not isinstance(r, str) for r in roles_raw):
                raise ProfileLoadError(
                    f"profile {path}: node_state '{name}' 'roles' must be list of strings"
                )
            out.append(StateSpec(name=name, roles=tuple(roles_raw)))
        else:
            raise ProfileLoadError(f"profile {path}: node_state entry must be string or object")
    return tuple(out)


def _parse_probes(items: Any, path: Path) -> tuple[ProbeSpec, ...]:
    out: list[ProbeSpec] = []
    for entry in _require_list(items, section="probes", path=path):
        if not isinstance(entry, dict):
            raise ProfileLoadError(f"profile {path}: probe entry must be an object")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ProfileLoadError(f"profile {path}: probe entry missing 'name'")
        mechanism = entry.get("mechanism", False)
        if not isinstance(mechanism, bool):
            raise ProfileLoadError(
                f"profile {path}: probe '{name}' 'mechanism' must be a boolean"
            )
        module = entry.get("module")
        if module is not None and not isinstance(module, str):
            raise ProfileLoadError(
                f"profile {path}: probe '{name}' 'module' must be a string or null"
            )
        out.append(
            ProbeSpec(
                name=name,
                mechanism=mechanism,
                module=module,
                inputs=tuple(entry.get("inputs", []) or []),
                outputs=tuple(entry.get("outputs", []) or []),
            )
        )
    return tuple(out)


def _parse_gates(items: Any, path: Path) -> tuple[GateSpec, ...]:
    out: list[GateSpec] = []
    for entry in _require_list(items, section="gates", path=path):
        if not isinstance(entry, dict):
            raise ProfileLoadError(f"profile {path}: gate entry must be an object")
        gid = entry.get("id")
        if not isinstance(gid, str) or not gid:
            raise ProfileLoadError(f"profile {path}: gate entry missing 'id'")
        mechanism = entry.get("mechanism", False)
        if not isinstance(mechanism, bool):
            raise ProfileLoadError(
                f"profile {path}: gate '{gid}' 'mechanism' must be a boolean"
            )
        out.append(
            GateSpec(
                id=gid,
                mechanism=mechanism,
                rule=entry.get("rule", "") or "",
                requires_verdicts=tuple(entry.get("requires_verdicts", []) or []),
            )
        )
    return tuple(out)


def _parse_routing(items: Any, path: Path) -> tuple[RoutingRule, ...]:
    out: list[RoutingRule] = []
    for entry in _require_list(items, section="routing_rules", path=path):
        if not isinstance(entry, dict):
            raise ProfileLoadError(f"profile {path}: routing rule must be an object")
        cause = entry.get("cause")
        actions = entry.get("actions", [])
        if not isinstance(cause, str) or not cause:
            raise ProfileLoadError(f"profile {path}: routing rule missing 'cause'")
        if not isinstance(actions, list):
            raise ProfileLoadError(f"profile {path}: routing rule '{cause}' actions must be list")
        out.append(RoutingRule(cause=cause, actions=tuple(actions)))
    return tuple(out)


def _parse_scope(items: Any, path: Path) -> tuple[ScopeRule, ...]:
    out: list[ScopeRule] = []
    for entry in _require_list(items, section="scope_semantics", path=path):
        if not isinstance(entry, dict):
            raise ProfileLoadError(f"profile {path}: scope rule must be an object")
        when = entry.get("when_state")
        tag = entry.get("tag_scope")
        if not isinstance(when, str) or not when:
            raise ProfileLoadError(f"profile {path}: scope rule missing 'when_state'")
        if not isinstance(tag, str) or not tag:
            raise ProfileLoadError(f"profile {path}: scope rule missing 'tag_scope'")
        out.append(ScopeRule(when_state=when, tag_scope=tag))
    return tuple(out)


def _parse_scope_order(items: Any, path: Path) -> tuple[str, ...]:
    """Ordered scope vocabulary, narrowest → widest (§19.9 vmp #2 / B1).

    Empty or absent = no ordering declared; ScopeBoundaryGate then
    returns ``undetermined`` instead of failing.
    """
    if items is None:
        return ()
    if not isinstance(items, list) or any(not isinstance(v, str) or not v for v in items):
        raise ProfileLoadError(
            f"profile {path}: 'scope_order' must be a non-empty list of strings"
        )
    seen: set[str] = set()
    for v in items:
        if v in seen:
            raise ProfileLoadError(
                f"profile {path}: 'scope_order' contains duplicate '{v}'"
            )
        seen.add(v)
    return tuple(items)


def _parse_task_templates(items: Any, path: Path) -> tuple[TaskTemplateRef, ...]:
    """Profile-recommended reusable task templates (§20.1.2 task ↔ profile).

    JSON shape: ``[{"name": "merge_cross_check", "spec": {...task spec...}}, ...]``.
    The ``spec`` body is stored opaquely; consumers call
    :func:`engine.task.parse_task_spec` to materialise the
    :class:`engine.task.TaskSpec`.
    """
    if items is None:
        return ()
    if not isinstance(items, list):
        raise ProfileLoadError(
            f"profile {path}: 'task_templates' must be a list"
        )
    out: list[TaskTemplateRef] = []
    seen: set[str] = set()
    for i, entry in enumerate(items):
        if not isinstance(entry, dict):
            raise ProfileLoadError(
                f"profile {path}: task_templates[{i}] must be an object"
            )
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ProfileLoadError(
                f"profile {path}: task_templates[{i}] missing 'name' string"
            )
        if name in seen:
            raise ProfileLoadError(
                f"profile {path}: task_templates has duplicate entry '{name}'"
            )
        seen.add(name)
        spec = entry.get("spec")
        if spec is not None and not isinstance(spec, dict):
            raise ProfileLoadError(
                f"profile {path}: task_templates['{name}'].spec must be an "
                f"object or null"
            )
        out.append(TaskTemplateRef(name=name, spec=spec or {}))
    return tuple(out)


def _parse_cap_mapping(items: Any, path: Path) -> tuple[CapMappingEntry, ...]:
    """``constant_provenance`` category → evidence_class id table (§19.9 vmp #4).

    Two accepted shapes:

      * list of objects: ``[{"category": "hardcoded_fixed", "class_id": "A"}, ...]``
      * object form:     ``{"hardcoded_fixed": "A", ...}``

    Both produce a tuple of :class:`CapMappingEntry`. Object form is
    more ergonomic for profile authors; list form preserves declared
    ordering if a profile author wants explicit control.
    """
    out: list[CapMappingEntry] = []
    if items is None:
        return ()
    if isinstance(items, dict):
        for category, class_id in items.items():
            if not isinstance(category, str) or not category:
                raise ProfileLoadError(
                    f"profile {path}: cap_mapping key must be a non-empty string"
                )
            if not isinstance(class_id, str):
                raise ProfileLoadError(
                    f"profile {path}: cap_mapping['{category}'] must be a string "
                    f"(use empty string for 'no cap')"
                )
            out.append(CapMappingEntry(category=category, class_id=class_id))
        return tuple(out)
    if not isinstance(items, list):
        raise ProfileLoadError(
            f"profile {path}: 'cap_mapping' must be a list or object"
        )
    seen: set[str] = set()
    for entry in items:
        if not isinstance(entry, dict):
            raise ProfileLoadError(
                f"profile {path}: cap_mapping entry must be an object"
            )
        category = entry.get("category")
        class_id = entry.get("class_id", entry.get("cap"))
        if not isinstance(category, str) or not category:
            raise ProfileLoadError(
                f"profile {path}: cap_mapping entry missing 'category'"
            )
        if not isinstance(class_id, str):
            raise ProfileLoadError(
                f"profile {path}: cap_mapping['{category}'] missing 'class_id' string"
            )
        if category in seen:
            raise ProfileLoadError(
                f"profile {path}: cap_mapping has duplicate entry for '{category}'"
            )
        seen.add(category)
        out.append(CapMappingEntry(category=category, class_id=class_id))
    return tuple(out)
