"""Task JSON loader (PLAN §20).

Parses one task spec file (or in-memory dict) into a :class:`TaskSpec`.
Missing ``done_criterion`` is the headline failure mode — without an
objective termination check the task has no business existing.

JSON shape (runner-correction applied — PLAN §20.1):

.. code-block:: json

  {
    "id": "restore_sm3_sign",
    "goal": "restore full sign + end-to-end byte-equal cross-check",
    "uses_runner": "reference_target",
    "runner_capabilities": ["trace", "re_execute", "memregion_watch"],
    "profile": "vmp_algorithm_extraction",
    "nodes": [{"id": "scratch21"}, {"id": "template"}],
    "current_focus": "template",
    "done_criterion": {
      "kind": "all_of",
      "items": [
        {"kind": "node_closed", "node": "scratch21"},
        {"kind": "node_closed", "node": "template"},
        {"kind": "named_artefact", "name": "merge_cross_check"}
      ]
    },
    "children": [],
    "input_contract": null,
    "description": "..."
  }

``uses_runner`` may be ``null``/absent — pure-procedure tasks (no
runner) are a regular case under the runner-correction, not a
separate "standalone" class.  ``input_contract`` is required when the
task is referenced by another task's child list / call site
(enforced at compose time by :mod:`engine.task.tree` /
:mod:`engine.task.contract`); the loader itself does not flag a
top-level goal task missing a contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from engine.task.contract import InputContract
from engine.task.done_criterion import (
    VALID_KINDS,
    CriterionItem,
    referenced_children,
    referenced_nodes,
)
from engine.task.implementation_path import (
    ImplementationPath,
    ImplementationPathError,
)
from engine.task.types import NodeRef, NodeState, TaskSpec


class TaskLoadError(Exception):
    """Raised when a task spec is missing required fields or
    self-inconsistent (criterion references unknown node etc.)."""


def load_task_spec(path: str | Path) -> TaskSpec:
    """Read a task spec from disk."""
    p = Path(path)
    if not p.exists():
        raise TaskLoadError(f"task spec file not found: {p}")
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise TaskLoadError(f"task spec {p} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise TaskLoadError(f"task spec {p}: top level must be a JSON object")
    return parse_task_spec(raw, source=str(p))


def parse_task_spec(raw: dict, *, source: str = "<dict>") -> TaskSpec:
    """Parse an in-memory dict into a :class:`TaskSpec`.

    Used by the on-disk loader, profile-bundled templates, and tests.
    """
    if not isinstance(raw, dict):
        raise TaskLoadError(f"task spec {source}: must be a JSON object")

    task_id = raw.get("id")
    if not isinstance(task_id, str) or not task_id:
        raise TaskLoadError(f"task spec {source}: missing string 'id'")

    goal = raw.get("goal")
    if not isinstance(goal, str) or not goal:
        raise TaskLoadError(
            f"task spec '{task_id}' ({source}): missing string 'goal'"
        )

    if "done_criterion" not in raw:
        raise TaskLoadError(
            f"task spec '{task_id}' ({source}): missing 'done_criterion' — a "
            f"task without an objective termination check has no business "
            f"existing (PLAN §20.1.3 invariant #2)"
        )
    done_criterion = _parse_criterion(
        raw["done_criterion"], path=f"{source}::done_criterion"
    )

    nodes = _parse_nodes(raw.get("nodes", []), source=source, task_id=task_id)
    current_focus = raw.get("current_focus")
    if current_focus is not None and not isinstance(current_focus, str):
        raise TaskLoadError(
            f"task spec '{task_id}': 'current_focus' must be a string or null"
        )
    if isinstance(current_focus, str) and not any(
        n.id == current_focus for n in nodes
    ):
        raise TaskLoadError(
            f"task spec '{task_id}': current_focus '{current_focus}' is not a "
            f"declared node id"
        )

    # Runner-correction (PLAN §20.1): runner usage is a uniform
    # declaration; no "bound vs standalone" carve-out.
    uses_runner = raw.get("uses_runner")
    if uses_runner is not None and not isinstance(uses_runner, str):
        raise TaskLoadError(
            f"task spec '{task_id}': 'uses_runner' must be a string or null"
        )
    runner_capabilities_raw = raw.get("runner_capabilities", []) or []
    if not isinstance(runner_capabilities_raw, list) or any(
        not isinstance(c, str) or not c for c in runner_capabilities_raw
    ):
        raise TaskLoadError(
            f"task spec '{task_id}': 'runner_capabilities' must be a list of "
            f"non-empty strings"
        )
    runner_capabilities = tuple(runner_capabilities_raw)
    if runner_capabilities and uses_runner is None:
        raise TaskLoadError(
            f"task spec '{task_id}': declares runner_capabilities but no "
            f"uses_runner — capabilities are addressed to a workbench, not to "
            f"thin air"
        )

    profile = raw.get("profile")
    if profile is not None and (not isinstance(profile, str) or not profile):
        raise TaskLoadError(
            f"task spec '{task_id}': 'profile' must be a non-empty string or null"
        )

    contract_raw = raw.get("input_contract")
    input_contract = (
        InputContract.parse(contract_raw, source=f"{source}::input_contract")
        if contract_raw is not None else None
    )

    # Optional light-to-heavy staged plan. Shape-only validation here; brief
    # quality review (the "三审") is deliberately not run — held for a separate
    # decision (it must stay on form, not task content).
    impl_path_raw = raw.get("implementation_path")
    if impl_path_raw is not None:
        try:
            implementation_path = ImplementationPath.parse(
                impl_path_raw, source=f"{source}::implementation_path"
            )
        except ImplementationPathError as exc:
            raise TaskLoadError(str(exc)) from exc
    else:
        implementation_path = None

    children = tuple(
        parse_task_spec(c, source=f"{source}::children[{i}]")
        for i, c in enumerate(raw.get("children", []) or [])
    )

    # Dangling-reference checks: every node_closed / child_done atom in
    # the criterion must resolve to a declared node / child id.
    node_ids = {n.id for n in nodes}
    child_ids = {c.id for c in children}
    missing_nodes = referenced_nodes(done_criterion) - node_ids
    if missing_nodes:
        raise TaskLoadError(
            f"task spec '{task_id}': done_criterion references undeclared "
            f"node(s): {sorted(missing_nodes)}"
        )
    missing_children = referenced_children(done_criterion) - child_ids
    if missing_children:
        raise TaskLoadError(
            f"task spec '{task_id}': done_criterion references undeclared "
            f"child task(s): {sorted(missing_children)}"
        )

    description = raw.get("description", "")
    if not isinstance(description, str):
        raise TaskLoadError(
            f"task spec '{task_id}': 'description' must be a string"
        )

    return TaskSpec(
        id=task_id,
        goal=goal,
        done_criterion=done_criterion,
        nodes=nodes,
        current_focus=current_focus,
        uses_runner=uses_runner,
        runner_capabilities=runner_capabilities,
        profile=profile,
        children=children,
        input_contract=input_contract,
        implementation_path=implementation_path,
        description=description,
    )


def _parse_criterion(raw: Any, *, path: str) -> CriterionItem:
    if not isinstance(raw, dict):
        raise TaskLoadError(f"{path}: criterion entry must be an object")
    kind = raw.get("kind")
    if not isinstance(kind, str) or kind not in VALID_KINDS:
        raise TaskLoadError(
            f"{path}: criterion 'kind' must be one of {sorted(VALID_KINDS)} "
            f"(got {kind!r})"
        )
    if kind == "node_closed":
        node = raw.get("node")
        if not isinstance(node, str) or not node:
            raise TaskLoadError(
                f"{path}: node_closed requires non-empty 'node' string"
            )
        return CriterionItem(kind=kind, node=node)
    if kind == "child_done":
        child = raw.get("child")
        if not isinstance(child, str) or not child:
            raise TaskLoadError(
                f"{path}: child_done requires non-empty 'child' string"
            )
        return CriterionItem(kind=kind, child=child)
    if kind == "named_artefact":
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            raise TaskLoadError(
                f"{path}: named_artefact requires non-empty 'name' string"
            )
        return CriterionItem(kind=kind, name=name)
    # all_of / any_of
    items_raw = raw.get("items")
    if not isinstance(items_raw, list) or not items_raw:
        raise TaskLoadError(
            f"{path}: {kind} requires a non-empty 'items' list"
        )
    items = tuple(
        _parse_criterion(item, path=f"{path}::items[{i}]")
        for i, item in enumerate(items_raw)
    )
    return CriterionItem(kind=kind, items=items)


def _parse_nodes(raw: Any, *, source: str, task_id: str) -> tuple[NodeRef, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise TaskLoadError(
            f"task spec '{task_id}' ({source}): 'nodes' must be a list"
        )
    out: list[NodeRef] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw):
        if isinstance(entry, str):
            node_id = entry
            state = NodeState.OPEN
            description = ""
        elif isinstance(entry, dict):
            node_id = entry.get("id")
            if not isinstance(node_id, str) or not node_id:
                raise TaskLoadError(
                    f"task spec '{task_id}': nodes[{i}] missing string 'id'"
                )
            state_raw = entry.get("state", "open")
            if not isinstance(state_raw, str):
                raise TaskLoadError(
                    f"task spec '{task_id}': nodes[{i}] 'state' must be a string"
                )
            try:
                state = NodeState(state_raw)
            except ValueError as exc:
                raise TaskLoadError(
                    f"task spec '{task_id}': nodes[{i}] 'state' must be one of "
                    f"{[s.value for s in NodeState]}"
                ) from exc
            description = entry.get("description", "") or ""
            if not isinstance(description, str):
                raise TaskLoadError(
                    f"task spec '{task_id}': nodes[{i}] 'description' must be a string"
                )
        else:
            raise TaskLoadError(
                f"task spec '{task_id}': nodes[{i}] must be a string id or object"
            )
        if node_id in seen:
            raise TaskLoadError(
                f"task spec '{task_id}': duplicate node id '{node_id}'"
            )
        seen.add(node_id)
        out.append(NodeRef(id=node_id, state=state, description=description))
    return tuple(out)
