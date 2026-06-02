"""Task audit log + structural mutations (PLAN §20 v2 §4.3).

The v2 §4.3 boundary: **inserts / replaces / deletes** of children are
legitimate runtime workflow adjustments and must be supported, but
two things stay locked:

  1. ``done_criterion`` on any task in the tree may NOT be altered by
     a runtime op.  Altering it = agent redefining "done" = violates
     objective-task-termination (PLAN §20.1.3 invariant #2).
  2. The parent-task references to child-task done relationships may
     not be rewritten *by the children themselves* (no reverse-write
     — invariant #1's spirit applied to structure).

Every accepted op is logged to a JSONL audit trail with ``who``,
``what``, ``why``, ``when`` so the workflow change history is
reviewable at the same tier as finding-mutation audits (PLAN §20.1.3
invariant #4).

The mutations return a NEW :class:`TaskSpec` tree — the input spec is
frozen and never modified.  Callers wrap the new tree in a new
:class:`engine.task.tree.TaskTree` via :func:`assemble_task_tree`.
"""

from __future__ import annotations

import enum
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from engine.task.contract import ContractMismatchError
from engine.task.done_criterion import CriterionItem
from engine.task.types import TaskSpec


class TaskAuditError(Exception):
    """Raised when a mutation violates the v2 §4.3 boundary
    (done_criterion alteration, mechanism-floor bypass, etc.) or
    references a task id not in the tree."""


class TaskAuditOp(enum.Enum):
    """Every mutation the audit log records."""

    INSERT_CHILD   = "insert_child"
    REPLACE_CHILD  = "replace_child"
    DELETE_CHILD   = "delete_child"
    CREATE_TASK    = "create_task"
    ARCHIVE_TASK   = "archive_task"


@dataclass(frozen=True)
class TaskAuditEntry:
    """One row in the audit log."""

    op: TaskAuditOp
    target_task_id: str               # the task being mutated or its parent
    detail_id: str                    # child id, replacement id, …
    who: str                          # caller identity (agent / user / clark)
    why: str                          # natural-language motivation
    when: float                       # unix timestamp
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op":             self.op.value,
            "target_task_id": self.target_task_id,
            "detail_id":      self.detail_id,
            "who":            self.who,
            "why":            self.why,
            "when":           self.when,
            "extra":          dict(self.extra),
        }


class TaskAuditLog:
    """Append-only JSONL audit log for task ops.

    ``path = None`` keeps everything in memory (useful in tests and
    in-process auditing); construction with a path appends to the
    file at every entry.  The in-memory list is the source of truth
    for :meth:`read_all`.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._entries: list[TaskAuditEntry] = []

    def append(self, entry: TaskAuditEntry) -> TaskAuditEntry:
        self._entries.append(entry)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        return entry

    def read_all(self) -> list[TaskAuditEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Structural mutations — every one logs an audit entry.
# ---------------------------------------------------------------------------


def insert_child(
    root: TaskSpec,
    parent_id: str,
    new_child: TaskSpec,
    *,
    who: str,
    why: str,
    log: TaskAuditLog,
    now: Optional[float] = None,
) -> TaskSpec:
    """Insert ``new_child`` under the task identified by ``parent_id``.

    Returns a NEW root spec with the insertion applied.  Logs an
    :attr:`TaskAuditOp.INSERT_CHILD` entry.
    """
    _check_new_child(new_child)
    if not _has_task(root, parent_id):
        raise TaskAuditError(
            f"insert_child: parent '{parent_id}' not in tree rooted at "
            f"'{root.id}'"
        )
    new_root = _map_tree(
        root,
        on_target=parent_id,
        transform=lambda parent: _spec_with_children(
            parent, parent.children + (new_child,)
        ),
    )
    log.append(TaskAuditEntry(
        op=TaskAuditOp.INSERT_CHILD,
        target_task_id=parent_id,
        detail_id=new_child.id,
        who=who,
        why=why,
        when=now if now is not None else time.time(),
    ))
    return new_root


def replace_child(
    root: TaskSpec,
    parent_id: str,
    child_id: str,
    new_child: TaskSpec,
    *,
    who: str,
    why: str,
    log: TaskAuditLog,
    now: Optional[float] = None,
) -> TaskSpec:
    """Replace ``child_id`` under ``parent_id`` with ``new_child``.

    The replacement child MUST carry the same id as the one being
    replaced — replacing 'A' with 'B' is a delete-then-insert, not a
    replace.  This invariant keeps the parent's done_criterion
    references intact (which the agent cannot rewrite).

    Note the replaced child's findings are archived (caller's
    responsibility; the audit entry records the archival).
    """
    if new_child.id != child_id:
        raise TaskAuditError(
            f"replace_child: replacement id '{new_child.id}' must equal "
            f"replaced id '{child_id}' — renaming via replace is forbidden "
            f"(use delete+insert)"
        )
    _check_new_child(new_child)
    if not _has_task(root, parent_id):
        raise TaskAuditError(
            f"replace_child: parent '{parent_id}' not in tree"
        )
    parent_spec = _find_task(root, parent_id)
    if not any(c.id == child_id for c in parent_spec.children):
        raise TaskAuditError(
            f"replace_child: child '{child_id}' not under parent '{parent_id}'"
        )

    def swap(parent: TaskSpec) -> TaskSpec:
        return _spec_with_children(parent, tuple(
            new_child if c.id == child_id else c for c in parent.children
        ))

    new_root = _map_tree(root, on_target=parent_id, transform=swap)
    log.append(TaskAuditEntry(
        op=TaskAuditOp.REPLACE_CHILD,
        target_task_id=parent_id,
        detail_id=child_id,
        who=who,
        why=why,
        when=now if now is not None else time.time(),
        extra={"new_spec_id": new_child.id},
    ))
    return new_root


def delete_child(
    root: TaskSpec,
    parent_id: str,
    child_id: str,
    *,
    who: str,
    why: str,
    log: TaskAuditLog,
    now: Optional[float] = None,
) -> TaskSpec:
    """Remove ``child_id`` from ``parent_id``'s children.

    The parent's ``done_criterion`` references to the deleted child
    are NOT rewritten — they remain in the spec, and the next
    ``evaluate_root_done`` will list them as unmet gaps.  This is
    intentional: the criterion is the source of truth; deleting a
    child while leaving its reference in the criterion is the agent's
    explicit choice and the gate will keep flagging it.
    """
    if not _has_task(root, parent_id):
        raise TaskAuditError(
            f"delete_child: parent '{parent_id}' not in tree"
        )
    parent_spec = _find_task(root, parent_id)
    if not any(c.id == child_id for c in parent_spec.children):
        raise TaskAuditError(
            f"delete_child: child '{child_id}' not under parent '{parent_id}'"
        )

    def drop(parent: TaskSpec) -> TaskSpec:
        return _spec_with_children(
            parent,
            tuple(c for c in parent.children if c.id != child_id),
        )

    new_root = _map_tree(root, on_target=parent_id, transform=drop)
    log.append(TaskAuditEntry(
        op=TaskAuditOp.DELETE_CHILD,
        target_task_id=parent_id,
        detail_id=child_id,
        who=who,
        why=why,
        when=now if now is not None else time.time(),
    ))
    return new_root


# ---------------------------------------------------------------------------
# Forbidden mutations — done_criterion immutability + structural integrity.
# ---------------------------------------------------------------------------


def _check_new_child(new_child: TaskSpec) -> None:
    """Sanity check on a freshly-supplied child.

    The v2 §4.3 lock against ``done_criterion`` rewrite applies
    *to existing tasks*: an inserted child carries whatever
    done_criterion the spec declares, and that criterion is then
    immutable.  No new check is needed here — the lock fires when
    callers try to mutate an EXISTING task's criterion via these
    helpers (none of them expose a "change criterion" knob; that's
    the structural defence).
    """
    # Defensive: a TaskSpec without a done_criterion can't be
    # constructed via the loader, but a hand-built spec might be —
    # refuse anything that would let an inserted task escape its
    # objective end-point.
    if new_child.done_criterion is None:  # type: ignore[truthy-bool]
        raise TaskAuditError(
            f"insert/replace: new task '{new_child.id}' lacks "
            f"done_criterion — refused (PLAN §20.1.3 invariant #2)"
        )


def assert_done_criterion_unchanged(
    before: TaskSpec, after: TaskSpec,
) -> None:
    """Walk both trees and refuse if any same-id pair has a different
    ``done_criterion``.

    Callers use this as a post-condition check after a structural op
    — the structural ops themselves never touch existing criteria,
    but a hostile caller composing multiple ops in one pass could
    smuggle a criterion edit through.  This is the post-condition
    defence (PLAN §20.1.3 invariant #2).
    """
    before_index: dict[str, CriterionItem] = {}
    _collect_criteria(before, before_index)
    after_index: dict[str, CriterionItem] = {}
    _collect_criteria(after, after_index)
    for task_id, criterion in before_index.items():
        if task_id not in after_index:
            # The task was archived / removed — that's fine; the
            # criterion just disappears with it.
            continue
        if after_index[task_id] != criterion:
            raise TaskAuditError(
                f"task '{task_id}': done_criterion was modified during a "
                f"structural op — forbidden (PLAN §20.1.3 invariant #2)"
            )


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------


def _has_task(root: TaskSpec, task_id: str) -> bool:
    if root.id == task_id:
        return True
    return any(_has_task(c, task_id) for c in root.children)


def _find_task(root: TaskSpec, task_id: str) -> TaskSpec:
    if root.id == task_id:
        return root
    for c in root.children:
        if _has_task(c, task_id):
            return _find_task(c, task_id)
    raise TaskAuditError(f"_find_task: '{task_id}' not in tree")


def _map_tree(
    spec: TaskSpec,
    *,
    on_target: str,
    transform,
) -> TaskSpec:
    """Build a new tree where the task with id ``on_target`` is
    replaced by ``transform(task)``; every other task is recursively
    rebuilt with its (possibly transformed) children."""
    if spec.id == on_target:
        return transform(spec)
    new_children = tuple(
        _map_tree(c, on_target=on_target, transform=transform)
        for c in spec.children
    )
    return _spec_with_children(spec, new_children)


def _spec_with_children(
    spec: TaskSpec, children: tuple[TaskSpec, ...],
) -> TaskSpec:
    """Return a copy of ``spec`` with the given children list.  All
    other fields preserved verbatim — done_criterion in particular
    is **not** touched."""
    return TaskSpec(
        id=spec.id,
        goal=spec.goal,
        done_criterion=spec.done_criterion,
        nodes=spec.nodes,
        current_focus=spec.current_focus,
        uses_runner=spec.uses_runner,
        runner_capabilities=spec.runner_capabilities,
        profile=spec.profile,
        children=children,
        input_contract=spec.input_contract,
        implementation_path=spec.implementation_path,
        description=spec.description,
    )


def _collect_criteria(
    spec: TaskSpec, out: dict[str, CriterionItem],
) -> None:
    out[spec.id] = spec.done_criterion
    for c in spec.children:
        _collect_criteria(c, out)


__all__ = [
    "TaskAuditEntry",
    "TaskAuditError",
    "TaskAuditLog",
    "TaskAuditOp",
    "assert_done_criterion_unchanged",
    "delete_child",
    "insert_child",
    "replace_child",
]
