"""T6 — Audit log + insert / replace / delete (PLAN §20 v2 §4.3).

Workflow-edit ops are legitimate and supported; the boundaries:

  * ``done_criterion`` on any task is locked — no helper exposes a
    "change criterion" knob, and the post-condition checker
    :func:`assert_done_criterion_unchanged` rejects any tree pair
    where the same id carries different criteria.
  * Every accepted op writes one :class:`TaskAuditEntry` to the log
    (in-memory + JSONL on disk).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.task import (
    CriterionItem,
    NodeRef,
    NodeState,
    TaskAuditError,
    TaskAuditLog,
    TaskAuditOp,
    TaskSpec,
    assemble_task_tree,
    assert_done_criterion_unchanged,
    delete_child,
    insert_child,
    parse_task_spec,
    replace_child,
)


def _tc3_spec() -> TaskSpec:
    return parse_task_spec({
        "id": "root",
        "goal": "g",
        "done_criterion": {
            "kind": "all_of",
            "items": [
                {"kind": "child_done", "child": "A1"},
                {"kind": "child_done", "child": "A2"},
            ],
        },
        "children": [
            {
                "id": "A1",
                "goal": "g",
                "done_criterion": {"kind": "node_closed", "node": "n1"},
                "nodes": [{"id": "n1"}],
                "input_contract": {},
            },
            {
                "id": "A2",
                "goal": "g",
                "done_criterion": {"kind": "node_closed", "node": "n2"},
                "nodes": [{"id": "n2"}],
                "input_contract": {},
            },
        ],
    })


def _new_child(child_id: str = "A3") -> TaskSpec:
    return parse_task_spec({
        "id": child_id,
        "goal": "g",
        "done_criterion": {"kind": "node_closed", "node": "x"},
        "nodes": [{"id": "x"}],
        "input_contract": {},
    })


# ---------------------------------------------------------------------------
# insert_child
# ---------------------------------------------------------------------------


def test_insert_child_adds_to_parent():
    log = TaskAuditLog()
    new_root = insert_child(
        _tc3_spec(),
        "root",
        _new_child("A3"),
        who="agent",
        why="discovered need for end-to-end merge cross-check",
        log=log,
    )
    child_ids = {c.id for c in new_root.children}
    assert child_ids == {"A1", "A2", "A3"}
    # Audit log has one INSERT entry.
    entries = log.read_all()
    assert len(entries) == 1
    assert entries[0].op is TaskAuditOp.INSERT_CHILD
    assert entries[0].target_task_id == "root"
    assert entries[0].detail_id == "A3"
    assert entries[0].who == "agent"


def test_insert_child_persists_to_jsonl(tmp_path: Path):
    audit_path = tmp_path / "audit.jsonl"
    log = TaskAuditLog(path=audit_path)
    insert_child(
        _tc3_spec(), "root", _new_child(),
        who="clark", why="auto-inserted gap-closing task", log=log,
    )
    lines = audit_path.read_text().strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["op"] == "insert_child"
    assert payload["detail_id"] == "A3"
    assert payload["who"] == "clark"


def test_insert_child_unknown_parent_rejected():
    with pytest.raises(TaskAuditError, match="not in tree"):
        insert_child(
            _tc3_spec(), "ghost", _new_child(),
            who="agent", why="—", log=TaskAuditLog(),
        )


def test_insert_child_preserves_original_spec():
    """The frozen-tree invariant: the input spec is untouched."""
    original = _tc3_spec()
    original_child_ids = {c.id for c in original.children}
    insert_child(
        original, "root", _new_child(),
        who="agent", why="—", log=TaskAuditLog(),
    )
    # Original unchanged.
    assert {c.id for c in original.children} == original_child_ids


# ---------------------------------------------------------------------------
# replace_child
# ---------------------------------------------------------------------------


def test_replace_child_swaps_in_place():
    log = TaskAuditLog()
    replacement = parse_task_spec({
        "id": "A1",  # must match the replaced child's id
        "goal": "alternative implementation of A1",
        "done_criterion": {"kind": "node_closed", "node": "n1"},
        "nodes": [{"id": "n1"}],
        "input_contract": {},
    })
    new_root = replace_child(
        _tc3_spec(), "root", "A1", replacement,
        who="agent",
        why="initial implementation went wrong, switching approach",
        log=log,
    )
    a1 = next(c for c in new_root.children if c.id == "A1")
    assert "alternative implementation" in a1.goal
    assert log.read_all()[0].op is TaskAuditOp.REPLACE_CHILD


def test_replace_child_id_must_match_old_id():
    """Renaming via replace is forbidden — the parent's done_criterion
    references the original id, replacing with a new id would create a
    dangling ref the agent could exploit."""
    with pytest.raises(TaskAuditError, match="renaming"):
        replace_child(
            _tc3_spec(), "root", "A1", _new_child("A1_renamed"),
            who="agent", why="—", log=TaskAuditLog(),
        )


def test_replace_child_unknown_child_rejected():
    with pytest.raises(TaskAuditError, match="not under parent"):
        replace_child(
            _tc3_spec(), "root", "ghost", _new_child("ghost"),
            who="agent", why="—", log=TaskAuditLog(),
        )


# ---------------------------------------------------------------------------
# delete_child
# ---------------------------------------------------------------------------


def test_delete_child_removes_from_parent():
    log = TaskAuditLog()
    new_root = delete_child(
        _tc3_spec(), "root", "A2",
        who="agent",
        why="back half no longer needed for this run",
        log=log,
    )
    assert {c.id for c in new_root.children} == {"A1"}
    assert log.read_all()[0].op is TaskAuditOp.DELETE_CHILD


def test_delete_child_does_not_rewrite_parent_criterion():
    """The parent's done_criterion still references A2; after delete,
    evaluating root done will still flag A2 as a gap.  This is
    intentional — done_criterion is the source of truth."""
    new_root = delete_child(
        _tc3_spec(), "root", "A2",
        who="agent", why="—", log=TaskAuditLog(),
    )
    # The criterion text is preserved verbatim.
    assert any(
        item.kind == "child_done" and item.child == "A2"
        for item in new_root.done_criterion.items
    )


# ---------------------------------------------------------------------------
# done_criterion immutability — post-condition check
# ---------------------------------------------------------------------------


def test_done_criterion_unchanged_check_passes_after_normal_insert():
    """insert / replace / delete must preserve every existing task's
    criterion."""
    original = _tc3_spec()
    new_root = insert_child(
        original, "root", _new_child(),
        who="agent", why="—", log=TaskAuditLog(),
    )
    assert_done_criterion_unchanged(original, new_root)


def test_done_criterion_unchanged_check_catches_hostile_edit():
    """Construct a tampered tree by hand to confirm the post-condition
    check fires when the criterion is altered."""
    original = _tc3_spec()
    # Build a parallel tree where root's done_criterion has been
    # rewritten — simulates a hostile composition.
    tampered_root = TaskSpec(
        id=original.id,
        goal=original.goal,
        done_criterion=CriterionItem(kind="all_of", items=()),  # vacuously satisfied
        nodes=original.nodes,
        current_focus=original.current_focus,
        uses_runner=original.uses_runner,
        runner_capabilities=original.runner_capabilities,
        profile=original.profile,
        children=original.children,
        input_contract=original.input_contract,
        description=original.description,
    )
    with pytest.raises(TaskAuditError, match="done_criterion was modified"):
        assert_done_criterion_unchanged(original, tampered_root)


# ---------------------------------------------------------------------------
# Audit log behaviour
# ---------------------------------------------------------------------------


def test_log_records_when_and_extra_fields():
    log = TaskAuditLog()
    insert_child(
        _tc3_spec(), "root", _new_child(),
        who="agent", why="why-text",
        log=log,
        now=1_700_000_000.0,
    )
    entry = log.read_all()[0]
    assert entry.when == 1_700_000_000.0
    assert entry.why == "why-text"


def test_log_in_memory_works_without_path():
    log = TaskAuditLog()
    insert_child(
        _tc3_spec(), "root", _new_child("X"),
        who="x", why="x", log=log,
    )
    delete_child(
        _tc3_spec(), "root", "A1",
        who="y", why="y", log=log,
    )
    assert len(log) == 2
