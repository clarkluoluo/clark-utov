"""T4 — parent / child task tree (PLAN §20 v2 §4.1).

Two essential behaviours:

  * **Parent done depends on child done state** — a parent whose
    ``done_criterion`` names ``child_done(X)`` must wait for X to be
    done; closing every leaf node alone is not enough.
  * **Contract requirement at compose time** — a child named in the
    parent's ``done_criterion`` must carry an ``input_contract``
    (PLAN §20.1.3 invariant #5).  No contract = compose-time error,
    not run-time error.
"""

from __future__ import annotations

import pytest

from engine.task import (
    ContractMismatchError,
    TaskLoadError,
    TaskTree,
    assemble_task_tree,
    parse_task_spec,
)


def _tc3_parent_with_children() -> dict:
    """The tc3 canonical shape with parent ↦ {A1, A2, A3}."""
    return {
        "id": "restore_sign",
        "goal": "restore full sign + end-to-end byte-equal cross-check",
        "uses_runner": "reference_target",
        "runner_capabilities": ["trace"],
        "profile": "vmp_algorithm_extraction",
        "done_criterion": {
            "kind": "all_of",
            "items": [
                {"kind": "child_done", "child": "front_half"},
                {"kind": "child_done", "child": "back_half"},
                {"kind": "child_done", "child": "merge_check"},
            ],
        },
        "children": [
            {
                "id": "front_half",
                "goal": "restore appkey → template",
                "nodes": [{"id": "scratch21"}, {"id": "template"}],
                "done_criterion": {
                    "kind": "all_of",
                    "items": [
                        {"kind": "node_closed", "node": "scratch21"},
                        {"kind": "node_closed", "node": "template"},
                    ],
                },
                "input_contract": {"produces": ["front_half_impl"]},
            },
            {
                "id": "back_half",
                "goal": "restore template → digest → output",
                "nodes": [
                    {"id": "prefix"}, {"id": "SM3"},
                    {"id": "digest"}, {"id": "output"},
                ],
                "done_criterion": {
                    "kind": "all_of",
                    "items": [
                        {"kind": "node_closed", "node": "prefix"},
                        {"kind": "node_closed", "node": "SM3"},
                        {"kind": "node_closed", "node": "digest"},
                        {"kind": "node_closed", "node": "output"},
                    ],
                },
                "input_contract": {"produces": ["back_half_impl"]},
            },
            {
                "id": "merge_check",
                "goal": "end-to-end merge cross-check, byte-equal",
                "done_criterion": {
                    "kind": "named_artefact", "name": "byte_equal_pass",
                },
                "input_contract": {
                    "accepts": ["front_half_impl", "back_half_impl"],
                    "capabilities": ["re_execute"],
                    "produces": ["byte_equal_pass"],
                },
            },
        ],
    }


# ---------------------------------------------------------------------------
# Tree construction + indexing
# ---------------------------------------------------------------------------


def test_assemble_indexes_every_task():
    tree = assemble_task_tree(parse_task_spec(_tc3_parent_with_children()))
    ids = {t.id for t in tree.iter_all()}
    assert ids == {"restore_sign", "front_half", "back_half", "merge_check"}


def test_parent_of_is_correct():
    tree = assemble_task_tree(parse_task_spec(_tc3_parent_with_children()))
    assert tree.parent_of("front_half") == "restore_sign"
    assert tree.parent_of("back_half") == "restore_sign"
    assert tree.parent_of("restore_sign") is None


def test_duplicate_task_id_rejected():
    """Two tasks in the tree with the same id breaks indexing — refuse
    at assembly time. Construct via direct dict so loader-level
    dangling-ref checks don't fire first (the loader sees no
    duplicates yet because the duplicate is a grandchild)."""
    raw = {
        "id": "root",
        "goal": "g",
        "done_criterion": {"kind": "child_done", "child": "a"},
        "children": [
            {
                "id": "a",
                "goal": "g",
                "done_criterion": {"kind": "child_done", "child": "twin"},
                "input_contract": {},
                "children": [
                    {
                        "id": "twin",
                        "goal": "g",
                        "done_criterion": {"kind": "node_closed", "node": "x"},
                        "nodes": [{"id": "x"}],
                        "input_contract": {},
                    },
                ],
            },
            {
                # Sibling at root that shares an id with the grandchild.
                "id": "twin",
                "goal": "g",
                "done_criterion": {"kind": "node_closed", "node": "y"},
                "nodes": [{"id": "y"}],
            },
        ],
    }
    # The root references only 'a' so the loader is happy.  The tree
    # assembler catches the cross-branch duplicate.
    spec = parse_task_spec(raw)
    with pytest.raises(TaskLoadError, match="duplicate task id"):
        assemble_task_tree(spec)


# ---------------------------------------------------------------------------
# Parent done depends on child done (the tc3 fix)
# ---------------------------------------------------------------------------


def test_parent_done_requires_every_child_done():
    """Closing every leaf node of A1 and A2 alone is not enough — the
    parent's done_criterion also names merge_check."""
    tree = assemble_task_tree(parse_task_spec(_tc3_parent_with_children()))
    closed_all_leaves = frozenset({
        "scratch21", "template", "prefix", "SM3", "digest", "output",
    })
    # No artefacts — merge_check task can't be done either.
    assert tree.is_done(
        "front_half", closed_nodes=closed_all_leaves,
    ) is True
    assert tree.is_done(
        "back_half", closed_nodes=closed_all_leaves,
    ) is True
    assert tree.is_done(
        "merge_check", closed_nodes=closed_all_leaves,
    ) is False
    # Parent therefore not done — even though front_half and
    # back_half have closed every leaf.
    assert tree.is_done(
        "restore_sign", closed_nodes=closed_all_leaves,
    ) is False


def test_parent_done_when_every_child_done():
    tree = assemble_task_tree(parse_task_spec(_tc3_parent_with_children()))
    closed_all_leaves = frozenset({
        "scratch21", "template", "prefix", "SM3", "digest", "output",
    })
    # The artefact closes merge_check; both halves are also closed.
    assert tree.is_done(
        "restore_sign",
        closed_nodes=closed_all_leaves,
        present_artefacts=frozenset({"byte_equal_pass"}),
    ) is True


def test_evaluate_root_done_refusal_message_names_pending_child():
    """The end-to-end refusal: agent declares 'restore_sign done'
    while merge_check hasn't run; root gate refuses and names the
    missing child."""
    tree = assemble_task_tree(parse_task_spec(_tc3_parent_with_children()))
    closed_all_leaves = frozenset({
        "scratch21", "template", "prefix", "SM3", "digest", "output",
    })
    result = tree.evaluate_root_done(closed_nodes=closed_all_leaves)
    assert result.passed is False
    assert any("merge_check" in g for g in result.gaps)


# ---------------------------------------------------------------------------
# Contract requirement at compose time
# ---------------------------------------------------------------------------


def test_child_referenced_in_parent_criterion_must_have_contract():
    """Compose-time invariant: a child the parent names as a done
    dependency MUST carry an input_contract."""
    raw = _tc3_parent_with_children()
    # Strip the merge_check contract.
    raw["children"][2]["input_contract"] = None
    spec = parse_task_spec(raw)
    with pytest.raises(ContractMismatchError, match="merge_check"):
        assemble_task_tree(spec)


def test_unreferenced_child_without_contract_is_allowed():
    """A child not named by the parent's done_criterion may omit the
    contract — only referenced children must declare one."""
    raw = _tc3_parent_with_children()
    # Add a fourth child the parent does not reference, no contract.
    raw["children"].append({
        "id": "side_quest",
        "goal": "informational sibling task",
        "done_criterion": {"kind": "node_closed", "node": "q"},
        "nodes": [{"id": "q"}],
    })
    spec = parse_task_spec(raw)
    # Assembly succeeds.
    tree = assemble_task_tree(spec)
    assert tree.get("side_quest") is not None
