"""T2 — done_criterion grammar + evaluator (PLAN §20.1.4 atoms).

Grammar: ``node_closed`` / ``child_done`` / ``named_artefact`` /
``all_of`` / ``any_of``.  Evaluator returns ``(satisfied, gaps)``;
``gaps`` is what the task gate's refusal message quotes back to the
agent so it knows exactly what to do next.
"""

from __future__ import annotations

import pytest

from engine.task import (
    CriterionEvalContext,
    CriterionItem,
    evaluate_done_criterion,
    referenced_artefacts,
    referenced_children,
    referenced_nodes,
)


# ---------------------------------------------------------------------------
# Leaf atoms
# ---------------------------------------------------------------------------


def test_node_closed_pass():
    c = CriterionItem(kind="node_closed", node="n1")
    ctx = CriterionEvalContext(closed_nodes=frozenset({"n1"}))
    r = evaluate_done_criterion(c, ctx)
    assert r.satisfied is True
    assert r.gaps == ()


def test_node_closed_fail_lists_node():
    c = CriterionItem(kind="node_closed", node="n1")
    r = evaluate_done_criterion(c, CriterionEvalContext())
    assert r.satisfied is False
    assert r.gaps == ("node 'n1' not closed",)


def test_child_done_pass():
    c = CriterionItem(kind="child_done", child="sub")
    ctx = CriterionEvalContext(done_children=frozenset({"sub"}))
    assert evaluate_done_criterion(c, ctx).satisfied


def test_child_done_fail_lists_child():
    c = CriterionItem(kind="child_done", child="sub")
    r = evaluate_done_criterion(c, CriterionEvalContext())
    assert r.satisfied is False
    assert "child task 'sub' not done" in r.gaps


def test_named_artefact_pass():
    c = CriterionItem(kind="named_artefact", name="merge_cross_check")
    ctx = CriterionEvalContext(present_artefacts=frozenset({"merge_cross_check"}))
    assert evaluate_done_criterion(c, ctx).satisfied


def test_named_artefact_fail_lists_name():
    c = CriterionItem(kind="named_artefact", name="merge_cross_check")
    r = evaluate_done_criterion(c, CriterionEvalContext())
    assert r.satisfied is False
    assert "artefact 'merge_cross_check' missing" in r.gaps


# ---------------------------------------------------------------------------
# Composites — all_of / any_of
# ---------------------------------------------------------------------------


def test_all_of_pass_when_every_item_satisfied():
    c = CriterionItem(kind="all_of", items=(
        CriterionItem(kind="node_closed", node="a"),
        CriterionItem(kind="node_closed", node="b"),
    ))
    ctx = CriterionEvalContext(closed_nodes=frozenset({"a", "b"}))
    assert evaluate_done_criterion(c, ctx).satisfied is True


def test_all_of_fails_when_any_item_unsatisfied_lists_each_gap():
    """The canonical reference case: two nodes closed but merge missing."""
    c = CriterionItem(kind="all_of", items=(
        CriterionItem(kind="node_closed", node="front_half"),
        CriterionItem(kind="node_closed", node="back_half"),
        CriterionItem(kind="named_artefact", name="merge_cross_check"),
    ))
    ctx = CriterionEvalContext(
        closed_nodes=frozenset({"front_half", "back_half"}),
        # no merge_cross_check
    )
    r = evaluate_done_criterion(c, ctx)
    assert r.satisfied is False
    assert "artefact 'merge_cross_check' missing" in r.gaps
    # The two closed nodes contribute no gap — only the unmet ones list.
    assert not any("front_half" in g for g in r.gaps)
    assert not any("back_half" in g for g in r.gaps)


def test_any_of_pass_when_one_branch_satisfied():
    c = CriterionItem(kind="any_of", items=(
        CriterionItem(kind="node_closed", node="primary"),
        CriterionItem(kind="node_closed", node="fallback"),
    ))
    ctx = CriterionEvalContext(closed_nodes=frozenset({"fallback"}))
    assert evaluate_done_criterion(c, ctx).satisfied is True


def test_any_of_fail_lists_all_branches_with_index():
    c = CriterionItem(kind="any_of", items=(
        CriterionItem(kind="node_closed", node="a"),
        CriterionItem(kind="node_closed", node="b"),
    ))
    r = evaluate_done_criterion(c, CriterionEvalContext())
    assert r.satisfied is False
    assert any("any_of[0]" in g and "a" in g for g in r.gaps)
    assert any("any_of[1]" in g and "b" in g for g in r.gaps)


# ---------------------------------------------------------------------------
# Nested composites — multi-level
# ---------------------------------------------------------------------------


def test_nested_all_of_with_any_of_branch():
    """Real-world-style: '(node a OR node b) AND merge'."""
    c = CriterionItem(kind="all_of", items=(
        CriterionItem(kind="any_of", items=(
            CriterionItem(kind="node_closed", node="a"),
            CriterionItem(kind="node_closed", node="b"),
        )),
        CriterionItem(kind="named_artefact", name="merge"),
    ))
    ctx = CriterionEvalContext(
        closed_nodes=frozenset({"b"}),
        present_artefacts=frozenset({"merge"}),
    )
    assert evaluate_done_criterion(c, ctx).satisfied is True


def test_nested_all_of_with_unsatisfied_any_branch_fails():
    c = CriterionItem(kind="all_of", items=(
        CriterionItem(kind="any_of", items=(
            CriterionItem(kind="node_closed", node="a"),
            CriterionItem(kind="node_closed", node="b"),
        )),
        CriterionItem(kind="named_artefact", name="merge"),
    ))
    ctx = CriterionEvalContext(present_artefacts=frozenset({"merge"}))
    r = evaluate_done_criterion(c, ctx)
    assert r.satisfied is False


# ---------------------------------------------------------------------------
# Reference collectors
# ---------------------------------------------------------------------------


def test_referenced_nodes_walks_tree():
    c = CriterionItem(kind="all_of", items=(
        CriterionItem(kind="node_closed", node="a"),
        CriterionItem(kind="any_of", items=(
            CriterionItem(kind="node_closed", node="b"),
            CriterionItem(kind="child_done", child="sub"),
        )),
    ))
    assert referenced_nodes(c) == frozenset({"a", "b"})


def test_referenced_children_walks_tree():
    c = CriterionItem(kind="all_of", items=(
        CriterionItem(kind="child_done", child="x"),
        CriterionItem(kind="child_done", child="y"),
    ))
    assert referenced_children(c) == frozenset({"x", "y"})


def test_referenced_artefacts_walks_tree():
    c = CriterionItem(kind="any_of", items=(
        CriterionItem(kind="named_artefact", name="alpha"),
        CriterionItem(kind="named_artefact", name="beta"),
    ))
    assert referenced_artefacts(c) == frozenset({"alpha", "beta"})


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_any_of_is_unsatisfiable():
    c = CriterionItem(kind="any_of", items=())
    r = evaluate_done_criterion(c, CriterionEvalContext())
    assert r.satisfied is False
    assert any("unsatisfiable" in g for g in r.gaps)


def test_unknown_kind_fails_gracefully():
    c = CriterionItem(kind="totally_made_up")
    r = evaluate_done_criterion(c, CriterionEvalContext())
    assert r.satisfied is False
    assert any("unknown criterion kind" in g for g in r.gaps)
