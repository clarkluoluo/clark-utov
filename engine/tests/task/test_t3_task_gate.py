"""T3 — TaskGate (PLAN §20.1.2 task ↔ gate, two-layer M1).

The reference case pothole: agent reads node M1 pass as task M1 pass and
declares "task done" while a merge cross-check is still pending.
TaskGate is the anchor that refuses such a declaration and names the
gap.
"""

from __future__ import annotations

import pytest

from engine.task import (
    CriterionEvalContext,
    CriterionItem,
    NodeRef,
    NodeState,
    TaskDoneRefusal,
    TaskGate,
    TaskSpec,
    parse_task_spec,
)


# ---------------------------------------------------------------------------
# Fixtures — the canonical tc3 task shape
# ---------------------------------------------------------------------------


def _tc3_parent_task() -> TaskSpec:
    """Parent task naming front-half + back-half + merge."""
    return parse_task_spec({
        "id": "restore_sign",
        "goal": "restore full sign + end-to-end byte-equal cross-check",
        "uses_runner": "reference_target",
        "runner_capabilities": ["trace", "re_execute"],
        "profile": "vmp_algorithm_extraction",
        "nodes": [
            {"id": "front_half"},
            {"id": "back_half"},
        ],
        "done_criterion": {
            "kind": "all_of",
            "items": [
                {"kind": "node_closed", "node": "front_half"},
                {"kind": "node_closed", "node": "back_half"},
                {"kind": "named_artefact", "name": "merge_cross_check"},
            ],
        },
    })


# ---------------------------------------------------------------------------
# Refusal: agent declares done without merge
# ---------------------------------------------------------------------------


def test_task_done_refused_when_merge_artefact_missing():
    """The tc3 case verbatim: both nodes closed but merge cross-check
    artefact not in the ledger → task gate refuses."""
    gate = TaskGate(spec=_tc3_parent_task())
    ctx = CriterionEvalContext(
        closed_nodes=frozenset({"front_half", "back_half"}),
        # no merge_cross_check in present_artefacts
    )
    result = gate.evaluate_task_done(ctx=ctx)
    assert result.passed is False
    assert "merge_cross_check" in result.refusal_message
    # The closed nodes do not appear as gaps — only the missing
    # artefact does.
    assert any("merge_cross_check" in g for g in result.gaps)


def test_task_done_refused_when_node_still_open():
    gate = TaskGate(spec=_tc3_parent_task())
    ctx = CriterionEvalContext(
        closed_nodes=frozenset({"front_half"}),  # back_half still open
        present_artefacts=frozenset({"merge_cross_check"}),
    )
    result = gate.evaluate_task_done(ctx=ctx)
    assert result.passed is False
    assert any("back_half" in g and "not closed" in g for g in result.gaps)


def test_task_done_accepted_when_everything_satisfied():
    gate = TaskGate(spec=_tc3_parent_task())
    ctx = CriterionEvalContext(
        closed_nodes=frozenset({"front_half", "back_half"}),
        present_artefacts=frozenset({"merge_cross_check"}),
    )
    result = gate.evaluate_task_done(ctx=ctx)
    assert result.passed is True
    assert result.refusal_message == ""


# ---------------------------------------------------------------------------
# Refusal message format
# ---------------------------------------------------------------------------


def test_refusal_message_lists_each_gap_separately():
    """Refusal message must enumerate gaps so the agent can act on
    each — handing back a single 'criterion not satisfied' line was
    exactly the failure mode the task gate fixes."""
    gate = TaskGate(spec=_tc3_parent_task())
    ctx = CriterionEvalContext()  # nothing closed, no artefacts
    result = gate.evaluate_task_done(ctx=ctx)
    msg = result.refusal_message
    assert "front_half" in msg
    assert "back_half" in msg
    assert "merge_cross_check" in msg
    assert "[TASK-GATE/REFUSE]" in msg
    assert "restore_sign" in msg


def test_assert_task_done_raises_on_refusal():
    gate = TaskGate(spec=_tc3_parent_task())
    with pytest.raises(TaskDoneRefusal, match="merge_cross_check"):
        gate.assert_task_done(ctx=CriterionEvalContext(
            closed_nodes=frozenset({"front_half", "back_half"}),
        ))


def test_assert_task_done_returns_result_on_pass():
    gate = TaskGate(spec=_tc3_parent_task())
    result = gate.assert_task_done(ctx=CriterionEvalContext(
        closed_nodes=frozenset({"front_half", "back_half"}),
        present_artefacts=frozenset({"merge_cross_check"}),
    ))
    assert result.passed is True


# ---------------------------------------------------------------------------
# ctx defaulting: derive from spec's own node states
# ---------------------------------------------------------------------------


def test_default_ctx_derives_closed_nodes_from_spec():
    """When the caller doesn't supply a ctx, the gate uses the spec's
    own node states.  Useful for spec-stamped closure attestations and
    for unit tests."""
    spec = TaskSpec(
        id="t",
        goal="g",
        done_criterion=CriterionItem(kind="node_closed", node="n"),
        nodes=(NodeRef(id="n", state=NodeState.CLOSED),),
    )
    gate = TaskGate(spec=spec)
    result = gate.evaluate_task_done()  # no ctx
    assert result.passed is True


def test_default_ctx_treats_artefacts_as_absent():
    """No external ctx means the conservative answer: no artefacts
    present yet, no children done yet."""
    spec = TaskSpec(
        id="t",
        goal="g",
        done_criterion=CriterionItem(kind="named_artefact", name="x"),
    )
    gate = TaskGate(spec=spec)
    result = gate.evaluate_task_done()
    assert result.passed is False
    assert any("artefact 'x'" in g for g in result.gaps)


# ---------------------------------------------------------------------------
# Two-layer separation: node gate ≠ task gate
# ---------------------------------------------------------------------------


def test_all_nodes_closed_alone_does_not_satisfy_task_with_artefact():
    """The whole point of the two-layer split: the agent declaring
    'all nodes closed' is necessary but not sufficient for task done
    when the criterion names extra artefacts (like a merge check)."""
    spec = TaskSpec(
        id="parent",
        goal="restore",
        done_criterion=CriterionItem(kind="all_of", items=(
            CriterionItem(kind="node_closed", node="a"),
            CriterionItem(kind="named_artefact", name="merge"),
        )),
        nodes=(
            NodeRef(id="a", state=NodeState.CLOSED),
        ),
    )
    gate = TaskGate(spec=spec)
    # ctx supplies the closed nodes but NOT the merge artefact
    result = gate.evaluate_task_done(ctx=CriterionEvalContext(
        closed_nodes=frozenset({"a"}),
    ))
    assert result.passed is False
    assert any("merge" in g for g in result.gaps)
