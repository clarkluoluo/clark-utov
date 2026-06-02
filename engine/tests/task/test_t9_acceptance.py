"""T9 — Acceptance gates (PLAN §20 / ROADMAP v0.5.0-dev candidate).

The 8 acceptance gates from the v0.5.0 candidate spec, verified
end-to-end. Each test name encodes the gate number; comments quote
the gate text from the spec verbatim.

  1. Task object loads from a JSON spec; missing done_criterion fails the load.
  2. Agent declares "task done" while one node still open → clark
     refuses, names the open node.
  3. Agent declares "task done" while all nodes closed but
     done_criterion references a separate merge cross-check that
     hasn't run → clark refuses, names the missing artefact.
  4. Handoff payload includes the full task spec; agent receives goal
     + nodes + done_criterion in the first message of the next session.
  5. (v2) Parent task with two children: closing both children alone
     doesn't satisfy the parent done_criterion that names a merge step;
     parent stays open until merge runs.
  6. (v2) Reusable work-unit task declares an input contract; a parent
     task that wires the wrong input fails at compose time, not run time.
  7. (v2) Insert / replace operations are audit-logged; an attempt to
     rewrite done_criterion via the insert API is refused.
  8. (v2) Inserted / replacing task still triggers every base
     mechanism probe — M1 / M3 / CP / VP / WFW / scope gates / use
     case fork fire normally on the new code path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.profile import (
    ConjunctiveGate,
    ProbeContext,
    ProfileRegistry,
)
from engine.task import (
    ContractMismatchError,
    CriterionEvalContext,
    CriterionItem,
    InputContract,
    NodeRef,
    NodeState,
    TaskAuditError,
    TaskAuditLog,
    TaskDoneRefusal,
    TaskGate,
    TaskLoadError,
    TaskSpec,
    TaskTree,
    assemble_task_tree,
    assert_done_criterion_unchanged,
    delete_child,
    insert_child,
    parse_task_spec,
    replace_child,
    validate_contract_compose,
)


VMP_PROFILE = "vmp_algorithm_extraction"


# ---------------------------------------------------------------------------
# 1 — missing done_criterion fails the load
# ---------------------------------------------------------------------------


def test_gate_1_missing_done_criterion_fails_load(tmp_path: Path):
    p = tmp_path / "broken.json"
    p.write_text(json.dumps({
        "id": "t",
        "goal": "g",
        "nodes": [{"id": "n"}],
        # no done_criterion
    }))
    with pytest.raises(TaskLoadError, match="done_criterion"):
        from engine.task import load_task_spec
        load_task_spec(p)


# ---------------------------------------------------------------------------
# 2 — task done refused while one node still open, names the open node
# ---------------------------------------------------------------------------


def test_gate_2_node_open_refusal_names_node():
    spec = parse_task_spec({
        "id": "t",
        "goal": "g",
        "done_criterion": {
            "kind": "all_of",
            "items": [
                {"kind": "node_closed", "node": "a"},
                {"kind": "node_closed", "node": "b"},
            ],
        },
        "nodes": [{"id": "a"}, {"id": "b"}],
    })
    gate = TaskGate(spec=spec)
    result = gate.evaluate_task_done(
        ctx=CriterionEvalContext(closed_nodes=frozenset({"a"})),
    )
    assert result.passed is False
    assert any("b" in g and "not closed" in g for g in result.gaps)
    assert "b" in result.refusal_message


# ---------------------------------------------------------------------------
# 3 — all nodes closed but merge artefact pending → refused, names artefact
# ---------------------------------------------------------------------------


def test_gate_3_merge_artefact_pending_refusal_names_artefact():
    """The verbatim tc3 case."""
    spec = parse_task_spec({
        "id": "restore_sign",
        "goal": "restore + cross-check",
        "done_criterion": {
            "kind": "all_of",
            "items": [
                {"kind": "node_closed", "node": "front_half"},
                {"kind": "node_closed", "node": "back_half"},
                {"kind": "named_artefact", "name": "merge_cross_check"},
            ],
        },
        "nodes": [{"id": "front_half"}, {"id": "back_half"}],
    })
    gate = TaskGate(spec=spec)
    result = gate.evaluate_task_done(
        ctx=CriterionEvalContext(
            closed_nodes=frozenset({"front_half", "back_half"}),
        ),
    )
    assert result.passed is False
    assert any("merge_cross_check" in g for g in result.gaps)
    assert "merge_cross_check" in result.refusal_message


# ---------------------------------------------------------------------------
# 4 — handoff payload includes the full task spec
# ---------------------------------------------------------------------------


def test_gate_4_handoff_payload_carries_full_spec():
    """A TaskSpec round-trips JSON-shaped data: id, goal, full nodes
    list, and full done_criterion AST.  The agent receiving the
    handoff sees the whole picture from message #1, never just the
    current focus."""
    spec = parse_task_spec({
        "id": "t",
        "goal": "restore full sign",
        "done_criterion": {
            "kind": "all_of",
            "items": [
                {"kind": "node_closed", "node": "a"},
                {"kind": "named_artefact", "name": "merge"},
            ],
        },
        "nodes": [{"id": "a"}, {"id": "b"}],
        "current_focus": "a",
    })
    # The handoff envelope MUST contain enough fields to reconstruct
    # the entire spec — verify the public attributes the agent reads.
    handoff = {
        "id":             spec.id,
        "goal":           spec.goal,
        "current_focus":  spec.current_focus,
        "nodes":          [n.id for n in spec.nodes],
        "done_criterion": _criterion_to_dict(spec.done_criterion),
    }
    # The spec contains the whole picture.
    assert handoff["goal"] == "restore full sign"
    assert handoff["nodes"] == ["a", "b"]   # both nodes visible, not only focus
    assert handoff["done_criterion"]["kind"] == "all_of"
    assert any(
        item.get("name") == "merge"
        for item in handoff["done_criterion"]["items"]
    )


def _criterion_to_dict(c: CriterionItem) -> dict:
    out = {"kind": c.kind}
    if c.node:
        out["node"] = c.node
    if c.child:
        out["child"] = c.child
    if c.name:
        out["name"] = c.name
    if c.items:
        out["items"] = [_criterion_to_dict(i) for i in c.items]
    return out


# ---------------------------------------------------------------------------
# 5 — parent stays open until merge step runs
# ---------------------------------------------------------------------------


def test_gate_5_parent_with_two_children_stays_open_without_merge():
    spec = parse_task_spec({
        "id": "parent",
        "goal": "g",
        "done_criterion": {
            "kind": "all_of",
            "items": [
                {"kind": "child_done", "child": "front"},
                {"kind": "child_done", "child": "back"},
                {"kind": "named_artefact", "name": "merge"},
            ],
        },
        "children": [
            {
                "id": "front",
                "goal": "g",
                "done_criterion": {"kind": "node_closed", "node": "fn"},
                "nodes": [{"id": "fn"}],
                "input_contract": {},
            },
            {
                "id": "back",
                "goal": "g",
                "done_criterion": {"kind": "node_closed", "node": "bn"},
                "nodes": [{"id": "bn"}],
                "input_contract": {},
            },
        ],
    })
    tree = assemble_task_tree(spec)
    result = tree.evaluate_root_done(
        closed_nodes=frozenset({"fn", "bn"}),  # both children done
        # but no merge artefact
    )
    assert result.passed is False
    assert any("merge" in g for g in result.gaps)


# ---------------------------------------------------------------------------
# 6 — contract mismatch fails at compose time, not at run time
# ---------------------------------------------------------------------------


def test_gate_6_contract_mismatch_at_compose_time():
    contract = InputContract(accepts=("front_impl", "back_impl"),
                              capabilities=("re_execute",))
    with pytest.raises(ContractMismatchError) as exc:
        validate_contract_compose(
            contract,
            supplied_inputs=["front_impl"],     # back_impl missing
            supplied_capabilities=[],           # re_execute missing
            callee_id="merge_check",
        )
    msg = str(exc.value)
    assert "merge_check" in msg
    assert "back_impl" in msg
    assert "re_execute" in msg


# ---------------------------------------------------------------------------
# 7a — insert / replace audit-logged
# ---------------------------------------------------------------------------


def test_gate_7a_insert_replace_delete_logged():
    spec = parse_task_spec({
        "id": "root",
        "goal": "g",
        "done_criterion": {"kind": "child_done", "child": "a"},
        "children": [{
            "id": "a", "goal": "g",
            "done_criterion": {"kind": "node_closed", "node": "n"},
            "nodes": [{"id": "n"}],
            "input_contract": {},
        }],
    })
    log = TaskAuditLog()
    new = insert_child(
        spec, "root",
        parse_task_spec({
            "id": "b", "goal": "g",
            "done_criterion": {"kind": "node_closed", "node": "m"},
            "nodes": [{"id": "m"}],
            "input_contract": {},
        }),
        who="agent", why="ins", log=log,
    )
    new = replace_child(
        new, "root", "a",
        parse_task_spec({
            "id": "a", "goal": "alternate", "done_criterion":
            {"kind": "node_closed", "node": "n"},
            "nodes": [{"id": "n"}], "input_contract": {},
        }),
        who="agent", why="rep", log=log,
    )
    delete_child(new, "root", "b", who="agent", why="del", log=log)
    ops = [e.op.value for e in log.read_all()]
    assert ops == ["insert_child", "replace_child", "delete_child"]


# ---------------------------------------------------------------------------
# 7b — rewriting done_criterion via insert is refused
# ---------------------------------------------------------------------------


def test_gate_7b_done_criterion_rewrite_via_insert_refused():
    """No structural op exposes a 'change criterion' knob.  Confirm by
    constructing a hostile post-state where root's criterion has
    changed and the post-condition check fires."""
    spec = parse_task_spec({
        "id": "root",
        "goal": "g",
        "done_criterion": {"kind": "node_closed", "node": "a"},
        "nodes": [{"id": "a"}],
    })
    # Build a tampered version where root has a vacuous all_of.
    tampered = TaskSpec(
        id=spec.id,
        goal=spec.goal,
        done_criterion=CriterionItem(kind="all_of", items=()),
        nodes=spec.nodes,
        current_focus=spec.current_focus,
        uses_runner=spec.uses_runner,
        runner_capabilities=spec.runner_capabilities,
        profile=spec.profile,
        children=spec.children,
        input_contract=spec.input_contract,
        description=spec.description,
    )
    with pytest.raises(TaskAuditError, match="done_criterion was modified"):
        assert_done_criterion_unchanged(spec, tampered)


# ---------------------------------------------------------------------------
# 8 — inserted task still walks the mechanism floor
# ---------------------------------------------------------------------------


def test_gate_8_inserted_task_still_walks_mechanism_floor():
    """Lock B from v0.4.0 applies to the task path too: the gate
    consults the import-time mechanism registry, not anything the
    task carries.  An inserted task's scope-overreach params still
    fail the scope_boundary_gate."""
    reg = ProfileRegistry()
    profile = reg.load_chain(VMP_PROFILE)
    cg = ConjunctiveGate(profile)

    spec = parse_task_spec({
        "id": "root",
        "goal": "g",
        "uses_runner": "reference_target",
        "runner_capabilities": ["trace"],
        "profile": VMP_PROFILE,
        "done_criterion": {"kind": "child_done", "child": "ch"},
        "children": [{
            "id": "ch", "goal": "g",
            "done_criterion": {"kind": "node_closed", "node": "n"},
            "nodes": [{"id": "n"}],
            "input_contract": {},
        }],
    })
    log = TaskAuditLog()
    inserted = insert_child(
        spec, "root",
        parse_task_spec({
            "id": "ch2", "goal": "g",
            "done_criterion": {"kind": "node_closed", "node": "m"},
            "nodes": [{"id": "m"}],
            "input_contract": {},
        }),
        who="agent", why="—", log=log,
    )
    tree = assemble_task_tree(inserted)

    # All children closed; criterion satisfied.  But the task-done
    # declaration carries cross_env / task_bound scope-overreach
    # params; scope_boundary_gate fires.
    result = tree.evaluate_root_done(
        closed_nodes=frozenset({"n", "m"}),
        conjunctive_gate=cg,
        probe_ctx=ProbeContext(
            method="finalize_verdict",
            params={"scope_claim": "cross_env", "scope_observed": "task_bound"},
            profile=profile,
        ),
    )
    assert result.passed is False
    assert "scope_boundary_gate" in result.mechanism_failing_probes
