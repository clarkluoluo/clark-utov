"""T1 — TaskSpec / NodeRef / loader (PLAN §20).

Runner-correction acceptance (PLAN §20.1, 2026-05-29):

  * No "bound vs standalone" carve-out: a task with ``uses_runner=null``
    parses cleanly; runner_capabilities without uses_runner is rejected.
  * input_contract is no longer mandatory for runner-less tasks; it is
    a general reusability declaration.

Loader headline failure: missing ``done_criterion`` always fails.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.task import (
    CriterionItem,
    NodeRef,
    NodeState,
    TaskLoadError,
    TaskSpec,
    load_task_spec,
    parse_task_spec,
)


# ---------------------------------------------------------------------------
# Mandatory fields
# ---------------------------------------------------------------------------


def _minimal_spec(**overrides) -> dict:
    base = {
        "id": "t1",
        "goal": "demo goal",
        "done_criterion": {
            "kind": "all_of",
            "items": [{"kind": "node_closed", "node": "n1"}],
        },
        "nodes": [{"id": "n1"}],
    }
    base.update(overrides)
    return base


def test_missing_done_criterion_raises():
    raw = {"id": "t1", "goal": "g", "nodes": [{"id": "n1"}]}
    with pytest.raises(TaskLoadError, match="done_criterion"):
        parse_task_spec(raw)


def test_missing_id_raises():
    raw = _minimal_spec()
    del raw["id"]
    with pytest.raises(TaskLoadError, match="'id'"):
        parse_task_spec(raw)


def test_missing_goal_raises():
    raw = _minimal_spec()
    del raw["goal"]
    with pytest.raises(TaskLoadError, match="'goal'"):
        parse_task_spec(raw)


def test_minimal_spec_parses():
    spec = parse_task_spec(_minimal_spec())
    assert spec.id == "t1"
    assert spec.goal == "demo goal"
    assert spec.done_criterion.kind == "all_of"
    assert len(spec.nodes) == 1
    assert spec.nodes[0].id == "n1"
    assert spec.nodes[0].state is NodeState.OPEN


# ---------------------------------------------------------------------------
# Runner-correction: uses_runner is optional; runner_capabilities follows
# ---------------------------------------------------------------------------


def test_no_runner_usage_is_ok():
    """Pure-procedure tasks (no runner) are a regular case, not a
    separate 'standalone' carve-out."""
    spec = parse_task_spec(_minimal_spec())
    assert spec.uses_runner is None
    assert spec.runner_capabilities == ()
    assert spec.declares_runner_usage is False


def test_uses_runner_recorded():
    spec = parse_task_spec(_minimal_spec(
        uses_runner="reference_target",
        runner_capabilities=["trace", "re_execute"],
    ))
    assert spec.uses_runner == "reference_target"
    assert spec.runner_capabilities == ("trace", "re_execute")
    assert spec.declares_runner_usage is True


def test_runner_capabilities_without_uses_runner_rejected():
    """Capabilities addressed to thin air is a category error."""
    with pytest.raises(TaskLoadError, match="thin air"):
        parse_task_spec(_minimal_spec(runner_capabilities=["trace"]))


def test_runner_capabilities_must_be_strings():
    with pytest.raises(TaskLoadError, match="runner_capabilities"):
        parse_task_spec(_minimal_spec(
            uses_runner="r",
            runner_capabilities=[123],
        ))


# ---------------------------------------------------------------------------
# Runner-correction: input_contract is no longer mandatory for runner-less
# ---------------------------------------------------------------------------


def test_runnerless_task_without_contract_is_legal():
    """Old §4.2 required standalone tasks to carry a contract.  The
    runner-correction makes the contract a general reusability marker;
    a runner-less top-level goal is a perfectly valid no-contract task."""
    spec = parse_task_spec(_minimal_spec())  # uses_runner absent, no contract
    assert spec.input_contract is None
    assert spec.is_reusable is False


def test_runnerless_task_with_contract_is_reusable():
    spec = parse_task_spec(_minimal_spec(
        input_contract={
            "accepts": ["front_half", "back_half"],
            "produces": ["merged_pass_rate"],
        },
    ))
    assert spec.is_reusable is True
    assert spec.input_contract.accepts == ("front_half", "back_half")
    assert spec.input_contract.produces == ("merged_pass_rate",)


def test_bound_task_with_contract_is_legal():
    """A runner-using reusable algorithm-extraction task — common."""
    spec = parse_task_spec(_minimal_spec(
        uses_runner="reference_target",
        runner_capabilities=["trace"],
        input_contract={"accepts": ["target_addr"], "produces": ["algo_id"]},
    ))
    assert spec.declares_runner_usage is True
    assert spec.is_reusable is True


# ---------------------------------------------------------------------------
# Profile (intent-class) declaration
# ---------------------------------------------------------------------------


def test_profile_recorded():
    spec = parse_task_spec(_minimal_spec(profile="vmp_algorithm_extraction"))
    assert spec.profile == "vmp_algorithm_extraction"


def test_two_tasks_can_share_runner_with_different_profiles():
    """Same runner, different intents → different profiles is legal.
    The loader does not enforce uniqueness because there is none —
    the runner is a neutral workbench."""
    a = parse_task_spec(_minimal_spec(
        id="a", uses_runner="reference_target",
        profile="vmp_algorithm_extraction",
    ))
    b = parse_task_spec(_minimal_spec(
        id="b", uses_runner="reference_target",
        profile="integrity_check",
    ))
    assert a.uses_runner == b.uses_runner
    assert a.profile != b.profile


# ---------------------------------------------------------------------------
# done_criterion dangling references
# ---------------------------------------------------------------------------


def test_done_criterion_references_undeclared_node_rejected():
    with pytest.raises(TaskLoadError, match="undeclared node"):
        parse_task_spec(_minimal_spec(
            done_criterion={
                "kind": "node_closed",
                "node": "ghost",  # not declared
            },
            nodes=[{"id": "n1"}],
        ))


def test_done_criterion_references_undeclared_child_rejected():
    with pytest.raises(TaskLoadError, match="undeclared child"):
        parse_task_spec(_minimal_spec(
            done_criterion={"kind": "child_done", "child": "ghost"},
            children=[],
        ))


def test_current_focus_must_be_declared_node():
    with pytest.raises(TaskLoadError, match="current_focus"):
        parse_task_spec(_minimal_spec(current_focus="ghost"))


# ---------------------------------------------------------------------------
# Children — recursive parse
# ---------------------------------------------------------------------------


def test_children_parse_recursively():
    raw = _minimal_spec(
        done_criterion={
            "kind": "all_of",
            "items": [{"kind": "child_done", "child": "subA"}],
        },
        nodes=[],
        children=[{
            "id": "subA",
            "goal": "child goal",
            "done_criterion": {"kind": "node_closed", "node": "sub_n"},
            "nodes": [{"id": "sub_n"}],
        }],
    )
    spec = parse_task_spec(raw)
    assert len(spec.children) == 1
    assert spec.children[0].id == "subA"
    assert spec.children[0].nodes[0].id == "sub_n"


def test_duplicate_node_id_rejected():
    with pytest.raises(TaskLoadError, match="duplicate node"):
        parse_task_spec(_minimal_spec(nodes=[{"id": "n1"}, {"id": "n1"}]))


# ---------------------------------------------------------------------------
# File loader smoke test
# ---------------------------------------------------------------------------


def test_load_task_spec_from_file(tmp_path: Path):
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(_minimal_spec()))
    spec = load_task_spec(p)
    assert spec.id == "t1"


def test_file_not_found(tmp_path: Path):
    with pytest.raises(TaskLoadError, match="not found"):
        load_task_spec(tmp_path / "nope.json")


def test_invalid_json(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid")
    with pytest.raises(TaskLoadError, match="valid JSON"):
        load_task_spec(p)
