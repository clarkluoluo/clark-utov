"""Reference-case regression driven through hypotask.interface (additive integration).

This exercises the clark-Hypotask cognitive layer end-to-end via the utov
integration layer, WITHOUT touching the RE engine. It reproduces the structural
shape of the reference case — a parent "recover full sign" task split into two halves
plus a cross-node end-to-end byte-parity gate — and proves the two pitfalls
the reference case hit are now structurally blocked:

  1. node-closed ≠ task-done: closing the nodes of a subtask doesn't make the
     parent done; the parent gate also needs both subtasks done AND the
     cross-node parity probe to pass.
  2. missing the merge parity: even with both subtasks done, the parent stays
     blocked until end_to_end_bytewise parity actually passes.

Skips cleanly if clark-hypotask isn't installed, so the engine baseline
(752 passed, 1 skipped) is unaffected when the extra is absent.
"""

from __future__ import annotations

import pytest

pytest.importorskip("hypotask")

from engine.integration.bootstrap import (  # noqa: E402
    bootstrap_hypotask,
    make_test_session,
)


@pytest.fixture
def db_path(tmp_path):
    """Fresh Hypotask sqlite db per test (agent never touches it directly)."""
    return str(tmp_path / "reference_case.db")


# Strong evidence pack: dataflow provenance, full chain, producer reads nothing
# runtime-derived, recomputable, multi-input — enough for the verifier to
# adjudicate a closed_form (class A) node state.
def _strong_ev() -> dict:
    return {
        "kind": "dataflow_provenance",
        "provenance_depth": "full_chain",
        "producer_reads": [],
        "inputs_tested": 3,
        "recomputable_function_provided": True,
        "provenance_does_not_read_runtime_quantity": True,
    }


def _subtask_spec(goal: str, node_names: list[str]) -> dict:
    return {
        "goal": goal,
        "uses_runner": "unidbg_signrunner",
        "runner_capabilities_required": ["trace", "re-execute"],
        "profile_domain": "vmp_algorithm_extraction",
        "nodes": [{"name": n, "deps": []} for n in node_names],
        "done_criterion": {"all_of": [{"all_nodes_closed": True}]},
    }


def _close_all_nodes(session, task_id):
    """Write a strong finding to every node and let the verifier upgrade it."""
    from hypotask.interface import get_task_state, update_node_state, write_finding

    st = get_task_state(session, task_id)
    for node in st["nodes"]:
        nid = node["node_id"]
        wf = write_finding(
            session,
            node_id=nid,
            content=f"{node['name']}: dataflow provenance, producer reads no runtime quantity",
            evidence_source=_strong_ev(),
            claimed_evidence_class="B",
            claimed_scope={"task_bound": True},
        )
        assert wf["ok"], wf
        r = update_node_state(session, nid, evidence=_strong_ev())
        assert r["ok"], r


def test_reference_case_parent_blocked_until_subtasks_done_and_parity_passes(db_path):
    """Full lifecycle: parent stays blocked through every partial state, only
    passes once both halves are done AND end-to-end parity holds."""
    from hypotask.interface import claim_task_done, get_task_state

    bootstrap_hypotask(db_path)

    # Mutable parity result so we can flip it within one registered probe.
    parity_state = {"ok": False}

    def parity_fn(task_id, cfg):
        return parity_state["ok"], {"checked": cfg.get("name")}

    with make_test_session(db_path, parity_fn=parity_fn) as s:
        # Two halves as standalone subtasks; capture their ids for the parent's
        # done_criterion (which is locked at creation).
        sub_a = s.create_task(_subtask_spec("recover first half", ["scratch21", "template"]))
        sub_b = s.create_task(_subtask_spec("recover second half", ["prefix", "sm3_digest"]))

        parent = s.create_task(
            {
                "goal": "recover full sign + end-to-end bytewise parity",
                "uses_runner": "unidbg_signrunner",
                "runner_capabilities_required": [
                    "trace", "re-execute", "memregion_watch", "phase_install",
                ],
                "profile_domain": "vmp_algorithm_extraction",
                "nodes": [{"name": "merge_parity", "deps": []}],
                "done_criterion": {
                    "all_of": [
                        {"subtask_done": sub_a},
                        {"subtask_done": sub_b},
                        {"cross_node_parity": {"name": "end_to_end_bytewise"}},
                    ]
                },
            }
        )

        # (1) Nothing done yet → parent blocked, missing lists every gap.
        v = claim_task_done(s, parent)
        assert not v["ok"]
        assert any("subtask_done" in m for m in v["missing"])

        # (2) Close + claim subtask A done. Parent still blocked (B + parity).
        _close_all_nodes(s, sub_a)
        assert claim_task_done(s, sub_a)["ok"]
        v = claim_task_done(s, parent)
        assert not v["ok"], "one half done is not the whole task — reference-case pitfall #1"

        # (3) Close + claim subtask B done. Both halves done, but parity not yet
        #     proven → parent STILL blocked (reference-case pitfall #2: the missed merge).
        _close_all_nodes(s, sub_b)
        assert claim_task_done(s, sub_b)["ok"]
        v = claim_task_done(s, parent)
        assert not v["ok"], "both halves done but no parity = not done"
        assert any("cross_node_parity" in m for m in v["missing"])

        # (4) End-to-end byte parity actually passes → parent done at last.
        parity_state["ok"] = True
        v = claim_task_done(s, parent)
        assert v["ok"], v
        assert get_task_state(s, parent)["task"]["status"] == "done"


def test_reference_case_done_criterion_is_locked_at_runtime(db_path):
    """The parent's done_criterion can't be weakened at runtime (objectivity)."""
    from hypotask.interface import log_task_op
    from hypotask.task.invariants import InvariantViolation

    bootstrap_hypotask(db_path)
    with make_test_session(db_path, parity_fn=lambda t, c: (True, {})) as s:
        sub = s.create_task(_subtask_spec("half", ["n1"]))
        parent = s.create_task(
            {
                "goal": "locked",
                "profile_domain": "vmp_algorithm_extraction",
                "nodes": [{"name": "root", "deps": []}],
                "done_criterion": {"all_of": [{"subtask_done": sub}]},
            }
        )
        # Try to replace the subtask while changing the parent's done_criterion
        # reference — invariant must reject (done_criterion immutability).
        with pytest.raises(InvariantViolation):
            log_task_op(
                s,
                "replace",
                {
                    "old_task_id": sub,
                    "new_spec": {
                        "goal": "half v2",
                        "profile_domain": "vmp_algorithm_extraction",
                        "nodes": [{"name": "n1", "deps": []}],
                        "done_criterion": {"all_of": [{"finding_count_min": {"node_id": "n1", "min": 1}}]},
                    },
                },
            )


def test_unidbg_runner_capabilities_from_adapter(db_path):
    """UnidbgSignRunner advertises only capabilities its adapter implements."""
    from engine.integration import UnidbgSignRunner
    from engine.runner_client import NullRunnerAdapter
    from engine.types import TargetMeta

    meta = TargetMeta(
        target_name="ReferenceTarget",
        arch="arm64",
        algo_entry_pc=0x40007D88,
        algo_exit_pc=0x40008000,
        input_length=None,
        output_length=32,
        algo_symbol="sign",
        emulator_name="unidbg",
        emulator_version="0.9",
    )
    # File-mode adapter implements only metadata() → no live capabilities.
    runner = UnidbgSignRunner(NullRunnerAdapter(meta))
    assert runner.name == "unidbg_signrunner"
    assert runner.capabilities() == []
    assert not runner.supports(["trace"])
