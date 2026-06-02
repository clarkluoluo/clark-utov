"""T7 — v0.4.0 mechanism floor penetration (PLAN §20.1.3 invariant #1).

Task gate + tree must let the v0.4.0 conjunctive gate fire at the
task-level "done" declaration: M1, M3, constant_provenance,
value_provenance, watch_first_write, scope_boundary_gate,
scope_upscale_gate — every probe with ``mechanism=True`` runs against
the params the caller supplies with the declaration.  Inserted /
replaced tasks inherit the same wiring because the gate consults the
import-time builtin registry (Lock B from v0.4.0), not anything the
task itself carries.
"""

from __future__ import annotations

import pytest

from engine.profile import (
    ConjunctiveGate,
    ProbeContext,
    ProfileRegistry,
)
from engine.task import (
    CriterionEvalContext,
    TaskAuditLog,
    TaskGate,
    TaskTree,
    assemble_task_tree,
    insert_child,
    parse_task_spec,
)


VMP_PROFILE = "vmp_algorithm_extraction"


@pytest.fixture()
def vmp_profile():
    return ProfileRegistry().load_chain(VMP_PROFILE)


# ---------------------------------------------------------------------------
# Fixture — small tc3-shaped task
# ---------------------------------------------------------------------------


def _tc3_root_with_merge_artefact_dep() -> dict:
    return {
        "id": "restore_sign",
        "goal": "restore full sign",
        "uses_runner": "reference_target",
        "runner_capabilities": ["trace"],
        "profile": VMP_PROFILE,
        "nodes": [{"id": "front_half"}, {"id": "back_half"}],
        "done_criterion": {
            "kind": "all_of",
            "items": [
                {"kind": "node_closed", "node": "front_half"},
                {"kind": "node_closed", "node": "back_half"},
                {"kind": "named_artefact", "name": "merge_cross_check"},
            ],
        },
    }


# ---------------------------------------------------------------------------
# Mechanism floor fires when probe_ctx supplied
# ---------------------------------------------------------------------------


def test_task_gate_passes_conjunctive_gate_when_clean(vmp_profile):
    """Headline floor wiring: TaskGate composes ConjunctiveGate when
    both criterion + mechanism floor agree."""
    spec = parse_task_spec(_tc3_root_with_merge_artefact_dep())
    cg = ConjunctiveGate(vmp_profile)
    gate = TaskGate(spec=spec, conjunctive_gate=cg)
    result = gate.evaluate_task_done(
        ctx=CriterionEvalContext(
            closed_nodes=frozenset({"front_half", "back_half"}),
            present_artefacts=frozenset({"merge_cross_check"}),
        ),
        probe_ctx=ProbeContext(
            method="finalize_verdict",
            params={"scope_claim": "task_bound", "scope_observed": "task_bound"},
            profile=vmp_profile,
        ),
    )
    assert result.passed is True
    assert result.mechanism_floor_passed is True
    assert result.mechanism_failing_probes == ()


def test_task_gate_fails_when_scope_overreach_on_task_done_declaration(vmp_profile):
    """Task-level declaration with scope_claim > observed → fails the
    floor.  Even though the criterion is satisfied, the mechanism
    layer refuses (PLAN §20.1.3 invariant #1)."""
    spec = parse_task_spec(_tc3_root_with_merge_artefact_dep())
    cg = ConjunctiveGate(vmp_profile)
    gate = TaskGate(spec=spec, conjunctive_gate=cg)
    result = gate.evaluate_task_done(
        ctx=CriterionEvalContext(
            closed_nodes=frozenset({"front_half", "back_half"}),
            present_artefacts=frozenset({"merge_cross_check"}),
        ),
        probe_ctx=ProbeContext(
            method="finalize_verdict",
            params={"scope_claim": "cross_env", "scope_observed": "task_bound"},
            profile=vmp_profile,
        ),
    )
    assert result.passed is False
    assert result.mechanism_floor_passed is False
    assert "scope_boundary_gate" in result.mechanism_failing_probes


def test_task_gate_without_probe_ctx_skips_mechanism_floor(vmp_profile):
    """Backwards-compat default: when the caller doesn't supply
    probe_ctx, the floor stays inert.  done_criterion still fires."""
    spec = parse_task_spec(_tc3_root_with_merge_artefact_dep())
    cg = ConjunctiveGate(vmp_profile)
    gate = TaskGate(spec=spec, conjunctive_gate=cg)
    result = gate.evaluate_task_done(
        ctx=CriterionEvalContext(
            closed_nodes=frozenset({"front_half", "back_half"}),
            present_artefacts=frozenset({"merge_cross_check"}),
        ),
        # no probe_ctx
    )
    assert result.passed is True
    assert result.mechanism_floor_passed is True


# ---------------------------------------------------------------------------
# Inserted task inherits mechanism floor (Lock B from v0.4.0)
# ---------------------------------------------------------------------------


def test_inserted_task_still_walks_mechanism_floor(vmp_profile):
    """Insert a child + evaluate the new root's done.  Mechanism floor
    fires on the inserted node's params the same way it would on the
    original; the gate isn't 'opted in' on a per-task basis."""
    spec = parse_task_spec({
        "id": "root",
        "goal": "g",
        "uses_runner": "reference_target",
        "runner_capabilities": ["trace"],
        "profile": VMP_PROFILE,
        "done_criterion": {"kind": "child_done", "child": "child1"},
        "children": [{
            "id": "child1",
            "goal": "g",
            "done_criterion": {"kind": "node_closed", "node": "c1n"},
            "nodes": [{"id": "c1n"}],
            "input_contract": {},
        }],
    })
    log = TaskAuditLog()
    # Insert a sibling that would also need to be closed for root done.
    # But the parent's criterion only names child1, so child1 closing
    # is enough for done_criterion.
    new_root = insert_child(
        spec, "root",
        parse_task_spec({
            "id": "child2",
            "goal": "newly inserted gap-closing task",
            "done_criterion": {"kind": "node_closed", "node": "c2n"},
            "nodes": [{"id": "c2n"}],
            "input_contract": {},
        }),
        who="agent", why="discovered gap", log=log,
    )
    tree = assemble_task_tree(new_root)

    cg = ConjunctiveGate(vmp_profile)
    # Hostile params: cross_env claim with task_bound observed → scope
    # gate must fire on the task-done call exactly as it would for any
    # other archival surface.
    result = tree.evaluate_root_done(
        closed_nodes=frozenset({"c1n", "c2n"}),
        conjunctive_gate=cg,
        probe_ctx=ProbeContext(
            method="finalize_verdict",
            params={"scope_claim": "cross_env", "scope_observed": "task_bound"},
            profile=vmp_profile,
        ),
    )
    assert result.passed is False
    assert "scope_boundary_gate" in result.mechanism_failing_probes
