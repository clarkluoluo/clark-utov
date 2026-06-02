"""Adversarial: prove the reference case three-layer concept-spoof can't pass in utov.

The reference case showed an agent "completing" a task in 10 min by concept-spoofing at
three layers:
  L1. declare a node closed without real closure evidence,
  L2. write a done_criterion that only checks a bare field (state == "closed"),
  L3. use a vague state name ("closed"/"done"/"ok") to pass off form as substance.

Under the locked architecture (utov judges, Hypotask stores), the defence lives
in utov's own task system. These tests pin that each spoof layer is blocked HERE
and that the refusal names the missing substance. If a test fails, that layer
has a real form/substance seam — fix it, don't weaken the test.
"""

from __future__ import annotations

import pytest

from engine.task.done_criterion import CriterionEvalContext, CriterionItem
from engine.task.gate import TaskGate
from engine.task.loader import TaskLoadError, parse_task_spec
from engine.task.types import NodeRef, NodeState, TaskSpec


# ---------------------------------------------------------------------------
# Layer 2 (loader grammar): you cannot even EXPRESS a bare-field criterion.
# ---------------------------------------------------------------------------

def test_L2_done_criterion_cannot_be_a_bare_field_comparison():
    """done_criterion has no 'state == X' atom. A spoofer trying a bare-field
    check is rejected at parse — the only atoms are node_closed / child_done /
    named_artefact / all_of / any_of, each referencing real closure/done/
    artefact state, never a raw field value."""
    spoofs = [
        {"kind": "state_equals", "field": "state", "value": "closed"},
        {"kind": "field_check", "node": "n1", "state": "closed"},
        {"kind": "file_exists", "path": "out.bin"},
        {"kind": "looks_done"},
    ]
    for bad in spoofs:
        with pytest.raises(TaskLoadError):
            parse_task_spec(
                {"id": "t", "goal": "g", "nodes": [{"id": "n1"}],
                 "done_criterion": bad}
            )


def test_L2_unknown_criterion_kind_names_the_substance_atoms():
    """Rejection names the valid atoms so the spoofer learns what substance is
    required — not a silent fail."""
    with pytest.raises(TaskLoadError) as ei:
        parse_task_spec(
            {"id": "t", "goal": "g", "nodes": [{"id": "n1"}],
             "done_criterion": {"kind": "looks_done"}}
        )
    assert "node_closed" in str(ei.value)


# ---------------------------------------------------------------------------
# Layer 3 (state naming): node state is a fixed lifecycle enum — no vague
# custom name can be injected to pass off form as substance.
# ---------------------------------------------------------------------------

def test_L3_node_state_is_a_closed_enum_no_vague_names():
    """A spoofer cannot declare a node with a vague custom state name; the
    loader only accepts the fixed NodeState lifecycle (open/closed/stuck)."""
    for vague in ("done", "ok", "completed", "finished", "success", "passed"):
        with pytest.raises(TaskLoadError):
            parse_task_spec(
                {"id": "t", "goal": "g",
                 "nodes": [{"id": "n1", "state": vague}],
                 "done_criterion": {"kind": "node_closed", "node": "n1"}}
            )


# ---------------------------------------------------------------------------
# Layer 1 (self-declared closure): done-ness must key on runtime judgement
# (ctx = utov's real verdict), not the spec's own say-so.
# ---------------------------------------------------------------------------

def test_L1_task_done_uses_runtime_judgement_not_self_declared_state():
    """When the runtime ctx (utov's real judgement) reports NO closed nodes, a
    'task done' declaration is refused even if the spec pre-stamps the node
    closed. The gate evaluates against ctx — self-declaration is not a verdict."""
    spec = TaskSpec(
        id="t", goal="g",
        done_criterion=CriterionItem(kind="node_closed", node="n1"),
    )
    gate = TaskGate(spec=spec)
    ctx = CriterionEvalContext(closed_nodes=frozenset())  # utov: n1 NOT closed
    result = gate.evaluate_task_done(ctx=ctx)
    assert not result.passed
    assert any("n1" in g and "not closed" in g for g in result.gaps)


def test_L1_task_done_passes_only_with_real_closure_in_ctx():
    """Symmetric: real closure in the runtime ctx → same spec passes. The gate
    keys on substance (ctx verdict); a pass needs the real verdict."""
    spec = TaskSpec(
        id="t", goal="g",
        done_criterion=CriterionItem(kind="node_closed", node="n1"),
    )
    gate = TaskGate(spec=spec)
    ctx = CriterionEvalContext(closed_nodes=frozenset({"n1"}))
    assert gate.evaluate_task_done(ctx=ctx).passed


def test_L1_KNOWN_SEAM_derive_ctx_from_spec_trusts_self_declared_state():
    """KNOWN SEAM (documents current behaviour): with NO runtime ctx,
    TaskGate._derive_ctx_from_spec() derives closed_nodes from the spec's OWN
    node.state. So a spec self-stamping state=CLOSED passes if the caller omits
    ctx. This is the L1 spoof's only foothold in utov — NOT reachable on the
    production path where the caller supplies utov's real judgement as ctx.

    Pinned so the contract is explicit: callers MUST pass a runtime ctx from
    real judgement; relying on _derive_ctx_from_spec for a done decision is the
    spoofable path. Fix (if form==substance wanted without caller discipline):
    make _derive_ctx_from_spec refuse self-declared closure for done decisions.
    """
    spec = TaskSpec(
        id="t", goal="g",
        nodes=(NodeRef(id="n1", state=NodeState.CLOSED),),
        done_criterion=CriterionItem(kind="node_closed", node="n1"),
    )
    gate = TaskGate(spec=spec)
    assert gate.evaluate_task_done().passed  # ctx=None → trusts self-declaration
