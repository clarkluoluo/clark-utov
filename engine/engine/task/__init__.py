"""Task object model (v0.5.0-dev · PLAN §20).

The third top-level object alongside ``Node`` and ``Finding``. A
:class:`TaskSpec` declares ``goal / nodes / done_criterion / focus``
plus optional runner usage (the runner is a neutral workbench, used
by the task, not the parent of it — PLAN §20.1 runner-correction).
``done_criterion`` is the **objective** termination check the agent
cannot redefine at runtime. Two-layer M1 lives here: a node-level
gate signs each node's closure (the v0.4.0 conjunctive gate); a
task-level gate signs the entire task's termination
(``TaskGate`` — composes ``done_criterion`` + the floor of mechanism
probes; lands incrementally — see T3).

Why this exists — the reference case: the agent mistook node M1 pass for task
termination, dropping the merge cross-check.  No object carried the
distinction "this task isn't done until the merge runs."  Task
supplies that anchor.

Public re-exports kept narrow; see module docstrings for the full
surface. As more modules ship (TaskGate, TaskTree, TaskAuditLog), they
join the re-export list.
"""

from engine.task.audit import (
    TaskAuditEntry,
    TaskAuditError,
    TaskAuditLog,
    TaskAuditOp,
    assert_done_criterion_unchanged,
    delete_child,
    insert_child,
    replace_child,
)
from engine.task.contract import (
    ContractMismatchError,
    InputContract,
    validate_contract_compose,
)
from engine.task.done_criterion import (
    CriterionEvalContext,
    CriterionEvalResult,
    CriterionItem,
    evaluate_done_criterion,
    referenced_artefacts,
    referenced_children,
    referenced_nodes,
)
from engine.task.gate import (
    TaskDoneRefusal,
    TaskGate,
    TaskGateResult,
)
from engine.task.loader import (
    TaskLoadError,
    load_task_spec,
    parse_task_spec,
)
from engine.task.tree import (
    TaskTree,
    assemble_task_tree,
)
from engine.task.types import (
    NodeRef,
    NodeState,
    TaskSpec,
)


__all__ = [
    "ContractMismatchError",
    "CriterionEvalContext",
    "CriterionEvalResult",
    "CriterionItem",
    "InputContract",
    "NodeRef",
    "NodeState",
    "TaskAuditEntry",
    "TaskAuditError",
    "TaskAuditLog",
    "TaskAuditOp",
    "TaskDoneRefusal",
    "TaskGate",
    "TaskGateResult",
    "TaskLoadError",
    "TaskSpec",
    "TaskTree",
    "assemble_task_tree",
    "assert_done_criterion_unchanged",
    "delete_child",
    "evaluate_done_criterion",
    "insert_child",
    "load_task_spec",
    "parse_task_spec",
    "referenced_artefacts",
    "referenced_children",
    "referenced_nodes",
    "replace_child",
    "validate_contract_compose",
]
