"""Task object model — core data types (PLAN §20).

The runner-correction (PLAN §20.1, revised 2026-05-29): runner is a
**neutral workbench**, not the parent of a task.  A task *uses* a
runner and *needs* certain runner capabilities; the runner does not
know any task exists.  Consequences for the types here:

  * No "bound vs standalone" carve-out.  Every TaskSpec uniformly
    declares ``uses_runner`` (optional) and ``runner_capabilities``
    (sequence of capability names the runner must support).
  * ``input_contract`` is no longer reserved for "standalone tasks".
    Any TaskSpec that is referenced by another task's child list or
    call site MUST carry an ``input_contract``; a top-level concrete
    goal task without callers may omit it.

All containers are frozen so registry / gate consumers can hand them
out without worrying about downstream mutation. The mutation surfaces
(insert / replace / delete via the audit log) construct new
TaskSpec instances rather than mutating in place.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional


class NodeState(enum.Enum):
    """Per-node closure status the task gate consults.

    ``OPEN`` is the default — the node has been declared as part of
    the task but no closure attestation has been recorded.
    ``CLOSED`` means the node-level gate has signed; this is the only
    state that satisfies a ``node_closed`` criterion item.
    ``STUCK`` means the node hit a block_cause class-3 (true boundary)
    — informational; satisfies neither closure nor failure.
    """

    OPEN = "open"
    CLOSED = "closed"
    STUCK = "stuck"


@dataclass(frozen=True)
class NodeRef:
    """A node entry on a task spec.

    Carries the declared identity and current closure state. The state
    transitions through the task lifecycle but the spec object is
    re-created (frozen) on each transition so the audit log records
    the change.
    """

    id: str
    state: NodeState = NodeState.OPEN
    description: str = ""

    def with_state(self, new_state: NodeState) -> "NodeRef":
        return NodeRef(id=self.id, state=new_state, description=self.description)


@dataclass(frozen=True)
class TaskSpec:
    """One task — a user-intent instance.

    Mandatory: ``id``, ``goal``, ``done_criterion``.

    Runner usage is **declared, not bound** (PLAN §20.1 runner
    correction):

      * ``uses_runner`` — the runner id the task uses, or ``None``
        for pure-procedure tasks.  The runner does not know it is
        being used; the task spec is the only place the link is
        recorded.
      * ``runner_capabilities`` — the abilities the task needs from
        the runner (``trace`` / ``re_execute`` / ``memregion_watch``
        / ``timing`` / …).  At compose time the bound runner must
        declare it supports each one.

    Profile usage:

      * ``profile`` — the intent-class profile this task loads (e.g.
        ``vmp_algorithm_extraction``).  Two tasks on the same runner
        with different profiles do not share semantics; the runner
        does not constrain which profile a task may load.

    Cross-task composition:

      * ``children`` — recursive sub-tasks; each is itself a
        TaskSpec.  The parent's ``done_criterion`` may reference
        child ids via ``child_done`` atoms.
      * ``input_contract`` — required when this task is invoked from
        another task; optional for top-level goal tasks with no
        callers (PLAN §20.1.3 invariant #5).
      * ``implementation_path`` — OPTIONAL light-to-heavy staged plan
        (roadmap §8.11/§9.4). A tool, not a mandate: a brief that wants
        a rhythm may declare it; omitting it still loads. See
        :mod:`engine.task.implementation_path`.

    ``done_criterion`` is the objective task-termination check.  It
    cannot be mutated at runtime — every audit operation that touches
    a task spec preserves the criterion verbatim.  See
    :mod:`engine.task.audit` for the enforcement.
    """

    id: str
    goal: str
    done_criterion: "CriterionItem"  # forward ref — defined in done_criterion.py
    nodes: tuple[NodeRef, ...] = ()
    current_focus: Optional[str] = None
    uses_runner: Optional[str] = None
    runner_capabilities: tuple[str, ...] = ()
    profile: Optional[str] = None
    children: tuple["TaskSpec", ...] = ()
    input_contract: Optional["InputContract"] = None  # forward ref — contract.py
    implementation_path: Optional["ImplementationPath"] = None  # forward ref — implementation_path.py
    description: str = ""

    @property
    def is_parent(self) -> bool:
        return bool(self.children)

    @property
    def declares_runner_usage(self) -> bool:
        """True when the task spec names a runner workbench."""
        return self.uses_runner is not None

    @property
    def is_reusable(self) -> bool:
        """True when this task carries an input_contract — i.e. is
        designed to be invoked by another task."""
        return self.input_contract is not None

    def node(self, node_id: str) -> Optional[NodeRef]:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def child(self, child_id: str) -> Optional["TaskSpec"]:
        for c in self.children:
            if c.id == child_id:
                return c
        return None
