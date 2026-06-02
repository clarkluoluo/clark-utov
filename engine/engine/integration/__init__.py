"""clark-Hypotask integration layer (additive — does NOT touch the RE engine).

Architecture (locked): **utov judges, Hypotask stores.**

  - utov engine  → owns ALL real judgement: the RE heavy lifting (trace → DFG →
    slice → simplify → hypothesis), AND every verdict (what counts as closed,
    evidence_class, scope, source provenance, done-or-not). utov enforces its own
    bottom lines. Judgement is tuned against VMP and user-adjustable; it stays
    closed in utov because an open generic judge can't keep the VMP task chain
    from collapsing into dead ends.
  - Hypotask    → a ledger + topology + checklist. Stores what utov submits,
    links finding→node→task, reports "is the thing present / where are we".
    It does NOT judge correctness.

So this layer uses ONLY Hypotask's store / link / query surface (open_session,
create_task, write_finding to record a utov verdict, get_task_state,
get_ledger_trail, the neutral-workbench runner abstraction, and cross-node
checklist hooks whose judgement is a utov callable). It deliberately does NOT
register Hypotask's builtin mechanism probes or verifier plugins — those make
Hypotask judge, which is utov's role.

Nothing in `engine.*` imports this package; only application entry points and
tests do. Importing requires `clark-hypotask` (the `hypotask` extra); imported
lazily so the engine baseline stays green when hypotask is absent.

See `hypotask_overreach_log.md` (repo root) for points where Hypotask still
tries to judge on the write path (to be fixed Hypotask-side).
"""

from __future__ import annotations

from .unidbg_runner import UnidbgSignRunner
from .bootstrap import (
    bootstrap_hypotask,
    make_test_session,
    make_dev_session,
)
from .adapter import (
    FindingResult,
    NodeView,
    TaskStateView,
    ClaimResult,
    store_finding,
    task_state,
    active_findings,
    claim_done,
)

__all__ = [
    "UnidbgSignRunner",
    "bootstrap_hypotask",
    "make_test_session",
    "make_dev_session",
    # adapter: the single choke point for Hypotask return shapes (insulates utov
    # from Hypotask schema/key changes — see adapter.py).
    "FindingResult",
    "NodeView",
    "TaskStateView",
    "ClaimResult",
    "store_finding",
    "task_state",
    "active_findings",
    "claim_done",
]
