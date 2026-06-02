"""Adapter: single choke point between utov and Hypotask's return shapes.

utov never touches Hypotask's DB (rule held — verified), so Hypotask may change
its storage schema freely. utov's ONLY exposure is the key names in the dicts
`hypotask.interface` returns. If Hypotask renames a key, only THIS file breaks.

All write/query calls go through here; every raw Hypotask dict is unpacked into
a utov-owned frozen dataclass. utov business/test code consumes the dataclasses,
never the raw dicts. The adapter normalises shapes only — utov judges, Hypotask
stores; nothing here judges.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hypotask.interface import (
    claim_task_done as _claim_task_done,
    get_active_findings as _get_active_findings,
    get_task_state as _get_task_state,
    write_finding as _write_finding,
)


@dataclass(frozen=True)
class FindingResult:
    finding_id: str
    evidence_class: str
    scope: dict
    ok: bool
    raw: dict = field(repr=False, default_factory=dict)


@dataclass(frozen=True)
class NodeView:
    node_id: str
    name: str
    state: str
    raw: dict = field(repr=False, default_factory=dict)


@dataclass(frozen=True)
class TaskStateView:
    task_id: str
    status: str
    nodes: tuple
    raw: dict = field(repr=False, default_factory=dict)


@dataclass(frozen=True)
class ClaimResult:
    ok: bool
    passed: tuple
    missing: tuple
    raw: dict = field(repr=False, default_factory=dict)


def store_finding(
    session,
    *,
    node_id: str,
    content: str,
    evidence_source: dict,
    evidence_class: str,
    scope: dict | None = None,
) -> FindingResult:
    """Store a finding utov has ALREADY judged; Hypotask records it verbatim."""
    r = _write_finding(
        session,
        node_id=node_id,
        content=content,
        evidence_source=evidence_source,
        claimed_evidence_class=evidence_class,
        claimed_scope=scope or {},
    )
    verdict = r.get("verifier_verdict", {})
    return FindingResult(
        finding_id=r["finding_id"],
        evidence_class=verdict.get("evidence_class", evidence_class),
        scope=verdict.get("scope", scope or {}),
        ok=r.get("ok", True),
        raw=r,
    )


def task_state(session, task_id: str) -> TaskStateView:
    st = _get_task_state(session, task_id)
    task = st.get("task", {}) or {}
    nodes = tuple(
        NodeView(
            node_id=n["node_id"],
            name=n.get("name", ""),
            state=n.get("state", ""),
            raw=n,
        )
        for n in st.get("nodes", [])
    )
    return TaskStateView(
        task_id=task.get("task_id", task_id),
        status=task.get("status", ""),
        nodes=nodes,
        raw=st,
    )


def active_findings(session, *, task_id=None, node_id=None) -> list:
    """Active findings as raw rows; normalise here if row keys ever shift."""
    return _get_active_findings(session, task_id=task_id, node_id=node_id)


def claim_done(session, task_id: str, evidence: dict | None = None) -> ClaimResult:
    r = _claim_task_done(session, task_id, evidence=evidence)
    return ClaimResult(
        ok=r.get("ok", False),
        passed=tuple(r.get("passed", []) or []),
        missing=tuple(r.get("missing", []) or []),
        raw=r,
    )
