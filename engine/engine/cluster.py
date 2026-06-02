"""Cluster-finding cascade primitives (capability_request.md §P1-4).

The reference target hindsight: the ``strb`` cluster (finding ``#296`` plus 8
member ``idxs``) had a hand-rolled invalidation path inside one
pipeline script — when the parent was refuted, ledger ops cascading
the 8 members back to ``pending`` were manual. That worked for one
shape but it was bespoke. The agent needs a general primitive:

    parent finding invalidated
      → every member finding's origin_hyp flips to ``pending``
      → audit row per flip, batched under one cascade id

This module owns the primitive (decoupled from the SQL helpers in
``store.py``); ``Core.invalidate_cluster`` is the public wrapper.

The cluster relation lives in the existing ``finding_groups`` table —
no new schema. Anyone who can ``link_finding_group_members`` already
has a cluster; this module gives them a uniform way to roll it back.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CascadeReport:
    """Outcome of one ``invalidate_cluster`` call."""
    cascade_id: str               # short uuid; lands in audit `reason`
    parent_finding_id: int
    parent_hyp_id: int | None
    member_hyp_ids: tuple[int, ...]
    skipped_member_finding_ids: tuple[int, ...]  # members with no origin_hyp_id
    parent_origin_was_already_failed: bool


def _origin_hyp_id_for_finding(
    findings_conn: sqlite3.Connection,
    finding_id: int,
) -> int | None:
    row = findings_conn.execute(
        "SELECT origin_hyp_id FROM findings WHERE id = ?",
        (finding_id,),
    ).fetchone()
    if row is None:
        return None
    return row[0]


def collect_member_finding_ids(
    findings_conn: sqlite3.Connection,
    parent_finding_id: int,
) -> list[int]:
    """Return all member finding ids registered for ``parent_finding_id``
    in ``finding_groups``. Empty list = not a cluster (or no members
    registered yet)."""
    rows = findings_conn.execute(
        "SELECT member_finding_id FROM finding_groups "
        "WHERE parent_finding_id = ? ORDER BY id ASC",
        (parent_finding_id,),
    ).fetchall()
    return [int(r[0]) for r in rows]


def collect_member_origin_hyp_ids(
    findings_conn: sqlite3.Connection,
    parent_finding_id: int,
) -> tuple[list[int], list[int]]:
    """Return ``(hyp_ids, skipped_member_finding_ids)``.

    ``hyp_ids`` is every member's ``origin_hyp_id`` (deduplicated, order
    preserved by finding_groups insertion order). ``skipped`` contains
    members whose origin_hyp is null — those can't be flipped back to
    pending and the caller surfaces them in the audit row.
    """
    member_ids = collect_member_finding_ids(findings_conn, parent_finding_id)
    if not member_ids:
        return [], []
    placeholders = ",".join("?" * len(member_ids))
    rows = findings_conn.execute(
        f"SELECT id, origin_hyp_id FROM findings WHERE id IN ({placeholders})",
        member_ids,
    ).fetchall()
    by_member: dict[int, int | None] = {int(r[0]): r[1] for r in rows}
    hyp_ids: list[int] = []
    skipped: list[int] = []
    seen: set[int] = set()
    # Preserve finding_groups order.
    for mid in member_ids:
        oh = by_member.get(mid)
        if oh is None:
            skipped.append(mid)
            continue
        oh_i = int(oh)
        if oh_i in seen:
            continue
        seen.add(oh_i)
        hyp_ids.append(oh_i)
    return hyp_ids, skipped


def make_cascade_id() -> str:
    """Short ``invalidate_cluster``-#xxxxxx tag that lands in the
    audit reason so all per-hyp interventions of one cascade can be
    grouped after the fact."""
    return f"cluster_cascade:{uuid.uuid4().hex[:8]}"
