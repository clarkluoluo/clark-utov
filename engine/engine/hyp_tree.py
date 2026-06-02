"""Hypothesis ledger CRUD over the refactored schema (DECISIONS D-027).

Key model change vs the old design:
  - Claim templates are deduplicated by (kind, source, payload_hash). A claim
    type with 10k occurrences is ONE template row, NOT 10k.
  - Big payloads live in hyp_payloads (content-addressed). Multiple templates
    or verifier results pointing at identical JSON share one row.
  - Anchors (trace_idx) are normalized: one hyp can cover N points, one point
    can be touched by many hyps. Bidirectional indexes for position queries.
  - Tags are multi-axis. Free to add new axes without schema change.
  - Dependencies beyond parent_id are explicit (`hyp_dependencies` table).

Public API:

  HypTree.add(...)               — insert a hypothesis, return id
  HypTree.add_anchor(hyp_id, idx, pc)
  HypTree.add_tag(hyp_id, axis, value)
  HypTree.add_dependency(from_id, to_id, kind)
  HypTree.get(hyp_id)            — fetch one
  HypTree.mark_verdict(hyp_id, verdict, verifier_result)
  HypTree.next_pending_sibling(hyp_id)
  HypTree.abandon_subtree(hyp_id)   — mark + optionally archive
  HypTree.query(filters)         — flexible multi-axis query (kind/status/tag/anchor_idx/anchor_pc)
  HypTree.downstream_of(hyp_id)  — explicit dependency graph BFS
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable

from .store import (
    _now_iso,
    template_hash,
    upsert_payload,
)


@dataclass
class HypNode:
    id: int
    parent_id: int | None
    depth: int
    status: str
    kind: str
    source: str
    subject: str
    payload: dict[str, Any]
    confidence: float | None


class HypTree:
    """Thin layer over the refactored hypotheses.sqlite (7-table schema)."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # --------------------------------- WRITE ---------------------------------

    def add(
        self,
        parent_id: int | None,
        kind: str,
        subject: str,
        payload: dict[str, Any],
        confidence: float | None,
        source: str = "plugin",
        anchors: Iterable[tuple[int, int]] | None = None,
        tags: Iterable[tuple[str, str]] | None = None,
        created_in_stage: str | None = None,
    ) -> int:
        """Insert a hypothesis. Returns its id.

        Args:
            anchors: iterable of (trace_idx, pc) the hyp applies at.
            tags:    iterable of (axis, value) multi-dim labels.

        The payload + template are upserted (deduplicated by content-hash);
        if a claim with identical (kind, source, payload) already exists, we
        reuse its template id.
        """
        depth = 0 if parent_id is None else (self.get(parent_id).depth + 1)
        payload_ref = upsert_payload(self.conn, payload)
        thash = template_hash(kind, source, payload_ref)

        row = self.conn.execute(
            "SELECT id FROM claim_templates WHERE template_hash = ?", (thash,),
        ).fetchone()
        if row is not None:
            template_id = row[0]
        else:
            cur = self.conn.execute(
                "INSERT INTO claim_templates(kind, source, payload_ref,"
                " template_hash, confidence, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (kind, source, payload_ref, thash, confidence, _now_iso()),
            )
            template_id = cur.lastrowid

        cur = self.conn.execute(
            "INSERT INTO hypotheses(template_id, parent_id, depth, status, subject,"
            " verifier_result_ref, verdict_at, created_at, created_in_stage)"
            " VALUES (?, ?, ?, 'pending', ?, NULL, NULL, ?, ?)",
            (template_id, parent_id, depth, subject, _now_iso(), created_in_stage),
        )
        hyp_id = cur.lastrowid

        if anchors:
            self.conn.executemany(
                "INSERT OR IGNORE INTO hyp_anchors(hyp_id, trace_idx, pc) VALUES (?, ?, ?)",
                [(hyp_id, idx, pc) for idx, pc in anchors],
            )
        if tags:
            self.conn.executemany(
                "INSERT OR IGNORE INTO hyp_tags(hyp_id, axis, value) VALUES (?, ?, ?)",
                [(hyp_id, axis, value) for axis, value in tags],
            )
        # commit at the end so a batched .add_many is possible later if needed
        self.conn.commit()
        return hyp_id  # type: ignore[return-value]

    def add_anchor(self, hyp_id: int, trace_idx: int, pc: int) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO hyp_anchors(hyp_id, trace_idx, pc) VALUES (?, ?, ?)",
            (hyp_id, trace_idx, pc),
        )
        self.conn.commit()

    def add_tag(self, hyp_id: int, axis: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO hyp_tags(hyp_id, axis, value) VALUES (?, ?, ?)",
            (hyp_id, axis, value),
        )
        self.conn.commit()

    def add_dependency(self, from_hyp_id: int, to_hyp_id: int, kind: str = "supports") -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO hyp_dependencies(from_hyp_id, to_hyp_id, kind)"
            " VALUES (?, ?, ?)",
            (from_hyp_id, to_hyp_id, kind),
        )
        self.conn.commit()

    def mark_verdict(
        self,
        hyp_id: int,
        verdict: str,
        verifier_result: dict[str, Any],
    ) -> None:
        status = {"pass": "passed", "fail": "failed", "inconclusive": "pending"}[verdict]
        result_ref = upsert_payload(self.conn, verifier_result)
        self.conn.execute(
            "UPDATE hypotheses SET status = ?, verifier_result_ref = ?, verdict_at = ?"
            " WHERE id = ?",
            (status, result_ref, _now_iso(), hyp_id),
        )
        self.conn.commit()

    # --------------------------------- READ ---------------------------------

    def get(self, hyp_id: int) -> HypNode:
        row = self.conn.execute(
            "SELECT h.id, h.parent_id, h.depth, h.status, h.subject,"
            " t.kind, t.source, t.payload_ref, t.confidence"
            " FROM hypotheses h JOIN claim_templates t ON h.template_id = t.id"
            " WHERE h.id = ?",
            (hyp_id,),
        ).fetchone()
        if row is None:
            raise KeyError(hyp_id)
        return self._row_to_node(row)

    def _row_to_node(self, row: tuple) -> HypNode:
        (hid, parent_id, depth, status, subject, kind, source, payload_ref, conf) = row
        payload_row = self.conn.execute(
            "SELECT payload FROM hyp_payloads WHERE content_hash = ?", (payload_ref,),
        ).fetchone()
        payload = json.loads(payload_row[0]) if payload_row else {}
        return HypNode(
            id=hid, parent_id=parent_id, depth=depth, status=status,
            kind=kind, source=source, subject=subject,
            payload=payload, confidence=conf,
        )

    def next_pending_sibling(self, hyp_id: int) -> HypNode | None:
        """Highest-confidence pending sibling (DFS ordering, DECISIONS D-004)."""
        cur_parent = self.conn.execute(
            "SELECT parent_id FROM hypotheses WHERE id = ?", (hyp_id,),
        ).fetchone()
        if cur_parent is None:
            return None
        parent_id = cur_parent[0]
        row = self.conn.execute(
            "SELECT h.id, h.parent_id, h.depth, h.status, h.subject,"
            " t.kind, t.source, t.payload_ref, t.confidence"
            " FROM hypotheses h JOIN claim_templates t ON h.template_id = t.id"
            " WHERE h.parent_id IS ? AND h.id != ? AND h.status = 'pending'"
            " ORDER BY t.confidence DESC NULLS LAST, h.id ASC LIMIT 1",
            (parent_id, hyp_id),
        ).fetchone()
        return self._row_to_node(row) if row else None

    def descendants_of(self, hyp_id: int) -> list[int]:
        """BFS over parent_id tree."""
        out: list[int] = []
        stack = [hyp_id]
        seen: set[int] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur != hyp_id:
                out.append(cur)
            for r in self.conn.execute(
                "SELECT id FROM hypotheses WHERE parent_id = ?", (cur,),
            ).fetchall():
                stack.append(r[0])
        return out

    def downstream_of(self, hyp_id: int) -> list[int]:
        """BFS over hyp_dependencies (NOT parent_id) — explicit semantic graph."""
        out: list[int] = []
        stack = [hyp_id]
        seen: set[int] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur != hyp_id:
                out.append(cur)
            for r in self.conn.execute(
                "SELECT to_hyp_id FROM hyp_dependencies WHERE from_hyp_id = ?", (cur,),
            ).fetchall():
                stack.append(r[0])
        return out

    def query(
        self,
        *,
        status: str | None = None,
        kind: str | None = None,
        source: str | None = None,
        tag: tuple[str, str] | None = None,        # (axis, value)
        anchor_trace_idx: int | None = None,
        anchor_pc: int | None = None,
        limit: int | None = None,
    ) -> list[HypNode]:
        """Multi-axis query. All non-None filters are AND-ed."""
        sql = (
            "SELECT DISTINCT h.id, h.parent_id, h.depth, h.status, h.subject,"
            " t.kind, t.source, t.payload_ref, t.confidence"
            " FROM hypotheses h JOIN claim_templates t ON h.template_id = t.id"
        )
        wheres: list[str] = []
        args: list[Any] = []
        if tag is not None:
            sql += " JOIN hyp_tags g ON g.hyp_id = h.id"
            wheres.append("g.axis = ? AND g.value = ?")
            args.extend(tag)
        if anchor_trace_idx is not None:
            sql += " JOIN hyp_anchors a1 ON a1.hyp_id = h.id"
            wheres.append("a1.trace_idx = ?")
            args.append(anchor_trace_idx)
        if anchor_pc is not None:
            sql += " JOIN hyp_anchors a2 ON a2.hyp_id = h.id"
            wheres.append("a2.pc = ?")
            args.append(anchor_pc)
        if status is not None:
            wheres.append("h.status = ?")
            args.append(status)
        if kind is not None:
            wheres.append("t.kind = ?")
            args.append(kind)
        if source is not None:
            wheres.append("t.source = ?")
            args.append(source)
        if wheres:
            sql += " WHERE " + " AND ".join(wheres)
        sql += " ORDER BY h.depth ASC, h.id ASC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = self.conn.execute(sql, args).fetchall()
        return [self._row_to_node(r) for r in rows]

    def anchors_of(self, hyp_id: int) -> list[tuple[int, int]]:
        return [
            (r[0], r[1]) for r in self.conn.execute(
                "SELECT trace_idx, pc FROM hyp_anchors WHERE hyp_id = ? ORDER BY trace_idx",
                (hyp_id,),
            ).fetchall()
        ]

    def tags_of(self, hyp_id: int) -> list[tuple[str, str]]:
        return [
            (r[0], r[1]) for r in self.conn.execute(
                "SELECT axis, value FROM hyp_tags WHERE hyp_id = ?", (hyp_id,),
            ).fetchall()
        ]

    # --------------------------------- ABANDON ---------------------------------

    def abandon_subtree(self, hyp_id: int) -> None:
        """Mark this node + descendants as abandoned. They stay in the table;
        the archival job (store.archive_subtree) can later move them out."""
        ids = [hyp_id] + self.descendants_of(hyp_id)
        if not ids:
            return
        ids_csv = ",".join(str(i) for i in ids)
        self.conn.execute(f"UPDATE hypotheses SET status = 'abandoned' WHERE id IN ({ids_csv})")
        self.conn.commit()
