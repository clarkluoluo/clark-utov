"""Core mixin: hypotheses, findings, interventions, and run control."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..conformance import (
    ConformanceReport,
    require_pass_or_die,
    run_conformance,
    write_report,
)
from ..hyp_tree import HypNode, HypTree
from ..runner_client import (
    JsonlTraceReader,
    RunnerAdapter,
    TraceReader,
    UnidbgTextTraceReader,
)
from ..store import (
    WorkDir, _now_iso,
    archive_subtree as _archive_subtree,
    log_intervention as _log_intervention,
    open_findings_db, open_hypotheses_db,
    read_interventions as _read_interventions,
    read_payload,
)
from ..stages import (
    s0_5_normalize, s1_segment, s1b_fingerprint, s2_dedupe,
    s3_triton, s4_slice, s5_simplify, s6_taint,
)
from ..types import Instruction, TargetMeta
from ..verifier import Verifier

from ._base import *  # noqa: F401,F403
from ._base import _STAGES, _algorithm_hyp_trap, _run_io_equivalence  # noqa: F401


class _CoreHypMixin:
    """Core methods: hypotheses, findings, interventions, and run control (split from the monolithic Core)."""
    def submit_hypothesis(
        self, *,
        kind: str, subject: str, payload: dict[str, Any],
        confidence: float | None, parent_id: int | None = None,
        source: str = "agent",
        anchors: list[tuple[int, int]] | None = None,
        tags: list[tuple[str, str]] | None = None,
    ) -> int:
        conn = open_hypotheses_db(self.work)
        try:
            return HypTree(conn).add(
                parent_id=parent_id, kind=kind, subject=subject,
                payload=payload, confidence=confidence,
                source=source, anchors=anchors, tags=tags,
            )
        finally:
            conn.close()

    def get_hypotheses(
        self, *,
        status: str | None = None,
        kind: str | None = None,
        source: str | None = None,
        tag: tuple[str, str] | None = None,
        anchor_trace_idx: int | None = None,
        anchor_pc: int | None = None,
        limit: int | None = None,
    ) -> list[HypNode]:
        """Return hypotheses matching all provided filters (multi-axis)."""
        conn = open_hypotheses_db(self.work)
        try:
            return HypTree(conn).query(
                status=status, kind=kind, source=source, tag=tag,
                anchor_trace_idx=anchor_trace_idx, anchor_pc=anchor_pc,
                limit=limit,
            )
        finally:
            conn.close()

    def get_findings(
        self, *,
        source: str | None = None,
        stage: str | None = None,
        kind: str | None = None,
        subject_like: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Query findings.sqlite with optional filters (0526Plan B1).

        Returns one dict per row with keys: id, stage, kind, subject,
        source, verifier_strategy, verified_at, origin_hyp_id, payload_ref.
        """
        sql = (
            "SELECT id, stage, kind, subject, source, verifier_strategy,"
            " verified_at, origin_hyp_id, payload_ref FROM findings"
        )
        clauses, args = [], []
        if source is not None:
            clauses.append("source = ?")
            args.append(source)
        if stage is not None:
            clauses.append("stage = ?")
            args.append(stage)
        if kind is not None:
            clauses.append("kind = ?")
            args.append(kind)
        if subject_like is not None:
            clauses.append("subject LIKE ?")
            args.append(subject_like)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id ASC"
        if limit is not None:
            sql += " LIMIT ?"
            args.append(int(limit))

        conn = open_findings_db(self.work)
        try:
            rows = conn.execute(sql, args).fetchall()
        finally:
            conn.close()
        keys = ("id", "stage", "kind", "subject", "source",
                "verifier_strategy", "verified_at", "origin_hyp_id",
                "payload_ref")
        return [dict(zip(keys, r)) for r in rows]

    # --- driver mode lifecycle (D-021) ---

    def checkpoint(self) -> None:
        """Flush state. Idempotent. Writes session.json + updated meta.json
        so the other driver mode can pick up exactly where we paused."""
        self._save_session()
        path = self.work.root / "meta.json"
        meta = json.loads(path.read_text()) if path.exists() else {}
        meta.update({
            "driver_mode": self.config.driver_mode,
            "run_id":      self.work.run_id,
            "stages_done": self.work.read_stage_state(),
            "session_keys": sorted(self.session.keys()),
        })
        path.write_text(json.dumps(meta, indent=2))

    def pause(self, reason: str, hint: dict[str, Any] | None = None) -> None:
        """Cooperative pause for mode switch (D-021)."""
        self.checkpoint()
        path = self.work.root / "meta.json"
        meta = json.loads(path.read_text())
        meta["paused"] = True
        meta["pause_reason"] = reason
        meta["next_action_hint"] = hint or {}
        path.write_text(json.dumps(meta, indent=2))

    # --- intervention API (PLAN §15 — agent operability + 留痕) ---

    def _audit(self, *, actor: str, action: str, target_table: str | None = None,
               target_id: Any = None, before: dict | None = None,
               after: dict | None = None, reason: str | None = None) -> int:
        conn = open_hypotheses_db(self.work)
        try:
            return _log_intervention(
                conn, actor=actor, action=action,
                target_table=target_table, target_id=target_id,
                before=before, after=after, reason=reason,
            )
        finally:
            conn.close()

    def override_verdict(self, hyp_id: int, new_verdict: str, *,
                         reason: str, actor: str = "agent") -> None:
        """Manually flip a verifier verdict. Logged."""
        if new_verdict not in ("pass", "fail", "inconclusive"):
            raise ValueError(f"invalid verdict: {new_verdict}")
        before = self._hyp_snapshot(hyp_id)
        conn = open_hypotheses_db(self.work)
        try:
            from ..hyp_tree import HypTree
            HypTree(conn).mark_verdict(hyp_id, new_verdict,
                                       {"override_by": actor, "reason": reason})
        finally:
            conn.close()
        after = self._hyp_snapshot(hyp_id)
        self._audit(actor=actor, action="override_verdict",
                    target_table="hypotheses", target_id=hyp_id,
                    before=before, after=after, reason=reason)

    def force_status(self, hyp_id: int, new_status: str, *,
                     reason: str, actor: str = "agent") -> None:
        """Manually flip a hypothesis status (e.g. revive an abandoned node)."""
        valid = {"pending", "verifying", "passed", "failed", "abandoned"}
        if new_status not in valid:
            raise ValueError(f"invalid status: {new_status}")
        before = self._hyp_snapshot(hyp_id)
        conn = open_hypotheses_db(self.work)
        try:
            conn.execute("UPDATE hypotheses SET status = ? WHERE id = ?",
                         (new_status, hyp_id))
            conn.commit()
        finally:
            conn.close()
        after = self._hyp_snapshot(hyp_id)
        self._audit(actor=actor, action="force_status",
                    target_table="hypotheses", target_id=hyp_id,
                    before=before, after=after, reason=reason)

    def localize_divergence(
        self,
        good_input: bytes,
        bad_input:  bytes,
        *,
        resync_look_ahead: int = 200,
    ) -> dict[str, Any]:
        """capability_request.md §P1-3: first-class differential localiser.

        Run the runner against two inputs (good vs. bad), compute the
        first divergent instruction, and return ranked candidate
        hypotheses. The caller can then ``submit_hypothesis`` from the
        returned payloads (their shapes follow the standard wire form).

        Falls back to the engine's current trace_reader if the runner
        does not implement ``get_trace`` — in that case ``bad_input``
        runs through the runner only if Live mode is available; File-
        mode adapters raise ``NotImplementedError`` and the agent uses
        the static-trace path (or supplies pre-recorded traces via
        :func:`engine.localize.localize_divergence` directly).
        """
        from ..localize import localize_divergence as _ld

        good_path = self.rerun.get_trace(
            good_input,
            self.config.target_meta.algo_entry_pc,
            self.config.target_meta.algo_exit_pc,
        )
        bad_path = self.rerun.get_trace(
            bad_input,
            self.config.target_meta.algo_entry_pc,
            self.config.target_meta.algo_exit_pc,
        )
        good_items = list(JsonlTraceReader(good_path))
        bad_items = list(JsonlTraceReader(bad_path))
        result = _ld(good_items, bad_items, resync_look_ahead=resync_look_ahead)
        return {
            "divergence":
                None if result.divergence is None else {
                    "kind":     result.divergence.kind,
                    "good_idx": result.divergence.good_idx,
                    "bad_idx":  result.divergence.bad_idx,
                    "pc":       f"0x{result.divergence.pc:x}",
                    "good":     result.divergence.good,
                    "bad":      result.divergence.bad,
                },
            "candidates": [
                {
                    "rank":       c.rank,
                    "kind":       c.kind,
                    "subject":    c.subject,
                    "payload":    c.payload,
                    "confidence": c.confidence,
                    "rationale":  c.rationale,
                }
                for c in result.candidates
            ],
            "resync_at": result.resync_at,
        }

    def invalidate_cluster(
        self,
        parent_finding_id: int,
        *,
        reason: str,
        actor: str = "agent",
    ) -> dict[str, Any]:
        """capability_request.md §P1-4: cascade-invalidate a cluster.

        Marks the parent's origin hyp as ``failed`` and flips every
        member's origin hyp back to ``pending`` so the verifier can
        re-evaluate them under fresh evidence. All flips share one
        cascade id in their audit ``reason`` so the operation can be
        traced as a single event.

        Returns a dict containing ``cascade_id``, ``parent_hyp_id``,
        ``member_hyp_ids`` (the ones that were flipped to pending), and
        ``skipped_member_finding_ids`` (members with no origin_hyp).
        """
        from ..cluster import (
            CascadeReport,
            _origin_hyp_id_for_finding,
            collect_member_origin_hyp_ids,
            make_cascade_id,
        )

        cascade_id = make_cascade_id()
        fc = open_findings_db(self.work)
        try:
            parent_hyp_id = _origin_hyp_id_for_finding(fc, parent_finding_id)
            hyp_ids, skipped = collect_member_origin_hyp_ids(fc, parent_finding_id)
        finally:
            fc.close()

        parent_was_already_failed = False
        if parent_hyp_id is not None:
            snap_before = self._hyp_snapshot(parent_hyp_id)
            current_status = (snap_before or {}).get("status")
            if current_status == "failed":
                parent_was_already_failed = True
            else:
                self.force_status(
                    parent_hyp_id, "failed",
                    reason=f"{cascade_id} parent invalidated: {reason}",
                    actor=actor,
                )

        flipped: list[int] = []
        for hid in hyp_ids:
            snap = self._hyp_snapshot(hid)
            if not snap:
                continue
            if snap.get("status") == "pending":
                # Already where we'd send it — skip the write but still
                # log so the audit shows the cascade touched it.
                flipped.append(hid)
                continue
            self.force_status(
                hid, "pending",
                reason=f"{cascade_id} cascade from parent {parent_finding_id}: {reason}",
                actor=actor,
            )
            flipped.append(hid)

        report = CascadeReport(
            cascade_id=cascade_id,
            parent_finding_id=parent_finding_id,
            parent_hyp_id=parent_hyp_id,
            member_hyp_ids=tuple(flipped),
            skipped_member_finding_ids=tuple(skipped),
            parent_origin_was_already_failed=parent_was_already_failed,
        )
        return {
            "cascade_id":                report.cascade_id,
            "parent_finding_id":         report.parent_finding_id,
            "parent_hyp_id":             report.parent_hyp_id,
            "member_hyp_ids":            list(report.member_hyp_ids),
            "skipped_member_finding_ids": list(report.skipped_member_finding_ids),
            "parent_was_already_failed": report.parent_origin_was_already_failed,
        }

    def inject_finding(self, *, kind: str, subject: str, payload: dict,
                       reason: str, actor: str = "agent",
                       verifier_strategy: str = "manual") -> int:
        """Inject a finding manually (without a hyp passing verifier).

        Injecting a `fold_idiom` or `algo_signature` finding auto-triggers
        `recompute_algorithm_fits()` so any existing `algorithm_identified`
        payload reflects the new anchor immediately (BUG_REPORT-6 #5).
        """
        from ..store import upsert_payload as _upsert
        fc = open_findings_db(self.work)
        try:
            payload_ref = _upsert(fc, payload)
            cur = fc.execute(
                "INSERT INTO findings(stage, kind, subject, payload_ref, verified_at,"
                " verifier_strategy, origin_hyp_id, source)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("manual_inject", kind, subject, payload_ref, _now_iso(),
                 verifier_strategy, None, "agent_override" if actor == "agent" else "manual_inject"),
            )
            fc.commit()
            finding_id = cur.lastrowid
        finally:
            fc.close()
        self._audit(actor=actor, action="inject_finding",
                    target_table="findings", target_id=finding_id,
                    after={"kind": kind, "subject": subject, "payload": payload},
                    reason=reason)
        if kind in ("fold_idiom", "algo_signature"):
            try:
                self.recompute_algorithm_fits()
            except Exception:
                # Refit is best-effort — never block an inject on a downstream
                # template error.
                pass
        return finding_id  # type: ignore[return-value]

    def add_tag_with_reason(self, hyp_id: int, axis: str, value: str, *,
                            reason: str, actor: str = "agent") -> None:
        before = self._hyp_snapshot(hyp_id)
        conn = open_hypotheses_db(self.work)
        try:
            from ..hyp_tree import HypTree
            HypTree(conn).add_tag(hyp_id, axis, value)
        finally:
            conn.close()
        self._audit(actor=actor, action="add_tag",
                    target_table="hyp_tags", target_id=hyp_id,
                    before=before,
                    after={"axis": axis, "value": value}, reason=reason)

    def add_dependency_with_reason(self, from_hyp_id: int, to_hyp_id: int,
                                    kind: str = "supports", *,
                                    reason: str, actor: str = "agent") -> None:
        conn = open_hypotheses_db(self.work)
        try:
            from ..hyp_tree import HypTree
            HypTree(conn).add_dependency(from_hyp_id, to_hyp_id, kind)
        finally:
            conn.close()
        self._audit(actor=actor, action="add_dependency",
                    target_table="hyp_dependencies",
                    target_id=f"{from_hyp_id}->{to_hyp_id}",
                    after={"from": from_hyp_id, "to": to_hyp_id, "kind": kind},
                    reason=reason)

    def resume_run(self, *, actor: str = "agent", reason: str = "agent resumed") -> None:
        """Clear paused state. Caller should then re-invoke pipeline."""
        path = self.work.root / "meta.json"
        meta = json.loads(path.read_text()) if path.exists() else {}
        before = {"paused": meta.get("paused", False),
                  "reason": meta.get("pause_reason")}
        meta["paused"] = False
        meta["pause_reason"] = None
        path.write_text(json.dumps(meta, indent=2))
        self._audit(actor=actor, action="resume",
                    target_table="meta",
                    before=before, after={"paused": False}, reason=reason)

    def list_interventions(self, *, limit: int = 100,
                            action: str | None = None) -> list[dict]:
        conn = open_hypotheses_db(self.work)
        try:
            return _read_interventions(conn, limit=limit, action=action)
        finally:
            conn.close()

    def _hyp_snapshot(self, hyp_id: int) -> dict | None:
        try:
            n = self._hyp_node(hyp_id)
        except KeyError:
            return None
        return {"id": n.id, "status": n.status, "subject": n.subject,
                "kind": n.kind, "confidence": n.confidence,
                "payload": n.payload}

    def _hyp_node(self, hyp_id: int):
        from ..hyp_tree import HypTree
        conn = open_hypotheses_db(self.work)
        try:
            return HypTree(conn).get(hyp_id)
        finally:
            conn.close()

    # --- rerun-from-stage cascade (PLAN §15 agent operability) ---

    _STAGE_ORDER = ("s1", "s1b", "s2", "s3", "s4", "s5", "s6")

    def rerun_from_stage(self, stage: str, *,
                         actor: str = "agent",
                         reason: str = "agent requested re-run") -> dict:
        """Reset stage_state and tear down derived state for `stage` and every
        downstream stage. After this returns, calling run_stage(stage) starts
        fresh.

        Cascade scope:
          - stage_state.json: remove stage and all downstream from done map
          - stage_outputs/sN*: delete output files for the cascade range
          - hypotheses with created_in_stage in cascade range: STATUS=abandoned
            (we don't delete — keep audit trail intact). Their findings get
            invalidated (origin_hyp_id matches → DELETE row, FK preserved).
        """
        if stage not in self._STAGE_ORDER:
            raise ValueError(f"unknown stage: {stage}; expected one of {self._STAGE_ORDER}")

        cascade = list(self._STAGE_ORDER[self._STAGE_ORDER.index(stage):])

        # 1) Snapshot before-state for audit
        stage_state_before = self.work.read_stage_state()

        # 2) stage_state.json — drop cascade stages
        new_state = {k: v for k, v in stage_state_before.items() if k not in cascade}
        self.work.stage_state_path.write_text(json.dumps(new_state, indent=2))

        # 3) stage_outputs/sN*.jsonl — delete
        files_deleted: list[str] = []
        for st in cascade:
            for p in (self.work.root / "stage_outputs").glob(f"{st}*"):
                try:
                    p.unlink()
                    files_deleted.append(p.name)
                except OSError:
                    pass

        # 4) hypotheses created in cascade → abandoned (kept for audit)
        conn = open_hypotheses_db(self.work)
        try:
            csv = ",".join(f"'{s}'" for s in cascade)
            row = conn.execute(
                f"SELECT id FROM hypotheses WHERE created_in_stage IN ({csv})"
            ).fetchall()
            cascade_hyp_ids = [r[0] for r in row]
            if cascade_hyp_ids:
                ids_csv = ",".join(str(i) for i in cascade_hyp_ids)
                conn.execute(
                    f"UPDATE hypotheses SET status = 'abandoned' WHERE id IN ({ids_csv})"
                )
            conn.commit()
        finally:
            conn.close()

        # 5) findings with origin_hyp_id pointing at abandoned hyps → DELETE
        findings_deleted = 0
        if cascade_hyp_ids:
            fc = open_findings_db(self.work)
            try:
                ids_csv = ",".join(str(i) for i in cascade_hyp_ids)
                cur = fc.execute(
                    f"DELETE FROM findings WHERE origin_hyp_id IN ({ids_csv})"
                )
                findings_deleted = cur.rowcount or 0
                fc.commit()
            finally:
                fc.close()

        # 6) Session — wipe stage-derived hints so feedback context restarts.
        # Keep algo_hints (cross-stage), drop fingerprint_anchor_idxs etc.
        for k in ("fingerprint_anchor_idxs", "stuck_points"):
            self.session.pop(k, None)
        self._save_session()

        # 7) Audit one big row
        self._audit(
            actor=actor, action="rerun_from_stage",
            target_table="stage_state", target_id=stage,
            before={"stage_state": stage_state_before},
            after={
                "stage_state": new_state,
                "cascade_stages": cascade,
                "stage_files_deleted": files_deleted,
                "hyps_abandoned": cascade_hyp_ids,
                "findings_deleted": findings_deleted,
            },
            reason=reason,
        )
        return {
            "cascade_stages": cascade,
            "stage_files_deleted": files_deleted,
            "hyps_abandoned_count": len(cascade_hyp_ids),
            "findings_deleted": findings_deleted,
        }

    # --- archival (D-027 item 6) ---

    def archive_abandoned(self, threshold: int = 100) -> dict[str, Any]:
        """If abandoned subtrees together exceed `threshold` rows, move them
        all into archived/hyps_<ts>.sqlite so the active DB stays slim."""
        conn = open_hypotheses_db(self.work)
        try:
            count_row = conn.execute(
                "SELECT COUNT(*) FROM hypotheses WHERE status = 'abandoned'"
            ).fetchone()
            count = count_row[0] if count_row else 0
            if count < threshold:
                return {"archived": 0, "remaining_abandoned": count}
            # Roots are abandoned hyps whose parent is NOT abandoned.
            roots = [r[0] for r in conn.execute(
                "SELECT h.id FROM hypotheses h"
                " WHERE h.status = 'abandoned'"
                "   AND (h.parent_id IS NULL"
                "        OR (SELECT status FROM hypotheses WHERE id = h.parent_id) != 'abandoned')"
            ).fetchall()]
            arch_path = _archive_subtree(conn, self.work, roots)
            return {"archived": count, "archive_path": str(arch_path)}
        finally:
            conn.close()

