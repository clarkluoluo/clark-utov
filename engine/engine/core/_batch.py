"""Core mixin: preprocess batches, discard, and stuck statistics."""
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


class _CoreBatchMixin:
    """Core methods: preprocess batches, discard, and stuck statistics (split from the monolithic Core)."""
    # layer-1/2 step). Default = full chain in dependency order.
    PREPROCESS_PASS_MAP = {
        "plugin":    "verify_and_promote_plugin_findings",
        "binop":     "verify_and_promote_handler_binops",
        "unary":     "verify_and_promote_handler_unaries",
        "imm":       "verify_and_promote_handler_imm_binops",
        "ext":       "verify_and_promote_handler_extended_binops",
        "bfx":       "verify_and_promote_handler_bfx",
        "ch":        "verify_and_promote_handler_ch_idioms",
        "maj":       "verify_and_promote_handler_maj_idioms",
        "triton":    "verify_and_promote_triton_simplifications",
        "sigma":     "verify_and_promote_sigma_idioms",
        "algorithm": "verify_and_promote_algorithm_templates",
        "rescan":    "self_rescan_missing_anchors",
    }
    PREPROCESS_DEFAULT_ORDER = [
        "plugin", "binop", "unary", "imm", "ext", "bfx", "ch", "maj",
        "triton", "sigma", "algorithm", "rescan",
    ]

    def preprocess_batch(
        self,
        *,
        passes: list[str] | None = None,
    ) -> dict[str, Any]:
        """One-call deterministic batch with selectable pass list (0527).

        Args:
            passes: subset of canonical pass names. None = full chain in
                    dependency order (plugin → binop → unary → imm → ext →
                    bfx → ch → triton → sigma → algorithm). Unknown names
                    raise ValueError. Order in the list dictates execution
                    order — caveat: the layer-1 (sigma) and layer-2
                    (algorithm) passes consume layer-0 outputs and only
                    work if their dependencies ran earlier in the chain.

        Returns:
            {
              "batch_id":         short uuid,
              "ran":              [pass_names in execution order],
              "results":          {pass_name: pass_summary_dict},
              "totals":           {
                  "promoted":            int,
                  "by_source":           {source: count},
                  "by_kind":             {kind: count},
                  "matched_algorithms":  [str],
              },
              "next_step_hints":  [str],   # agent UX hints
            }

        Every finding promoted while this call runs carries the returned
        batch_id (via Core._current_batch_id picked up by promote_to_finding).
        Agents can later call discard_batch(batch_id, ...) to bulk-fail any
        subset they reject — leaves an audit trail per hyp.
        """
        import uuid

        if passes is None:
            passes = list(self.PREPROCESS_DEFAULT_ORDER)
        bad = [p for p in passes if p not in self.PREPROCESS_PASS_MAP]
        if bad:
            raise ValueError(
                f"unknown preprocess pass names: {bad}; "
                f"valid: {sorted(self.PREPROCESS_PASS_MAP)}"
            )

        batch_id = uuid.uuid4().hex[:12]
        self._current_batch_id = batch_id
        results: dict[str, Any] = {}
        try:
            for pname in passes:
                method = getattr(self, self.PREPROCESS_PASS_MAP[pname])
                try:
                    results[pname] = method()
                except Exception as e:
                    results[pname] = {"error": f"{type(e).__name__}: {e}"}
        finally:
            self._current_batch_id = None

        f_conn = open_findings_db(self.work)
        try:
            row_total = f_conn.execute(
                "SELECT COUNT(*) FROM findings WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()[0]
            by_source = {r[0]: r[1] for r in f_conn.execute(
                "SELECT source, COUNT(*) FROM findings "
                "WHERE batch_id = ? GROUP BY source",
                (batch_id,),
            ).fetchall()}
            by_kind = {r[0]: r[1] for r in f_conn.execute(
                "SELECT kind, COUNT(*) FROM findings "
                "WHERE batch_id = ? GROUP BY kind",
                (batch_id,),
            ).fetchall()}
            algo_rows = f_conn.execute(
                "SELECT subject FROM findings "
                "WHERE batch_id = ? AND kind IN ('algorithm_hyp','algorithm_identified')",
                (batch_id,),
            ).fetchall()
        finally:
            f_conn.close()
        matched_algos = [r[0] for r in algo_rows]

        # FEATURE-REQUEST-1 auto-emit: drop a pseudocode.md alongside
        # findings.sqlite whenever at least one algorithm_identified
        # finding exists. Best-effort; never raises.
        emit_path = None
        if matched_algos:
            from ..emitter import emit_to_run_dir
            emit_path = emit_to_run_dir(self.work.root)

        return {
            "batch_id":         batch_id,
            "ran":              passes,
            "results":          results,
            "totals": {
                "promoted":           row_total,
                "by_source":          by_source,
                "by_kind":            by_kind,
                "matched_algorithms": matched_algos,
            },
            "next_step_hints":  self._preprocess_batch_hints(
                results, by_source, matched_algos),
            "pseudocode_path":  str(emit_path) if emit_path else None,
        }

    def _preprocess_batch_hints(
        self,
        results: dict[str, Any],
        by_source: dict[str, int],
        matched_algos: list[str],
    ) -> list[str]:
        """Heuristic next-step hints for the agent. Pure observation of the
        result dict — no recommendation the agent can't independently
        derive from the same data. Each hint is one short line."""
        hints: list[str] = []
        plugin_n = (results.get("plugin") or {}).get("promoted", 0)
        fold_n   = (results.get("sigma")  or {}).get("promoted", 0)
        triton_r = results.get("triton")  or {}

        if matched_algos:
            hints.append(
                f"algorithm_hyp: {', '.join(matched_algos)} (HYPOTHESIS — "
                f"pre-oracle-closure, carries a LOCAL_CLOSURE_ONLY trap; not a final "
                f"identification). Confidence + anchors_seen + the closure trap are in "
                f"the finding's payload; promote to algorithm_identified only on "
                f"whole-case oracle closure (sink confirmed + provenance closed + "
                f"multi-input parity EXACT)."
            )
        elif plugin_n > 0 and fold_n == 0:
            hints.append(
                f"plugin fingerprints hit ({plugin_n}) but no σ/Σ fold "
                f"idiom matched. Trace probably doesn't cover the round "
                f"function — check trace bounds (entry/exit PC) vs the "
                f"library's actual algorithm entry."
            )
        elif plugin_n == 0 and fold_n == 0 and not matched_algos:
            hints.append(
                "no algorithmic anchor found (no plugin fingerprint, no "
                "fold idiom, no algorithm fit). Call `stuck_statistics` to "
                "see what shapes remain and decide whether `--mode "
                "aggressive` with an LLM backend is worth the spend."
            )

        if triton_r and triton_r.get("skipped_reason"):
            hints.append(
                "triton pass skipped: "
                + str(triton_r.get("skipped_reason"))
            )
        elif triton_r:
            mm = triton_r.get("model_mismatch", 0)
            checked = triton_r.get("checked", 0)
            if mm and (mm > max(checked, 1) * 2):
                hints.append(
                    f"triton model_mismatch high ({mm} skipped vs "
                    f"{checked} verified). Most non-layer-0 instructions "
                    f"touch status flags / memory Triton can't model in "
                    f"fresh-context mode; not a verifier failure, just "
                    f"unmodellable instructions."
                )

        if not hints:
            hints.append(
                "batch produced no notable anchors. Call "
                "`stuck_statistics` for the next decision step."
            )
        return hints

    def discard_batch(
        self,
        batch_id: str,
        *,
        sources: list[str] | None = None,
        kinds: list[str] | None = None,
        reason: str = "agent discarded preprocess batch",
        actor: str = "agent",
    ) -> dict[str, Any]:
        """Bulk-fail the hypotheses behind a preprocess batch (留 audit).

        PLAN §1.1 says the ledger is append-only — we don't DELETE finding
        rows. discard_batch calls override_verdict("fail", ...) on every
        hypothesis whose finding row carries the given batch_id (optionally
        filtered by source / kind). Each override writes an `interventions`
        audit row, so the full promote → discard cycle stays auditable.

        Args:
            batch_id: tag returned by preprocess_batch.
            sources: optional source filter (e.g. ["s5_triton"] to discard
                     only Triton-net findings from this batch, keep layer-0).
            kinds:   optional kind filter (e.g. ["fold_idiom"] to discard
                     layer-1 findings but keep their constituent layer-0).
            reason:  stored in the audit trail.
            actor:   "agent" by default.

        Returns: {batch_id, candidate_count, discarded, hyp_ids, errors}
        """
        f_conn = open_findings_db(self.work)
        try:
            sql = (
                "SELECT origin_hyp_id, kind, source FROM findings "
                "WHERE batch_id = ? AND origin_hyp_id IS NOT NULL"
            )
            args: list[Any] = [batch_id]
            if sources:
                sql += " AND source IN (" + ",".join("?" * len(sources)) + ")"
                args += list(sources)
            if kinds:
                sql += " AND kind IN (" + ",".join("?" * len(kinds)) + ")"
                args += list(kinds)
            rows = f_conn.execute(sql, args).fetchall()
        finally:
            f_conn.close()

        discarded: list[int] = []
        errors: dict[str, str] = {}
        for hyp_id, _kind, _source in rows:
            try:
                self.override_verdict(
                    int(hyp_id), "fail", reason=reason, actor=actor,
                )
                discarded.append(int(hyp_id))
            except Exception as e:
                errors[str(hyp_id)] = f"{type(e).__name__}: {e}"

        return {
            "batch_id":        batch_id,
            "candidate_count": len(rows),
            "discarded":       len(discarded),
            "hyp_ids":         discarded,
            "errors":          errors,
        }

    def stuck_statistics(self, *, max_points: int | None = None,
                          cluster_gap: int = 0x40) -> dict[str, Any]:
        """Group S5 stuck points by mnemonic / verifiable_shape / pc_cluster
        (0526Plan B2). agent-user-advice §3.2 turns the 12K-item flat list
        into a structured view an agent can read at a glance.

        Args:
            max_points: cap the scan at N stuck points; None = no cap.
            cluster_gap: max byte gap that still counts as one PC cluster
                         (default 0x40 = 16 instructions).
        """
        import re as _re

        from ..orchestrators.script_mode import _find_stuck_points

        stuck = _find_stuck_points(self, max_points=max_points)
        total = len(stuck)

        _reg_re = _re.compile(r"^[wx]\d+$|^wzr$|^xzr$")
        _ext_kinds = {"lsl", "lsr", "asr", "ror",
                      "sxtw", "sxth", "sxtb", "uxtw", "uxth", "uxtb"}
        _binop_mnem = {"eor", "and", "orr", "add", "sub", "mul",
                       "lsl", "lsr", "ror", "asr"}
        _unary_mnem = {"mov", "mvn", "neg", "sxtw", "uxtw", "rev", "clz"}
        _bfx_mnem = {"ubfx", "sbfx", "bfm", "ubfm", "sbfm"}

        by_mnem: dict[str, int] = {}
        by_shape = {
            "reg_reg_reg_binop":  0,
            "reg_imm_binop":      0,
            "extended_register":  0,
            "unary":              0,
            "bitfield_extract":   0,
            "memory_load":        0,
            "memory_store":       0,
            "branch":             0,
            "fp_neon":            0,
            "other":              0,
        }

        for s in stuck:
            snippet = (s.snippet or "").strip()
            toks = [t for t in _re.split(r"[,\s]+", snippet) if t]
            if not toks:
                by_shape["other"] += 1
                continue
            m = toks[0].lower()
            by_mnem[m] = by_mnem.get(m, 0) + 1

            if m.startswith(("ldr", "ldur", "ldp", "ldnp", "ldax", "ldar")):
                by_shape["memory_load"] += 1
            elif m.startswith(("str", "stur", "stp", "stnp", "stlr", "stxr")):
                by_shape["memory_store"] += 1
            elif m.startswith(("b.", "br", "bl", "cb", "tb")) or m in ("b", "ret"):
                by_shape["branch"] += 1
            elif m.startswith(("f", "v")) or m in ("fmov", "fadd", "fsub", "fmul"):
                by_shape["fp_neon"] += 1
            elif m in _bfx_mnem:
                by_shape["bitfield_extract"] += 1
            elif (m in _binop_mnem and len(toks) == 4
                  and all(_reg_re.match(t) for t in toks[1:4])):
                by_shape["reg_reg_reg_binop"] += 1
            elif (m in _binop_mnem and len(toks) in (5, 6)
                  and len(toks) >= 5 and toks[4].lower() in _ext_kinds):
                by_shape["extended_register"] += 1
            elif (m in _binop_mnem and len(toks) == 4 and toks[3].startswith("#")):
                by_shape["reg_imm_binop"] += 1
            elif m in _unary_mnem and len(toks) == 3:
                by_shape["unary"] += 1
            else:
                by_shape["other"] += 1

        # by_pc_cluster
        pc_set: set[int] = set()
        for s in stuck:
            if s.instr_idx is not None and 0 <= s.instr_idx < len(self._items):
                pc_set.add(self._items[s.instr_idx].pc)
        pcs_sorted = sorted(pc_set)
        clusters: list[dict[str, Any]] = []
        if pcs_sorted:
            cur_lo = cur_hi = pcs_sorted[0]
            cur_count = 1
            for pc in pcs_sorted[1:]:
                if pc - cur_hi <= cluster_gap:
                    cur_hi = pc
                    cur_count += 1
                else:
                    clusters.append({
                        "lo":    f"0x{cur_lo:x}",
                        "hi":    f"0x{cur_hi:x}",
                        "count": cur_count,
                    })
                    cur_lo = cur_hi = pc
                    cur_count = 1
            clusters.append({
                "lo":    f"0x{cur_lo:x}",
                "hi":    f"0x{cur_hi:x}",
                "count": cur_count,
            })

        return {
            "total":               total,
            "by_mnemonic":         dict(sorted(by_mnem.items(), key=lambda kv: -kv[1])),
            "by_verifiable_shape": by_shape,
            "by_pc_cluster":       clusters,
        }

