"""Core mixin: algorithm/dataflow/static/mode verify-and-promote passes."""
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


class _CoreAlgorithmMixin:
    """Core methods: algorithm/dataflow/static/mode verify-and-promote passes (split from the monolithic Core)."""
    def verify_and_promote_algorithm_templates(self) -> dict[str, Any]:
        """Layer-2 algorithm-template fit (0526Plan E1.4 + E1.5).

        Aggregates the trace's layer-1 fold_idiom findings + layer-0 plugin
        fingerprint findings and matches them against a small algorithm
        template table. Each template lists a set of anchor names (idiom
        subjects + plugin fingerprint constants); a trace "fits" when the
        observed anchors clear the template's minimum-evidence threshold.

        IO equivalence is wired against the duck-typed runner (see
        `_run_io_equivalence`). 0527/BUG_REPORT-7 §C: this is no longer a
        class-name string match — the engine actually invokes runner.rerun
        with a NIST/FIPS canonical vector and compares output bytes.
        """
        from ..store import link_finding_group_members

        f_conn = open_findings_db(self.work)
        try:
            rows = f_conn.execute(
                "SELECT id, subject, kind FROM findings "
                "WHERE kind IN ('fold_idiom', 'algo_signature')"
            ).fetchall()
        finally:
            f_conn.close()

        observed_anchors: dict[str, list[int]] = {}
        for fid, subj, _kind in rows:
            name = subj.rsplit("@", 1)[0] if "@" in subj else subj
            observed_anchors.setdefault(name, []).append(fid)

        # BUG_REPORT-6 #5: a second invocation (e.g. after inject_finding adds
        # fresh σ/Σ folds) used to silently create a duplicate
        # `algorithm_identified` finding per algo. Skip algos that already
        # have a finding — the recompute path keeps their payload current.
        f_conn = open_findings_db(self.work)
        try:
            existing_algos = {r[0] for r in f_conn.execute(
                "SELECT subject FROM findings "
                "WHERE kind IN ('algorithm_hyp','algorithm_identified')"
            ).fetchall()}
        finally:
            f_conn.close()

        promoted = 0
        matched_algos: list[str] = []
        io_test_by_algo: dict[str, dict[str, Any]] = {}
        io_test_summary = "no algorithms matched"   # back-compat string field

        for algo, spec in ALGORITHM_TEMPLATES.items():
            if algo in existing_algos:
                continue
            anchors_seen = [a for a in spec["anchors"] if a in observed_anchors]
            if len(anchors_seen) < spec["min_unique_anchors"]:
                continue
            evidence_score = round(len(anchors_seen) / len(spec["anchors"]), 3)
            # §B: templates with `confidence_override` skip the fraction-of-
            # anchors formula — used when even a single STRONG anchor (e.g.
            # AES.Te0[0]) is enough to identify the algorithm.
            if "confidence_override" in spec:
                confidence = round(float(spec["confidence_override"]), 3)
            else:
                # Confidence: 0.50 floor + up to 0.35 from anchor coverage.
                # 0.85 ceiling is the structural-match-only cap; lifted only
                # when IO-equivalence (below) returns status=passed.
                confidence = round(0.50 + 0.35 * evidence_score, 3)

            io_result = _run_io_equivalence(
                self.rerun, self.config.target_meta, algo, spec,
            )
            io_test_by_algo[algo] = io_result
            io_test_summary = io_result.get("status", "skipped")
            # §C: passing IO test lifts the structural-only confidence cap.
            if io_result.get("status") == "passed":
                confidence = round(min(0.99, confidence + 0.10), 3)
            # Failure on a canonical vector contradicts the structural match
            # — keep the hypothesis but mark it; downstream consumers
            # decide whether to suppress.
            elif io_result.get("status") == "failed":
                confidence = round(max(0.0, confidence - 0.30), 3)

            try:
                hid = self.submit_hypothesis(
                    kind=ALGORITHM_HYP_KIND,
                    subject=algo,
                    payload={
                        "algorithm":        algo,
                        "anchors_seen":     anchors_seen,
                        "anchors_expected": spec["anchors"],
                        "evidence_score":   evidence_score,
                        "io_test":          io_result,
                        "reference_impl":   spec.get("reference_impl"),
                        # task 7②: a pre-oracle-closure algorithm guess MUST carry
                        # the explicit local-closure-trap marker (not silently a
                        # renamed "identified").
                        "closure":          _algorithm_hyp_trap(io_result),
                        "rationale": (
                            f"{len(anchors_seen)}/{len(spec['anchors'])} "
                            f"{algo} anchors present: {sorted(anchors_seen)}. "
                            f"IO-equivalence: {io_result.get('status')} — "
                            f"{io_result.get('detail')}. "
                            f"HYPOTHESIS (algorithm_hyp): pre-oracle-closure — not a "
                            f"final identification."
                        ),
                    },
                    confidence=confidence,
                    source="s5_algorithm_fit",
                    anchors=[],
                )
                conn_h = open_hypotheses_db(self.work)
                try:
                    HypTree(conn_h).mark_verdict(hid, "pass", {
                        "strategy":       "structural_anchor_set_match",
                        "evidence_score": evidence_score,
                    })
                finally:
                    conn_h.close()
                algo_finding_id = self.promote_to_finding(
                    hid, verifier_strategy="structural_anchor_set_match",
                    stage="s5-algorithm-fit",
                )
                promoted += 1
                matched_algos.append(algo)

                # Link constituent anchor findings into the algorithm group.
                f_conn = open_findings_db(self.work)
                try:
                    members: list[tuple[int, str | None]] = []
                    for anchor in anchors_seen:
                        for member_fid in observed_anchors[anchor]:
                            members.append((member_fid, anchor))
                    if members:
                        link_finding_group_members(
                            f_conn,
                            parent_finding_id=algo_finding_id,
                            idiom_name=algo,
                            members=members,
                        )
                finally:
                    f_conn.close()
            except (KeyError, ValueError):
                pass

        return {
            "stage":               "s5-algorithm-fit",
            "matched_algorithms":  matched_algos,
            "promoted":            promoted,
            "io_test":             io_test_summary,
            "io_test_by_algo":     io_test_by_algo,
        }

    # 0527/BUG_REPORT-6 #5: refit existing algorithm_identified findings
    # against the current pool of fold_idiom + algo_signature findings, so
    # `anchors_seen` / `evidence_score` don't drift stale after
    # inject_finding() adds fresh anchors. Also promotes new algorithms
    # that just crossed their min-evidence threshold.
    def recompute_algorithm_fits(self) -> dict[str, Any]:
        """Refresh `algorithm_identified` findings after the anchor pool changes.

        Walks current `fold_idiom` + `algo_signature` findings, recomputes
        each known template's anchor coverage, and:
          - **Updates** existing `algorithm_identified` payloads in-place when
            `anchors_seen` shifted (e.g. agent injected a missed σ₁ fold).
          - **Promotes** new algorithms that just crossed
            `min_unique_anchors`.

        Returns a per-algo summary including the before/after anchor delta.
        Safe to call repeatedly; idempotent when no anchors changed.
        """
        from ..store import read_payload as _read_payload
        from ..store import upsert_payload as _upsert_payload

        f_conn = open_findings_db(self.work)
        try:
            rows = f_conn.execute(
                "SELECT id, subject, kind FROM findings "
                "WHERE kind IN ('fold_idiom', 'algo_signature')"
            ).fetchall()
            observed: dict[str, list[int]] = {}
            for fid, subj, _k in rows:
                name = subj.rsplit("@", 1)[0] if "@" in subj else subj
                observed.setdefault(name, []).append(fid)

            existing: dict[str, tuple[int, str]] = {}
            for r in f_conn.execute(
                "SELECT id, subject, payload_ref FROM findings "
                "WHERE kind IN ('algorithm_hyp','algorithm_identified')"
            ).fetchall():
                existing[r[1]] = (r[0], r[2])
        finally:
            f_conn.close()

        updated_algos: dict[str, dict] = {}
        unchanged_algos: list[str] = []

        for algo, spec in ALGORITHM_TEMPLATES.items():
            if algo not in existing:
                continue
            anchors_seen = [a for a in spec["anchors"] if a in observed]
            if len(anchors_seen) < spec["min_unique_anchors"]:
                # Coverage dropped below threshold (rare, but possible if
                # findings were superseded). Leave the finding as-is rather
                # than deleting; just note it in the report.
                continue
            evidence_score = round(len(anchors_seen) / len(spec["anchors"]), 3)
            io_result = _run_io_equivalence(
                self.rerun, self.config.target_meta, algo, spec,
            )
            new_payload = {
                "algorithm":        algo,
                "anchors_seen":     anchors_seen,
                "anchors_expected": spec["anchors"],
                "evidence_score":   evidence_score,
                "io_test":          io_result,
                "reference_impl":   spec.get("reference_impl"),
                "closure":          _algorithm_hyp_trap(io_result),
                "rationale": (
                    f"{len(anchors_seen)}/{len(spec['anchors'])} "
                    f"{algo} anchors present: {sorted(anchors_seen)}. "
                    f"IO-equivalence: {io_result.get('status')} — "
                    f"{io_result.get('detail')}. "
                    f"HYPOTHESIS (algorithm_hyp): pre-oracle-closure — not a final "
                    f"identification."
                ),
            }
            fid, old_ref = existing[algo]
            f_conn = open_findings_db(self.work)
            try:
                try:
                    old_payload = _read_payload(f_conn, old_ref) if old_ref else {}
                except KeyError:
                    old_payload = {}
                old_anchors = old_payload.get("anchors_seen", [])
                if set(old_anchors) == set(anchors_seen):
                    unchanged_algos.append(algo)
                    continue
                new_ref = _upsert_payload(f_conn, new_payload)
                f_conn.execute(
                    "UPDATE findings SET payload_ref = ? WHERE id = ?",
                    (new_ref, fid),
                )
                f_conn.commit()
            finally:
                f_conn.close()
            updated_algos[algo] = {
                "anchors_before": sorted(old_anchors),
                "anchors_after":  sorted(anchors_seen),
                "evidence_score": evidence_score,
            }
            self._audit(
                actor="engine", action="recompute_algorithm_fit",
                target_table="findings", target_id=fid,
                before={"anchors_seen": sorted(old_anchors)},
                after={"anchors_seen": sorted(anchors_seen),
                       "evidence_score": evidence_score},
                reason="agent invoked recompute_algorithm_fits",
            )

        # Catch algorithms whose anchors just crossed the threshold. The
        # dedupe at the top of verify_and_promote_algorithm_templates makes
        # this a safe no-op for already-present algos.
        new_pass = self.verify_and_promote_algorithm_templates()

        return {
            "stage":          "s5-algorithm-refit",
            "updated":        updated_algos,
            "unchanged":      unchanged_algos,
            "newly_promoted": new_pass.get("matched_algorithms", []),
        }

    def self_rescan_missing_anchors(self) -> dict[str, Any]:
        """BR-8 #3: re-scan layer-1 / layer-0 matchers when an
        algorithm_identified finding still has missing anchors.

        Algorithm-fit promotes a hypothesis even when only some anchors are
        present (down to `min_unique_anchors`). The σ/Σ + Ch + Maj matchers
        all dedupe by PC / triple-key, so re-running them after the initial
        fit is a cheap second pass that may pick up shapes the original
        pass missed (e.g. a σ₁ that needed Phase 3's DFG-grouped scan but
        wasn't a hot spot earlier). After re-scanning we call
        recompute_algorithm_fits so the existing finding's payload reflects
        the now-complete anchor set.

        Idempotent: no missing anchors → no-op. Safe to call repeatedly.
        """
        from ..store import read_payload as _read_payload

        f_conn = open_findings_db(self.work)
        try:
            rows = f_conn.execute(
                "SELECT id, subject, payload_ref FROM findings "
                "WHERE kind IN ('algorithm_hyp','algorithm_identified')"
            ).fetchall()
        finally:
            f_conn.close()
        if not rows:
            return {
                "stage":          "s5-anchor-rescan",
                "checked":        0,
                "missing_before": {},
                "sigma_promoted": 0,
                "ch_promoted":    0,
                "maj_promoted":   0,
                "refit":          None,
                "still_missing":  {},
            }

        f_conn = open_findings_db(self.work)
        try:
            missing_by_algo: dict[str, list[str]] = {}
            for _fid, subj, ref in rows:
                try:
                    pl = _read_payload(f_conn, ref) if ref else {}
                except KeyError:
                    continue
                seen = set(pl.get("anchors_seen") or [])
                exp = list(pl.get("anchors_expected") or [])
                missing = [a for a in exp if a not in seen]
                if missing:
                    missing_by_algo[subj] = missing
        finally:
            f_conn.close()
        if not missing_by_algo:
            return {
                "stage":          "s5-anchor-rescan",
                "checked":        len(rows),
                "missing_before": {},
                "sigma_promoted": 0,
                "ch_promoted":    0,
                "maj_promoted":   0,
                "refit":          None,
                "still_missing":  {},
            }

        # Re-run the σ/Σ + Ch + Maj matchers. All three dedupe internally
        # (seen_triples / seen_pcs) so this is a cheap retry that only adds
        # new anchors. Phase 3 DFG-grouped scan inside the σ/Σ matcher is
        # the practical win for ILP / dst==input cases.
        sigma_res = self.verify_and_promote_sigma_idioms()
        ch_res    = self.verify_and_promote_handler_ch_idioms()
        maj_res   = self.verify_and_promote_handler_maj_idioms()
        refit     = self.recompute_algorithm_fits()

        f_conn = open_findings_db(self.work)
        try:
            rows = f_conn.execute(
                "SELECT id, subject, payload_ref FROM findings "
                "WHERE kind IN ('algorithm_hyp','algorithm_identified')"
            ).fetchall()
        finally:
            f_conn.close()
        f_conn = open_findings_db(self.work)
        try:
            still_missing: dict[str, list[str]] = {}
            for _fid, subj, ref in rows:
                try:
                    pl = _read_payload(f_conn, ref) if ref else {}
                except KeyError:
                    continue
                seen = set(pl.get("anchors_seen") or [])
                exp = list(pl.get("anchors_expected") or [])
                missing = [a for a in exp if a not in seen]
                if missing:
                    still_missing[subj] = missing
        finally:
            f_conn.close()

        return {
            "stage":          "s5-anchor-rescan",
            "checked":        len(missing_by_algo),
            "missing_before": missing_by_algo,
            "sigma_promoted": sigma_res.get("promoted", 0),
            "ch_promoted":    ch_res.get("promoted", 0),
            "maj_promoted":   maj_res.get("promoted", 0),
            "refit":          refit,
            "still_missing":  still_missing,
        }

    def dataflow_query(
        self,
        *,
        kind: str,
        input_reg: str | None = None,
        within_pc_range: tuple[int, int] | None = None,
        target_reg: str | None = None,
        dst_reg: str | None = None,
        from_pc: int | None = None,
        max_depth: int = 8,
        max_results: int = 256,
    ) -> list[dict[str, Any]]:
        """BR-8 #4: agent-facing trace queries that avoid grepping s3_dfg.jsonl.

        Supported `kind` values:
          - "rotations_on_input": every rotate/shift acting on `input_reg`
            (standalone `ror|lsr dst, x, #N` and the shifted-source operand
            of `eor dst, lhs, x, ror|lsr #N`).
          - "xor_chain_to":       eor-chain instructions that ultimately
            write `target_reg`, walked backwards through producers.
          - "producer_chain":     the producer chain of `dst_reg` starting
            at `from_pc` (or the latest writer if from_pc is None), depth
            bounded by `max_depth`.
          - "boolean_subgraph":   instructions matching boolean ops
            (and/orr/eor/bic/mvn) within `within_pc_range`, grouped by
            output register — first cut at the 8-bit TT helper described
            in BUG_REPORT-8 §2 (the truth-table reduction is left to the
            agent; this returns the candidate set).

        Filters:
          - within_pc_range: [lo_pc, hi_pc] (inclusive) — applied to every
            kind. Defaults to no filter.
          - max_results:      cap on returned rows (default 256).

        Returns a list of plain dicts with hex PCs (so the result is JSON-
        serialisable for the agent RPC).
        """
        import re as _re

        _reg_re = _re.compile(r"^[wx]\d+$|^wzr$|^xzr$")
        items = self._items

        def _split(s: str) -> list[str]:
            return [t for t in _re.split(r"[,\s]+", s.strip()) if t]

        def _parse_imm(tok: str) -> int | None:
            if not tok.startswith("#"):
                return None
            body = tok[1:]
            try:
                return int(body, 16) if body.lower().startswith("0x") else int(body)
            except ValueError:
                return None

        def _in_range(pc: int) -> bool:
            if within_pc_range is None:
                return True
            lo, hi = within_pc_range
            return lo <= pc <= hi

        def _row(ins) -> dict[str, Any]:
            return {
                "idx":        ins.idx,
                "pc":         f"0x{ins.pc:x}",
                "mnemonic":   ins.mnemonic,
                "regs_read":  {k: f"0x{v:x}" for k, v in ins.regs_read.items()},
                "regs_write": {k: f"0x{v:x}" for k, v in ins.regs_write.items()},
            }

        results: list[dict[str, Any]] = []

        if kind == "rotations_on_input":
            if not input_reg:
                raise ValueError("rotations_on_input requires input_reg")
            for ins in items:
                if not _in_range(ins.pc):
                    continue
                toks = _split(ins.mnemonic)
                op = toks[0].lower() if toks else ""
                if (op in ("ror", "lsr") and len(toks) == 4
                        and all(_reg_re.match(t) for t in toks[1:3])
                        and toks[2] == input_reg):
                    n = _parse_imm(toks[3])
                    if n is None:
                        continue
                    results.append({
                        **_row(ins), "kind": op, "amount": n,
                        "dst": toks[1], "form": "standalone",
                    })
                elif (op == "eor" and len(toks) == 6
                        and toks[4].lower() in ("ror", "lsr")
                        and toks[3] == input_reg
                        and all(_reg_re.match(t) for t in toks[1:4])):
                    n = _parse_imm(toks[5])
                    if n is None:
                        continue
                    results.append({
                        **_row(ins), "kind": toks[4].lower(), "amount": n,
                        "dst": toks[1], "form": "eor_shifted",
                    })
                if len(results) >= max_results:
                    break
            return results

        if kind == "xor_chain_to":
            if not target_reg:
                raise ValueError("xor_chain_to requires target_reg")
            # Walk backward: at each step find the latest eor that writes
            # the current target, recurse on its source registers, stop at
            # max_depth.
            seen_pcs: set[int] = set()
            queue: list[tuple[str, int]] = [(target_reg, 0)]
            # last_writer per reg up to each idx — precompute a single map.
            while queue and len(results) < max_results:
                reg, depth = queue.pop(0)
                if depth > max_depth:
                    continue
                writers = [ins for ins in items
                           if reg in ins.regs_write and ins.pc not in seen_pcs
                           and _in_range(ins.pc)]
                if not writers:
                    continue
                latest = writers[-1]
                toks = _split(latest.mnemonic)
                if not toks or toks[0].lower() != "eor":
                    continue
                seen_pcs.add(latest.pc)
                results.append({**_row(latest), "depth": depth})
                # Recurse on operands that look like register names.
                for src in toks[2:]:
                    if _reg_re.match(src):
                        queue.append((src, depth + 1))
            return results

        if kind == "producer_chain":
            if not dst_reg:
                raise ValueError("producer_chain requires dst_reg")
            # Pick a starting instruction: the latest writer of dst_reg
            # before from_pc (or the last writer overall).
            start_idx = None
            for ins in items:
                if dst_reg in ins.regs_write and _in_range(ins.pc):
                    if from_pc is None or ins.pc <= from_pc:
                        start_idx = ins.idx
            if start_idx is None:
                return []
            seen_idx: set[int] = set()
            queue: list[tuple[int, str, int]] = [(start_idx, dst_reg, 0)]
            while queue and len(results) < max_results:
                idx, reg, depth = queue.pop(0)
                if idx in seen_idx or depth > max_depth:
                    continue
                seen_idx.add(idx)
                ins = items[idx]
                results.append({**_row(ins), "depth": depth, "reg": reg})
                # Each read register's producer is the highest-idx writer
                # before `idx`.
                for r in ins.regs_read:
                    if r in ("sp", "wzr", "xzr"):
                        continue
                    prod = None
                    for cand in items[:idx]:
                        if r in cand.regs_write:
                            prod = cand.idx
                    if prod is not None:
                        queue.append((prod, r, depth + 1))
            return results

        if kind == "boolean_subgraph":
            BOOL_OPS = {"and", "orr", "eor", "bic", "mvn"}
            for ins in items:
                if not _in_range(ins.pc):
                    continue
                toks = _split(ins.mnemonic)
                if not toks or toks[0].lower() not in BOOL_OPS:
                    continue
                if len(toks) < 3:
                    continue
                if not all(_reg_re.match(t) for t in toks[1:]
                           if not t.startswith("#")):
                    continue
                results.append({
                    **_row(ins), "op": toks[0].lower(),
                    "dst": toks[1], "srcs": toks[2:],
                })
                if len(results) >= max_results:
                    break
            return results

        raise ValueError(f"unknown dataflow_query kind: {kind!r}")

    def emit_pseudocode(
        self,
        *,
        out: "Any" = None,
        fmt: str = "text",
    ) -> str:
        """FEATURE-REQUEST-1: render Tier 1 pseudocode for this run.

        Convenience wrapper around `engine.emitter.emit` that targets the
        current run's directory (`self.work.root`). Returns the rendered
        text and optionally writes it to ``out`` (any file-like with a
        ``write`` method).

        Raises `engine.emitter.EmitterError` when the run has no
        `algorithm_identified` finding or hits an unsupported algorithm
        label. Callers that want a soft failure (e.g. preprocess_batch
        auto-emit) should use `engine.emitter.emit_to_run_dir` instead.
        """
        from ..emitter import emit as _emit
        return _emit(self.work.root, out=out, fmt=fmt)

    def verify_and_promote_static_artifacts(self) -> dict[str, Any]:
        """0527 BUG_REPORT-7 §J.4: scan the target .so's .rodata for likely
        keys / IVs / tables next to identified algorithm code.

        Prerequisite (BUG_REPORT-6 ceiling 2): the .so path must be
        propagated through CLI → session. We read it from
        ``self.session["so_path"]``. No-op when unavailable.

        Output: ``static_artifact_candidate`` findings for every word-aligned
        16/24/32-byte window inside .rodata whose Shannon entropy exceeds the
        threshold AND whose zero-byte ratio is bounded — i.e. windows that
        look like real key material rather than zero padding or a known
        LUT layout.

        For BUG_REPORT-6 §6 ceiling 2's "key recovery for libEncryptor.so":
        the AES-256 key (32 bytes) sits inside .rodata adjacent to the Te0
        table base; an agent can grep this finding's bytes_hex for entropy
        outliers and try them as candidate AES keys via §J.6's
        ``utov verify-construction``.
        """
        from .. import static_tools

        so_path_str = self.session.get("so_path")
        if not so_path_str:
            return {
                "stage":   "s5-static-artifacts",
                "promoted": 0,
                "skipped":  "no --so provided",
            }
        so_path = Path(so_path_str)
        if not so_path.exists():
            return {
                "stage":   "s5-static-artifacts",
                "promoted": 0,
                "skipped":  f"so not found: {so_path}",
            }
        if not static_tools.is_available("objdump"):
            return {
                "stage":   "s5-static-artifacts",
                "promoted": 0,
                "skipped":  "objdump not on PATH",
            }

        result = static_tools.run_tool(
            "objdump", ["-s", "-j", ".rodata", str(so_path)],
        )
        if result.exit_code != 0:
            return {
                "stage":   "s5-static-artifacts",
                "promoted": 0,
                "skipped":  f"objdump failed: {result.stderr[:120]}",
            }

        import re
        rodata = bytearray()
        rodata_start: int | None = None
        line_rx = re.compile(
            r"^\s*([0-9a-f]+)\s+"             # offset
            r"((?:[0-9a-f]{2,8}\s+){1,4})"    # 1-4 hex groups
        )
        for ln in result.stdout.splitlines():
            m = line_rx.match(ln)
            if not m:
                continue
            offset = int(m.group(1), 16)
            if rodata_start is None:
                rodata_start = offset
            # Skip gap if objdump dropped trailing zero rows (shouldn't
            # happen with -s, but be defensive).
            elif offset != rodata_start + len(rodata):
                rodata.extend(b"\x00" * (offset - rodata_start - len(rodata)))
            hex_str = re.sub(r"\s+", "", m.group(2))
            try:
                rodata.extend(bytes.fromhex(hex_str))
            except ValueError:
                pass

        if not rodata:
            return {
                "stage":   "s5-static-artifacts",
                "promoted": 0,
                "skipped":  ".rodata empty or unparseable",
            }

        # Shannon entropy over byte values.
        import math
        from collections import Counter
        def _entropy(data: bytes) -> float:
            if not data:
                return 0.0
            counts = Counter(data)
            n = len(data)
            return -sum((c / n) * math.log2(c / n) for c in counts.values())

        HIGH_ENTROPY    = 4.5    # bits/byte — empirical: random 32B ≈ ~4.5-5
        ZERO_RATIO_MAX  = 0.2    # < 20% zeros
        KEY_SIZES       = (16, 24, 32)
        STEP            = 4
        CANDIDATE_CAP   = 50

        candidates: list[dict[str, Any]] = []
        last_seen_at_offset: dict[int, bool] = {}
        for sz in KEY_SIZES:
            for off in range(0, max(0, len(rodata) - sz + 1), STEP):
                window = bytes(rodata[off:off + sz])
                if window.count(0) / sz > ZERO_RATIO_MAX:
                    continue
                entropy = _entropy(window)
                if entropy < HIGH_ENTROPY:
                    continue
                file_offset = (rodata_start or 0) + off
                # Suppress overlapping smaller-size redundant emits at the
                # same start offset (32B already wins over 16B at same off).
                if last_seen_at_offset.get(file_offset):
                    continue
                last_seen_at_offset[file_offset] = True
                candidates.append({
                    "offset":      file_offset,
                    "size":        sz,
                    "entropy":     round(entropy, 3),
                    "bytes_hex":   window.hex(),
                })

        # Sort by entropy descending so the agent sees the strongest first.
        candidates.sort(key=lambda c: -c["entropy"])

        promoted = 0
        for cand in candidates[:CANDIDATE_CAP]:
            subject = (
                f"static_artifact@.rodata+0x{cand['offset']:x}"
                f"/{cand['size']}B"
            )
            f_conn = open_findings_db(self.work)
            try:
                already = f_conn.execute(
                    "SELECT 1 FROM findings "
                    "WHERE kind='static_artifact_candidate' AND subject=?",
                    (subject,),
                ).fetchone()
            finally:
                f_conn.close()
            if already:
                continue

            sz = cand["size"]
            kind_guess = (
                "AES-256 key (32 bytes; high entropy)"      if sz == 32 else
                "AES-192 key (24 bytes; high entropy)"      if sz == 24 else
                "AES-128 key / HMAC key (16 bytes)"
            )
            payload = {
                "kind":         "static_artifact_candidate",
                "section":      ".rodata",
                "offset":       f"0x{cand['offset']:x}",
                "size_bytes":   sz,
                "shannon_entropy_bits_per_byte": cand["entropy"],
                "bytes_hex":    cand["bytes_hex"],
                "kind_guess":   kind_guess,
            }
            try:
                hid = self.submit_hypothesis(
                    kind="static_artifact_candidate",
                    subject=subject,
                    payload=payload,
                    confidence=0.45,    # candidate, not confirmed
                    source="s5_static_artifacts",
                    anchors=[],
                )
                conn_h = open_hypotheses_db(self.work)
                try:
                    HypTree(conn_h).mark_verdict(hid, "pass", {
                        "strategy": "rodata_entropy_scan",
                    })
                finally:
                    conn_h.close()
                self.promote_to_finding(
                    hid, verifier_strategy="rodata_entropy_scan",
                    stage="s5-static-artifacts",
                )
                promoted += 1
            except (KeyError, ValueError):
                pass

        return {
            "stage":                 "s5-static-artifacts",
            "promoted":              promoted,
            "candidates_seen":       len(candidates),
            "rodata_bytes_scanned":  len(rodata),
        }

    def verify_and_promote_mode_evidence_ledger(self) -> dict[str, Any]:
        """0527 BUG_REPORT-7 §J.3: mode-detection negative-evidence ledger.

        For each known mode (HMAC, AES-GCM, …), evaluate predicates against
        the trace and emit a ledger finding describing the evidence
        for/against that mode. The agent uses these ledgers to refuse to
        upgrade a structural match (e.g. SHA-512) to a mode hypothesis
        (HMAC-SHA-512) when the distinguishing anchors are missing.

        Counter to BUG_REPORT-6's "HMAC-SHA-512/256" guess from structural
        block-count alone: that hypothesis had zero direct evidence in the
        trace; the ipad/opad anchors never fire. This pass makes that
        absence loud instead of silent.
        """
        MODE_TEMPLATES = {
            "HMAC": {
                "applies_to": ["SHA-1", "SHA-256", "SHA-512", "SHA-3", "SM3"],
                "predicates": [
                    {"name": "ipad_init_scalar", "kind": "fingerprint_hit",
                     "needle": "HMAC.ipad"},
                    {"name": "opad_init_scalar", "kind": "fingerprint_hit",
                     "needle": "HMAC.opad"},
                    {"name": "ipad_init_simd",   "kind": "fingerprint_hit",
                     "needle": "HMAC.ipad.simd_movi"},
                    {"name": "opad_init_simd",   "kind": "fingerprint_hit",
                     "needle": "HMAC.opad.simd_movi"},
                ],
            },
            "AES-GCM": {
                "applies_to": ["AES"],
                "predicates": [
                    # GHASH R = 0xe1 << 120 — the leading byte / word
                    # appears as an immediate or table entry in nearly every
                    # AES-GCM implementation.
                    {"name": "ghash_R_constant", "kind": "imm_in_trace",
                     "values": [0xe1000000, 0xe1000000_00000000,
                                0xe100_0000_0000_0000_0000_0000_0000_0000]},
                ],
            },
        }

        f_conn = open_findings_db(self.work)
        try:
            algos = {r[0] for r in f_conn.execute(
                "SELECT subject FROM findings "
                "WHERE kind IN ('algorithm_hyp','algorithm_identified')"
            ).fetchall()}
            fps: set[str] = set()
            for (subj,) in f_conn.execute(
                "SELECT subject FROM findings WHERE kind='algo_signature'"
            ).fetchall():
                fps.add(subj.rsplit("@", 1)[0] if "@" in subj else subj)
        finally:
            f_conn.close()

        if not algos:
            return {
                "stage":           "s5-mode-ledger",
                "modes_evaluated": 0,
                "promoted":        0,
            }

        # Build the immediate-value set once.
        imm_values: set[int] = set()
        for ins in self._items:
            imm_values.update(ins.regs_write.values())

        evaluated = 0
        promoted  = 0
        for mode_name, spec in MODE_TEMPLATES.items():
            applies_overlap = sorted(a for a in spec["applies_to"] if a in algos)
            if not applies_overlap:
                continue
            evaluated += 1

            predicate_rows: list[dict[str, Any]] = []
            for pred in spec["predicates"]:
                if pred["kind"] == "fingerprint_hit":
                    hit = pred["needle"] in fps
                    predicate_rows.append({
                        "name":    pred["name"],
                        "checked": pred["needle"],
                        "hit":     hit,
                        "verdict": "PRESENT" if hit else "MISSING",
                    })
                elif pred["kind"] == "imm_in_trace":
                    hit = any(v in imm_values for v in pred["values"])
                    predicate_rows.append({
                        "name":    pred["name"],
                        "checked": [f"0x{v:x}" for v in pred["values"]],
                        "hit":     hit,
                        "verdict": "PRESENT" if hit else "MISSING",
                    })

            n_present = sum(1 for p in predicate_rows if p["verdict"] == "PRESENT")
            n_total   = len(predicate_rows)
            if n_present == 0:
                verdict_text   = f"MISSING — no direct evidence for {mode_name}"
                confidence_cap = 0.30
            elif n_present >= n_total - (n_total // 2):
                # ≥ ceil(n_total/2) — call it strong present
                verdict_text   = (
                    f"PRESENT — {n_present}/{n_total} {mode_name} "
                    "evidence anchors fired"
                )
                confidence_cap = 0.90
            else:
                verdict_text   = (
                    f"PARTIAL — {n_present}/{n_total} {mode_name} "
                    "evidence anchors fired (treat as weak)"
                )
                confidence_cap = 0.55

            subject = f"{mode_name}@{','.join(applies_overlap)}"
            f_conn = open_findings_db(self.work)
            try:
                already = f_conn.execute(
                    "SELECT 1 FROM findings WHERE kind='mode_evidence_ledger' "
                    "AND subject=?", (subject,),
                ).fetchone()
            finally:
                f_conn.close()
            if already:
                continue

            payload = {
                "mode_candidate":        mode_name,
                "applies_to_algorithms": applies_overlap,
                "predicates":            predicate_rows,
                "verdict":               verdict_text,
                "confidence_cap":        confidence_cap,
            }
            try:
                hid = self.submit_hypothesis(
                    kind="mode_evidence_ledger",
                    subject=subject,
                    payload=payload,
                    confidence=0.95,
                    source="s5_mode_ledger",
                    anchors=[],
                )
                conn_h = open_hypotheses_db(self.work)
                try:
                    HypTree(conn_h).mark_verdict(hid, "pass", {
                        "strategy": "mode_evidence_aggregation",
                    })
                finally:
                    conn_h.close()
                self.promote_to_finding(
                    hid, verifier_strategy="mode_evidence_aggregation",
                    stage="s5-mode-ledger",
                )
                promoted += 1
            except (KeyError, ValueError):
                pass

        return {
            "stage":           "s5-mode-ledger",
            "modes_evaluated": evaluated,
            "promoted":        promoted,
        }

    def verify_and_promote_indexed_load_table(self) -> dict[str, Any]:
        """0527 BUG_REPORT-7 §J.1: tabulate indexed loads against a base.

        Walks the trace for `ldr/ldrh/ldrb/ldrsw wD, [bReg, iReg, uxtw #N]`
        (and `lsl` variants). For each distinct (base register value,
        element size) pair seen ≥ MIN_LOADS times, emits one
        `indexed_load_table` finding describing the lookup-table-style
        access pattern (base, element bytes, total loads, distinct index
        count, trace/PC range, sample (idx, value) pairs).

        Generic across crypto: Te0/Td0 (AES), GMUL tables (GHASH/GCM),
        CRC32 tables, DES S-boxes, MixColumns LUTs, ChaCha quarter-round
        LUTs, etc. The base address + load count pair is a strong
        structural fingerprint regardless of which algorithm wraps it.
        """
        import re

        # ldr / ldrh / ldrb / ldrsw with index reg + optional uxtw|sxtw|lsl shift.
        # The dst/base/index regs are captured; the post-bracket shift width
        # is optional. We do NOT match plain immediate offsets — that's array
        # iteration, not LUT indexing.
        LDR_RX = re.compile(
            r"^\s*(?P<mnem>ldr(?:sw|h|b)?)\s+"
            r"(?P<dst>[wx][0-9]+|wzr|xzr),\s*"
            r"\[(?P<base>x[0-9]+|sp),\s*"
            r"(?P<idx>[wx][0-9]+|wzr|xzr)"
            r"(?:,\s*(?:uxtw|sxtw|lsl|sxtx|uxtx)(?:\s*#(?P<shift>\d+))?)?"
            r"\]\s*$"
        )

        MIN_LOADS = 8
        SAMPLE_CAP = 8

        by_base: dict[tuple[int, int], dict[str, Any]] = {}

        for ins in self._items:
            m = LDR_RX.match(ins.mnemonic)
            if not m:
                continue
            mnem  = m.group("mnem")
            dst   = m.group("dst")
            bReg  = m.group("base")
            iReg  = m.group("idx")
            shift = m.group("shift")

            if mnem == "ldrb":
                ebytes = 1
            elif mnem == "ldrh":
                ebytes = 2
            elif mnem == "ldrsw":
                ebytes = 4
            elif mnem == "ldr":
                ebytes = 8 if dst.startswith("x") else 4
            else:
                continue

            # Cross-check the shift if present: must be log2(ebytes) for a
            # canonical LUT access (uxtw #2 → 4-byte, uxtw #3 → 8-byte).
            # If shift contradicts, this is some other addressing pattern —
            # skip rather than misclassify.
            if shift is not None:
                try:
                    if (1 << int(shift)) != ebytes:
                        continue
                except ValueError:
                    continue

            base_val = ins.regs_read.get(bReg)
            if base_val is None:
                continue
            idx_val  = 0 if iReg in ("wzr", "xzr") else ins.regs_read.get(iReg, 0)
            load_val = 0 if dst  in ("wzr", "xzr") else ins.regs_write.get(dst, 0)

            key = (base_val, ebytes)
            d = by_base.get(key)
            if d is None:
                d = {
                    "trace_idx_min": ins.idx, "trace_idx_max": ins.idx,
                    "pc_min": ins.pc, "pc_max": ins.pc,
                    "loads": 0, "unique_idxs": set(), "samples": [],
                }
                by_base[key] = d
            if ins.idx < d["trace_idx_min"]: d["trace_idx_min"] = ins.idx
            if ins.idx > d["trace_idx_max"]: d["trace_idx_max"] = ins.idx
            if ins.pc  < d["pc_min"]:        d["pc_min"]        = ins.pc
            if ins.pc  > d["pc_max"]:        d["pc_max"]        = ins.pc
            d["loads"] += 1
            d["unique_idxs"].add(idx_val)
            if len(d["samples"]) < SAMPLE_CAP:
                d["samples"].append({
                    "idx":       f"0x{idx_val:x}",
                    "value":     f"0x{load_val:x}",
                    "trace_idx": ins.idx,
                })

        promoted = 0
        tables_found = 0
        for (base_val, ebytes), d in by_base.items():
            if d["loads"] < MIN_LOADS:
                continue
            tables_found += 1
            subject = f"indexed_load_table@0x{base_val:x}/{ebytes}B"
            f_conn = open_findings_db(self.work)
            try:
                already = f_conn.execute(
                    "SELECT 1 FROM findings WHERE kind='indexed_load_table' "
                    "AND subject=?", (subject,),
                ).fetchone()
            finally:
                f_conn.close()
            if already:
                continue
            payload = {
                "kind":            "indexed_load_table",
                "base_addr":       f"0x{base_val:x}",
                "element_bytes":   ebytes,
                "total_loads":     d["loads"],
                "unique_indexes":  len(d["unique_idxs"]),
                "trace_idx_range": [d["trace_idx_min"], d["trace_idx_max"]],
                "pc_range":        [f"0x{d['pc_min']:x}", f"0x{d['pc_max']:x}"],
                "sample_values":   d["samples"],
            }
            try:
                hid = self.submit_hypothesis(
                    kind="indexed_load_table",
                    subject=subject,
                    payload=payload,
                    confidence=0.95,
                    source="s5_indexed_load",
                    anchors=[],
                )
                conn_h = open_hypotheses_db(self.work)
                try:
                    HypTree(conn_h).mark_verdict(hid, "pass", {
                        "strategy": "indexed_load_aggregation",
                    })
                finally:
                    conn_h.close()
                self.promote_to_finding(
                    hid, verifier_strategy="indexed_load_aggregation",
                    stage="s5-indexed-load",
                )
                promoted += 1
            except (KeyError, ValueError):
                pass

        return {
            "stage":        "s5-indexed-load",
            "tables_found": tables_found,
            "promoted":     promoted,
        }

    def verify_and_promote_primitive_timeline(self) -> dict[str, Any]:
        """0527 BUG_REPORT-7 §J.2: cross-primitive timeline.

        Whenever ≥2 distinct cryptographic fingerprint families fire in the
        same trace, aggregate their trace-idx + PC ranges and emit a
        `primitive_timeline` finding. Lets the agent reason about composite
        constructions (encrypt-then-MAC, KDF-then-AES, AEAD with MAC tail)
        without hand-grepping the trace.

        Idempotent: a re-invocation with the same trace produces the same
        finding shape, deduped by subject.
        """
        from ..data.fingerprints import FINGERPRINTS, INSTR_PATTERNS

        def _family(fp_name: str) -> str:
            """Normalize fingerprint name into a primitive family label."""
            prefix = fp_name.split(".", 1)[0] if "." in fp_name else fp_name
            return {
                "SHA1":   "SHA-1",
                "SHA256": "SHA-256",
                "SHA512": "SHA-512",
                "SHA3":   "SHA-3",
            }.get(prefix, prefix)

        by_magic = {fp.magic: fp for fp in FINGERPRINTS}
        by_family: dict[str, dict[str, Any]] = {}

        def _accumulate(family: str, trace_idx: int, pc: int) -> None:
            d = by_family.setdefault(family, {
                "trace_idx_min": trace_idx, "trace_idx_max": trace_idx,
                "pc_min": pc, "pc_max": pc, "hits": 0,
            })
            if trace_idx < d["trace_idx_min"]: d["trace_idx_min"] = trace_idx
            if trace_idx > d["trace_idx_max"]: d["trace_idx_max"] = trace_idx
            if pc < d["pc_min"]: d["pc_min"] = pc
            if pc > d["pc_max"]: d["pc_max"] = pc
            d["hits"] += 1

        for ins in self._items:
            for val in ins.regs_write.values():
                fp = by_magic.get(val)
                if fp is not None:
                    _accumulate(_family(fp.name), ins.idx, ins.pc)
            for pat in INSTR_PATTERNS:
                if pat.match_text in ins.mnemonic:
                    _accumulate(_family(pat.primitive), ins.idx, ins.pc)

        if len(by_family) < 2:
            return {
                "stage":           "s5-primitive-timeline",
                "promoted":        0,
                "families_seen":   len(by_family),
                "segments_count":  0,
            }

        segments = sorted(
            [
                {
                    "primitive":       fam,
                    "trace_idx_range": [d["trace_idx_min"], d["trace_idx_max"]],
                    "pc_range":        [f"0x{d['pc_min']:x}", f"0x{d['pc_max']:x}"],
                    "hits":            d["hits"],
                }
                for fam, d in by_family.items()
            ],
            key=lambda s: s["trace_idx_range"][0],
        )

        overlaps: list[dict[str, str]] = []
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                a, b = segments[i], segments[j]
                lo = max(a["trace_idx_range"][0], b["trace_idx_range"][0])
                hi = min(a["trace_idx_range"][1], b["trace_idx_range"][1])
                if lo <= hi:
                    overlaps.append({"a": a["primitive"], "b": b["primitive"]})

        order_str = " → ".join(s["primitive"] for s in segments)
        suffix    = " (sequential, no overlap)" if not overlaps else " (interleaved)"

        # Dedupe on the same trace: subject encodes the family list + order.
        subject = f"timeline:{order_str}"
        f_conn = open_findings_db(self.work)
        try:
            already = f_conn.execute(
                "SELECT 1 FROM findings WHERE kind = 'primitive_timeline' "
                "AND subject = ?", (subject,),
            ).fetchone()
        finally:
            f_conn.close()
        if already:
            return {
                "stage":          "s5-primitive-timeline",
                "promoted":       0,
                "families_seen":  len(by_family),
                "segments_count": len(segments),
                "ordering":       order_str + suffix,
            }

        payload = {
            "segments": segments,
            "overlaps": overlaps,
            "ordering": order_str + suffix,
        }
        try:
            hid = self.submit_hypothesis(
                kind="primitive_timeline",
                subject=subject,
                payload=payload,
                confidence=0.95,    # deterministic aggregation, no inference
                source="s5_primitive_timeline",
                anchors=[],
            )
            conn_h = open_hypotheses_db(self.work)
            try:
                HypTree(conn_h).mark_verdict(hid, "pass", {
                    "strategy": "primitive_timeline_aggregation",
                })
            finally:
                conn_h.close()
            self.promote_to_finding(
                hid, verifier_strategy="primitive_timeline_aggregation",
                stage="s5-primitive-timeline",
            )
            promoted = 1
        except (KeyError, ValueError):
            promoted = 0

        return {
            "stage":          "s5-primitive-timeline",
            "promoted":       promoted,
            "families_seen":  len(by_family),
            "segments_count": len(segments),
            "ordering":       order_str + suffix,
        }

    def promote_to_finding(
        self,
        hyp_id: int,
        verifier_strategy: str,
        stage: str = "s6",
    ) -> int:
        """Upgrade a verifier-passed hypothesis into a finding (PLAN §1.1).

        Reads the hyp's claim template + payload via refs, copies payload into
        the findings DB's hyp_payloads (content-addressed; may already exist),
        then inserts the finding row.
        """
        from ..store import upsert_payload as _upsert

        conn = open_hypotheses_db(self.work)
        try:
            row = conn.execute(
                "SELECT h.status, t.kind, h.subject, t.payload_ref, p.payload,"
                "       t.source"
                " FROM hypotheses h"
                " JOIN claim_templates t ON h.template_id = t.id"
                " JOIN hyp_payloads p ON p.content_hash = t.payload_ref"
                " WHERE h.id = ?",
                (hyp_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(hyp_id)
        status, kind, subject, _payload_ref, payload_json, hyp_source = row
        if status != "passed":
            raise ValueError(f"hyp {hyp_id} status={status}, must be 'passed' to promote")

        # 0527: pick up the current batch_id (if any) from instance state
        # so findings promoted during preprocess_batch share a tag agents
        # can later filter / bulk-discard on.
        current_batch = getattr(self, "_current_batch_id", None)

        fc = open_findings_db(self.work)
        try:
            new_ref = _upsert(fc, payload_json)
            cur = fc.execute(
                "INSERT INTO findings(stage, kind, subject, payload_ref, verified_at,"
                " verifier_strategy, origin_hyp_id, source, batch_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (stage, kind, subject, new_ref, _now_iso(),
                 verifier_strategy, hyp_id, hyp_source or "unknown",
                 current_batch),
            )
            fc.commit()
            return cur.lastrowid  # type: ignore[return-value]
        finally:
            fc.close()
