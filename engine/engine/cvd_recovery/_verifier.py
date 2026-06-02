"""cvd_recovery.verifier section (split from the monolithic module)."""
from __future__ import annotations


import hashlib
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_log = logging.getLogger(__name__)

from ..capabilities import collect_build_capabilities, coverage_for_terminal
from ..closure_classification import (
    classify_closure,
    parity_exact_from_report,
    provenance_closed_from_verdict,
)
from ..cohort_diff import InputDependenceMap
from ..cvd import (
    Candidate,
    CandidateGenerator,
    CvdBudget,
    CvdResult,
    CvdState,
    Registry,
    Terminal,
    TerminalClassifier,
    Verdict,
    Verifier,
    VStatus,
    run_cvd_collect_to_json,
)
from ..dispatch_coverage import CoverageMap
from ..oracle_provenance import ProvenanceVerdict, trace_provenance
from ..recapture_loop import (
    LoopOutcome,
    _is_closed as _provenance_closed_by_observation,
    run_recapture_loop,
)
from ..recapture_target import derive_recapture_directive
from ..opaque_staging import (
    PointerChainSpec,
    VERDICT_SYMBOLIC_ADDRESS,
    diagnose_opaque_staging,
)
from ..setup_symex import (
    CaseConfig,
    DrivePause,
    DriveResult,
    MemLiveIn,
    SetupSymexConfig,
    derive_window_mem_live_in,
    drive,
)
from ..trace_observability import assess_trace_observability
from ..types import Instruction
from ._cohort import MEM_SINK_UNPLACEABLE_NEEDED, OPAQUE_STAGING_FRONTIER, RECOVER_WINDOW, REMEDY_ADD_SEEDS, REMEDY_DIVERSIFY_SEEDS, REMEDY_RE_ANCHOR, SIG_GENERATION_BUDGET_EXHAUSTED, TERMINAL_BAND_PARITY_FAIL, TERMINAL_COMPOSITE_REQUIRED, TERMINAL_COMPOSITE_TOO_EXPENSIVE, TERMINAL_MEM_SINK_UNPLACEABLE, TERMINAL_MEMORY_DISPOSITION_MISSING, _compact, _current_run_loaded_value, _drive_evidence, recommend_mem_disposition
from ._compose import _band_window, _classify_block_kind, _classify_drive_result, _f_references_inputs, _terminal_coverage, _window_closure, compose_band_transforms, derive_mem_sink_interval, plan_composite_recovery


class RecoverWindowVerifier(Verifier):
    """The heavy (T2) Verifier: one ``recover_window`` candidate → one full
    ``setup_symex.drive()`` run → a CVD :class:`Verdict`.

    ``base_config`` is the case identity + fixed knobs (target / input_hash /
    run_id / reg_file / inputs / seed+sink hints / backing); the per-window
    geometry (``window`` / ``window_kind``) comes from the candidate payload and
    overrides the base. ``triton_runner`` is the Level-1/Level-2 symex runner
    (e.g. ``setup_symex_runner.build_level2_runner``). The verdict mapping:

      * closed (parity EXACT)                 → CONFIRMED  (evidence = F/parity)
      * fixable geometry/feed/backing/parity  → ELIMINATED (+ corrected spawn)
      * un-modeled opcode                     → TERMINAL   (capability_request)
      * opaque staging / seed-invariant       → TERMINAL   (frontier / not-a-window)
      * DrivePause (agent judgment)           → PENDING    (checkpoint carried)
    """

    name = "recover_window"
    version = "1"
    owner = "core"

    def __init__(
        self,
        *,
        base_config: CaseConfig,
        triton_runner: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        ledger: Any = None,
        decisions: Mapping[str, Any] | None = None,
        pointer_chain: "PointerChainSpec | None" = None,
        cohort_traces: "Sequence[Sequence[Instruction]]" = (),
        input_keys: "Sequence[str] | None" = None,
        disp_thresholds: "Mapping[str, float] | None" = None,
        cohort_mem_sidecars: "Sequence[Any] | None" = None,
        cohort_load_diagnostics: "Mapping[str, Any] | None" = None,
        dependence: "InputDependenceMap | None" = None,
        onpath_bands: "Sequence[tuple[int, int]] | None" = None,
        band_registry: "_OnpathBandRegistry | None" = None,
        budget: "CvdBudget | None" = None,
        mem_sink: "Mapping[str, Any] | None" = None,
    ) -> None:
        self.base_config = base_config
        # Issue 7 — EXPLICIT mem-write recovery sink descriptor
        # (spec_f0_mem_write_window_sink.md). When set, the window's OUTPUT is a
        # memory store (not a register): the descriptor names the output store via
        # ``sink_idx`` / ``sink_addr`` / ``sink_size``. None → register path,
        # byte-for-byte today's x8 behaviour (the regression guard). A per-candidate
        # payload ``mem_sink`` overrides this base descriptor (the per-window sink).
        self.mem_sink = dict(mem_sink) if mem_sink is not None else None
        self.triton_runner = triton_runner
        self.ledger = ledger
        self.decisions = dict(decisions or {})
        # Composite recovery (Req6): the full on-path BAND set this run generated, so a
        # BAND_PARITY_FAIL band knows its ADJACENT bands to combine. None → the verifier
        # plans the composite over just THIS band (a single-band plan stays
        # BAND_PARITY_FAIL — nothing to combine). The composite-cost budget rides the
        # SAME CvdBudget the generator uses. Construct-symmetry red line: the registry
        # passes both transparently (no new caller obligation); absent → byte-for-byte
        # today's parity ELIMINATED for non-band windows (invariant 7).
        self.onpath_bands = (
            [(int(lo), int(hi)) for lo, hi in onpath_bands]
            if onpath_bands is not None else None)
        # A3 collect-layer aggregation: the run-level chain_id -> [bands] index the
        # generator POPULATED (recovery_registry wires the SAME instance into both — one
        # source, no caller obligation). A BAND_PARITY_FAIL candidate looks up ALL bands
        # of its own chain_id here so plan_composite_recovery sees the WHOLE group, not
        # the single band drive happened to hand it. None → fall back to the static
        # onpath_bands / this band alone (graceful, never an error; today's behaviour).
        self.band_registry = band_registry
        self.budget = budget or CvdBudget()
        # pre-flight observable-variance gate (output-side dual of
        # check_seed_independence's seed_block_note): the SAME cohort dependence map
        # the generator/terminal classifier hold. Only consulted for an early BLOCK
        # when ``verdict == "localized"`` AND the candidate window contains zero
        # varying position — every other dependence state (opaque / low-observability
        # / absent / window has variance) stands down to the normal drive() flow
        # byte-for-byte (invariant 7). None → never triggers (today's behaviour).
        self.dependence = dependence
        # Opaque-staging Phase 0 inputs (case-specific → injected, never hardcoded):
        # the pointer-chain SHAPE (config) and any cohort traces for byte-level
        # corroboration. Both optional — the diagnosis degrades to single-trace
        # (verdict + EA-symbolic from the DFG) when absent, never silent.
        self.pointer_chain = pointer_chain
        self.cohort_traces = list(cohort_traces)
        # Evidence-backed mem disposition (invariant 8): the cohort's input_keys
        # (the "inputs truly varied" gate) and the variance thresholds. Both
        # optional — no cohort/keys → zero prefill → byte-for-byte today's PENDING.
        self.input_keys = list(input_keys) if input_keys is not None else None
        self.disp_thresholds = dict(disp_thresholds) if disp_thresholds else None
        # 坎1: per-cohort-vector _mem.jsonl sidecar paths, parallel to
        # cohort_traces. Restores merge symmetry so a bare-fed cohort vector is not
        # blind in the mem dimension purely for lack of merge. None → bare cohort
        # (today's behaviour). The last-mem-disposition diagnostics (merge report +
        # symmetry/degradation decision) are captured here for the PENDING evidence.
        self.cohort_mem_sidecars = (
            list(cohort_mem_sidecars) if cohort_mem_sidecars is not None else None)
        # 坎1 重改: the load-layer report from ``load_cohort_traces`` (which vector
        # had no mem sidecar resolved). Surfaced into the per-window disposition
        # diagnostics so the "no sidecar, auto-resolution missed" WARN reaches the
        # gap-map evidence at the boundary (invariant 1), not buried. None → cohort
        # was supplied as already-loaded Instruction sequences (no load layer ran).
        self.cohort_load_diagnostics = (
            dict(cohort_load_diagnostics) if cohort_load_diagnostics is not None else None)
        self._last_disp_diag: dict[str, Any] = {}

    def applies(self, c: Candidate, state: CvdState) -> bool:
        return c.kind == RECOVER_WINDOW

    def cost(self, c: Candidate, state: CvdState) -> float:
        return 100.0   # heavy / T2 — a full symex run

    def _case_config(self, c: Candidate) -> CaseConfig:
        p = c.payload or {}
        win = tuple(p.get("window", list(self.base_config.window)))
        wk = p.get("window_kind", self.base_config.window_kind)
        return replace(self.base_config, window=(win[0], win[1]), window_kind=wk)

    def _effective_mem_sink(self, c: Candidate) -> dict[str, Any] | None:
        """The mem-write sink descriptor for THIS candidate, or None (Issue 7).

        A per-candidate payload ``mem_sink`` (the per-window output store) overrides
        the verifier's base descriptor; ``sink_form="reg"`` (or absent) → None, the
        register path (the regression guard). Only ``sink_form=="mem"`` activates the
        explicit mem-write recovery path."""
        payload = c.payload or {}
        desc = payload.get("mem_sink") if payload.get("mem_sink") is not None \
            else self.mem_sink
        if desc is None:
            return None
        if str(desc.get("sink_form", "mem")) != "mem":
            return None                          # reg / other → register path
        return dict(desc)

    def _mem_sink_unplaceable_verdict(
        self, c: Candidate, descriptor: Mapping[str, Any], why: str,
    ) -> Verdict:
        """The structured MEM_SINK_UNPLACEABLE terminal (Issue 7, spec §blocker).

        The window's OUTPUT is a memory store but its effective address could not be
        derived (no trace mem op at sink_idx / no byte-granular dep / B5-mem could not
        pin a symbolic EA) OR its symbolic bytes could not be read after the sink
        store. A structured dead end — NEVER a silent fall-back to a register/constant
        (A8④ / 契约③). Carries clark's VERBATIM ``needed[]`` list."""
        sink_idx = descriptor.get("sink_idx")
        ev = _compact({
            "terminal":  TERMINAL_MEM_SINK_UNPLACEABLE,
            "reason":    ("cannot derive store effective address / cannot read "
                          "symbolic memory bytes after sink store"),
            "detail":    why,
            "sink_idx":  sink_idx,
            "sink_form": "mem",
            "needed":    list(MEM_SINK_UNPLACEABLE_NEEDED),
        })
        return Verdict(
            VStatus.TERMINAL, terminal_kind=TERMINAL_MEM_SINK_UNPLACEABLE,
            reason=("mem-write recovery sink is unplaceable: " + why),
            evidence=ev, located_base=c.locus,
            capability_request=(
                "supply the store's effective address / size for the mem-write sink "
                "(a trace mem op at sink_idx, pc-local regs to decode the EA, or "
                "B5-mem symbolic-EA resolution) so the runner can read the store's "
                "symbolic output bytes — needed: "
                + ", ".join(MEM_SINK_UNPLACEABLE_NEEDED)))

    def _mem_disposition_recs(self, cc: CaseConfig, items) -> dict[int, MemDispositionRec]:
        """Cohort-variance recommendations for this window's external mem inputs.

        No cohort → empty (zero prefill, today's behaviour). Pure / advisory."""
        self._last_disp_diag = {}
        if not self.cohort_traces:
            return {}
        win = (cc.window[0], cc.window[1])
        wk = (cc.window_kind == "idx")
        mem_live_in, _ = derive_window_mem_live_in(items, window=win, window_is_idx=wk)
        if not mem_live_in:
            return {}
        diag: dict[str, Any] = {}
        # Surface the load-layer report (no-mem-sidecar WARN per vector) alongside
        # the per-window merge/observability diagnostics so a vector that arrived
        # bare because its sibling was missing is visible in the gap map, not silent.
        if self.cohort_load_diagnostics is not None:
            diag["cohort_load"] = self.cohort_load_diagnostics
        recs = recommend_mem_disposition(
            mem_live_in, self.cohort_traces, window=win, window_is_idx=wk,
            input_keys=self.input_keys, thresholds=self.disp_thresholds,
            cohort_mem_sidecars=self.cohort_mem_sidecars, diagnostics=diag)
        self._last_disp_diag = diag
        return recs

    def _mem_disposition_audit(
        self, cc: CaseConfig, items,
        recs: "Mapping[int, MemDispositionRec]",
    ) -> dict[str, Any]:
        """Per-window memory-disposition coverage audit (Req5).

        For THIS candidate's own window (self-dispatched — never the early map), find
        the external memory live-in addresses (``derive_window_mem_live_in``) and split
        them into DECIDED (a usable disposition is available — ``symbolize``/``back``,
        confidence ``auto``/``recommend``, OR the caller pinned it explicitly in
        ``decisions['mem_input_symbolize_vs_back']``) vs UNDECIDED (no usable
        disposition: no cohort to compare, a blind window, a shallow cohort → the
        recommender returns ``confidence=='none'`` / the addr is absent from recs).

        Returns ``{"n_live_in", "live_in", "decided", "undecided", "all_undecided",
        "any_live_in"}`` — a small structured summary (invariant 4). ``all_undecided``
        is True iff there IS external mem live-in and NONE of it was decided: that is
        the "this window's memory was never classified" condition the
        MEMORY_DISPOSITION_MISSING terminal keys on (separating a missing decision
        from a genuine opaque/constant collapse). Pure read of already-computed facts;
        feeds NO close / parity / G4 gate (invariant 7)."""
        win = (cc.window[0], cc.window[1])
        wk = (cc.window_kind == "idx")
        mem_live_in, _ = derive_window_mem_live_in(items, window=win, window_is_idx=wk)
        live_addrs = [int(m.addr) for m in mem_live_in]
        # Caller's explicit pins (mem_input_symbolize_vs_back) ARE a decision — a
        # window the agent already classified is not "missing" (the override wins in
        # verify()'s prefill too). Keys may be int or hex-str; normalise to int.
        pinned: set[int] = set()
        caller_md = self.decisions.get("mem_input_symbolize_vs_back") or {}
        for k in caller_md:
            try:
                pinned.add(int(k, 16) if isinstance(k, str) else int(k))
            except (TypeError, ValueError):
                continue
        decided: list[int] = []
        undecided: list[int] = []
        for a in live_addrs:
            rec = recs.get(a)
            usable = (a in pinned) or (
                rec is not None and rec.confidence in ("auto", "recommend")
                and rec.disposition in ("symbolize", "back"))
            (decided if usable else undecided).append(a)
        any_live_in = bool(live_addrs)
        return {
            "n_live_in":     len(live_addrs),
            "live_in":       [f"0x{a:x}" for a in live_addrs],
            "decided":       [f"0x{a:x}" for a in decided],
            "undecided":     [f"0x{a:x}" for a in undecided],
            "any_live_in":   any_live_in,
            # ALL of the window's external mem live-in is undecided (and there is some)
            # → the window's memory was never classified for THIS window.
            "all_undecided": any_live_in and not decided,
        }

    def _memory_disposition_missing_verdict(
        self, c: Candidate, cc: CaseConfig, base_evidence: dict[str, Any],
        audit: dict[str, Any], collapse_disposition: str,
    ) -> Verdict:
        """The honest MEMORY_DISPOSITION_MISSING terminal (Req5).

        A window whose external memory live-in was NEVER classified collapsed to
        opaque / a constant. That collapse is the MISSING decision, NOT a known
        algorithm property — surface it as its OWN terminal (WARN-loud, never the
        misleading opaque_staging / constant trap). Carries WHICH live-in addresses
        are undecided + WHY (no cohort to compare across the window), so the agent
        knows the next move is to SUPPLY the window's disposition (cohort to classify
        symbolize-vs-back, or an explicit pin), not to read it as an algorithm fact."""
        ev = dict(base_evidence)
        ev["memory_disposition_audit"] = _compact(audit)
        ev["collapsed_disposition"] = collapse_disposition   # what it WOULD have been
        if self._last_disp_diag:
            ev["mem_disposition_diagnostics"] = _compact(self._last_disp_diag)
        reason = (
            f"this window's external memory live-in ({audit['n_live_in']} address(es): "
            f"{audit['undecided']}) was never classified (symbolize-vs-back), so the "
            f"window's output collapsed to '{collapse_disposition}' — that collapse is "
            f"a MISSING per-window memory disposition, NOT a known algorithm property. "
            f"The early window's disposition map does not cover this window (its live-in "
            f"reads land on different addresses); supply this window's disposition "
            f"(a cohort to classify each live-in's value variance, or an explicit "
            f"mem_input_symbolize_vs_back pin) — do NOT read the collapse as opaque / a "
            f"constant algorithm result")
        return Verdict(
            VStatus.TERMINAL, terminal_kind=TERMINAL_MEMORY_DISPOSITION_MISSING,
            reason=reason, evidence=ev, located_base=c.locus,
            capability_request=(
                "classify this window's external memory live-in (supply a cohort to "
                "compute symbolize-vs-back per address, or pin it via "
                "mem_input_symbolize_vs_back) — the window's memory disposition is "
                "missing, not an algorithm property"))

    def _preflight_observable_variance(self, cc: CaseConfig, c: Candidate) -> Verdict | None:
        """Pre-flight observable-variance gate — the output-side dual of
        ``check_seed_independence`` (seed-side, pre-symex), front-running the
        post-hoc closability gate (setup_symex UNCLOSABLE) to BEFORE drive().

        Returns an early-BLOCK :class:`Verdict` (and the verify() caller must NOT
        call drive) on a held ``localized`` map in two cases:

          (A) **Zero window variance** — the candidate payload window ``[lo,hi]``
              contains NO varying position (``window_is_seed_varying`` False). The
              output is input-independent over this window, so a full symex run
              would only validate a constant — BLOCK early, save the round, and
              ANCHOR recovery at ``divergence_idx`` (or the window-external nearest
              varying position) via a spawned corrected candidate (a comfortable
              exit / precise door — feedback_red_line_needs_comfortable_exit).

          (B) **Cohort too shallow** — the window DOES vary, but the cohort carries
              fewer than ``parity_min_vectors`` vectors. The multi-vector parity
              floor needs >= N distinct independent observed outputs; ``n_vectors <
              N`` can never supply them → UNCLOSABLE regardless of F. Early BLOCK
              with "cohort output diversity insufficient; need diverse seeds" — the
              pre-flight dual of the post-hoc UNCLOSABLE verdict, no spawn (the fix
              is upstream: supply more output-diverse seeds, not re-anchor).

        Variance is the cohort's own measurement (invariant 8: ``varying`` /
        ``n_vectors``), never fabricated.

        Every other state STANDS DOWN (returns None → normal drive flow,
        byte-for-byte; the post-hoc observed-variance gate backstops it):
          * no dependence map held;
          * ``verdict`` is ``opaque`` / ``inconclusive_low_observability`` /
            ``insufficient`` (no trustworthy ``varying`` set to gate on);
          * the window DOES contain a varying position (invariant 7: a window with
            variance takes the unchanged path).
        """
        dep = self.dependence
        if dep is None or dep.verdict != "localized":
            return None                          # stand down — no localized variance map
        lo, hi = int(cc.window[0]), int(cc.window[1])
        if lo > hi:
            lo, hi = hi, lo
        by_idx = (cc.window_kind == "idx")
        if dep.window_is_seed_varying(lo, hi, by_idx=by_idx):
            # The window HAS variance, but the cohort may still be too SHALLOW to
            # ever EXACT-close: the multi-vector parity floor (setup_symex's
            # ``parity_min_vectors``, the same min_vectors check_parity_vectors uses)
            # demands >= N distinct independent observed outputs, and a cohort of
            # ``n_vectors < N`` can supply at most ``n_vectors`` distinct ones —
            # UNCLOSABLE no matter what F is. This is the pre-flight (pre-symex) dual
            # of the post-hoc UNCLOSABLE verdict: catch the cohort-diversity floor
            # BEFORE a wasted full symex round. (invariant 8: n_vectors is the
            # cohort's own measured size, never fabricated.)
            min_vectors = SetupSymexConfig.from_env().parity_min_vectors
            if dep.n_vectors < min_vectors:
                reason = (
                    f"cohort output diversity insufficient here; the cohort carries "
                    f"only {dep.n_vectors} vector(s) but EXACT-close needs >= "
                    f"{min_vectors} distinct independent observed outputs — no F can "
                    f"close it; need diverse seeds (supply >= {min_vectors} "
                    f"output-diverse cohort vectors), do NOT spend a symex round here")
                evidence = _compact({
                    "preflight": "observable_variance",
                    "window": [lo, hi], "window_kind": cc.window_kind,
                    "disposition": "cohort_diversity_insufficient",
                    "drive_skipped": True,
                    "n_vectors": dep.n_vectors,
                    "min_vectors": min_vectors,
                    # Three-factor remedy tag (task 2 — factor 2 / add-seeds): the
                    # window HAS variance but the cohort is too SMALL (n_vectors <
                    # min) → the fix is MORE seeds, not re-anchoring. Makes df0f95c's
                    # implicit "need diverse seeds" reason an explicit, machine-
                    # routable tag; coexists with the reason text (does not replace
                    # it). Pure evidence key — does not touch the verdict (inv 7).
                    "remedy": REMEDY_ADD_SEEDS,
                    "remedy_reason": (
                        "cohort has variance but too few vectors (n_vectors="
                        f"{dep.n_vectors} < min_vectors={min_vectors}) — supply MORE "
                        "seeds to reach the distinct-output floor (re-anchoring will "
                        "not help; the window is already on the varying path)"),
                    "reason": reason,
                })
                return Verdict(
                    VStatus.ELIMINATED, reason=reason, spawn=[],
                    evidence=evidence, located_base=c.locus)
            return None                          # window HAS variance — normal flow (inv 7)
        # Window has zero varying position under a localized cohort → input-
        # independent here. Early BLOCK before drive (save the whole symex round).
        div = dep.divergence_idx
        anchor = self._nearest_varying_anchor(dep, lo, hi, by_idx=by_idx)
        reason = (
            "no input variance in this window — output is input-independent here; "
            f"cohort_diff locates variance at idx >= {div}; anchor recovery there")
        evidence = _compact({
            "preflight": "observable_variance",
            "window": [lo, hi], "window_kind": cc.window_kind,
            "disposition": "no_window_variance",
            "drive_skipped": True,
            "divergence_idx": div,
            "anchor_idx": anchor,
            "varying_idxs": list(dep.varying_idxs),
            # Three-factor remedy tag (task 2 — factor 1 / re-anchor): the window has
            # ZERO input-variance (the input never excited it; structurally outside
            # the input-dependent path) → the fix is to RE-ANCHOR to divergence_idx,
            # NOT supply more/different seeds. Explicit tag over df0f95c's existing
            # anchor reason; coexists with it. Pure evidence key (inv 7).
            "remedy": REMEDY_RE_ANCHOR,
            "remedy_reason": (
                "window carries no input-variance (zero varying position) — the input "
                f"does not excite it; re-anchor recovery to the varying path (idx >= "
                f"{div}), do NOT add or diversify seeds for this window"),
            "reason": reason,
        })
        spawn = self._anchor_spawn(c, dep, anchor)
        return Verdict(
            VStatus.ELIMINATED, reason=reason, spawn=spawn,
            evidence=evidence, located_base=c.locus)

    @staticmethod
    def _nearest_varying_anchor(
        dep: "InputDependenceMap", lo: int, hi: int, *, by_idx: bool,
    ) -> int | None:
        """The idx to anchor recovery at: ``divergence_idx`` when it is a real
        position, else the varying position whose key is nearest the (empty) window
        band. Pure / deterministic over the cohort's own ``varying`` set (inv 8)."""
        if dep.divergence_idx is not None:
            return dep.divergence_idx
        best: int | None = None
        best_dist = None
        for p in dep.varying:
            key = p.idx if by_idx else p.pc
            dist = 0 if lo <= key <= hi else min(abs(key - lo), abs(key - hi))
            if best_dist is None or dist < best_dist:
                best, best_dist = p.idx, dist
        return best

    @staticmethod
    def _anchor_spawn(
        c: Candidate, dep: "InputDependenceMap", anchor: int | None,
    ) -> list[Candidate]:
        """Spawn one corrected candidate anchored at the variance idx (the direction
        the agent should take instead). No anchor known → no spawn (the reason still
        names ``divergence_idx``). Anchored at the SAME window the generator would
        emit for that varying position (``[anchor, anchor]``, idx band)."""
        if anchor is None:
            return []
        p = dict(c.payload or {})
        if p.get("_variance_anchored"):
            return []                            # already corrected — do not loop
        is_div = (dep.divergence_idx is not None and anchor == dep.divergence_idx)
        payload = _compact({
            "window": [anchor, anchor], "window_kind": "idx",
            "source": "preflight_variance_anchor",
            "anchor": "divergence_idx" if is_div else "nearest_varying",
            "divergence_idx": dep.divergence_idx,
            "_variance_anchored": True,
        })
        return [Candidate(
            RECOVER_WINDOW, locus=anchor, signal="variance_anchor",
            entry_reason=(
                f"pre-flight re-anchor: previous window had no input variance — "
                f"anchor recovery at {'divergence' if is_div else 'nearest varying'} "
                f"idx{anchor} (where the input actually reaches observable state)"),
            base_value=(5.0 if is_div else 4.0),
            payload=payload)]

    def verify(self, c: Candidate, state: CvdState) -> Verdict:
        # Generation-budget marker (dev-recovery-generation-budget-spec): NOT a window
        # to symex — it is the explicit "generation was capped" report the generator
        # emitted when the candidate set exceeded budget.max_gen_candidates. Surface
        # it as a TERMINAL verdict carrying the truncation report (what/how-many/order
        # was dropped), never run drive() on it. Additive branch for a NEW signal →
        # every existing signal's path is byte-for-byte unchanged (invariant 7).
        if c.signal == SIG_GENERATION_BUDGET_EXHAUSTED:
            return Verdict(
                status=VStatus.TERMINAL,
                terminal_kind="GENERATION_BUDGET_EXHAUSTED",
                reason=c.entry_reason,
                evidence=_compact(dict(c.payload or {})),
                capability_request="raise budget.max_gen_candidates or narrow the goal")
        cc = self._case_config(c)
        # Issue 7 — EXPLICIT mem-write sink (spec_f0_mem_write_window_sink.md). When
        # this window's OUTPUT is a memory store, resolve the sink descriptor (a
        # per-candidate payload ``mem_sink`` overrides the verifier's base) and derive
        # the store interval [sink_addr, sink_addr+sink_size) from the trace mem op /
        # the descriptor — the caller is NOT required to fill addr+size. If the EA
        # cannot be pinned (no trace mem op at sink_idx, no byte-granular dep, B5-mem
        # could not resolve) → MEM_SINK_UNPLACEABLE BEFORE drive (the structured
        # blocker, never a silent register/constant fallback). None → register path,
        # byte-for-byte today's x8 behaviour (the regression guard).
        mem_sink_desc = self._effective_mem_sink(c)
        eff_mem_sink: dict[str, Any] | None = None
        if mem_sink_desc is not None:
            interval, why = derive_mem_sink_interval(state.items, mem_sink_desc)
            if interval is None:
                return self._mem_sink_unplaceable_verdict(c, mem_sink_desc, why or "")
            sink_addr, sink_size = interval
            eff_mem_sink = {
                "sink_addr": sink_addr, "sink_size": sink_size,
                "sink_idx": (int(mem_sink_desc["sink_idx"])
                             if mem_sink_desc.get("sink_idx") is not None else None)}
        # PRE-FLIGHT observable-variance gate — runs BEFORE any disposition/drive():
        # a localized cohort whose candidate window carries zero varying position is
        # input-independent here → early BLOCK + anchor, drive() never called (the
        # whole symex round saved). Stands down (None) for opaque / low-obs / no map
        # / window-with-variance → normal flow, post-hoc gate backstops (invariant 7).
        preflight = self._preflight_observable_variance(cc, c)
        if preflight is not None:
            return preflight
        # Evidence-backed mem disposition: compute recs, AUTO-prefill only the
        # reliable direction (confidence=="auto" → symbolize), under any caller
        # decisions (caller's explicit value always wins). recommend/none are NOT
        # prefilled — those addrs stay undecided → drive PENDING (invariant 8).
        recs = self._mem_disposition_recs(cc, state.items)
        # Req5 — per-window memory-disposition coverage audit (self-dispatched for THIS
        # window, never the early map). If the window's external mem live-in is wholly
        # un-classified, a downstream opaque / constant collapse is a MISSING decision,
        # not an algorithm property — the audit lets the opaque/constant branches route
        # to the honest MEMORY_DISPOSITION_MISSING terminal instead of a false one.
        mem_audit = self._mem_disposition_audit(cc, state.items, recs)
        eff_decisions = dict(self.decisions)
        if recs:
            caller_md = dict((self.decisions.get("mem_input_symbolize_vs_back") or {}))
            prefill: dict[Any, Any] = {}
            for addr, rec in recs.items():
                if rec.confidence == "auto" and rec.disposition == "symbolize":
                    # drive applies a symbolize disposition by materialising the
                    # SymVar at the load's seed value; the value is this run's loaded
                    # value at addr (the concrete operand symex seeds from). Encode
                    # the value-bearing form drive consumes ({"symbolize": val}).
                    val = _current_run_loaded_value(
                        state.items, addr,
                        window=(cc.window[0], cc.window[1]),
                        window_is_idx=(cc.window_kind == "idx"))
                    if val is not None:
                        prefill[addr] = {"symbolize": val}
            # caller's explicit mem decisions override the prefill.
            prefill.update(caller_md)
            if prefill:
                eff_decisions["mem_input_symbolize_vs_back"] = prefill
        result = drive(trace=state.items, case_config=cc,
                       triton_runner=self.triton_runner, ledger=self.ledger,
                       decisions=dict(eff_decisions),
                       pointer_chain=self.pointer_chain,
                       # cohort→parity feed leg: drive runs the runner once per
                       # cohort vector to build REAL cross-run parity vectors. The
                       # verifier already holds these (construct_symmetry red line:
                       # transparent forward, no new caller obligation). Empty →
                       # byte-for-byte today (invariant 7).
                       cohort_traces=self.cohort_traces,
                       cohort_keys=self.input_keys,
                       # Issue 7 — the resolved mem-sink interval (addr+size filled).
                       # drive forwards it to the runner as output_mem so expression()
                       # reads the store's symbolic bytes; observed/predicted are then
                       # bytewise. None → register path, unchanged (invariant 7).
                       mem_sink=eff_mem_sink)
        if isinstance(result, DrivePause):
            # An agent-judgment checkpoint — NOT a capability gap. Carry only the
            # checkpoint (small, bounded), never a trace dump (invariant 4). Attach
            # the full disposition recs as evidence (invariant 4 prior, like phase0
            # diagnosis) so the agent sees utov's computed prior (recommended back /
            # ambiguity reason), not a bare question.
            ev = dict(_compact(result.to_dict()))
            if recs:
                ev["mem_disposition_recs"] = _compact(
                    [r.to_dict() for r in recs.values()])
            # Surface the cohort merge symmetry / degradation decision so the WARN
            # (asymmetric cohort / batch degradation) is visible in the gap map,
            # not buried inside the primitive (invariant 1: WARN at the boundary).
            if self._last_disp_diag:
                ev["mem_disposition_diagnostics"] = _compact(self._last_disp_diag)
            return Verdict(
                VStatus.PENDING,
                reason=f"agent judgment needed: {result.checkpoint.name}",
                evidence=ev,
                capability_request="")
        # DriveResult — evidence is a compact, necessary-only gap-map summary.
        # SAFETY GATE (closure-evidence layering): a window's ``closed`` means parity
        # EXACT for THAT window — it is NOT whole-case oracle closure. Stamp the
        # three-layer closure classification on EVERY confirmed window so a window-
        # local parity-EXACT is presented at its true level (candidate_formula when
        # on-path, local_formula otherwise), never auto-promoted to algorithm closure.
        if result.closed:
            closure = _window_closure(result, c, cc)
            ev = _drive_evidence(result, cc.window, cc.window_kind,
                                 disposition="closed")
            ev = dict(ev)
            ev["closure"] = closure
            if not closure.get("algorithm_closed"):
                # Window parity-EXACT but NOT oracle-closed (output sink not confirmed
                # / provenance not closed) → demote: it is a candidate / local formula,
                # carrying its trap marker LOUDLY (A8④). Still CONFIRMED at the WINDOW
                # level (the window transform holds), but the closure label tells the
                # consumer it is not a whole-case algorithm closure.
                ev["closure_trap"] = closure.get("trap_state")
            return Verdict(
                VStatus.CONFIRMED,
                evidence=ev,
                located_base=c.locus)
        disposition, reason = _classify_drive_result(result)
        evidence = _drive_evidence(result, cc.window, cc.window_kind,
                                   disposition=disposition)
        # Issue 7 — mem-sink read outcome (runs BEFORE the opaque / constant
        # terminals). The runner ran in EXPLICIT mem-sink mode but could not read the
        # sink bytes back symbolically. Two DISTINCT conditions, two DISTINCT routes
        # (A8④ — never conflated, never a silent fallback):
        #   * input-invariant store (no byte became symbolic — seed/driver-independent)
        #     → NOT unplaceable and NOT a new ad-hoc verdict: route through the EXISTING
        #     seed-independence exclusion (the seed_invariant TERMINAL). The store is
        #     simply not a recovery target (spec rule 4 / the F0 raw3 constant stores).
        #   * everything else (EA never resolved to where the value landed / Triton read
        #     failed) → MEM_SINK_UNPLACEABLE with the structured needed[] list.
        if mem_sink_desc is not None and result.mem_sink_unreadable:
            why = str(result.mem_sink_unreadable)
            if "input-invariant" in why:
                ev = dict(evidence)
                ev["mem_sink_seed_independent"] = why
                return Verdict(
                    VStatus.TERMINAL, terminal_kind="seed_invariant",
                    reason=("the mem-sink store is input-invariant (seed/driver-"
                            "independent) — not a recovery target; surfaced through "
                            "the seed-independence exclusion, NOT a mem-sink terminal: "
                            + why),
                    evidence=ev, located_base=c.locus, capability_request="")
            return self._mem_sink_unplaceable_verdict(c, mem_sink_desc, why)
        # Req5 — MEMORY_DISPOSITION_MISSING gate (runs BEFORE the opaque / constant
        # terminals). This window's external memory live-in was WHOLLY un-classified
        # (no symbolize-vs-back disposition available for ANY of it — the early map
        # does not cover this window's addresses). A resulting collapse to opaque, or
        # an emitted F that references no input (a constant collapse), is then the
        # MISSING decision, not a known algorithm property: route it to the honest
        # terminal that keeps "memory never classified" SEPARATE from "symbol genuinely
        # does not propagate" (A8④ / never the misleading opaque/constant mask).
        # Only fires when there IS undecided live-in AND the result actually collapsed
        # — a window with a decided/empty live-in, or one that did not collapse, takes
        # its existing path byte-for-byte (invariant 7).
        is_constant_collapse = (
            result.emitted_F is not None
            and not _f_references_inputs(result.emitted_F, cc.inputs))
        if mem_audit["all_undecided"] and (
                disposition == "opaque" or is_constant_collapse):
            return self._memory_disposition_missing_verdict(
                c, cc, evidence, mem_audit,
                collapse_disposition=("constant" if is_constant_collapse
                                      else "opaque"))
        # SAFETY GATE for the non-closed paths too: a window that emitted a constant
        # F (emit_picked_constant / opaque collapse) with no output-provenance support
        # is the PSEUDO_CLOSURE_TRAP — stamp the closure classification so the constant
        # is never silently advanced (task 2). Pure read; never feeds a gate (inv 7).
        if result.emitted_F is not None:
            evidence = dict(evidence)
            evidence["closure"] = _window_closure(result, c, cc)
        if disposition == "parity" and result.parity_report is not None:
            # Surface the parity floor numbers into ELIMINATED evidence so collect
            # can tell a feed pit (supplied < need) from an F error (supplied >=
            # need but matched < need) without re-deriving. _compact already caps
            # the per-vector list (invariant 4).
            pr = result.parity_report
            evidence = dict(evidence)
            evidence["parity_detail"] = _compact({
                "need":     pr.get("min_vectors"),
                "supplied": pr.get("total"),
                "matched":  pr.get("independent_pass"),
                "vectors":  pr.get("vectors"),
            })
            # Three-factor remedy tag (task 2 — factor 3 / diversify-seeds): the
            # post-hoc UNCLOSABLE verdict (setup_symex.check_parity_vectors) means the
            # window HAS variance and the cohort is big enough to reach the floor, yet
            # the INDEPENDENT side's observed outputs COLLIDE (observed_distinct <
            # min_vectors) — no F can EXACT-close. The implicit "fix the cohort with
            # output-diverse seeds" semantics of df0f95c is made an explicit, machine-
            # routable tag here: supply more OUTPUT-DIVERSE seeds (NOT just more — the
            # cohort is already deep enough; it is the outputs that are not distinct).
            # Distinct from add-seeds (cohort too SMALL) and re-anchor (no variance).
            # Pure evidence key — does not change the ELIMINATED verdict (inv 7); the
            # observed_distinct口径 is df0f95c's independent-side count (inv 8).
            if pr.get("verdict") == "UNCLOSABLE":
                od = pr.get("observed_distinct")
                mv = pr.get("min_vectors")
                evidence["remedy"] = REMEDY_DIVERSIFY_SEEDS
                evidence["remedy_reason"] = (
                    "cohort deep enough but the independent side's observed outputs "
                    f"collide (observed_distinct={od} < min_vectors={mv}) — no F can "
                    "EXACT-close; supply more OUTPUT-DIVERSE seeds (change WHICH "
                    "inputs, not just how many), do NOT keep tuning F or merely add "
                    "more same-output seeds")
        if disposition == "capability":
            return Verdict(VStatus.TERMINAL, terminal_kind="unmodeled_instruction",
                           reason=reason, evidence=evidence,
                           located_base=c.locus,
                           capability_request=reason)
        if disposition == "opaque":
            # Phase 0: split the opaque window into known_addr (→P1) vs
            # symbolic_address (→P2) vs inconclusive, and attach the diagnosis to
            # the gap-map evidence so collect no longer reports a flat "藏 staging".
            # Diagnosis is deterministic (DFG + cohort byte diff), zero LLM; the
            # large staging-byte list is digested inside StagingDiagnosis.to_dict
            # (invariant 4). Best-effort: a diagnosis error must not lose the verdict.
            diag = None
            try:
                diag = diagnose_opaque_staging(
                    state.items,
                    window=cc.window,
                    window_is_idx=(cc.window_kind == "idx"),
                    pointer_chain=self.pointer_chain,
                    cohort_traces=self.cohort_traces,
                    symbolic_inputs=(cc.symbolic_regs or ()),
                )
                evidence = dict(evidence)
                evidence["opaque_staging_diagnosis"] = _compact(diag.to_dict())
            except Exception as e:                     # diagnosis is advisory only
                evidence = dict(evidence)
                evidence["opaque_staging_diagnosis_error"] = str(e)
            # §需求1: split the opaque MERGE POINT into a specific block_kind (the
            # additive root-cause field). Deterministic; never feeds a gate. §需求3:
            # a symbol_not_on_output_path block carries its symbol_trace inside the
            # detail. §需求2: stamp the block_kind's capability coverage so the agent
            # sees whether THIS build carries the feature the terminal relies on.
            block_kind, block_detail = _classify_block_kind(
                result,
                items=state.items,
                window=cc.window,
                window_is_idx=(cc.window_kind == "idx"),
                inputs=cc.inputs,
                staging_diag=diag)
            evidence = dict(evidence)
            evidence["block_kind"] = block_kind
            evidence["block_kind_detail"] = _compact(block_detail)
            evidence.update(_terminal_coverage("opaque_staging"))
            return Verdict(VStatus.TERMINAL, terminal_kind="opaque_staging",
                           reason=reason, evidence=evidence,
                           located_base=c.locus,
                           capability_request=OPAQUE_STAGING_FRONTIER)
        if disposition == "seed_invariant":
            return Verdict(VStatus.TERMINAL, terminal_kind="seed_invariant",
                           reason=reason, evidence=evidence,
                           located_base=c.locus,
                           capability_request="")
        # Composite recovery (Req6): an on-path BAND whose ISOLATED parity did not close
        # the whole output is BAND_PARITY_FAIL — a SIGNAL (the slice is real but
        # incomplete), not silence. Route to the composite path: within budget + >= 2
        # adjacent bands → RUN the real chained-symbolic-state execution (compose each
        # band's symbolic output state into the next band's live-in, reg + mem, then
        # multi-vector parity on the composite); over budget → COMPOSITE_TOO_EXPENSIVE
        # (comfortable exit, not run); a band un-chainable at the emit layer → the
        # honest deep-primitive stop-report. Only for on-path band candidates; a
        # non-band parity ELIMINATED is byte-for-byte unchanged (invariant 7).
        if disposition == "parity" and (c.payload or {}).get("band") is True:
            return self._band_parity_verdict(c, cc, evidence, state)
        # fixable / unsound / parity → ELIMINATED, with a corrected spawn where the
        # fix is a well-defined geometry flip (try the other window convention).
        spawn = _corrected_spawn(c) if disposition == "fixable" else []
        return Verdict(VStatus.ELIMINATED, reason=reason, spawn=spawn,
                       evidence=evidence)

    def _band_parity_verdict(
        self, c: Candidate, cc: CaseConfig, evidence: dict[str, Any],
        state: CvdState,
    ) -> Verdict:
        """BAND_PARITY_FAIL + composite EXECUTION for an isolated on-path band (Req6).

        The band's own parity did NOT close the output. Assemble the composite plan
        over the adjacent on-path bands (``self.onpath_bands`` when the registry passed
        them, else just this band) and route:
          * COMPOSITE_TOO_EXPENSIVE — combined symex over budget (band list + estimate,
                                      a comfortable exit; execution NOT attempted);
          * COMPOSITE_REQUIRED      — adjacent bands available within budget → RUN the
                                      real chained-symbolic-state execution. If it
                                      closes parity → CONFIRMED (composite F); if it
                                      runs but parity fails → BAND_PARITY_FAIL (even the
                                      composite does not close — this band set is not the
                                      whole algorithm / still missing a segment); if a
                                      band cannot be chained at the emit layer → the
                                      COMPOSITE_REQUIRED terminal naming the primitive
                                      gap (deeper concolic-seed injection), never faked;
          * BAND_PARITY_FAIL        — a lone band with no neighbour to combine (signal).
        """
        this_band = _band_window(c.payload or {})
        # A3 collect-layer aggregation: plan the composite over the WHOLE same-chain_id
        # band group, not just the one band drive handed us. Source priority:
        #   1. the run-level shared band_registry, looked up by THIS candidate's
        #      chain_id (the generator recorded every same-chain band there) — this is
        #      what turns N isolated BAND_PARITY_FAILs into one COMPOSITE plan;
        #   2. else the static onpath_bands (legacy direct injection);
        #   3. else this band alone (lone band → stays BAND_PARITY_FAIL, the signal).
        chain_id = (c.payload or {}).get("chain_id")
        group = (self.band_registry.group(chain_id)
                 if self.band_registry is not None and chain_id is not None else [])
        if group:
            bands = group
        elif self.onpath_bands:
            bands = list(self.onpath_bands)
        else:
            bands = [this_band] if this_band is not None else []
        plan = plan_composite_recovery(bands, budget=self.budget)
        ev = dict(evidence)
        ev["composite_plan"] = _compact(plan)
        ev["band"] = list(this_band) if this_band else None
        ev["chain_band_group"] = [list(b) for b in bands]
        ev["chain_band_group_size"] = len(bands)
        ev["composite_aggregation_min"] = self.budget.composite_aggregation_min
        terminal = plan.get("terminal") if plan else None

        if terminal == TERMINAL_COMPOSITE_TOO_EXPENSIVE:
            # Over budget — a comfortable exit (band list + estimate), do NOT run.
            return Verdict(
                VStatus.TERMINAL, terminal_kind=terminal,
                reason=plan.get("reason", ""), evidence=ev,
                located_base=c.locus,
                capability_request=(
                    "raise budget.max_composite_symex_items or narrow the band set"))

        if terminal == TERMINAL_COMPOSITE_REQUIRED:
            # Within budget + >= 2 adjacent bands → RUN the real chained execution.
            comp = self._execute_composite(
                sorted({(int(lo), int(hi)) for lo, hi in bands}), cc, state)
            ev["composite_execution"] = _compact(comp)
            if comp.get("closed"):
                # The composite F closed multi-vector parity → CONFIRMED, carrying the
                # real composed expression + its parity (the G4/parity gates were NOT
                # relaxed — composite is one more verified expression).
                ev["composite_F"] = comp.get("composite_F")
                ev["parity"] = comp.get("parity")
                return Verdict(
                    VStatus.CONFIRMED, evidence=ev, located_base=c.locus,
                    reason=("chained-symbolic-state composite closed: composed the "
                            "adjacent bands' symbolic output states (reg + mem "
                            "hand-off) into one expression that passed multi-vector "
                            "parity"))
            if comp.get("primitive_gap"):
                # A band could NOT be chained at the emit layer → the honest deep-
                # primitive stop-report (which band, what change is needed). The reg/
                # emit-composable hand-off that DID chain is reported as progress.
                return Verdict(
                    VStatus.TERMINAL, terminal_kind=terminal,
                    reason=comp.get("reason", plan.get("reason", "")),
                    evidence=ev, located_base=c.locus,
                    capability_request=(
                        "implement the deeper concolic-seed primitive: inject a band's "
                        "symbolic OUTPUT expression as the next window's symbolic "
                        "live-in seed inside one shared concolic context (the Level-2 "
                        f"runner stops at band {comp.get('stopped_at')}); the emit-layer "
                        f"reg+mem hand-off chained {comp.get('n_chained')} band(s)"))
            # The composite RAN (chained the bands) but its parity did not close →
            # BAND_PARITY_FAIL: even composed, this band set is not the whole algorithm
            # (or still missing a segment). The composite F + parity ride the evidence
            # (it is a real run, not a scaffold — A8④).
            ev["composite_F"] = comp.get("composite_F")
            ev["parity"] = comp.get("parity")
            return Verdict(
                VStatus.TERMINAL, terminal_kind=TERMINAL_BAND_PARITY_FAIL,
                reason=("chained-symbolic-state composite RAN but its parity did not "
                        "close — even composed, these adjacent bands are not the whole "
                        "algorithm (a segment is still missing or off-set); the "
                        "composed expression + its parity ride the evidence"),
                evidence=ev, located_base=c.locus, capability_request="")

        # lone band / no plan → BAND_PARITY_FAIL: the isolated slice is the signal.
        return Verdict(
            VStatus.TERMINAL, terminal_kind=TERMINAL_BAND_PARITY_FAIL,
            reason=(plan.get("reason") if plan else
                    "isolated on-path band failed parity — the slice is real but does "
                    "not close the whole output on its own (no adjacent band to "
                    "combine)"),
            evidence=ev, located_base=c.locus,
            capability_request="")

    def _execute_composite(
        self, bands: Sequence[tuple[int, int]], cc: CaseConfig, state: CvdState,
    ) -> dict[str, Any]:
        """RUN the chained-symbolic-state composite over adjacent on-path bands (Req6).

        For each band IN PRODUCER-CHAIN ORDER: run its symex (``drive`` on the band's
        window) to obtain its emitted transform — the band's symbolic OUTPUT state as a
        closed-form function of its symbolic live-in. Then COMPOSE the bands
        (:func:`compose_band_transforms`): each band's live-in symbol is bound to the
        previous band's output expression — the reg + mem hand-off (the producer→
        consumer edge決定 which output threads into which live-in; a mem hand-off via a
        staging/heap cell is the SAME named-symbol substitution as a reg one). Finally
        run MULTI-VECTOR parity on the composite over the cohort (observed = the END
        band's true exit output per vector; predicted = composite F on that vector's
        original seed) — the G4/parity/seed gates are NOT relaxed.

        Returns ``{closed, composite_F, parity, n_chained, primitive_gap, stopped_at,
        per_band, reason}``. ``primitive_gap`` is True (with ``stopped_at``) when a band
        did not emit a chainable named transform — the deep concolic-seed primitive the
        Level-2 runner does not expose; surfaced, never faked. Bounded by the SAME
        ``max_composite_symex_items`` the plan estimated against (defensive double-gate
        — the plan already routed COMPOSITE_TOO_EXPENSIVE; here we re-check the live
        per-band symex item count and bail to a primitive-gap-shaped honest stop)."""
        from ..setup_symex import _eval_emitted_on_seed, ParityVector, check_parity_vectors
        items = list(state.items)
        per_band: list[dict[str, Any]] = []
        band_fns: list[tuple[str, str]] = []
        for (lo, hi) in bands:
            band_cc = replace(cc, window=(lo, hi), window_kind="idx")
            result = drive(trace=items, case_config=band_cc,
                           triton_runner=self.triton_runner, ledger=self.ledger,
                           decisions=dict(self.decisions),
                           pointer_chain=self.pointer_chain,
                           cohort_traces=self.cohort_traces,
                           cohort_keys=self.input_keys)
            label = f"[{lo},{hi}]"
            emitted = getattr(result, "emitted_F", None) if not isinstance(
                result, DrivePause) else None
            per_band.append({"band": [lo, hi], "emitted": bool(emitted)})
            band_fns.append((label, emitted or ""))
        composed = compose_band_transforms(band_fns, outer_input=cc.inputs[0]
                                           if cc.inputs else "carrier")
        base = {
            "bands": [list(b) for b in bands],
            "n_chained": composed.get("n_chained", 0),
            "per_band": per_band,
            "composite_F": composed.get("composite_F"),
        }
        if not composed.get("ok"):
            # A band's symbolic output state could not be chained at the emit layer —
            # the deep concolic-seed primitive. Honest stop-report (never faked).
            return {**base, "closed": False, "primitive_gap": True,
                    "stopped_at": composed.get("stopped_at"),
                    "parity": None, "reason": composed.get("reason")}
        composite_F = composed["composite_F"]
        # Multi-vector parity on the composite: observed = each cohort vector's true
        # WHOLE-output (the end band's exit, computed by the runner's oracle); predicted
        # = composite F on that vector's own original seed. Reuse the end band's drive
        # facts per vector (the runner surfaces trace_self_check sink + seed values).
        inputs = list(cc.inputs) if cc.inputs else ["carrier"]
        end_lo, end_hi = bands[-1]
        vecs: list[Any] = []
        for k, ct in enumerate(self.cohort_traces):
            ctl = list(ct or ())
            if not ctl:
                continue
            key = (str(self.input_keys[k]) if self.input_keys
                   and k < len(self.input_keys) and self.input_keys[k] is not None
                   else f"cohort-{k}")
            try:
                run_v = dict(self.triton_runner({
                    "entry": {"entry_pc": cc.entry_pc,
                              "symbolic_regs": list(cc.symbolic_regs or cc.reg_file)},
                    "mode": "forward_symbolic",
                    "window": [end_lo, end_hi], "window_kind": "idx",
                    "items": ctl, "decisions": dict(self.decisions)}))
            except Exception:
                continue
            tsc = run_v.get("trace_self_check") or {}
            observed = tsc.get("sink_value")
            seed_v = tsc.get("seed_values")
            if observed is None or seed_v is None:
                continue
            ok, value, _why = _eval_emitted_on_seed(composite_F, inputs, dict(seed_v))
            if not ok:
                continue
            exec_id = run_v.get("exec_id") or f"composite-{k}:{key}"
            vecs.append(ParityVector(
                input_key=key, observed=str(observed), predicted=str(value),
                exec_id=str(exec_id)))
        from ..setup_symex import SetupSymexConfig
        min_vectors = SetupSymexConfig.from_env().parity_min_vectors
        report = check_parity_vectors(
            vecs, window=(end_lo, end_hi), min_vectors=min_vectors)
        parity = f"{report.independent_pass}/{report.counted}"
        return {
            **base, "closed": bool(report.sufficient), "primitive_gap": False,
            "stopped_at": None, "parity": parity,
            "parity_verdict": report.verdict,
            "reason": (composed.get("reason", "") + f"; composite parity {parity} "
                       f"({report.verdict})"),
        }


def _corrected_spawn(c: Candidate) -> list[Candidate]:
    """A well-defined geometry correction: flip the window convention (idx↔pc).

    This is the one mechanical retry CVD can self-route without agent input; other
    fixes (re-capture, inject backing) need data CVD does not hold, so they stay as
    an ELIMINATED reason for the agent, not an auto-spawn."""
    p = dict(c.payload or {})
    if p.get("_geometry_flipped"):
        return []   # already tried the flip — do not loop
    cur = p.get("window_kind", "idx")
    flipped = "pc" if cur == "idx" else "idx"
    p2 = dict(p, window_kind=flipped, _geometry_flipped=True)
    return [Candidate(
        RECOVER_WINDOW, locus=c.locus, signal="geometry_retry",
        entry_reason=f"retry {c.entry_reason} with window_kind={flipped}",
        base_value=c.base_value, payload=p2)]


# --------------------------------------------------------------------------- #
# 3 — TerminalClassifier: claim the genuine global dead ends.
# --------------------------------------------------------------------------- #

