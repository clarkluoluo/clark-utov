"""cvd_recovery.generator section (split from the monolithic module)."""
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
from ._cohort import RECOVER_WINDOW, SIG_GENERATION_BUDGET_EXHAUSTED, SIG_PROVENANCE_BLOCKED_UNPLACEABLE, SIG_PROVENANCE_OFFPATH_VARIANCE, SIG_PROVENANCE_ONPATH, SIG_PROVENANCE_UNANCHORED, SIG_RECAPTURE_DIRECTIVE, SRC_OUTPUT_PROVENANCE, _OnpathBandRegistry, _ProvenanceAnchor, _compact, _placeable_next_watch, _unplaceable_next_watch


class RecoveryWindowGenerator(CandidateGenerator):
    """Produce one ``recover_window`` candidate per window worth a symex run.

    PRIMARY anchor — output provenance (when the target output is known):
      * ``sink_base`` + the run's ``expected`` oracle bytes locate the target
        output; :func:`oracle_provenance.trace_provenance` backtraces its producer
        chain. Candidates are generated ONLY on that provenance path (``on_path=
        True`` + a ``path_distance`` from the sink). cohort/dispatch variance is
        DEMOTED to a secondary ordering filter WITHIN those on-path windows (the
        ``_distinct_potential`` bonus), never the primary selector — an input-
        varying-but-off-path window (e.g. a dispatch intermediate that never feeds
        the output) is no longer surfaced as a main candidate. This is the F0-A
        root-cause fix: variance found "where the input reaches state", which is NOT
        the same as "on the path that produces the target output".
      * If the output writer was never OBSERVED (``sink_captured`` False /
        ``NEEDS_OBSERVATION``) → a :func:`recapture_target.derive_recapture_directive`
        candidate is produced (collect the output first, then re-anchor), NOT a
        silent fall-back to off-path variance windows (that silent fall-back was the
        绕路 root cause).
      * If the output cannot be anchored to any provenance at all → an EXPLICIT
        unanchored diagnostic candidate (A8④ degradation), not a silent fall-back.

    SECONDARY / fall-back sources (used as-is when NO target output is supplied —
    invariant 7, byte-for-byte today):
      * a :class:`CoverageMap` — the ~N handler-TYPE representative windows (solve
        each type once, compose along the sequence);
      * an :class:`InputDependenceMap` — the ``localized`` seed-varying positions
        (recover F only where the seed actually reaches observable state).

    Windows are trace-idx bands (``window_kind='idx'``), matching how the Level-2
    runner and the coverage map bound a handler body."""

    name = "recovery_window_gen"
    version = "1"
    owner = "core"
    kind = RECOVER_WINDOW

    def __init__(
        self,
        *,
        coverage: CoverageMap | None = None,
        dependence: InputDependenceMap | None = None,
        window_kind: str = "idx",
        sink_base: int | None = None,
        value_name: str = "target_output",
        budget: "CvdBudget | None" = None,
        band_registry: "_OnpathBandRegistry | None" = None,
    ) -> None:
        self.coverage = coverage
        self.dependence = dependence
        self.window_kind = window_kind
        # Generation/backtrace budget (dev-recovery-generation-budget-spec): bounds
        # the provenance backtrace (depth/breadth) and caps the on-path candidate
        # count (ROI-ordered top-N). The SAME CvdBudget the verify loop uses — not a
        # new budget system; the recovery generator just reads its three generation
        # fields. A default CvdBudget()'s generous ceilings never trip a budget-
        # internal case (invariant 7); only a long trace / wide fan-out / candidate
        # explosion does.
        self.budget = budget or CvdBudget()
        # Output-provenance PRIMARY anchor (the F0-A root-cause fix). ``sink_base``
        # is THIS case's located target-output address (case-specific → injected,
        # never hardcoded; from ``base_config.sink_hint_addr``). The oracle expected
        # bytes arrive per-run on ``state.expected``. Both present (sink_base set +
        # a non-trivial expected) → provenance-anchored generation is the primary
        # source; absent → invariant 7, today's coverage/variance generation
        # byte-for-byte. ``value_name`` only labels the recapture directive.
        self.sink_base = sink_base
        self.value_name = value_name
        # The non-window provenance signals (recapture directive / unanchored
        # diagnostic) are emitted at most ONCE per run — the same widen-regenerate
        # cycle that motivates ``_opaque_windows_emitted`` would otherwise re-offer
        # them forever. They are not windows, so they get their own one-shot guard.
        self._provenance_signal_emitted = False
        # 坎2 anti-loop: the opaque-window candidate is emitted at most ONCE per
        # run. The generator instance persists across the driver's widen-regenerate
        # cycles (_widen builds a fresh CvdState but reuses the registry, hence this
        # generator), so instance state — not the disposable CvdState — is the
        # correct place to remember an already-emitted opaque window. Without it,
        # every widen would regenerate the same window (verify → still opaque →
        # widen → regenerate …), an infinite loop. Once emitted, the window is not
        # re-offered → frontier truly empties → the terminal classifier claims, but
        # only AFTER the per-window forward has run and left its symbolic_forwards.
        self._opaque_windows_emitted: set[tuple[int, int]] = set()
        # A3 collect-layer aggregation: the run-level chain_id -> [bands] index. The
        # generator records every on-path band here so the verifier can plan a
        # composite over the WHOLE same-chain group (not the one band drive hands it).
        # Shared by construction (recovery_registry wires the same instance into the
        # verifier); None → a private index (still correct within this generator,
        # group-aware ranking still works; the verifier just won't see cross-candidate
        # groups — graceful, never an error).
        self.band_registry = band_registry or _OnpathBandRegistry()

    # On-path windows are the PRIMARY tier; an off-path (variance-only) window is
    # never allowed to out-rank them. Base tiers (a window's source determines its
    # floor; the small ``_distinct_potential`` bonus only re-orders WITHIN a tier):
    #   on-path provenance window           : _BASE_ONPATH (top — the output path)
    #   coverage / variance window, on-path  : their own base (4.0/5.0) + on_path tag
    #   coverage / variance window, off-path : their base MINUS _OFFPATH_PENALTY,
    #                                          floored, so it sinks below every
    #                                          on-path window (visible, never主候选)
    _BASE_ONPATH = 6.0
    _OFFPATH_PENALTY = 5.0          # demote off-path variance windows below on-path

    def generate(self, state: CvdState, *, diag: list | None = None) -> list[Candidate]:
        cands: list[Candidate] = []

        def _log(event: str, **kw) -> None:
            """Progress / truncation breadcrumb into the run's records (if wired).

            The generation/backtrace phase is otherwise a black box on a long trace
            (the F0-A 8.5min-no-output symptom). These let a human/agent see "still
            advancing vs stuck" and exactly what a budget cut dropped (A8④)."""
            if diag is not None:
                diag.append({"event": event, "phase": "generation",
                             "tool": f"{self.name}@{self.version}", **kw})

        # ---- PRIMARY anchor: output provenance (only when a target output exists) ----
        # ``anchor`` is None when no target output was supplied (sink_base is None or
        # the run carries no/empty expected) → invariant 7: the coverage/variance
        # generation below is byte-for-byte today's behaviour. When a target output
        # IS supplied, ``anchor`` carries the provenance result: either on-path window
        # candidates (the primary tier) and the on-path idx set used to demote off-
        # path variance windows, or a non-window terminal-shaped candidate (recapture
        # directive / explicit unanchored) that REPLACES the variance fall-back.
        anchor = self._output_provenance_anchor(state, log=_log)
        onpath_idxs: set[int] | None = None
        if anchor is not None:
            cands.extend(anchor.candidates)
            onpath_idxs = anchor.onpath_idxs    # None ⇒ no on-path window to filter on
            if anchor.suppress_secondary:
                # The output writer was never observed, or the output cannot be
                # anchored to any provenance: emit the recapture / unanchored
                # candidate and STOP — do NOT silently fall back to off-path variance
                # windows (that silent fall-back is the 绕路 root cause / A8④).
                return self._cap_candidates(cands, log=_log)

        def _on_path(lo: int, hi: int) -> bool | None:
            """on-path iff any in-band idx is a provenance producer. None when no
            provenance anchor (today's windows carry no on/off-path axis)."""
            if onpath_idxs is None:
                return None
            return any(lo <= i <= hi for i in onpath_idxs)

        if self.coverage is not None:
            for t in self.coverage.types:
                lo, hi = t.representative
                op = _on_path(lo, hi)
                base, extra = self._secondary_tier(4.0, op, lo, hi, anchor)
                cands.append(Candidate(
                    RECOVER_WINDOW, locus=lo,
                    signal=("dispatch_type_rep" if op is not False
                            else SIG_PROVENANCE_OFFPATH_VARIANCE),
                    entry_reason=(f"handler type {t.type_id} representative window "
                                  f"idx[{lo},{hi}] (×{t.occurrences})"
                                  + ("" if op is None else
                                     f" [{'on' if op else 'off'}-path]")),
                    base_value=base,
                    payload=_compact({
                        "window": [lo, hi], "window_kind": self.window_kind,
                        "source": "dispatch_coverage", "type_id": t.type_id,
                        "occurrences": t.occurrences,
                        "reg_live_in": list(t.reg_live_in),
                        "mem_live_in": [m.to_dict() for m in t.mem_live_in],
                        "unmodeled_opcodes": list(t.unmodeled_opcodes),
                        **extra,
                    })))
        if self.dependence is not None and self.dependence.verdict == "localized":
            # Backtracking is ANCHORED, not free-roaming (output-backtrack addendum
            # §2): divergence_idx — where the seed first reaches observable state —
            # is the real start point. It is the anchor the recover/provenance
            # Verifier backtracks from; the other varying positions follow. CVD
            # reuses the native provenance machinery (drive's own backtrace) — this
            # only supplies the anchored candidates, it does NOT rebuild backtracking.
            div = self.dependence.divergence_idx
            n_vectors = self.dependence.n_vectors
            for p in self.dependence.varying:
                is_anchor = (p.idx == div)
                # 独立侧 distinct 潜力 (task 1): a PRE-SYMEX proxy for how distinct the
                # independent side could be once symex runs — added ON TOP of today's
                # fixed base (anchor 5.0 / other 4.0). Higher potential → higher
                # base_value → CVD frontier verifies it FIRST (spend the symex budget
                # on the windows most likely to EXACT-close; low-potential windows are
                # still early-BLOCKed by the pre-flight gate). Invariant 7: a position
                # with no varying signal contributes a 0.0 bonus → byte-for-byte
                # today's fixed base_value (the保序兜底 below proves it in a fixture).
                bonus = self._distinct_potential(
                    [p], n_vectors=n_vectors, divergence_idx=div)
                # When a provenance anchor is active, variance is SECONDARY: a varying
                # position ON the output path keeps its tier (the bonus orders within
                # it — c5f936a's distinct-potential is exactly this secondary filter);
                # a varying position OFF the path (high-variance dispatch intermediate
                # that never feeds the output — the F0-A off-path windows) is DEMOTED
                # below every on-path window so the frontier never picks it as the main
                # candidate. None ⇒ no provenance anchor → today's behaviour exactly.
                op = _on_path(p.idx, p.idx)
                base, extra = self._secondary_tier(
                    (5.0 if is_anchor else 4.0) + bonus, op, p.idx, p.idx, anchor)
                signal = ("divergence_anchor" if is_anchor else "input_varying")
                if op is False:
                    signal = SIG_PROVENANCE_OFFPATH_VARIANCE
                cands.append(Candidate(
                    RECOVER_WINDOW, locus=p.idx,
                    signal=signal,
                    entry_reason=(
                        (f"DIVERGENCE ANCHOR idx{p.idx} — seed's first observable "
                         "entry point (backtrack start)" if is_anchor else
                         f"seed-varying position idx{p.idx}")
                        + f" (regs={list(p.varying_regs)} "
                          f"mem={[hex(a) for a in p.varying_mem]})"
                        + ("" if op is None else
                           f" [{'on' if op else 'off'}-path — variance is "
                           f"{'corroborating' if op else 'OFF the output path, demoted'}]")),
                    base_value=base,
                    payload=_compact({
                        "window": [p.idx, p.idx], "window_kind": "idx",
                        "source": "cohort_diff",
                        "anchor": "divergence_idx" if is_anchor else None,
                        "divergence_idx": div,
                        "varying_regs": list(p.varying_regs),
                        "varying_mem": list(p.varying_mem),
                        "control_flow": p.control_flow,
                        "distinct_potential": round(bonus, 4),
                        **extra,
                    })))
        # 坎2 — opaque is NOT a direct global dead end: try a per-window forward
        # ONCE before the terminal classifier claims. An ``opaque`` cohort carries
        # an ``opaque_staging_advisory`` (the EA-varying staging PCs) — the very
        # window coordinates the per-window Verifier needs to run drive()'s opaque
        # fallback forward (0388e3b) and leave a ``symbolic_forwards`` count. With
        # no coverage and no localized varying, this is the ONLY candidate source,
        # so without it the frontier starts empty and the terminal claims opaque
        # before any forward ran (clark's "抢跑"). Emitted only on verdict==opaque
        # + a real advisory window + not already tried (invariant 7: every other
        # path is byte-for-byte unchanged).
        if (self.dependence is not None
                and self.dependence.verdict == "opaque"
                and self.dependence.opaque_staging_advisory is not None):
            win = self._opaque_advisory_window(self.dependence.opaque_staging_advisory)
            if win is not None and win not in self._opaque_windows_emitted:
                self._opaque_windows_emitted.add(win)
                lo, hi = win
                cands.append(Candidate(
                    RECOVER_WINDOW, locus=lo, signal="opaque_staging_forward",
                    entry_reason=(
                        f"opaque cohort — run a per-window forward over the EA-varying "
                        f"staging window idx[{lo},{hi}] BEFORE claiming the opaque "
                        f"terminal (try the symbolic forward once; if it still "
                        f"collapses, the terminal claim then ships a symbolic_forwards "
                        f"count as evidence it was tried)"),
                    base_value=4.0,
                    payload=_compact({
                        "window": [lo, hi], "window_kind": "idx",
                        "source": "opaque_staging_advisory",
                    })))
        return self._cap_candidates(cands, log=_log)

    # ----------------------------------------------------------------------- #
    # On-path candidate cap (dev-recovery-generation-budget-spec, task 2). The
    # generated candidate set is bounded by ``budget.max_gen_candidates`` —
    # ROI-ORDERED (by base_value: distance-to-output / on-path / distinct-potential
    # are already folded into base_value by 8fad88f + c5f936a), keep the top-N, drop
    # the long tail. The drop is NEVER silent: an explicit
    # GENERATION_BUDGET_EXHAUSTED candidate carries what was cut, how many, and the
    # retained order (A8④ / No silent caps). A budget-internal set (<= cap, the
    # common case) is returned UNTOUCHED — same objects, same order (invariant 7).
    # ----------------------------------------------------------------------- #

    def _cap_candidates(
        self, cands: list[Candidate], *, log: "Callable[..., None] | None" = None,
    ) -> list[Candidate]:
        cap = self.budget.max_gen_candidates
        if cap is None or len(cands) <= cap:
            # Within budget → byte-for-byte today's behaviour (invariant 7). No cap
            # field, no reorder, no extra candidate — the list is returned as-is.
            return cands
        # ROI order = base_value descending (the dynamic ROI weights already encode
        # near-sink / on-path / distinct-potential). Stable sort preserves the
        # generator's original order WITHIN a tie, so the retained set is deterministic.
        ordered = sorted(cands, key=lambda c: -c.base_value)
        kept = ordered[:cap]
        dropped = ordered[cap:]
        dropped_signals: dict[str, int] = {}
        for c in dropped:
            dropped_signals[c.signal] = dropped_signals.get(c.signal, 0) + 1
        report = {
            "stage": "candidate_cap",
            "max_gen_candidates": cap,
            "generated": len(cands),
            "kept": len(kept),
            "dropped": len(dropped),
            "retained_order": "base_value (ROI) descending — near-sink / on-path / "
                              "distinct-potential first; the long tail is dropped",
            "dropped_by_signal": dropped_signals,
            "kept_base_value_range": [round(kept[-1].base_value, 3),
                                      round(kept[0].base_value, 3)],
        }
        if log is not None:
            log("GENERATION_BUDGET_EXHAUSTED", **report)
        # Explicit, non-silent truncation candidate — top base_value so a consumer
        # sees it first and knows the generation set was capped (not exhausted search).
        kept.append(Candidate(
            RECOVER_WINDOW, locus=self.sink_base or 0,
            signal=SIG_GENERATION_BUDGET_EXHAUSTED,
            entry_reason=(
                f"generation budget exhausted — {len(cands)} candidate windows "
                f"generated, capped to the top {cap} by ROI (base_value); "
                f"{len(dropped)} long-tail window(s) dropped. Raise "
                f"budget.max_gen_candidates or narrow the goal to recover them."),
            base_value=self._BASE_ONPATH + 3.0,
            payload=_compact({
                "source": "generation_budget",
                "budget_exhausted": True,
                **report,
            })))
        return kept

    # ----------------------------------------------------------------------- #
    # Output-provenance PRIMARY anchor (dev-output-provenance-anchored-window-
    # gen-spec.md). Reuses oracle_provenance.trace_provenance + recapture_target —
    # builds nothing new; only re-orders "what is the primary anchor".
    # ----------------------------------------------------------------------- #

    def _output_provenance_anchor(
        self, state: CvdState, *, log: "Callable[..., None] | None" = None,
    ) -> "_ProvenanceAnchor | None":
        """Anchor window generation to what feeds the target output.

        Returns ``None`` (→ invariant 7, today's coverage/variance generation
        byte-for-byte) when NO target output is supplied: ``sink_base`` is None, or
        the run carries no/empty ``expected`` oracle bytes. Otherwise backtraces the
        output's producer chain (``trace_provenance``) and returns a
        :class:`_ProvenanceAnchor`:

          * output writer never OBSERVED (``sink_captured`` False / a
            NEEDS_OBSERVATION verdict) → a single recapture-directive candidate +
            ``suppress_secondary`` (collect the output, then re-anchor; NEVER fall
            back to off-path variance — the 绕路 root cause).
          * producer chain found → on-path window candidates (the primary tier) +
            the on-path idx set (used to demote off-path variance windows).
          * output anchored nowhere (no producer chain, no observation) → an EXPLICIT
            unanchored diagnostic candidate + ``suppress_secondary`` (A8④)."""
        if self.sink_base is None:
            return None
        expected = bytes(state.expected or b"")
        # An empty / 1-byte sentinel expected (the b"\x00" many call sites pass when
        # there is no real oracle) is NOT a target output to anchor on — invariant 7.
        if len(expected) < 2:
            return None
        if self._provenance_signal_emitted:
            # The recapture / unanchored signal is one-shot; once emitted, this
            # generator contributes no further provenance candidates on a re-generate.
            # (On-path window anchors are dedup'd by the driver's per-window logic;
            # re-deriving them is harmless but we avoid re-emitting the terminal one.)
            return None
        items = list(state.items)
        # ROUTING SHORT-CIRCUIT (dev-closure-evidence-layering, task 3): when the
        # OUTPUT SINK is not even captured on this run, the expensive provenance
        # backtrace can only end at NEEDS_OBSERVATION — so DO NOT pay for it. Run the
        # CHEAP sink check first (validate_sink, the existing primitive) and, if the
        # output is OUTPUT_NOT_OBSERVABLE, route straight to a recapture directive
        # BEFORE the heavy generation/backtrace. This is the F0-A "8.5min-no-output"
        # fix: the unobserved gate must run before, not after, the costly walk
        # (commit a420cf9 only BOUNDED the walk; it did not re-order it). dispatch /
        # variance fall-back is NEVER allowed here (suppress_secondary). Reuses
        # oracle_sink.validate_sink — builds nothing new (A8①).
        try:
            from ..oracle_sink import SinkVerdict as _SV, validate_sink as _vsink
            sv = _vsink(items, expected, snapshots=state.snapshots)
        except Exception:
            sv = None
        if sv is not None and sv.verdict is _SV.OUTPUT_NOT_OBSERVABLE:
            if log is not None:
                log("SINK_UNOBSERVED_SHORT_CIRCUIT",
                    sink_verdict=sv.verdict.value,
                    detail="output sink not captured — recapture directive BEFORE "
                           "the provenance backtrace (no long generation)")
            self._provenance_signal_emitted = True
            return _ProvenanceAnchor(
                candidates=[self._recapture_candidate_from_sink(items, expected, sv)],
                onpath_idxs=None, onpath_distance={}, suppress_secondary=True)
        if log is not None:
            log("BACKTRACE_START", trace_len=len(items),
                sink_base=f"0x{self.sink_base:x}",
                max_backtrace_depth=self.budget.max_backtrace_depth,
                max_backtrace_breadth=self.budget.max_backtrace_breadth)
        try:
            # Backtrace is BOUNDED (dev-recovery-generation-budget-spec): depth via
            # max_steps, breadth via max_breadth — the SAME CvdBudget ceilings the
            # verify loop honours, applied upstream so a 57k-step trace cannot make
            # the producer walk O(huge) (the F0-A black-box symptom).
            prov = trace_provenance(
                items, expected, sink_base=self.sink_base,
                snapshots=state.snapshots, assess_observability=True,
                max_steps=self.budget.max_backtrace_depth,
                max_breadth=self.budget.max_backtrace_breadth)
        except Exception as e:
            # Provenance is the primary anchor, but a backtrace error must not lose
            # the run — degrade EXPLICITLY to an unanchored report, do NOT silently
            # fall back to variance (A8④ / invariant: degradation is never silent).
            self._provenance_signal_emitted = True
            return _ProvenanceAnchor(
                candidates=[self._unanchored_candidate(
                    detail=f"trace_provenance raised {type(e).__name__}: {e}",
                    prov=None)],
                onpath_idxs=None, onpath_distance={}, suppress_secondary=True)

        if log is not None:
            log("BACKTRACE_DONE", verdict=prov.verdict.value,
                chain_len=len(prov.chain or []),
                truncated=prov.backtrace_truncated)
            if prov.backtrace_truncated is not None:
                # The producer walk hit a depth/breadth ceiling — surface it as an
                # explicit GENERATION_BUDGET_EXHAUSTED breadcrumb (never silent, A8④).
                log("GENERATION_BUDGET_EXHAUSTED", stage="backtrace",
                    detail=prov.backtrace_truncated)

        # Output sink not confirmed on this run → recapture FIRST (safety gate; the
        # sink-未确认优先-recapture invariant is untouched by A2 — this branch is not
        # in scope and stays exactly as before).
        if prov.sink_captured is False:
            self._provenance_signal_emitted = True
            return _ProvenanceAnchor(
                candidates=[self._recapture_candidate(items, expected, prov)],
                onpath_idxs=None, onpath_distance={}, suppress_secondary=True)

        # NEEDS_OBSERVATION title vs. the gap list (A2). recapture_loop already owns the
        # single-source closure predicate (``_is_closed`` — NEEDS_OBSERVATION + EMPTY
        # next_watch = every producer is captured = CLOSED). cvd_recovery used to honour
        # only the title and bounce EVERY NEEDS_OBSERVATION back to recapture even when
        # the gap list was already empty (the provenance was closed). Three-way split,
        # all explicit (A8④), conservatively BLOCKED over a false-close:
        #   (a) closed (empty next_watch)        → fall through to on-path candidates.
        #   (b) placeable gaps remain (pc set)   → recapture directive (regression).
        #   (c) ONLY unplaceable gaps (pc None)  → BLOCKED, NOT closed, NOT recapturable.
        if prov.verdict is ProvenanceVerdict.NEEDS_OBSERVATION:
            if not _provenance_closed_by_observation(prov):
                placeable = _placeable_next_watch(prov)
                unplaceable = _unplaceable_next_watch(prov)
                if placeable:
                    # (b) a hookable gap still open → recapture, exactly as before.
                    self._provenance_signal_emitted = True
                    return _ProvenanceAnchor(
                        candidates=[self._recapture_candidate(items, expected, prov)],
                        onpath_idxs=None, onpath_distance={}, suppress_secondary=True)
                # (c) gaps remain but NONE is placeable → cannot close, cannot recapture
                # (no PC to arm a watch). BLOCKED — never silently treated as closed.
                if log is not None:
                    log("PROVENANCE_BLOCKED_UNPLACEABLE",
                        n_unplaceable=len(unplaceable),
                        detail="NEEDS_OBSERVATION with only un-hookable gaps (no "
                               "reading PC) — neither closed nor recapturable")
                self._provenance_signal_emitted = True
                return _ProvenanceAnchor(
                    candidates=[self._unplaceable_blocked_candidate(prov, unplaceable)],
                    onpath_idxs=None, onpath_distance={}, suppress_secondary=True)
            # (a) closed: gap list empty → DO NOT recapture; continue to on-path.

        # Build the on-path producer-chain idx set + per-idx distance-from-sink.
        onpath_idxs, distance = self._provenance_onpath(items, prov)
        if not onpath_idxs:
            # Output is observed but no producer chain could be anchored (e.g. an
            # OPAQUE_CALLEE with nothing on our side, or an empty chain) → explicit
            # unanchored report, never a silent variance fall-back (A8④).
            self._provenance_signal_emitted = True
            return _ProvenanceAnchor(
                candidates=[self._unanchored_candidate(
                    detail=("output is observed but its production is not anchorable "
                            "to a traced producer chain on this side"),
                    prov=prov)],
                onpath_idxs=None, onpath_distance={}, suppress_secondary=True)

        cands = self._onpath_window_candidates(prov, onpath_idxs, distance)
        if log is not None:
            log("ONPATH_CANDIDATES", n=len(cands), onpath_idxs=len(onpath_idxs))
        return _ProvenanceAnchor(
            candidates=cands, onpath_idxs=onpath_idxs,
            onpath_distance=distance, suppress_secondary=False)

    @staticmethod
    def _provenance_onpath(
        items: Sequence[Instruction], prov: Any,
    ) -> tuple[set[int], dict[int, int]]:
        """The producer-chain trace idxs + each idx's hop-distance from the sink.

        The chain (``prov.chain``) is the backtrace steps, sorted by idx; the
        sink-writers are the LAST (highest-idx) producers. Distance = sink-writer
        idx is 0, each earlier producer one more hop back (rank in idx-descending
        order). Pure read of the provenance result — no new backtrace."""
        chain_idxs = sorted({
            int(s["idx"]) for s in (prov.chain or []) if "idx" in s})
        distance = {idx: rank for rank, idx in enumerate(reversed(chain_idxs))}
        return set(chain_idxs), distance

    # On-path band coalescing (dev-recovery-bands-decisions-composite-spec Req4).
    # A long producer chain (tc2: 7191 idxs) generated one single-idx window per idx
    # → the candidate cap dropped 6167 and every terminal was an isolated-store window
    # ("symbol does not propagate through a lone store/load"), burning the budget. The
    # fix: COALESCE consecutive producer-chain idxs (gap <= band_gap_threshold) into a
    # single BAND candidate window=[start,end] — a contiguous slice is ONE algorithm
    # segment, not N isolated points. Longer bands rank higher (a real slice carries
    # more), so the budget is spent on the slices most likely to recover F; but the
    # near-sink isolated-store diagnostic is RETAINED (demoted, not coalesced away) —
    # that "lone store" signal is still useful, it is just no longer the whole story.

    # A band ranks ABOVE a single-idx window of the same near-sink distance: a real
    # contiguous algorithm slice is worth more than one isolated store/load. The span
    # bonus is bounded (a saturating log) so it RE-ORDERS within the on-path tier and
    # never escapes it (invariant 7 — on-path stays the primary tier). The retained
    # near-sink single-store DIAGNOSTIC is demoted just below the bands so it is
    # visible (still tried if budget allows) but never out-ranks a real slice.
    _BAND_SPAN_CEILING = 0.8
    _SINGLE_STORE_DIAG_PENALTY = 0.4
    # A3 group-aware ranking: a band that belongs to a same-chain GROUP (>= the
    # composite_aggregation_min floor) gets this small cohesion bonus so the group's
    # bands cluster and out-rank a discrete single-chain band. Bounded (well under the
    # recapture/unanchored +2.0 floor) → re-orders WITHIN the on-path tier only
    # (invariant 7); a lone-chain band gets 0.0 (byte-for-byte today's ordering).
    _GROUP_COHESION_BONUS = 0.15

    def _coalesce_onpath_bands(
        self, onpath_idxs: set[int], gap_threshold: int,
    ) -> list[tuple[int, int]]:
        """Merge sorted on-path idxs into ``[start, end]`` bands.

        Two consecutive idxs join the same band when their gap is ``<= gap_threshold``
        (a gap of 1 = strictly contiguous; a larger threshold tolerates the small
        holes a producer walk leaves in a dense slice). Pure / deterministic; the
        threshold is the universal "same contiguous slice" knob, never a case idx."""
        ordered = sorted(onpath_idxs)
        if not ordered:
            return []
        bands: list[tuple[int, int]] = []
        start = prev = ordered[0]
        for idx in ordered[1:]:
            if idx - prev <= max(1, int(gap_threshold)):
                prev = idx
            else:
                bands.append((start, prev))
                start = prev = idx
        bands.append((start, prev))
        return bands

    def _onpath_window_candidates(
        self, prov: Any, onpath_idxs: set[int], distance: dict[int, int],
    ) -> list[Candidate]:
        """COALESCED on-path ``recover_window`` band candidates (Req4).

        These are the PRIMARY tier: BANDS (``window=[start,end]``) anchored on the
        output's producer chain, NOT one window per idx. Consecutive producer-chain
        idxs (gap <= ``budget.band_gap_threshold``) are merged into one band — a
        contiguous algorithm slice. Ranked so the bands NEAREST the sink and the
        LONGEST verify first (output slice first, then its inputs; a real slice over a
        lone store). The near-sink isolated-store diagnostic is RETAINED as a demoted
        single-idx candidate (the lone-store signal is still useful, just not the main
        candidate). cohort/dispatch variance is layered ON TOP later as the secondary
        filter. Each payload exposes ``on_path: True`` + ``path_distance`` (the band's
        nearest-sink distance), ``window``, ``window_span``, ``nearest_sink_distance``,
        and the chain id so a consumer separates on/off-path + sizes the slice at a
        glance."""
        import math
        cands: list[Candidate] = []
        producer_pcs = {int(p) for p in (prov.producer_pcs or ())}
        producer_pcs_repr = [f"0x{p:x}" for p in sorted(producer_pcs)]
        # The chain identity: the located output address (sink_base) anchors THIS
        # producer chain — a stable id a consumer can group bands by ("有则带").
        chain_id = (f"0x{self.sink_base:x}" if self.sink_base is not None
                    else getattr(prov, "verdict", None)
                    and prov.verdict.value)
        bands = self._coalesce_onpath_bands(
            onpath_idxs, self.budget.band_gap_threshold)
        # A3 collect-layer aggregation + group-aware ranking. Record EVERY band of THIS
        # chain into the shared run-level index so the verifier (a BAND_PARITY_FAIL
        # candidate) can plan a composite over the whole same-chain group, and so the
        # generator can rank a multi-band chain's bands TOGETHER (a chain that is one
        # group of bricks should cluster, not be敲ed one brick at a time). The bonus is
        # bounded so it only RE-ORDERS within the on-path tier (invariant 7).
        for lo, hi in bands:
            self.band_registry.record(chain_id, lo, hi)
        group_size = self.band_registry.group_size(chain_id)
        # >= composite_aggregation_min same-chain bands → this is a GROUP the planner
        # should combine; give its bands a small cohesion bonus so they sort adjacent
        # and ABOVE a discrete single-chain band (data-driven floor, no case number).
        is_group = group_size >= max(1, int(self.budget.composite_aggregation_min))
        group_bonus = self._GROUP_COHESION_BONUS if is_group else 0.0
        for lo, hi in bands:
            in_band = [distance.get(i, 0) for i in onpath_idxs if lo <= i <= hi]
            near_sink = min(in_band) if in_band else distance.get(lo, 0)
            span = hi - lo + 1
            # Nearer the sink → higher value (verify the output, then walk back); a
            # LONGER band → higher value (a real contiguous slice over a lone point).
            # span bonus is a bounded saturating log so it only re-orders WITHIN the
            # on-path tier (never escapes _BASE_ONPATH..+1; invariant 7).
            span_bonus = min(self._BAND_SPAN_CEILING,
                             self._BAND_SPAN_CEILING * math.log2(span + 1) / 8.0)
            base = (self._BASE_ONPATH + max(0.0, 1.0 - 0.05 * near_sink)
                    + span_bonus + group_bonus)
            cands.append(Candidate(
                RECOVER_WINDOW, locus=lo, signal=SIG_PROVENANCE_ONPATH,
                entry_reason=(
                    f"on output-provenance path: producer band idx[{lo},{hi}] "
                    f"(span {span}) feeds the target output (nearest {near_sink} "
                    f"hop(s) from sink; producer PCs {producer_pcs_repr[:4]}) — a "
                    f"contiguous algorithm slice anchored on WHAT writes the output, "
                    f"not where input-variance is"
                    + (f"; one of {group_size} same-chain bands (a composite group)"
                       if is_group else "")),
                base_value=base,
                payload=_compact({
                    "window": [lo, hi], "window_kind": "idx",
                    "window_span": span,
                    "source": SRC_OUTPUT_PROVENANCE,
                    "on_path": True,
                    "band": True,
                    "path_distance": near_sink,
                    "nearest_sink_distance": near_sink,
                    "chain_id": chain_id,
                    "chain_band_group_size": group_size,
                    "chain_band_group": is_group,
                    "provenance_verdict": prov.verdict.value,
                    "producer_pcs": producer_pcs_repr,
                })))
        # RETAIN the near-sink isolated-store diagnostic (Req4 ④): the single producer
        # idx NEAREST the sink, kept as its OWN single-idx candidate, DEMOTED below the
        # bands. The lone-store signal ("does an isolated store/load propagate the
        # symbol?") is still useful evidence — band coalescing must not swallow it. It
        # is only emitted when there is a band wider than 1 (otherwise the band already
        # IS that single-idx window — no separate diagnostic needed, invariant 7).
        if any((hi - lo + 1) > 1 for lo, hi in bands) and onpath_idxs:
            sink_idx = min(onpath_idxs, key=lambda i: distance.get(i, 0))
            dist0 = distance.get(sink_idx, 0)
            cands.append(Candidate(
                RECOVER_WINDOW, locus=sink_idx, signal=SIG_PROVENANCE_ONPATH,
                entry_reason=(
                    f"near-sink isolated-store DIAGNOSTIC: lone producer idx{sink_idx} "
                    f"({dist0} hop(s) from sink) — retained (demoted) so the "
                    f"single-store-propagation signal is still tried, not swallowed by "
                    f"band coalescing"),
                # demoted just below the band tier (bands floor at _BASE_ONPATH); the
                # diagnostic stays visible/tried but never the main candidate.
                base_value=max(0.1, self._BASE_ONPATH - self._SINGLE_STORE_DIAG_PENALTY),
                payload=_compact({
                    "window": [sink_idx, sink_idx], "window_kind": "idx",
                    "window_span": 1,
                    "source": SRC_OUTPUT_PROVENANCE,
                    "on_path": True,
                    "band": False,
                    "single_store_diagnostic": True,
                    "path_distance": dist0,
                    "nearest_sink_distance": dist0,
                    "chain_id": chain_id,
                    "provenance_verdict": prov.verdict.value,
                    "producer_pcs": producer_pcs_repr,
                })))
        return cands

    def _recapture_candidate(
        self, items: Sequence[Instruction], expected: bytes, prov: Any,
    ) -> Candidate:
        """A recapture-directive candidate for an UNOBSERVED output writer.

        Reuses :func:`recapture_target.derive_recapture_directive` (4531f8b) to
        produce the precise watch contract (collect the output first, then re-anchor
        provenance). This REPLACES the off-path variance fall-back — surfacing a
        watch directive instead of generating off-path variance windows was the
        whole point (the silent fall-back was the 绕路 root cause)."""
        try:
            directive = derive_recapture_directive(
                list(items), expected, self.value_name,
                reason=("target output writer not observed (provenance verdict "
                        f"{prov.verdict.value}, sink_captured={prov.sink_captured}) — "
                        "collect the output, then re-anchor provenance"))
            directive_dict = directive.to_dict()
        except Exception as e:                      # degrade explicitly, never silent
            directive_dict = {
                "kind": "recapture_directive",
                "status": "DERIVE_ERROR",
                "detail": f"derive_recapture_directive raised {type(e).__name__}: {e}",
            }
        return Candidate(
            RECOVER_WINDOW, locus=self.sink_base or 0, signal=SIG_RECAPTURE_DIRECTIVE,
            entry_reason=(
                "target output writer NOT observed — produce a recapture directive "
                "(collect the output first, then re-anchor provenance), do NOT fall "
                "back to off-path variance windows"),
            base_value=self._BASE_ONPATH + 2.0,     # top: must precede any window try
            payload=_compact({
                "source": SRC_OUTPUT_PROVENANCE,
                "on_path": None,
                "needs_observation": True,
                "provenance_verdict": prov.verdict.value,
                "sink_captured": prov.sink_captured,
                "next_watch": list(prov.next_watch or []),
                "recapture_directive": directive_dict,
            }))

    def _recapture_candidate_from_sink(
        self, items: Sequence[Instruction], expected: bytes, sv: Any,
    ) -> Candidate:
        """A recapture-directive candidate from the CHEAP sink short-circuit (task 3).

        Built BEFORE the provenance backtrace ran (the output sink is not captured at
        all), so there is no ProvenanceResult — only the SinkValidation. Reuses
        :func:`recapture_target.derive_recapture_directive` (the same precise watch
        contract). Marks the dispatch / variance fall-back as suppressed and tags WHY
        (sink unobserved) so the consumer sees a fast recapture, not a long stall."""
        try:
            directive = derive_recapture_directive(
                list(items), expected, self.value_name,
                sink_validation=sv,
                reason=("target output sink NOT captured on this run "
                        f"(validate_sink={getattr(sv, 'verdict', None)}) — recapture "
                        "the output, then re-anchor provenance; routed BEFORE the "
                        "provenance backtrace (no long generation)"))
            directive_dict = directive.to_dict()
        except Exception as e:                      # degrade explicitly, never silent
            directive_dict = {
                "kind": "recapture_directive",
                "status": "DERIVE_ERROR",
                "detail": f"derive_recapture_directive raised {type(e).__name__}: {e}",
            }
        return Candidate(
            RECOVER_WINDOW, locus=self.sink_base or 0, signal=SIG_RECAPTURE_DIRECTIVE,
            entry_reason=(
                "target output sink NOT observed — produce a recapture directive "
                "FIRST (short-circuit BEFORE the provenance backtrace / any long "
                "generation), do NOT fall back to dispatch/off-path variance windows"),
            base_value=self._BASE_ONPATH + 2.5,     # top: precedes even on-path tries
            payload=_compact({
                "source": SRC_OUTPUT_PROVENANCE,
                "on_path": None,
                "needs_observation": True,
                "short_circuit": "sink_unobserved_preflight",
                "sink_verdict": getattr(getattr(sv, "verdict", None), "value", None),
                "sink_captured": False,
                "recapture_directive": directive_dict,
            }))

    def _unanchored_candidate(self, *, detail: str, prov: Any) -> Candidate:
        """An EXPLICIT 'output cannot be anchored to provenance' diagnostic (A8④).

        The degenerate tail: a target output was supplied but its production cannot
        be tied to a traced producer chain. Report it as a candidate the agent sees
        — never a silent fall-back to off-path variance windows."""
        payload: dict[str, Any] = {
            "source": SRC_OUTPUT_PROVENANCE,
            "on_path": None,
            "unanchored": True,
            "detail": detail,
        }
        if prov is not None:
            payload["provenance_verdict"] = prov.verdict.value
            payload["sink_captured"] = prov.sink_captured
        return Candidate(
            RECOVER_WINDOW, locus=self.sink_base or 0, signal=SIG_PROVENANCE_UNANCHORED,
            entry_reason=(
                "target output supplied but NOT anchorable to a traced producer "
                f"chain — {detail}; reported explicitly (no silent fall-back to "
                "off-path variance windows)"),
            base_value=self._BASE_ONPATH + 2.0,
            payload=_compact(payload))

    def _unplaceable_blocked_candidate(
        self, prov: Any, unplaceable: list[dict],
    ) -> Candidate:
        """An EXPLICIT BLOCKED diagnostic for a NEEDS_OBSERVATION whose ONLY remaining
        gaps are unplaceable (A2 坎 / A8④).

        The producer chain reaches a read of an address with NO reading PC to hang a
        watch on — so the provenance is NOT closed (the gap is real) yet ALSO NOT
        recapturable (there is nothing to arm). Conservatively BLOCKED: surfaced as a
        candidate the agent sees with the un-hookable addresses listed, never silently
        promoted to closed and never bounced into a recapture that cannot be placed."""
        return Candidate(
            RECOVER_WINDOW, locus=self.sink_base or 0,
            signal=SIG_PROVENANCE_BLOCKED_UNPLACEABLE,
            entry_reason=(
                "provenance gap remains but is UNPLACEABLE — the chain reads an "
                "address with no reading PC to arm a watch on; neither closed nor "
                "recapturable. BLOCKED (reported explicitly, not treated as closed)"),
            base_value=self._BASE_ONPATH + 2.0,
            payload=_compact({
                "source": SRC_OUTPUT_PROVENANCE,
                "on_path": None,
                "blocked": True,
                "unplaceable": True,
                "provenance_verdict": prov.verdict.value,
                "sink_captured": prov.sink_captured,
                "unplaceable_gaps": list(unplaceable),
                "detail": ("NEEDS_OBSERVATION with only un-hookable gaps (no reading "
                           "PC) — cannot close, cannot recapture"),
            }))

    def _secondary_tier(
        self, base_value: float, on_path: bool | None,
        lo: int, hi: int, anchor: "_ProvenanceAnchor | None",
    ) -> tuple[float, dict[str, Any]]:
        """Demote a coverage/variance window to its SECONDARY tier under a provenance
        anchor; tag it on/off-path. Returns ``(base_value, payload_extra)``.

        * no anchor (``on_path`` None) → unchanged base, no extra (invariant 7).
        * on-path → keep base (variance corroborates the on-path window) + tag.
        * off-path → base floored well below the on-path tier so the frontier never
          picks a high-variance OFF-path window as the main candidate (the F0-A fix);
          still emitted (visible), just demoted."""
        if on_path is None:
            return base_value, {}
        dist = None
        if anchor is not None:
            in_band = [d for i, d in anchor.onpath_distance.items() if lo <= i <= hi]
            dist = min(in_band) if in_band else None
        extra: dict[str, Any] = {"on_path": bool(on_path)}
        if dist is not None:
            extra["path_distance"] = dist
        if on_path:
            return base_value, extra
        # off-path: sink it below every on-path window (which floor at _BASE_ONPATH).
        return max(0.1, base_value - self._OFFPATH_PENALTY), extra

    # The distinct-potential bonus is intentionally SMALL and bounded so it only
    # RE-ORDERS within a base tier (a high-potential non-anchor never out-ranks the
    # divergence anchor's 5.0 floor): it is a tie-break / priority nudge, not a new
    # verdict axis (invariant 7 — ordering only). Each component is capped, then the
    # sum is capped at this ceiling.
    _POTENTIAL_CEILING = 0.9
    _DIVERGENCE_BONUS = 0.3       # window covers divergence_idx (seed's entry point)

    @classmethod
    def _distinct_potential(
        cls,
        positions: "Sequence[Any]",
        *,
        n_vectors: int,
        divergence_idx: int | None,
    ) -> float:
        """Pre-symex PROXY for the window's independent-side distinct potential.

        A bounded, additive bonus added to a candidate's fixed ``base_value`` so the
        CVD frontier verifies the windows MOST LIKELY to EXACT-close first (true
        distinct still needs a symex run; this only orders the budget). Built purely
        from the cohort_diff structural signals (invariant 8 — never fabricated; zero
        case-specific knowledge, invariant 2/6):

          * **VaryingPosition diversity** — the dimensions (``varying_regs`` +
            ``varying_mem`` + a control-flow divergence) that vary across positions
            in the window, normalised by the position count. More varying dimensions
            per position → more ways the input can drive a distinct output.
          * **divergence coverage** — a window that covers ``divergence_idx`` (the
            seed's first observable entry) is where the input is freshest → higher.
          * **n_vectors** — the cohort size is the HARD upper bound on independent
            distinct outputs; a thin cohort caps the achievable distinct, so it
            scales the diversity term.

        保序兜底 (invariant 7): with NO varying signal — no positions, or every
        position carries zero varying dims — the bonus is exactly ``0.0``, leaving
        the caller's fixed base_value byte-for-byte unchanged. A thin cohort
        (``n_vectors <= 1``) likewise scales the diversity term to ~0 (its real fate
        is the pre-flight早 BLOCK, not the ordering)."""
        pos = list(positions)
        if not pos:
            return 0.0
        # total varying dimensions across the window's positions
        dims = 0
        for p in pos:
            dims += len(getattr(p, "varying_regs", ()) or ())
            dims += len(getattr(p, "varying_mem", ()) or ())
            if getattr(p, "control_flow", False):
                dims += 1
        if dims == 0:
            # No varying dimensions → no distinct potential to add (保序兜底): even a
            # window covering divergence carries no input-distinct output to order on
            # without a varying dimension. Byte-for-byte today's fixed base_value.
            return 0.0
        dims_per_pos = dims / len(pos)
        # Normalise diversity into [0,1): saturating — 1 dim ≈ 0.5, many dims → ~1.
        diversity = dims_per_pos / (dims_per_pos + 1.0)
        # Cohort size is the distinct upper bound: 1 vector → 0 (can't be distinct),
        # saturating toward 1 as the cohort widens. n_vectors<=1 ⇒ scale 0 ⇒ the
        # diversity term contributes nothing (its fate is the pre-flight BLOCK).
        nv = max(0, int(n_vectors) - 1)
        cohort_scale = nv / (nv + 1.0)          # 0 at nv=0, →1 as cohort widens
        bonus = diversity * cohort_scale
        if (divergence_idx is not None
                and any(getattr(p, "idx", None) == divergence_idx for p in pos)):
            bonus += cls._DIVERGENCE_BONUS
        return min(bonus, cls._POTENTIAL_CEILING)

    @staticmethod
    def _opaque_advisory_window(advisory: Mapping[str, Any]) -> tuple[int, int] | None:
        """A representative idx window for an opaque cohort, derived purely from the
        advisory's structural coordinates (zero case-specific knowledge).

        Prefer the band spanning the EA-varying staging sites' idxs (where the
        staging actually is); fall back to the cohort's ``aligned_region`` band.
        Returns ``None`` only when neither is usable (no window → nothing to try →
        the terminal classifier claims as before)."""
        idxs: list[int] = []
        sites = advisory.get("ea_varying_sites")
        # _compact may have trimmed the list to a {_trimmed_list, sample, …} dict;
        # the sample is enough for a representative window.
        if isinstance(sites, Mapping):
            sites = sites.get("sample") or ()
        for s in sites or ():
            if isinstance(s, Mapping):
                i = s.get("idx")
                if isinstance(i, int) and i >= 0:
                    idxs.append(i)
        if idxs:
            return (min(idxs), max(idxs))
        region = advisory.get("aligned_region")
        if (isinstance(region, (list, tuple)) and len(region) == 2
                and all(isinstance(x, int) for x in region)):
            lo, hi = int(region[0]), int(region[1])
            return (min(lo, hi), max(lo, hi))
        return None


# --------------------------------------------------------------------------- #
# 2 — Verifier (heavy / T2): run the whole drive() per window, map the verdict.
# --------------------------------------------------------------------------- #

