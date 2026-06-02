"""cvd_recovery.driver section (split from the monolithic module)."""
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
from ._cohort import SIG_RECAPTURE_DIRECTIVE, _compact, load_cohort_traces
from ._registry import _DEFAULT_RECOVERY_DECISIONS, recovery_registry


def _reconcile_anchors(cc: CaseConfig, items: list) -> CaseConfig:
    """Rebase ``cc``'s absolute pc anchors onto THIS run's actual module base.

    Reuses ``conformance.detect_relocation``'s two-anchor cross-validation (entry
    AND exit must share ONE page-aligned delta — the anti-masking guard): the
    config's ``entry_pc`` / ``sink_hint_addr`` are the pre-relocation anchors, the
    trace's first / last pc are the observed ones. A clean rebase shifts the pc
    anchors (and a pc-window); no clean rebase (already aligned, or anchors
    disagree) leaves the config untouched — never reconcile a genuinely wrong
    anchor away. Closes the 'sink pinned at the pre-relocation address' error."""
    if not items:
        return cc
    from ..conformance import detect_relocation, rebase_pc
    from ..types import TargetMeta
    meta = TargetMeta(
        target_name=cc.target, arch="", algo_entry_pc=cc.entry_pc,
        algo_exit_pc=cc.sink_hint_addr, input_length=None, output_length=0)
    reloc = detect_relocation(meta, items[0].pc, items[-1].pc)
    if reloc is None:
        return cc
    new = replace(
        cc,
        entry_pc=rebase_pc(cc.entry_pc, reloc),
        seed_hint_addr=rebase_pc(cc.seed_hint_addr, reloc),
        sink_hint_addr=rebase_pc(cc.sink_hint_addr, reloc))
    if cc.window_kind == "pc":
        lo, hi = cc.window
        new = replace(new, window=(rebase_pc(lo, reloc), rebase_pc(hi, reloc)))
    return new


# --------------------------------------------------------------------------- #
# B2 — verifier-internal recapture closure helpers (dev-recovery-verifier-internal-
# recapture-spec). The DECISION to recapture lives in the generator/verify path (it
# emits the recapture-directive candidate); the ORCHESTRATION (drive the runner,
# re-enter collect with same-run snapshots) lives here, in the one-call entry (DP1).
# --------------------------------------------------------------------------- #

def _wants_recapture(res: "CvdResult") -> bool:
    """True iff a collect run surfaced a RECAPTURE DIRECTIVE — the output writer was
    not observed (NEEDS_OBSERVATION with a placeable gap, or an unobserved sink) and
    the run is asking for the output to be collected before provenance can close.

    Detected structurally from the collected entries: the recapture directive
    surfaces as a ``candidate.signal == recapture_directive`` entry — depending on the
    verify path it lands in ``extension_requests`` (terminal-shaped) OR
    ``pending_judgments`` (drive-pause-shaped), so BOTH are scanned (and ``confirmed``
    defensively). A pure read of the result — no re-derivation."""
    for bucket in (res.extension_requests, res.pending_judgments, res.confirmed):
        for entry in (bucket or []):
            cand = entry.get("candidate") if isinstance(entry, dict) else None
            if isinstance(cand, dict) and cand.get("signal") == SIG_RECAPTURE_DIRECTIVE:
                return True
    return False


def _recapture_reenter_loop(
    *,
    res: "CvdResult",
    path: "Path",
    collect_once: "Callable[[list], tuple[CvdResult, Path]]",
    items: list,
    sink_base: int | None,
    adapter: Any,
    loop_input: bytes,
    output_observe_pc: int | None,
    budget: "CvdBudget",
) -> "tuple[CvdResult, Path]":
    """Close the recapture loop in-process: drive run_recapture_loop, RE-ENTER collect
    with the loop's same-run snapshots, repeat until closed (→ on-path bands, DP2) or a
    non-CLOSED loop terminal or the re-entry budget is spent.

    G1 BY CONSTRUCTION: each re-entry feeds collect EXACTLY the snapshot set of the
    loop's FINAL round — one rerun, one nonce (run_recapture_loop never accumulates
    snapshots across reruns). We never merge a prior re-entry's snapshots here, so no
    cross-rerun stitch is ever possible from this layer either.

    No usable ``sink_base`` → cannot anchor a recapture loop → degrade explicitly
    (WARN-loud, return today's directive gap map)."""
    if sink_base is None:
        _log.warning(
            "run_recovery: recapture directive surfaced but base_config has no "
            "sink_hint_addr to anchor the recapture loop on — cannot self-close. "
            "Returning the recapture-directive gap map (NOT a closure).")
        return res, path

    max_reentries = int(getattr(budget, "max_recapture_reentries", 4))
    for attempt in range(1, max_reentries + 1):
        # 1. ONE recapture loop (the reused closed-loop engine — G1/G2/G3 inside it).
        loop_res = run_recapture_loop(
            items, sink_base=int(sink_base), adapter=adapter,
            loop_input=loop_input, output_observe_pc=output_observe_pc)

        # 2. The loop hit a non-CLOSED terminal (STALLED / UNPLACEABLE / BUDGET) — an
        #    EXPLICIT structured exit (G3). Do NOT spin: surface it on the result and
        #    return the current (directive) gap map; never a silent close (A8④).
        if loop_res.outcome is not LoopOutcome.CLOSED:
            term = loop_res.terminal() or {}
            _log.warning(
                "run_recovery: recapture loop did not close (outcome=%s after %d "
                "round(s)) on re-entry %d/%d — surfacing the structured loop terminal; "
                "the run is NOT closed. See result.provenance['recapture_loop'].",
                loop_res.outcome.value, len(loop_res.rounds), attempt, max_reentries)
            _stamp_recapture_loop(res, loop_res, attempt, truncated=False)
            return res, path

        # 3. CLOSED — feed the FINAL round's same-run snapshots back into collect (DP2:
        #    the re-entered collect, now seeing closed provenance, continues straight
        #    into on-path band generation — closure → bands in the same call).
        round_snaps = list(loop_res.snapshots)
        # Any truncated round in the loop means the runner hit a record cap — WARN and
        # propagate truncated; an incomplete ledger is not silently treated as complete.
        loop_truncated = any(r.truncated for r in loop_res.rounds)
        if loop_truncated:
            _log.warning(
                "run_recovery: the recapture loop hit a runner RECORD CAP (truncated "
                "round) on re-entry %d — the captured snapshot ledger is INCOMPLETE; "
                "the re-entered provenance may not fully close. Truncation is "
                "propagated (never silently treated as complete).", attempt)
        res, path = collect_once(round_snaps)
        _stamp_recapture_loop(res, loop_res, attempt, truncated=loop_truncated)

        # 4. The re-entered run no longer asks for recapture → closed (the generator now
        #    sees an observed output and continues to on-path bands). Done.
        if not _wants_recapture(res):
            _log.info(
                "run_recovery: recapture loop closed the output observation in %d "
                "re-entry/-ies; the recovery run continued to on-path band generation "
                "in the same call (B2 self-close).", attempt)
            return res, path
        # else: still NEEDS_OBSERVATION (a deeper producer gap surfaced) → loop again
        #       under the re-entry budget.

    # Budget spent while still asking for recapture — WARN-loud + truncated, never a
    # silent spin / silent close (A8④). Return the last (still-directive) gap map.
    _log.warning(
        "run_recovery: recapture self-close hit the re-entry cap "
        "(budget.max_recapture_reentries=%d) while the run still wanted observation — "
        "stopping (NOT closed). Raise the cap or collect the residual by hand; the "
        "last gap map carries the recapture directive.", max_reentries)
    return res, path


def _stamp_recapture_loop(
    res: "CvdResult", loop_res: Any, attempt: int, *, truncated: bool,
) -> None:
    """Record the in-process recapture-loop terminal onto the collect result's
    provenance channel so the self-close is VISIBLE in the gap map (which loop
    outcome, how many rounds, truncation) — never a hidden side path (invariant 1)."""
    prov = dict(res.provenance or {})
    prov["recapture_loop"] = _compact({
        "kind": "verifier_internal_recapture",
        "reentry": attempt,
        "outcome": loop_res.outcome.value,
        "n_rounds": len(loop_res.rounds),
        "closed": loop_res.closed,
        "truncated": truncated,
        "terminal": loop_res.terminal(),
    })
    res.provenance = prov


# --------------------------------------------------------------------------- #
# P6 — output-determinism evidence (dev-output-determinism-evidence-spec).
#
# A FIRST-CLASS, run-level EVIDENCE channel: "this runner + this input, rerun K
# times, produced an OUTPUT that was observed identical (or not)". It saves the
# agent a hand-rolled "this is not a nonce" assertion by RECORDING the bounded
# empirical observation utov can make for free with the rerun-capable adapter.
#
# Honesty / invariant 3 + 4 (the spec's hard red lines):
#   * It is EVIDENCE ONLY — it NEVER feeds any close / parity / G4 gate, never
#     auto-promotes a closure (the PSEUDO_CLOSURE_TRAP). verdict judgment does not
#     read it; an agent/human reads it to decide "treat this as deterministic output?".
#   * Its wording is strictly ``observed-stable-across-K`` — a BOUNDED empirical
#     observation, NOT a proof. The forbidden over-strong tokens (deterministic /
#     no nonce / proven) NEVER appear in the field or its reasons; a coarse time-seed
#     could still vary over a longer period than K reruns.
#   * No adapter / probe error / EMPTY output → ``observed: false`` + an explicit
#     reason — it NEVER defaults to "stable" (that silent default is the false-closure
#     risk). Observed instability → ``stable: false`` + byte-level varying/constant
#     ranges (reported GENERICALLY, never curve-fit to one case's byte layout).
#   * Hitting the rerun cap (a truncated rerun) → ``truncated: true`` propagated +
#     WARN (same discipline as B2), never silently treated as a clean stable result.
#
# ORTHOGONAL to the other two determinism notions (do NOT conflate — spec §复用):
#   * ``setup_symex.check_seed_independence`` = SEED side (does the symbolic SEED
#     reach state); ``cohort_diff`` = cross-INPUT variance (different inputs). P6 is
#     the SAME-input, cross-RERUN dimension. Different axis, different question.
# G1 is untouched: this probe runs its OWN reruns and compares only ``rr.output``;
# it accumulates NO snapshot across reruns and feeds nothing back into provenance.
# --------------------------------------------------------------------------- #

# Forbidden over-strong determinism claims (invariant 3). Asserted absent in the
# evidence + reasons by the acceptance tests; only ``observed-stable-across-K`` is
# the sanctioned phrasing.
_OUTPUT_DET_FORBIDDEN_WORDS = ("deterministic", "no nonce", "proven")

# Public reasons / kinds (stable strings the agent can route on).
OUTPUT_DET_NO_ADAPTER = "no rerun-capable adapter"


def _byte_variance_ranges(
    outputs: Sequence[bytes],
) -> tuple[list[list[int]], list[list[int]]]:
    """Per-byte constant/varying ranges across K observed outputs (GENERIC, A8).

    Compares the K outputs byte-position by byte-position and coalesces consecutive
    positions of the SAME kind into ``[start, end]`` (inclusive) ranges. A position
    is *varying* iff the K outputs disagree there OR not all outputs even reach that
    position (a length difference is itself a variance). Reported universally — the
    spec's red line forbids hardcoding tc2's "first 6 fixed / last 26 vary" shape;
    this derives the ranges from whatever the outputs actually show. Returns
    ``(varying_ranges, constant_ranges)``."""
    if not outputs:
        return [], []
    max_len = max(len(o) for o in outputs)
    min_len = min(len(o) for o in outputs)
    varying: list[list[int]] = []
    constant: list[list[int]] = []

    def _flush(target: list[list[int]], start: int, end: int) -> None:
        target.append([start, end])

    run_kind: bool | None = None      # True = varying, False = constant
    run_start = 0
    for i in range(max_len):
        if i >= min_len:
            is_var = True             # not every output reaches here → variance
        else:
            col = {o[i] for o in outputs}
            is_var = len(col) > 1
        if run_kind is None:
            run_kind, run_start = is_var, i
        elif is_var != run_kind:
            _flush(varying if run_kind else constant, run_start, i - 1)
            run_kind, run_start = is_var, i
    if run_kind is not None:
        _flush(varying if run_kind else constant, run_start, max_len - 1)
    return varying, constant


def probe_output_determinism(
    adapter: Any,
    loop_input: bytes,
    *,
    reruns: int = 3,
) -> dict[str, Any]:
    """Observe output stability across K SAME-input reruns — a P6 evidence record.

    Drives ``adapter.rerun(loop_input, [])`` (EMPTY observe_points — we only want the
    produced ``output``, no snapshots, so G1 is never even approached) K times and
    compares the K outputs. Pure observation; reads only ``rr.output`` (+ ``rr
    .truncated``). Returns the ``output_determinism`` evidence dict:

      * all K identical →
        ``{observed, stable: true, reruns: K, sample_hex, truncated}``
      * any disagreement →
        ``{observed, stable: false, reruns: K, varying_byte_ranges,
           constant_byte_ranges, truncated}``
      * adapter has no ``rerun`` / a rerun raised / an EMPTY output →
        ``{observed: false, reason: ...}`` (NEVER defaults to stable).

    Wording is ``observed-stable-across-K`` only (invariant 3); the forbidden
    over-strong tokens never appear. ``truncated`` rides through when any rerun hit a
    runner record cap (the caller WARNs)."""
    k = max(1, int(reruns))
    rerun = getattr(adapter, "rerun", None)
    if not callable(rerun):
        return {"observed": False, "reason": OUTPUT_DET_NO_ADAPTER}
    outputs: list[bytes] = []
    truncated = False
    for r in range(k):
        try:
            rr = rerun(loop_input, [])
        except Exception as e:        # a rerun error → honest non-observation, never stable
            return {
                "observed": False,
                "reruns_attempted": r,
                "reason": (f"rerun-capable adapter raised on rerun {r + 1}/{k}: "
                           f"{type(e).__name__}: {e} — output stability NOT observed"),
            }
        out = bytes(getattr(rr, "output", b"") or b"")
        if not out:
            return {
                "observed": False,
                "reruns_attempted": r + 1,
                "reason": (f"adapter.rerun returned an EMPTY output on rerun "
                           f"{r + 1}/{k} — output stability cannot be observed; "
                           "NOT treated as stable"),
            }
        if getattr(rr, "truncated", False):
            truncated = True
        outputs.append(out)
    stable = len({bytes(o) for o in outputs}) == 1
    ev: dict[str, Any] = {
        "observed": True,
        "stable": stable,
        "reruns": k,
        "note": ("output observed identical across all K reruns of this runner+input "
                 "— an observed-stable-across-K empirical observation (a bounded "
                 "K-rerun observation, NOT a determinism proof; a coarse time/sequence "
                 "seed could still vary over a longer period than K)"
                 if stable else
                 "output observed to VARY across K reruns of this runner+input — the "
                 "varying byte ranges are an observed-unstable-across-K signal (the "
                 "G1 nonce/time signal; do NOT treat the run as a fixed output)"),
        "truncated": truncated,
    }
    if stable:
        ev["sample_hex"] = outputs[0].hex()
    else:
        varying, constant = _byte_variance_ranges(outputs)
        ev["varying_byte_ranges"] = varying
        ev["constant_byte_ranges"] = constant
        ev["samples_hex"] = [o.hex() for o in outputs[:4]]
    return _compact(ev)


def _stamp_output_determinism(res: "CvdResult", det: Mapping[str, Any]) -> None:
    """Record the P6 output-determinism observation onto the result's run-level
    ``provenance`` channel (additive, parallel to ``recapture_loop``). EVIDENCE
    ONLY — it touches no verdict/close/parity field (invariant 3)."""
    prov = dict(res.provenance or {})
    prov["output_determinism"] = dict(det)
    res.provenance = prov


def run_recovery(
    items,
    *,
    base_config: CaseConfig,
    triton_runner: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    expected: bytes,
    work_root: str | Path,
    ts: str,
    coverage: CoverageMap | None = None,
    dependence: InputDependenceMap | None = None,
    decisions: Mapping[str, Any] | None = None,
    cohort_traces: "Sequence[Sequence[Instruction]]" = (),
    cohort_trace_paths: "Sequence[str | Path] | None" = None,
    input_keys: "Sequence[str] | None" = None,
    disp_thresholds: "Mapping[str, float] | None" = None,
    cohort_mem_sidecars: "Sequence[Any] | None" = None,
    pointer_chain: "PointerChainSpec | None" = None,
    budget: "CvdBudget | None" = None,
    exec_identity: Mapping[str, Any] | None = None,
    snapshots: "Sequence[Any] | None" = None,
    recapture_adapter: Any = None,
    loop_input: bytes | None = None,
    output_observe_pc: int | None = None,
    mem_sink: "Mapping[str, Any] | None" = None,
    filename: str = "cvd_gap_map.json",
) -> tuple[CvdResult, Path]:
    """One-call consumer entry for a recovery CVD run — the whole wiring, once.

    Encodes the four things a hand-rolled invoker gets wrong (the reference case evidence):
      1. **pure recovery registry** — ``recovery_registry`` only, NOT mixed with
         ``default_registry`` (whose sink chain emits ``OUTPUT_NOT_OBSERVABLE``
         noise that drowns the real frontier);
      2. **safe decision defaults, real judgment still escalated** — defaults the
         non-judgment checkpoints so ``drive`` reaches symex, but leaves
         ``mem_input_symbolize_vs_back`` to PENDING (the agent decides; the engine
         must not). With ``cohort_traces`` + ``input_keys`` the evidence-backed
         disposition AUTO-prefills only the reliable direction (value varies →
         symbolize); back is recommend-only, ambiguous stays PENDING. Caller
         ``decisions`` override both the defaults and the prefill;
      3. **anchor reconcile** — rebases pc anchors onto this run's actual base via
         ``_reconcile_anchors`` (no pinning a pre-relocation address);
      4. **stamped JSON, no sqlite** — delegates to ``cvd.run_cvd_collect_to_json``
         (actively persists ``cvd_gap_map.json``, passes no ledger).

    **Cohort symmetry by construction (68a873e anti-pattern correction).** Prefer
    ``cohort_trace_paths`` (a list of cohort trace file paths) over pre-loaded
    ``cohort_traces``: each path is loaded through ``load_cohort_traces`` →
    ``JsonlTraceReader(p).merged()``, the SAME entry the main trace uses, with
    automatic ``<stem>_mem.jsonl`` sibling resolution. The caller gives ONLY paths
    and structurally cannot feed one merged + three bare — symmetry is guaranteed,
    not a caller obligation. ``cohort_mem_sidecars`` survives ONLY as an optional
    explicit OVERRIDE (parallel to the paths); omitting it is already symmetric. A
    vector with neither an override nor an on-disk sibling raises NO exception but
    is WARN-ed into the gap-map evidence (no silent batch degradation). If both
    ``cohort_trace_paths`` and a pre-loaded ``cohort_traces`` are given, the paths
    win (the symmetric entry).

    **Pointer-chain shape by self-derivation (坎3).** ``pointer_chain`` is an
    optional OVERRIDE only; by default drive self-derives the staging pointer-chain
    shape from its own opaque diagnosis, so the caller never hand-types a
    case-specific shape and the store-side forward narrow is not perpetually empty.

    **Same-execution oracle snapshots (A1).** ``snapshots`` are memory snapshots
    captured in the SAME execution that produced ``items`` (G1: never accumulated
    across reruns — the watch-batching reconnaissance hands back exactly the run's own
    captures). They are forwarded to ``run_cvd`` so ``validate_sink`` /
    ``trace_provenance`` can read them (turning an OUTPUT_NOT_OBSERVABLE sink into a
    SINK_CONFIRMED located_via=snapshot). Because the verify-path reads through
    ``state.scoped_snapshots()`` (gated on ``obs_scope >= 1``, cvd.py:144-145), a
    NON-EMPTY ``snapshots`` opens ``obs_scope`` at PLACE (existing obs_scope field, no
    new gate) so the captured evidence is actually used — NOT silently swallowed by
    the scope gate (A8④). Empty / ``None`` is a strict regression (obs_scope stays 0,
    the widen ladder behaves exactly as before).

    **Verifier-internal recapture closure (B2,
    dev-recovery-verifier-internal-recapture-spec).** When the collect run surfaces a
    recapture-directive (the output writer was not observed → NEEDS_OBSERVATION with a
    placeable ``next_watch``, the A2 'collect-the-output-first' branch) AND a
    rerun-capable ``recapture_adapter`` (``adapter.rerun(input, observe_points) ->
    RerunResult``) plus ``loop_input`` are supplied, run_recovery CLOSES THE LOOP BY
    CONSTRUCTION (DP1 — verify() stays a pure judgment that only EMITS the directive;
    the one-call entry owns the runner + the re-entry orchestration):

      1. drive :func:`engine.recapture_loop.run_recapture_loop` (the SAME closed-loop
         engine the standalone path uses — reused, NOT re-written, G1/G2/G3 already
         enforced there) with this run's ``sink_hint_addr`` and ``loop_input``;
      2. take the loop's FINAL-round snapshots — by construction all from ONE rerun
         (one nonce; the loop never accumulates across reruns — G1) — and RE-ENTER
         collect with ``snapshots=`` those (DP2 — the re-entered collect, now seeing a
         closed provenance, continues straight into on-path band generation in the
         same call; closure → bands 一气呵成, no second caller step);
      3. repeat until the run no longer asks for recapture (closed → bands) or the
         loop hits a non-CLOSED terminal (STALLED / UNPLACEABLE / BUDGET) or the
         ``budget.max_recapture_reentries`` cap is hit (WARN-loud + truncated, never a
         silent spin / never a silent close).

    The caller therefore NO LONGER hand-wires ``run_recapture_loop -> run_recovery(
    snapshots=)``; that two-step is internalised so a narrow agent cannot forget it or
    pass a cross-rerun snapshot mix (the construct-symmetry red line).

    **No runner → explicit degrade, never silent (A8④ / 契约③).** When a recapture
    directive is surfaced but NO ``recapture_adapter`` (or no ``loop_input``) is
    supplied, run_recovery returns TODAY's behaviour — the recapture-directive gap map
    (the agent collects the output by hand) — and WARNs LOUD that the verifier had no
    runner to self-close. It NEVER treats the directive as a closure and NEVER drops
    the gap.

    **Mem-write recovery sink (Issue 7, spec_f0_mem_write_window_sink.md).**
    ``mem_sink`` is an OPTIONAL recovery sink descriptor — ``{"sink_form": "mem",
    "sink_idx", "sink_addr", "sink_size", ...}`` — for a window whose OUTPUT is a
    memory store, not a register. When supplied, the verifier derives the store
    interval (from the trace mem op at ``sink_idx`` / S3 byte-granular deps when
    addr+size are not filled — the caller is NOT required to), drives the runner in
    mem-sink mode (it reads the store's symbolic bytes), and compares observed/
    predicted BYTEWISE across vectors. A store whose EA cannot be pinned / whose
    bytes cannot be read → the structured ``MEM_SINK_UNPLACEABLE`` terminal (never a
    silent register/constant fallback); an input-invariant store → the existing
    seed-independence exclusion. ``None`` (default) → the register path, byte-for-
    byte today's x8 behaviour (the regression guard).

    Returns ``(CvdResult, path_to_gap_map_json)``."""
    eff_decisions = dict(_DEFAULT_RECOVERY_DECISIONS)
    eff_decisions.update(decisions or {})
    # A1: same-execution snapshots. A non-empty list MUST land in obs_scope >= 1 or the
    # verify-path scope gate (scoped_snapshots) silently discards it — open scope here
    # (existing field) and WARN-loud if anything would still drop it (never silent).
    snaps = list(snapshots or [])
    eff_obs_scope = 0
    if snaps:
        eff_obs_scope = 1
        _log.info(
            "run_recovery: %d same-execution snapshot(s) supplied — opening "
            "obs_scope=1 at PLACE so the verify path (validate_sink / "
            "trace_provenance via scoped_snapshots) actually reads them; without "
            "this they would be silently dropped by the obs_scope gate (cvd.py:144).",
            len(snaps))
    cc = _reconcile_anchors(base_config, list(items))
    # Symmetric cohort load (preferred): paths in → each loaded via the SAME
    # JsonlTraceReader(p).merged() the main trace uses (auto _mem.jsonl sibling), so
    # the caller cannot bare-feed the cohort. The load report (no-sidecar WARN per
    # vector) rides into the verifier so it surfaces in the gap-map evidence.
    cohort_load_diagnostics: dict[str, Any] | None = None
    if cohort_trace_paths is not None:
        cohort_traces, cohort_load_diagnostics = load_cohort_traces(
            cohort_trace_paths, cohort_mem_sidecars=cohort_mem_sidecars)
        # paths already folded the sibling/override → don't re-merge in the verifier.
        cohort_mem_sidecars = None
    # Generation/backtrace budget (dev-recovery-generation-budget-spec): the SAME
    # CvdBudget governs the generator's backtrace/candidate ceilings AND the verify
    # loop. A default CvdBudget() applies generous, parameterised ceilings (no F0
    # number); a budget-internal case never trips them (invariant 7).
    eff_budget = budget or CvdBudget()

    def _collect_once(round_snaps: list) -> tuple[CvdResult, Path]:
        """ONE recovery collect run on a given (same-execution) snapshot set.

        Factored out so the B2 recapture re-entry can re-run collect with the
        loop's freshly-captured snapshots. A non-empty ``round_snaps`` opens
        obs_scope=1 (A1) so the verify path actually reads them; empty stays the
        obs_scope=0 regression. The registry is rebuilt per call because the
        generator/verifier hold one-shot per-run state (e.g. the recapture-directive
        guard) — a fresh registry per collect is the correct, side-effect-free re-run."""
        ob = 1 if round_snaps else 0
        reg = recovery_registry(
            base_config=cc, triton_runner=triton_runner,
            coverage=coverage, dependence=dependence, decisions=eff_decisions,
            cohort_traces=cohort_traces, input_keys=input_keys,
            disp_thresholds=disp_thresholds, cohort_mem_sidecars=cohort_mem_sidecars,
            cohort_load_diagnostics=cohort_load_diagnostics,
            pointer_chain=pointer_chain, budget=eff_budget,
            # Issue 7 — EXPLICIT mem-write recovery sink descriptor (the window's
            # output is a store, not a register). None → register path (regression
            # guard); the verifier derives addr/size from the trace mem op / S3 dep.
            mem_sink=mem_sink)
        return run_cvd_collect_to_json(
            items, expected, work_root=work_root, ts=ts,
            exec_identity=exec_identity or {}, filename=filename, registry=reg,
            budget=eff_budget,
            # A1: forward the SAME-execution snapshots + the scope that makes them visible.
            snapshots=list(round_snaps), obs_scope=ob,
            # Run-level out-layer bubble: the cohort-load report rides to the TOP of the
            # gap map so the no-sidecar WARN is visible on EVERY run outcome — including
            # an opaque TERMINAL / all-CONFIRMED run that emits no PENDING evidence (the
            # per-window mem_disposition_diagnostics channel only fires on PENDING). This
            # is additive to that channel, not a replacement. None → no load layer ran.
            cohort_load=cohort_load_diagnostics)

    res, path = _collect_once(snaps)

    # B2 — verifier-internal recapture closure. If the run is asking for recapture
    # (the output writer was not observed), close the loop BY CONSTRUCTION when a
    # rerun-capable adapter is in hand; otherwise degrade EXPLICITLY (never silent).
    if _wants_recapture(res):
        if recapture_adapter is not None and loop_input is not None:
            res, path = _recapture_reenter_loop(
                res=res, path=path, collect_once=_collect_once,
                items=list(items), sink_base=cc.sink_hint_addr,
                adapter=recapture_adapter, loop_input=loop_input,
                output_observe_pc=output_observe_pc, budget=eff_budget)
        else:
            # 契约③: a recapture directive but NO runner to self-close → keep today's
            # gap map (the agent collects by hand) and WARN LOUD. NEVER a silent close,
            # NEVER a dropped gap (A8④). This is the symmetric degrade the spec mandates.
            _log.warning(
                "run_recovery: the recovery run surfaced a RECAPTURE DIRECTIVE "
                "(output writer not observed → NEEDS_OBSERVATION) but no usable runner "
                "was supplied (recapture_adapter=%s, loop_input=%s) — the verifier "
                "CANNOT self-close the observation loop. Returning the recapture-"
                "directive gap map for the agent to collect the output by hand; this is "
                "NOT a closure. Pass recapture_adapter (a rerun-capable RunnerAdapter) "
                "+ loop_input to let run_recovery close the loop in-process (B2).",
                "set" if recapture_adapter is not None else None,
                "set" if loop_input is not None else None)

    # P6 — output-determinism evidence (dev-output-determinism-evidence-spec).
    # When a rerun-capable adapter + loop_input are in hand, OBSERVE output stability
    # across K same-input reruns and stamp it RUN-LEVEL (additive, parallel to the B2
    # recapture_loop stamp). EVIDENCE ONLY: it feeds no close/parity/G4 gate, never
    # auto-promotes a closure (invariant 3). No adapter / no input → an EXPLICIT
    # observed:false record (契约②), never a defaulted "stable". A truncated rerun
    # (runner record cap) → truncated propagated + WARN, never silent.
    if recapture_adapter is not None and loop_input is not None:
        det = probe_output_determinism(
            recapture_adapter, loop_input,
            reruns=getattr(eff_budget, "max_output_determinism_reruns", 3))
        if det.get("observed") and det.get("truncated"):
            _log.warning(
                "run_recovery: the output-determinism probe hit a runner RECORD CAP "
                "(a truncated rerun) — the observed output may be INCOMPLETE; "
                "output_determinism.truncated propagated (never silently treated as a "
                "clean observed-stable-across-K result).")
        elif not det.get("observed"):
            _log.info(
                "run_recovery: output-determinism NOT observed (%s) — recording "
                "output_determinism.observed=false (never defaulting to stable).",
                det.get("reason"))
    else:
        det = {"observed": False, "reason": OUTPUT_DET_NO_ADAPTER}
        _log.info(
            "run_recovery: no rerun-capable adapter (+ loop_input) supplied — "
            "output_determinism.observed=false (explicit, never assumed stable).")
    _stamp_output_determinism(res, det)
    # Re-export so the on-disk gap map carries the run-level output_determinism stamp
    # (the export inside run_cvd_collect_to_json ran BEFORE this stamp). The utov-export
    # header + filename-honesty gate ride through export_gap_map (契约④, same path).
    from ..cvd import export_gap_map
    export_gap_map(res, path, ts=ts, exec_identity=exec_identity or {})
    return res, path


