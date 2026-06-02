"""cvd_recovery.registry section (split from the monolithic module)."""
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
from ._cohort import OPAQUE_STAGING_FRONTIER, _OnpathBandRegistry, _compact
from ._generator import RecoveryWindowGenerator
from ._verifier import RecoverWindowVerifier


class RecoveryTerminalClassifier(TerminalClassifier):
    """Claim the recovery run's global terminal frontier.

    When the cohort diff reports ``opaque`` (the seeds vary yet no observable
    state moves — pointer-indirect staging), the whole locus is the opaque-staging
    frontier: a genuine, named dead end (not a silent stall), pointing at
    dev-symbolic-input-through-opaque-staging.md. Without a dependence map it
    declines (returns None) so the driver's normal terminal path / extension
    request still applies."""

    name = "recovery_terminal"
    version = "1"
    owner = "core"

    def __init__(self, *, dependence: InputDependenceMap | None = None) -> None:
        self.dependence = dependence

    def classify(self, state: CvdState) -> Terminal | None:
        # 坎2 timing — no separate "forward already tried" guard is needed here:
        # the driver only reaches a TerminalClassifier when the frontier is empty
        # AND widening is exhausted, and RecoveryWindowGenerator now emits an
        # opaque-staging-forward candidate on the FIRST generate for any opaque
        # cohort with an advisory window. So a forward is always tried (and leaves
        # its symbolic_forwards) before the frontier can empty; the global opaque
        # terminal here is claimed only AFTER that — honestly, with the forward
        # having run. When no advisory window exists (generator emits nothing),
        # there is genuinely nothing to forward, and claiming directly is correct.
        dep = self.dependence
        if dep is not None and dep.is_opaque:
            evidence = _compact(dep.to_dict())   # invariant 4: no big lists inline
            # Phase 3: surface the localize-side opaque advisory (the EA-varying
            # staging PCs) into the terminal evidence so the opaque dead end ships
            # with WHERE-to-pierce coordinates, not just "needs symex". Already a
            # to_dict() inside dep.to_dict(); also lift it to a top-level key so a
            # consumer need not dig (kept _compact for invariant 4).
            adv = dep.opaque_staging_advisory
            if adv is not None:
                evidence["opaque_staging_advisory"] = _compact(adv)
            return Terminal(
                kind="opaque_staging",
                evidence=evidence,
                sink_base=None,
                capability_request=OPAQUE_STAGING_FRONTIER,
                success=False)
        return None


# --------------------------------------------------------------------------- #
# Registry helper — wire the three roles into a CVD Registry.
# --------------------------------------------------------------------------- #

def recovery_registry(
    *,
    base_config: CaseConfig,
    triton_runner: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    coverage: CoverageMap | None = None,
    dependence: InputDependenceMap | None = None,
    ledger: Any = None,
    decisions: Mapping[str, Any] | None = None,
    window_kind: str = "idx",
    cohort_traces: "Sequence[Sequence[Instruction]]" = (),
    input_keys: "Sequence[str] | None" = None,
    disp_thresholds: "Mapping[str, float] | None" = None,
    cohort_mem_sidecars: "Sequence[Any] | None" = None,
    cohort_load_diagnostics: "Mapping[str, Any] | None" = None,
    pointer_chain: "PointerChainSpec | None" = None,
    budget: "CvdBudget | None" = None,
    mem_sink: "Mapping[str, Any] | None" = None,
) -> Registry:
    """Build a CVD :class:`Registry` for a recovery run (the three roles only).

    Pass the result as ``registry=`` to ``cvd.run_cvd``; pair with
    ``collect_extensions=True`` to get the whole gap map from one run.

    ``cohort_traces`` / ``input_keys`` enable the evidence-backed mem disposition
    (auto-prefill symbolize for input-varying loads; recommend back; leave the
    truly ambiguous PENDING). Absent → zero prefill (today's all-PENDING).
    ``cohort_mem_sidecars`` (parallel to ``cohort_traces``) carries each cohort
    vector's ``_mem.jsonl`` so the cohort is merged SYMMETRICALLY with the main
    trace (坎1) — otherwise a bare-fed vector is blind in the mem dimension and the
    observability gate vetoes the whole cohort.

    ``pointer_chain`` is an OPTIONAL override only (坎3): by default drive
    SELF-DERIVES the staging pointer-chain shape from its own opaque diagnosis, so
    the caller never has to hand-type a case-specific shape. Supply it only to
    override that auto-derivation."""
    # A3 collect-layer aggregation: ONE shared chain_id -> [bands] index wired into
    # BOTH the generator (populates it as it emits on-path bands) and the verifier
    # (reads the same-chain group for the composite plan). Symmetry by construction —
    # the registry guarantees both sides see the same source, never a caller obligation
    # to keep two band lists in sync.
    band_registry = _OnpathBandRegistry()
    return (Registry()
            .register(RecoveryWindowGenerator(
                coverage=coverage, dependence=dependence, window_kind=window_kind,
                band_registry=band_registry,
                # Output-provenance PRIMARY anchor: the located target-output address
                # is the case's sink hint (case-specific → injected, never hardcoded);
                # the oracle expected bytes arrive per-run on state.expected. Both
                # present → on-path generation is primary, variance secondary; absent
                # → invariant 7 (today's coverage/variance generation byte-for-byte).
                sink_base=base_config.sink_hint_addr,
                # Generation/backtrace budget (dev-recovery-generation-budget-spec):
                # bounds the provenance backtrace + caps the on-path candidate count.
                # Default CvdBudget()'s generous ceilings never trip a small case.
                budget=budget))
            .register(RecoverWindowVerifier(
                base_config=base_config, triton_runner=triton_runner,
                ledger=ledger, decisions=decisions, cohort_traces=cohort_traces,
                input_keys=input_keys, disp_thresholds=disp_thresholds,
                cohort_mem_sidecars=cohort_mem_sidecars,
                cohort_load_diagnostics=cohort_load_diagnostics,
                pointer_chain=pointer_chain,
                band_registry=band_registry,
                # Composite recovery (Req6): the SAME budget the generator uses carries
                # the composite-cost ceiling (max_composite_symex_items) so a band that
                # fails parity in isolation gets a COMPOSITE_REQUIRED /
                # COMPOSITE_TOO_EXPENSIVE plan with an honest cost estimate.
                budget=budget,
                # Issue 7 — EXPLICIT mem-write recovery sink descriptor. When the
                # window's OUTPUT is a store (not a register), this names it; the
                # verifier derives the store interval + drives the runner in mem-sink
                # mode. None → register path, byte-for-byte today (the regression guard).
                mem_sink=mem_sink,
                # pre-flight observable-variance gate: the verifier holds the SAME
                # dependence map the generator/terminal classifier do, so a localized
                # candidate window with zero variance is BLOCKed before drive().
                dependence=dependence))
            .register(RecoveryTerminalClassifier(dependence=dependence)))


# --------------------------------------------------------------------------- #
# Consumer one-shot entry — close the hand-wired half (the other half is
# cvd.run_cvd_collect_to_json, which only owns the OUTPUT layer). The wiring a
# test-agent otherwise hand-rolls (and gets wrong) is encoded here once.
# --------------------------------------------------------------------------- #

# Non-judgment checkpoints with a safe default the run can proceed on (so drive
# reaches symex instead of immediately PENDING). A GENUINE judgment
# (mem_input_symbolize_vs_back) is deliberately ABSENT — it stays PENDING and is
# handed back to the agent (arch invariant 8: the engine never auto-decides it).
_DEFAULT_RECOVERY_DECISIONS: dict[str, Any] = {
    "alias_vs_compute": "compute",
    "which_static": [],
}


