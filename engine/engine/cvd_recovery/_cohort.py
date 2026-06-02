"""cvd_recovery.cohort section (split from the monolithic module)."""
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


# The hard symbolic-recovery frontier this run is allowed to surface but not
# pretend to solve — a symbol entering through pointer-indirect opaque staging.
OPAQUE_STAGING_FRONTIER = (
    "symbol does not propagate through the window: the input arrives via an opaque "
    "staging buffer (pointer-indirect), so the recovered F collapses to a constant. "
    "Surfacing as the opaque-staging frontier — see "
    "dev-symbolic-input-through-opaque-staging.md. Do NOT hand-fill over it.")

RECOVER_WINDOW = "recover_window"

# Output-provenance anchoring (dev-output-provenance-anchored-window-gen-spec.md):
# the recovery window generator's PRIMARY anchor is "what feeds the target output"
# (its provenance producer chain), NOT "where input-variance is" (cohort/dispatch
# variance). variance is right for "where the input reaches state" but it is NOT
# the same as "on the output path": an input-varying dispatch intermediate can be
# OFF the path that produces the target bytes. So when the target output is known
# (sink_base + oracle expected bytes), windows are generated ON the provenance
# path and variance is demoted to a secondary ordering filter WITHIN those on-path
# windows. With no target output, the generator is byte-for-byte today's coverage/
# variance generation (invariant 7).
SIG_PROVENANCE_ONPATH = "output_provenance_onpath"
SIG_PROVENANCE_OFFPATH_VARIANCE = "offpath_variance_demoted"
SIG_RECAPTURE_DIRECTIVE = "recapture_directive"
SIG_PROVENANCE_UNANCHORED = "output_provenance_unanchored"
# A NEEDS_OBSERVATION provenance whose remaining gaps are ALL unplaceable (a read of
# an address with NO reading PC to hang a watch on, ``next_watch`` entry ``pc is
# None``, oracle_provenance L433/L479) is NOT closed (the gap is real) but also NOT
# recapturable (no PC to arm) → an explicit BLOCKED terminal, never silently treated
# as closed (A2 坎 / A8④). Distinct from UNANCHORED (output observed, no producer
# chain) — here the producer chain exists but terminates at an un-hookable read.
SIG_PROVENANCE_BLOCKED_UNPLACEABLE = "output_provenance_blocked_unplaceable"
SRC_OUTPUT_PROVENANCE = "output_provenance"


def _placeable_next_watch(prov: Any) -> list[dict]:
    """The subset of ``prov.next_watch`` that CAN be armed as a watch: each gap that
    names a reading ``pc`` (``pc is not None``). Mirrors the placeable/unplaceable
    split in :func:`engine.recapture.observe_points_from_provenance` (recapture.py:147)
    — a gap with no reading PC cannot have a code hook hung on it (one shape, reused,
    not re-derived)."""
    out: list[dict] = []
    for w in getattr(prov, "next_watch", None) or []:
        if w.get("pc") is not None:
            out.append(w)
    return out


def _unplaceable_next_watch(prov: Any) -> list[dict]:
    """The complement of :func:`_placeable_next_watch`: gaps with ``pc is None`` —
    reads of an address with no PC to arm a watch on. A non-empty list here is a real
    UNplaceable gap → BLOCKED, not closed (A2)."""
    out: list[dict] = []
    for w in getattr(prov, "next_watch", None) or []:
        if w.get("pc") is None:
            out.append(w)
    return out


class _OnpathBandRegistry:
    """Run-level ``chain_id -> [bands]`` index (A3 collect-layer aggregation).

    The single shared source the generator POPULATES (every on-path BAND it emits,
    keyed by its ``chain_id`` = the located sink_base) and the verifier READS (a
    BAND_PARITY_FAIL candidate looks up ALL same-chain bands so the composite planner
    sees the whole group, not just the one band the driver happened to hand it).

    Symmetry is by construction: ``recovery_registry`` wires ONE instance into BOTH
    plugins (no caller obligation to keep two lists in sync). Bands are stored unique
    + sorted so the planner's "≥2 adjacent on-path bands" rule sees a stable group.
    The G1 boundary holds — this is a within-run, same-execution index that is rebuilt
    fresh each run (the registry is per-run); it never accumulates across reruns."""

    def __init__(self) -> None:
        self._by_chain: dict[Any, list[tuple[int, int]]] = {}

    def record(self, chain_id: Any, lo: int, hi: int) -> None:
        if chain_id is None:
            return
        group = self._by_chain.setdefault(chain_id, [])
        band = (int(lo), int(hi))
        if band not in group:
            group.append(band)

    def group(self, chain_id: Any) -> list[tuple[int, int]]:
        """The unique, sorted band group for ``chain_id`` ([] when unknown)."""
        if chain_id is None:
            return []
        return sorted(set(self._by_chain.get(chain_id, [])))

    def group_size(self, chain_id: Any) -> int:
        return len(self.group(chain_id))

    def chains(self) -> dict[Any, list[tuple[int, int]]]:
        return {cid: sorted(set(bands)) for cid, bands in self._by_chain.items()}
# Generation/backtrace budget exhausted (dev-recovery-generation-budget-spec): the
# on-path candidate cap and/or the provenance backtrace depth/breadth ceiling
# truncated this run's generation. Surfaced as an explicit candidate (never a silent
# drop, A8④); its payload reports what was cut, how many, and the retained order.
SIG_GENERATION_BUDGET_EXHAUSTED = "generation_budget_exhausted"

# Composite recovery (dev-recovery-bands-decisions-composite-spec Req6): when a single
# on-path BAND is a genuine algorithm slice but its ISOLATED parity does not close the
# whole output, recovery has a middle tier between "isolated band" and "one huge symex
# window" — COMBINE adjacent on-path bands via chained symbolic state. The three
# terminals (aligned with the existing verdict style, A8④ — every degenerate end
# still emits a verdict):
#   * BAND_PARITY_FAIL      — the band's own parity did NOT close (an isolated slice
#                             is a SIGNAL, not silence): emitted F failed the floor.
#   * COMPOSITE_REQUIRED    — a single band cannot close it but ADJACENT on-path bands
#                             are available to combine: carries the band list to combine.
#   * COMPOSITE_TOO_EXPENSIVE — the combined symex work exceeds the cost budget: carries
#                             the band list + a cost estimate (a comfortable exit, never
#                             a >90s hang — feedback_red_line_needs_comfortable_exit).
TERMINAL_BAND_PARITY_FAIL        = "BAND_PARITY_FAIL"
TERMINAL_COMPOSITE_REQUIRED      = "COMPOSITE_REQUIRED"
TERMINAL_COMPOSITE_TOO_EXPENSIVE = "COMPOSITE_TOO_EXPENSIVE"

# Per-window memory disposition (dev-recovery-bands-decisions-composite-spec Req5).
# The recovery generator produces windows FAR from the initial (0,800) window the
# early disposition map covered; once recovery generates an idx28000+/41000+ window
# its live-in reads land on DIFFERENT staging/heap/table/stack addresses. A window
# whose external memory live-in could NOT be classified (no symbolize-vs-back
# disposition is available for it — no cohort to compare, a blind window, a shallow
# cohort) has a MISSING decision, NOT a known algorithm property. If such a window
# then collapses to opaque / a constant, that collapse is the MISSING-DECISION
# artifact wearing an algorithm-property mask (the symex treated an un-classified
# load as un-resolved input / a propagatable value as a constant). Surfacing it as
# opaque_staging / a constant trap would be a FALSE terminal. This terminal keeps
# "the window's memory was never classified" strictly SEPARATE from "the symbol
# genuinely does not propagate" (A8④: a degenerate window still emits an honest
# verdict; the WARN tops the terminal, never silent — feedback_construct_symmetry_
# not_caller_obligation: the disposition is self-dispatched per window, never the
# early map silently reused, and a missing one is named, not collapsed).
TERMINAL_MEMORY_DISPOSITION_MISSING = "MEMORY_DISPOSITION_MISSING"

# Issue 7 — mem-write / window-output recovery (spec_f0_mem_write_window_sink.md).
# A recovery window whose OUTPUT is a memory write (a store), not a register: the
# sink descriptor names the store interval [sink_addr, sink_addr+sink_size) whose
# bytes are the recovered output. When the runner CANNOT derive the store's
# effective address (no trace mem op at sink_idx, no S3 byte-granular mem_deps to
# pin it, B5-mem could not resolve a symbolic EA) OR cannot read the symbolic
# memory bytes after the sink store, the window is UNPLACEABLE — a structured dead
# end, NEVER a silent fall-back to a register/constant (A8④ / 契约③). The
# verdict carries the exact ``needed[]`` list the spec pins. NOTE: an input-
# invariant store (constant across vectors, seed/driver-independent) is a DIFFERENT
# condition — it is NOT unplaceable; it routes through the EXISTING seed-
# independence exclusion / UNCLOSABLE path, not this terminal (A8④).
TERMINAL_MEM_SINK_UNPLACEABLE = "MEM_SINK_UNPLACEABLE"

# The structured ``needed[]`` list the MEM_SINK_UNPLACEABLE blocker carries —
# clark's VERBATIM shape (spec_f0_mem_write_window_sink.md §"Structured blocker").
MEM_SINK_UNPLACEABLE_NEEDED = ["trace mem op", "pc-local regs", "EA decode",
                               "memory backing"]


@dataclass(frozen=True)
class _ProvenanceAnchor:
    """The output-provenance generation outcome (internal to the generator).

    * ``candidates`` — the candidates this anchor contributes (on-path windows, OR a
      single recapture-directive / unanchored diagnostic candidate).
    * ``onpath_idxs`` — the trace idxs on the provenance producer chain, used to tag
      coverage/variance windows on/off-path and DEMOTE the off-path ones. ``None``
      when there is no on-path window axis (recapture / unanchored paths).
    * ``onpath_distance`` — {idx: hops-from-sink} for the producer-chain idxs (the
      sink writer is distance 0). Exposed on each on-path window payload.
    * ``suppress_secondary`` — True when the variance/coverage fall-back MUST NOT run
      (output writer unobserved → recapture; or output unanchored → explicit report).
      That silent fall-back to off-path variance windows is the 绕路 root cause."""

    candidates: list[Candidate]
    onpath_idxs: set[int] | None
    onpath_distance: dict[int, int]
    suppress_secondary: bool

# --------------------------------------------------------------------------- #
# Three-factor remedy tags (task 2) — name WHICH cohort fix a BLOCK/UNCLOSABLE
# calls for, so the agent修对 (not the laundry-list "fix cohort"). These are
# evidence keys only — additive, machine-routable, and they do NOT touch any
# close / UNCLOSABLE / pre-flight verdict (invariant 7). Mutually exclusive by the
# three structural cohort states they tag:
#   * RE_ANCHOR       — window has NO input-variance (zero varying position): the
#                       input never excited it → re-anchor to divergence_idx
#                       (more/different seeds will not help). [pre-flight (A)]
#   * ADD_SEEDS       — window HAS variance but n_vectors < min: the cohort is too
#                       SMALL to reach the distinct floor → supply MORE seeds.
#                       [pre-flight (B)]
#   * DIVERSIFY_SEEDS — window has variance, n_vectors >= min, yet the independent
#                       side's observed_distinct < min (outputs COLLIDE): the
#                       cohort is big enough but its outputs are not distinct →
#                       supply more OUTPUT-DIVERSE seeds (not more of the same).
#                       [post-hoc UNCLOSABLE]
# --------------------------------------------------------------------------- #
REMEDY_RE_ANCHOR      = "re-anchor"
REMEDY_ADD_SEEDS      = "add-seeds"
REMEDY_DIVERSIFY_SEEDS = "diversify-seeds"

# Gap-map output discipline (utov-arch-index invariant 4 / output-backtrack
# addendum §1): the recovery output is a STRUCTURED gap map — necessary info only,
# never a trace dump. A list longer than this becomes {count, sha1, sample} so a
# 41416-step trace is never inlined into a candidate payload / verdict evidence.
_MAX_INLINE_LIST = 16


def _compact(value: Any, _depth: int = 0) -> Any:
    """Replace oversized lists with a count+hash digest, recursing into dicts/lists.

    Same shape as ``cvd_ledger._trim_payload`` (kept dependency-free here): the
    recovery gap map stays a small structured summary, not a register/mem dump."""
    if _depth > 8:
        return "<deep>"
    if isinstance(value, dict):
        return {k: _compact(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_INLINE_LIST:
            return {
                "_trimmed_list": True,
                "count": len(value),
                "sha1": hashlib.sha1(
                    json.dumps(list(value), default=str, sort_keys=True).encode()
                ).hexdigest(),
                "sample": [_compact(v, _depth + 1) for v in list(value)[:8]],
            }
        return [_compact(v, _depth + 1) for v in value]
    return value


def _drive_evidence(result: DriveResult, window, window_kind: str,
                    *, disposition: str | None = None) -> dict[str, Any]:
    """A compact, necessary-only summary of a drive() outcome for the gap map.

    Keeps what the agent needs to act (the recovered F source, parity, which gate
    it stopped at, what is missing) and DROPS the bulk (full per-step trail,
    address-closure / mem-backing lists, the trace itself). Invariant 4: the gap
    map is a map, not a trace dump."""
    stopped_at = result.per_step[-1]["step"] if result.per_step else None
    ev: dict[str, Any] = {
        "window": list(window), "window_kind": window_kind,
        "closed": result.closed, "parity": result.parity,
        "emitted_F": result.emitted_F,          # the recovered F — necessary, kept
        "note": result.note, "stopped_at": stopped_at,
        "backing_ok": result.backing_ok,
        "self_check": (result.self_check or {}).get("status")
        if result.self_check else None,
        # Issue 7 — the sink shape the self-check compared in (reg | mem). "reg" /
        # absent → the register path (byte-for-byte today). "mem" surfaces that this
        # window's output is a store whose BYTES were compared (not an x8 value).
        "sink_form": (result.self_check or {}).get("sink_form")
        if result.self_check else None,
        "unmodeled": bool(result.unmodeled),
        # Opaque-staging Phase 2(i) "+ record-a-line": how many loads forwarded this
        # run. With per_step's symbolic_staging_injected, injected>0 & forwards==0 is
        # the tc4 "wired but never hit" signal, visible without guessing on closed=0.
        "symbolic_forwards": result.symbolic_forwards,
    }
    if disposition is not None:
        ev["disposition"] = disposition
    return _compact(ev)


# --------------------------------------------------------------------------- #
# Evidence-backed mem disposition — recommend symbolize/back from cohort variance.
#
# ``mem_input_symbolize_vs_back`` is the ONE genuine agent judgment of a recovery
# run (arch invariant 8). Its essence is "does this loaded value VARY across the
# input cohort (→ symbolize: a new input that arrived) or stay CONSTANT (→ back:
# a state carrier / table base)". Once ⑥ gives the trace ``Instruction.mem``,
# that question is deterministically *computable* from cohort value variance —
# gated by ③ observability (don't trust a blind window) and the cohort
# ``input_keys`` gate (don't trust a single trace / a cohort whose inputs never
# actually varied). Three confidence tiers (honesty / invariant 8):
#   * value VARIES + observable        → "symbolize" / "auto"      (reliable dir,
#                                                                    auto-prefill)
#   * value CONSTANT + wide + observable → "back" / "recommend"    (RISK dir — a
#                                            constant true input, e.g. the same key
#                                            captured each run, would be wrongly
#                                            backed; recommend only, NEVER auto)
#   * cohort<2 / inputs not varied / blind → None / "none"         (genuinely
#                                                                    ambiguous →
#                                                                    stays PENDING)
# --------------------------------------------------------------------------- #

_DEFAULT_DISP_THRESHOLDS: dict[str, float] = {
    "min_cohort": 2.0,        # need >= 2 cohort traces to compare values
    "min_input_keys": 2.0,    # need >= 2 DISTINCT input_keys (inputs truly varied)
}


def _merge_cohort_mem_sidecars(
    cohort_traces: Sequence[Sequence[Instruction]],
    cohort_mem_sidecars: Sequence[Any] | None,
) -> tuple[list[list[Instruction]], dict[str, Any]]:
    """Symmetrically fold each cohort vector's ``_mem.jsonl`` into its main stream.

    The main trace is enriched (its ``_mem.jsonl`` merged via
    ``runner_client.JsonlTraceReader.merged`` → ``merge_trace_sources``) *before*
    it reaches recovery; the cohort vectors arrived as bare ``Instruction``
    sequences (``mem=()``). This restores the symmetry: each cohort vector with a
    sidecar path gets the SAME ``read_mem_events`` + ``merge_trace_sources``
    treatment the main trace already had. ``cohort_mem_sidecars`` is parallel to
    ``cohort_traces`` (index k = vector k); a ``None``/missing/falsy entry leaves
    that vector verbatim (invariant 7: no sidecar → byte-for-byte unchanged).

    Reuses existing primitives only (``obs_readers.read_mem_events`` /
    ``trace_merge.merge_trace_sources``) — no new parse. Returns
    ``(merged_traces, report)`` where ``report`` is a small structured summary of
    what was folded (never a trace dump; invariant 4). Best-effort per vector: a
    sidecar read/merge error is recorded, not raised — a missing sidecar must not
    lose the cohort (it just degrades that vector to its bare form)."""
    traces = [list(t) for t in cohort_traces]
    sidecars = list(cohort_mem_sidecars or ())
    report: dict[str, Any] = {
        "vectors": len(traces),
        "sidecars_supplied": sum(1 for s in sidecars if s),
        "merged": [],          # per-vector {vector, mem_events_merged, unaligned}
        "errors": [],          # per-vector {vector, error}
    }
    if not any(sidecars):
        return traces, report
    from ..obs_readers import read_mem_events
    from ..trace_merge import merge_trace_sources
    for k, trace in enumerate(traces):
        side = sidecars[k] if k < len(sidecars) else None
        if not side:
            continue
        try:
            events = read_mem_events(side)
            merged = merge_trace_sources(trace, mem_events=events)
            traces[k] = list(merged.items)
            rep = merged.report
            report["merged"].append({
                "vector": k,
                "mem_events_merged": rep.mem_events_merged if rep else 0,
                "unaligned": len(rep.unaligned) if rep else 0,
            })
        except Exception as e:          # best-effort: a bad sidecar degrades, not raises
            report["errors"].append({"vector": k, "error": str(e)})
    return traces, report


def load_cohort_traces(
    cohort_trace_paths: Sequence[str | Path],
    *,
    cohort_mem_sidecars: Sequence[Any] | None = None,
) -> tuple[list[list[Instruction]], dict[str, Any]]:
    """Load cohort traces from PATHS the same way the main trace is loaded.

    The 68a873e anti-pattern was: the main trace went through
    ``JsonlTraceReader(path).merged()`` (auto-folding its ``_mem.jsonl``) while the
    cohort was loaded bare OUTSIDE utov (no ``merged()``) and then asked to be
    "repaired" via a parallel ``cohort_mem_sidecars`` array the caller had to
    remember. This loader closes that asymmetry at the SOURCE: every vector is read
    through the SAME ``JsonlTraceReader(p).merged()`` entry (with automatic
    ``<stem>_mem.jsonl`` sibling resolution — see
    ``runner_client.mem_sidecar_sibling``), so symmetry is guaranteed BY
    CONSTRUCTION — the caller gives only paths and cannot feed one rich + three
    bare.

    ``cohort_mem_sidecars`` is an OPTIONAL explicit OVERRIDE (parallel to the
    paths): a non-falsy entry k forces that exact sidecar for vector k instead of
    the auto-resolved sibling; ``None``/missing → auto-resolution. It is no longer
    a caller obligation — the default (omit it) is already symmetric.

    Returns ``(traces, report)``. ``report`` records, per vector, the resolved
    sidecar (or that NONE was found) so the WARN for a vector with neither an
    explicit sidecar nor an on-disk sibling is surfaced at the layer boundary
    (invariant 1) — degradation is allowed, silence is not. Best-effort per vector:
    a read/merge error degrades that vector to bare, never raises (a bad sidecar
    must not lose the cohort)."""
    from ..runner_client import (
        JsonlTraceReader,
        mem_sidecar_sibling,
        unmerged_mem_sidecars,
    )
    overrides = list(cohort_mem_sidecars or ())
    traces: list[list[Instruction]] = []
    report: dict[str, Any] = {
        "vectors": len(cohort_trace_paths),
        "loaded": [],            # per-vector {vector, path, mem_sidecar, mem_events}
        "no_mem_sidecar": [],    # per-vector {vector, path, warn} (degraded, visible)
        "unmerged_sidecar_looking": [],  # per-vector {vector, path, files, warn}
        "errors": [],            # per-vector {vector, path, error}
    }
    for k, path in enumerate(cohort_trace_paths):
        override = overrides[k] if k < len(overrides) and overrides[k] else None
        try:
            reader = JsonlTraceReader(path, mem_sidecar=override) if override \
                else JsonlTraceReader(path)
            resolved = reader.resolve_mem_sidecar()
            merged = reader.merged()
            traces.append(list(merged.items))
            rep = merged.report
            report["loaded"].append({
                "vector": k,
                "path": str(path),
                "mem_sidecar": str(resolved) if resolved else None,
                "mem_events": rep.mem_events_merged if rep else 0,
            })
            if resolved is None:
                # Neither an explicit override nor a conventional sibling found →
                # this vector stays bare in the mem dimension. Do NOT silently batch-
                # degrade: record a WARN the caller surfaces in the gap map evidence.
                report["no_mem_sidecar"].append({
                    "vector": k,
                    "path": str(path),
                    "warn": (f"cohort vector {k} has no mem sidecar: neither an "
                             f"explicit override nor a conventional sibling "
                             f"('{mem_sidecar_sibling(path).name}' or "
                             f"'{Path(path).stem}_mem_sidecar.jsonl') was found "
                             "— vector loaded bare (mem dimension blind over the "
                             "window); the all()-veto degradation applies"),
                })
            # De-silence (task 5): a mem-sidecar-LOOKING file in the directory that
            # was NOT the one merged (a differently-stemmed sidecar, a stray family
            # member) must be WARN-ed, never silently ignored. invariant 1: surface
            # at the boundary so a present-but-unmerged mem dimension is visible.
            stray = unmerged_mem_sidecars(path, resolved)
            if stray:
                report["unmerged_sidecar_looking"].append({
                    "vector": k,
                    "path": str(path),
                    "files": [str(s) for s in stray],
                    "warn": (f"cohort vector {k}: {len(stray)} mem-sidecar-looking "
                             f"file(s) in the directory were NOT merged "
                             f"({[s.name for s in stray]}) — if one of these is the "
                             "vector's real mem sidecar, name it "
                             f"'{mem_sidecar_sibling(path).name}' / "
                             f"'{Path(path).stem}_mem_sidecar.jsonl' or pass it "
                             "explicitly; mem dimension may be silently blind"),
                })
        except Exception as e:    # best-effort: a bad path/sidecar degrades, not raises
            traces.append([])
            report["errors"].append({"vector": k, "path": str(path), "error": str(e)})
    return traces, report


@dataclass(frozen=True, slots=True)
class MemDispositionRec:
    """Evidence-backed recommendation for one external memory input's disposition.

    ``confidence`` is the honest tier — only ``"auto"`` (symbolize) is safe to
    prefill; ``"recommend"`` (back) is surfaced to the agent but never applied;
    ``"none"`` is a genuine ambiguity left fully to the agent (invariant 8)."""

    addr:         int
    disposition:  str | None      # "symbolize" | "back" | None
    confidence:   str             # "auto" | "recommend" | "none"
    reason:       str
    value_varies: bool | None     # None = could not determine (gated out)
    n_cohort:     int             # cohort traces that actually loaded this addr
    observable:   bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "addr":         f"0x{self.addr:x}",
            "disposition":  self.disposition,
            "confidence":   self.confidence,
            "reason":       self.reason,
            "value_varies": self.value_varies,
            "n_cohort":     self.n_cohort,
            "observable":   self.observable,
        }


def _cohort_loaded_values(
    cohort_traces: Sequence[Sequence[Instruction]],
    addr: int,
    *,
    window: tuple[int, int],
    window_is_idx: bool,
) -> list[int]:
    """The value each cohort trace loads from ``addr`` inside the window.

    Aligned by effective address (``MemOp.addr``) — the same external input
    address recurs across vectors (the cohort_diff alignment thesis, address
    granularity). One value per vector (the FIRST read covering ``addr`` in that
    vector's window); a vector that never reads ``addr`` contributes nothing.
    Pure / deterministic, no DFG, no symbolic input."""
    vals: list[int] = []
    for trace in cohort_traces:
        lo, hi = int(window[0]), int(window[1])
        if lo > hi:
            lo, hi = hi, lo
        key = (lambda ins: ins.idx) if window_is_idx else (lambda ins: ins.pc)
        found: int | None = None
        for ins in trace:
            if not (lo <= key(ins) <= hi):
                continue
            for op in ins.mem:
                if op.rw == "r" and op.addr <= addr < op.addr + op.size:
                    found = op.val
                    break
            if found is not None:
                break
        if found is not None:
            vals.append(found)
    return vals


def _current_run_loaded_value(
    items: Sequence[Instruction],
    addr: int,
    *,
    window: tuple[int, int],
    window_is_idx: bool,
) -> int | None:
    """This run's loaded value at ``addr`` inside the window (seed for symbolize).

    ``None`` when the address is not read in the window. Same alignment as
    :func:`_cohort_loaded_values` (one trace)."""
    vals = _cohort_loaded_values(
        [items], addr, window=window, window_is_idx=window_is_idx)
    return vals[0] if vals else None


def recommend_mem_disposition(
    mem_inputs: Sequence[MemLiveIn],
    cohort_traces: Sequence[Sequence[Instruction]],
    *,
    window: tuple[int, int],
    window_is_idx: bool = True,
    input_keys: Sequence[str] | None = None,
    thresholds: Mapping[str, float] | None = None,
    cohort_mem_sidecars: Sequence[Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[int, MemDispositionRec]:
    """Deterministic evidence for each external memory input's disposition.

    Returns ``{addr: MemDispositionRec}``. Zero LLM, pure function (invariant 1).
    Only the reliable direction (value VARIES → symbolize) earns ``confidence
    ="auto"``; the risk direction (value CONSTANT → back) is ``"recommend"`` only;
    everything gated out (cohort<2 / inputs not truly varied / blind window) is
    ``None`` / ``"none"`` and stays the agent's call (invariant 8).

    ``cohort_mem_sidecars`` (parallel to ``cohort_traces``) restores merge
    SYMMETRY: each cohort vector's ``_mem.jsonl`` is folded in exactly as the main
    trace's already was, so a bare-fed vector no longer reads ``mem`` rate=0 and
    fails the observability gate (坎1 root cause). If a ``diagnostics`` dict is
    supplied, the merge report and the observability-symmetry decision (WARN +
    degradation marker) are written there for the caller to surface as evidence —
    keeping the WARN at the layer boundary, not buried (invariant 1)."""
    th = dict(_DEFAULT_DISP_THRESHOLDS)
    if thresholds:
        th.update(thresholds)
    min_cohort = int(th.get("min_cohort", 2))
    min_keys = int(th.get("min_input_keys", 2))

    # 坎1 main fix — cohort symmetric merge: fold each vector's _mem.jsonl in the
    # SAME way the main trace's was, BEFORE the observability gate (so a bare-fed
    # vector is no longer blind in the mem dimension purely for lack of merge).
    traces, merge_report = _merge_cohort_mem_sidecars(
        cohort_traces, cohort_mem_sidecars)
    n_cohort = len(traces)
    n_distinct_keys = len(set(input_keys)) if input_keys is not None else None

    # ③ observability gate: assess each cohort trace over the window. The mem
    # dimension must be observable to trust value comparison (a blind vector cannot
    # corroborate constancy/variance). Thresholds ride the SAME ``thresholds`` map.
    obs_per_trace = [
        assess_trace_observability(
            t, window=window, window_is_idx=window_is_idx, thresholds=th,
        ).overall_sufficient_for("mem")
        for t in traces
    ]
    n_observable = sum(1 for o in obs_per_trace if o)
    # all()-veto DEGRADATION (兜底): instead of one bad vector vetoing the whole
    # cohort (→ 93 independent nulls), use the OBSERVABLE SUBSET when it is wide
    # enough (>= min_cohort). The value comparison below still runs only over
    # vectors that actually loaded the addr, so trusting an observable subset only
    # NARROWS the evidence — it never invents symbolize/back (invariant 8: the
    # degradation gives a batch-processable outcome, never a fabricated verdict).
    symmetric = bool(obs_per_trace) and all(obs_per_trace)
    if symmetric:
        observable = True
        degraded = None
    elif n_observable >= min_cohort:
        # asymmetric but enough observable vectors → judge on the subset, WARN.
        observable = True
        traces = [t for t, o in zip(traces, obs_per_trace) if o]
        n_cohort = len(traces)
        degraded = {
            "mode": "observable_subset",
            "observable": n_observable,
            "total": len(obs_per_trace),
            "warn": (f"cohort merge asymmetric: {len(obs_per_trace) - n_observable} "
                     f"of {len(obs_per_trace)} vectors lack an observable mem "
                     "dimension over the window (missing/unmerged sidecar?) — "
                     f"judging on the observable subset of {n_observable}"),
        }
    else:
        # not enough observable vectors even to compare → one UNIFIED batch
        # degradation, not 93 independent nulls. Still invariant-8 honest: every
        # rec stays disposition=None/confidence="none" (no symbolize/back invented);
        # the single marker tells the agent "cohort unusable, handled in batch".
        observable = False
        degraded = {
            "mode": "cohort_unusable_batch",
            "observable": n_observable,
            "total": len(obs_per_trace),
            "warn": (f"cohort unusable for mem disposition: only {n_observable} of "
                     f"{len(obs_per_trace)} vectors have an observable mem dimension "
                     f"(need >= {min_cohort}). All external mem inputs handled in a "
                     "single batch degradation (left to agent), not per-addr nulls — "
                     "check cohort sidecar merge symmetry"),
        }

    if diagnostics is not None:
        diagnostics["cohort_mem_merge"] = merge_report
        diagnostics["observability"] = {
            "per_vector": list(obs_per_trace),
            "n_observable": n_observable,
            "symmetric": symmetric,
            "degraded": degraded,
        }

    out: dict[int, MemDispositionRec] = {}
    for m in mem_inputs:
        # --- front gates (invariant 8: low evidence → ambiguous, never invented) ---
        if n_cohort < min_cohort:
            out[m.addr] = MemDispositionRec(
                m.addr, None, "none",
                reason=(f"no cohort variance evidence (have {n_cohort} trace(s), "
                        f"need >= {min_cohort}) — agent must decide symbolize vs back"),
                value_varies=None, n_cohort=n_cohort, observable=observable)
            continue
        if n_distinct_keys is not None and n_distinct_keys < min_keys:
            out[m.addr] = MemDispositionRec(
                m.addr, None, "none",
                reason=(f"inputs did not truly vary (distinct input_keys="
                        f"{n_distinct_keys} < {min_keys}) — value constancy is not "
                        "informative; agent must decide"),
                value_varies=None, n_cohort=n_cohort, observable=observable)
            continue
        if not observable:
            batch = ("; cohort unusable → batch degradation (see diagnostics)"
                     if degraded and degraded.get("mode") == "cohort_unusable_batch"
                     else "")
            out[m.addr] = MemDispositionRec(
                m.addr, None, "none",
                reason=("low observability: the mem dimension is not sufficiently "
                        "covered across the cohort window — cannot trust the value "
                        "comparison; agent must decide" + batch),
                value_varies=None, n_cohort=n_cohort, observable=observable)
            continue

        # --- value variance across the input-varying cohort ---
        vals = _cohort_loaded_values(
            traces, m.addr, window=window, window_is_idx=window_is_idx)
        n_seen = len(vals)
        if n_seen < min_cohort:
            # The address was not actually loaded in >=2 cohort vectors — cannot
            # compare. Genuinely ambiguous (not a fabricated verdict).
            out[m.addr] = MemDispositionRec(
                m.addr, None, "none",
                reason=(f"address loaded in only {n_seen} cohort vector(s) "
                        f"(need >= {min_cohort}) — no value comparison; agent decides"),
                value_varies=None, n_cohort=n_seen, observable=observable)
            continue
        varies = len(set(vals)) > 1
        if varies:
            # Reliable direction: a value that changes with the input IS an input.
            out[m.addr] = MemDispositionRec(
                m.addr, "symbolize", "auto",
                reason=(f"value varies across the input-varying cohort "
                        f"({len(set(vals))} distinct over {n_seen} vectors) — an "
                        "input that arrived; auto-symbolize"),
                value_varies=True, n_cohort=n_seen, observable=observable)
        else:
            # RISK direction: constant could be a carrier OR a constant true input
            # (same key each run). Recommend back, but NEVER auto-apply (invariant 8).
            out[m.addr] = MemDispositionRec(
                m.addr, "back", "recommend",
                reason=(f"value constant across {n_seen} input-varying cohort "
                        "vectors — likely a state carrier / table base; RECOMMEND "
                        "back, but confirm: a constant true input (e.g. a fixed key) "
                        "would also look constant — agent decides"),
                value_varies=False, n_cohort=n_seen, observable=observable)
    return out


# --------------------------------------------------------------------------- #
# 1 — CandidateGenerator: windows from the dispatch coverage map / cohort diff.
# --------------------------------------------------------------------------- #

