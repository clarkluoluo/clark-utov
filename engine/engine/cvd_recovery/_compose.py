"""cvd_recovery.compose section (split from the monolithic module)."""
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
from ._cohort import OPAQUE_STAGING_FRONTIER, TERMINAL_COMPOSITE_REQUIRED, TERMINAL_COMPOSITE_TOO_EXPENSIVE


def _classify_drive_result(res: DriveResult) -> tuple[str, str]:
    """Route a non-closed :class:`DriveResult` to a CVD disposition + reason.

    Keyed on drive's STRUCTURED fields (never case data, never note text):
      * ``fixable`` — a correctable geometry/feed/backing gap → ELIMINATED, with a
        spawned corrected candidate where one is well-defined;
      * ``opaque``  — the symbol does not propagate (collapsed F / external mem
        input neither symbolized nor backed) → the opaque-staging frontier;
      * ``seed_invariant`` — the window's whole seed set is constant → not driven
        by the recovery variable;
      * ``capability`` — a Level-2 un-modeled instruction → needs its semantics;
      * ``unsound`` — the G4 self-check BLOCKED (F fails to reproduce its trace);
      * ``parity`` — emitted but failed the multi-vector parity floor.
    """
    if not res.backing_ok:
        return ("fixable", "backing gate not satisfied (blind address closure) — "
                "inject same-execution backing or re-capture, then re-run")
    if res.decode_audit and res.decode_audit.get("systematic"):
        return ("fixable", "systematic decode/byte-feed inconsistency — fix the "
                "byte-feed (endianness/arch/slice), do NOT S-expr fill over it")
    if res.unmodeled is not None:
        return ("capability", "Level-2 symex hit an un-modeled instruction — supply "
                "its symbolic semantics (escape-hatch fill)")
    if res.self_check is not None and res.self_check.get("status") == "BLOCK":
        return ("unsound", "G4 emit self-check BLOCK — recovered F does not "
                "reproduce its own trace (symex unsound); do not emit")
    # An emitted F that failed only the cross-vector parity floor. Surface the
    # numbers (matched/supplied vs need) so "feed pit" (supplied < need) is
    # distinguishable at a glance from "F wrong" (supplied >= need, matched < need).
    if res.emitted_F is not None and res.parity_report is not None \
            and not res.parity_report.get("sufficient", False):
        pr = res.parity_report
        need = pr.get("min_vectors")
        supplied = pr.get("total")
        matched = pr.get("independent_pass")
        return ("parity",
                f"emitted F failed the multi-vector parity floor — "
                f"parity BLOCK: matched {matched}/supplied {supplied}, "
                f"need >= {need} independent cross-run vector(s)")
    # Nothing emitted and none of the above structured gaps fired → the symbol did
    # not propagate. Distinguish a constant-seed window (seed-invariant) from an
    # opaque-staging collapse using the seed step's auto-seed verdict.
    for step in res.per_step:
        if step.get("step") == "seed_entry_state":
            auto = step.get("auto_seed") or {}
            if auto.get("seed_blocked"):
                return ("seed_invariant", "every live-in register is concretely "
                        "backed / pinned — the window is not driven by the recovery "
                        "variable (a setup segment), nothing symbolic to recover")
    return ("opaque", OPAQUE_STAGING_FRONTIER)


# --------------------------------------------------------------------------- #
# Composite recovery planner (dev-recovery-bands-decisions-composite-spec Req6).
#
# After Req4, on-path candidates are BANDS (contiguous algorithm slices). When a
# single band is a real slice but its ISOLATED parity does not close the whole output
# (BAND_PARITY_FAIL), recovery's middle tier is to COMBINE adjacent on-path bands. The
# truly chained symbolic-state execution (run band A, thread its symbolic output state
# as band B's symbolic live-in, compose the expression) is a DEEP symex change — the
# drive() / runner expose no chained-symbolic-state primitive (each drive() run starts
# from a fresh entry state and emits one window's F). Per the spec, this planner lands
# the SCAFFOLDING — band-list assembly, adjacency, the cost estimate, and the
# COMPOSITE_REQUIRED vs COMPOSITE_TOO_EXPENSIVE determination — and the REAL combined
# execution STOPS and reports (status="not_executed"), rather than faking a composite
# that never threaded state. Pure / deterministic over already-computed band geometry
# (invariant 8); no case idx baked (cost is universal: total span × per-item factor).
# --------------------------------------------------------------------------- #

# Per-item symex cost proxy (relative units): the combined window's total span is the
# dominant cost driver (symex is ~linear in modelled items). Universal engineering
# constant, parameterised against budget.max_composite_symex_items — no case number.
_COMPOSITE_ITEM_COST = 1.0


def derive_mem_sink_interval(
    items: Sequence[Instruction],
    descriptor: Mapping[str, Any],
) -> tuple[tuple[int, int] | None, str | None]:
    """Derive ``(sink_addr, sink_size)`` for a mem-write recovery sink (Issue 7).

    The recovery sink descriptor pins the OUTPUT store via ``sink_idx`` (the store
    instruction in the window that writes the output). Per the spec, ``sink_addr`` /
    ``sink_size`` are derived from the trace mem op when the caller did not fill
    them — the caller is NOT required to:

      1. an explicit ``sink_addr`` + ``sink_size`` in the descriptor wins (hex-str
         or int);
      2. else the WRITE :class:`MemOp` of the instruction at ``sink_idx`` (a store's
         ``(addr, size)`` is the byte-granular interval — the same S3 byte-granular
         mem-dep fact the trace records for that store);
      3. else ``(None, reason)`` — the EA could not be pinned (no trace mem op at
         sink_idx, no byte-granular dep, B5-mem could not resolve a symbolic EA).
         The caller routes that to MEM_SINK_UNPLACEABLE; it is NEVER guessed.

    Returns ``((addr, size), None)`` on success, or ``(None, reason)`` when the EA
    cannot be derived (so the verifier surfaces the structured terminal). Pure read
    of the trace / descriptor — never fabricates an address."""
    # 1. explicit addr+size (the caller already pinned it) — accept int or hex-str.
    raw_addr = descriptor.get("sink_addr")
    raw_size = descriptor.get("sink_size")
    if raw_addr is not None and raw_size is not None:
        try:
            addr = int(raw_addr, 0) if isinstance(raw_addr, str) else int(raw_addr)
            size = int(raw_size, 0) if isinstance(raw_size, str) else int(raw_size)
            if size > 0:
                return (addr, size), None
            return None, f"descriptor sink_size={size!r} is non-positive"
        except (TypeError, ValueError):
            return None, (f"descriptor sink_addr/sink_size not parseable: "
                          f"{raw_addr!r}/{raw_size!r}")
    # 2. derive from the trace WRITE mem op at sink_idx (the byte-granular store).
    raw_idx = descriptor.get("sink_idx")
    if raw_idx is None:
        return None, ("no sink_addr/sink_size and no sink_idx — cannot pin the "
                      "store interval (need the trace mem op / EA decode)")
    try:
        sink_idx = int(raw_idx)
    except (TypeError, ValueError):
        return None, f"descriptor sink_idx={raw_idx!r} is not an index"
    for ins in items:
        if ins.idx != sink_idx:
            continue
        writes = [op for op in (ins.mem or ()) if op.rw == "w" and op.size > 0]
        if not writes:
            return None, (f"the instruction at sink_idx={sink_idx} carries no trace "
                          f"write mem op (EA not decoded / store value not observed) "
                          f"— cannot pin the store interval")
        # The store's recorded (addr, size). Multiple writes → the widest covering
        # interval (still a single contiguous store on AArch64; defensive). A caller-
        # given ``sink_size`` narrows the recorded store width (e.g. a 4-byte output
        # inside an 8-byte str), so it is honoured when present and positive.
        lo = min(op.addr for op in writes)
        hi = max(op.addr + op.size for op in writes)
        size = hi - lo
        if raw_size is not None:
            try:
                cand = int(raw_size, 0) if isinstance(raw_size, str) else int(raw_size)
                if cand > 0:
                    size = cand
            except (TypeError, ValueError):
                pass                              # junk size → fall back to the store width
        return (lo, size), None
    return None, (f"no trace instruction at sink_idx={sink_idx} (the store is not "
                  f"in the trace) — cannot pin the store interval")


def _band_window(payload: Mapping[str, Any]) -> tuple[int, int] | None:
    """The ``[lo, hi]`` band window from a candidate/evidence payload, or None."""
    w = (payload or {}).get("window")
    if isinstance(w, (list, tuple)) and len(w) == 2:
        try:
            lo, hi = int(w[0]), int(w[1])
            return (min(lo, hi), max(lo, hi))
        except (TypeError, ValueError):
            return None
    return None


def estimate_composite_cost(bands: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """Deterministic cost estimate for COMBINING ``bands`` into one symex run.

    The combined window spans from the lowest band start to the highest band end; its
    item count (the span) is the dominant symex cost driver. Returns the total span,
    the per-band spans, and the estimated relative symex work. Pure (invariant 8)."""
    spans = [hi - lo + 1 for lo, hi in bands]
    if not bands:
        return {"n_bands": 0, "combined_span": 0, "band_spans": [],
                "estimated_symex_items": 0}
    combined_lo = min(lo for lo, _ in bands)
    combined_hi = max(hi for _, hi in bands)
    combined_span = combined_hi - combined_lo + 1
    return {
        "n_bands": len(bands),
        "combined_window": [combined_lo, combined_hi],
        "combined_span": combined_span,
        "band_spans": spans,
        "summed_band_span": sum(spans),
        # Cost ~ the combined window the chained run must model (the dominant driver).
        "estimated_symex_items": int(combined_span * _COMPOSITE_ITEM_COST),
    }


def plan_composite_recovery(
    bands: Sequence[tuple[int, int]],
    *,
    budget: "CvdBudget | None" = None,
) -> dict[str, Any]:
    """Decide the composite terminal for a set of adjacent on-path bands (Req6).

    Called once an isolated band's parity has FAILED (BAND_PARITY_FAIL) and the agent
    needs the next move. Determines, deterministically over the band geometry:

      * ``terminal``  — ``COMPOSITE_REQUIRED`` when >= 2 adjacent on-path bands exist
        to combine (a single band could not close it; combine its neighbours), or
        ``COMPOSITE_TOO_EXPENSIVE`` when the combined symex cost exceeds
        ``budget.max_composite_symex_items`` (a comfortable exit — band list + estimate
        — instead of a >90s hang). With < 2 bands there is nothing to combine →
        ``COMPOSITE_REQUIRED`` is not raised (the caller stays at BAND_PARITY_FAIL).
      * ``bands``     — the band list to combine (the consumer sees WHICH segments).
      * ``cost``      — the estimate (combined span + estimated symex items).
      * ``executed``  — ALWAYS False here: the real chained symbolic-state execution
        is NOT implemented (a deep symex change — see the module note). The scaffold
        reports honestly; it never fakes a composite that did not thread state.

    Returns ``{}`` (no composite decision) when there are no bands. Pure / det."""
    b = sorted({(lo, hi) for lo, hi in bands})
    budget = budget or CvdBudget()
    if not b:
        return {}
    cost = estimate_composite_cost(b)
    over_budget = cost["estimated_symex_items"] > budget.max_composite_symex_items
    if over_budget:
        terminal = TERMINAL_COMPOSITE_TOO_EXPENSIVE
        reason = (
            f"combining {len(b)} adjacent on-path band(s) would symex a combined "
            f"window of {cost['combined_span']} item(s) (~{cost['estimated_symex_items']} "
            f"symex items), over the budget ({budget.max_composite_symex_items}) — "
            f"surfaced as a comfortable exit (band list + cost estimate) instead of a "
            f"long symex hang; raise budget.max_composite_symex_items or narrow the "
            f"band set to combine fewer segments")
    elif len(b) >= 2:
        terminal = TERMINAL_COMPOSITE_REQUIRED
        reason = (
            f"a single band did not close the output; {len(b)} adjacent on-path bands "
            f"are available to combine via chained symbolic state (combined window "
            f"{cost.get('combined_window')}, ~{cost['estimated_symex_items']} symex "
            f"items, within budget). NOTE: the chained-state composite EXECUTION is "
            f"not yet implemented (a deep symex change) — this names the required "
            f"combination + its cost; it does not run it")
    else:
        # only one band → nothing to combine; the caller stays at BAND_PARITY_FAIL.
        return {
            "terminal": None,
            "bands": [list(x) for x in b],
            "cost": cost,
            "executed": False,
            "reason": ("a single isolated band failed parity but there is no adjacent "
                       "on-path band to combine — stays BAND_PARITY_FAIL (the isolated "
                       "slice is the signal)"),
        }
    return {
        "terminal": terminal,
        "bands": [list(x) for x in b],
        "cost": cost,
        "over_budget": over_budget,
        # The plan NAMES the required combination + its cost. Whether the chained
        # EXECUTION ran is decided by ``execute_composite_recovery`` (the verifier calls
        # it when it has the runner/trace); the plan itself stays a pure decision.
        "executed": False,
        "composite_execution": "planned",
        "reason": reason,
    }


# --------------------------------------------------------------------------- #
# Composite recovery EXECUTION (dev-recovery-bands-decisions-composite-spec Req6
# "Req6 执行设计") — the REAL chained-symbolic-state run.
#
# The plan above names WHICH adjacent on-path bands to combine + the cost. This
# executes the combination by CHAINING each band's symbolic output state into the
# next band's symbolic live-in (reg + mem), composing one expression over the
# ORIGINAL input, and running multi-vector parity on it.
#
# How the chaining is real (not faked) without a deep Triton-internal change: each
# band's symex (``setup_symex.drive``) already emits a CLOSED-FORM transform
# ``def f(<live-in>): return <expr>`` — a function of that band's symbolic live-in.
# Band A's emitted transform IS its symbolic output state (the symbolic expression A
# produces for the reg/mem band B reads as live-in). The chain THREADS that state by
# COMPOSITION: band B's live-in symbol is bound to band A's output expression
# (``f_B(f_A(carrier))``), and so on down the producer-chain order. The handoff is
# reg + mem BOTH (the spec's hard requirement): the producer→consumer edge — which of
# A's outputs is B's live-in — is what determines the substitution; whether that
# value rode a register or a staging/heap memory cell does not change the composition
# (a mem hand-off is the SAME named-symbol substitution as a reg hand-off). The final
# end-band expression is a function of the original input; parity validates it over
# the cohort (the G4/parity/seed gates are NOT relaxed — composite is just one more
# expression to verify).
#
# The honest boundary (spec's "若撞 primitive 缺口"): a band whose symex did NOT emit
# a usable named transform (it collapsed / hit an un-modeled op / produced a raw
# context-local Triton expr that is not a function of a named input) CANNOT be chained
# at the expression layer — that band needs the deeper "inject a symbolic expression
# as the next window's live-in SEED inside one shared concolic context" primitive,
# which the Level-2 runner does not expose (its entry seed takes only a concrete
# shadow value, never a symbolic expression). When that happens the executor STOPS at
# that band and reports ``composite_execution="reg_handoff_only"`` /
# ``"primitive_gap"`` with WHICH band needs the deeper change — it never fakes a
# composite. The reg+mem handoff over emit-layer-composable bands is done; the deep
# concolic-seed injection is the named gap.
# --------------------------------------------------------------------------- #

_COMPOSITE_FN_PREFIX = "_band"


def _band_emitted_inputs(emitted_F: str | None) -> list[str] | None:
    """The parameter names of an emitted ``def f(<params>): ...`` transform.

    ``None`` when there is no emitted function (the band collapsed / emitted a bare
    expression that is not a named function — not chainable at the emit layer). Pure
    textual parse of utov's own emit form (``emit_python`` renders ``def f(...)``)."""
    if not emitted_F:
        return None
    import re
    m = re.search(r"def\s+f\s*\(([^)]*)\)\s*:", str(emitted_F))
    if not m:
        return None
    params = [p.strip() for p in m.group(1).split(",") if p.strip()]
    return params


def compose_band_transforms(
    band_fns: Sequence[tuple[str, str]],
    *,
    outer_input: str,
) -> dict[str, Any]:
    """Compose ordered band transforms into ONE function of ``outer_input``.

    ``band_fns`` is the producer-chain-ordered list of ``(band_label, emitted_F)`` —
    each ``emitted_F`` a ``def f(<live-in>): return <expr>`` transform whose single
    live-in is fed by the PREVIOUS band's output (reg or mem hand-off; the composition
    is the same named-symbol substitution either way). The composite is
    ``f_N(f_{N-1}(... f_1(outer_input) ...))``: band 1 reads the original input, each
    later band reads the prior band's output expression as its symbolic live-in.

    Returns ``{"ok", "composite_F", "n_chained", "stopped_at", "reason"}``. ``ok`` is
    False (with ``stopped_at`` = the first un-chainable band) when a band did not emit
    a single-parameter named transform — that band needs the deeper concolic-seed
    primitive (the executor surfaces it, never fakes the chain). Pure / deterministic;
    builds the composite SOURCE only (evaluation/parity is the caller's, gated)."""
    if not band_fns:
        return {"ok": False, "composite_F": None, "n_chained": 0,
                "stopped_at": None, "reason": "no bands to compose"}
    # Each band becomes a named helper ``_band{k}(x)``; the composite calls them in
    # producer-chain order, threading x = prior band's output (reg/mem hand-off).
    helpers: list[str] = []
    for k, (label, fn) in enumerate(band_fns):
        params = _band_emitted_inputs(fn)
        if not params or len(params) != 1:
            return {
                "ok": False, "composite_F": None, "n_chained": k,
                "stopped_at": label,
                "reason": (
                    f"band {label} did not emit a single-live-in named transform "
                    f"(params={params}); its symbolic output state cannot be chained at "
                    f"the emit layer — this band needs the deeper primitive: inject a "
                    f"symbolic EXPRESSION as the next window's live-in seed inside one "
                    f"shared concolic context (the Level-2 runner's entry seed takes "
                    f"only a concrete shadow value, not a symbolic expression). Surfaced, "
                    f"never faked."),
            }
        # Re-home the band's ``f`` to a unique helper name, renaming its single param to
        # a fixed ``x`` so the helpers compose regardless of each band's input label.
        import re
        helper = re.sub(r"def\s+f\s*\(([^)]*)\)\s*:",
                        f"def {_COMPOSITE_FN_PREFIX}{k}(x):", str(fn), count=1)
        # bind the original param name to x inside the body (so ``carrier``/``b_in`` →
        # x) — a leading alias line, robust to the body referencing the param by name.
        helper = re.sub(
            rf"def {_COMPOSITE_FN_PREFIX}{k}\(x\):",
            f"def {_COMPOSITE_FN_PREFIX}{k}(x):\n    {params[0]} = x", helper, count=1)
        helpers.append(helper)
    # Composite: nest the helper calls in producer-chain order.
    call = outer_input
    for k in range(len(band_fns)):
        call = f"{_COMPOSITE_FN_PREFIX}{k}({call})"
    composite = (
        "\n".join(helpers)
        + f"\n\ndef f({outer_input}):\n    return {call}\n")
    return {
        "ok": True, "composite_F": composite, "n_chained": len(band_fns),
        "stopped_at": None,
        "reason": (f"chained {len(band_fns)} band transform(s) by composing each "
                   f"band's symbolic output state into the next band's symbolic "
                   f"live-in (reg + mem hand-off)"),
    }


# --------------------------------------------------------------------------- #
# §需求1 — terminal block_kind: split the opaque MERGE POINT into 5 mutually
# exclusive root causes (additive; disposition/terminal_kind unchanged).
#
# The disposition above (``_classify_drive_result``) routes a non-closed drive to
# ELIMINATED / TERMINAL / PENDING exactly as before. The opaque disposition is a
# MERGE POINT: F0 (opaque staging) and TC2 (symbol off the output path) both land
# there, but their root cause / fix / re-run pre-condition differ. ``block_kind``
# is a SEPARATE, orthogonal field that names *why* the output collapsed, computed
# only for the opaque disposition (every other disposition is already specific).
# Pure / deterministic decision tree (invariant 8); judged on already-computed
# structured signals (no case data — invariant 2/6). Ordering = mutually exclusive
# priority. Nothing here feeds close / parity / G4 / seed (invariant 7).
# --------------------------------------------------------------------------- #

BLOCK_OPAQUE_STAGING            = "opaque_staging"
BLOCK_WINDOW_BOUNDARY_MISMATCH  = "window_boundary_mismatch"
BLOCK_SYMBOL_NOT_ON_OUTPUT_PATH = "symbol_not_on_output_path"
BLOCK_EMIT_PICKED_CONSTANT      = "emit_picked_constant"
BLOCK_UNDETERMINED_CONSTANT     = "undetermined_constant"


def _terminal_coverage(terminal_kind: str) -> dict[str, Any]:
    """§需求2 — the capability-coverage stamp for a terminal that carries a
    block_kind: ``{coverage_ok, capabilities, [coverage_warn]}``.

    Lets the agent see, ON the terminal evidence, whether THIS build provides the
    capabilities the terminal's diagnosis relies on (a stale build that "saw
    opaque" for a pre-feature reason is WARNed, not silently trusted). Pure;
    nothing here feeds a gate (invariant 7)."""
    coverage_ok, _missing, warn = coverage_for_terminal(terminal_kind)
    out: dict[str, Any] = {
        "coverage_ok": coverage_ok,
        "capabilities": sorted(collect_build_capabilities()),
    }
    if warn:
        out["coverage_warn"] = warn
    return out


def _seed_mem_info(result: DriveResult) -> dict[str, Any]:
    """The seed step's ``mem_live_in`` info (symbolized / unpinned / n_window_items),
    or ``{}`` when the window seeded no external memory live-in. Pure read."""
    for step in result.per_step:
        if step.get("step") == "seed_entry_state":
            return dict(step.get("mem_live_in") or {})
    return {}


def _n_window_items(result: DriveResult) -> int | None:
    """The window's matched item count (from the seed mem info), or ``None`` when
    not recorded (no external mem live-in step ran)."""
    mi = _seed_mem_info(result)
    n = mi.get("n_window_items")
    return int(n) if n is not None else None


def _symbolized_addrs(result: DriveResult) -> list[int]:
    """External memory addresses this run symbolized (from the seed step)."""
    return [int(a) for a in (_seed_mem_info(result).get("symbolized") or [])]


def _symbol_byte_src_idxs(
    items: Sequence[Instruction], addrs: Sequence[int],
    window: tuple[int, int], window_is_idx: bool,
) -> list[int]:
    """The in-window trace idxs of the loads that read a symbolized address.

    One or more reads can cover the same address; all in-window reading idxs are
    returned (sorted). Pure / deterministic — aligns by ``MemOp.addr`` exactly
    like the disposition recommender."""
    lo, hi = int(window[0]), int(window[1])
    if lo > hi:
        lo, hi = hi, lo
    key = (lambda ins: ins.idx) if window_is_idx else (lambda ins: ins.pc)
    want = {int(a) for a in addrs}
    out: list[int] = []
    for ins in items:
        if not (lo <= key(ins) <= hi):
            continue
        for op in ins.mem:
            if op.rw == "r" and any(op.addr <= a < op.addr + op.size for a in want):
                out.append(ins.idx)
                break
    return sorted(set(out))


def _f_references_inputs(emitted_F: str | None, inputs: Sequence[str]) -> bool:
    """Does the emitted F's BODY reference any declared input name?

    A constant-collapsed F (``def f(carrier):\\n    return 7``) names the input ONLY
    in its signature — the body uses none of them; a real recovered F
    (``... carrier ^ 0x5a ...``) references it in the body. The signature
    parameter list is stripped before searching so the declaration itself is not
    mistaken for a use. Pure / textual — emit renders the symex expr verbatim, so a
    symbol that survived to emit shows up as its input name in the body, one that
    collapsed does not."""
    if not emitted_F:
        return False
    import re
    # Drop the FIRST ``def ...(...):`` header (the parameter declaration) so the
    # input names in the signature are not counted as body uses.
    body = re.sub(r"def\s+\w+\s*\([^)]*\)\s*:", "", str(emitted_F), count=1)
    for name in inputs:
        if re.search(rf"\b{re.escape(str(name))}\b", body):
            return True
    return False


def _symex_emitted(result: DriveResult) -> bool:
    """Did the symex step run and the runner return a non-empty expr (→ emitted_F
    rendered)? ``emitted_F is not None`` is exactly that (emit only sets it when
    ``expr_source`` was non-empty)."""
    return result.emitted_F is not None


def _dfg_symbol_trace(
    items: Sequence[Instruction],
    seed_idxs: Sequence[int],
    window: tuple[int, int],
    window_is_idx: bool,
) -> dict[str, Any]:
    """§需求3 — backtrace a symbol's values through ``build_dfg`` to localize where
    the symbolic chain ends relative to the window's output (the sink/exit).

    Deterministic, zero LLM (invariant 8). Builds the window's concrete DFG, marks
    every node DATA-reachable from a seed load (forward over reg + mem edges), then:
      * ``last_seen_idx``       — the LAST window idx still carrying the symbol;
      * ``entered_emit_dfg``    — whether the window's EXIT producer (the last node)
        is in the symbol's forward cone (the symbol reaches the output computation);
      * ``constantized_at_pc``  — the PC of the last symbol-carrying node when the
        symbol did NOT reach the exit (where the chain breaks), else ``None``.

    Returns ``{}`` when there is no seed idx in the window (nothing to trace)."""
    from ..stages.s3_triton import build_dfg

    lo, hi = int(window[0]), int(window[1])
    if lo > hi:
        lo, hi = hi, lo
    key = (lambda ins: ins.idx) if window_is_idx else (lambda ins: ins.pc)
    win_items = [ins for ins in items if lo <= key(ins) <= hi]
    if not win_items:
        return {}
    seeds = {int(i) for i in seed_idxs if any(ins.idx == i for ins in win_items)}
    if not seeds:
        return {}
    nodes = build_dfg(win_items)
    idx_of_node = {n.idx: n for n in nodes}
    order = [n.idx for n in nodes]                      # execution order

    # Forward propagation: a node "carries the symbol" if it IS a seed, or it
    # depends (reg or mem edge) on a node that carries it. One pass in order is
    # sufficient because every dep idx precedes the consumer (build_dfg links to
    # PRIOR writers only).
    carries: set[int] = set(seeds)
    for n in nodes:
        if n.idx in carries:
            continue
        deps = [d for d in n.reg_deps.values() if d is not None]
        deps += list(n.mem_deps)
        if any(d in carries for d in deps):
            carries.add(n.idx)

    if not carries:                                     # defensive (seeds ⊆ carries)
        return {}
    last_seen = max(carries)
    exit_idx = order[-1]                                # the window's last node = exit
    entered = exit_idx in carries
    constantized_pc: str | None = None
    if not entered:
        # the chain breaks AFTER the last carrying node: report that node's PC as
        # where the symbol was last alive before the output computation dropped it.
        node = idx_of_node.get(last_seen)
        if node is not None:
            constantized_pc = f"0x{node.pc:x}"
    return {
        "last_seen_idx":      last_seen,
        "entered_emit_dfg":   bool(entered),
        "constantized_at_pc": constantized_pc,
    }


def _classify_block_kind(
    result: DriveResult,
    *,
    items: Sequence[Instruction],
    window: tuple[int, int],
    window_is_idx: bool,
    inputs: Sequence[str],
    staging_diag: Any | None,
) -> tuple[str, dict[str, Any]]:
    """Split the opaque MERGE POINT into one of 5 mutually exclusive block kinds.

    Returns ``(block_kind, detail)``. ``detail`` carries the deciding signals (and,
    for the degenerate tail, ``excluded`` — which classes were ruled out and why,
    so the agent still gets a verdict, invariant 4/A8④). Ordering below IS the
    priority; the first matching branch wins (mutual exclusivity by construction).
    All signals are already-computed structured facts (invariant 2/6/8)."""
    excluded: list[str] = []
    n_items = _n_window_items(result)
    symbolized = _symbolized_addrs(result)
    fwd = int(result.symbolic_forwards or 0)
    ea_symbolic = bool(
        staging_diag is not None
        and getattr(staging_diag, "verdict", None) == VERDICT_SYMBOLIC_ADDRESS)

    # 1 — opaque_staging: the symbol enters via an opaque (pointer-indirect) staging
    #     buffer (symbolic EA) and forwarding did NOT rescue it (forwarded nothing,
    #     incl. the fallback re-run which would have raised symbolic_forwards).
    if ea_symbolic and fwd == 0:
        return BLOCK_OPAQUE_STAGING, {
            "block_kind": BLOCK_OPAQUE_STAGING,
            "ea_symbolic": True, "symbolic_forwards": fwd,
            "reason": ("EA backtraces to a symbolic / pointer-indirect root and "
                       "forwarding rescued nothing — the symbol enters through "
                       "opaque staging (route: opaque-staging frontier)."),
        }
    if ea_symbolic:
        excluded.append("opaque_staging: EA symbolic but forwarding fired "
                        f"(symbolic_forwards={fwd}) — not an un-rescued staging miss")
    else:
        excluded.append("opaque_staging: EA is not symbolic (no pointer-indirect "
                        "staging root)")

    # 2 — window_boundary_mismatch: the window itself is mis-bounded — it matched 0
    #     items, OR a symbolized byte's load idx/pc falls OUTSIDE the window.
    if n_items == 0:
        return BLOCK_WINDOW_BOUNDARY_MISMATCH, {
            "block_kind": BLOCK_WINDOW_BOUNDARY_MISMATCH,
            "n_window_items": 0,
            "reason": ("the window matched 0 trace items — an out-of-range / "
                       "mis-specified window boundary; re-check the bounds."),
        }
    if symbolized:
        in_window = _symbol_byte_src_idxs(items, symbolized, window, window_is_idx)
        out_of_window = [a for a in symbolized
                         if not _symbol_byte_src_idxs(
                             items, [a], window, window_is_idx)]
        if out_of_window and not in_window:
            return BLOCK_WINDOW_BOUNDARY_MISMATCH, {
                "block_kind": BLOCK_WINDOW_BOUNDARY_MISMATCH,
                "symbolized_outside_window": [f"0x{a:x}" for a in out_of_window],
                "reason": ("symbolized byte(s) are loaded outside the window band — "
                           "the window/boundary was taken wrong; re-pick the bounds."),
            }
    excluded.append("window_boundary_mismatch: window matched "
                    f"{n_items if n_items is not None else 'some'} item(s) and "
                    "symbolized bytes (if any) fall inside it")

    # 3 — symbol_not_on_output_path: a symbol WAS symbolized, EA is concrete, the
    #     byte is in-window, but the symbol value never reaches the output (the
    #     emitted_F) computation DFG (TC2). Carries the §需求3 symbol_trace evidence.
    if symbolized and not ea_symbolic:
        seed_idxs = _symbol_byte_src_idxs(items, symbolized, window, window_is_idx)
        if seed_idxs:
            trace = _dfg_symbol_trace(items, seed_idxs, window, window_is_idx)
            if trace and not trace.get("entered_emit_dfg", False):
                return BLOCK_SYMBOL_NOT_ON_OUTPUT_PATH, {
                    "block_kind": BLOCK_SYMBOL_NOT_ON_OUTPUT_PATH,
                    "symbol_trace": trace,
                    "reason": ("the input was symbolized in-window with a concrete "
                               "EA, but its value never reaches the output (emitted_F) "
                               "data-flow — the symbol is off the output path."),
                }
            if trace and trace.get("entered_emit_dfg"):
                excluded.append("symbol_not_on_output_path: the symbol DOES reach "
                                "the output DFG (entered_emit_dfg=true)")
            else:
                excluded.append("symbol_not_on_output_path: could not trace the "
                                "symbol's data-flow to the output")
        else:
            excluded.append("symbol_not_on_output_path: no in-window load of a "
                            "symbolized byte to trace")
    else:
        excluded.append("symbol_not_on_output_path: nothing was symbolized (or EA "
                        "is symbolic → opaque-staging family)")

    # 4 — emit_picked_constant: symex emitted a real F that nonetheless does not
    #     reference any input (it collapsed to a constant during emit/propagation).
    if _symex_emitted(result) and not _f_references_inputs(result.emitted_F, inputs):
        return BLOCK_EMIT_PICKED_CONSTANT, {
            "block_kind": BLOCK_EMIT_PICKED_CONSTANT,
            "emitted_F": result.emitted_F,
            "reason": ("symex emitted an F but it references no input — the symbolic "
                       "expression collapsed to a constant at emit time."),
        }
    if _symex_emitted(result):
        excluded.append("emit_picked_constant: the emitted F DOES reference an "
                        "input (not a constant collapse)")
    else:
        excluded.append("emit_picked_constant: nothing was emitted (symex produced "
                        "no expression)")

    # 5 — undetermined_constant: degenerate tail. Still a verdict (A8④): attach the
    #     excluded set so the agent sees exactly which classes were ruled out + why.
    return BLOCK_UNDETERMINED_CONSTANT, {
        "block_kind": BLOCK_UNDETERMINED_CONSTANT,
        "excluded": excluded,
        "reason": ("the output collapsed to a constant but none of the specific "
                   "block kinds fit — see ``excluded`` for what was ruled out."),
    }


# --------------------------------------------------------------------------- #
# Closure-evidence layering + trap state (dev-closure-evidence-layering-trap-state
# -spec, tasks 1/2). A window's drive() ``closed`` means PARITY EXACT for THAT
# window — it is NOT whole-case oracle closure (output sink confirmed + provenance
# closed). The safety gate stamps every CVD window result with the three-layer
# closure classification so a window-local constant / a window parity-EXACT is never
# silently presented as an ALGORITHM closure. Pure read of already-computed signals
# (A8①): the candidate payload (output-provenance anchor: on_path / provenance_
# verdict / sink_captured / source coords) + the drive result (emitted_F, parity).
# --------------------------------------------------------------------------- #


def _constant_source_from_payload(
    payload: Mapping[str, Any], window, window_kind: str,
) -> dict[str, Any]:
    """The provenance coordinate of a (possibly constant) window result — task 2's
    mandatory ``{window, idx_range, reg}``. "实有什么报什么" (A7): whatever the
    candidate payload carries (window band / source / divergence / producer regs)."""
    src: dict[str, Any] = {
        "window": list(window),
        "window_kind": window_kind,
        "idx_range": list(window),
    }
    p = payload or {}
    for key in ("source", "type_id", "divergence_idx", "path_distance",
                "provenance_verdict"):
        if p.get(key) is not None:
            src[key] = p[key]
    regs = p.get("reg_live_in") or p.get("varying_regs") or p.get("producer_pcs")
    if regs:
        src["reg"] = list(regs) if isinstance(regs, (list, tuple)) else regs
    return src


def _window_closure(
    result: DriveResult, c: Candidate, cc: CaseConfig,
) -> dict[str, Any]:
    """Classify a window drive result through the three-layer closure model.

    ``output_sink_confirmed`` / ``provenance_closed`` come from the candidate's
    output-provenance anchor signals (only an on-path / provenance-anchored window
    carries them); ``parity_exact`` from the drive's parity report. A window with no
    provenance anchor at all (no target output was supplied — invariant 7) carries
    sink/provenance False → it can be at most structural/local, never auto-promoted
    to algorithm closure. ``is_constant`` = the emitted F references no input."""
    p = c.payload or {}
    prov_verdict = p.get("provenance_verdict")
    on_path = p.get("on_path")
    sink_captured = p.get("sink_captured")
    # The window is on the confirmed output path only when the generator anchored it
    # there (on_path True) AND provenance is a closing verdict; an absent anchor
    # (today's coverage/variance generation) yields False — local at most.
    output_sink_confirmed = bool(on_path is True and sink_captured is True)
    provenance_closed = bool(
        on_path is True
        and provenance_closed_from_verdict(prov_verdict, sink_captured=sink_captured))
    parity_exact = parity_exact_from_report(result.parity_report) or bool(result.closed)
    structural_closed = result.emitted_F is not None or bool(result.closed)
    is_constant = (
        result.emitted_F is not None
        and not _f_references_inputs(result.emitted_F, cc.inputs))
    cls = classify_closure(
        structural_closed=structural_closed,
        output_sink_confirmed=output_sink_confirmed,
        provenance_closed=provenance_closed,
        parity_exact=parity_exact,
        is_constant=is_constant,
        provenance_supported=provenance_closed,
        constant_source=(
            _constant_source_from_payload(p, cc.window, cc.window_kind)
            if is_constant else None),
    )
    return cls.to_dict()


