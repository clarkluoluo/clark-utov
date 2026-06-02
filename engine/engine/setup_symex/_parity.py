"""setup_symex.parity section (split from the monolithic module)."""
from __future__ import annotations


import enum
import os
import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..dataflow import classify_semop
from ..types import Instruction, MemSnapshot
from ..watch_first_write import (
    WatchFirstWriteConfig,
    WatchFirstWriteSpec,
    request_watch_first_write,
)


@dataclass(frozen=True, slots=True)
class ParityVector:
    """One cross-run input→output check of an emitted handler/window transform.

    ``observed`` is what THIS run's oracle produced for ``input_key``; ``predicted``
    is the emitted transform applied to the same input. Both must come from the
    SAME execution (``exec_id``) — comparing a transform's output against another
    run's observed output is the cross-run mixing the determinism guard forbids.
    ``derived_from`` marks the trace the transform was recovered from; such a
    vector is tautological and never counts toward the independent floor."""

    input_key:    str
    observed:     str
    predicted:    str
    exec_id:      str | None = None
    derived_from: bool = False

    @property
    def matches(self) -> bool:
        return self.observed == self.predicted

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_key":    self.input_key,
            "observed":     self.observed,
            "predicted":    self.predicted,
            "exec_id":      self.exec_id,
            "derived_from": self.derived_from,
            "matches":      self.matches,
        }


@dataclass(frozen=True, slots=True)
class ParityVectorReport:
    """Whether an emitted transform is EXACT under multi-vector cross-run parity.

    EXACT requires at least ``min_vectors`` INDEPENDENT vectors (distinct inputs,
    none the deriving trace) that match their own-execution observed output, with
    no cross-run mixing detected. Anything less is BLOCK — the transform may be
    wrong/incomplete (the case's handler10) and must not be stamped exact off a
    tautological 1/1."""

    window_pcs:       tuple[int, int]
    min_vectors:      int
    independent_pass: int                  # distinct, non-derivation, matching
    counted:          int                  # distinct, non-derivation vectors examined
    total:            int                  # all vectors supplied
    mismatches:       tuple[str, ...]      # input_keys whose predicted != observed
    determinism_ok:   bool                 # no cross-run mixing detected
    determinism_seen: bool                 # exec_ids present to verify determinism
    verdict:          str                  # "EXACT" | "BLOCK" | "UNCLOSABLE"
    reasons:          tuple[str, ...]
    observed_distinct: int = 0             # distinct observed values across evidenced (independent-side) vectors
    vectors:          tuple[dict[str, Any], ...] = ()   # per-vector observed/predicted/matches detail
    next_actions:     tuple[dict[str, Any], ...] = ()   # additive/advisory: suggested existing helper(s) for this verdict (spec A)

    @property
    def sufficient(self) -> bool:
        return self.verdict == "EXACT"

    def to_dict(self) -> dict[str, Any]:
        # Invariant 4: cap the per-vector detail at a small sample + count, never a
        # full dump (the cohort can be wide). The cap is on what crosses the layer.
        _SAMPLE = 8
        vecs = list(self.vectors)
        vector_detail: Any = [dict(v) for v in vecs[:_SAMPLE]]
        if len(vecs) > _SAMPLE:
            vector_detail = {
                "_trimmed_list": True,
                "count": len(vecs),
                "sample": vector_detail,
            }
        out: dict[str, Any] = {
            "window_pcs":       [f"0x{self.window_pcs[0]:x}", f"0x{self.window_pcs[1]:x}"],
            "min_vectors":      self.min_vectors,
            "independent_pass": self.independent_pass,
            "counted":          self.counted,
            "total":            self.total,
            "mismatches":       list(self.mismatches),
            "determinism_ok":   self.determinism_ok,
            "determinism_seen": self.determinism_seen,
            "verdict":          self.verdict,
            "reasons":          list(self.reasons),
            "observed_distinct": self.observed_distinct,
            # The independent-side (main/deriving trace excluded) distinct observed
            # count, named explicitly — this is the closability axis: a cohort whose
            # independent side collapses to < min_vectors distinct outputs cannot be
            # EXACT-closed by ANY F (UNCLOSABLE). Equal to observed_distinct (which is
            # already computed over the evidenced/independent side); surfaced as a
            # first-class key so a consumer reads "closable?" without inferring it.
            "independent_observed_distinct": self.observed_distinct,
            "vectors":          vector_detail,
            "kind":             "setup_symex_parity_vectors",
        }
        # Additive + advisory (spec A): surface suggested existing helper(s) ONLY when
        # non-empty, so a report with no mapped next-action serializes byte-for-byte
        # as before (regression: verdict values/reasons unchanged).
        if self.next_actions:
            out["next_actions"] = [dict(a) for a in self.next_actions]
        return out

    @property
    def advisory(self) -> str:
        if self.sufficient:
            return (f"parity EXACT: {self.independent_pass} independent cross-run "
                    f"vector(s) matched (>= {self.min_vectors}) — the transform holds "
                    f"beyond the trace it was derived from")
        if self.verdict == "UNCLOSABLE":
            return (
                f"parity UNCLOSABLE: the INDEPENDENT side (main/deriving trace "
                f"excluded) observed collapses to {self.observed_distinct} distinct "
                f"value(s) < min_vectors={self.min_vectors} — NO F can EXACT-close "
                f"this window: with fewer than {self.min_vectors} distinct observed "
                f"outputs the cohort cannot independently confirm an input-dependent "
                f"transform (a constant / near-constant gold trivially matches a "
                f"constant predicted, whatever F is). This is a COHORT problem, not "
                f"an F problem: fix the cohort (supply output-diverse seeds so the "
                f"independent side carries >= {self.min_vectors} distinct outputs), "
                f"do NOT keep tuning F. Judged independently of whether predicted "
                f"matched. If the cohort IS diverse upstream, the window/sink is "
                f"mis-located (reading a constant) or it is a trivial pass-through.")
        mm = f"; mismatches on {list(self.mismatches)}" if self.mismatches else ""
        mix = ("" if self.determinism_ok else "; cross-run mixing detected (an observed "
               "output was reused across distinct inputs — never mix executions)")
        return (
            f"parity BLOCK: only {self.independent_pass}/{self.min_vectors} independent "
            f"cross-run vector(s) passed{mm}{mix} — do NOT stamp exact; supply "
            f">= {self.min_vectors} independent cross-run vectors, each checked against "
            f"its own execution's observed output"
        )


def check_parity_vectors(
    vectors: Iterable["ParityVector"],
    *,
    window: tuple[int, int],
    min_vectors: int,
    trace_exec_id: str | None = None,
) -> ParityVectorReport:
    """Decide EXACT / UNCLOSABLE / BLOCK for a handler/window transform.

    A vector COUNTS toward the independent floor only if it is not the deriving
    trace (``derived_from``, or sharing ``trace_exec_id``) and its ``input_key``
    is distinct from earlier counted vectors — a repeated input cannot stand in
    for an independent one.

    **Closability is judged FIRST, independently of whether predicted matched.**
    The INDEPENDENT side (main/deriving trace already excluded by the ``counted``
    filter) must carry at least ``min_vectors`` distinct *observed* outputs. If it
    collapses to ``< min_vectors`` distinct, the verdict is UNCLOSABLE: no F can
    EXACT-close this window, because fewer than ``min_vectors`` distinct outputs
    cannot independently confirm an input-dependent transform (a constant / near-
    constant gold trivially matches a constant predicted, whatever F is). This is
    the OUTPUT-side dual of :func:`check_seed_independence` and is a COHORT defect
    (fix the seeds — supply output-diverse vectors), NOT an F defect — so it is
    reported BEFORE and regardless of the match floor, to tell the consumer to fix
    the cohort, not keep tuning F. (F0 @1769: the independent side collapsed to 1
    distinct observed while min=3 → UNCLOSABLE even though predicted matched.)

    Only when the independent side is genuinely diverse (>= ``min_vectors``
    distinct observed) does the match floor decide: EXACT requires ``min_vectors``
    distinct counted vectors that all match with no detected cross-run mixing;
    otherwise BLOCK.

    Independence is on the INPUT/seed dimension, not the execution: ``input_key``
    is the per-vector seed-assignment fingerprint, so two runs of the same input
    (different ``exec_id``) collapse to one counted vector — this is the same
    "independent evidence" notion CVD's ``evaluate_credibility`` uses, applied to
    set-up symex. The complementary pre-symex check that the seed actually VARIES
    across the cohort (so F isn't a function of a constant) is
    :func:`check_seed_independence`.

    Determinism: when vectors carry ``exec_id`` we verify no two DISTINCT inputs
    share one execution (which would mean one run's observed output was reused for
    another input — the mixing the same-execution guard forbids)."""
    vecs = list(vectors)
    seen_inputs: set[str] = set()
    counted: list[ParityVector] = []
    for v in vecs:
        derived = v.derived_from or (
            trace_exec_id is not None and v.exec_id == trace_exec_id)
        if derived or v.input_key in seen_inputs:
            continue
        seen_inputs.add(v.input_key)
        counted.append(v)

    mismatches = tuple(v.input_key for v in counted if not v.matches)
    independent_pass = sum(1 for v in counted if v.matches)

    # Determinism: a single exec_id must not back two distinct counted inputs
    # (that would be one run's observed output reused for another input).
    exec_seen = [v.exec_id for v in counted if v.exec_id is not None]
    determinism_seen = bool(exec_seen)
    determinism_ok = len(exec_seen) == len(set(exec_seen))

    # Closability gate — the OUTPUT-side dual of check_seed_independence, judged
    # FIRST and INDEPENDENTLY of whether predicted matched. The INDEPENDENT side
    # (the ``counted`` filter already dropped the deriving trace ``derived_from`` /
    # ``trace_exec_id`` — main/derived excluded, per the held_out/independent-side
    # caliber) must carry >= min_vectors DISTINCT observed outputs. If it collapses
    # to < min_vectors distinct, NO F can EXACT-close: fewer than min_vectors
    # distinct outputs cannot independently confirm an input-dependent transform
    # (a constant / near-constant gold trivially matches a constant predicted,
    # whatever F is). That is a COHORT defect (need output-diverse seeds), not an
    # F defect → UNCLOSABLE, reported BEFORE and regardless of the match floor.
    #
    # The gate may only judge REAL observed evidence: a vector whose ``observed`` is
    # an actual oracle reading carries an ``exec_id`` (the cohort feed-leg and the
    # explicit-vector runner path both stamp one — see _cohort_parity_vectors). The
    # scalar "m/n" gold FALLBACK (_parity_vectors_from_run) carries NO exec_id and a
    # PLACEHOLDER observed ("1"), which encodes the pass count, not a real output —
    # judging its diversity would itself be a fabrication (invariant 8) and would
    # mislabel "no observed data" as "collapsed output". So the gate runs ONLY over
    # counted vectors that carry an exec_id; with none, there is no real observed
    # signal and the gate STANDS DOWN (byte-for-byte the prior EXACT/BLOCK path —
    # invariant 7). observed_distinct is the REAL count over those evidenced
    # (independent-side) vectors (invariant 8 — never fabricated).
    evidenced = [v for v in counted if v.exec_id is not None]
    observed_distinct = len({v.observed for v in evidenced})
    observed_evidence = observed_distinct >= 1     # real observed signal present
    # The threshold is the SAME floor as the match floor: the independent side must
    # be at least as output-diverse as the number of independent vectors we demand.
    # F0 @1769: independent distinct=1 (or =2 under a min=3) → still UNCLOSABLE.
    independent_observed_diverse = observed_distinct >= min_vectors

    reasons: list[str] = []
    # UNCLOSABLE is the FIRST (closability) check — does not depend on the match
    # floor / determinism. It fires whenever there IS real observed evidence and the
    # independent side carries < min_vectors distinct outputs. (Stand-down: with no
    # evidenced observed at all — scalar-gold fallback only — the gate is silent.)
    unclosable = observed_evidence and not independent_observed_diverse
    if unclosable:
        reasons.append(
            f"independent-side observed collapses to {observed_distinct} < "
            f"{min_vectors} distinct — no F can EXACT-close; fix the cohort "
            f"(output-diverse seeds), not F")

    if independent_pass < min_vectors:
        reasons.append(
            f"only {independent_pass} independent cross-run vector(s) matched; need "
            f">= {min_vectors} (a 1/1 ≈ verifying the transform with the trace it was "
            f"derived from — tautological)")
    if mismatches:
        reasons.append(
            f"transform mismatches its own-execution observed output on "
            f"{list(mismatches)} — the recovered transform is wrong or incomplete")
    if not determinism_ok:
        reasons.append(
            "cross-run mixing: one execution's observed output backs two distinct "
            "inputs — compare each vector within its own execution, never merge")

    # Verdict precedence: UNCLOSABLE (cohort not good enough) is decided FIRST and
    # independently of the match floor — even a fully-matching predicted cannot
    # close a collapsed independent side (this is what stops the agent white-tuning
    # F when it should fix the cohort). Only when the independent side is genuinely
    # output-diverse (>= min_vectors distinct) does the match floor + determinism
    # decide EXACT vs BLOCK. (The old DEGENERATE — "floor met but observed constant"
    # — is now a strict subset of UNCLOSABLE: distinct < min implies distinct == 1
    # is no longer carved out separately; >= min distinct cannot be "constant", so
    # the two collapse cleanly into UNCLOSABLE.)
    would_be_exact = independent_pass >= min_vectors and determinism_ok
    if unclosable:
        verdict = "UNCLOSABLE"
    elif would_be_exact:
        verdict = "EXACT"
    else:
        verdict = "BLOCK"
    # Per-vector detail (over ALL supplied vectors, so the agent sees observed vs
    # predicted on every one — including the derived/tautological ones that don't
    # count toward the floor). matches = observed == predicted (invariant 8: the
    # number is the real comparison, not recomputed differently).
    vector_detail = tuple({
        "input_key":    v.input_key,
        "observed":     v.observed,
        "predicted":    v.predicted,
        "matches":      v.matches,
    } for v in vecs)
    # Additive + advisory (spec A): consult the declarative verdict->helper registry
    # at the verdict-construction site. This does NOT touch ``verdict`` or
    # ``reasons`` (byte-for-byte unchanged); it only attaches a pointer to an
    # existing helper when a (kind, verdict, reason) mapping matches — empty
    # otherwise. Imported locally to avoid any import cycle through the package.
    from ..next_actions import suggest_next_actions, PARITY_VECTORS_KIND
    next_actions = suggest_next_actions(PARITY_VECTORS_KIND, verdict, tuple(reasons))
    return ParityVectorReport(
        window_pcs=(int(window[0]), int(window[1])),
        min_vectors=int(min_vectors),
        independent_pass=independent_pass,
        counted=len(counted),
        total=len(vecs),
        mismatches=mismatches,
        determinism_ok=determinism_ok,
        determinism_seen=determinism_seen,
        verdict=verdict,
        reasons=tuple(reasons),
        observed_distinct=observed_distinct,
        vectors=vector_detail,
        next_actions=next_actions,
    )


# ---------------------------------------------------------------------------
# Seed-independence gate — the UPSTREAM (pre-symex) companion to the parity gate.
#
# check_parity_vectors guards AFTER emit (a transform that only matches the trace
# it was derived from). It already counts independence by distinct ``input_key``
# (the per-vector seed-assignment fingerprint — input-dimension, not exec_id), so
# it does NOT degrade "independent" to "same execution". What it cannot see is the
# case where the seed we symbolized never VARIES across the cohort: then the
# recovered F is a function of a constant → it degenerates to a constant and
# trivially matches every vector (a constant gold matches a constant predicted),
# a false EXACT that the post-emit gate cannot tell from a real one. This gate
# catches it BEFORE the symex runs.
#
# The subject is the SEED (the recovery variable we symbolize) — NOT hard-wired
# "input". F may be F(input), F(nonce), F(input, nonce), … so "must vary across
# the cohort" is asked of whatever we symbolized: input vectors must vary the
# input seed, a nonce-only F's cohort must vary the nonce seed. (dev addendum M1.)
# ---------------------------------------------------------------------------


