"""Cohort-diff: localize WHERE the recovery-variable-dependent computation lives.

Given >= 2 aligned cohort traces that differ only in the recovery variable we
want to recover F over (the *seed* — input, nonce, time, …; dev addendum: the
subject is the seed, NOT hard-wired "input"), this answers the purely mechanical
question "which windows/states vary with the seed (= to recover) vs stay constant
(= setup, carry as a constant)". The agent then symbolizes+recovers ONLY the
seed-varying windows and carries the invariant ones as constants — no wasted
symex, no false EXACT on a constant window.

This is the full-trace extension of :func:`engine.setup_symex.check_seed_independence`
(which verifies a SINGLE window): same criterion (does state vary across the
cohort), one localizes, one verifies.

Three things the naive "diff entry/exit registers" misses, and which this handles
(dev verification-localization addendum):

  * MEMORY, not just registers (M-mem / M6): an opaque value enters through a
    memory store (the F0 60-byte staging buffer), invisible to a register-only
    diff. Memory write values are diffed alongside registers.
  * control-flow divergence (M5): different seeds can take different code paths,
    so the traces do NOT align by index. We align by PC; the first PC that
    differs across vectors is itself a strong seed-dependence signal (an input-
    dependent branch), not an alignment failure to crash on.
  * the all-invariant / opaque outcome (M6): there may be NO window whose
    OBSERVABLE state varies with the seed (F0: the seed arrives as opaque
    staging, every handler I/O is constant). We must NOT conclude "no
    dependence"; we return a distinct ``opaque`` verdict — the dependence is
    real but invisible to a state diff, so it needs symex to pierce the staging,
    not localization.

Coupling axes (M4): when the cohort co-varies a second axis (a per-run nonce /
time) alongside the seed, a raw diff attributes that axis's state to the seed.
Pass the registers / addresses driven by the coupling axis as ``ignore_regs`` /
``ignore_addrs`` to control it out; what remains is the seed's own footprint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .types import Instruction

__all__ = [
    "VaryingPosition",
    "InputDependenceMap",
    "localize_input_dependence",
]


@dataclass(frozen=True, slots=True)
class VaryingPosition:
    """One aligned trace position whose state varies across the cohort.

    ``varying_regs`` are the written registers whose value differs across vectors;
    ``varying_mem`` are the addresses whose stored value differs. ``control_flow``
    marks a position where the PC itself diverges (a seed-dependent branch)."""

    idx:          int
    pc:           int
    varying_regs: tuple[str, ...] = ()
    varying_mem:  tuple[int, ...] = ()
    control_flow: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "idx":          self.idx,
            "pc":           f"0x{self.pc:x}",
            "varying_regs": list(self.varying_regs),
            "varying_mem":  [f"0x{a:x}" for a in self.varying_mem],
            "control_flow": self.control_flow,
        }


@dataclass(frozen=True, slots=True)
class InputDependenceMap:
    """Where (if anywhere) observable state varies with the seed across a cohort.

    ``verdict``:
      * ``localized``     — at least one position varies; ``divergence_idx`` is the
                            FIRST one (the seed's real entry point).
      * ``opaque``        — the cohort genuinely differs yet NO observable state
                            varies, AND the diffed dimensions (regs_write / mem-
                            write) were observably populated enough to trust that
                            "no variation" really means hidden: the dependence is
                            real but hidden (opaque staging / obfuscation). Needs
                            symex to pierce it; do NOT read this as "no dependence"
                            (dev addendum M6).
      * ``inconclusive_low_observability`` — we saw no variation, but the diffed
                            dimensions were basically empty in this trace
                            (regs_write/mem-write coverage below threshold) and/or
                            the variation lives on the read side (regs_read varies
                            across vectors while regs_write+mem do not). This is a
                            MEASUREMENT BLIND SPOT, not a true opaque frontier — we
                            were fed the wrong dimension / incomplete data. Honest
                            non-result; do NOT send it to the opaque-staging symex
                            investment (dev opaque trust-gate).
      * ``insufficient``  — fewer than 2 vectors, or the cohort did not vary the
                            seed (every supplied input_key identical) — can't tell.
    """

    n_vectors:        int
    alignment:        str                  # "by_pc" | "insufficient"
    verdict:          str                  # localized|opaque|inconclusive_low_observability|insufficient
    divergence_idx:   int | None           # first seed-varying position
    aligned_len:      int                  # positions compared before any divergence stop
    varying:          tuple[VaryingPosition, ...] = ()
    ignored_regs:     tuple[str, ...] = ()
    ignored_addrs:    tuple[int, ...] = ()
    reasons:          tuple[str, ...] = ()
    # opaque trust-gate observability evidence (gap-map visible):
    observable_positions: int = 0          # aligned positions with non-empty regs_write or mem
    observability_rate:   float = 0.0      # observable_positions / aligned_len
    regs_read_varies:     bool = False     # did any read value differ across vectors
    # Phase 3 — localize-side opaque advisory (the EA-varying staging PCs). Only
    # populated on an ``opaque`` verdict; ``None`` on every other path (default
    # keeps non-opaque to_dict byte-for-byte unchanged). The dict shape is
    # CohortStagingAdvisory.to_dict() (or None).
    opaque_staging_advisory: dict | None = None

    @property
    def varying_idxs(self) -> tuple[int, ...]:
        return tuple(p.idx for p in self.varying)

    @property
    def is_opaque(self) -> bool:
        return self.verdict == "opaque"

    @property
    def is_low_observability(self) -> bool:
        """Distinct from :attr:`is_opaque`: "no variation seen" but the diffed
        dimensions were too sparse to trust it (measurement blind spot). The
        gap-map must NOT treat this as a true opaque-staging frontier."""
        return self.verdict == "inconclusive_low_observability"

    def window_is_seed_varying(self, lo: int, hi: int, *, by_idx: bool = True) -> bool:
        """Does any seed-varying position fall inside the inclusive band? The band
        is a trace-idx range (``by_idx``) or a PC range. An ``opaque`` map answers
        False everywhere AND is flagged via :attr:`is_opaque` — callers must treat
        opaque as "can't localize", never as "this window is constant/safe"."""
        for p in self.varying:
            key = p.idx if by_idx else p.pc
            if lo <= key <= hi:
                return True
        return False

    # back-compat alias (the localization is over the seed, of which input is one kind)
    window_is_input_varying = window_is_seed_varying

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n_vectors":      self.n_vectors,
            "alignment":      self.alignment,
            "verdict":        self.verdict,
            "divergence_idx": self.divergence_idx,
            "aligned_len":    self.aligned_len,
            "varying":        [p.to_dict() for p in self.varying],
            "ignored_regs":   list(self.ignored_regs),
            "ignored_addrs":  [f"0x{a:x}" for a in self.ignored_addrs],
            "reasons":        list(self.reasons),
            "observable_positions": self.observable_positions,
            "observability_rate":   round(self.observability_rate, 4),
            "regs_read_varies":     self.regs_read_varies,
            "kind":           "cohort_input_dependence_map",
        }
        # Phase 3: only emitted on the opaque path (None elsewhere) so every
        # non-opaque verdict's serialization stays byte-for-byte unchanged.
        if self.opaque_staging_advisory is not None:
            out["opaque_staging_advisory"] = self.opaque_staging_advisory
        return out

    @property
    def advisory(self) -> str:
        if self.verdict == "insufficient":
            return ("cohort cannot localize: " + "; ".join(self.reasons))
        if self.verdict == "inconclusive_low_observability":
            return (
                f"INCONCLUSIVE (low observability): no register/memory variation "
                f"was seen across {self.aligned_len} aligned position(s), but the "
                f"diffed dimensions (regs_write / mem-write) were basically empty "
                f"in this trace (observability_rate={self.observability_rate:.2%}) "
                f"{'and regs_read DID vary across vectors ' if self.regs_read_varies else ''}"
                f"— this is a MEASUREMENT BLIND SPOT (wrong dimension / incomplete "
                f"data), NOT a true opaque frontier. Feed traces with populated "
                f"regs_write / merged memory events, or diff regs_read. Do NOT send "
                f"this to opaque-staging symex.")
        if self.verdict == "opaque":
            return (
                f"OPAQUE: the cohort's seeds genuinely differ yet NO observable "
                f"register/memory state varies across {self.aligned_len} aligned "
                f"position(s). The seed dependence is real but hidden (opaque "
                f"staging / obfuscation) — a state diff cannot localize it; symex "
                f"must pierce the staging. Do NOT conclude 'no seed dependence'.")
        return (
            f"localized: {len(self.varying)} seed-varying position(s); the seed "
            f"first reaches observable state at idx {self.divergence_idx}. Recover F "
            f"only on the seed-varying windows; carry the invariant rest as constants.")


def _mem_writes(ins: Instruction) -> dict[int, int]:
    """Address -> stored value for this step's memory WRITES (canonical MemOp)."""
    out: dict[int, int] = {}
    for op in ins.mem:
        if op.rw == "w":
            out[op.addr] = op.val
    return out


# Diffed-dimension presence predicate is the SINGLE shared kernel in
# engine.trace_observability (item ③ A8 unification): cohort_diff no longer keeps
# its own definition — it imports the same predicate opaque_staging's coverage
# also draws from, so there is one source of "does this step carry the diffed
# dimension". The per-aligned-position observability_rate is still computed here
# over cohort_diff's OWN aligned region (it stops at control-flow divergence), but
# the predicate is unified — keeping the verdict byte-for-byte while removing the
# parallel definition.
from .trace_observability import has_write_dim as _has_observable_diff_dim


def localize_input_dependence(
    cohort_traces: Sequence[Sequence[Instruction]],
    *,
    input_keys: Sequence[str] | None = None,
    ignore_regs: Sequence[str] = (),
    ignore_addrs: Sequence[int] = (),
    min_observability: float = 0.05,
) -> InputDependenceMap:
    """Localize the seed-dependent computation across an aligned cohort.

    ``cohort_traces`` are >= 2 traces of the SAME code path under different seed
    values. Positions are aligned by PC: as long as every vector shares the PC at
    a position, its written-register values and memory-write values are diffed
    across vectors; a position varies when any survives the ``ignore_*`` filter
    and differs. The first PC that disagrees across vectors is a control-flow
    divergence (a seed-dependent branch) — recorded as a varying position and the
    scan stops there (the paths no longer align).

    ``ignore_regs`` / ``ignore_addrs`` control out a coupling axis (a per-run
    nonce/time): state they drive is excluded so what remains is the seed's own
    footprint (dev addendum M4).

    Opaque trust-gate (dev opaque trust-gate): before concluding ``opaque`` from
    "no observable variation", we self-check that the diffed dimensions were
    actually worth trusting. If the aligned window's fraction of positions with a
    non-empty ``regs_write`` or memory write is below ``min_observability``, OR if
    ``regs_read`` values DID vary across vectors while regs_write+mem did not, then
    "no variation" is a MEASUREMENT BLIND SPOT (we were fed the wrong dimension /
    incomplete data), not a true opaque frontier — we return
    ``inconclusive_low_observability`` instead of ``opaque``. This keeps the
    gap-map from mistaking a sparse trace for a real staging frontier. Cohorts
    whose diffed dimensions are well populated take the original path unchanged.

    Returns an :class:`InputDependenceMap`. A cohort that differs yet shows no
    varying observable state — with trustworthy observability — is reported
    ``opaque`` (M6); never silently "no dependence"."""
    traces = [list(t) for t in cohort_traces]
    ig_regs = set(ignore_regs)
    ig_addrs = set(ignore_addrs)
    reasons: list[str] = []

    if len(traces) < 2:
        reasons.append(f"need >= 2 cohort traces, got {len(traces)}")
        return InputDependenceMap(
            n_vectors=len(traces), alignment="insufficient", verdict="insufficient",
            divergence_idx=None, aligned_len=0, reasons=tuple(reasons),
            ignored_regs=tuple(sorted(ig_regs)), ignored_addrs=tuple(sorted(ig_addrs)))

    # A cohort that did not actually vary the seed cannot localize anything — the
    # CVD/seed-independence axis (every supplied input_key identical) is "no seed
    # axis", distinct from "varied but opaque".
    cohort_varied_seed = input_keys is None or len(set(input_keys)) >= 2
    if input_keys is not None and not cohort_varied_seed:
        reasons.append(
            "all cohort input_keys identical — the cohort did not vary the seed; "
            "supply vectors with genuinely different seed values")
        return InputDependenceMap(
            n_vectors=len(traces), alignment="insufficient", verdict="insufficient",
            divergence_idx=None, aligned_len=0, reasons=tuple(reasons),
            ignored_regs=tuple(sorted(ig_regs)), ignored_addrs=tuple(sorted(ig_addrs)))

    n = min(len(t) for t in traces)
    varying: list[VaryingPosition] = []
    divergence_idx: int | None = None
    aligned_len = 0
    observable_positions = 0     # positions where the diff dimensions had anything
    regs_read_varies = False     # input-side (read) value variation across vectors

    for i in range(n):
        steps = [t[i] for t in traces]
        pcs = {s.pc for s in steps}
        if len(pcs) > 1:
            # Control-flow divergence — a seed-dependent branch. Record it and stop:
            # beyond here the traces are different paths and no longer align by index.
            ref = steps[0]
            pos = VaryingPosition(idx=ref.idx, pc=ref.pc, control_flow=True)
            varying.append(pos)
            if divergence_idx is None:
                divergence_idx = ref.idx
            reasons.append(
                f"control-flow divergence at idx {ref.idx}: PCs differ across vectors "
                f"(a seed-dependent branch) — alignment stops here")
            break
        aligned_len += 1
        ref = steps[0]
        # Observability of the diffed dimensions at this position (trust-gate).
        if any(_has_observable_diff_dim(s) for s in steps):
            observable_positions += 1
        # Input-side variation: did any commonly-read register's value differ
        # across vectors? (regs_write+mem are the diffed dims; regs_read is the
        # input side the diff has never looked at — strong "wrong dimension" signal
        # when it moves but the written dims do not.)
        if not regs_read_varies:
            common_reads = set(steps[0].regs_read)
            for s in steps[1:]:
                common_reads &= set(s.regs_read)
            for r in common_reads:
                if r in ig_regs:
                    continue
                if len({s.regs_read[r] for s in steps}) > 1:
                    regs_read_varies = True
                    break
        # Register write values across vectors (only regs every vector wrote).
        v_regs: list[str] = []
        common_regs = set(steps[0].regs_write)
        for s in steps[1:]:
            common_regs &= set(s.regs_write)
        for r in common_regs:
            if r in ig_regs:
                continue
            if len({s.regs_write[r] for s in steps}) > 1:
                v_regs.append(r)
        # Memory write values across vectors (addresses every vector wrote).
        mem_maps = [_mem_writes(s) for s in steps]
        v_mem: list[int] = []
        common_addrs = set(mem_maps[0])
        for m in mem_maps[1:]:
            common_addrs &= set(m)
        for a in common_addrs:
            if a in ig_addrs:
                continue
            if len({m[a] for m in mem_maps}) > 1:
                v_mem.append(a)
        if v_regs or v_mem:
            varying.append(VaryingPosition(
                idx=ref.idx, pc=ref.pc,
                varying_regs=tuple(sorted(v_regs)),
                varying_mem=tuple(sorted(v_mem))))
            if divergence_idx is None:
                divergence_idx = ref.idx

    observability_rate = (observable_positions / aligned_len) if aligned_len else 0.0

    if varying:
        verdict = "localized"
    else:
        # No observable register/memory variation. Before calling this a true
        # opaque frontier, run the trust-gate: was the diffed dimension worth
        # trusting? Low coverage of regs_write/mem-write, or input-side (regs_read)
        # variation while the written dims stayed flat, means we were fed the wrong
        # dimension / incomplete data — a MEASUREMENT BLIND SPOT, not real opacity.
        low_coverage = observability_rate < min_observability
        if low_coverage or regs_read_varies:
            verdict = "inconclusive_low_observability"
            if low_coverage:
                reasons.append(
                    f"opaque trust-gate: only {observable_positions}/{aligned_len} "
                    f"aligned position(s) had any non-empty regs_write / mem-write "
                    f"(observability_rate={observability_rate:.2%} < "
                    f"{min_observability:.2%}) — the diffed dimensions are basically "
                    f"empty in this trace; 'no variation' is a measurement blind spot, "
                    f"not true opacity. Feed populated regs_write / merged memory "
                    f"events, or diff regs_read")
            if regs_read_varies:
                reasons.append(
                    "opaque trust-gate: regs_read values DID vary across vectors while "
                    "regs_write+mem did not — the seed difference lives on the read "
                    "side / a dimension this diff does not cover; not true opacity")
        else:
            # The cohort varied the seed (checked above) yet NO observable state
            # moved AND the diffed dimensions were well populated: opaque (the seed
            # enters through staging the diff can't see). Real dependence, just
            # hidden — needs symex, NOT a "no dependence" verdict.
            verdict = "opaque"
            reasons.append(
                f"cohort varied the seed but no register/memory state varied across "
                f"{aligned_len} aligned position(s) "
                f"(observability_rate={observability_rate:.2%}, regs_read flat) "
                f"— opaque staging; localize via symex")

    # Phase 3 — localize-side opaque advisory. ONLY on a true ``opaque`` verdict
    # (localized/inconclusive_low_observability/insufficient stay advisory-free,
    # so their serialization is unchanged). The value diff is empty under opaque;
    # the non-redundant signal is per-PC EA variance — surface those staging PCs
    # so symex knows where to pierce. import is local to keep the module-load
    # dependency direction cohort_diff → opaque_staging (verified acyclic:
    # opaque_staging imports only setup_symex / types).
    opaque_advisory: dict | None = None
    if verdict == "opaque":
        from .opaque_staging import cohort_staging_advisory
        # The aligned region is the first ``aligned_len`` positions cohort_diff
        # compared by list-position; express it as the reference trace's inclusive
        # idx band (robust to traces that do not start at idx 0).
        if aligned_len > 0:
            region = (traces[0][0].idx, traces[0][aligned_len - 1].idx)
        else:
            region = (0, 0)
        adv = cohort_staging_advisory(
            traces, region=region, window_is_idx=True,
            ignore_addrs=tuple(sorted(ig_addrs)))
        opaque_advisory = adv.to_dict()

    return InputDependenceMap(
        n_vectors=len(traces),
        alignment="by_pc",
        verdict=verdict,
        divergence_idx=divergence_idx,
        aligned_len=aligned_len,
        varying=tuple(varying),
        ignored_regs=tuple(sorted(ig_regs)),
        ignored_addrs=tuple(sorted(ig_addrs)),
        reasons=tuple(reasons),
        observable_positions=observable_positions,
        observability_rate=observability_rate,
        regs_read_varies=regs_read_varies,
        opaque_staging_advisory=opaque_advisory,
    )
