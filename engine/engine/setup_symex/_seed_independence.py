"""setup_symex.seed_independence section (split from the monolithic module)."""
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
class SeedIndependenceReport:
    """Whether the symbolized seed actually VARIES across the verification cohort.

    A seed (a live-in register or a ``symbolic_mem`` region) that takes the SAME
    entry value in every cohort vector is a constant wearing a symbol's hat —
    symbolizing it makes F a function of a constant, which degenerates and then
    falsely passes parity (every vector's gold is the same constant). EXACT off
    such a cohort is the F0 false-EXACT. When EVERY seed is constant across the
    cohort the window is not driven by the recovery variable at all → BLOCK. When
    only SOME are constant they are surfaced (they belong in concrete backing, not
    symbolized) but the gate passes."""

    n_vectors:            int
    min_vectors:          int
    constant_seeds:       tuple[str, ...]   # same entry value in every vector
    varying_seeds:        tuple[str, ...]   # >= 2 distinct entry values across vectors
    distinct_vector_count: int              # distinct seed-assignment tuples across vectors
    verdict:              str               # "OK" | "BLOCK" | "INSUFFICIENT"
    reasons:              tuple[str, ...]

    @property
    def sufficient(self) -> bool:
        return self.verdict == "OK"

    @property
    def blocked(self) -> bool:
        return self.verdict == "BLOCK"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_vectors":             self.n_vectors,
            "min_vectors":           self.min_vectors,
            "constant_seeds":        list(self.constant_seeds),
            "varying_seeds":         list(self.varying_seeds),
            "distinct_vector_count": self.distinct_vector_count,
            "verdict":               self.verdict,
            "reasons":               list(self.reasons),
            "kind":                  "setup_symex_seed_independence",
        }

    @property
    def advisory(self) -> str:
        if self.verdict == "INSUFFICIENT":
            return (f"seed-independence undecidable: only {self.n_vectors} cohort "
                    f"vector(s) (need >= {self.min_vectors}) — supply more vectors "
                    f"with genuinely different seed values")
        if self.blocked:
            return (
                f"seed BLOCK: every symbolized seed {sorted(self.constant_seeds)} is "
                f"CONSTANT across {self.n_vectors} cohort vector(s) — this window is "
                f"not driven by the recovery variable (a setup segment, or the cohort "
                f"never varied the seed: same input AND same nonce/time). Symbolizing "
                f"a constant degenerates F and falsely passes parity. Pick a window on "
                f"the recovery variable's path, or a cohort that varies the seed.")
        msg = (f"seed varies: {sorted(self.varying_seeds)} change across "
               f"{self.distinct_vector_count} distinct vector(s) — F is exercised")
        if self.constant_seeds:
            msg += (f"; constant seeds {sorted(self.constant_seeds)} should be CONCRETE "
                    f"backing, not symbolized (they never vary → not inputs)")
        return msg


def check_seed_independence(
    seed_values: Mapping[str, Sequence[int]],
    *,
    min_vectors: int = 2,
) -> SeedIndependenceReport:
    """Decide whether the symbolized seed is exercised by the cohort (pre-symex).

    ``seed_values`` maps each seed name (a register, or ``"mem@0x..."`` for a
    symbolic memory region) to its ENTRY value in each cohort vector — the value
    the seed holds when control reaches the window, read from THAT vector's own
    execution (determinism, A6). A seed is *constant* when those values are all
    equal, *varying* when it takes >= 2 distinct values.

    BLOCK iff there are enough vectors AND no seed varies (the whole seed set is
    constant across the cohort → the window is not driven by the recovery
    variable). Otherwise OK, surfacing any constant seeds (which belong in
    concrete backing). With fewer than ``min_vectors`` vectors the answer is
    INSUFFICIENT (undecidable — surfaced, never a silent pass). Zero case-specific
    knowledge: it compares values, never addresses/handler ids."""
    names = list(seed_values)
    seqs = {k: list(v) for k, v in seed_values.items()}
    n_vectors = max((len(v) for v in seqs.values()), default=0)
    constant: list[str] = []
    varying: list[str] = []
    for name in names:
        vals = seqs[name]
        if len(set(vals)) >= 2:
            varying.append(name)
        else:
            constant.append(name)
    # Distinct seed-assignment tuples: how many genuinely-different vectors we have
    # (M3 — a repeated assignment is not extra independent evidence).
    if names and n_vectors:
        tuples = {
            tuple(seqs[name][i] if i < len(seqs[name]) else None for name in names)
            for i in range(n_vectors)
        }
        distinct_vector_count = len(tuples)
    else:
        distinct_vector_count = 0

    reasons: list[str] = []
    if n_vectors < min_vectors:
        verdict = "INSUFFICIENT"
        reasons.append(
            f"only {n_vectors} cohort vector(s); need >= {min_vectors} to decide "
            f"seed independence")
    elif not varying:
        verdict = "BLOCK"
        reasons.append(
            f"every symbolized seed is constant across {n_vectors} vector(s) "
            f"({sorted(constant)}) — the window is not driven by the recovery "
            f"variable; symbolizing a constant degenerates F and falsely passes parity")
    else:
        verdict = "OK"
        if constant:
            reasons.append(
                f"constant seeds {sorted(constant)} never vary across the cohort — "
                f"they are concrete backing, not symbolic inputs")
    return SeedIndependenceReport(
        n_vectors=n_vectors,
        min_vectors=int(min_vectors),
        constant_seeds=tuple(constant),
        varying_seeds=tuple(varying),
        distinct_vector_count=distinct_vector_count,
        verdict=verdict,
        reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# G4 emit self-check — the recovered F must reproduce its OWN trace.
#
# Concolic symex runs ON a trace: every non-symbolic state is, by construction,
# the trace's concrete value (that is what the concolic shadow is for). So the
# recovered F, evaluated on the trace's concrete seed values, MUST equal the
# trace's concrete sink value at the window exit. If it does not, the symex is
# unsound (shadow / reconcile / sink leak) and the emitted F is wrong — it must
# NOT be emitted silently. The real F0 case: handler56 symex computed exit x8=69
# while the trace's own exit was 0xfb9881b1, yet nothing BLOCKED.
#
# This is a NECESSARY (not sufficient) gate, BEFORE the cross-vector parity gate:
# parity proves generality across vectors; this proves the most basic thing —
# that F reproduces the very trace it was derived from. A 1/1 trace match is not
# enough for EXACT (that is parity's job), but FAILING it is a hard BLOCK.
# ---------------------------------------------------------------------------


