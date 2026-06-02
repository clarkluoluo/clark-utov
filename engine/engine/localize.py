"""Differential localisation API (capability_request.md §P1-3).

The reference target hindsight: pinning ``random_007`` to ``SignRunner`` (not the
input) walked three layers of diff —

  1. ``localize_divergence``  — find the first trace step where the
     known-good sample and the failing sample disagree.
  2. ``calltrace alignment``  — align the surviving traces so a step in
     one maps to the equivalent step in the other.
  3. ``byte-graft``           — patch a candidate byte / register from
     the good sample into the bad sample's state at the divergence and
     re-run to confirm the bug *is* that one byte.

Today this lives in ad-hoc script bodies (``clark_r1_digest_hook.py``).
Promote it into a first-class engine API the agent calls once with two
sample groups; the engine returns the divergence point plus a list of
candidate hypotheses ranked by likelihood.

This module is the data plumbing — the actual hypothesis generation is
intentionally heuristic (mnemonic / written-reg / mem-write
fingerprints) so it stays deterministic. Agents that want richer
hypothesis text route through the standard ``submit_hypothesis`` flow.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .types import Instruction


@dataclass(frozen=True, slots=True)
class DivergencePoint:
    """First instruction where the two trace sequences diverge.

    ``good_idx`` / ``bad_idx`` are absolute trace indices; ``pc`` is the
    PC on the good side (which the bad side either reached late, early
    or never).  ``kind`` summarises *what* diverged (one of
    ``pc | regs_write | mem | length``)."""
    kind: str
    good_idx: int
    bad_idx: int
    pc: int
    good: dict[str, Any] = field(default_factory=dict)
    bad: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CandidateHypothesis:
    """One ranked candidate the engine raises after divergence is
    pinned.  Subject + payload follow the standard hyp wire shape so the
    caller can ``submit_hypothesis`` directly."""
    rank: int
    kind: str
    subject: str
    payload: dict[str, Any]
    confidence: float
    rationale: str


@dataclass(frozen=True, slots=True)
class LocalizeResult:
    """Full ``localize_divergence`` output.

    ``divergence`` is None when the two traces agree up to the shorter
    one's length AND have the same length — in that case there is no
    divergence to pin."""
    divergence: DivergencePoint | None
    candidates: tuple[CandidateHypothesis, ...] = ()
    # First aligned pair beyond divergence (idx_good, idx_bad), helps a
    # subsequent byte-graft attempt skip to a clean resumption point.
    resync_at: tuple[int, int] | None = None


def _ins_fingerprint(ins: Instruction) -> tuple[Any, ...]:
    """A position-independent identifier of an instruction. Used for
    calltrace alignment after divergence."""
    return (ins.pc, ins.mnemonic, tuple(sorted(ins.regs_write.keys())))


def _ins_compare(a: Instruction, b: Instruction) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """Return (kind, good_detail, bad_detail) if ``a`` and ``b`` differ
    materially, else None.  Mnemonic alone is *not* enough — same PC and
    written-reg names but different written values is the canonical
    "divergence" we're chasing.
    """
    if a.pc != b.pc:
        return ("pc", {"pc": f"0x{a.pc:x}"}, {"pc": f"0x{b.pc:x}"})
    if a.regs_write != b.regs_write:
        return ("regs_write", {"regs_write": dict(a.regs_write)},
                              {"regs_write": dict(b.regs_write)})
    # Memory writes also count — compare addr+val+size tuples
    a_writes = [(m.addr, m.val, m.size) for m in a.mem if m.rw == "w"]
    b_writes = [(m.addr, m.val, m.size) for m in b.mem if m.rw == "w"]
    if a_writes != b_writes:
        return ("mem", {"writes": a_writes}, {"writes": b_writes})
    return None


def find_first_divergence(
    good: Sequence[Instruction],
    bad:  Sequence[Instruction],
) -> DivergencePoint | None:
    """Walk ``good`` and ``bad`` in lock-step until they differ or one
    ends.  Returns None when both run out simultaneously."""
    n = min(len(good), len(bad))
    for i in range(n):
        cmp = _ins_compare(good[i], bad[i])
        if cmp is not None:
            kind, gd, bd = cmp
            return DivergencePoint(
                kind=kind, good_idx=good[i].idx, bad_idx=bad[i].idx,
                pc=good[i].pc, good=gd, bad=bd,
            )
    if len(good) != len(bad):
        if len(good) < len(bad):
            head = bad[n]
            return DivergencePoint(
                kind="length", good_idx=-1, bad_idx=head.idx,
                pc=head.pc, good={"length": len(good)},
                bad={"length": len(bad), "extra_pc": f"0x{head.pc:x}"},
            )
        head = good[n]
        return DivergencePoint(
            kind="length", good_idx=head.idx, bad_idx=-1,
            pc=head.pc, good={"length": len(good), "extra_pc": f"0x{head.pc:x}"},
            bad={"length": len(bad)},
        )
    return None


def _find_resync(
    good: Sequence[Instruction], gi: int,
    bad:  Sequence[Instruction], bi: int,
    *,
    look_ahead: int = 200,
) -> tuple[int, int] | None:
    """After divergence, scan forward up to ``look_ahead`` steps on each
    side to find a (gi', bi') pair whose fingerprints match. Returns
    None if no resync was found within window — caller knows the two
    traces have structurally drifted."""
    good_marks = {_ins_fingerprint(good[i]): i
                  for i in range(gi, min(gi + look_ahead, len(good)))}
    for j in range(bi, min(bi + look_ahead, len(bad))):
        fp = _ins_fingerprint(bad[j])
        match = good_marks.get(fp)
        if match is not None:
            return (match, j)
    return None


def _build_candidates(
    dp: DivergencePoint,
    good: Sequence[Instruction],
    bad:  Sequence[Instruction],
) -> list[CandidateHypothesis]:
    """Heuristic-rank candidate hypotheses from a divergence point."""
    candidates: list[CandidateHypothesis] = []
    if dp.kind == "regs_write":
        diff_regs = []
        gw = dp.good.get("regs_write") or {}
        bw = dp.bad.get("regs_write") or {}
        for r in set(gw) | set(bw):
            if gw.get(r) != bw.get(r):
                diff_regs.append((r, gw.get(r), bw.get(r)))
        diff_regs.sort()
        for i, (reg, gv, bv) in enumerate(diff_regs):
            candidates.append(CandidateHypothesis(
                rank=i,
                kind="divergent_reg_write",
                subject=f"reg_diverge@{reg}@0x{dp.pc:x}",
                payload={
                    "pc":       f"0x{dp.pc:x}",
                    "register": reg,
                    "good":     None if gv is None else f"0x{gv:x}",
                    "bad":      None if bv is None else f"0x{bv:x}",
                },
                confidence=0.6,
                rationale=(
                    f"At pc=0x{dp.pc:x} good wrote {reg}=0x{gv:x} but bad "
                    f"wrote {reg}=0x{bv:x}. Graft this single register and "
                    f"re-run to see whether the rest of the divergence "
                    f"vanishes."
                ),
            ))
    elif dp.kind == "mem":
        candidates.append(CandidateHypothesis(
            rank=0,
            kind="divergent_mem_write",
            subject=f"mem_diverge@0x{dp.pc:x}",
            payload={"pc": f"0x{dp.pc:x}",
                     "good_writes": dp.good.get("writes"),
                     "bad_writes":  dp.bad.get("writes")},
            confidence=0.5,
            rationale=(
                f"Memory writes differ at pc=0x{dp.pc:x}. Likely cause: "
                f"earlier divergent register fed an str-family op. Walk "
                f"backward from this PC."
            ),
        ))
    elif dp.kind == "pc":
        candidates.append(CandidateHypothesis(
            rank=0,
            kind="control_flow_divergence",
            subject=f"cf_diverge@step{dp.good_idx}",
            payload={"good_pc": dp.good.get("pc"), "bad_pc": dp.bad.get("pc")},
            confidence=0.55,
            rationale=(
                "Same step index reached different PCs — an earlier "
                "conditional branched on a flag set by un-aligned "
                "register state. Look upstream for the predicate."
            ),
        ))
    elif dp.kind == "length":
        candidates.append(CandidateHypothesis(
            rank=0,
            kind="trace_length_mismatch",
            subject=f"len_diverge@{dp.pc:x}",
            payload={"good": dp.good, "bad": dp.bad},
            confidence=0.4,
            rationale="Traces diverged in length — one side terminated early.",
        ))
    return candidates


def localize_divergence(
    good: Iterable[Instruction] | Sequence[Instruction],
    bad:  Iterable[Instruction] | Sequence[Instruction],
    *,
    resync_look_ahead: int = 200,
) -> LocalizeResult:
    """Main entry point. Compare two trace sequences and surface the
    first divergence plus ranked hypotheses.

    Both arguments must be in trace order. They do not need equal
    length; a shorter prefix that agrees, then runs out, registers as
    a ``length`` divergence with the next instruction from the longer
    side reported in ``extra_pc``.
    """
    good_seq = list(good) if not isinstance(good, list) else good
    bad_seq  = list(bad)  if not isinstance(bad, list) else bad

    dp = find_first_divergence(good_seq, bad_seq)
    if dp is None:
        return LocalizeResult(divergence=None)
    candidates = _build_candidates(dp, good_seq, bad_seq)
    if dp.good_idx >= 0 and dp.bad_idx >= 0:
        # only attempt resync when both sides actually had an instruction
        gi = next((i for i, ins in enumerate(good_seq) if ins.idx == dp.good_idx), -1)
        bi = next((i for i, ins in enumerate(bad_seq)  if ins.idx == dp.bad_idx),  -1)
        if gi >= 0 and bi >= 0:
            r = _find_resync(good_seq, gi + 1, bad_seq, bi + 1,
                             look_ahead=resync_look_ahead)
            if r is not None:
                resync_at = (good_seq[r[0]].idx, bad_seq[r[1]].idx)
                return LocalizeResult(divergence=dp,
                                      candidates=tuple(candidates),
                                      resync_at=resync_at)
    return LocalizeResult(divergence=dp, candidates=tuple(candidates))
