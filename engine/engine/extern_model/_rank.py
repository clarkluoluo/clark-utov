"""spec #2 — family-agnostic candidate ranker over #1's extern models.

The layer ABOVE :mod:`engine.extern_model` (#1): given an extern ``symbol`` and a
few same-execution observations ``(seed, first-N returns)``, rank the candidate
implementation FAMILIES registered for that symbol and *explain* the ranking, so
"this is a PRNG" auto-advances to "most likely family X, here's why" instead of a
manual trial of host-libc / ANSI-LCG / bionic.

Design (A8, mirrors the spec four-check):

  1. **Reuse, don't rebuild** — the scorer is a generalization of
     :func:`engine.core._base._run_io_equivalence` (score a candidate by
     behavioural match against observed I/O). Each candidate's behaviour comes
     from #1's callable (``ModelFamilyHint.eval`` is the registry ``_eval``),
     evaluated via the SAME projection surface #1's ``eval_sequence`` uses. The
     ranker contains ZERO PRNG-specific logic.
  2. **Family-agnostic** — any ``symbol`` with ≥2 registered
     :class:`ModelFamilyHint` plugs in with NO ranker changes. Hints are DATA:
     they are derived from :data:`engine.extern_model.MODEL_REGISTRY`
     (stateful entries) and may be extended with :func:`register_family_hint`.
  3. **Subject = observed behaviour**, never the symbol name. Ranking is the
     prefix-match score of each family's projected stream against
     ``observed_returns``.
  4. **Preserve / advisory** — a :class:`CandidateRanking` is ADVISORY. It never
     auto-promotes a family to a closed finding; the verifier / parity gate stays
     the sole truth (架构原则 #1). Hence ``evidence_level`` rides along per
     candidate and the verdict is a ranking, not a confirmation.
  5. **Degenerate → structured** — no family matches ⇒ ``NO_CANDIDATE``; too few
     observations to discriminate ⇒ ``INSUFFICIENT_OBSERVATION`` (it names how
     many more it needs). NEVER a low-confidence silent pick.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "ModelFamilyHint",
    "CandidateScore",
    "CandidateRanking",
    "register_family_hint",
    "family_hints_for",
    "rank_model_candidates",
    "RANK_RANKED",
    "RANK_NO_CANDIDATE",
    "RANK_INSUFFICIENT_OBSERVATION",
]

# Verdicts (spec Contract). A ranking is advisory — these are NOT closed findings.
RANK_RANKED = "RANKED"
RANK_NO_CANDIDATE = "NO_CANDIDATE"
RANK_INSUFFICIENT_OBSERVATION = "INSUFFICIENT_OBSERVATION"

# Floor on observations: with zero observations there is nothing to score; with a
# single observation many families collide on one byte. The ranker also requires
# the winner to be DISCRIMINATED from the runner-up (see _discriminating); this
# is the hard floor below which we never even try.
_MIN_OBSERVATIONS = 1


@dataclass(frozen=True)
class ModelFamilyHint:
    """A declarative candidate family for one ``symbol`` (attaches to #1).

    The ranker treats this purely as DATA — ``eval`` is #1's registry callable
    (``MODEL_REGISTRY[*]["_eval"]``), so the candidate's behaviour is NOT
    re-implemented here. ``project`` selects the observable surface the candidate
    is compared on (matching #1's ``eval_sequence`` projections); ``seed_form`` /
    ``out_width`` / ``quick_check`` are self-describing provenance carried into
    the ranking output (spec Contract family-hint block)."""

    symbol: str
    family: str
    model_id: str
    eval: Callable[[int, int], list[int]]
    project: str = "low8"
    seed_form: str = "u32"
    out_width: int = 8
    quick_check: str = ""
    version: str = ""
    evidence_level: str = "reference"

    def project_word(self, word: int) -> int:
        return _project_word(word, self.project)


@dataclass(frozen=True)
class CandidateScore:
    """One family's score against ``observed_returns`` (ranking row).

    ``score`` ∈ [0,1] is the matched-prefix fraction. Exactly one of ``match`` /
    ``mismatch`` is populated (a full prefix match → ``match`` summary; any
    divergence → ``mismatch`` naming where + got/want). ``evidence_level`` rides
    along so a ``conjectured`` candidate is never dressed up as a confirmed one."""

    family: str
    model_id: str
    score: float
    matched: int
    total: int
    evidence_level: str
    version: str = ""
    match: str = ""
    mismatch: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "family": self.family,
            "model_id": self.model_id,
            "score": self.score,
            "evidence_level": self.evidence_level,
            "version": self.version,
        }
        if self.match:
            d["match"] = self.match
        if self.mismatch:
            d["mismatch"] = self.mismatch
        return d


@dataclass(frozen=True)
class CandidateRanking:
    """The advisory ranking result (spec Contract output).

    NOT a closed finding (A8③): ``verdict`` ranks candidates; the verifier /
    parity gate must still confirm. ``why_top`` is an EXPLICIT reason the #1 row
    won — not just its name — citing the match count and the runner-up's
    divergence."""

    symbol: str
    verdict: str
    ranked: tuple[CandidateScore, ...] = ()
    why_top: str = ""
    detail: str = ""
    observations: int = 0
    needed_observations: int = 0

    @property
    def top(self) -> CandidateScore | None:
        return self.ranked[0] if self.ranked else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "verdict": self.verdict,
            "ranked": [c.to_dict() for c in self.ranked],
            "why_top": self.why_top,
            "detail": self.detail,
            "observations": self.observations,
            "needed_observations": self.needed_observations,
        }


def _project_word(word: int, project: str) -> int:
    if project == "raw":
        return word
    if project == "low8":
        return word & 0xFF
    if project == "low16":
        return word & 0xFFFF
    if project == "bit0":
        return word & 1
    raise ValueError(
        f"unknown projection {project!r} (expected raw|low8|low16|bit0) — "
        "no silent raw fallback")


# --------------------------------------------------------------------------- #
# Family-hint registry — DATA, derived from #1's MODEL_REGISTRY (reuse).
# --------------------------------------------------------------------------- #
# A symbol → its registered candidate families. Built lazily from the stateful
# entries of engine.extern_model.MODEL_REGISTRY so the callables have ONE source
# of truth (#1). Tests / callers may add synthetic families via
# register_family_hint — that is the "second symbol, no code change" path.
_EXTRA_HINTS: dict[str, list[ModelFamilyHint]] = {}

# Per-model projection/observable overrides (the family-hint block of the spec).
# Keyed by model_id; absent → low8 default (the TC2 rand() & 0xff surface).
_MODEL_HINT_META: dict[str, dict[str, Any]] = {
    "bionic-random-type3": {
        "project": "low8", "out_width": 8, "seed_form": "u32",
        "quick_check": "low8-prefix",
    },
    "glibc-random-type3": {
        "project": "low8", "out_width": 8, "seed_form": "u32",
        "quick_check": "low8-prefix",
    },
    "glibc-rand-type0": {
        "project": "low8", "out_width": 8, "seed_form": "u32",
        "quick_check": "low8-prefix",
    },
    "ansi-lcg": {
        "project": "low8", "out_width": 8, "seed_form": "u32",
        "quick_check": "low8-prefix",
    },
}


def _registry_hints(symbol: str) -> list[ModelFamilyHint]:
    """Derive family hints from #1's stateful registry entries for ``symbol``.

    Reuse, not rebuild: the candidate callables are the registry ``_eval`` (the
    #1 reference impls). Imported lazily to avoid a circular import with the
    package ``__init__`` that owns the registry."""
    from . import MODEL_REGISTRY, KIND_STATEFUL  # local import: avoids cycle

    base = symbol.split("@", 1)[0]
    hints: list[ModelFamilyHint] = []
    for entry in MODEL_REGISTRY:
        if entry.get("model_kind") != KIND_STATEFUL:
            continue
        if base not in entry.get("symbols", ()):
            continue
        ev = entry.get("_eval")
        if ev is None:
            continue
        meta = _MODEL_HINT_META.get(entry["model_id"], {})
        hints.append(ModelFamilyHint(
            symbol=base,
            family=entry.get("family", entry["model_id"]),
            model_id=entry["model_id"],
            eval=ev,
            project=meta.get("project", "low8"),
            seed_form=meta.get("seed_form", "u32"),
            out_width=meta.get("out_width", 8),
            quick_check=meta.get("quick_check", ""),
            version=entry.get("version", ""),
            evidence_level=entry.get("evidence_level", "reference"),
        ))
    return hints


def register_family_hint(hint: ModelFamilyHint) -> None:
    """Register an extra candidate family (the family-agnostic extension point).

    This is how a NEW symbol becomes rankable with zero ranker changes (spec
    Fixture (b)): register ≥2 hints for it, then call
    :func:`rank_model_candidates`. Idempotent on ``(symbol, model_id)``."""
    bucket = _EXTRA_HINTS.setdefault(hint.symbol, [])
    if any(h.model_id == hint.model_id for h in bucket):
        return
    bucket.append(hint)


# Seed window used to collapse behavioural-alias families (see family_hints_for).
_ALIAS_PROBE_SEED = 1
_ALIAS_PROBE_WINDOW = 16


def _behaviour_key(hint: ModelFamilyHint) -> tuple[int, ...] | None:
    """A fingerprint of the family's projected stream for alias collapsing.

    Two registered families that produce a BYTE-IDENTICAL stream are the same
    behavioural candidate (they can never diverge, so the ranker can never tell
    them apart — listing both as rivals would manufacture a false tie). Returns
    ``None`` if the candidate cannot be probed (kept distinct, never collapsed)."""
    try:
        words = hint.eval(_ALIAS_PROBE_SEED, _ALIAS_PROBE_WINDOW)
        return tuple(hint.project_word(w) for w in words)
    except Exception:
        return None


def family_hints_for(symbol: str) -> tuple[ModelFamilyHint, ...]:
    """All DISTINCT candidate families registered for ``symbol`` (registry +
    extras).

    De-duplicated by ``model_id`` AND by behaviour: families whose projected
    stream is byte-identical (e.g. bionic ``random()`` vs glibc ``random()``,
    both the BSD TYPE_3 table — kept as two #1 registry entries only to prove tag
    DISPATCH) collapse to one behavioural candidate, since the ranker scores
    behaviour and could never split them. Registry order wins; the first
    occurrence of a behaviour is kept."""
    base = symbol.split("@", 1)[0]
    candidates: list[ModelFamilyHint] = list(_registry_hints(base))
    seen_ids = {h.model_id for h in candidates}
    for h in _EXTRA_HINTS.get(base, ()):
        if h.model_id not in seen_ids:
            candidates.append(h)
            seen_ids.add(h.model_id)

    out: list[ModelFamilyHint] = []
    seen_behaviour: dict[tuple[int, ...], str] = {}
    for h in candidates:
        key = _behaviour_key(h)
        if key is not None and key in seen_behaviour:
            continue  # behavioural alias of an already-kept family
        if key is not None:
            seen_behaviour[key] = h.model_id
        out.append(h)
    return tuple(out)


# --------------------------------------------------------------------------- #
# The scorer — generalized _run_io_equivalence (behavioural prefix match).
# --------------------------------------------------------------------------- #
def _prefix_match(observed: Sequence[int], produced: Sequence[int]) -> int:
    """Length of the longest common prefix (the behavioural-equivalence span)."""
    n = 0
    for o, p in zip(observed, produced):
        if int(o) != int(p):
            break
        n += 1
    return n


def _score_family(
    hint: ModelFamilyHint,
    observed_seed: int,
    observed: Sequence[int],
) -> CandidateScore:
    """Score ONE family: run its #1 callable for ``len(observed)`` draws, project
    to the family's observable surface, measure the matched prefix. This is the
    generalized ``_run_io_equivalence`` step — candidate behaviour vs observed
    behaviour, with a structured (never raising) score row."""
    total = len(observed)
    try:
        words = hint.eval(int(observed_seed), total)
        produced = [hint.project_word(w) for w in words]
    except Exception as exc:  # a broken candidate scores 0, honestly labelled
        return CandidateScore(
            family=hint.family, model_id=hint.model_id, score=0.0,
            matched=0, total=total, evidence_level=hint.evidence_level,
            version=hint.version,
            mismatch=f"eval errored: {type(exc).__name__}: {exc}")
    matched = _prefix_match(observed, produced)
    score = matched / total if total else 0.0
    if matched == total and total > 0:
        match = (f"{matched}/{total} of {hint.project}({hint.symbol}) matched "
                 f"(seed={observed_seed})")
        return CandidateScore(
            family=hint.family, model_id=hint.model_id, score=score,
            matched=matched, total=total, evidence_level="observed_match",
            version=hint.version, match=match)
    got = produced[matched] if matched < len(produced) else None
    want = observed[matched] if matched < total else None
    mismatch = (f"diverges at return #{matched + 1} "
                f"(got {got} want {want}); matched {matched}/{total} "
                f"of {hint.project}({hint.symbol})")
    return CandidateScore(
        family=hint.family, model_id=hint.model_id, score=score,
        matched=matched, total=total, evidence_level=hint.evidence_level,
        version=hint.version, mismatch=mismatch)


def _discriminating(ranked: Sequence[CandidateScore]) -> bool:
    """True iff the top family is BEHAVIOURALLY distinguished from the runner-up.

    The winner must be a full match AND strictly out-match every other family on
    the observed prefix. If two families tie on a full match, the observation is
    too short to pick between them (→ INSUFFICIENT_OBSERVATION); we never silently
    break the tie."""
    if not ranked:
        return False
    top = ranked[0]
    if top.matched != top.total or top.total == 0:
        return False
    if len(ranked) == 1:
        return True
    return top.matched > ranked[1].matched


def rank_model_candidates(
    symbol: str,
    *,
    observed_seed: int,
    observed_returns: Sequence[int],
    runtime_tags: Mapping[str, str] | None = None,
) -> CandidateRanking:
    """Rank the candidate families for ``symbol`` against observed behaviour.

    Family-agnostic (spec Acceptance): every family registered for ``symbol``
    (registry-derived + :func:`register_family_hint` extras) is scored by the
    same generalized ``_run_io_equivalence`` prefix-match — the ranker knows
    nothing PRNG-specific. ``runtime_tags`` is accepted for parity with
    :func:`engine.extern_model.resolve_extern_model` and reserved for future
    tag-narrowed candidate sets; it never silences a degenerate verdict.

    Verdicts (A8④, structured — never a silent low-confidence pick):
      * ``NO_CANDIDATE`` — no family registered, or no family matched even the
        first observation.
      * ``INSUFFICIENT_OBSERVATION`` — fewer observations than needed to
        discriminate the winner from the runner-up; ``needed_observations`` names
        how many it wants.
      * ``RANKED`` — a discriminated winner; ``why_top`` explains WHY (match count
        + runner-up divergence), and every row carries match/mismatch + evidence.

    The result is ADVISORY (A8③): it produces candidates the verifier/parity gate
    must still confirm — a family is never auto-promoted to a closed finding."""
    base = symbol.split("@", 1)[0]
    hints = family_hints_for(base)
    observed = list(observed_returns)
    n = len(observed)

    if not hints:
        return CandidateRanking(
            symbol=base, verdict=RANK_NO_CANDIDATE, observations=n,
            detail=(f"no candidate family registered for {base!r}; register ≥2 "
                    "ModelFamilyHint (or a MODEL_REGISTRY stateful entry) to make "
                    "it rankable. No silent pick."))

    if n < _MIN_OBSERVATIONS:
        return CandidateRanking(
            symbol=base, verdict=RANK_INSUFFICIENT_OBSERVATION, observations=n,
            needed_observations=_MIN_OBSERVATIONS,
            detail=(f"{n} observation(s) for {base!r} with {len(hints)} candidate "
                    f"families; need ≥{_MIN_OBSERVATIONS} return value(s) to score "
                    "any family. Capture more same-execution returns."))

    scores = [_score_family(h, observed_seed, observed) for h in hints]
    ranked = tuple(sorted(scores, key=lambda c: (-c.matched, c.model_id)))
    top = ranked[0]

    # No family matched even the first observation → NO_CANDIDATE (not a guess).
    if top.matched == 0:
        worst = "; ".join(
            f"{c.family}: {c.mismatch}" for c in ranked) or "(no detail)"
        return CandidateRanking(
            symbol=base, verdict=RANK_NO_CANDIDATE, ranked=ranked,
            observations=n,
            detail=(f"none of {len(hints)} registered families for {base!r} "
                    f"matched even return #1 under seed={observed_seed}. "
                    f"Per-family divergence — {worst}. The observed behaviour "
                    "belongs to an unregistered family; add a candidate."))

    # A partial-but-not-full top, or a full-match tie, means we cannot yet commit.
    if not _discriminating(ranked):
        # How many more observations would help: at least enough to extend past
        # the current matched prefix (one more than the best matched span).
        needed = max(top.matched + 1, _MIN_OBSERVATIONS + 1)
        if top.matched == top.total and len(ranked) > 1 \
                and ranked[1].matched == top.matched:
            tie = ranked[1].family
            detail = (f"{base!r}: top family {top.family} and {tie} both match "
                      f"all {n} observed returns — the prefix is too short to "
                      f"discriminate them. Need ≥{needed} returns to split the "
                      "tie. No silent pick between equal candidates.")
        else:
            detail = (f"{base!r}: best family {top.family} matched only "
                      f"{top.matched}/{n} returns — not a full-prefix match, so "
                      f"the family is not yet committable. Need ≥{needed} "
                      "consistent returns (or a different candidate set).")
        return CandidateRanking(
            symbol=base, verdict=RANK_INSUFFICIENT_OBSERVATION, ranked=ranked,
            observations=n, needed_observations=needed, detail=detail)

    runner_up = ranked[1] if len(ranked) > 1 else None
    if runner_up is not None:
        why_top = (
            f"{top.family} ({top.model_id}) is the only family whose "
            f"{top.match.split(' (')[0]}; the runner-up {runner_up.family} "
            f"{runner_up.mismatch}. Ranked by behavioural prefix match on "
            f"{n} same-execution returns (seed={observed_seed}).")
    else:
        why_top = (
            f"{top.family} ({top.model_id}) is the sole registered family and "
            f"its {top.match.split(' (')[0]} on {n} returns "
            f"(seed={observed_seed}).")
    return CandidateRanking(
        symbol=base, verdict=RANK_RANKED, ranked=ranked, why_top=why_top,
        observations=n,
        detail=("advisory ranking — the verifier/parity gate remains the sole "
                "truth; a family is never auto-promoted to a closed finding."))
