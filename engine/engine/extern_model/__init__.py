"""spec #1 — generic extern executable-model registry.

A platform capability: resolve an extern SYMBOL (+ runtime tags) to an
*executable, versioned, evidence-tagged* reference model with a UNIFORM
interface, so the verifier never hand-writes per-target PRNG/libc code. General
first — TC2 (bionic ``rand``) is only a proof-point, not the design shape.

Architecture (A8①, mirrors :data:`engine.core._base.ALGORITHM_TEMPLATES`):
  * :data:`MODEL_REGISTRY` is declarative DATA keyed by ``(symbol, tag-matchers)``.
    Adding a model = one registry entry, ZERO resolver/verifier edits.
  * :func:`resolve_extern_model` dispatches symbol + ``runtime_tags`` → exactly
    one :class:`ModelSpec`, or — on unknown / ambiguous / family-hint-only — a
    structured :class:`ModelUnavailable` (NEVER a silent pick; A8④).
  * Two shapes behind ONE door (A8②): ``stateful`` PRNG/time models expose a
    callable ``eval_sequence``; ``mem_effect`` models (memcpy/memset/memmove)
    WRAP :func:`engine.libc_boundary.synthesize_boundary_edge` — they do NOT
    re-implement the effect (A8① reuse).

Preserve (A8③): this layer sits ABOVE the existing
:func:`engine.import_map.extern_summary` classification, which is unchanged. A
caller that never calls :func:`resolve_extern_model` sees zero behaviour change.

Evidence (A8④): ``evidence_level`` is a FIRST-CLASS field — a ``conjectured``
model can never be presented as ``reference``. PRNG reference impls are MIT-clean
reimplementations (see :mod:`engine.extern_model._prng`), never copied from GPL
libc, each carrying ``version`` + ``source`` provenance.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

from ..import_map import extern_summary
from ..libc_boundary import synthesize_boundary_edge
from ._prng import (
    ansi_lcg,
    bionic_random_type3,
    glibc_rand_type0,
    glibc_random,
)

from ._rank import (
    CandidateRanking,
    CandidateScore,
    ModelFamilyHint,
    RANK_INSUFFICIENT_OBSERVATION,
    RANK_NO_CANDIDATE,
    RANK_RANKED,
    family_hints_for,
    rank_model_candidates,
    register_family_hint,
)

__all__ = [
    "ModelSpec",
    "ModelUnavailable",
    "MODEL_REGISTRY",
    "resolve_extern_model",
    "EXTERN_MODEL_UNAVAILABLE",
    "KIND_STATEFUL",
    "KIND_MEM_EFFECT",
    # spec #2 — candidate ranker
    "ModelFamilyHint",
    "CandidateScore",
    "CandidateRanking",
    "rank_model_candidates",
    "register_family_hint",
    "family_hints_for",
    "RANK_RANKED",
    "RANK_NO_CANDIDATE",
    "RANK_INSUFFICIENT_OBSERVATION",
]

EXTERN_MODEL_UNAVAILABLE = "EXTERN_MODEL_UNAVAILABLE"

# Recognised evidence levels, ordered weakest→strongest. ``evidence_level`` is a
# first-class field: a model may NEVER be presented one notch stronger than it is
# (A8④). ``reference`` = a faithful reimplementation of a published algorithm;
# ``observed_match`` = matched against captured outputs for this target;
# ``conjectured`` = a guess, not yet confirmed.
EVIDENCE_LEVELS = ("conjectured", "observed_match", "reference")

# Model kinds.
KIND_STATEFUL = "stateful"      # PRNG / time: callable eval_sequence + state_update
KIND_MEM_EFFECT = "mem_effect"  # memcpy/memset/memmove: WRAPS libc_boundary


# --------------------------------------------------------------------------- #
# ModelSpec — the uniform resolved model (one shape for every kind).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """A resolved, executable, versioned, evidence-tagged extern model.

    Uniform across kinds (A8②): the metadata block (``model_id`` … ``source``)
    is identical whether the extern is a PRNG or a memcpy. The callable surface
    differs by ``model_kind``:

      * ``stateful`` → :meth:`eval_sequence` (and :meth:`state_update`).
      * ``mem_effect`` → :meth:`apply` (delegates to ``libc_boundary``).

    Calling the wrong-kind method raises ``TypeError`` (honest, never a silent
    no-op). ``_eval`` / ``_provider`` hold the registry callable; they are not
    part of the serialized spec."""

    model_id: str
    symbol: str
    model_kind: str          # KIND_STATEFUL | KIND_MEM_EFFECT
    state_kind: str          # "prng" | "time" | "none"
    evidence_level: str      # one of EVIDENCE_LEVELS
    version: str             # "bionic-2023" | "glibc-2.31" | "ansi-c89" ...
    source: str              # registry entry id + provenance (algorithm citation)
    family: str = ""         # the dispatch family (e.g. "bionic", "glibc")
    # callable backends (kind-specific; not serialized)
    _eval: Callable[[int, int], list[int]] | None = field(
        default=None, repr=False, compare=False)
    _provider: Callable[..., Any] | None = field(
        default=None, repr=False, compare=False)

    # -- uniform metadata ---------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "symbol": self.symbol,
            "model_kind": self.model_kind,
            "state_kind": self.state_kind,
            "evidence_level": self.evidence_level,
            "version": self.version,
            "source": self.source,
            "family": self.family,
        }

    # -- stateful surface (PRNG / time) -------------------------------------- #
    def eval_sequence(
        self,
        seed: int,
        count: int,
        *,
        project: str = "raw",
    ) -> list[int]:
        """``count`` consecutive draws of this stateful model for ``seed``.

        ``project`` selects the observable surface of each word:
          * ``raw``  → the full generator word (e.g. 31-bit ``random()``).
          * ``low8`` → ``word & 0xff`` (the TC2 ``rand() & 0xff`` proof-point).
          * ``low16``→ ``word & 0xffff``.
          * ``bit0`` → ``word & 1``.
        Unknown ``project`` → ``ValueError`` (never a silent raw fallback)."""
        if self.model_kind != KIND_STATEFUL or self._eval is None:
            raise TypeError(
                f"eval_sequence is only valid for a {KIND_STATEFUL} model; "
                f"{self.model_id} is {self.model_kind}")
        words = self._eval(int(seed), int(count))
        return [_project_word(w, project) for w in words]

    def state_update(self, state: int, *, count: int = 1) -> list[int]:
        """Advance the model ``count`` steps from ``state`` (a seed), returning
        the raw words. Thin convenience over :meth:`eval_sequence` raw — kept so
        the spec's ``state_update(state) -> state`` contract has a home; the PRNG
        is reseed-deterministic so a seed fully identifies the state."""
        if self.model_kind != KIND_STATEFUL or self._eval is None:
            raise TypeError(
                f"state_update is only valid for a {KIND_STATEFUL} model; "
                f"{self.model_id} is {self.model_kind}")
        return self._eval(int(state), int(count))

    # -- mem_effect surface (delegates to libc_boundary) --------------------- #
    def apply(
        self,
        trace: Iterable[Any],
        call_site: int,
        sink_region: tuple[int, int],
        import_map: Any,
        **kwargs: Any,
    ) -> Any:
        """Synthesize the sink→source effect for this mem-op by WRAPPING
        :func:`engine.libc_boundary.synthesize_boundary_edge` (A8① reuse — this
        model does NOT re-implement memcpy/memset). Returns whatever the boundary
        synthesizer returns: a ``BoundaryEdge`` or a structured
        ``BoundaryEdgeUnresolved`` (the honest "cannot synthesize" verdict)."""
        if self.model_kind != KIND_MEM_EFFECT or self._provider is None:
            raise TypeError(
                f"apply is only valid for a {KIND_MEM_EFFECT} model; "
                f"{self.model_id} is {self.model_kind}")
        return self._provider(trace, call_site, sink_region, import_map, **kwargs)


@dataclass(frozen=True)
class ModelUnavailable:
    """Structured "no model" verdict (A8④ — never a silent pick).

    ``reason`` is one of ``no registry entry`` / ``ambiguous-needs-tags`` /
    ``family-hint-only``. ``family_hints`` is the candidate model-id set handed
    to #2 (the candidate ranker) when the symbol is known but a tag could not
    disambiguate the family."""

    symbol: str
    reason: str
    family_hints: tuple[str, ...] = ()
    detail: str = ""
    verdict: str = EXTERN_MODEL_UNAVAILABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "symbol": self.symbol,
            "reason": self.reason,
            "family_hints": list(self.family_hints),
            "detail": self.detail,
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
# mem_effect provider — WRAPS libc_boundary.synthesize_boundary_edge.
# --------------------------------------------------------------------------- #
def _mem_effect_provider(
    trace: Iterable[Any],
    call_site: int,
    sink_region: tuple[int, int],
    import_map: Any,
    **kwargs: Any,
) -> Any:
    """Adapter: present ``synthesize_boundary_edge`` as a model callable. Reuse,
    not rebuild — the COPY/CONST edge logic and the ``BoundaryEdgeUnresolved``
    degenerate path live in :mod:`engine.libc_boundary` and are untouched."""
    return synthesize_boundary_edge(trace, call_site, sink_region, import_map, **kwargs)


# --------------------------------------------------------------------------- #
# MODEL_REGISTRY — declarative. Add a model = add an entry here, NOTHING else.
# --------------------------------------------------------------------------- #
# Each entry is a dict of ModelSpec metadata + a ``tags`` matcher + a backend.
# The resolver groups entries by symbol, filters by tag match, and dispatches.
#
#   tags : the runtime_tags this entry REQUIRES (subset match). An entry with an
#          empty/absent matcher for a key is family-agnostic on that key.
#   family : the dispatch family label (surfaces in family_hints on ambiguity).
#
# PRNG reference impls are MIT-clean (see _prng.py); ``source`` cites the public
# algorithm, never the GPL libc source.
MODEL_REGISTRY: list[dict[str, Any]] = [
    # ── stateful PRNG: bionic rand/random (TYPE_3) — the TC2 proof-point ──── #
    {
        "model_id": "bionic-random-type3",
        "symbols": ("rand", "random", "srand"),
        "tags": {"libc_family": "bionic"},
        "family": "bionic",
        "model_kind": KIND_STATEFUL,
        "state_kind": "prng",
        "evidence_level": "reference",
        "version": "bionic-2023",
        "source": ("extern_model/_prng.py:bionic_random_type3 — MIT-clean "
                   "reimplementation of BSD random(3) TYPE_3 (deg 31, sep 3); "
                   "Park-Miller 16807 seeding (CACM 31(10) 1988). Not copied "
                   "from bionic source."),
        "_eval": bionic_random_type3,
    },
    # ── stateful PRNG: glibc random (TYPE_3) — second family for dispatch ─── #
    {
        "model_id": "glibc-random-type3",
        "symbols": ("rand", "random", "srand", "srandom"),
        "tags": {"libc_family": "glibc"},
        "family": "glibc",
        "model_kind": KIND_STATEFUL,
        "state_kind": "prng",
        "evidence_level": "reference",
        "version": "glibc-2.31",
        "source": ("extern_model/_prng.py:glibc_random — MIT-clean "
                   "reimplementation of glibc default random() TYPE_3 (deg 31, "
                   "sep 3). Not copied from glibc source."),
        "_eval": glibc_random,
    },
    # ── stateful PRNG: glibc rand() TYPE_0 simple LCG — divergent family ─── #
    {
        "model_id": "glibc-rand-type0",
        "symbols": ("rand", "srand"),
        "tags": {"libc_family": "glibc", "rand_state": "type0"},
        "family": "glibc-lcg",
        "model_kind": KIND_STATEFUL,
        "state_kind": "prng",
        "evidence_level": "reference",
        "version": "glibc-type0",
        "source": ("extern_model/_prng.py:glibc_rand_type0 — MIT-clean "
                   "reimplementation of glibc random(3) TYPE_0 simple LCG "
                   "(state*1103515245+12345 & 0x7fffffff). Not copied from glibc."),
        "_eval": glibc_rand_type0,
    },
    # ── stateful PRNG: ANSI C89 rand() LCG — third divergent family ──────── #
    {
        "model_id": "ansi-lcg",
        "symbols": ("rand", "srand"),
        "tags": {"libc_family": "ansi"},
        "family": "ansi-lcg",
        "model_kind": KIND_STATEFUL,
        "state_kind": "prng",
        "evidence_level": "reference",
        "version": "ansi-c89",
        "source": ("extern_model/_prng.py:ansi_lcg — MIT-clean reimplementation "
                   "of the ISO C89 §7.20.2.1 example rand() LCG (RAND_MAX 32767). "
                   "Not copied."),
        "_eval": ansi_lcg,
    },
    # ── mem_effect: memcpy/memset/memmove → wrap libc_boundary ───────────── #
    {
        "model_id": "libc-mem-effect",
        "symbols": ("memcpy", "memmove", "memset"),
        "tags": {},                       # family-agnostic: the ABI is universal
        "family": "libc-mem",
        "model_kind": KIND_MEM_EFFECT,
        "state_kind": "none",
        "evidence_level": "reference",
        "version": "ansi-c89",
        "source": ("extern_model wraps engine.libc_boundary."
                   "synthesize_boundary_edge (reuse, not rebuild); ABI from "
                   "import_map.EXTERN_SUMMARIES."),
        "_provider": _mem_effect_provider,
    },
]


def _entry_to_spec(
    entry: dict[str, Any],
    symbol: str,
    runtime_tags: Mapping[str, str],
) -> ModelSpec:
    """Materialize a registry entry into a :class:`ModelSpec` for ``symbol``.

    Reads (never mutates, A8③) the existing
    :func:`engine.import_map.extern_summary` classification and cross-checks the
    registry's ``state_kind`` against it. A mismatch is surfaced in ``source``
    (audit), never silently reconciled — the model layer sits ABOVE the
    classification and must stay honest about a disagreement."""
    source = entry["source"]
    summ = extern_summary(symbol)
    if summ is not None and summ.state_kind != entry["state_kind"]:
        source = (f"{source} [WARN: import_map classifies {symbol!r} state_kind="
                  f"{summ.state_kind!r}, registry says {entry['state_kind']!r}]")
    return ModelSpec(
        model_id=entry["model_id"],
        symbol=symbol,
        model_kind=entry["model_kind"],
        state_kind=entry["state_kind"],
        evidence_level=entry["evidence_level"],
        version=entry["version"],
        source=source,
        family=entry.get("family", ""),
        _eval=entry.get("_eval"),
        _provider=entry.get("_provider"),
    )


def _tags_match(required: Mapping[str, str], runtime_tags: Mapping[str, str]) -> bool:
    """An entry matches when every tag it REQUIRES is present and equal in
    ``runtime_tags`` (subset match). An entry with no required tags matches any
    runtime_tags (family-agnostic, e.g. the universal mem ABI)."""
    for k, v in required.items():
        if runtime_tags.get(k) != v:
            return False
    return True


# --------------------------------------------------------------------------- #
# resolve_extern_model — the single entry door (A8②).
# --------------------------------------------------------------------------- #
def resolve_extern_model(
    symbol: str,
    *,
    runtime_tags: Mapping[str, str] | None = None,
) -> ModelSpec | ModelUnavailable:
    """Resolve an extern ``symbol`` (+ runtime tags) to a uniform model.

    Dispatch (A8② / spec Contract):
      1. Strip any ``@plt`` decoration; collect every registry entry that lists
         ``symbol``.
      2. No entry → :class:`ModelUnavailable` ``reason="no registry entry"``.
      3. Filter entries by ``runtime_tags`` (subset match). Tag-agnostic entries
         (no required tags, e.g. the mem ABI) always match.
      4. Exactly one match → resolve to that :class:`ModelSpec`.
      5. >1 distinct family match → :class:`ModelUnavailable`
         ``reason="ambiguous-needs-tags"`` with ``family_hints`` (NEVER a silent
         pick; hands off to #2's ranker).
      6. Entries exist but NONE matched the tags → if the symbol carries multiple
         families, ``reason="family-hint-only"`` with the full family-hint set;
         if a single family entry simply didn't match its required tags, also
         ``family-hint-only`` (the tag is needed to commit).

    The existing :func:`engine.import_map.extern_summary` classification is read
    (state_kind / ABI cross-check) but never mutated (A8③)."""
    if not symbol:
        return ModelUnavailable(
            symbol="", reason="no registry entry",
            detail="empty symbol — nothing to resolve")

    base = symbol.split("@", 1)[0]
    runtime_tags = dict(runtime_tags or {})

    candidates = [e for e in MODEL_REGISTRY if base in e["symbols"]]
    if not candidates:
        return ModelUnavailable(
            symbol=base, reason="no registry entry",
            detail=(f"no extern model registered for {base!r}; the import_map "
                    "extern_summary may still classify it (opaque external "
                    "state), but no executable model exists — add a "
                    "MODEL_REGISTRY entry"))

    # The family-hint set for this symbol (every registered family), surfaced on
    # any ambiguity so #2's ranker has the full candidate set.
    all_families = tuple(dict.fromkeys(e["model_id"] for e in candidates))

    matched = [e for e in candidates if _tags_match(e["tags"], runtime_tags)]

    if len(matched) == 1:
        return _entry_to_spec(matched[0], base, runtime_tags)

    if len(matched) > 1:
        # Multiple distinct families matched (or a tag is too weak to pick one).
        fams = tuple(dict.fromkeys(e["model_id"] for e in matched))
        if len(fams) == 1:
            # Same family, multiple symbol aliases — first is fine (one model).
            return _entry_to_spec(matched[0], base, runtime_tags)
        return ModelUnavailable(
            symbol=base, reason="ambiguous-needs-tags", family_hints=fams,
            detail=(f"{base!r} matched {len(fams)} families {fams} under tags "
                    f"{runtime_tags or '{}'} — supply a disambiguating tag "
                    "(e.g. libc_family). No silent pick."))

    # No entry matched the supplied tags → family-hint-only (hand off to #2).
    return ModelUnavailable(
        symbol=base, reason="family-hint-only", family_hints=all_families,
        detail=(f"{base!r} has registered families {all_families} but none "
                f"matched runtime_tags {runtime_tags or '{}'} — a family tag is "
                "required to commit to a model. Family hints handed to the "
                "ranker (#2). No silent pick."))
