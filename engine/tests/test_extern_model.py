"""spec #1 — generic extern executable-model registry: fixtures.

Covers the spec §Fixtures matrix:
  (a) multi-family dispatch       — rand + bionic → type3; + glibc → glibc.
  (b) mem-op via libc_boundary    — memcpy resolves to a uniform model whose
                                     .apply() delegates to synthesize_boundary_edge.
  (c) unknown symbol              — EXTERN_MODEL_UNAVAILABLE / no registry entry.
  (d) ambiguous (no tag)          — family-hint-only, NO silent pick.
  (e) TC2 proof-point             — eval_sequence("rand", bionic, 32, low8)
                                     reproduces a known seed→sequence vector.
  + uniform ModelSpec shape, evidence_level first-class, wrong-kind guards.

Vector SOURCE (proof-point): the 32-byte ``low8`` vector below is COMPUTED from
THIS module's own MIT-clean bionic model (``eval_sequence(seed=1, 32, low8)``)
and pinned here as a regression anchor. Its raw words for seed=1 begin
1804289383, 846930886, 1681692777, 1714636915 — the canonical published BSD/
glibc ``rand()`` sequence for ``srand(1)``, which independently confirms the
reimplementation is algorithmically correct (not an arbitrary self-consistent
vector).
"""

from __future__ import annotations

from engine.extern_model import (
    EXTERN_MODEL_UNAVAILABLE,
    KIND_MEM_EFFECT,
    ModelSpec,
    ModelUnavailable,
    resolve_extern_model,
)
from engine.libc_boundary import BoundaryEdgeUnresolved
from engine.oracle_provenance import BoundaryEdge
from engine.types import Instruction


# Known seed=1 → 32 × (rand() & 0xff) for the bionic TYPE_3 model (see header).
_BIONIC_SEED1_LOW8 = [
    103, 198, 105, 115, 81, 255, 74, 236, 41, 205, 186, 171, 242, 251, 227, 70,
    124, 194, 84, 248, 27, 232, 231, 141, 118, 90, 46, 99, 51, 159, 201, 154,
]


# --------------------------------------------------------------------------- #
# (a) multi-family dispatch.
# --------------------------------------------------------------------------- #
def test_multi_family_dispatch_bionic_vs_glibc():
    bionic = resolve_extern_model("rand", runtime_tags={"libc_family": "bionic"})
    glibc = resolve_extern_model("rand", runtime_tags={"libc_family": "glibc"})
    assert isinstance(bionic, ModelSpec)
    assert isinstance(glibc, ModelSpec)
    assert bionic.model_id == "bionic-random-type3"
    assert glibc.model_id == "glibc-random-type3"
    assert bionic.family == "bionic"
    assert glibc.family == "glibc"
    # Distinct registry entries / versions prove the dispatch is real.
    assert bionic.version != glibc.version


def test_dispatch_strips_plt_decoration():
    m = resolve_extern_model("rand@plt", runtime_tags={"libc_family": "bionic"})
    assert isinstance(m, ModelSpec)
    assert m.symbol == "rand"
    assert m.model_id == "bionic-random-type3"


def test_srand_and_random_aliases_resolve_same_family():
    for sym in ("srand", "random"):
        m = resolve_extern_model(sym, runtime_tags={"libc_family": "bionic"})
        assert isinstance(m, ModelSpec)
        assert m.model_id == "bionic-random-type3"


# --------------------------------------------------------------------------- #
# (b) mem-op via libc_boundary provider — uniform model, delegates to wrap.
# --------------------------------------------------------------------------- #
def _ins(idx, pc, mnem, regs_read):
    return Instruction(idx=idx, pc=pc, bytes_=b"\x00\x00\x00\x00",
                       mnemonic=mnem, regs_read=regs_read, regs_write={})


class _FakeImportMap:
    """Minimal import_map stand-in: resolve the call target → 'memcpy'."""

    def __init__(self, target, symbol):
        self._target = target
        self._symbol = symbol
        self.by_plt_addr = {target: symbol}

    def symbol_for(self, addr):
        return self._symbol if addr == self._target else None


def test_mem_effect_resolves_to_uniform_model():
    m = resolve_extern_model("memcpy")
    assert isinstance(m, ModelSpec)
    assert m.model_kind == KIND_MEM_EFFECT
    assert m.state_kind == "none"
    # Uniform metadata block — same shape as a PRNG model.
    assert set(m.to_dict()) == {
        "model_id", "symbol", "model_kind", "state_kind",
        "evidence_level", "version", "source", "family",
    }


def test_mem_effect_apply_delegates_to_libc_boundary():
    # A memcpy(dst=0x1000, src=0x2000, n=16) call; sink ⊆ dst → COPY edge.
    imap = _FakeImportMap(0x400A80, "memcpy")
    call = _ins(5, 0x400500, "bl", {"x0": 0x1000, "x1": 0x2000, "x2": 16})
    m = resolve_extern_model("memcpy")
    # An explicit boundary_edge is honored VERBATIM by synthesize_boundary_edge
    # (A8③ override), proving apply() is a true pass-through to the wrapped
    # libc_boundary synthesizer — not a re-implementation.
    edge = BoundaryEdge(sink_surface=0x1000, boundary_pc_from=0x400500,
                        boundary_pc_to=0x400500, source_ptr=0x2000)
    out = m.apply([call], 5, (0x1000, 16), imap, boundary_edge=edge)
    assert isinstance(out, BoundaryEdge)
    assert out.source_ptr == 0x2000


def test_mem_effect_apply_returns_structured_unresolved():
    # No call at the site → libc_boundary returns BoundaryEdgeUnresolved, and the
    # wrapper surfaces it verbatim (structured, never a fabricated edge).
    imap = _FakeImportMap(0x400A80, "memcpy")
    m = resolve_extern_model("memcpy")
    out = m.apply([], 99, (0x1000, 16), imap)
    assert isinstance(out, BoundaryEdgeUnresolved)
    assert out.verdict == "BOUNDARY_EDGE_UNRESOLVED"


# --------------------------------------------------------------------------- #
# (c) unknown symbol → EXTERN_MODEL_UNAVAILABLE.
# --------------------------------------------------------------------------- #
def test_unknown_symbol_is_structured_unavailable():
    out = resolve_extern_model("getrandom_v2", runtime_tags={"libc_family": "bionic"})
    assert isinstance(out, ModelUnavailable)
    assert out.verdict == EXTERN_MODEL_UNAVAILABLE
    assert out.reason == "no registry entry"
    assert out.family_hints == ()


def test_empty_symbol_is_unavailable():
    out = resolve_extern_model("")
    assert isinstance(out, ModelUnavailable)
    assert out.reason == "no registry entry"


# --------------------------------------------------------------------------- #
# (d) ambiguous (no tag) → family-hint-only, NO silent pick.
# --------------------------------------------------------------------------- #
def test_ambiguous_no_tag_yields_family_hints_no_silent_pick():
    out = resolve_extern_model("rand")  # no libc_family tag
    assert isinstance(out, ModelUnavailable)
    assert out.verdict == EXTERN_MODEL_UNAVAILABLE
    assert out.reason == "family-hint-only"
    # Both registered PRNG families handed to the ranker (#2).
    assert "bionic-random-type3" in out.family_hints
    assert "glibc-random-type3" in out.family_hints


def test_wrong_tag_value_yields_family_hints():
    out = resolve_extern_model("rand", runtime_tags={"libc_family": "musl"})
    assert isinstance(out, ModelUnavailable)
    assert out.reason == "family-hint-only"
    assert "bionic-random-type3" in out.family_hints


# --------------------------------------------------------------------------- #
# (e) TC2 proof-point — reproduce the known seed→sequence vector.
# --------------------------------------------------------------------------- #
def test_proof_point_bionic_rand_low8_reproduces_known_vector():
    m = resolve_extern_model("rand", runtime_tags={"libc_family": "bionic"})
    assert isinstance(m, ModelSpec)
    seq = m.eval_sequence(seed=1, count=32, project="low8")
    assert len(seq) == 32
    assert all(0 <= b <= 0xFF for b in seq)
    assert seq == _BIONIC_SEED1_LOW8


def test_proof_point_raw_words_match_canonical_bsd_rand():
    # Independent cross-check: the FIRST raw words for srand(1) are the published
    # canonical BSD/glibc rand() values — anchors the reimpl to a public oracle.
    m = resolve_extern_model("rand", runtime_tags={"libc_family": "bionic"})
    raw = m.eval_sequence(seed=1, count=4, project="raw")
    assert raw == [1804289383, 846930886, 1681692777, 1714636915]


def test_eval_sequence_is_deterministic_and_reseed_stable():
    m = resolve_extern_model("rand", runtime_tags={"libc_family": "bionic"})
    a = m.eval_sequence(seed=42, count=16)
    b = m.eval_sequence(seed=42, count=16)
    assert a == b
    c = m.eval_sequence(seed=43, count=16)
    assert c != a  # different seed → different stream


# --------------------------------------------------------------------------- #
# Uniform shape + evidence first-class + wrong-kind guards.
# --------------------------------------------------------------------------- #
def test_evidence_level_is_first_class_and_versioned():
    m = resolve_extern_model("rand", runtime_tags={"libc_family": "bionic"})
    assert m.evidence_level == "reference"
    assert m.version == "bionic-2023"
    assert "MIT-clean" in m.source  # provenance: not copied from GPL libc


def test_stateful_model_rejects_mem_apply():
    m = resolve_extern_model("rand", runtime_tags={"libc_family": "bionic"})
    try:
        m.apply([], 0, (0, 0), None)
    except TypeError as e:
        assert "mem_effect" in str(e)
    else:
        raise AssertionError("expected TypeError calling apply() on a stateful model")


def test_mem_model_rejects_eval_sequence():
    m = resolve_extern_model("memcpy")
    try:
        m.eval_sequence(1, 4)
    except TypeError as e:
        assert "stateful" in str(e)
    else:
        raise AssertionError("expected TypeError calling eval_sequence() on a mem model")


def test_unknown_projection_is_loud():
    m = resolve_extern_model("rand", runtime_tags={"libc_family": "bionic"})
    try:
        m.eval_sequence(1, 4, project="low4")
    except ValueError as e:
        assert "unknown projection" in str(e)
    else:
        raise AssertionError("expected ValueError on an unknown projection")
