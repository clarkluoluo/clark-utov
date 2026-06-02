"""MIT-clean reference reimplementations of C-library PRNG families.

These are CLEAN-ROOM reimplementations written from the *published algorithm
description* of the BSD ``random(3)`` additive-feedback generator (Park & Miller
seeding + the TYPE_3 lagged-Fibonacci feedback table). They are NOT, and must
never be, copied from glibc / bionic source. Each callable is registry DATA
carrying its own ``version`` + ``source`` provenance so a resolved model is
self-describing and auditable.

The two families modelled here share the SAME TYPE_3 structure (degree 31,
separation 3) — historically accurate: bionic inherited the BSD ``random()`` and
``#define rand() random()`` shape, and glibc's default ``random()`` is the same
TYPE_3 generator. They are kept as TWO distinct registry entries (different
``model_id`` / ``version``) so the resolver can prove *multi-family dispatch*
(spec §Fixtures (a)); a future divergent family (e.g. a musl LCG) is added as a
third entry with zero resolver edits.

Algorithm provenance (what these were reimplemented FROM, never copied):
  * S. K. Park & K. W. Miller, "Random Number Generators: Good Ones Are Hard to
    Find", CACM 31(10), 1988 — the 16807 multiplicative seeding congruence
    (Schrage decomposition to stay within int32).
  * 4.3BSD ``random(3)`` public algorithm description: degree 31, separation 3,
    additive feedback ``r[f] += r[r]``, output ``(word >> 1) & 0x7fffffff``,
    10*deg warm-up draws after seeding.
No proprietary constants beyond these published mathematical parameters.
"""

from __future__ import annotations

__all__ = [
    "BsdRandomType3",
    "bionic_random_type3",
    "glibc_random",
    "glibc_rand_type0",
    "ansi_lcg",
]

_INT32_MASK = 0xFFFF_FFFF
_INT31_MASK = 0x7FFF_FFFF
_M = 2_147_483_647  # 2**31 - 1, the Park-Miller modulus


def _as_s32(v: int) -> int:
    """Interpret ``v`` as a signed 32-bit integer (the C table word type)."""
    v &= _INT32_MASK
    return v - (1 << 32) if v & 0x8000_0000 else v


def _seed_table(seed: int, deg: int) -> list[int]:
    """Park-Miller multiplicative seeding of the ``deg``-word feedback table.

    ``r[0] = seed`` then ``r[i] = 16807 * r[i-1] (mod 2**31-1)`` via the Schrage
    decomposition documented for ``random(3)`` to avoid 64-bit overflow on the
    int32 table. Values are kept as signed 32-bit words."""
    if seed == 0:
        seed = 1  # the man-page rule: a 0 seed is treated as 1
    table = [0] * deg
    table[0] = _as_s32(seed)
    for i in range(1, deg):
        prev = table[i - 1]
        hi = prev // 127773
        lo = prev % 127773
        word = 16807 * lo - 2836 * hi
        if word < 0:
            word += _M
        table[i] = _as_s32(word)
    return table


class BsdRandomType3:
    """The BSD ``random(3)`` TYPE_3 additive-feedback generator (deg 31, sep 3).

    Stateful: ``seed(s)`` (re)initialises, ``next_word()`` advances one step and
    returns the 31-bit non-negative ``random()`` value. On bionic ``rand()`` is
    ``#define rand() random()`` so the raw word IS the ``rand()`` return; the
    ``project`` layer (e.g. ``low8`` = ``& 0xff``) is applied by ``eval_sequence``.

    Pure value model — no global process state; each instance owns its table so
    parallel evaluations never interfere (clean for the verifier's vectoring)."""

    DEG = 31
    SEP = 3

    def __init__(self, *, warmup_mul: int = 10) -> None:
        self._warmup_mul = warmup_mul
        self._table: list[int] = []
        self._fptr = 0
        self._rptr = 0

    def seed(self, s: int) -> "BsdRandomType3":
        self._table = _seed_table(int(s) & _INT32_MASK, self.DEG)
        self._fptr = self.SEP
        self._rptr = 0
        # Warm-up: discard 10 * DEG outputs so the seeding transient is gone
        # (the documented BSD convention).
        for _ in range(self._warmup_mul * self.DEG):
            self._step()
        return self

    def _step(self) -> int:
        """Advance one position; return the new fptr word (signed 32-bit)."""
        t = self._table
        val = _as_s32(t[self._fptr] + t[self._rptr])
        t[self._fptr] = val
        self._fptr = (self._fptr + 1) % self.DEG
        self._rptr = (self._rptr + 1) % self.DEG
        return val

    def next_word(self) -> int:
        """One ``random()`` draw: ``(word >> 1) & 0x7fffffff`` (non-negative)."""
        val = self._step()
        return (val & _INT32_MASK) >> 1 & _INT31_MASK

    def state(self) -> dict[str, object]:
        """A snapshot of the generator state (for ``state_update`` round-trips)."""
        return {"table": list(self._table), "fptr": self._fptr, "rptr": self._rptr}

    def load_state(self, st: dict[str, object]) -> "BsdRandomType3":
        self._table = list(st["table"])  # type: ignore[arg-type]
        self._fptr = int(st["fptr"])  # type: ignore[arg-type]
        self._rptr = int(st["rptr"])  # type: ignore[arg-type]
        return self


def bionic_random_type3(seed: int, count: int) -> list[int]:
    """``count`` consecutive bionic ``rand()`` words for ``srand(seed)``.

    bionic: ``rand() == random()`` (the TYPE_3 generator), so each word is the
    raw 31-bit ``random()`` return. Projection (``& 0xff`` etc.) is the caller's
    (``eval_sequence``)."""
    gen = BsdRandomType3(warmup_mul=10).seed(seed)
    return [gen.next_word() for _ in range(int(count))]


def glibc_random(seed: int, count: int) -> list[int]:
    """``count`` consecutive glibc ``random()`` words for ``srandom(seed)``.

    Same TYPE_3 structure as bionic (kept as a SEPARATE family entry to prove
    multi-family dispatch + independent provenance/version). glibc's default
    ``random()`` uses the identical degree-31 additive-feedback table."""
    gen = BsdRandomType3(warmup_mul=10).seed(seed)
    return [gen.next_word() for _ in range(int(count))]


_GLIBC_T0_A = 1103515245
_GLIBC_T0_C = 12345


def glibc_rand_type0(seed: int, count: int) -> list[int]:
    """``count`` consecutive glibc ``rand()`` words for the TYPE_0 simple LCG.

    glibc's ``random()`` falls back to a single-word linear-congruential
    generator when the state array is too small to run the additive-feedback
    table (``TYPE_0``, ``random(3)`` man page). Recurrence (published):

        state = (state * 1103515245 + 12345) & 0x7fffffff
        word  = state

    A genuinely DIVERGENT family from the bionic TYPE_3 stream (different
    algorithm class, not just a relabelled table) — this is the host-libc
    ``rand()`` shape that #2's ranker must rule out by behaviour, not by name.
    MIT-clean: written from the published LCG constants, not copied."""
    state = int(seed) & _INT32_MASK
    if state == 0:
        state = 1
    out: list[int] = []
    for _ in range(int(count)):
        state = (state * _GLIBC_T0_A + _GLIBC_T0_C) & _INT31_MASK
        out.append(state)
    return out


_ANSI_A = 1103515245
_ANSI_C = 12345


def ansi_lcg(seed: int, count: int) -> list[int]:
    """``count`` consecutive ANSI C ``rand()`` words (the K&R / C89 LCG).

    The reference ``rand()`` of the C89 standard (ISO/IEC 9899:1990 §7.20.2.1
    example implementation):

        next = next * 1103515245 + 12345
        word = (next >> 16) & 0x7fff      # RAND_MAX == 32767

    Distinct OUTPUT WIDTH (15-bit) and a high-bits projection — a third family
    that diverges from both BSD TYPE_3 and the glibc TYPE_0 low word. MIT-clean,
    written from the published standard, not copied."""
    next_ = int(seed) & _INT32_MASK
    out: list[int] = []
    for _ in range(int(count)):
        next_ = (next_ * _ANSI_A + _ANSI_C) & _INT32_MASK
        out.append((next_ >> 16) & 0x7FFF)
    return out
