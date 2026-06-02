"""Canonical byte order for AArch64 opcode bytes — one order, end-to-end.

Root cause this fixes (VMP class-8 re-occurrence, 2026-05-31): a trace's per-
instruction ``"bytes"`` field can be recorded in EITHER convention —
little-endian memory order (``[69,3a,00,f9]`` for the word ``0x f9003a69``) or
the word's MSB-first hex (``f9003a69`` stored verbatim). capstone and Triton both
want little-endian memory order; feeding them the MSB-first form makes them
disassemble *a different instruction* or fail outright (e.g. ``f9003a69`` →
nothing; reversed ``693a00f9`` → ``str x9, [x19, #0x70]``). That looked like
"Triton doesn't support this opcode" but was pure byte-order — a config/feed bug,
NOT an escape-hatch (un-modeled-opcode) scenario.

The cure is a **single canonical order = little-endian memory order**, applied
once at the trace READERS (the source), so every downstream consumer — the Triton
/ capstone decode feeds AND every ``bytes_.hex()`` consumer (the SemanticsTable
key, dispatch-coverage opcodes, findings) — sees one order. No site reverses a
second time (there is no per-feed reversal to begin with), so there is no
double-reversal risk.

Orientation is a property of the **trace/window CONVENTION, not of an isolated
instruction.** A blind reverse is wrong (an already-LE word can spuriously decode
in the WRONG order too — ``b8696835`` decodes as ``cbnz`` as-is yet the correct
instruction is the reversed ``ldr``), and a per-instruction head-match against the
recorded mnemonic is ALSO insufficient: AArch64 has pervasive **alias mnemonics**
where capstone's canonical spelling differs from the trace's recorded spelling
(trace ``orr w6,wzr,#0x20`` vs capstone ``mov w6,#0x20``; also cmp/subs, cmn/adds,
tst/ands, neg/sub, mov/movz/movn, lsl|lsr|asr/ubfm|sbfm, ror/extr, …). For such an
instruction NEITHER orientation's head matches the recorded mnemonic, so a
per-instruction oracle would leave it as-stored — i.e. MSB-first on a uniformly
MSB-first trace — which then either makes the §2 guard false-BLOCK a recoverable
trace or feeds silently-wrong bytes into symex.

So we decide the convention by an **oracle MAJORITY VOTE** over the *decidable*
instructions (those where exactly one orientation's head matches the recorded
mnemonic), then apply that one convention UNIFORMLY across the window. An
alias / undecidable instruction **inherits** the window convention rather than
being left as-stored. A correctly-stored LE trace votes "as-stored" → it is
untouched (idempotent). A window with no clear majority among decidable
instructions (a genuinely garbled / mixed feed) is left as-stored, and the §2
``decode_audit`` guard is the backstop — a systematic mismatch BLOCKs rather than
risk a false flip. capstone is an optional dependency (no capstone → leave
as-stored, guard backstops); never raises.

Entry points:
  * :func:`canonical_aarch64_bytes` — pure per-call normalizer, kept for callers
    that hold a single instruction (it still does the head-match disambiguation,
    but cannot recover an isolated alias instruction — only the window-level path
    can). The trace readers do NOT use this directly.
  * :class:`ConventionDetector` / :func:`detect_convention` — vote a convention
    over a sample of instructions.
  * :func:`normalize_window` — apply a window's voted convention to a list of
    ``(raw_bytes, mnemonic)`` pairs (or the reader helper below).

Streaming impact: ``TraceReader`` is a streaming, file-backed iterator that
re-opens per ``__iter__``. The readers do a cheap **two-pass** over the file: a
first pass votes the convention on the first N decidable instructions (bounded
memory — it stops once it has enough decidable evidence), caches the decision,
then a second pass yields instructions with the cached convention applied. No
whole-trace in-memory buffering.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable


# Lazy import so the engine package keeps working when capstone isn't installed
# (engine-vs-fixture boundary — same pattern as stages/s3_triton_symex.py).
try:  # pragma: no cover — exercised only on hosts with capstone
    import capstone as _capstone  # type: ignore
    _CAPSTONE_OK = True
    _CAPSTONE_IMPORT_ERR: str | None = None
    _CS = _capstone.Cs(_capstone.CS_ARCH_ARM64, _capstone.CS_MODE_LITTLE_ENDIAN)
except Exception as _e:  # ImportError or any binding-load error
    _capstone = None  # type: ignore
    _CAPSTONE_OK = False
    _CAPSTONE_IMPORT_ERR = f"{type(_e).__name__}: {_e}"
    _CS = None


# AArch64 is a fixed-width 4-byte ISA; byte-order normalization only applies to a
# 4-byte word. Anything else (variable-length / unexpected slice) is left as-is —
# we do not guess a swap we can't justify.
AARCH64_INSN_BYTES = 4

# How many DECIDABLE instructions a convention vote needs before it stops sampling
# (bounded memory for the streaming readers). A handful of unambiguous non-alias
# instructions is plenty to fix the window orientation.
CONVENTION_VOTE_SAMPLE = 64


def capstone_available() -> bool:
    """True iff capstone bindings imported cleanly at module load."""
    return _CAPSTONE_OK


def capstone_unavailable_reason() -> str | None:
    """When :func:`capstone_available` is False, the import error text; else None."""
    return None if _CAPSTONE_OK else _CAPSTONE_IMPORT_ERR


def capstone_mnemonic(raw: bytes) -> str | None:
    """Disassemble one AArch64 word with capstone; return ``"mnem op_str"`` lower-
    cased, or ``None`` when capstone is unavailable or can't decode these bytes.

    The oracle the §2 guard cross-checks Triton against — capstone decodes the
    *bulk* of a real trace, so a Triton-failed-but-capstone-succeeded step is a
    feed bug, not a true blind spot."""
    if not _CAPSTONE_OK or _CS is None:
        return None
    try:
        for insn in _CS.disasm(bytes(raw), 0):
            text = f"{insn.mnemonic} {insn.op_str}".strip()
            return text.lower()
    except Exception:
        return None
    return None


def _heads_match(disasm: str | None, mnemonic: str | None) -> bool:
    """True iff a capstone disasm string and a recorded trace mnemonic name the
    same instruction (leading-token compare — operand syntax/spacing differs
    across emitters, but the mnemonic head is stable)."""
    if not disasm or not mnemonic:
        return False
    a = disasm.split()
    b = str(mnemonic).split()
    if not a or not b:
        return False
    return a[0].lower() == b[0].lower()


class Convention(enum.Enum):
    """The byte-order convention a trace/window is stored in."""
    AS_STORED = "as_stored"   # already canonical little-endian — leave untouched
    REVERSED = "reversed"     # MSB-first — every word must be 4-byte-reversed
    UNKNOWN = "unknown"       # no clear majority / no oracle — leave as-stored


def _vote_one(raw: bytes, mnemonic: str | None) -> Convention | None:
    """Vote a single instruction's orientation, or ``None`` if it is undecidable.

    Decidable iff EXACTLY one orientation's capstone head matches the recorded
    mnemonic. An alias instruction (neither head matches the recorded alias
    spelling) or one that decodes the same family both ways is undecidable and
    casts no vote — it inherits the window convention later."""
    if len(raw) != AARCH64_INSN_BYTES or not _CAPSTONE_OK or not mnemonic:
        return None
    rev = raw[::-1]
    asis_ok = _heads_match(capstone_mnemonic(raw), mnemonic)
    rev_ok = _heads_match(capstone_mnemonic(rev), mnemonic)
    if asis_ok and not rev_ok:
        return Convention.AS_STORED
    if rev_ok and not asis_ok:
        return Convention.REVERSED
    return None  # neither, or both (palindromic/ambiguous) — undecidable


def detect_convention(
    instructions: Iterable[tuple[bytes, str | None]],
    *,
    sample: int = CONVENTION_VOTE_SAMPLE,
) -> Convention:
    """Vote the window/trace byte-order convention over a sample of instructions.

    ``instructions`` yields ``(raw_bytes, recorded_mnemonic)`` pairs. Tallies the
    decidable ones (see :func:`_vote_one`) until ``sample`` decidable votes are
    seen, then returns the majority:

      * capstone unavailable / no decidable votes → ``UNKNOWN`` (leave as-stored);
      * a CLEAR majority (winner > 0 and at least twice the loser) → that
        convention, applied uniformly (an alias instruction inherits it);
      * otherwise (tie / mixed feed with no clear winner) → ``UNKNOWN`` so we do
        NOT false-flip a garbled window — the §2 guard BLOCKs it instead.

    Pure aside from the capstone handle; never raises."""
    if not _CAPSTONE_OK:
        return Convention.UNKNOWN
    asis = rev = 0
    for raw, mnem in instructions:
        v = _vote_one(bytes(raw), mnem)
        if v is Convention.AS_STORED:
            asis += 1
        elif v is Convention.REVERSED:
            rev += 1
        if asis + rev >= sample:
            break
    if asis == 0 and rev == 0:
        return Convention.UNKNOWN
    if asis > rev and asis >= 2 * rev:
        return Convention.AS_STORED
    if rev > asis and rev >= 2 * asis:
        return Convention.REVERSED
    return Convention.UNKNOWN  # no clear majority — don't false-flip


class ConventionDetector:
    """Holds a voted :class:`Convention` and applies it uniformly to each word.

    Build via :meth:`from_samples` (votes over an instruction sample), then call
    :meth:`apply` per instruction. ``apply`` reverses every 4-byte word iff the
    convention is ``REVERSED`` — INCLUDING alias / undecidable instructions, which
    is the whole point: they inherit the window convention instead of being left
    MSB-first. ``AS_STORED`` / ``UNKNOWN`` leave bytes untouched (idempotent)."""

    __slots__ = ("convention",)

    def __init__(self, convention: Convention):
        self.convention = convention

    @classmethod
    def from_samples(
        cls,
        instructions: Iterable[tuple[bytes, str | None]],
        *,
        sample: int = CONVENTION_VOTE_SAMPLE,
    ) -> "ConventionDetector":
        return cls(detect_convention(instructions, sample=sample))

    def apply(self, raw: bytes) -> bytes:
        b = bytes(raw)
        if len(b) != AARCH64_INSN_BYTES:
            return b
        if self.convention is Convention.REVERSED:
            return b[::-1]
        return b  # AS_STORED / UNKNOWN — canonical already, or don't guess


def normalize_window(
    instructions: Iterable[tuple[bytes, str | None]],
    *,
    sample: int = CONVENTION_VOTE_SAMPLE,
) -> list[bytes]:
    """Normalize a whole window of ``(raw_bytes, mnemonic)`` pairs to canonical LE.

    Votes the convention over the (materialized) window then applies it uniformly,
    so an alias instruction whose individual head-match is inconclusive is still
    flipped along with its non-alias neighbours. For the streaming readers prefer
    the two-pass :class:`ConventionDetector` over a re-openable file (this helper
    materializes the input list)."""
    items = [(bytes(r), m) for r, m in instructions]
    det = ConventionDetector.from_samples(items, sample=sample)
    return [det.apply(r) for r, _ in items]


def canonical_aarch64_bytes(raw: bytes, mnemonic: str | None = None) -> bytes:
    """Normalize ONE instruction's raw bytes to canonical little-endian order.

    Pure per-call path, kept for callers that hold a single instruction. It uses
    the recorded ``mnemonic`` as a capstone oracle to disambiguate orientation
    (of as-stored vs the 4-byte reverse, pick the head that matches the recorded
    mnemonic). NOTE: an isolated **alias** instruction cannot be oriented this way
    (capstone says ``mov``, the trace says ``orr`` — neither head matches), so for
    aliases this leaves bytes as-stored; only the window-level path
    (:class:`ConventionDetector` / :func:`normalize_window`, which the readers use)
    can recover an alias by inheriting the window convention. The §2 guard
    backstops the leftover case.

    Resolution order:
      1. not a 4-byte word → return unchanged (no justified swap);
      2. capstone unavailable → return unchanged (the §2 guard backstops);
      3. recorded mnemonic given → the orientation whose head matches it
         (prefer as-stored on a tie so a correctly-stored LE trace is untouched);
      4. no mnemonic → the orientation that decodes at all (prefer as-stored);
      5. neither decodes / both ambiguous (incl. aliases) → return unchanged.

    Pure aside from the module-level capstone handle; never raises."""
    b = bytes(raw)
    if len(b) != AARCH64_INSN_BYTES:
        return b
    if not _CAPSTONE_OK:
        return b
    asis_d = capstone_mnemonic(b)
    if mnemonic:
        # Common path (an already-LE trace): as-stored matches → keep it, and skip
        # the reverse disasm entirely (tie → prefer as-stored, so normalization is
        # idempotent and never gratuitously swaps a correctly-stored word).
        if _heads_match(asis_d, mnemonic):
            return b
        rev = b[::-1]
        if _heads_match(capstone_mnemonic(rev), mnemonic):
            return rev               # stored MSB-first → reverse to canonical LE
        # Neither matches the recorded mnemonic (alias, or both ambiguous): leave
        # as-stored. An isolated call can't break this tie — the window path does.
        return b
    # No recorded mnemonic — fall back to "the orientation that decodes".
    if asis_d is not None:
        return b
    rev = b[::-1]
    if capstone_mnemonic(rev) is not None:
        return rev
    return b


__all__ = [
    "AARCH64_INSN_BYTES",
    "CONVENTION_VOTE_SAMPLE",
    "Convention",
    "ConventionDetector",
    "capstone_available",
    "capstone_unavailable_reason",
    "capstone_mnemonic",
    "canonical_aarch64_bytes",
    "detect_convention",
    "normalize_window",
]
