"""FramingTransform — a parameterised instance of the CVD ``Transform`` interface
(cvd.py §10.1 encoding edge), filling the slot that was interface-only / Deferred
(cvd.py docstring §10.6: "interfaces exist … but no non-trivial instances").

Background (F0, 2026-06-01 compose): the final body was
``base64("BEFOR" + base64(raw))`` and ``raw`` did NOT appear byte-aligned anywhere
in the captured base64 *stream*, because recovering it from the stream means
dropping a 5-char prefix at the base64-symbol level — 5 base64 chars = 30 bits,
which is NOT a byte multiple, so the remaining symbols decode with a sub-byte
**bit shift**. The agent hand-RE'd that bit map. This module makes any such framing
(prefix / encoding / nesting / drop-N / bit-vs-byte level) a *parameter*, so the
driver's transform-aware retreat (cvd.py §10.6) has a real instance and no future
case has to hand-RE the bit map.

ZERO case-fit: F0's concrete values (``"BEFOR"``, drop 5, base64, nested,
bit-level) are NOT in this code — they are constructor parameters. A plain
base64 stream with no prefix is the same class with different params.

Honest degenerate (A8④): a region that does not match the configured framing is
NOT silently guessed at — :meth:`detect` returns False and :meth:`inverse` raises
``ValueError("unrecognized framing; cannot transcode")``.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .cvd import Transform

if TYPE_CHECKING:                       # avoid import cycle at runtime
    from .cvd import CvdState


# --- base64 symbol alphabet (the 6-bit code) --------------------------------

_STD_ALPHABET = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
_URL_ALPHABET = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"

_ENCODINGS = ("base64", "base64url", "hex")


def _alphabet_for(encoding: str) -> bytes:
    return _URL_ALPHABET if encoding == "base64url" else _STD_ALPHABET


# --- bit-level base64 core (the hard point) ---------------------------------
#
# A base64 stream is a sequence of 6-bit symbols (MSB-first). Concatenating those
# 6-bit groups gives one continuous bit string; the decoder re-groups it into
# 8-bit bytes (MSB-first). When you DROP a prefix of N symbols you remove 6*N
# bits — and if 6*N is not a multiple of 8 the surviving symbols no longer sit on
# byte boundaries. The only correct way to recover the raw bytes is to operate on
# the bit string, not on whole-byte slices.


def _symbols_to_bits(symbols: bytes, encoding: str) -> list[int]:
    """Expand base64 chars into the flat MSB-first bit list (6 bits / symbol)."""
    alphabet = _alphabet_for(encoding)
    index = {c: i for i, c in enumerate(alphabet)}
    bits: list[int] = []
    for c in symbols:
        if c == ord("="):               # padding carries no data bits
            continue
        if c not in index:
            raise ValueError(
                f"unrecognized framing; cannot transcode "
                f"(byte 0x{c:02x} not in {encoding} alphabet)")
        v = index[c]
        for shift in range(5, -1, -1):
            bits.append((v >> shift) & 1)
    return bits


def _bits_to_bytes(bits: list[int]) -> bytes:
    """Pack an MSB-first bit list into bytes, discarding a trailing partial group
    (< 8 bits) — that remainder is padding/shift slack, not a whole raw byte."""
    out = bytearray()
    n = (len(bits) // 8) * 8
    for i in range(0, n, 8):
        b = 0
        for j in range(8):
            b = (b << 1) | bits[i + j]
        out.append(b)
    return bytes(out)


def _bytes_to_bits(data: bytes) -> list[int]:
    bits: list[int] = []
    for b in data:
        for shift in range(7, -1, -1):
            bits.append((b >> shift) & 1)
    return bits


def _bits_to_symbols(bits: list[int], encoding: str) -> bytes:
    """Pack an MSB-first bit list into base64 symbols (6 bits each). A trailing
    partial group is right-padded with zero bits into one final symbol."""
    alphabet = _alphabet_for(encoding)
    out = bytearray()
    i = 0
    n = len(bits)
    while i < n:
        v = 0
        taken = 0
        for j in range(6):
            v = (v << 1) | (bits[i + j] if i + j < n else 0)
            taken += 1 if i + j < n else 0
        out.append(alphabet[v])
        i += 6
    return bytes(out)


# --- the parameterised Transform instance -----------------------------------

@dataclass
class FramingTransform(Transform):
    """Parameterised raw↔framed transcoder for a (possibly nested, possibly
    prefixed, possibly bit-shifted) text framing.

    Parameters (all of F0's specifics live here, never in code):

    * ``encoding``   — ``base64`` | ``base64url`` | ``hex`` (inner/outer codec).
    * ``prefix``     — literal bytes inserted before the inner blob in the framed
      form (F0: ``b"BEFOR"``); ``b""`` for a plain stream.
    * ``nesting``    — base64-of-base64: the framed form is the codec applied a
      second time over ``prefix + codec(raw)``.
    * ``drop_chars`` — number of leading *symbols* removed from the (inner) stream
      before decoding raw. For ``prefix`` this is ``len(prefix)``; it is exposed
      separately so a pure drop-N (prefix-less) stream is expressible.
    * ``bit_level``  — when True, ``drop_chars`` is applied at the 6-bit base64
      symbol level so a non-byte-aligned drop (6*N not a multiple of 8) is handled
      by bit-shifting the surviving symbols; when False it is a whole-byte/char
      slice after decode.
    """

    encoding: str = "base64"
    prefix: bytes = b""
    nesting: bool = False
    drop_chars: int = 0
    bit_level: bool = False
    name: str = "framing"
    version: str = "1"
    owner: str = "core"

    def __post_init__(self) -> None:
        if self.encoding not in _ENCODINGS:
            raise ValueError(
                f"unrecognized framing; cannot transcode "
                f"(unknown encoding {self.encoding!r})")
        if isinstance(self.prefix, str):
            self.prefix = self.prefix.encode()
        if self.drop_chars < 0:
            raise ValueError("drop_chars must be >= 0")

    # --- codec primitives ---------------------------------------------------

    def _codec_decode(self, data: bytes) -> bytes:
        try:
            if self.encoding == "hex":
                return binascii.unhexlify(data)
            if self.encoding == "base64url":
                return base64.urlsafe_b64decode(_pad_b64(data))
            return base64.b64decode(_pad_b64(data))
        except (binascii.Error, ValueError) as exc:
            raise ValueError(
                f"unrecognized framing; cannot transcode ({exc})") from exc

    def _codec_encode(self, data: bytes) -> bytes:
        if self.encoding == "hex":
            return binascii.hexlify(data)
        if self.encoding == "base64url":
            return base64.urlsafe_b64encode(data)
        return base64.b64encode(data)

    # --- detection ----------------------------------------------------------

    def detect(self, region: bytes, state: "CvdState | None" = None) -> bool:
        """True iff ``region`` looks like this framing: it is a valid stream in
        the configured alphabet and (if a prefix is configured) the decoded outer
        layer starts with that prefix. Never guesses — a region that does not fit
        returns False (the driver then leaves it alone / asks for a transform)."""
        if not region:
            return False
        try:
            symbols = _strip_ws(region)
            if self.encoding == "hex":
                outer = self._codec_decode(symbols)
            else:
                # must be drawn entirely from the symbol alphabet
                if not _all_in_alphabet(symbols, self.encoding):
                    return False
                outer = self._codec_decode(symbols)
        except ValueError:
            return False
        if self.nesting:
            # outer should itself be a (prefixed) stream in the same alphabet
            inner_region = outer[len(self.prefix):] if self.prefix else outer
            if self.prefix and not outer.startswith(self.prefix):
                return False
            return _all_in_alphabet(_strip_ws(inner_region), self.encoding)
        if self.prefix:
            return outer.startswith(self.prefix)
        return True

    # --- framed → raw -------------------------------------------------------

    def inverse(self, encoded: bytes) -> bytes:
        """Recover raw bytes from the framed output (F0: stream → drop prefix →
        bit-shift → raw). Raises on a region that does not fit the framing."""
        symbols = _strip_ws(encoded)
        if not symbols:
            raise ValueError("unrecognized framing; cannot transcode (empty)")

        # Bit-level: drop_chars symbols are removed at the 6-bit level. When
        # nested, the input is codec(inner_symbol_text), so decode the outer layer
        # first to recover the inner symbol stream; otherwise the input IS the
        # symbol stream and must NOT be outer-decoded (a bare 5-symbol stream is
        # not valid base64 to decode as a whole).
        if self.bit_level and self.encoding != "hex":
            stream = _strip_ws(self._codec_decode(symbols)) if self.nesting else symbols
            return self._bit_drop_decode(stream)

        # Non-bit-level: outer layer = whole-byte/char decode, then slice.
        inner_stream = self._codec_decode(symbols)
        if self.prefix:
            if not inner_stream.startswith(self.prefix):
                raise ValueError(
                    "unrecognized framing; cannot transcode "
                    "(configured prefix not present)")
            inner_stream = inner_stream[len(self.prefix):]
        if self.drop_chars:
            inner_stream = inner_stream[self.drop_chars:]
        if self.nesting:
            return self._codec_decode(_strip_ws(inner_stream))
        return inner_stream

    def _bit_drop_decode(self, stream: bytes) -> bytes:
        """Drop ``drop_chars`` symbols at the 6-bit level, then decode the rest.
        6*drop_chars bits are removed from the flat bit string; the survivors are
        re-grouped MSB-first into bytes. This is the load-bearing bit-shift path."""
        bits = _symbols_to_bits(_strip_ws(stream), self.encoding)
        drop_bits = 6 * self.drop_chars
        if drop_bits > len(bits):
            raise ValueError(
                "unrecognized framing; cannot transcode "
                "(drop exceeds stream length)")
        return _bits_to_bytes(bits[drop_bits:])

    # --- raw → framed -------------------------------------------------------

    def forward(self, raw: bytes) -> bytes:
        """Build the framed output from raw bytes (inverse of :meth:`inverse`),
        used by transform-aware retreat to re-frame a recovered raw window."""
        if self.bit_level and self.encoding != "hex":
            return self._bit_drop_encode(raw)

        inner = self._codec_encode(raw) if self.nesting else raw
        framed_inner = self.prefix + inner
        return self._codec_encode(framed_inner)

    def _bit_drop_encode(self, raw: bytes) -> bytes:
        """Inverse of :meth:`_bit_drop_decode`: prepend EXACTLY ``6*drop_chars``
        frame bits (the symbols :meth:`inverse` will drop), then re-symbolise the
        ``frame_bits + raw_bits`` stream. The frame region is seeded from
        ``prefix`` (its bits carry the leading symbols) and zero-extended to the
        full ``6*drop_chars`` width so forward∘inverse == identity on raw."""
        need = 6 * self.drop_chars
        frame_bits = _bytes_to_bits(self.prefix)[:need]
        frame_bits = frame_bits + [0] * (need - len(frame_bits))
        symbols = _bits_to_symbols(frame_bits + _bytes_to_bits(raw), self.encoding)
        # In nesting mode the inner symbol stream is treated as TEXT that the outer
        # codec wraps (outer = codec(inner_symbol_text)); on inverse we recover that
        # text and bit-drop-decode it ourselves — so the inner stream length is
        # unconstrained (no base64 ≡1-mod-4 rule applies to arbitrary text).
        return self._codec_encode(symbols) if self.nesting else symbols


# --- small helpers ----------------------------------------------------------

def _strip_ws(data: bytes) -> bytes:
    return bytes(b for b in data if b not in b" \t\r\n")


def _pad_b64(data: bytes) -> bytes:
    return data + b"=" * ((-len(data)) % 4)


def _all_in_alphabet(symbols: bytes, encoding: str) -> bool:
    if encoding == "hex":
        hexset = set(b"0123456789abcdefABCDEF")
        return len(symbols) > 0 and all(c in hexset for c in symbols)
    allowed = set(_alphabet_for(encoding)) | set(b"=")
    return len(symbols) > 0 and all(c in allowed for c in symbols)


def default_transforms() -> list[FramingTransform]:
    """The default-registered framing transforms. Kept deliberately general: a
    plain base64 stream (no prefix) and a plain base64url stream. Case-specific
    framings (a particular prefix / drop-N / bit-level) are constructed by the
    caller with the right parameters and registered additionally — they are not
    baked in here (zero case-fit)."""
    return [
        FramingTransform(encoding="base64"),
        FramingTransform(encoding="base64url"),
    ]


__all__ = ["FramingTransform", "default_transforms"]
