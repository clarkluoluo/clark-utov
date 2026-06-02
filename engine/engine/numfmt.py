"""Number-formatting helpers (capability_request.md §P2-1).

`utov` ledger / calltrace / report output has been bitten by ambiguous
numerics — "20" read as decimal when it was hex 0x20=32 nearly poisoned
the random_007 write-back model on the reference target run. From now on every
number that lands in agent-facing text (calltrace columns, hook PCs, mem
addresses, register values, sizes) goes through one of the helpers here
so the encoding is unambiguous.

Rules:
  - PC / mem-addr / register-value: ``as_hex(0x1234) == "0x1234"`` —
    always lowercase, ``0x`` prefix, no padding.
  - byte-count / index / count: ``as_dec(32) == "32"`` — bare decimal.
  - ambiguous (e.g. a fingerprint cell parser hands you a string and you
    don't know which base it was written with): use ``as_either(s)``
    which REQUIRES the caller to supply ``base=`` and refuses to guess.

This is intentionally tiny — the value is in eliminating ``"20" decimal
vs hex`` ambiguity at the SOURCE, not in clever formatting.
"""

from __future__ import annotations

from typing import Any


def as_hex(value: int, *, width: int | None = None) -> str:
    """PC / mem addr / register value → ``0xNN`` lowercase.

    ``width`` pads with leading zeros to that many hex digits (no ``0x``
    counted). ``as_hex(0x40, width=8)`` → ``"0x00000040"``. Default is no
    pad (matches the original engine output).
    """
    if not isinstance(value, int):
        raise TypeError(f"as_hex() requires int; got {type(value).__name__}")
    if value < 0:
        return f"-0x{(-value):0{width or 0}x}"
    if width:
        return f"0x{value:0{width}x}"
    return f"0x{value:x}"


def as_dec(value: int) -> str:
    """Counts / indices / sizes → bare decimal string (no prefix)."""
    if not isinstance(value, int):
        raise TypeError(f"as_dec() requires int; got {type(value).__name__}")
    return str(value)


def parse_explicit(s: str, *, base: int) -> int:
    """Strict parse: caller MUST state the base. Refuses to guess.

    Lets call-sites that pull numbers out of legacy tables (calltrace
    columns, CSV cells, parquet text fields) record their decoding choice
    in the call rather than relying on ``int(s, 0)``'s magic-prefix
    behaviour. ``base=16`` accepts both ``"0x40"`` and ``"40"``;
    ``base=10`` rejects ``"0x40"``.
    """
    if base not in (10, 16):
        raise ValueError(f"only base=10 or base=16 supported; got {base!r}")
    s = s.strip()
    if base == 16:
        if s.lower().startswith("0x"):
            return int(s, 16)
        return int(s, 16)
    # base == 10
    if s.lower().startswith("0x"):
        raise ValueError(
            f"parse_explicit(base=10) refusing to parse {s!r} — looks like hex"
        )
    return int(s, 10)


def normalize_record(row: dict[str, Any]) -> dict[str, Any]:
    """Project a mixed-encoding row to canonical strings.

    Conventions used by the engine output layer:
      - keys ending in ``_pc``, ``_addr``, ``_rva``, ``_value`` → hex
      - keys ending in ``_count``, ``_len``, ``_size``, ``_idx`` → dec
      - other int values pass through unchanged

    A dict (not in-place) is returned so the caller decides whether to
    overwrite ``row``.
    """
    HEX_SUFFIXES = ("_pc", "_addr", "_rva", "_value", "_offset")
    DEC_SUFFIXES = ("_count", "_len", "_size", "_idx", "_index")
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, int) and not isinstance(v, bool):
            if any(k.endswith(suf) for suf in HEX_SUFFIXES):
                out[k] = as_hex(v)
                continue
            if any(k.endswith(suf) for suf in DEC_SUFFIXES):
                out[k] = as_dec(v)
                continue
        out[k] = v
    return out
