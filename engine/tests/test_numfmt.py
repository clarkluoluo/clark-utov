"""capability_request.md §P2-1 — number-formatting helpers."""

from __future__ import annotations

import pytest

from engine.numfmt import as_dec, as_hex, normalize_record, parse_explicit


def test_as_hex_default():
    assert as_hex(0x40) == "0x40"
    assert as_hex(0) == "0x0"
    assert as_hex(0xDEAD) == "0xdead"  # lowercase


def test_as_hex_width_pads():
    assert as_hex(0x40, width=8) == "0x00000040"


def test_as_hex_rejects_non_int():
    with pytest.raises(TypeError):
        as_hex("0x40")  # type: ignore[arg-type]


def test_as_dec_basic():
    assert as_dec(32) == "32"
    assert as_dec(0) == "0"


def test_parse_explicit_base16_accepts_both_prefix_forms():
    assert parse_explicit("0x40", base=16) == 0x40
    assert parse_explicit("40",   base=16) == 0x40


def test_parse_explicit_base10_rejects_hex_prefix():
    """The whole point of P2-1: '20' must not silently decode as 0x20."""
    with pytest.raises(ValueError, match="looks like hex"):
        parse_explicit("0x20", base=10)


def test_parse_explicit_unsupported_base_rejected():
    with pytest.raises(ValueError):
        parse_explicit("10", base=2)


def test_normalize_record_uses_suffix_conventions():
    row = {
        "pc_addr":      0x40007d88,
        "trace_idx":    1234,
        "byte_count":   32,
        "free_text":    "hi",
    }
    out = normalize_record(row)
    # _addr → hex
    assert out["pc_addr"] == "0x40007d88"
    # _idx → dec
    assert out["trace_idx"] == "1234"
    # _count → dec
    assert out["byte_count"] == "32"
    # str passes through
    assert out["free_text"] == "hi"


def test_normalize_record_does_not_mutate_input():
    row = {"foo_pc": 0x100}
    _ = normalize_record(row)
    assert row["foo_pc"] == 0x100   # still int
