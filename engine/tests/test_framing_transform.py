"""FramingTransform — the parameterised instance that fills the CVD Transform
slot (cvd.py §10.1 encoding edge, previously interface-only / §10.6 Deferred).

Coverage proves ZERO case-fit: F0's framing (prefix="BEFOR", nested base64,
drop-5, bit-level) is ONE parameter combination; plain base64, base64url, hex,
prefixed, nested, and bit-shifted (drop-N) framings are all the same class with
different parameters. Plus: the load-bearing bit-level (sub-byte) drop, the
degenerate "unrecognized framing" raise, registry wiring, and round-trip
(forward∘inverse == identity).
"""

from __future__ import annotations

import base64
import os

import pytest

from engine.cvd import Registry, Transform, default_registry
from engine.framing_transform import FramingTransform, default_transforms


# --- form 1: plain base64, no prefix (the simplest framing) ------------------

def test_plain_base64_no_prefix():
    t = FramingTransform(encoding="base64")
    raw = b"hello world payload"
    framed = base64.b64encode(raw)
    assert t.detect(framed) is True
    assert t.inverse(framed) == raw
    assert t.forward(raw) == framed


def test_plain_base64url_no_prefix():
    t = FramingTransform(encoding="base64url")
    raw = bytes(range(40))                  # contains bytes that map to -/_ vs +/
    framed = base64.urlsafe_b64encode(raw)
    assert t.detect(framed) is True
    assert t.inverse(framed) == raw


def test_plain_hex():
    t = FramingTransform(encoding="hex")
    raw = b"\x00\xde\xad\xbe\xef"
    framed = t.forward(raw)
    assert framed == b"00deadbeef"
    assert t.detect(framed) is True
    assert t.inverse(framed) == raw


# --- form 2: prefixed (F0's "BEFOR") -----------------------------------------

def test_prefixed_nested_base64_f0_shape():
    # F0: body = base64("BEFOR" + base64(raw))  (byte-level prefix view)
    raw = os.urandom(89)
    framed = base64.b64encode(b"BEFOR" + base64.b64encode(raw))
    t = FramingTransform(encoding="base64", prefix=b"BEFOR", nesting=True)
    assert t.detect(framed) is True
    assert t.inverse(framed) == raw                      # raw recovered, no hand-RE


def test_prefix_is_a_parameter_not_baked_in():
    # the SAME class with a DIFFERENT prefix handles a different framing.
    raw = os.urandom(33)
    framed = base64.b64encode(b"HDR9" + base64.b64encode(raw))
    t = FramingTransform(encoding="base64", prefix=b"HDR9", nesting=True)
    assert t.inverse(framed) == raw
    # the F0-tuned transform must NOT claim this one
    t_f0 = FramingTransform(encoding="base64", prefix=b"BEFOR", nesting=True)
    assert t_f0.detect(framed) is False


def test_string_prefix_is_normalised_to_bytes():
    t = FramingTransform(encoding="base64", prefix="PFX", nesting=True)
    assert t.prefix == b"PFX"
    raw = os.urandom(20)
    framed = base64.b64encode(b"PFX" + base64.b64encode(raw))
    assert t.inverse(framed) == raw


# --- form 3: bit-level drop-N (the hard point: sub-byte shift) ---------------

@pytest.mark.parametrize("drop_chars", [1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("length", [1, 3, 16, 89, 128])
def test_bit_level_drop_round_trip(drop_chars, length):
    # drop_chars * 6 bits removed at the SYMBOL level; for drop in {1,2,3,5,6,7}
    # that is NOT a byte multiple -> survivors are bit-shifted.
    raw = os.urandom(length)
    t = FramingTransform(encoding="base64", drop_chars=drop_chars, bit_level=True)
    assert t.inverse(t.forward(raw)) == raw


def test_bit_level_drop5_is_not_byte_aligned():
    # F0's drop=5 -> 30 bits -> 30 % 8 == 6 (a genuine sub-byte shift).
    assert (5 * 6) % 8 == 6
    raw = os.urandom(89)
    t = FramingTransform(encoding="base64", drop_chars=5, bit_level=True)
    framed = t.forward(raw)
    assert t.inverse(framed) == raw


def test_bit_level_differs_from_byte_level():
    # Prove bit-level actually bit-shifts: for a non-byte-aligned drop the
    # bit-level result is NOT the same as a naive whole-symbol byte slice.
    raw = os.urandom(64)
    t_bit = FramingTransform(encoding="base64", drop_chars=5, bit_level=True)
    framed = t_bit.forward(raw)
    # naive byte-level decode of the SAME stream after dropping 5 chars: corrupt
    naive = FramingTransform(encoding="base64", drop_chars=5, bit_level=False)
    # naive treats the stream as outer-codec text; it will NOT recover raw
    try:
        wrong = naive.inverse(framed)
    except ValueError:
        wrong = b"<raised>"
    assert wrong != raw


@pytest.mark.parametrize("nesting", [False, True])
@pytest.mark.parametrize("encoding", ["base64", "base64url"])
def test_bit_level_nested_and_plain_round_trip(nesting, encoding):
    raw = os.urandom(89)
    t = FramingTransform(encoding=encoding, prefix=b"BEFOR",
                         drop_chars=5, bit_level=True, nesting=nesting)
    assert t.inverse(t.forward(raw)) == raw


# --- form 4 / degenerate: unrecognized framing -> explicit raise (A8④) -------

def test_unrecognized_framing_raises_not_silent():
    t = FramingTransform(encoding="base64", prefix=b"BEFOR", nesting=True)
    inner = base64.b64encode(os.urandom(20))
    wrong = base64.b64encode(b"AFTER" + inner)           # wrong prefix
    with pytest.raises(ValueError, match="unrecognized framing; cannot transcode"):
        t.inverse(wrong)


def test_non_alphabet_bytes_raise():
    t = FramingTransform(encoding="base64", drop_chars=1, bit_level=True)
    with pytest.raises(ValueError, match="unrecognized framing; cannot transcode"):
        t.inverse(b"\x00\x01\x02 not base64 !!!")


def test_unknown_encoding_rejected_at_construction():
    with pytest.raises(ValueError, match="unrecognized framing; cannot transcode"):
        FramingTransform(encoding="rot13")


def test_detect_rejects_non_matching_region():
    t = FramingTransform(encoding="base64", prefix=b"BEFOR", nesting=True)
    assert t.detect(b"") is False
    assert t.detect(b"\xff\xfe\x00 raw bytes") is False
    # a valid base64 stream whose decode lacks the prefix is also rejected
    assert t.detect(base64.b64encode(b"no prefix here")) is False


def test_drop_exceeds_stream_raises():
    t = FramingTransform(encoding="base64", drop_chars=100, bit_level=True)
    framed = base64.b64encode(b"short")
    with pytest.raises(ValueError, match="unrecognized framing; cannot transcode"):
        t.inverse(framed)


def test_negative_drop_rejected():
    with pytest.raises(ValueError):
        FramingTransform(encoding="base64", drop_chars=-1)


# --- registry wiring: the §10.6 slot now has instances (deferred -> active) ---

def test_default_registry_has_framing_transforms():
    reg = default_registry()
    assert len(reg.transforms) >= 2
    assert all(isinstance(t, Transform) for t in reg.transforms)
    encodings = {t.encoding for t in reg.transforms}
    assert {"base64", "base64url"} <= encodings


def test_framing_transform_is_registrable_via_register_path():
    # additive registration through the generic Transform path (cvd.py:259)
    reg = Registry()
    tf = FramingTransform(encoding="base64", prefix=b"BEFOR",
                          nesting=True, drop_chars=5, bit_level=True)
    reg.register(tf)
    assert reg.transforms == [tf]
    # registering a Transform must not touch the other slots (invariant 7)
    assert reg.verifiers == [] and reg.generators == [] and reg.rules == []


def test_default_transforms_are_general_not_case_fit():
    # the default set carries NO F0-specific prefix / drop / bit-level values
    for t in default_transforms():
        assert t.prefix == b""
        assert t.drop_chars == 0
        assert t.bit_level is False


# --- round-trip identity across the whole parameter space --------------------

@pytest.mark.parametrize("encoding", ["base64", "base64url", "hex"])
@pytest.mark.parametrize("prefix", [b"", b"BEFOR", b"X"])
@pytest.mark.parametrize("nesting", [False, True])
def test_byte_level_round_trip_matrix(encoding, prefix, nesting):
    raw = os.urandom(45)
    t = FramingTransform(encoding=encoding, prefix=prefix, nesting=nesting)
    assert t.inverse(t.forward(raw)) == raw
