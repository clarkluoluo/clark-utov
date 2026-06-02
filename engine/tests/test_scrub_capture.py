"""Scrub Capture: recover a write-then-wiped transient secret (TOOL_SCRUB_CAPTURE.md).

Synthetic traces only. Covers the unknown-target detector (recover the pre-wipe
value; noise control rejects benign slot reuse) and the CVD-Plus integration
(register the tool, driver unchanged → recovered secret confirmed against the
oracle → SUCCESS).
"""

from __future__ import annotations

from engine.cvd import CvdOutcome, Registry, default_registry, run_cvd
from engine.scrub_capture import register_scrub, scrub_capture
from engine.types import Instruction, MemOp


def _ins(idx, mnem, mem=()):
    return Instruction(idx=idx, pc=0x70000 + idx * 4, bytes_=b"\x00\x00\x00\x00",
                       mnemonic=mnem, regs_read={}, regs_write={}, mem=mem)


def _le(b):
    return int.from_bytes(b, "little")


SECRET = bytes([0x9f, 0x3c, 0xe1, 0x42])     # high-entropy, non-ASCII
STACK = 0xbef00                              # a "stack" slot


# --- the unknown-target detector --------------------------------------------

def test_recovers_pre_wipe_secret_from_write_then_wipe():
    # secret written to a stack slot at idx0, zero-wiped at idx1.
    trace = [
        _ins(0, "str x8, [sp, #16]", mem=(MemOp("w", STACK, _le(SECRET), 4),)),  # place
        _ins(5, "str xzr, [sp, #16]", mem=(MemOp("w", STACK, 0, 4),)),           # wipe
    ]
    found = scrub_capture(trace)
    assert len(found) == 1
    sc = found[0]
    assert sc.value == SECRET                 # the pre-wipe value, recovered
    assert sc.region == (STACK, 4)
    assert sc.wrote_at[0] == 0 and sc.wiped_at[0] == 5
    assert sc.entropy > 0


def test_benign_overwrite_is_not_flagged():
    # printable "ABCD" overwritten by "EFGH" — not a wipe, not a secret.
    trace = [
        _ins(0, "str x8, [x9]", mem=(MemOp("w", STACK, _le(b"ABCD"), 4),)),
        _ins(1, "str x8, [x9]", mem=(MemOp("w", STACK, _le(b"EFGH"), 4),)),
    ]
    assert scrub_capture(trace) == []


def test_constant_value_overwritten_is_not_a_secret():
    # V_old is a constant fill (low entropy) -> not a secret, even if wiped.
    trace = [
        _ins(0, "str x8, [x9]", mem=(MemOp("w", STACK, _le(bytes([0x11] * 4)), 4),)),
        _ins(1, "str xzr, [x9]", mem=(MemOp("w", STACK, 0, 4),)),
    ]
    assert scrub_capture(trace) == []


def test_long_lifetime_not_transient():
    trace = [
        _ins(0, "str x8, [x9]", mem=(MemOp("w", STACK, _le(SECRET), 4),)),
        _ins(500, "str xzr, [x9]", mem=(MemOp("w", STACK, 0, 4),)),   # wiped far later
    ]
    assert scrub_capture(trace, lifetime_window=64) == []


def test_stack_band_filters_non_stack_writes():
    trace = [
        _ins(0, "str x8, [x9]", mem=(MemOp("w", 0x4000, _le(SECRET), 4),)),  # not stack
        _ins(1, "str xzr, [x9]", mem=(MemOp("w", 0x4000, 0, 4),)),
    ]
    assert scrub_capture(trace, stack_band=(0xb0000, 0xc0000)) == []   # outside band


def test_ranked_by_entropy():
    lo = bytes([0x10, 0x11, 0x10, 0x11])       # 2 distinct -> below ratio, not secret
    hi = SECRET
    trace = [
        _ins(0, "str x8, [x9]", mem=(MemOp("w", 0xbe000, _le(hi), 4),)),
        _ins(1, "str xzr, [x9]", mem=(MemOp("w", 0xbe000, 0, 4),)),
        _ins(2, "str x8, [x10]", mem=(MemOp("w", 0xbe100, _le(lo), 4),)),
        _ins(3, "str xzr, [x10]", mem=(MemOp("w", 0xbe100, 0, 4),)),
    ]
    found = scrub_capture(trace)
    assert [sc.value for sc in found] == [hi]   # only the high-entropy one qualifies


# --- CVD-Plus integration: register the tool, driver unchanged --------------

def test_scrub_tool_registered_recovers_secret_as_success():
    # the run's oracle IS the secret; the scrub tool recovers it and the
    # ScrubVerifier confirms it against the oracle -> SUCCESS(RECOVERED_SECRET).
    trace = [
        _ins(0, "str x8, [sp, #16]", mem=(MemOp("w", STACK, _le(SECRET), 4),)),
        _ins(5, "str xzr, [sp, #16]", mem=(MemOp("w", STACK, 0, 4),)),
    ]
    reg = register_scrub(default_registry())     # driver code unchanged
    res = run_cvd(trace, SECRET, registry=reg)
    assert res.outcome is CvdOutcome.SUCCESS
    assert res.verdict == "RECOVERED_SECRET"
    assert res.sink_base == STACK


def test_scrub_recovered_value_not_matching_oracle_is_not_trusted():
    # the recovered secret != the run's oracle -> eliminated as unconfirmed,
    # reported in the log (ranked candidate), never asserted.
    trace = [
        _ins(0, "str x8, [sp, #16]", mem=(MemOp("w", STACK, _le(SECRET), 4),)),
        _ins(5, "str xzr, [sp, #16]", mem=(MemOp("w", STACK, 0, 4),)),
    ]
    other = bytes([0x01, 0x02, 0x03, 0x04])
    reg = Registry()
    register_scrub(reg)                          # only the scrub tool registered
    res = run_cvd(trace, other, registry=reg)
    assert any(e.get("event") == "ELIMINATED"
               and e.get("reason") == "unconfirmed_secret" for e in res.log)
