"""Byte-order feed fix + systematic decode-failure guard (dev-triton-feed-decode-guard).

Three things are pinned here, all from a REAL trace's bytes (example/task-libEncryptor/libs/arm64-v8a/trace.txt),
never hand-crafted opcodes:

  §1  source normalization → canonical little-endian: after the readers normalize,
      Triton's decode == capstone's on the real trace and feed_mismatch == 0.
  §2  the decode-feed guard: a byte-order-REVERSED window is a systematic feed bug
      (capstone decodes what Triton can't) → BLOCK + the fix-the-feed note, and the
      escape hatch / semantics-table is NOT taken.
  §0  the auto-seed edge: when every live-in register is concretely backed, the seed
      is reg_file − backed (not None → full reg_file); when that is empty too, BLOCK.

Triton is optional on the host; the §1/§2 real-Triton assertions are skipped when it
is unavailable, but the framework-level guard (DecodeAudit on a fake probe) and the
§0 seed fix run everywhere.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from engine.byte_order import (
    Convention,
    ConventionDetector,
    canonical_aarch64_bytes,
    capstone_available,
    capstone_mnemonic,
    detect_convention,
    normalize_window,
)
from engine.runner_client import JsonlTraceReader, UnidbgTextTraceReader
from engine.setup_symex import (
    CaseConfig,
    DriveResult,
    build_concrete_backing,
    drive,
)
from engine.setup_symex_runner import (
    DecodeAudit,
    audit_window_decode,
    triton_available,
)
from engine.types import Instruction, MemOp

REPO_ROOT = Path(__file__).resolve().parents[2]
VMP_TRACE = REPO_ROOT / "example" / "task-libEncryptor" / "libs" / "arm64-v8a" / "trace.txt"

requires_capstone = pytest.mark.skipif(
    not capstone_available(), reason="capstone not installed on host")
requires_triton = pytest.mark.skipif(
    not triton_available(), reason="Triton not installed on host")


def _real_window(lo: int, hi: int) -> list[Instruction]:
    """A real handler window from the VMP trace, by trace idx."""
    out: list[Instruction] = []
    for ins in UnidbgTextTraceReader(VMP_TRACE):
        if lo <= ins.idx <= hi:
            out.append(ins)
        if ins.idx > hi:
            break
    return out


# --- §1: source normalization is canonical + idempotent ---------------------

@requires_capstone
def test_real_trace_normalizes_to_capstone_decodable_bytes() -> None:
    # Every fed (non-branch) step in a real window decodes under capstone after
    # the reader normalized it — the canonical-LE order capstone/Triton want.
    win = _real_window(0, 200)
    assert win, "fixture window empty"
    undecodable = [
        ins for ins in win
        if capstone_mnemonic(bytes(ins.bytes_)) is None
    ]
    # A real AArch64 window decodes essentially in full; a handful of genuinely
    # capstone-unmodeled encodings is tolerable, a systematic miss is not.
    assert len(undecodable) <= 2, [ins.mnemonic for ins in undecodable]


@requires_capstone
def test_normalization_is_idempotent_on_already_le_trace() -> None:
    # example/task-libEncryptor/libs/arm64-v8a/trace.txt is already stored little-endian — re-normalizing a
    # bytes_ the reader produced must be a no-op (so phase_replay round-trips and
    # there is no double-reversal).
    for ins in _real_window(0, 120):
        again = canonical_aarch64_bytes(bytes(ins.bytes_), ins.mnemonic)
        assert again == bytes(ins.bytes_)


@requires_capstone
def test_msb_first_word_is_reversed_to_canonical_le() -> None:
    # The §★ evidence: a word stored MSB-first decodes to nothing as-is but to the
    # right instruction reversed. The normalizer, given the recorded mnemonic, must
    # pick the reversed (canonical-LE) orientation.
    # 0x f9003a69 stored MSB-first -> reversed 693a00f9 = str x9, [x19, #0x70].
    stored = bytes.fromhex("f9003a69")
    out = canonical_aarch64_bytes(stored, "str x9, [x19, #0x70]")
    assert out == stored[::-1]
    assert capstone_mnemonic(out) and capstone_mnemonic(out).startswith("str")
    # And an already-LE word with the same mnemonic family is left untouched.
    already_le = stored[::-1]
    assert canonical_aarch64_bytes(already_le, "str x9, [x19, #0x70]") == already_le


@requires_triton
@requires_capstone
def test_real_trace_feed_mismatch_is_zero_after_fix() -> None:
    # The acceptance bar: with the feed fixed, Triton decodes what capstone decodes
    # on the real trace -> feed_mismatch == 0 (no systematic decode/feed bug).
    win = _real_window(0, 400)
    audit = audit_window_decode(win, window=(0, 10_000_000), window_kind="idx")
    assert audit.total > 0
    assert audit.feed_mismatch == 0, audit.to_dict()
    assert not audit.systematic, audit.note


# --- §★ alias-on-MSB-first: window convention recovers aliases --------------

# Real/known LE words (reversed of the §★ table's MSB-first hex). The 4th is an
# ALIAS: capstone canonicalises it to `mov w6,#0x20` while the trace records the
# `orr w6,wzr,#0x20` spelling — neither head matches per-instruction, so only the
# WINDOW convention can orient it.
_ALIAS_WINDOW_LE: list[tuple[str, str]] = [
    (bytes.fromhex("f9003a69")[::-1].hex(), "str x9, [x19, #0x70]"),
    (bytes.fromhex("530a7eab")[::-1].hex(), "lsr w11, w21, #10"),
    (bytes.fromhex("b8696835")[::-1].hex(), "ldr w21, [x1, x9]"),
    (bytes.fromhex("321b03e6")[::-1].hex(), "orr w6, wzr, #0x20"),  # alias (mov)
]
_ALIAS_IDX = 3  # the alias instruction's position in the window


def _write_jsonl(path: Path, rows: list[tuple[str, str]], *, reverse: bool) -> Path:
    """Emit a JSONL trace; ``reverse`` stores each word MSB-first (4-byte flip)."""
    with path.open("w", encoding="utf-8") as f:
        for i, (le_hex, mnem) in enumerate(rows):
            raw = bytes.fromhex(le_hex)
            stored = raw[::-1] if reverse else raw
            f.write(json.dumps({
                "idx": i,
                "pc": f"0x{0x1000 + 4 * i:x}",
                "bytes": stored.hex(),
                "mnemonic": mnem,
                "regs_read": {},
                "regs_write": {},
            }) + "\n")
    return path


@requires_capstone
def test_alias_on_msb_first_recovers_via_window_convention(tmp_path) -> None:
    # §★ idx-63: a MSB-first window that INCLUDES an alias instruction. The reader's
    # two-pass convention vote (non-alias neighbours decide REVERSED) must flip ALL
    # words — including the alias one — to canonical-LE. The per-instruction oracle
    # left the alias MSB-first; the window path fixes it.
    path = _write_jsonl(tmp_path / "msb.jsonl", _ALIAS_WINDOW_LE, reverse=True)
    out = list(JsonlTraceReader(path))
    assert len(out) == len(_ALIAS_WINDOW_LE)
    # Every instruction — incl. the alias — comes back canonical-LE & capstone-decodable.
    for ins in out:
        assert capstone_mnemonic(bytes(ins.bytes_)) is not None, ins.mnemonic
    # The alias word specifically: canonical-LE == reversed-of-stored == capstone `mov`.
    alias = out[_ALIAS_IDX]
    assert bytes(alias.bytes_) == bytes.fromhex(_ALIAS_WINDOW_LE[_ALIAS_IDX][0])
    assert capstone_mnemonic(bytes(alias.bytes_)).startswith("mov")
    # The whole-window normalizer agrees (direct entry point).
    msb_pairs = [(bytes.fromhex(h)[::-1], m) for h, m in _ALIAS_WINDOW_LE]
    normed = normalize_window(msb_pairs)
    for le_hex, nb in zip([h for h, _ in _ALIAS_WINDOW_LE], normed):
        assert nb == bytes.fromhex(le_hex)


@requires_capstone
def test_guard_does_not_false_block_recoverable_msb_window(tmp_path) -> None:
    # The same MSB-first alias window, driven reader -> normalize -> audit. After the
    # reader normalizes via the window convention, capstone decodes everything, so a
    # Triton probe matching capstone yields feed_mismatch == 0 and NOT systematic —
    # the recoverable window is not mistaken for a feed bug / garbage.
    path = _write_jsonl(tmp_path / "msb.jsonl", _ALIAS_WINDOW_LE, reverse=True)
    win = list(JsonlTraceReader(path))
    audit = audit_window_decode(
        win, window=(0, 10_000_000), window_kind="idx",
        triton_probe=lambda code: capstone_mnemonic(bytes(code)) is not None)
    assert audit.total > 0
    assert audit.feed_mismatch == 0, audit.to_dict()
    assert audit.systematic is False, audit.note


@requires_capstone
def test_already_le_window_votes_as_stored_and_is_untouched(tmp_path) -> None:
    # An already-LE trace (incl. the alias) votes AS_STORED -> reader is a pure no-op
    # (idempotent): bytes_ equals the stored LE bytes verbatim, no gratuitous flip.
    path = _write_jsonl(tmp_path / "le.jsonl", _ALIAS_WINDOW_LE, reverse=False)
    assert detect_convention(
        [(bytes.fromhex(h), m) for h, m in _ALIAS_WINDOW_LE]) is Convention.AS_STORED
    for ins, (le_hex, _m) in zip(JsonlTraceReader(path), _ALIAS_WINDOW_LE):
        assert bytes(ins.bytes_) == bytes.fromhex(le_hex)


@requires_capstone
def test_garbled_no_majority_window_is_left_as_stored() -> None:
    # A window with NO consistent convention (each word's two orientations both fail
    # to match its recorded mnemonic, or the votes split) yields UNKNOWN -> bytes are
    # left as-stored so the §2 guard can BLOCK it, rather than a false flip.
    garbled = [
        (bytes.fromhex("deadbeef"), "totally bogus mnemonic"),
        (bytes.fromhex("00000000"), "also bogus"),
    ]
    det = ConventionDetector.from_samples(garbled)
    assert det.convention is Convention.UNKNOWN
    for raw, _m in garbled:
        assert det.apply(raw) == raw  # untouched


# --- §2: the systematic decode-failure guard --------------------------------

def test_decode_audit_flags_feed_mismatch_as_systematic() -> None:
    # A fake Triton probe that fails everything; capstone (oracle) decodes the real
    # bytes -> feed_mismatch > 0 -> systematic (the feed-bug discriminator), even at
    # a single mismatch. (Pure-framework: runs without real Triton.)
    if not capstone_available():
        pytest.skip("capstone not installed")
    win = _real_window(0, 40)
    audit = audit_window_decode(
        win, window=(0, 10_000_000), window_kind="idx",
        triton_probe=lambda _code: False)
    assert audit.feed_mismatch > 0
    assert audit.systematic
    assert "byte-FEED bug" in audit.note
    assert "do NOT S-expr fill" in audit.note


def test_decode_audit_clean_when_triton_matches_capstone() -> None:
    # Fake probe that decodes everything -> zero failures -> not systematic.
    if not capstone_available():
        pytest.skip("capstone not installed")
    win = _real_window(0, 40)
    audit = audit_window_decode(
        win, window=(0, 10_000_000), window_kind="idx",
        triton_probe=lambda _code: True)
    assert audit.decode_failed == 0
    assert audit.feed_mismatch == 0
    assert not audit.systematic
    assert audit.note == ""


def test_decode_audit_fail_rate_threshold() -> None:
    audit = DecodeAudit(total=100, decode_failed=6, feed_mismatch=0)
    assert audit.fail_rate == pytest.approx(0.06)
    assert audit.systematic            # 6% > 5% threshold, even with no oracle mismatch
    ok = DecodeAudit(total=100, decode_failed=5, feed_mismatch=0)
    assert not ok.systematic           # exactly at threshold is not over it


@requires_capstone
@requires_triton
def test_drive_blocks_on_byte_order_reversed_window_no_escape_hatch() -> None:
    # Craft a byte-order-reversed window from REAL trace bytes (re-reverse the
    # canonical-LE bytes the reader produced -> a MSB-first feed Triton can't decode
    # but capstone can). drive must BLOCK with the fix-the-feed note and NOT run the
    # symex/escape-hatch path (no S-expr fill over a feed bug).
    real = _real_window(0, 60)
    assert real
    lo, hi = 0x1000, 0x1000 + 4 * len(real)
    reversed_items = [
        replace(ins, pc=lo + 4 * i, idx=i, bytes_=bytes(ins.bytes_)[::-1])
        for i, ins in enumerate(real)
    ]

    called = []

    def runner_spy(_ctx):
        called.append(1)
        return {"propagated": True, "gold_parity": "8/8", "expr_source": "x"}

    # The decode-feed guard now runs BEFORE any analysis checkpoint, so a
    # systematic feed bug BLOCKs directly — no checkpoint (mem_input_symbolize_vs_back
    # / alias / static) ever fires. on_checkpoint refuses any checkpoint to pin
    # that none is asked on a byte-feed-bug window.
    def on_checkpoint(cp):
        raise AssertionError(
            f"no checkpoint must fire on a systematic feed bug: {cp.name}")

    cc = CaseConfig(
        target="libEncryptor.so", input_hash="rev", run_id="rev-1",
        seed_hint_addr=0x100, sink_hint_addr=0x200, entry_pc=lo - 1,
        window=(lo, hi), reg_file=("x0", "x1", "x19"),
        inputs=("carrier",), parity_min=8, symbolic_regs=("x0", "x1"),
        task="byte_order_reversed",
    )
    res = drive(trace=reversed_items, case_config=cc, triton_runner=runner_spy,
                on_checkpoint=on_checkpoint)
    assert isinstance(res, DriveResult)
    assert res.closed is False
    assert res.emitted_F is None
    assert not called, "symex/escape-hatch must NOT run on a systematic feed bug"
    assert res.decode_audit and res.decode_audit["systematic"] is True
    assert res.decode_audit["feed_mismatch"] > 0
    assert "byte-FEED bug" in res.note and "do NOT S-expr fill" in res.note


@requires_capstone
@requires_triton
def test_drive_feed_guard_precedes_mem_input_checkpoint_no_on_checkpoint() -> None:
    # A SYNTHETIC systematic-feed window that ALSO carries an external memory input
    # (a ldr off an un-backed address — the exact form that fires
    # mem_input_symbolize_vs_back). The decode-feed guard now runs BEFORE the mem
    # checkpoint, so drive must BLOCK directly on the feed bug — never DrivePause,
    # never call symex — with NO on_checkpoint / decisions supplied at all.
    real = _real_window(0, 16)
    assert real
    lo, hi = 0x2000, 0x2000 + 4 * (len(real) + 1)
    reversed_items = [
        replace(ins, pc=lo + 4 * i, idx=i, bytes_=bytes(ins.bytes_)[::-1])
        for i, ins in enumerate(real)
    ]
    # an external memory load (un-backed addr) that WOULD fire the mem checkpoint
    ldr_bytes = canonical_aarch64_bytes(bytes.fromhex("f9400262"), "ldr x2, [x19]")
    reversed_items.append(Instruction(
        idx=len(real), pc=lo + 4 * len(real), bytes_=bytes(ldr_bytes)[::-1],
        mnemonic="ldr x2, [x19]", regs_read={"x19": 0x9000}, regs_write={"x2": 0},
        mem=(MemOp("r", 0x9000, 0, 8),)))

    called = []

    def runner_spy(_ctx):
        called.append(1)
        return {"propagated": True, "gold_parity": "8/8", "expr_source": "x"}

    cc = CaseConfig(
        target="syn.so", input_hash="rev2", run_id="rev2-1",
        seed_hint_addr=0x100, sink_hint_addr=0x200, entry_pc=lo - 1,
        window=(lo, hi), reg_file=("x0", "x1", "x19"),
        inputs=("carrier",), parity_min=8, symbolic_regs=("x0", "x1"),
        task="syn_byte_order_reversed_with_mem",
    )
    res = drive(trace=reversed_items, case_config=cc, triton_runner=runner_spy)
    assert isinstance(res, DriveResult)          # NOT a DrivePause
    assert res.closed is False
    assert res.emitted_F is None
    assert not called, "symex must NOT run on a systematic feed bug"
    assert res.decode_audit and res.decode_audit["systematic"] is True
    assert "byte-FEED bug" in res.note and "do NOT S-expr fill" in res.note


# --- §0: auto-seed edge — all-backed -> reg_file − backed -> BLOCK -----------

def _two_step_window():
    # A load off base x19 then a mix. Bytes are real-decodable (str/add) so the
    # decode guard stays clean and we isolate the seed behavior.
    s = canonical_aarch64_bytes(bytes.fromhex("f9003a69"), "str x9, [x19, #0x70]")
    a = canonical_aarch64_bytes(bytes.fromhex("0000010b"), None)  # add w0, w0, w1 (LE)
    return [
        Instruction(idx=0, pc=0x1000, bytes_=s, mnemonic="str x9, [x19, #0x70]",
                    regs_read={"x9": 0, "x19": 0x9000}, regs_write={}),
        Instruction(idx=1, pc=0x1004, bytes_=a, mnemonic="add w0, w0, w1",
                    regs_read={"w0": 0, "w1": 0}, regs_write={"w0": 0}),
    ]


def _runner_ok(_ctx):
    return {"propagated": True, "gold_parity": "8/8",
            "expr_source": "def f(carrier):\n    return bytes(8)\n"}


def test_drive_seed_blocks_when_reg_file_minus_backed_is_empty() -> None:
    # reg_file == {x19}, x19 backed (the only live-in base) -> reg_file − backed is
    # empty -> BLOCK with the seed note; must NOT fall back to the full reg_file
    # (which would symbolize the backed base and clash).
    items = _two_step_window()
    cc = CaseConfig(
        target="t", input_hash="h", run_id="r", seed_hint_addr=0x1, sink_hint_addr=0x2,
        entry_pc=0xFFF, window=(0x1000, 0x10FF), reg_file=("x19",), inputs=("c",),
        parity_min=8, symbolic_regs=None,
        concrete_backing=build_concrete_backing(reg_values={"x19": 0x9000}),
        task="seed_all_backed")
    res = drive(trace=items, case_config=cc, triton_runner=_runner_ok,
                decisions={"alias_vs_compute": "compute", "which_static": []})
    assert isinstance(res, DriveResult)
    assert res.closed is False
    assert res.emitted_F is None
    assert "no symbolic input" in res.note
    assert "NOT falling back to the full reg_file" in res.note


def test_drive_seed_recovers_from_reg_file_minus_backed() -> None:
    # live-in {x19} is fully backed, but reg_file has a free register (x0) -> seed =
    # reg_file − backed = {x0}; drive proceeds (no clash, no spurious full-reg_file).
    items = _two_step_window()
    cc = CaseConfig(
        target="t", input_hash="h", run_id="r", seed_hint_addr=0x1, sink_hint_addr=0x2,
        entry_pc=0xFFF, window=(0x1000, 0x10FF), reg_file=("x0", "x19"), inputs=("c",),
        parity_min=8, symbolic_regs=None,
        concrete_backing=build_concrete_backing(reg_values={"x19": 0x9000}),
        task="seed_recover")
    res = drive(trace=items, case_config=cc, triton_runner=_runner_ok,
                decisions={"alias_vs_compute": "compute", "which_static": []})
    assert isinstance(res, DriveResult)
    # the seed step recorded the recovered symbolic set = reg_file − backed = {x0}
    seed_step = next(s for s in res.per_step if s["step"] == "seed_entry_state")
    assert seed_step["symbolic_regs"] == ["x0"]
    assert seed_step["auto_seed"].get("seed_from_reg_file_minus_backed") is True
    assert "no symbolic input" not in res.note
