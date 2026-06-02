"""Smoke tests for runner_client.

Verifies UnidbgTextTraceReader actually parses the example/task-libEncryptor/libs/arm64-v8a/trace.txt sample
to the same stats we got from grep earlier:
  - 41,416 instructions
  - 1,968 unique PCs
  - entry 0x40007d88, exit 0x40007ed8
  - hottest PC 0x40006cd8 with 309 hits
"""

from __future__ import annotations

from pathlib import Path

from engine.aarch64_mem import parse_mem_ops
from engine.runner_client import JsonlTraceReader, UnidbgTextTraceReader
from engine.types import MemOp

REPO_ROOT = Path(__file__).resolve().parents[2]
VMP_TRACE = REPO_ROOT / "example" / "task-libEncryptor" / "libs" / "arm64-v8a" / "trace.txt"


def test_unidbg_parser_on_vmp_sample() -> None:
    assert VMP_TRACE.exists(), f"missing fixture: {VMP_TRACE}"
    reader = UnidbgTextTraceReader(VMP_TRACE)
    n = 0
    pcs: dict[int, int] = {}
    first_pc: int | None = None
    last_pc: int | None = None
    for ins in reader:
        n += 1
        if first_pc is None:
            first_pc = ins.pc
        last_pc = ins.pc
        pcs[ins.pc] = pcs.get(ins.pc, 0) + 1

    assert n == 41416, f"expected 41,416 instructions, got {n}"
    assert len(pcs) == 1968, f"expected 1,968 unique PCs, got {len(pcs)}"
    assert first_pc == 0x40007D88
    assert last_pc == 0x40007ED8
    assert pcs[0x40006CD8] == 309


def test_first_instruction_fields() -> None:
    """First line is 'sub sp, sp, #0x60' — verify field extraction."""
    reader = UnidbgTextTraceReader(VMP_TRACE)
    first = next(iter(reader))
    assert first.idx == 0
    assert first.pc == 0x40007D88
    assert first.bytes_ == bytes.fromhex("ff8301d1")
    assert first.mnemonic == "sub sp, sp, #0x60"
    assert first.regs_read.get("sp") == 0xBFFFF700
    assert first.regs_write.get("sp") == 0xBFFFF6A0


def test_jsonl_reader_round_trip(tmp_path) -> None:
    """Spot-check the standard JSONL reader on a tiny synthetic trace."""
    p = tmp_path / "t.jsonl"
    p.write_text(
        '{"idx":0,"pc":"0x40001000","bytes":"00112233","mnemonic":"nop",'
        '"regs_read":{"x0":"0xa"},"regs_write":{"x0":"0xb"},'
        '"mem":[{"rw":"r","addr":"0x40002000","val":"0xdead","size":4}]}\n'
    )
    ins = next(iter(JsonlTraceReader(p)))
    assert ins.pc == 0x40001000
    assert ins.regs_read == {"x0": 0xA}
    assert ins.regs_write == {"x0": 0xB}
    assert len(ins.mem) == 1
    assert ins.mem[0].rw == "r"
    assert ins.mem[0].addr == 0x40002000


def test_jsonl_reader_skips_rows_without_bytes(tmp_path) -> None:
    """A JSONL whose rows do not ALL carry a "bytes" field must not crash the
    reader (the systematic recover_window KeyError('bytes') single point).

    The convention-detection FIRST pass (_samples) hard-subscripted obj["bytes"];
    a mem-event / annotation record interleaved in the stream raised KeyError and,
    because every recovery window re-loaded the cohort trace through this reader,
    all 38 windows reported a flat tool_error:KeyError. A non-instruction row is
    now skipped (in both the convention pass AND the iter body), the real
    instruction rows survive.
    """
    p = tmp_path / "mixed.jsonl"
    p.write_text(
        # a real instruction row
        '{"idx":0,"pc":"0x40001000","bytes":"00112233","mnemonic":"nop"}\n'
        # a row with NO "bytes" field (mem-event / annotation record)
        '{"idx":1,"pc":"0x40001004","mnemonic":"meta","mem":[]}\n'
        # an empty-string "bytes" (equally undecodable) — must also be skipped
        '{"idx":2,"pc":"0x40001008","bytes":"","mnemonic":"blank"}\n'
        # another real instruction row after the gaps
        '{"idx":3,"pc":"0x4000100c","bytes":"44556677","mnemonic":"add"}\n'
    )
    items = list(JsonlTraceReader(p))           # must NOT raise KeyError('bytes')
    assert [i.idx for i in items] == [0, 3]     # only the decodable rows survive
    assert items[0].pc == 0x40001000
    assert items[1].pc == 0x4000100C


# --- ⑥ parse_mem_ops: synthetic addressing-mode coverage --------------------
# Each case is design-driven (pure AArch64 addressing syntax + in-line register
# values), no address/handler/case constants.


def test_mem_base_only_load() -> None:
    # ldr x8, [x20] ; EA = reads[x20]; val = writes[x8]; size 8
    ops, unresolved = parse_mem_ops("ldr x8, [x20]", {"x20": 0x1000}, {"x8": 0xDEAD})
    assert not unresolved
    assert ops == (MemOp(rw="r", addr=0x1000, val=0xDEAD, size=8),)


def test_mem_base_imm_load() -> None:
    # ldr x8, [x25, #0x28] ; EA = base + 0x28
    ops, unresolved = parse_mem_ops("ldr x8, [x25, #0x28]", {"x25": 0x1000}, {"x8": 0x5})
    assert not unresolved
    assert ops == (MemOp(rw="r", addr=0x1028, val=0x5, size=8),)


def test_mem_base_imm_store() -> None:
    # str x8, [sp, #8] ; store -> rw=w, val from pre-state reads[x8]
    ops, unresolved = parse_mem_ops(
        "str x8, [sp, #8]", {"x8": 0xAA, "sp": 0x2000}, {"x8": 0xAA}
    )
    assert not unresolved
    assert ops == (MemOp(rw="w", addr=0x2008, val=0xAA, size=8),)


def test_mem_reg_offset() -> None:
    # ldr x0, [x1, x2] ; EA = base + index
    ops, unresolved = parse_mem_ops(
        "ldr x0, [x1, x2]", {"x1": 0x1000, "x2": 0x40}, {"x0": 0x9}
    )
    assert not unresolved
    assert ops == (MemOp(rw="r", addr=0x1040, val=0x9, size=8),)


def test_mem_reg_offset_lsl() -> None:
    # ldr x0, [x1, x2, lsl #3] ; EA = base + (index << 3)
    ops, unresolved = parse_mem_ops(
        "ldr x0, [x1, x2, lsl #3]", {"x1": 0x1000, "x2": 0x4}, {"x0": 0x9}
    )
    assert not unresolved
    assert ops == (MemOp(rw="r", addr=0x1000 + (0x4 << 3), val=0x9, size=8),)


def test_mem_pre_index() -> None:
    # stp x28, x19, [sp, #-0x20]! ; EA = sp + (-0x20); pair -> two ops
    ops, unresolved = parse_mem_ops(
        "stp x28, x19, [sp, #-0x20]!",
        {"x28": 0x0, "x19": 0x5D, "sp": 0xBFFFF6A0},
        {"x28": 0x0, "x19": 0x5D, "sp": 0xBFFFF680},
    )
    assert not unresolved
    ea = 0xBFFFF6A0 - 0x20
    assert ops == (
        MemOp(rw="w", addr=ea, val=0x0, size=8),
        MemOp(rw="w", addr=ea + 8, val=0x5D, size=8),
    )


def test_mem_post_index() -> None:
    # ldr x0, [x1], #0x10 ; EA = original base value (writeback ignored)
    ops, unresolved = parse_mem_ops(
        "ldr x0, [x1], #0x10", {"x1": 0x1000}, {"x0": 0x7, "x1": 0x1010}
    )
    assert not unresolved
    assert ops == (MemOp(rw="r", addr=0x1000, val=0x7, size=8),)


def test_mem_ldp_two_ops() -> None:
    # ldp x0, x1, [x2, #0x10] ; two reads at EA and EA+8
    ops, unresolved = parse_mem_ops(
        "ldp x0, x1, [x2, #0x10]", {"x2": 0x2000}, {"x0": 0xAA, "x1": 0xBB}
    )
    assert not unresolved
    assert ops == (
        MemOp(rw="r", addr=0x2010, val=0xAA, size=8),
        MemOp(rw="r", addr=0x2018, val=0xBB, size=8),
    )


def test_mem_stp_two_ops() -> None:
    # stp x24, x23, [sp, #0x20] ; two writes, store values from pre-state
    ops, unresolved = parse_mem_ops(
        "stp x24, x23, [sp, #0x20]",
        {"x24": 0x11, "x23": 0x22, "sp": 0x1000},
        {"x24": 0x11, "x23": 0x22},
    )
    assert not unresolved
    assert ops == (
        MemOp(rw="w", addr=0x1020, val=0x11, size=8),
        MemOp(rw="w", addr=0x1028, val=0x22, size=8),
    )


def test_mem_ldrb_size1() -> None:
    ops, unresolved = parse_mem_ops("ldrb w0, [x1]", {"x1": 0x1000}, {"w0": 0xFF})
    assert not unresolved
    assert ops == (MemOp(rw="r", addr=0x1000, val=0xFF, size=1),)


def test_mem_ldrh_size2() -> None:
    ops, unresolved = parse_mem_ops("ldrh w0, [x1]", {"x1": 0x1000}, {"w0": 0xBEEF})
    assert not unresolved
    assert ops == (MemOp(rw="r", addr=0x1000, val=0xBEEF, size=2),)


def test_mem_ldrsw_size4() -> None:
    ops, unresolved = parse_mem_ops("ldrsw x0, [x1]", {"x1": 0x1000}, {"x0": 0x1234})
    assert not unresolved
    assert ops == (MemOp(rw="r", addr=0x1000, val=0x1234, size=4),)


def test_mem_wreg_size4() -> None:
    # str w0, [x1] -> 32-bit store, size 4 from register class
    ops, unresolved = parse_mem_ops(
        "str w0, [x1]", {"w0": 0xCAFE, "x1": 0x1000}, {"w0": 0xCAFE}
    )
    assert not unresolved
    assert ops == (MemOp(rw="w", addr=0x1000, val=0xCAFE, size=4),)


def test_mem_non_memory_instruction() -> None:
    # Non ld*/st* op -> no mem, not unresolved (behaviour unchanged).
    ops, unresolved = parse_mem_ops("sub sp, sp, #0x60", {"sp": 0x1000}, {"sp": 0xFA0})
    assert ops == ()
    assert not unresolved


def test_mem_missing_base_value_unresolved() -> None:
    # Base register value not on the line -> empty + ea_unresolved (no fabrication).
    ops, unresolved = parse_mem_ops("ldr x8, [x25, #0x28]", {}, {"x8": 0x5})
    assert ops == ()
    assert unresolved


def test_mem_missing_index_value_unresolved() -> None:
    ops, unresolved = parse_mem_ops(
        "ldr x0, [x1, x2]", {"x1": 0x1000}, {"x0": 0x9}
    )
    assert ops == ()
    assert unresolved


def test_mem_missing_value_reg_unresolved() -> None:
    # EA resolvable but the data register value absent -> honest whole-step miss.
    ops, unresolved = parse_mem_ops("ldr x8, [x20]", {"x20": 0x1000}, {})
    assert ops == ()
    assert unresolved


def test_unidbg_reader_populates_mem(tmp_path) -> None:
    """Synthetic unidbg lines -> Instruction.mem populated; unresolved counted."""
    p = tmp_path / "t.txt"
    p.write_text(
        # resolvable load
        '[09:39:47 053][libX.so 0x07da8] [281740f9] 0x40007da8: '
        '"ldr x8, [x25, #0x28]" x25=0xbffff708 => x8=0x12\n'
        # resolvable store pair
        '[09:39:47 017][libX.so 0x07d90] [f85f02a9] 0x40007d90: '
        '"stp x24, x23, [sp, #0x20]" x24=0x1 x23=0x2 sp=0xbffff6a0 '
        '=> x24=0x1 x23=0x2\n'
        # non-memory instruction -> mem empty, not counted
        '[09:39:47 005][libX.so 0x07d88] [ff8301d1] 0x40007d88: '
        '"sub sp, sp, #0x60" sp=0xbffff700 => sp=0xbffff6a0\n'
    )
    reader = UnidbgTextTraceReader(p)
    inss = list(reader)
    assert inss[0].mem == (MemOp(rw="r", addr=0xBFFFF708 + 0x28, val=0x12, size=8),)
    assert inss[1].mem == (
        MemOp(rw="w", addr=0xBFFFF6A0 + 0x20, val=0x1, size=8),
        MemOp(rw="w", addr=0xBFFFF6A0 + 0x28, val=0x2, size=8),
    )
    assert inss[2].mem == ()
    assert reader.unresolved_mem_steps == 0


def test_unidbg_reader_counts_unresolved(tmp_path) -> None:
    """A memory line whose base value is absent -> empty mem + unresolved count."""
    p = tmp_path / "t.txt"
    p.write_text(
        # ldr literal-ish: base reg value not on the line
        '[09:39:47 053][libX.so 0x07da8] [281740f9] 0x40007da8: '
        '"ldr x8, [x25, #0x28]" => x8=0x12\n'
    )
    reader = UnidbgTextTraceReader(p)
    inss = list(reader)
    assert inss[0].mem == ()
    assert reader.unresolved_mem_steps == 1


# ---------------------------------------------------------------------------
# Task 5 — mem-sidecar SEMANTIC FAMILY recognition + de-silence WARN scan.
# dev-closure-evidence-layering-trap-state-spec.md.
# ---------------------------------------------------------------------------

import json as _json

from engine.runner_client import (
    looks_like_mem_sidecar,
    mem_sidecar_candidates,
    mem_sidecar_sibling,
    unmerged_mem_sidecars,
)


def _write_trace(p: Path) -> None:
    p.write_text(_json.dumps({
        "idx": 0, "pc": "0x1000", "bytes": "1f2003d5", "mnemonic": "nop",
    }) + "\n")


def _write_mem(p: Path) -> None:
    p.write_text(_json.dumps({
        "idx": 0, "addr": "0x2000", "rw": "r", "val": "0x41", "size": 1,
    }) + "\n")


def test_sidecar_candidates_are_the_family():
    cands = mem_sidecar_candidates("dir/trace.jsonl")
    names = [c.name for c in cands]
    assert names == ["trace_mem.jsonl", "trace_mem_sidecar.jsonl"]


def test_looks_like_mem_sidecar_recognises_both_suffixes():
    assert looks_like_mem_sidecar("x_mem.jsonl") is True
    assert looks_like_mem_sidecar("x_mem_sidecar.jsonl") is True
    assert looks_like_mem_sidecar("x.jsonl") is False


def test_resolve_picks_up_mem_sidecar_alt_family_member(tmp_path):
    """验收① — a *_mem_sidecar.jsonl is auto-recognised (no longer silently dropped)."""
    trace = tmp_path / "t.jsonl"
    _write_trace(trace)
    alt = tmp_path / "t_mem_sidecar.jsonl"
    _write_mem(alt)
    reader = JsonlTraceReader(trace)
    resolved = reader.resolve_mem_sidecar()
    assert resolved is not None
    assert resolved.name == "t_mem_sidecar.jsonl"
    # and it actually merges (mem events folded in)
    merged = reader.merged()
    assert merged.report is not None
    assert merged.report.mem_events_merged >= 1


def test_resolve_prefers_canonical_when_both_exist(tmp_path):
    """验收② — the canonical _mem.jsonl still wins (zero regression)."""
    trace = tmp_path / "t.jsonl"
    _write_trace(trace)
    _write_mem(tmp_path / "t_mem.jsonl")
    _write_mem(tmp_path / "t_mem_sidecar.jsonl")
    resolved = JsonlTraceReader(trace).resolve_mem_sidecar()
    assert resolved.name == "t_mem.jsonl"


def test_unmerged_sidecar_looking_files_are_surfaced(tmp_path):
    """验收③ — a sidecar-looking file in the dir that was NOT merged is reported
    (WARN material), not silently ignored."""
    trace = tmp_path / "main.jsonl"
    _write_trace(trace)
    # a differently-stemmed sidecar that the resolver will NOT pick for `main`
    stray = tmp_path / "other_mem.jsonl"
    _write_mem(stray)
    resolved = JsonlTraceReader(trace).resolve_mem_sidecar()
    assert resolved is None
    stray_found = unmerged_mem_sidecars(trace, resolved)
    assert stray in stray_found


def test_unmerged_excludes_the_resolved_one(tmp_path):
    trace = tmp_path / "t.jsonl"
    _write_trace(trace)
    canon = tmp_path / "t_mem.jsonl"
    _write_mem(canon)
    resolved = JsonlTraceReader(trace).resolve_mem_sidecar()
    assert resolved == canon
    # the resolved sidecar is NOT reported as unmerged
    assert canon not in unmerged_mem_sidecars(trace, resolved)


# ---------------------------------------------------------------------------
# #2 — sidecar auto-merge=0 divergence: when the mem-FILE itself is handed in as
# the trace path, the candidate stem doubled (..._mem_mem.jsonl) → auto-resolve
# None → 0 merged, while an explicit mem_sidecar= load found 110399. Fix: strip
# the family suffix to recover the base stem, and source the instruction skeleton
# from the base sibling so auto-merge == explicit. (F0 judgment 6/18.)
# ---------------------------------------------------------------------------

from engine.runner_client import base_trace_sibling


def test_candidates_strip_family_suffix_no_doubling():
    # A mem-file handed in as the trace path resolves to the SAME family as the
    # base trace would — not the doubled ..._mem_mem.jsonl that never exists.
    base_cands = [c.name for c in mem_sidecar_candidates("dir/t.jsonl")]
    mem_cands = [c.name for c in mem_sidecar_candidates("dir/t_mem.jsonl")]
    assert base_cands[:2] == ["t_mem.jsonl", "t_mem_sidecar.jsonl"]
    # mem-as-path must NOT produce t_mem_mem.jsonl
    assert "t_mem_mem.jsonl" not in mem_cands
    assert mem_cands[:2] == ["t_mem.jsonl", "t_mem_sidecar.jsonl"]


def test_base_trace_sibling_redirect():
    assert base_trace_sibling("dir/t.jsonl") is None              # already a base trace
    assert base_trace_sibling("dir/t_mem.jsonl").name == "t.jsonl"
    assert base_trace_sibling("dir/t_mem_sidecar.jsonl").name == "t.jsonl"


def test_auto_merge_matches_explicit_when_mem_file_passed_as_trace(tmp_path):
    # Regression for the F0 auto-merge=0 divergence.
    base = tmp_path / "t.jsonl"
    _write_trace(base)
    mem = tmp_path / "t_mem.jsonl"
    _write_mem(mem)
    # explicit reference: base trace + mem sidecar explicitly named
    explicit = JsonlTraceReader(base, mem_sidecar=mem).merged().report.mem_events_merged
    assert explicit == 1
    # buggy path: the mem-FILE itself handed in as the trace → auto must match.
    auto = JsonlTraceReader(mem).merged()
    assert auto.report.mem_events_merged == explicit          # was 0 pre-fix
    assert auto.report.n_items == 1                            # skeleton from base sibling


def test_auto_merge_on_base_trace_unchanged(tmp_path):
    # Zero-regression: passing the base trace still auto-resolves + merges.
    base = tmp_path / "t.jsonl"
    _write_trace(base)
    mem = tmp_path / "t_mem.jsonl"
    _write_mem(mem)
    auto = JsonlTraceReader(base).merged()
    assert auto.report.mem_events_merged == 1
    assert auto.report.n_items == 1


def test_mem_file_as_trace_without_base_sibling_does_not_redirect(tmp_path):
    # No base sibling on disk → no redirect (best-effort, never fabricates a path).
    mem = tmp_path / "orphan_mem.jsonl"
    _write_mem(mem)
    reader = JsonlTraceReader(mem)
    merged = reader.merged()
    # main stream is the mem file itself (no bytes rows) → empty skeleton, but it
    # must not crash; mem events go unaligned rather than vanish.
    assert merged.report.n_items == 0


# ---------------------------------------------------------------------------
# Task 6 — capture/mem -> MemSnapshot end-to-end conversion.
# ---------------------------------------------------------------------------

from engine.runner_client import (
    ObservedState,
    RerunResult,
    mem_snapshots_from_rerun,
)


def test_mem_snapshots_from_rerun_converts_observed_mem():
    res = RerunResult(
        output=b"\x00",
        observations=(
            ObservedState(pc=0x4000, when="after", regs={},
                          mem={0x7f00: b"\xde\xad", 0x7f10: b"\xbe\xef"}),
        ),
    )
    snaps = mem_snapshots_from_rerun(res)
    assert len(snaps) == 2
    addrs = {s.addr: s.data for s in snaps}
    assert addrs[0x7f00] == b"\xde\xad"
    assert addrs[0x7f10] == b"\xbe\xef"
    assert all(s.source == "snapshot" for s in snaps)


def test_mem_snapshots_empty_when_no_mem_captured():
    res = RerunResult(output=b"\x00", observations=(
        ObservedState(pc=0x4000, when="after", regs={"x0": 1}, mem={}),
    ))
    assert mem_snapshots_from_rerun(res) == []


# ---------------------------------------------------------------------------
# Bug1 — rerun() request must serialize capture + mem per observe point.
# Pre-fix the request dropped them ({pc,when,regs} only) so a runner could never
# capture mem → snapshots permanently empty → same-execution oracle path dead.
# ---------------------------------------------------------------------------

from engine.runner_client import ObservePoint, SubprocessRunnerAdapter


def _stub_subproc_adapter(capture_sink: list):
    """A SubprocessRunnerAdapter whose _call records params instead of spawning
    a process. Tests the REQUEST-side serialization in isolation."""
    a = object.__new__(SubprocessRunnerAdapter)
    a._next_id = 0

    def _fake_call(method, params=None):
        capture_sink.append((method, params))
        # canned rerun response so RerunResult parsing succeeds
        return {"output_hex": "00", "observations": []}

    a._call = _fake_call  # type: ignore[attr-defined]
    return a


def test_rerun_request_serializes_capture_and_mem():
    sink: list = []
    adapter = _stub_subproc_adapter(sink)
    op = ObservePoint(pc=0x70EC4, when="before", capture=("regs", "mem"),
                      regs=("x0", "x19"), mem=((0xBFFFF708, 8), (0x40002000, 4)))
    adapter.rerun(b"\x01\x02", observe_points=[op])
    assert len(sink) == 1
    method, params = sink[0]
    assert method == "rerun"
    pts = params["observe_points"]
    assert len(pts) == 1
    pt = pts[0]
    # When upper-cased for the Java side, full field set carried (the fix).
    assert pt["pc"] == "0x70ec4"
    assert pt["when"] == "BEFORE"
    assert pt["capture"] == ["REGS", "MEM"]  # Java Capture enum: UPPERCASE
    assert pt["regs"] == ["x0", "x19"]  # register NAMES, not an enum → unchanged
    assert pt["mem"] == [{"addr": "0xbffff708", "size": 8},
                         {"addr": "0x40002000", "size": 4}]


def test_rerun_request_uppercases_capture_for_java_enum():
    """Bug2: ``capture`` must be UPPER on the wire (Java ``Capture.valueOf``),
    symmetric with ``when``. Lowercase ``["mem"]`` tripped the runner →
    `isolated once exit 1`. register NAMES (``regs``) stay verbatim — not enums.
    """
    sink: list = []
    adapter = _stub_subproc_adapter(sink)
    # single-kind lowercase input
    op1 = ObservePoint(pc=0x1000, when="before", capture=("mem",),
                       regs=("x19",))
    adapter.rerun(b"\x00", observe_points=[op1])
    pt1 = sink[0][1]["observe_points"][0]
    assert pt1["capture"] == ["MEM"]
    assert pt1["regs"] == ["x19"]  # reg name unchanged (not an enum)

    # multi-kind preserves order, each upper-cased
    sink.clear()
    adapter._next_id = 0
    op2 = ObservePoint(pc=0x2000, when="after", capture=("regs", "mem"))
    adapter.rerun(b"\x00", observe_points=[op2])
    pt2 = sink[0][1]["observe_points"][0]
    assert pt2["capture"] == ["REGS", "MEM"]
    # symmetric with the long-fixed `when`
    assert pt2["when"] == "AFTER"


def test_rerun_request_mem_only_capture():
    # A mem-only capture point must still carry capture+mem (regs may be empty).
    sink: list = []
    adapter = _stub_subproc_adapter(sink)
    op = ObservePoint(pc=0x1000, when="after", capture=("mem",), mem=((0x2000, 16),))
    adapter.rerun(b"\xaa", observe_points=[op])
    pt = sink[0][1]["observe_points"][0]
    assert pt["capture"] == ["MEM"]  # Java Capture enum: UPPERCASE
    assert pt["mem"] == [{"addr": "0x2000", "size": 16}]
    assert pt["regs"] == []
    assert pt["when"] == "AFTER"


# ---------------------------------------------------------------------------
# Reg-relative point-watch wire (contracts §3.2.1): rerun() must serialize the
# new mem_regrel field; concrete-only points stay byte-for-byte as before.
# ---------------------------------------------------------------------------

from engine.runner_client import RegRelWatch  # noqa: E402


def test_rerun_request_serializes_mem_regrel():
    """① (base_reg,offset,width,pc[,kind]) → request carries mem_regrel."""
    sink: list = []
    adapter = _stub_subproc_adapter(sink)
    op = ObservePoint(
        pc=0x70EC4, when="before", capture=("mem",),
        mem_regrel=(RegRelWatch(base_reg="x19", offset=0x38, width=8,
                                pc=0x70EC4, kind="read"),),
    )
    adapter.rerun(b"\x01", observe_points=[op])
    pt = sink[0][1]["observe_points"][0]
    assert pt["mem_regrel"] == [
        {"base_reg": "x19", "offset": 0x38, "width": 8,
         "pc": "0x70ec4", "kind": "read"},
    ]
    # The concrete fields still ride alongside (additive, not a replacement).
    assert pt["mem"] == []
    assert pt["capture"] == ["MEM"]  # Java Capture enum: UPPERCASE
    assert pt["when"] == "BEFORE"
    # mem_regrel.kind stays lowercase: contract §3.2.1 wire form, no Java enum.
    assert pt["mem_regrel"][0]["kind"] == "read"


def test_rerun_request_mem_regrel_negative_offset_and_write():
    sink: list = []
    adapter = _stub_subproc_adapter(sink)
    op = ObservePoint(
        pc=0x1000, when="after", capture=("mem",),
        mem_regrel=(RegRelWatch(base_reg="sp", offset=-16, width=4,
                                pc=0x1004, kind="write"),),
    )
    adapter.rerun(b"\xaa", observe_points=[op])
    pt = sink[0][1]["observe_points"][0]
    assert pt["mem_regrel"] == [
        {"base_reg": "sp", "offset": -16, "width": 4,
         "pc": "0x1004", "kind": "write"},
    ]


def test_rerun_request_concrete_point_omits_mem_regrel():
    """③ default (no mem_regrel) → key absent; concrete wire unchanged."""
    sink: list = []
    adapter = _stub_subproc_adapter(sink)
    op = ObservePoint(pc=0x1000, when="after", capture=("mem",),
                      mem=((0x2000, 16),))
    adapter.rerun(b"\xaa", observe_points=[op])
    pt = sink[0][1]["observe_points"][0]
    assert "mem_regrel" not in pt
    assert pt["mem"] == [{"addr": "0x2000", "size": 16}]


def test_rerun_empty_observe_points_no_param():
    # Empty observe_points → no observe_points key (unchanged behaviour).
    sink: list = []
    adapter = _stub_subproc_adapter(sink)
    adapter.rerun(b"\x00", observe_points=[])
    assert "observe_points" not in sink[0][1]
