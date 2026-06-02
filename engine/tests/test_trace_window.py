"""capability_request.md §P0-2 — extra trace window plumbing test.

S4 backward slice must pick up instructions inside an ``extra_trace_windows``
band as additional sinks. This verifies the session.json write-back AND
the S4 slice integration.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from engine.core import Core, CoreConfig
from engine.runner_client import NullRunnerAdapter
from engine.stages import s3_triton, s4_slice
from engine.stages.s4_slice import _idxs_in_windows, _parse_window_endpoint
from engine.types import Instruction, TargetMeta


def _ins(idx, pc, regs_read=None, regs_write=None) -> Instruction:
    return Instruction(
        idx=idx, pc=pc, bytes_=b"\x00\x00\x00\x00", mnemonic="",
        regs_read=regs_read or {}, regs_write=regs_write or {}, mem=(),
    )


def test_idxs_in_windows_filters_by_band():
    items = [
        _ins(0, 0x10000),
        _ins(1, 0x322c90),
        _ins(2, 0x322ccc),
        _ins(3, 0x40000),
    ]
    windows = [["0x322c00", "0x322d00"]]
    out = _idxs_in_windows(items, windows)
    assert out == [1, 2]


def test_parse_window_endpoint_accepts_hex_string_and_int():
    assert _parse_window_endpoint("0x40") == 0x40
    assert _parse_window_endpoint("ff") == 0xff
    assert _parse_window_endpoint(0x40) == 0x40
    assert _parse_window_endpoint("not-a-hex") is None
    assert _parse_window_endpoint(3.14) is None


def test_session_json_records_extra_trace_windows():
    items = [_ins(0, 0x100), _ins(1, 0x104)]
    tm = TargetMeta(
        target_name="window-test", arch="arm64",
        algo_entry_pc=0x100, algo_exit_pc=0x104,
        input_length=None, output_length=4,
    )
    cfg = CoreConfig(
        work_root=Path(tempfile.mkdtemp(prefix="utov-test-window-")),
        target_meta=tm, input_hash="h", driver_mode="script",
        new_run=True,
        extra_trace_windows=((0x32302c, 0x325708),),
    )

    class _R:
        def __init__(self, xs): self.xs = xs
        def __iter__(self): return iter(self.xs)

    core = Core(cfg, _R(items), NullRunnerAdapter(tm), skip_conformance=True)

    session_path = core.work.root / "session.json"
    saved = json.loads(session_path.read_text())
    assert saved["extra_trace_windows"] == [["0x32302c", "0x325708"]]


def test_s4_picks_up_band_idxs_as_sinks():
    """An instruction whose PC is in extra_trace_windows ends up as an
    additional sink — backward slice reaches its ancestors."""
    items = [
        _ins(0, 0x10000, regs_write={"x0": 1}),
        _ins(1, 0x10004, regs_read={"x0": 1}, regs_write={"x1": 2}),
        _ins(2, 0x322c90, regs_read={"x1": 2}, regs_write={"x22": 3}),  # in-band
        _ins(3, 0x40000, regs_write={"x0": 9}),
    ]
    tm = TargetMeta(
        target_name="window-s4", arch="arm64",
        algo_entry_pc=0x10000, algo_exit_pc=0x40000,
        input_length=None, output_length=4,
    )
    cfg = CoreConfig(
        work_root=Path(tempfile.mkdtemp(prefix="utov-test-windows4-")),
        target_meta=tm, input_hash="h", driver_mode="script", new_run=True,
        extra_trace_windows=((0x322c00, 0x322d00),),
    )

    class _R:
        def __init__(self, xs): self.xs = xs
        def __iter__(self): return iter(self.xs)

    core = Core(cfg, _R(items), NullRunnerAdapter(tm), skip_conformance=True)
    # S3 produces the DFG. Run S3 first then S4 manually.
    s3_triton.run({"items": items, "work": core.work, "session": core.session})
    res = s4_slice.run({"items": items, "work": core.work, "session": core.session})

    # Slice must include the in-band instruction's idx (2) AND its
    # ancestor (1) AND that ancestor's parent (0).
    slice_path = core.work.root / "stage_outputs" / "s4_slice.jsonl"
    kept = []
    for ln in slice_path.read_text().splitlines():
        if ln.strip():
            kept.append(json.loads(ln)["idx"])
    assert 2 in kept, "in-band instruction must be a sink"
    assert 1 in kept, "its data ancestor must survive the slice"
    assert 0 in kept, "transitive ancestor must survive too"
