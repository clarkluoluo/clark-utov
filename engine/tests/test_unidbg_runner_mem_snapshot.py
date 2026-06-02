"""Task 6 — the integration adapter's MEMREGION_WATCH carries captured mem forward
as canonical MemSnapshots (so oracle_sink can consume them), and WARNs when the
capability claimed mem-capture but produced none.

dev-closure-evidence-layering-trap-state-spec.md task 6. Skips cleanly without
clark-hypotask installed.
"""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("hypotask")

from hypotask.runner.interface import RunnerCapability  # noqa: E402

from engine.integration.unidbg_runner import (  # noqa: E402
    MemRegionWatchResult,
    UnidbgSignRunner,
)
from engine.runner_client import (  # noqa: E402
    ObservedState,
    RerunResult,
    RunnerAdapter,
)
from engine.types import MemSnapshot, TargetMeta  # noqa: E402


class _MemCapturingAdapter(RunnerAdapter):
    """A live-ish adapter that DOES capture mem at observe points."""

    def metadata(self) -> TargetMeta:
        return TargetMeta(target_name="t", arch="arm64", algo_entry_pc=0,
                          algo_exit_pc=0, input_length=None, output_length=8)

    def rerun(self, input_bytes, observe_points=None):
        return RerunResult(
            output=b"\x00" * 8,
            observations=(
                ObservedState(pc=0x4000, when="after", regs={},
                              mem={0x7f00: b"\xde\xad\xbe\xef"}),
            ),
        )


class _NoMemAdapter(RunnerAdapter):
    """Claims to rerun (so MEMREGION_WATCH is advertised) but captures NO mem."""

    def metadata(self) -> TargetMeta:
        return TargetMeta(target_name="t", arch="arm64", algo_entry_pc=0,
                          algo_exit_pc=0, input_length=None, output_length=8)

    def rerun(self, input_bytes, observe_points=None):
        return RerunResult(output=b"\x00" * 8, observations=())


def test_memregion_watch_returns_canonical_snapshots():
    runner = UnidbgSignRunner(_MemCapturingAdapter())
    assert RunnerCapability.MEMREGION_WATCH in runner.capabilities()
    out = runner.invoke(
        RunnerCapability.MEMREGION_WATCH,
        input_bytes=b"x", pc=0x4000, base=0x7f00, size=4)
    assert isinstance(out, MemRegionWatchResult)
    assert len(out.mem_snapshots) == 1
    snap = out.mem_snapshots[0]
    assert isinstance(snap, MemSnapshot)
    assert snap.addr == 0x7f00
    assert snap.data == b"\xde\xad\xbe\xef"
    assert snap.source == "snapshot"


def test_memregion_watch_warns_when_no_snapshot_produced(caplog):
    runner = UnidbgSignRunner(_NoMemAdapter())
    with caplog.at_level(logging.WARNING):
        out = runner.invoke(
            RunnerCapability.MEMREGION_WATCH,
            input_bytes=b"x", pc=0x4000, base=0x7f00, size=4)
    assert isinstance(out, MemRegionWatchResult)
    assert out.mem_snapshots == ()
    # WARN surfaced (not a silent write/read fallback at the sink).
    assert any("NO mem snapshots" in r.message for r in caplog.records)


class _RecordingAdapter(RunnerAdapter):
    """Records the observe_points it was handed (request-side inspection)."""

    def __init__(self):
        self.seen = None

    def metadata(self) -> TargetMeta:
        return TargetMeta(target_name="t", arch="arm64", algo_entry_pc=0,
                          algo_exit_pc=0, input_length=None, output_length=8)

    def rerun(self, input_bytes, observe_points=None):
        self.seen = observe_points
        return RerunResult(output=b"\x00" * 8, observations=())


def test_memregion_watch_reg_relative_form():
    """② reg-relative invoke (base_reg,...) → ObservePoint.mem_regrel, no concrete mem."""
    adapter = _RecordingAdapter()
    runner = UnidbgSignRunner(adapter)
    runner.invoke(
        RunnerCapability.MEMREGION_WATCH,
        input_bytes=b"x", pc=0x70EC4, base_reg="x19", offset=0x38,
        width=8, kind="read")
    [pt] = adapter.seen
    assert pt.pc == 0x70EC4
    assert pt.mem == ()
    assert len(pt.mem_regrel) == 1
    w = pt.mem_regrel[0]
    assert (w.base_reg, w.offset, w.width, w.pc, w.kind) == ("x19", 0x38, 8, 0x70EC4, "read")


def test_memregion_watch_reg_relative_defaults_and_explicit_pc():
    """offset/width/kind default; an explicit point_watch_pc overrides the arm pc."""
    adapter = _RecordingAdapter()
    runner = UnidbgSignRunner(adapter)
    runner.invoke(
        RunnerCapability.MEMREGION_WATCH,
        input_bytes=b"x", pc=0x1000, base_reg="x24", point_watch_pc=0x70F84)
    w = adapter.seen[0].mem_regrel[0]
    assert (w.base_reg, w.offset, w.width, w.kind) == ("x24", 0, 8, "read")
    assert w.pc == 0x70F84            # explicit point_watch_pc wins
    assert adapter.seen[0].pc == 0x1000


def test_memregion_watch_concrete_form_still_works():
    """② concrete invoke (base/size) → ObservePoint.mem, no mem_regrel (zero regression)."""
    adapter = _RecordingAdapter()
    runner = UnidbgSignRunner(adapter)
    runner.invoke(
        RunnerCapability.MEMREGION_WATCH,
        input_bytes=b"x", pc=0x4000, base=0x7F00, size=4)
    [pt] = adapter.seen
    assert pt.mem == ((0x7F00, 4),)
    assert pt.mem_regrel == ()


def test_memregion_watch_mixed_forms_rejected():
    """Both forms at once is an ambiguity → ValueError, not a silent either-or."""
    runner = UnidbgSignRunner(_RecordingAdapter())
    with pytest.raises(ValueError, match="mutually exclusive"):
        runner.invoke(
            RunnerCapability.MEMREGION_WATCH,
            input_bytes=b"x", pc=0x4000, base=0x7F00, size=4,
            base_reg="x19", offset=0x38)


def test_snapshots_consumable_by_oracle_sink():
    """The whole point: the converted snapshots locate the output via 'snapshot'."""
    from engine.oracle_sink import SinkVerdict, validate_sink
    runner = UnidbgSignRunner(_MemCapturingAdapter())
    out = runner.invoke(
        RunnerCapability.MEMREGION_WATCH,
        input_bytes=b"x", pc=0x4000, base=0x7f00, size=4)
    sv = validate_sink([], b"\xde\xad\xbe\xef", snapshots=out.mem_snapshots)
    assert sv.verdict is SinkVerdict.SINK_CONFIRMED
    assert sv.located_via == "snapshot"
    assert sv.base == 0x7f00
