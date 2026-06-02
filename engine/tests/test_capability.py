"""capability_request.md §P0-1 / §P0-2 — instruction-level observation
+ dynamic trace window tests."""

from __future__ import annotations

from engine.capability import (
    CodeHookRange,
    HookSanity,
    TraceWindow,
    evaluate_hook_sanity,
    in_any_window,
    merge_windows,
    register_trace_from_instructions,
)
from engine.types import Instruction


def _mk(idx: int, pc: int, regs_read: dict[str, int],
        regs_write: dict[str, int]) -> Instruction:
    return Instruction(
        idx=idx, pc=pc, bytes_=b"\x00" * 4, mnemonic="",
        regs_read=regs_read, regs_write=regs_write, mem=(),
    )


def test_register_trace_in_band_emits_only_band_instructions():
    items = [
        _mk(0, 0x10000, {"x0": 1}, {"x0": 2}),
        _mk(1, 0x322c90, {"x22": 0x54}, {"x22": 0x22}),
        _mk(2, 0x322c94, {"x22": 0x22}, {"x22": 0x2a}),
        _mk(3, 0x40000, {"x0": 99}, {"x0": 100}),
    ]
    hook = CodeHookRange(start_pc=0x322c00, end_pc=0x322d00, regs=("x22",),
                         step="every")
    t = register_trace_from_instructions(items, hook)
    assert len(t.entries) == 2
    assert {e.pc for e in t.entries} == {0x322c90, 0x322c94}


def test_register_trace_on_change_filters_steady_steps():
    items = [
        _mk(0, 0x322c90, {"x22": 0x54}, {"x22": 0x54}),  # no change
        _mk(1, 0x322c94, {"x22": 0x54}, {"x22": 0x54}),  # no change
        _mk(2, 0x322c98, {"x22": 0x54}, {"x22": 0x22}),  # change!
        _mk(3, 0x322c9c, {"x22": 0x22}, {"x22": 0x22}),  # no change
    ]
    hook = CodeHookRange(0x322c00, 0x322d00, regs=("x22",), step="on_change")
    t = register_trace_from_instructions(items, hook)
    # first entry emitted (baseline), then one for the change
    assert len(t.entries) == 2
    assert t.entries[-1].regs["x22"] == 0x22


def test_unique_register_values_in_order():
    items = [
        _mk(0, 0x100, {}, {"x22": 0x54}),
        _mk(1, 0x104, {}, {"x22": 0x22}),
        _mk(2, 0x108, {}, {"x22": 0x54}),  # repeat
        _mk(3, 0x10c, {}, {"x22": 0x2a}),
    ]
    hook = CodeHookRange(0x100, 0x200, regs=("x22",), step="every")
    t = register_trace_from_instructions(items, hook)
    assert t.unique_register_values("x22") == (0x54, 0x22, 0x2a)


def test_first_change_idx_finds_compress_leg_transition():
    items = [
        _mk(0, 0x100, {"x22": 0x54}, {"x22": 0x54}),
        _mk(1, 0x104, {"x22": 0x54}, {"x22": 0x54}),
        _mk(2, 0x108, {"x22": 0x54}, {"x22": 0x22}),
        _mk(3, 0x10c, {"x22": 0x22}, {"x22": 0x22}),
    ]
    hook = CodeHookRange(0x100, 0x200, regs=("x22",), step="every")
    t = register_trace_from_instructions(items, hook)
    assert t.first_change_idx("x22") == 2


def test_hook_sanity_flags_constant_buffer():
    """3 inputs, all producing identical x22 trace ⇒ invalid_constant."""
    hook = CodeHookRange(0x100, 0x200, regs=("x22",), step="every")

    items_const = [
        _mk(0, 0x100, {}, {"x22": 0xe9a86ab9}),
        _mk(1, 0x104, {}, {"x22": 0xe9a86ab9}),
    ]
    t = register_trace_from_instructions(items_const, hook)
    sanity = evaluate_hook_sanity([t, t, t])
    assert sanity.invalid_constant is True
    assert sanity.unique_count == 1


def test_hook_sanity_passes_on_varying_inputs():
    hook = CodeHookRange(0x100, 0x200, regs=("x22",), step="every")
    traces = []
    for v in (0x11111111, 0x22222222, 0x33333333):
        items = [_mk(0, 0x100, {}, {"x22": v})]
        traces.append(register_trace_from_instructions(items, hook))
    sanity = evaluate_hook_sanity(traces)
    assert sanity.invalid_constant is False
    assert sanity.unique_count == 3


def test_merge_windows_collapses_overlap():
    primary = (0x10000, 0x20000)
    extra = (
        TraceWindow(0x15000, 0x25000),  # overlaps primary's tail
        TraceWindow(0x32302c, 0x325708),  # main-VMP band — disjoint
    )
    merged = merge_windows(primary, extra)
    assert merged == ((0x10000, 0x25000), (0x32302c, 0x325708))


def test_in_any_window():
    bands = ((0x10000, 0x20000), (0x32302c, 0x325708))
    assert in_any_window(0x15000, bands)
    assert in_any_window(0x32302c, bands)
    assert not in_any_window(0x21000, bands)
    assert not in_any_window(0x325708, bands)  # end exclusive


def test_runner_adapter_code_hook_range_default_raises():
    """A vanilla NullRunnerAdapter must raise NotImplementedError so
    callers know to fall back to get_trace + synthesis."""
    from engine.runner_client import NullRunnerAdapter
    from engine.types import TargetMeta

    tm = TargetMeta(
        target_name="t", arch="arm64",
        algo_entry_pc=0x100, algo_exit_pc=0x200,
        input_length=None, output_length=32,
    )
    r = NullRunnerAdapter(tm)
    hook = CodeHookRange(0x100, 0x200, regs=("x0",), step="every")
    try:
        r.code_hook_range(b"\x00", [hook])
    except NotImplementedError:
        return
    raise AssertionError("expected NotImplementedError")
