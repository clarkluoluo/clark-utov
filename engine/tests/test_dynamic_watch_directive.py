"""B3 — dynamic_watch_batch directive CONTRACT tests.

engine 开处方, runner 抓药. These tests pin the engine-side *shape* only:
the dataclass + serialization (to_dict/from_dict round-trip incl. mem_regrel) +
field validation (illegal snapshot_policy rejected, G1 coupling) + that the
observe-point shape is REUSED from runner_client (not a second shape) + a
capability-wall WARN. They assert engine does NOT implement attach/capture/rerun
(that is the runner's job, a separate repo).

Spec: todo/dev-dynamic-watch-directive-contract-spec.md
"""

from __future__ import annotations

import logging

import pytest

from engine.recapture import (
    DYNAMIC_WATCH_BATCH_KIND,
    DYNAMIC_WATCH_SNAPSHOT_POLICIES,
    DynamicWatchBatch,
    RecaptureSpec,
    dynamic_watch_batch_from_spec,
    warn_if_over_runner_capacity,
)
from engine.runner_client import (
    ObservePoint,
    RegRelWatch,
    SubprocessRunnerAdapter,
)


def _regrel_op() -> ObservePoint:
    """An observe point carrying both a register-offset mem_regrel (B6 index/scale)
    and a plain [base+offset] one, so round-trip must preserve index/scale."""
    return ObservePoint(
        pc=0x40007D88,
        when="before",
        capture=("mem",),
        regs=("x0", "x1"),
        mem_regrel=(
            RegRelWatch(base_reg="x20", offset=0x18, width=8, pc=0x40007D88,
                        kind="read", index="x9", scale=3),
            RegRelWatch(base_reg="sp", offset=-0x40, width=4, pc=0x40007D88,
                        kind="write"),
        ),
    )


def _concrete_op() -> ObservePoint:
    return ObservePoint(
        pc=0x221068,
        when="after",
        capture=("regs", "mem"),
        regs=("x0", "x2"),
        mem=((0xBFFFF700, 16), (0xBFFFF800, 4)),
    )


# --- round-trip -------------------------------------------------------------

def test_round_trip_lossless_incl_mem_regrel():
    """to_dict → from_dict reconstructs the directive byte-for-byte, including each
    observe point's mem_regrel with index/scale (A4: a dropped field breaks replay)."""
    d = DynamicWatchBatch(observe_points=(_regrel_op(), _concrete_op()))
    back = DynamicWatchBatch.from_dict(d.to_dict())
    assert back == d
    # and stable under a second pass
    assert back.to_dict() == d.to_dict()


def test_round_trip_preserves_index_scale_explicitly():
    d = DynamicWatchBatch(observe_points=(_regrel_op(),))
    w = DynamicWatchBatch.from_dict(d.to_dict()).observe_points[0].mem_regrel
    assert w[0].index == "x9" and w[0].scale == 3 and w[0].kind == "read"
    assert w[0].offset == 0x18 and w[0].width == 8 and w[0].pc == 0x40007D88
    # the plain form round-trips with index=None / scale=0
    assert w[1].index is None and w[1].scale == 0 and w[1].kind == "write"
    assert w[1].offset == -0x40


def test_to_dict_carries_g1_fields_and_kind():
    d = DynamicWatchBatch(observe_points=(_concrete_op(),))
    wire = d.to_dict()
    assert wire["kind"] == DYNAMIC_WATCH_BATCH_KIND == "dynamic_watch_batch"
    assert wire["must_capture_output_same_rerun"] is True
    assert wire["expected_source"] == "rr.output"
    assert wire["snapshot_policy"] == "same_execution_only"


# --- validation: illegal snapshot_policy rejected ---------------------------

def test_illegal_snapshot_policy_rejected():
    with pytest.raises(ValueError, match="snapshot_policy"):
        DynamicWatchBatch(observe_points=(_concrete_op(),),
                          snapshot_policy="cross_rerun_accumulate")


def test_illegal_snapshot_policy_rejected_via_from_dict():
    d = DynamicWatchBatch(observe_points=(_concrete_op(),)).to_dict()
    d["snapshot_policy"] = "bogus"
    with pytest.raises(ValueError, match="snapshot_policy"):
        DynamicWatchBatch.from_dict(d)


def test_legal_policy_set_fixed_and_includes_same_execution_only():
    assert "same_execution_only" in DYNAMIC_WATCH_SNAPSHOT_POLICIES


def test_empty_observe_points_rejected():
    with pytest.raises(ValueError, match="at least one observe point"):
        DynamicWatchBatch(observe_points=())


def test_g1_coupling_output_must_be_same_rerun():
    """same_execution_only with must_capture_output_same_rerun=False is incoherent
    (the output's nonce could come from another rerun) → rejected by construction."""
    with pytest.raises(ValueError, match="must_capture_output_same_rerun"):
        DynamicWatchBatch(observe_points=(_concrete_op(),),
                          must_capture_output_same_rerun=False)


def test_from_dict_rejects_wrong_kind():
    with pytest.raises(ValueError, match="kind"):
        DynamicWatchBatch.from_dict({"kind": "something_else",
                                     "observe_points": []})


# --- observe_points REUSE the runner_client shape (not a second shape) ------

def test_observe_point_wire_shape_matches_runner_client():
    """The directive's per-point keys must be the SAME set the rerun wire path
    (SubprocessRunnerAdapter._serialize_observe_point) emits — one shape, by
    construction. (Value conventions differ: the rerun path UPPERCASEs when/capture
    for the Java enums; we assert KEY parity, the structural contract.)"""
    op = _regrel_op()
    directive_pt = DynamicWatchBatch(observe_points=(op,)).to_dict()["observe_points"][0]
    runner_pt = SubprocessRunnerAdapter._serialize_observe_point(op)
    assert set(directive_pt) == set(runner_pt)
    # mem_regrel per-watch keys are identical too (shared _regrel_to_wire)
    assert set(directive_pt["mem_regrel"][0]) == set(runner_pt["mem_regrel"][0])


def test_observe_point_is_runner_client_type():
    d = DynamicWatchBatch(observe_points=(_concrete_op(),))
    assert all(isinstance(op, ObservePoint) for op in d.observe_points)
    back = DynamicWatchBatch.from_dict(d.to_dict())
    assert all(isinstance(op, ObservePoint) for op in back.observe_points)


# --- generation from a (provenance-prefilled) spec --------------------------

def test_generate_from_spec_uses_spec_observe_points():
    spec = RecaptureSpec(input=b"\x01\x02",
                         observe_points=[_regrel_op(), _concrete_op()])
    d = dynamic_watch_batch_from_spec(spec)
    assert tuple(spec.observe_points) == d.observe_points
    assert d.snapshot_policy == "same_execution_only"
    # generated directive is itself a valid, round-trippable prescription
    assert DynamicWatchBatch.from_dict(d.to_dict()) == d


def test_generate_from_spec_validates_eagerly():
    """A spec with no observe points → generation fails loudly (no silent empty
    prescription)."""
    spec = RecaptureSpec(input=b"\x01", observe_points=[])
    with pytest.raises(ValueError, match="at least one observe point"):
        dynamic_watch_batch_from_spec(spec)


# --- capability-wall WARN (runner reports its ceiling; engine does not know) -

def test_over_capacity_warns_loud(caplog):
    d = DynamicWatchBatch(observe_points=(_regrel_op(), _concrete_op()))
    with caplog.at_level(logging.WARNING, logger="engine.recapture"):
        over = warn_if_over_runner_capacity(d, {"max_watch_points": 1})
    assert over is True
    assert any("EXCEEDED runner watch capacity" in r.message for r in caplog.records)


def test_over_capacity_explicit_boolean_warns():
    d = DynamicWatchBatch(observe_points=(_concrete_op(),))
    assert warn_if_over_runner_capacity(d, {"watch_capacity_exceeded": True}) is True


def test_over_capacity_accepted_less_than_requested_warns():
    d = DynamicWatchBatch(observe_points=(_regrel_op(), _concrete_op()))
    assert warn_if_over_runner_capacity(d, {"accepted_watch_points": 1}) is True


def test_within_capacity_no_warn(caplog):
    d = DynamicWatchBatch(observe_points=(_concrete_op(),))
    with caplog.at_level(logging.WARNING, logger="engine.recapture"):
        over = warn_if_over_runner_capacity(d, {"max_watch_points": 8,
                                                "accepted_watch_points": 1})
    assert over is False
    assert not any("EXCEEDED" in r.message for r in caplog.records)


def test_shapeless_response_reports_nothing():
    d = DynamicWatchBatch(observe_points=(_concrete_op(),))
    assert warn_if_over_runner_capacity(d, None) is False
    assert warn_if_over_runner_capacity(d, {}) is False


# --- engine is CONTRACT-only: no attach / capture / rerun implementation -----

def test_engine_directive_module_has_no_runner_implementation():
    """engine 不抓药: the recapture module exposes the directive shape + generator +
    capacity WARN, but NO attach-observe-point / batch-capture / rerun primitive.
    (dispatch_recapture is an explicit DEFERRED NotImplementedError seam, not an
    implementation.)"""
    import inspect

    import engine.recapture as mod

    src = inspect.getsource(mod)
    # No directive helper performs a rerun or attaches/captures itself.
    for name in ("dynamic_watch_batch_from_spec", "warn_if_over_runner_capacity"):
        fn_src = inspect.getsource(getattr(mod, name))
        assert "adapter.rerun" not in fn_src
        assert ".rerun(" not in fn_src
    # The one rerun reference in the module is the DEFERRED dispatch seam, which
    # raises NotImplementedError rather than implementing the runner side.
    assert "NotImplementedError" in inspect.getsource(mod.dispatch_recapture)
    # The directive dataclass carries no method that runs the target.
    method_names = {n for n, _ in inspect.getmembers(DynamicWatchBatch,
                                                     predicate=inspect.isfunction)}
    assert "rerun" not in method_names
    assert "capture" not in method_names
    assert "attach" not in method_names
    assert src  # module is importable / non-empty
