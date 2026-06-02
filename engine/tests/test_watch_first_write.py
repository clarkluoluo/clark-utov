"""Watch-first-write primitive + auto-suggestion gate — acceptance tests."""

from __future__ import annotations

import pytest

from engine.discipline_wrapper import DisciplineWrapper
from engine.methodology import MethodologyConfig
from engine.watch_first_write import (
    WatchFirstWriteConfig,
    WatchFirstWriteResult,
    WatchFirstWriteSpec,
    maybe_suggest_watch,
    request_point_watch,
    request_watch_first_write,
    suggest_watches_in_params,
)


# ---------------------------------------------------------------------------
# Point-watch (PC-gated reg-relative single-point capture)
# ---------------------------------------------------------------------------


def test_point_watch_spec_shape():
    """A point-watch carries the PC gate + capture direction + reg-relative
    addressing; its dict reports kind=point_watch with watch_kind + pc."""
    spec = request_point_watch(
        0x70ec4, "x19", 0x38, 8, "tmp", kind="read",
        cfg=WatchFirstWriteConfig())
    assert spec.is_point_watch is True
    assert spec.is_reg_relative is True
    assert spec.pc == 0x70ec4
    assert spec.kind == "read"
    assert spec.addr_expr == "[x19 + 0x38]"
    d = spec.to_dict()
    assert d["kind"] == "point_watch"
    assert d["watch_kind"] == "read"
    assert d["pc"] == 0x70ec4
    assert d["pc_hex"] == "0x70ec4"
    assert d["addressing"] == "reg_relative"
    assert d["base_reg"] == "x19"
    assert d["offset"] == 0x38
    assert d["width_bytes"] == 8


def test_point_watch_write_kind():
    spec = request_point_watch(
        0x70f84, "x24", 0x84, 4, "v", kind="write",
        cfg=WatchFirstWriteConfig())
    assert spec.kind == "write"
    assert spec.to_dict()["watch_kind"] == "write"
    assert spec.to_dict()["width_bytes"] == 4


def test_point_watch_requires_reg_relative():
    """A point-watch must be reg-relative: a concrete-address point-watch is
    meaningless (no live register to resolve) and is rejected."""
    with pytest.raises(ValueError, match="reg-relative"):
        request_watch_first_write(
            0x1000, "v", pc=0x70ec4, kind="read", cfg=WatchFirstWriteConfig())


def test_point_watch_kind_must_be_read_or_write():
    with pytest.raises(ValueError):
        request_point_watch(0x70ec4, "x0", 0, 8, "v", kind="first_write",
                            cfg=WatchFirstWriteConfig())


def test_first_write_kind_with_pc_rejected():
    with pytest.raises(ValueError, match="point-watch needs kind"):
        request_watch_first_write(
            0x1000, "v", base_reg="x0", pc=0x70ec4, kind="first_write",
            cfg=WatchFirstWriteConfig())


def test_read_write_kind_without_pc_rejected():
    with pytest.raises(ValueError, match="no pc"):
        request_watch_first_write(
            0x1000, "v", base_reg="x0", kind="read", cfg=WatchFirstWriteConfig())


def test_first_write_spec_unchanged_no_point_fields():
    """④ zero regression: a plain first-write spec has pc=None, kind=first_write,
    and its dict carries NONE of the point-watch keys."""
    spec = request_watch_first_write(0x4000, "v", cfg=WatchFirstWriteConfig())
    assert spec.pc is None
    assert spec.kind == "first_write"
    assert spec.is_point_watch is False
    d = spec.to_dict()
    assert d["kind"] == "watch_first_write"
    assert "watch_kind" not in d and "pc" not in d and "pc_hex" not in d


# ---------------------------------------------------------------------------
# Primitive
# ---------------------------------------------------------------------------


def test_request_watch_first_write_builds_spec():
    spec = request_watch_first_write(
        0xbadc0de, "vmp_key",
        reason="trace producer",
        cfg=WatchFirstWriteConfig(),
    )
    assert isinstance(spec, WatchFirstWriteSpec)
    assert spec.addr == 0xbadc0de
    assert spec.value_name == "vmp_key"
    d = spec.to_dict()
    assert d["addr_hex"] == "0xbadc0de"
    assert d["kind"] == "watch_first_write"


def test_request_watch_first_write_rejects_bad_addr():
    with pytest.raises(ValueError):
        request_watch_first_write(-1, "x", cfg=WatchFirstWriteConfig())


def test_request_watch_first_write_disabled_raises():
    with pytest.raises(RuntimeError, match="disabled"):
        request_watch_first_write(
            0x1000, "x",
            cfg=WatchFirstWriteConfig(enabled=False),
        )


def test_watch_result_round_trip():
    spec = request_watch_first_write(0x4000, "v",
                                     cfg=WatchFirstWriteConfig())
    r = WatchFirstWriteResult(
        spec=spec, first_write_pc=0x12345, source_bytes=b"\xde\xad\xbe\xef",
    )
    d = r.to_dict()
    assert d["first_write_pc_hex"] == "0x12345"
    assert d["source_bytes"] == "deadbeef"


# ---------------------------------------------------------------------------
# Auto-suggestion gate
# ---------------------------------------------------------------------------


def test_suggests_when_observed_value_has_landing_addr():
    s = maybe_suggest_watch(
        {"value_name": "key", "source": "hook",
         "landing_address": 0x7ffabc1000,
         "recompute_fn_present": False},
        cfg=WatchFirstWriteConfig(),
    )
    assert s is not None
    assert s.spec is not None
    assert s.spec.addr == 0x7ffabc1000
    assert "0x7ffabc1000" in s.advisory


def test_no_suggestion_when_closed_form_already_verified():
    s = maybe_suggest_watch(
        {"value_name": "k", "source": "formula",
         "landing_address": 0x4000,
         "recompute_fn_present": True,
         "recompute_matches_measured": True},
        cfg=WatchFirstWriteConfig(),
    )
    assert s is None


def test_no_suggestion_without_landing_addr():
    s = maybe_suggest_watch(
        {"value_name": "k", "source": "hook"},
        cfg=WatchFirstWriteConfig(),
    )
    assert s is None


def test_advisory_only_mode_emits_no_spec():
    cfg = WatchFirstWriteConfig(auto_trigger=False)
    s = maybe_suggest_watch(
        {"value_name": "k", "source": "dump", "landing_address": 0x1000},
        cfg=cfg,
    )
    assert s is not None
    assert s.spec is None
    assert "watch_first_write" in s.advisory


def test_provenance_observed_flag_enough_to_trigger():
    """Even when ``source`` is missing, an upstream provenance tag of
    'observed' should make the value eligible."""
    s = maybe_suggest_watch(
        {"value_name": "k", "provenance": "observed",
         "landing_address": 0x1000},
        cfg=WatchFirstWriteConfig(),
    )
    assert s is not None


def test_env_toggle_off_disables_gate():
    cfg = WatchFirstWriteConfig.from_env({"UTOV_WATCH_FIRST_WRITE": "off"})
    assert cfg.enabled is False
    s = maybe_suggest_watch(
        {"value_name": "k", "source": "hook", "landing_address": 0x1000},
        cfg=cfg,
    )
    assert s is None


def test_env_toggle_auto_trigger_off_emits_advisory_only():
    cfg = WatchFirstWriteConfig.from_env(
        {"UTOV_WATCH_FIRST_WRITE_AUTO_TRIGGER": "off"},
    )
    assert cfg.auto_trigger is False


def test_walk_params_collects_multiple_suggestions():
    params = {
        "report": {
            "values": [
                {"value_name": "k1", "source": "hook",
                 "landing_address": 0x1000},
                {"value_name": "k2", "source": "formula",
                 "recompute_fn_present": True,
                 "recompute_matches_measured": True,
                 "landing_address": 0x2000},
                {"value_name": "k3", "source": "dump",
                 "landing_address": 0x3000},
            ],
        },
    }
    out = suggest_watches_in_params(params, cfg=WatchFirstWriteConfig())
    assert {s.value_name for s in out} == {"k1", "k3"}


# ---------------------------------------------------------------------------
# Wrapper integration
# ---------------------------------------------------------------------------


def test_wrapper_attaches_watch_suggestion_to_envelope():
    wrapper = DisciplineWrapper(config=MethodologyConfig())
    params = {
        "values": [
            {"value_name": "vmp_key", "source": "hook",
             "landing_address": 0xbadc0de},
        ],
    }
    _, env = wrapper.step("submit_finding", params, lambda m, p: None)
    assert env.watch_suggestions
    assert env.watch_suggestions[0]["value_name"] == "vmp_key"
    assert env.watch_suggestions[0]["spec"]["addr"] == 0xbadc0de
    assert any("WATCH-FIRST-WRITE" in a for a in env.alerts)
