"""block_cause — classifier + router + capability oracle + backlog."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.block_cause import (
    ANCHOR_TO_CAPABILITY,
    BacklogEntry,
    BacklogWriter,
    BlockCauseClass,
    BlockCauseRouter,
    NodeContext,
    RoutingAction,
    StaticCapabilityOracle,
    capability_for_anchor,
    classify,
    oracle_from_adapter,
    render_block_cause_alert,
)
from engine.phase import (
    ANCHOR_ADDR_FIRST_EXEC,
    ANCHOR_FUNC_ENTRY,
    ANCHOR_MEMREGION_FIRST_ACCESS,
    Anchor,
    PhaseBoundary,
)
from engine.phase_discovery import PhaseDiscoveryResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _discovery_crossing_out(reason: str = "no_writer_in_any_source") -> PhaseDiscoveryResult:
    return PhaseDiscoveryResult(
        value_addr=0xbabe0000,
        boundary=PhaseBoundary(
            name="producer",
            region=(0xbabe0000, 32),
            anchor=Anchor(
                anchor_type=ANCHOR_MEMREGION_FIRST_ACCESS,
                params={"base": 0xbabe0000, "length": 32, "access": "w"},
            ),
        ),
        crosses_out=True,
        reason=reason,
    )


def _discovery_in_window() -> PhaseDiscoveryResult:
    return PhaseDiscoveryResult(
        value_addr=0xbabe0000,
        boundary=None,
        crosses_out=False,
        reason="in_window_writer pc=0x40010000",
    )


# ---------------------------------------------------------------------------
# Anchor → capability mapping
# ---------------------------------------------------------------------------


def test_anchor_to_capability_known_anchors():
    assert capability_for_anchor(
        Anchor(anchor_type=ANCHOR_FUNC_ENTRY, params={"pc": 0x1000})
    ) == "func_entry_hook"
    assert capability_for_anchor(
        Anchor(anchor_type=ANCHOR_ADDR_FIRST_EXEC, params={"pc": 0x1000})
    ) == "pc_first_exec_hook"
    assert capability_for_anchor(
        Anchor(anchor_type=ANCHOR_MEMREGION_FIRST_ACCESS,
               params={"base": 0xa, "length": 4, "access": "w"})
    ) == "memregion_watch"


def test_anchor_to_capability_unknown_anchor_gets_prefix():
    # We don't synthesize a custom anchor here (the registry guards
    # that), but the mapping must produce a fallback name for
    # capabilities that don't have a static entry.
    fake = type("X", (), {"anchor_type": "some_future_type", "params": {}})()
    assert capability_for_anchor(fake).startswith("anchor:") or \
           capability_for_anchor(fake) == "unknown_capability"


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def test_classifier_crossing_out_is_collection_gap():
    c = classify(_discovery_crossing_out())
    assert c.cls is BlockCauseClass.COLLECTION_GAP
    names = {s.name for s in c.signals}
    assert "phase_discovery_crosses_out" in names
    assert "zero_writers_at_address" in names


def test_classifier_in_window_with_no_context_is_true_boundary():
    """Discovery says producer is in-window, but caller gave no
    further context — there's no way to know we're past L2/L3.
    With ``data_collected`` defaulting to False the classifier
    treats this as a collection gap with weaker evidence."""
    c = classify(_discovery_in_window())
    assert c.cls is BlockCauseClass.COLLECTION_GAP


def test_classifier_recognition_gap():
    c = classify(
        node_context=NodeContext(
            node_id="n1",
            data_collected=True,
            pattern_recognised=False,
            failure_summary="template fit produced no match",
        ),
    )
    assert c.cls is BlockCauseClass.RECOGNITION_GAP


def test_classifier_strategy_gap():
    c = classify(
        node_context=NodeContext(
            node_id="n1",
            data_collected=True,
            pattern_recognised=True,
            symbolised=True,
            strategy_resolved=False,
            failure_summary="rules layered but no paradigm picks the right one",
        ),
    )
    assert c.cls is BlockCauseClass.STRATEGY_GAP


def test_classifier_true_boundary():
    c = classify(
        node_context=NodeContext(
            node_id="n1",
            data_collected=True,
            pattern_recognised=True,
            symbolised=True,
            strategy_resolved=True,
            failure_summary="all layers ran; ambiguity remains",
        ),
    )
    assert c.cls is BlockCauseClass.TRUE_BOUNDARY


# ---------------------------------------------------------------------------
# Capability oracle
# ---------------------------------------------------------------------------


def test_static_oracle_yes_no():
    o = StaticCapabilityOracle(static=frozenset({"memregion_watch"}))
    assert o.has("memregion_watch") is True
    assert o.has("pc_first_exec_hook") is False


def test_static_oracle_metadata_override_adds_capability():
    o = StaticCapabilityOracle(
        static=frozenset({"a"}),
        metadata_override=frozenset({"b"}),
    )
    assert o.has("a") is True
    assert o.has("b") is True   # override adds


def test_oracle_from_adapter_reads_static_capabilities():
    class FakeAdapter:
        CAPABILITIES = frozenset({"func_entry_hook"})

    o = oracle_from_adapter(FakeAdapter())
    assert o.has("func_entry_hook") is True
    assert o.has("memregion_watch") is False


def test_oracle_from_adapter_metadata_overrides_static():
    class FakeAdapter:
        CAPABILITIES = frozenset({"func_entry_hook"})

    class FakeMeta:
        capabilities = ["memregion_watch"]

    o = oracle_from_adapter(FakeAdapter(), metadata=FakeMeta())
    assert o.has("func_entry_hook") is True   # static still in
    assert o.has("memregion_watch") is True   # added by metadata


def test_oracle_from_adapter_handles_dict_metadata():
    class FakeAdapter:
        CAPABILITIES = frozenset()

    o = oracle_from_adapter(
        FakeAdapter(),
        metadata={"capabilities": ["memregion_watch"]},
    )
    assert o.has("memregion_watch") is True


# ---------------------------------------------------------------------------
# Backlog writer
# ---------------------------------------------------------------------------


def test_backlog_writer_appends_jsonl(tmp_path):
    path = tmp_path / "capability_backlog.jsonl"
    w = BacklogWriter(path)
    e1 = BacklogEntry(
        gap_kind="needs_collection_capability",
        missing_capability="memregion_watch",
        node_id="n1",
        trigger_evidence={"value_addr_hex": "0xbabe0000"},
        timestamp=1.0,
        run_dir="/work/run1",
    )
    w.append(e1)
    w.append(BacklogEntry(
        gap_kind="needs_collection_capability",
        missing_capability="func_entry_hook",
        node_id="n2",
        trigger_evidence={},
        timestamp=2.0,
        run_dir="/work/run1",
    ))
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["missing_capability"] == "memregion_watch"
    assert json.loads(lines[1])["missing_capability"] == "func_entry_hook"


def test_backlog_writer_read_all_round_trip(tmp_path):
    path = tmp_path / "capability_backlog.jsonl"
    w = BacklogWriter(path)
    w.append(BacklogEntry(
        gap_kind="needs_collection_capability",
        missing_capability="memregion_watch",
        node_id="n",
        trigger_evidence={"k": "v"},
        timestamp=42.0,
    ))
    rows = w.read_all()
    assert len(rows) == 1
    assert rows[0].missing_capability == "memregion_watch"
    assert rows[0].trigger_evidence == {"k": "v"}


def test_backlog_writer_no_path_is_in_memory_only(tmp_path):
    w = BacklogWriter(None)
    entry = BacklogEntry(
        gap_kind="x",
        missing_capability="y",
        node_id="n",
        trigger_evidence={},
    )
    returned = w.append(entry)
    assert returned is entry
    assert w.read_all() == []


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def test_router_class1_capability_present_yields_auto_collect(tmp_path):
    router = BlockCauseRouter(
        oracle=StaticCapabilityOracle(static=frozenset({"memregion_watch"})),
        backlog_writer=BacklogWriter(tmp_path / "backlog.jsonl"),
    )
    r = router.route(_discovery_crossing_out())
    assert r.action is RoutingAction.AUTO_COLLECT
    assert r.rerun_request is not None
    assert r.rerun_request.instrument_spec.anchor.anchor_type == "memregion_first_access"
    assert r.backlog_entry is None
    # backlog file must NOT have been written.
    assert not (tmp_path / "backlog.jsonl").exists()


def test_router_class1_capability_missing_writes_backlog(tmp_path):
    backlog = tmp_path / "backlog.jsonl"
    router = BlockCauseRouter(
        oracle=StaticCapabilityOracle(static=frozenset()),    # nothing
        backlog_writer=BacklogWriter(backlog),
        run_dir=str(tmp_path),
    )
    r = router.route(_discovery_crossing_out())
    assert r.action is RoutingAction.REGISTER_BACKLOG
    assert r.backlog_entry is not None
    assert r.backlog_entry.missing_capability == "memregion_watch"
    assert r.backlog_entry.run_dir == str(tmp_path)
    assert backlog.exists()
    rows = BacklogWriter(backlog).read_all()
    assert len(rows) == 1
    assert rows[0].missing_capability == "memregion_watch"
    # The suggested spec is attached for the developer reading the
    # backlog to see what shape the runner would need to fulfil.
    assert r.backlog_entry.suggested_spec is not None


def test_router_class1_without_oracle_writes_backlog(tmp_path):
    """No oracle at all → treat as capability missing (safer
    default — never auto-trigger when we can't tell)."""
    backlog = tmp_path / "backlog.jsonl"
    router = BlockCauseRouter(
        oracle=None,
        backlog_writer=BacklogWriter(backlog),
    )
    r = router.route(_discovery_crossing_out())
    assert r.action is RoutingAction.REGISTER_BACKLOG


def test_router_class2_recognition_gap_escalates_l2():
    router = BlockCauseRouter()
    r = router.route(
        node_context=NodeContext(
            node_id="n",
            data_collected=True,
            pattern_recognised=False,
        ),
    )
    assert r.action is RoutingAction.ESCALATE_L2
    assert "layer-2" in r.escalation_hint


def test_router_class2_strategy_gap_escalates_l3():
    router = BlockCauseRouter()
    r = router.route(
        node_context=NodeContext(
            node_id="n",
            data_collected=True,
            pattern_recognised=True,
            symbolised=True,
            strategy_resolved=False,
        ),
    )
    assert r.action is RoutingAction.ESCALATE_L3
    assert "layer-3" in r.escalation_hint


def test_router_class3_escalates_user_with_decision_elements():
    router = BlockCauseRouter()
    r = router.route(
        node_context=NodeContext(
            node_id="n",
            data_collected=True,
            pattern_recognised=True,
            symbolised=True,
            strategy_resolved=True,
            failure_summary="L1/L2/L3 all ran",
        ),
    )
    assert r.action is RoutingAction.ESCALATE_USER
    assert r.decision_elements is not None
    # decision elements MUST be populated — class 3 never surfaces
    # without the prepared package.
    assert r.decision_elements.missing
    assert "L1/L2/L3 all ran" in r.decision_elements.missing


def test_router_alert_includes_class_and_action():
    router = BlockCauseRouter(
        oracle=StaticCapabilityOracle(static=frozenset({"memregion_watch"})),
    )
    r = router.route(_discovery_crossing_out())
    line = render_block_cause_alert([r])
    assert line is not None
    assert "collection_gap" in line
    assert "auto_collect" in line
