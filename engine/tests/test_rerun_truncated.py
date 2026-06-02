"""Tests: runner record-cap → RerunResult.truncated → consumer WARN + dirty mark.

Covers the construct-symmetry contract (contracts/runner_interface.md §3.2 撞 cap):
a runner that hits a record cap (e.g. X25_REGREL_CONCRETE_WRITE_MAX) MUST set
``truncated=True`` so the engine's consumer side WARNs (top-level, never silent)
and stamps every derived MemSnapshot ``truncated=True`` — the incomplete ledger
is NOT consumable as complete/clean provenance.

Zero-regression half: ``truncated=False`` (the default) leaves behaviour and the
derived snapshots completely unchanged.
"""

from __future__ import annotations

import logging

from engine.recapture import observations_to_snapshots
from engine.runner_client import (
    ObservedState,
    RerunResult,
    SubprocessRunnerAdapter,
    mem_snapshots_from_rerun,
)


def _result(truncated: bool, detail=None) -> RerunResult:
    return RerunResult(
        output=b"\x00" * 8,
        observations=(
            ObservedState(pc=0x70EC4, when="after", regs={},
                          mem={0x7F00: b"\xde\xad\xbe\xef"}),
        ),
        truncated=truncated,
        truncated_detail=detail,
    )


# --- ① truncated=True → WARN + derived snapshots stamped truncated -----------

def test_mem_snapshots_truncated_warns_and_marks(caplog):
    res = _result(True, detail={"cap": "X25_REGREL_CONCRETE_WRITE_MAX",
                                "limit": 8192, "kind": "write"})
    with caplog.at_level(logging.WARNING):
        snaps = mem_snapshots_from_rerun(res)
    # WARN surfaced at top level (not silent).
    assert any("TRUNCATED" in r.message for r in caplog.records)
    # The detail rode along into the WARN.
    assert any("X25_REGREL_CONCRETE_WRITE_MAX" in r.getMessage()
               for r in caplog.records)
    # Every derived snapshot is stamped truncated — downstream knows it is
    # NOT complete/clean provenance.
    assert snaps and all(s.truncated for s in snaps)


def test_observations_to_snapshots_truncated_warns_and_marks(caplog):
    res = _result(True)
    with caplog.at_level(logging.WARNING):
        snaps = observations_to_snapshots(res)
    assert any("TRUNCATED" in r.message for r in caplog.records)
    assert snaps and all(s.truncated for s in snaps)


# --- ② truncated=False (default) → zero regression ---------------------------

def test_mem_snapshots_not_truncated_unchanged(caplog):
    res = _result(False)
    with caplog.at_level(logging.WARNING):
        snaps = mem_snapshots_from_rerun(res)
    # No truncation WARN.
    assert not any("TRUNCATED" in r.message for r in caplog.records)
    # Snapshots produced exactly as before, with truncated=False.
    assert snaps and all(s.truncated is False for s in snaps)
    assert snaps[0].addr == 0x7F00
    assert snaps[0].data == b"\xde\xad\xbe\xef"
    assert snaps[0].source == "snapshot"


def test_observations_to_snapshots_not_truncated_unchanged(caplog):
    res = _result(False)
    with caplog.at_level(logging.WARNING):
        snaps = observations_to_snapshots(res)
    assert not any("TRUNCATED" in r.message for r in caplog.records)
    assert snaps and all(s.truncated is False for s in snaps)
    assert snaps[0].source == "recapture"


def test_default_rerun_result_is_not_truncated():
    res = RerunResult(output=b"\x00")
    assert res.truncated is False
    assert res.truncated_detail is None


# --- ③ to_dict / export exposes truncated ------------------------------------

def test_to_dict_exposes_truncated_default():
    d = RerunResult(output=b"\xab\xcd").to_dict()
    assert d["truncated"] is False
    assert d["output_hex"] == "abcd"
    assert d["n_observations"] == 0
    # Detail omitted when absent.
    assert "truncated_detail" not in d


def test_to_dict_exposes_truncated_with_detail():
    detail = {"cap": "X25_REGREL_CONCRETE_WRITE_MAX", "limit": 8192,
              "kind": "write", "dropped": 312}
    d = _result(True, detail=detail).to_dict()
    assert d["truncated"] is True
    assert d["truncated_detail"] == detail
    assert d["n_observations"] == 1


# --- adapter plumbing: subprocess runner wire → RerunResult.truncated --------

def _adapter_with_call(response: dict) -> SubprocessRunnerAdapter:
    """A SubprocessRunnerAdapter that never spawns a process: built via __new__
    with ``_call`` stubbed to return a canned wire ``response`` dict."""
    adapter = SubprocessRunnerAdapter.__new__(SubprocessRunnerAdapter)
    adapter._call = lambda method, params=None: response  # type: ignore[attr-defined]
    return adapter


def test_subprocess_rerun_plumbs_truncated_from_wire():
    detail = {"cap": "X25_REGREL_CONCRETE_WRITE_MAX", "limit": 8192}
    adapter = _adapter_with_call({
        "output_hex": "deadbeef",
        "observations": [],
        "truncated": True,
        "truncated_detail": detail,
    })
    res = adapter.rerun(b"x")
    assert res.output == b"\xde\xad\xbe\xef"
    assert res.truncated is True
    assert res.truncated_detail == detail


def test_subprocess_rerun_defaults_truncated_false_when_absent():
    # A runner that doesn't send the field at all → not truncated (back-compat).
    adapter = _adapter_with_call({"output_hex": "00", "observations": []})
    res = adapter.rerun(b"x")
    assert res.truncated is False
    assert res.truncated_detail is None


def test_subprocess_rerun_coerces_nondict_detail():
    adapter = _adapter_with_call({
        "output_hex": "00", "observations": [],
        "truncated": True, "truncated_detail": "write cap hit",
    })
    res = adapter.rerun(b"x")
    assert res.truncated is True
    assert res.truncated_detail == {"detail": "write cap hit"}
