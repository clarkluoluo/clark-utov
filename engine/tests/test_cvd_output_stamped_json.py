"""Consumer output = OUT-layer stamped JSON, never the COLLECT-layer sqlite.

Pins dev-consumer-output-stamped-json-not-sqlite.md: a CVD run handed to a
consumer (test-agent) produces a stamped JSON gap map (the authority
discriminator header + the structured CvdResult), and the run itself touches NO
sqlite — so no `cvd_ledger.sqlite` / `-wal` / `-shm` dead ledger is left behind.
The consumer reads the JSON; it never consumes utov's internal sqlite collect
layer. Synthetic shapes only — zero case-specific knowledge.
"""

from __future__ import annotations

import json

import pytest

from engine.cvd import (
    Candidate,
    CandidateGenerator,
    CvdOutcome,
    CvdState,
    Registry,
    Verdict,
    Verifier,
    VStatus,
    export_gap_map,
    run_cvd,
    run_cvd_collect_to_json,
)
from engine.export_stamp import (
    CONSUMER_EXPORT_AUTHORITY,
    export_stamped_json,
    is_utov_export,
    load_stamped_json,
)
from engine.setup_symex import (
    CaseConfig,
    build_concrete_backing,
    drive,
)
from engine.types import Instruction

_TS = "2026-06-01T00:00:00Z"


# --- a tiny consumer-facing collect run (no ledger) ------------------------- #

class _Gen(CandidateGenerator):
    name = "g"; version = "1"; owner = "test"; kind = "x"

    def generate(self, state):
        return [Candidate("ok", 0x10, "s", "c"),
                Candidate("gap", 0x20, "s", "needs tool")]


class _Ver(Verifier):
    name = "v"; version = "1"; owner = "test"

    def applies(self, c, state):
        return c.kind in ("ok", "gap")

    def verify(self, c, state):
        if c.kind == "ok":
            return Verdict(VStatus.CONFIRMED, evidence={"ok": True})
        return Verdict(VStatus.TERMINAL, terminal_kind="needs_tool",
                       reason="gap", capability_request="register tool X")


def _collect_result():
    reg = Registry().register(_Gen()).register(_Ver())
    return run_cvd([Instruction(0, 0x1000, b"\x00\x00\x00\x00", "nop", {}, {})],
                   b"\x00", registry=reg, collect_extensions=True)


def test_gap_map_is_stamped_json_and_loads_back():
    res = _collect_result()
    text = export_gap_map(res, ts=_TS,
                          exec_identity={"target": "synthetic.so", "run_id": "r1"})
    # the header is the authority discriminator.
    assert is_utov_export(text)
    header, payload = load_stamped_json(text)
    assert header is not None
    assert header["exported_by"] == "run_cvd"
    assert header["authority"] == CONSUMER_EXPORT_AUTHORITY
    assert header["exec_identity"]["target"] == "synthetic.so"
    # the body round-trips to the structured gap map.
    assert payload["outcome"] == CvdOutcome.COLLECTED.value
    assert len(payload["confirmed"]) == 1
    assert len(payload["extension_requests"]) == 1


def test_consumer_authority_does_not_point_at_sqlite():
    # The consumer artifact must NOT redirect the consumer back to the internal
    # sqlite ledger (that is what caused the dead-ledger / cross-layer bug).
    text = export_gap_map(_collect_result(), ts=_TS)
    header, _ = load_stamped_json(text)
    assert "为准" not in header["authority"] or "sqlite 账本" in header["authority"]
    # explicit: the standing sqlite-authority line is NOT used here.
    from engine.export_stamp import DEFAULT_AUTHORITY
    assert header["authority"] != DEFAULT_AUTHORITY


def test_collect_run_and_export_leave_no_sqlite(tmp_path):
    res = _collect_result()
    out = tmp_path / "cvd_gap_map.json"
    export_gap_map(res, out, ts=_TS)
    # the JSON artifact exists and is stamped...
    assert out.exists() and is_utov_export(out.read_text())
    # ...and NO sqlite dead ledger was produced anywhere in the work dir.
    assert list(tmp_path.glob("*.sqlite")) == []
    assert list(tmp_path.glob("*-wal")) == []
    assert list(tmp_path.glob("*-shm")) == []
    assert list(tmp_path.glob("*.sqlite-*")) == []


def test_consumer_one_shot_actively_persists_gap_map(tmp_path):
    # §③: utov ACTIVELY persists one durable JSON under work_root — the agent does
    # not have to remember to dump it (the empty-log failure mode).
    reg = Registry().register(_Gen()).register(_Ver())
    res, path = run_cvd_collect_to_json(
        [Instruction(0, 0x1000, b"\x00\x00\x00\x00", "nop", {}, {})], b"\x00",
        work_root=tmp_path / "run-out", ts=_TS,
        exec_identity={"target": "synthetic.so"}, registry=reg)
    assert res.outcome is CvdOutcome.COLLECTED
    assert path.exists() and path.name == "cvd_gap_map.json"
    assert is_utov_export(path.read_text())
    # no sqlite anywhere under the work root.
    assert list((tmp_path / "run-out").glob("*.sqlite*")) == []


def test_consumer_one_shot_refuses_a_ledger_handle():
    reg = Registry().register(_Gen()).register(_Ver())
    with pytest.raises(ValueError, match="never the sqlite ledger"):
        run_cvd_collect_to_json(
            [Instruction(0, 0x1000, b"\x00\x00\x00\x00", "nop", {}, {})], b"\x00",
            work_root="/tmp/x", ts=_TS, registry=reg, ledger=object())


def test_hand_written_file_is_discriminated_by_missing_header():
    # An agent hand-written JSON lacks the stamp → load reports header=None.
    plain = json.dumps({"outcome": "COLLECTED", "confirmed": []})
    assert not is_utov_export(plain)
    header, payload = load_stamped_json(plain)
    assert header is None                       # discriminator: not a utov export
    assert payload["outcome"] == "COLLECTED"    # body still readable


# --- DriveResult is the other named export object --------------------------- #

def _drive_result():
    trace = [Instruction(0, 0x1000, b"\x00\x00\x00\x00", "ldr w0, [x16]",
                         {"x16": 0x9000}, {}),
             Instruction(1, 0x1004, b"\x00\x00\x00\x00", "mul w0, w0, w1",
                         {"w0": 0, "w1": 0}, {"w0": 0})]
    cc = CaseConfig(
        target="synthetic.so", input_hash="ab", run_id="r", seed_hint_addr=0x100,
        sink_hint_addr=0x200, entry_pc=0x0FFF, window=(0x1000, 0x10FF),
        reg_file=("x0", "x1", "x16"), inputs=("carrier",), parity_min=8,
        symbolic_regs=("x0", "x1"),
        concrete_backing=build_concrete_backing(reg_values={"x16": 0x9000}),
        task="t")

    def runner(_ctx):
        return {"propagated": True, "gold_parity": "8/8",
                "expr_source": "def f(carrier):\n    return carrier & 0xff\n",
                "parity_vectors": [{"input_key": f"v{i}", "observed": f"o{i}",
                                    "predicted": f"o{i}", "exec_id": f"e{i}"} for i in range(3)],
                "trace_self_check": {"seed_values": {"carrier": 0x10},
                                     "sink_value": 0x10, "sink_mask": 0xFF}}
    # NO ledger handle → no sqlite for the consumer-facing run.
    return drive(trace=trace, case_config=cc, triton_runner=runner,
                 decisions={"alias_vs_compute": "compute", "which_static": []})


def test_drive_result_exports_as_stamped_json():
    res = _drive_result()
    text = export_stamped_json(
        res.to_dict(), source="setup_symex.drive (in-memory DriveResult)",
        exported_by="drive", exec_identity={"target": "synthetic.so"}, ts=_TS)
    assert is_utov_export(text)
    header, payload = load_stamped_json(text)
    assert payload["closed"] is True and payload["self_check"]["status"] == "PASS"
