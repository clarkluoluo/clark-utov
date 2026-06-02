"""CVD ledger — stamped, queryable verdicts that close the lane's record gap.

Pins the behaviour the VMP cipher case (163511) forced out: utov as the sole
authoritative writer (every verdict lands), "latest wins" bucketed by execution
identity (a different run never overwrites), a read-side pull that lets the next
round skip already-closed sub-goals, and a markdown stage-map projection that
replaces the hand-written notes.

Fixture shape mirrors the case (producer @0x223718 SUCCESS, the rodata template
TERMINAL_STATIC, the carrier NEEDS_OBSERVATION) but the addresses live HERE in
the test — the ledger mechanism is target-agnostic.
"""

from __future__ import annotations

import pytest

from engine.cvd_ledger import (
    ExecIdentity,
    LedgerKind,
    closed_subjects,
    entry_key,
    get_latest,
    history,
    kind_for_verdict,
    open_ledger,
    project_stage_map,
    record_call,
    record_verdict,
    should_skip,
    write_stage_view,
)
from engine.export_stamp import is_utov_export, parse_export_header


@pytest.fixture
def conn():
    c = open_ledger(":memory:")
    yield c
    c.close()


# An execution identity, and a SECOND one differing only by run_id / nonce —
# the determinism bucket the "latest wins" rule keys on.
RUN_A = ExecIdentity(target="libEncryptor.so", input_hash="ab12", run_id="run-A")
RUN_B = ExecIdentity(target="libEncryptor.so", input_hash="ab12", run_id="run-B")
RUN_A_NONCE = ExecIdentity(target="libEncryptor.so", input_hash="ab12",
                           run_id="run-A", nonce="e6_capture")

CVD_IN = {"sink_base": 0x12316078, "window": [0x2236AC, 0x2236E0]}


# --- write + query round-trip -----------------------------------------------

def test_record_and_get_latest_round_trips_result():
    c = open_ledger(":memory:")
    record_call(
        c, call_fn="trace_provenance", inputs=CVD_IN, exec_identity=RUN_A,
        result={"verdict": "SUCCESS", "producer": "0x223718"},
        kind=LedgerKind.PRODUCER, verdict="SUCCESS",
        subject="source_base", ts="2026-05-31T10:00:00Z",
    )
    e = get_latest(c, call_fn="trace_provenance", inputs=CVD_IN, exec_identity=RUN_A)
    assert e is not None
    assert e.verdict == "SUCCESS"
    assert e.result["producer"] == "0x223718"
    assert e.subject == "source_base"
    assert e.is_closed is True
    c.close()


def test_get_latest_none_when_absent(conn):
    assert get_latest(conn, call_fn="who_wrote", inputs={"a": 1},
                      exec_identity=RUN_A) is None


# --- rule 3: latest wins within a key, ts-ordered ---------------------------

def test_latest_wins_within_same_key(conn):
    record_verdict(conn, call_fn="trace_provenance", inputs=CVD_IN,
                   exec_identity=RUN_A, verdict="NEEDS_OBSERVATION",
                   subject="carrier_base", ts="2026-05-31T10:00:00Z")
    # a later round on the SAME execution re-runs and now closes it
    record_verdict(conn, call_fn="trace_provenance", inputs=CVD_IN,
                   exec_identity=RUN_A, verdict="SUCCESS",
                   subject="carrier_base", anchor=(7321, 0x223718),
                   ts="2026-05-31T11:30:00Z")
    e = get_latest(conn, call_fn="trace_provenance", inputs=CVD_IN, exec_identity=RUN_A)
    assert e.verdict == "SUCCESS"
    assert e.result["anchor"]["pc_hex"] == "0x223718"
    # history keeps both, newest first
    h = history(conn, call_fn="trace_provenance", inputs=CVD_IN, exec_identity=RUN_A)
    assert [x.verdict for x in h] == ["SUCCESS", "NEEDS_OBSERVATION"]


# --- rule 3: cross-run never overwrites -------------------------------------

def test_cross_run_does_not_overwrite(conn):
    # same call_fn + inputs, different execution identity -> different key.
    record_verdict(conn, call_fn="trace_provenance", inputs=CVD_IN,
                   exec_identity=RUN_A, verdict="SUCCESS", subject="source_base",
                   ts="2026-05-31T10:00:00Z")
    record_verdict(conn, call_fn="trace_provenance", inputs=CVD_IN,
                   exec_identity=RUN_B, verdict="NEEDS_OBSERVATION",
                   subject="source_base", ts="2026-05-31T12:00:00Z")
    a = get_latest(conn, call_fn="trace_provenance", inputs=CVD_IN, exec_identity=RUN_A)
    b = get_latest(conn, call_fn="trace_provenance", inputs=CVD_IN, exec_identity=RUN_B)
    assert a.verdict == "SUCCESS"            # run B's later ts did NOT bleed in
    assert b.verdict == "NEEDS_OBSERVATION"
    # the nonce-distinguished e6_capture round is a third, independent bucket
    record_verdict(conn, call_fn="trace_provenance", inputs=CVD_IN,
                   exec_identity=RUN_A_NONCE, verdict="SUCCESS",
                   subject="source_base", ts="2026-05-31T13:00:00Z")
    assert entry_key("trace_provenance", CVD_IN, RUN_A) \
        != entry_key("trace_provenance", CVD_IN, RUN_A_NONCE)
    assert get_latest(conn, call_fn="trace_provenance", inputs=CVD_IN,
                      exec_identity=RUN_A).verdict == "SUCCESS"


# --- verdict -> kind mapping + closed semantics -----------------------------

def test_kind_for_verdict_mapping():
    assert kind_for_verdict("SUCCESS") is LedgerKind.PRODUCER
    assert kind_for_verdict("TERMINAL_STATIC") is LedgerKind.STATIC_TERMINAL
    assert kind_for_verdict("NEEDS_OBSERVATION") is LedgerKind.OPEN_HYPOTHESIS
    assert kind_for_verdict("WHATEVER") is LedgerKind.VERDICT


def test_open_hypothesis_is_not_closed(conn):
    record_verdict(conn, call_fn="trace_provenance", inputs=CVD_IN,
                   exec_identity=RUN_A, verdict="NEEDS_OBSERVATION",
                   subject="carrier_base", watch=[], ts="2026-05-31T10:00:00Z")
    e = get_latest(conn, call_fn="trace_provenance", inputs=CVD_IN, exec_identity=RUN_A)
    assert e.is_closed is False
    assert e.result["watch_n"] == 0          # watch_n=0 lead is recorded, not lost


# --- read-side pull: skip already-closed sub-goals --------------------------

def test_should_skip_closed_subgoal_saves_a_round(conn):
    # Round 1: producer located + landed.
    record_verdict(conn, call_fn="who_wrote", inputs={"sink": 0x12316078},
                   exec_identity=RUN_A, verdict="SUCCESS",
                   subject="producer:0x12316078", anchor=(7321, 0x223718),
                   ts="2026-05-31T10:00:00Z")
    # Round 2: the same sub-goal would be re-derived — the pull says skip it.
    assert should_skip(conn, call_fn="who_wrote", inputs={"sink": 0x12316078},
                       exec_identity=RUN_A) is True
    # a different, un-recorded sub-goal is NOT skipped
    assert should_skip(conn, call_fn="who_wrote", inputs={"sink": 0x12316099},
                       exec_identity=RUN_A) is False


def test_closed_subjects_scoped_to_execution(conn):
    record_verdict(conn, call_fn="trace_provenance", inputs=CVD_IN,
                   exec_identity=RUN_A, verdict="SUCCESS", subject="source_base",
                   ts="2026-05-31T10:00:00Z")
    record_verdict(conn, call_fn="trace_provenance", inputs={"sink_base": 0x1},
                   exec_identity=RUN_A, verdict="NEEDS_OBSERVATION",
                   subject="carrier_base", ts="2026-05-31T10:05:00Z")
    # run B closes a different subject — must not appear in run A's closed set.
    record_verdict(conn, call_fn="trace_provenance", inputs={"sink_base": 0x2},
                   exec_identity=RUN_B, verdict="SUCCESS", subject="other_run",
                   ts="2026-05-31T10:10:00Z")
    assert closed_subjects(conn, RUN_A) == {"source_base"}
    assert closed_subjects(conn, RUN_B) == {"other_run"}


# --- projection: the global stage map ---------------------------------------

def _backfill_case(conn):
    """Backfill the cipher-body case's verdicts (synthetic, case-shaped)."""
    record_verdict(conn, call_fn="who_wrote", inputs={"sink": 0x12316078},
                   exec_identity=RUN_A, verdict="SUCCESS",
                   subject="producer:cipher_body", anchor=(7321, 0x223718),
                   ts="2026-05-31T10:00:00Z")
    record_verdict(conn, call_fn="trace_provenance", inputs={"rodata": 0x12216A80},
                   exec_identity=RUN_A, verdict="TERMINAL_STATIC",
                   subject="rodata_template", anchor=(0, 0x2236AC),
                   ts="2026-05-31T10:01:00Z")
    record_verdict(conn, call_fn="trace_provenance", inputs={"carrier": 0x12316060},
                   exec_identity=RUN_A, verdict="NEEDS_OBSERVATION",
                   subject="carrier_base", watch=[0x2236B4, 0x2236C0],
                   ts="2026-05-31T10:02:00Z")


def test_project_stage_map_renders_landed_verdicts(conn):
    _backfill_case(conn)
    md = project_stage_map(conn, RUN_A)
    assert "# CVD stage map — libEncryptor.so" in md
    assert "0x223718" in md                       # producer anchor
    assert "## Producers (SUCCESS)" in md
    assert "## Static terminals" in md
    assert "rodata_template" in md
    assert "## Open hypotheses (NEEDS_OBSERVATION)" in md
    assert "watch_n=2" in md                       # the carrier lead's watch list


def test_project_stage_map_scoped_per_execution(conn):
    _backfill_case(conn)
    # a different run's entry must not bleed into run A's map
    record_verdict(conn, call_fn="who_wrote", inputs={"sink": 0xDEAD},
                   exec_identity=RUN_B, verdict="SUCCESS",
                   subject="producer:other_run", ts="2026-05-31T11:00:00Z")
    md_a = project_stage_map(conn, RUN_A)
    md_b = project_stage_map(conn, RUN_B)
    assert "producer:other_run" not in md_a
    assert "producer:cipher_body" not in md_b
    assert "run run-A" in md_a and "run run-B" in md_b


# --- read-side consolidation: stamped view replaces hand-written json --------

def test_projection_carries_utov_export_stamp(conn):
    _backfill_case(conn)
    md = project_stage_map(conn, RUN_A, ts="2026-05-31T12:00:00Z")
    # the header is the discriminator: this is an authoritative utov export.
    assert is_utov_export(md)
    hdr = parse_export_header(md)
    assert hdr["exported_by"] == "project_stage_map"
    assert hdr["source"] == "utov/cvd_ledger.sqlite"
    assert hdr["exec_identity"]["run_id"] == "run-A"
    # from_entries is traceable: the keys it was generated from are the run's keys
    assert hdr["from_entries"]
    assert all(len(k) == 40 for k in hdr["from_entries"])      # sha1 entry keys
    # body still renders below the header
    assert "# CVD stage map — libEncryptor.so" in md


def test_projection_without_header_is_plain_body(conn):
    _backfill_case(conn)
    md = project_stage_map(conn, RUN_A, with_header=False)
    assert is_utov_export(md) is False
    assert md.lstrip().startswith("# CVD stage map")


def test_record_auto_materialises_stamped_view(tmp_path):
    # The dead-ledger fix: a file-backed write produces the readable stamped
    # view next to the sqlite WITHOUT any explicit projection call — so an agent
    # script that only does open_ledger + record_* still gets the view.
    db = tmp_path / "cvd_ledger.sqlite"
    c = open_ledger(str(db))
    record_verdict(c, call_fn="who_wrote", inputs={"sink": 0x12316078},
                   exec_identity=RUN_A, verdict="SUCCESS",
                   subject="producer:cipher_body", anchor=(7321, 0x223718),
                   ts="2026-05-31T10:00:00Z")
    c.close()
    view = tmp_path / "cvd_ledger_view.md"
    assert view.exists()
    text = view.read_text(encoding="utf-8")
    assert is_utov_export(text)                       # stamped, authoritative
    assert "producer:cipher_body" in text and "0x223718" in text


def test_auto_view_can_be_disabled(tmp_path):
    db = tmp_path / "cvd_ledger.sqlite"
    c = open_ledger(str(db))
    record_verdict(c, call_fn="who_wrote", inputs={"sink": 0x1},
                   exec_identity=RUN_A, verdict="SUCCESS", subject="p",
                   ts="2026-05-31T10:00:00Z", auto_view=False)
    c.close()
    assert not (tmp_path / "cvd_ledger_view.md").exists()


def test_inmemory_ledger_skips_auto_view():
    # :memory: has no directory — auto_view must no-op, not crash.
    c = open_ledger(":memory:")
    record_verdict(c, call_fn="who_wrote", inputs={"sink": 0x1},
                   exec_identity=RUN_A, verdict="SUCCESS", subject="p",
                   ts="2026-05-31T10:00:00Z")
    assert get_latest(c, call_fn="who_wrote", inputs={"sink": 0x1},
                      exec_identity=RUN_A).verdict == "SUCCESS"
    c.close()


# --- recording policy: durable findings only, big lists digested -------------

def test_transient_kind_dropped_by_default(conn):
    # an intermediate spec (default VERDICT kind) is transient -> not recorded.
    record_call(conn, call_fn="seed_entry_state", inputs={"x": 1},
                exec_identity=RUN_A, result={"spec": "..."},
                ts="2026-05-31T10:00:00Z")
    assert get_latest(conn, call_fn="seed_entry_state", inputs={"x": 1},
                      exec_identity=RUN_A) is None
    # force lands it anyway (rare escape hatch)
    record_call(conn, call_fn="seed_entry_state", inputs={"x": 1},
                exec_identity=RUN_A, result={"spec": "kept"},
                ts="2026-05-31T10:00:01Z", force=True)
    assert get_latest(conn, call_fn="seed_entry_state", inputs={"x": 1},
                      exec_identity=RUN_A).result["spec"] == "kept"


def test_durable_kinds_land(conn):
    cases = [(LedgerKind.SINK, "SINK_CONFIRMED"),
             (LedgerKind.EMIT, "PASS"),
             (LedgerKind.TRANSFORM, "OK")]
    for k, vd in cases:
        record_verdict(conn, call_fn="x", inputs={"k": k.value},
                       exec_identity=RUN_A, verdict=vd, kind=k,
                       subject=k.value, ts="2026-05-31T10:00:00Z")
        got = get_latest(conn, call_fn="x", inputs={"k": k.value}, exec_identity=RUN_A)
        assert got is not None and got.kind == k.value
    # SINK_CONFIRMED maps to the SINK kind via the verdict map (no override)
    assert kind_for_verdict("SINK_CONFIRMED") is LedgerKind.SINK


def test_big_lists_digested_not_inlined(conn):
    big = list(range(15497))                 # the blind_pcs explosion case
    record_verdict(conn, call_fn="trace_provenance", inputs={"sink": 1},
                   exec_identity=RUN_A, verdict="NEEDS_OBSERVATION",
                   subject="carrier", payload={"blind_pcs": big},
                   ts="2026-05-31T10:00:00Z")
    e = get_latest(conn, call_fn="trace_provenance", inputs={"sink": 1},
                   exec_identity=RUN_A)
    bp = e.result["blind_pcs"]
    assert isinstance(bp, dict) and bp["_trimmed_list"] is True
    assert bp["count"] == 15497 and len(bp["sha1"]) == 40 and len(bp["sample"]) == 8
    # the full 15497-element list is NOT inlined anywhere in the stored entry
    import json as _j
    assert "15496" not in _j.dumps(e.result)


def test_policy_keeps_db_to_durable_count(conn):
    # one "round": 4 transient steps (dropped) + 2 durable findings (kept).
    for fn in ("locate_boundary", "seed_entry_state", "pick_mode", "classify_hybrid_step"):
        record_call(conn, call_fn=fn, inputs={"step": fn}, exec_identity=RUN_A,
                    result={"spec": fn}, ts="2026-05-31T10:00:00Z")
    record_verdict(conn, call_fn="who_wrote", inputs={"sink": 1},
                   exec_identity=RUN_A, verdict="SUCCESS", subject="producer",
                   ts="2026-05-31T10:01:00Z")
    record_verdict(conn, call_fn="emit_python", inputs={"f": 1},
                   exec_identity=RUN_A, verdict="PASS", kind=LedgerKind.EMIT,
                   subject="cipher_body", payload={"parity": "8/8"},
                   ts="2026-05-31T10:02:00Z")
    n = conn.execute("SELECT COUNT(*) FROM cvd_ledger").fetchone()[0]
    assert n == 2                            # only the durable findings, not 6


def test_should_skip_and_map_unchanged_under_policy(conn):
    # durable closure verdict still drives should_skip + shows in the map —
    # dropping transient specs does not lose reuse or the stage map.
    record_verdict(conn, call_fn="who_wrote", inputs={"sink": 0x12316078},
                   exec_identity=RUN_A, verdict="SUCCESS",
                   subject="producer:cipher", anchor=(7321, 0x223718),
                   ts="2026-05-31T10:00:00Z")
    assert should_skip(conn, call_fn="who_wrote", inputs={"sink": 0x12316078},
                       exec_identity=RUN_A) is True
    assert "0x223718" in project_stage_map(conn, RUN_A)


def test_run_summary_rendered_in_projection(conn):
    record_call(conn, call_fn="drive", inputs={"task": "closure_fill"},
                exec_identity=RUN_A, kind=LedgerKind.RUN_SUMMARY,
                subject="closure_fill_x16",
                result={"gold_parity": "2/8", "emit_parity_ok": False,
                        "closure_gate_pass": True,
                        "check_mem_backing": {"sufficient": True,
                                              "false_fail_gate_bug": False}},
                ts="2026-05-31T10:00:00Z")
    md = project_stage_map(conn, RUN_A)
    assert "## Run summary — closure_fill_x16" in md
    assert "gold_parity" in md and "2/8" in md
    assert "false_fail_gate_bug" in md        # nested dict rendered


def test_write_stage_view_writes_fixed_path_stamped(tmp_path, conn):
    _backfill_case(conn)
    path = write_stage_view(conn, RUN_A, tmp_path, ts="2026-05-31T12:00:00Z")
    # fixed filename — the agent's per-round view, read instead of hand-written json
    assert path.name == "cvd_ledger_view.md"
    text = path.read_text(encoding="utf-8")
    assert is_utov_export(text)
    assert parse_export_header(text)["exec_identity"]["run_id"] == "run-A"
    assert "producer:cipher_body" in text
