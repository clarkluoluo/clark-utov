"""CLI mirrors of on-disk-state RPC methods: `findings` + the static `phases`
route command.

`findings` is the self-contained SELECT mirror of the get_findings RPC method;
these pin it end-to-end against a hand-built findings.sqlite. (`hyps`/`override`
are thin wrappers over the already-tested HypTree.query / mark_verdict +
log_intervention paths.)
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

from click.testing import CliRunner

from engine.cli import main


def _findings_db(tmp: Path) -> None:
    db = tmp / "findings.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE findings(id INTEGER PRIMARY KEY, stage TEXT, kind TEXT,"
        " subject TEXT, source TEXT, verifier_strategy TEXT, verified_at TEXT,"
        " origin_hyp_id INTEGER, payload_ref TEXT)"
    )
    conn.executemany(
        "INSERT INTO findings(id, stage, kind, subject, source, verifier_strategy,"
        " verified_at, origin_hyp_id, payload_ref) VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (1, "s5", "algorithm_identified", "AES-128", "s5_deterministic", "io", "t", None, None),
            (2, "s5", "fold_idiom", "sigma0", "s5_fold_idiom", "io", "t", None, None),
        ],
    )
    conn.commit()
    conn.close()


def test_findings_lists_rows_json():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _findings_db(tmp)
        r = CliRunner().invoke(main, ["findings", str(tmp), "--json"])
        assert r.exit_code == 0, r.output
        out = json.loads(r.output)
        assert {d["kind"] for d in out} == {"algorithm_identified", "fold_idiom"}


def test_findings_kind_filter():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _findings_db(tmp)
        r = CliRunner().invoke(main, ["findings", str(tmp), "--kind", "fold_idiom", "--json"])
        assert r.exit_code == 0, r.output
        out = json.loads(r.output)
        assert len(out) == 1 and out[0]["subject"] == "sigma0"


def test_findings_missing_db_errors():
    with tempfile.TemporaryDirectory() as td:
        r = CliRunner().invoke(main, ["findings", td, "--json"])
        assert r.exit_code == 2


def test_phases_prints_route_with_heavy_gated():
    r = CliRunner().invoke(main, ["phases"])
    assert r.exit_code == 0, r.output
    assert "phase_1_io_observe" in r.output
    assert "phase_heavy_vmtrace" in r.output
    assert "ESCALATION" in r.output
    assert "No candidate-guessing" in r.output
