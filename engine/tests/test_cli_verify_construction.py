"""0527 BUG_REPORT-7 §J.6: `utov verify-construction` subcommand."""

from __future__ import annotations

import json
import hashlib

from click.testing import CliRunner

from engine.cli import main


def test_verify_construction_no_runner_computes_expected():
    """Without --runner-cmd, the subcommand just computes the construction
    output for each input. Useful for vetting a candidate before wiring a
    runner."""
    runner = CliRunner()
    r = runner.invoke(main, [
        "verify-construction",
        "--construction", "lambda x: hashlib.sha256(x).digest()",
        "--inputs", "616263",
        "--json",
    ])
    assert r.exit_code == 0, r.output
    payload = json.loads(r.output)
    assert payload["construction_sha1"]
    assert payload["verdict"].startswith("COMPUTED")
    expected = hashlib.sha256(b"abc").digest().hex()
    assert payload["trials"][0]["expected_hex"] == expected


def test_verify_construction_rejects_non_lambda():
    """--construction must start with 'lambda'."""
    r = CliRunner().invoke(main, [
        "verify-construction",
        "--construction", "hashlib.sha256(b'abc').digest()",
        "--inputs", "616263",
    ])
    assert r.exit_code != 0
    assert "must be a lambda" in r.output


def test_verify_construction_rejects_forbidden_tokens():
    """Block dunder / import / subprocess tokens in --construction."""
    for bad in (
        "lambda x: __import__('os').system('rm -rf /')",
        "lambda x: open('/etc/passwd').read()",
        "lambda x: exec('print(1)')",
    ):
        r = CliRunner().invoke(main, [
            "verify-construction",
            "--construction", bad,
            "--inputs", "616263",
        ])
        assert r.exit_code != 0, f"should have rejected: {bad}"
        assert "forbidden token" in r.output


def test_verify_construction_kwargs_decode_hex():
    """Hex string kwargs are decoded to bytes for the lambda."""
    r = CliRunner().invoke(main, [
        "verify-construction",
        "--construction",
            "lambda x, k: hashlib.sha256(k + x).digest()",
        "--inputs", "616263",
        "--construction-args", json.dumps({"k": "deadbeef"}),
        "--json",
    ])
    assert r.exit_code == 0, r.output
    payload = json.loads(r.output)
    expected = hashlib.sha256(bytes.fromhex("deadbeef") + b"abc").digest().hex()
    assert payload["trials"][0]["expected_hex"] == expected


def test_verify_construction_handles_bad_input_hex():
    """Non-hex --inputs produces a per-trial error, not a crash."""
    r = CliRunner().invoke(main, [
        "verify-construction",
        "--construction", "lambda x: hashlib.sha256(x).digest()",
        "--inputs", "notahexstring",
        "--json",
    ])
    assert r.exit_code == 0, r.output
    payload = json.loads(r.output)
    assert "construction_error" in payload["trials"][0]


def test_verify_construction_construction_returning_nonbytes_is_error():
    """Lambda must return bytes; ints / strs become trial errors."""
    r = CliRunner().invoke(main, [
        "verify-construction",
        "--construction", "lambda x: 42",
        "--inputs", "616263",
        "--json",
    ])
    assert r.exit_code == 0, r.output
    payload = json.loads(r.output)
    assert "expected bytes" in payload["trials"][0]["construction_error"]
