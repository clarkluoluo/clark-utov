"""utov-export stamp — the provenance header on every file utov emits.

Pins the discriminator the read-side consolidation needs: a utov export carries
the header (authoritative, traceable), an agent's hand-written file does not.
"""

from __future__ import annotations

from engine.export_stamp import (
    DEFAULT_AUTHORITY,
    build_export_header,
    is_utov_export,
    parse_export_header,
)


def _header():
    return build_export_header(
        source="utov/cvd_ledger.sqlite",
        exported_by="project_stage_map",
        exec_identity={"target": "libEncryptor.so", "input_hash": "ab12",
                       "run_id": "run-A"},
        from_entries=["key1", "key2"],
        ts="2026-05-31T10:00:00Z",
    )


def test_header_is_recognised_as_utov_export():
    h = _header()
    assert is_utov_export(h)
    assert is_utov_export("  \n" + h)          # leading whitespace tolerated
    assert h.startswith("<!-- utov-export")
    assert DEFAULT_AUTHORITY in h


def test_hand_written_markdown_is_not_a_utov_export():
    assert is_utov_export("# my notes\n\nthe sink is at 0x223718\n") is False
    assert is_utov_export("") is False
    assert parse_export_header("# hand notes") is None


def test_parse_round_trips_fields():
    parsed = parse_export_header(_header())
    assert parsed is not None
    assert parsed["source"] == "utov/cvd_ledger.sqlite"
    assert parsed["exported_by"] == "project_stage_map"
    assert parsed["exec_identity"]["run_id"] == "run-A"      # JSON-decoded
    assert parsed["from_entries"] == ["key1", "key2"]        # JSON-decoded
    assert parsed["ts"] == "2026-05-31T10:00:00Z"


def test_header_prepended_to_body_still_parses():
    doc = _header() + "\n# CVD stage map — libEncryptor.so\n\n- producer @ 0x223718\n"
    assert is_utov_export(doc)
    assert parse_export_header(doc)["exported_by"] == "project_stage_map"
