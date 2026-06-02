"""Tests for the declarative ``related_helpers`` discoverability layer (spec C).

Covers the four spec §Fixtures:
  (a) verbose CLI prints related helpers for a mapped command;
  (b) an unmapped entry point surfaces nothing;
  (c) a relation pointing at a truly-bogus symbol fails the lint (a real bad
      link), while the planned-target allowlist exempts not-yet-landed symbols;
  (d) regression: default (non-verbose) command output is byte-for-byte
      unchanged.
"""
from __future__ import annotations

from click.testing import CliRunner

from engine.cli import main
from engine import related_helpers as rh


# --------------------------------------------------------------------------
# API: the lookup
# --------------------------------------------------------------------------
def test_lookup_returns_seeded_relations():
    assert rh.related_helpers("trace_provenance") == [
        "run_recapture_loop", "suggest_observations"]
    assert rh.related_helpers("check_parity_vectors") == [
        "real_gold.collect_real_gold", "check_seed_independence"]
    assert rh.related_helpers("import_map") == [
        "extern_model.resolve_extern_model",
        "libc_boundary.synthesize_boundary_edge"]
    assert rh.related_helpers("run_recovery") == [
        "check_emit_self_consistency", "derive_mem_sink_interval"]


def test_lookup_unmapped_is_empty_not_fake():
    # Fixture (b) at the API level: unmapped ⇒ empty list, never a fake claim.
    assert rh.related_helpers("no_such_entry_point") == []
    assert rh.format_related_line("no_such_entry_point") is None


def test_lookup_returns_a_fresh_copy():
    a = rh.related_helpers("trace_provenance")
    a.append("MUTATED")
    assert "MUTATED" not in rh.related_helpers("trace_provenance")


def test_format_line_shape():
    line = rh.format_related_line("run_recovery")
    assert line == "ℹ related: check_emit_self_consistency, derive_mem_sink_interval"


# --------------------------------------------------------------------------
# Lint: no silent bad links (Fixture (c))
# --------------------------------------------------------------------------
def test_seeded_map_is_clean():
    # Every real target resolves under `engine`; planned ones are allowlisted.
    errors = rh.lint_related_helpers()
    assert errors == [], "seeded relation map has bad links:\n" + "\n".join(errors)


def test_bogus_symbol_fails_lint(monkeypatch):
    # A relation pointing at a truly-nonexistent symbol must be caught.
    monkeypatch.setitem(
        rh.RELATED_HELPERS, "trace_provenance",
        ["run_recapture_loop", "this_symbol_does_not_exist_anywhere_xyz"])
    errors = rh.lint_related_helpers()
    assert any("this_symbol_does_not_exist_anywhere_xyz" in e for e in errors)


def test_planned_target_is_exempt_from_lint(monkeypatch):
    # A planned/not-yet-landed target in the allowlist keeps the map clean even
    # though its symbol exists nowhere. (Uses a synthetic placeholder so the
    # test does not depend on which real symbols are mid-flight this wave.)
    monkeypatch.setattr(rh, "PLANNED_TARGETS", frozenset({"future_helper_xyz"}))
    monkeypatch.setitem(rh.RELATED_HELPERS, "x", ["future_helper_xyz"])
    errors = rh.lint_related_helpers()
    assert not any("future_helper_xyz" in e for e in errors)


def test_planned_symbol_without_allowlist_would_fail(monkeypatch):
    # Sanity: the exemption is the allowlist, not a silent pass. The same
    # not-yet-landed relation with an empty allowlist now fails lint.
    monkeypatch.setattr(rh, "PLANNED_TARGETS", frozenset())
    monkeypatch.setitem(rh.RELATED_HELPERS, "x", ["future_helper_xyz"])
    errors = rh.lint_related_helpers()
    assert any("future_helper_xyz" in e for e in errors)


# --------------------------------------------------------------------------
# CLI: verbose surfacing (Fixtures (a), (b), (d))
# --------------------------------------------------------------------------
def test_verbose_cli_surfaces_related_for_mapped_command():
    # Fixture (a): `phases` is wired to surface the `trace_provenance` relation.
    r = CliRunner().invoke(main, ["--verbose", "phases"])
    assert r.exit_code == 0, r.output
    assert "ℹ related: run_recapture_loop, suggest_observations" in r.stderr


def test_verbose_via_env_debug(monkeypatch):
    # Debug-mode parity with the API: UTOV_DEBUG also surfaces the hint.
    monkeypatch.setenv("UTOV_DEBUG", "1")
    r = CliRunner().invoke(main, ["phases"])
    assert r.exit_code == 0, r.output
    assert "ℹ related:" in r.stderr


def test_unmapped_command_surfaces_nothing_even_in_verbose():
    # Fixture (b): `trace-info` has no relation entry ⇒ no hint, even verbose.
    # (Run against a trivial empty trace so the command itself succeeds.)
    import json
    from pathlib import Path
    runner = CliRunner()
    with runner.isolated_filesystem():
        ins = {"idx": 0, "pc": "0x1000", "bytes": "00000000",
               "mnemonic": "ret", "regs_read": {}, "regs_write": {}, "mem": []}
        Path("t.jsonl").write_text(json.dumps(ins) + "\n")
        r = runner.invoke(main, ["--verbose", "trace-info", "t.jsonl"])
        assert r.exit_code == 0, r.output
        assert "ℹ related:" not in r.stderr
        assert "ℹ related:" not in r.output


def test_default_mode_output_byte_for_byte_unchanged():
    # Fixture (d): non-verbose output must be identical with vs without the
    # discoverability layer. The hint goes to stderr and is gated off, so the
    # default-mode stdout has no `ℹ related:` line at all.
    plain = CliRunner().invoke(main, ["phases"])
    assert plain.exit_code == 0, plain.output
    assert "ℹ related:" not in plain.stdout
    assert "ℹ related:" not in plain.stderr
    # And the verbose stdout payload equals the default stdout payload exactly
    # (the hint is additive on stderr only — no command-output change).
    verbose = CliRunner().invoke(main, ["--verbose", "phases"])
    assert verbose.stdout == plain.stdout
