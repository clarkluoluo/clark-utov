"""§需求2 — capability-token coverage: 防串台 (cross-build confusion) guard.

A self-consistent / commit_ok build can still PRE-DATE a feature; the capability
token (present in source ⇔ build has the feature) + a universal ``required ⊆
build`` subset test is what tells an agent a stamp's build actually carries what a
terminal relies on. Pins: a build LACKING a required token →coverage_ok false +
WARN; a build with ALL tokens → coverage_ok true, no WARN (today's behaviour,
invariant 7). Token names are capability markers, never case names (invariant 6).
"""

from __future__ import annotations

from engine.capabilities import (
    PROVIDED_CAPABILITIES,
    TERMINAL_REQUIRES,
    build_capability_stamp,
    check_coverage,
    collect_build_capabilities,
    coverage_for_terminal,
)


def test_collect_returns_the_source_provided_set():
    # collect = the in-source token set (no git, no commit math — the source IS
    # the evidence).
    assert collect_build_capabilities() is PROVIDED_CAPABILITIES
    # the milestone tokens the spec asks to register are all present in THIS build.
    for tok in ("opaque_forward_v1", "symbolic_forward_record_v1",
                "opaque_forward_fallback_v1", "recovery_block_kind_v1"):
        assert tok in PROVIDED_CAPABILITIES


def test_coverage_full_build_ok_no_warn():
    # build_caps = ALL tokens → coverage_ok true, no missing, no WARN.
    ok, missing, warn = check_coverage(
        TERMINAL_REQUIRES["opaque_staging"],
        build_caps=PROVIDED_CAPABILITIES)
    assert ok is True and missing == frozenset() and warn is None


def test_coverage_missing_build_warns():
    # a pre-feature build (lacks opaque_forward_v1) → coverage_ok false + a WARN
    # naming the gap, telling the agent to rebuild from a commit that has it.
    pre_fix = PROVIDED_CAPABILITIES - {"opaque_forward_v1"}
    ok, missing, warn = check_coverage(
        TERMINAL_REQUIRES["opaque_staging"], build_caps=pre_fix)
    assert ok is False
    assert missing == frozenset({"opaque_forward_v1"})
    assert warn is not None and "opaque_forward_v1" in warn
    assert "pre-feature" in warn


def test_coverage_for_terminal_unknown_kind_requires_nothing():
    # a terminal kind with no declared requirement is always covered (additive:
    # adding the mapping later only ever ADDS a possible WARN).
    ok, missing, warn = coverage_for_terminal("some_unmapped_terminal")
    assert ok is True and missing == frozenset() and warn is None


def test_coverage_for_terminal_default_uses_build():
    # coverage_for_terminal with no build_caps reads THIS build → covered.
    ok, _missing, warn = coverage_for_terminal("opaque_staging")
    assert ok is True and warn is None


def test_block_kind_terminals_require_recovery_token():
    # every block_kind value declares the recovery_block_kind_v1 requirement (a
    # build emitting a block_kind must carry the feature — stale-stamp guard).
    for bk in ("opaque_staging_block", "window_boundary_mismatch",
               "symbol_not_on_output_path", "emit_picked_constant",
               "undetermined_constant"):
        assert TERMINAL_REQUIRES[bk] == frozenset({"recovery_block_kind_v1"})


def test_capability_stamp_block_is_sorted_list():
    stamp = build_capability_stamp()
    assert stamp["capabilities"] == sorted(PROVIDED_CAPABILITIES)


def test_doctor_surfaces_capabilities_and_warns_on_gap(monkeypatch):
    # doctor's [capabilities] group lists the build tokens; a build missing a
    # required token raises a per-terminal WARN finding.
    from engine import doctor

    findings_full = doctor._check_capabilities()
    assert any(f.component == "capabilities" and f.level == doctor.Level.OK
               for f in findings_full)
    # no WARN when the build has every token.
    assert not any(f.level == doctor.Level.WARN
                   and f.component.startswith("capability:")
                   for f in findings_full)

    # simulate a pre-feature build → coverage WARN surfaces.
    pre_fix = PROVIDED_CAPABILITIES - {"opaque_forward_v1"}
    monkeypatch.setattr(
        "engine.capabilities.collect_build_capabilities", lambda: pre_fix)
    findings_pre = doctor._check_capabilities()
    warns = [f for f in findings_pre
             if f.level == doctor.Level.WARN and f.component.startswith("capability:")]
    assert any("opaque_forward_v1" in f.detail for f in warns)
