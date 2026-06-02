"""base / domain boundary lint — acceptance §19.7 #8 (+ structural half of #1).

Step-1 surface of the v0.3.0 profile layer (PLAN §19 / IMPL_PLAN §P1.0).
The mechanism-lock adversarial tests (§19.7 #7 Locks A/B/C) and the
``vmp_algorithm_extraction`` regression land in steps 2-4 once the
first mechanism probes migrate; this file covers only the
file-shape rules that step-1 must enforce:

  - base profile may contain ONLY ``mechanism: true`` entries.
  - domain profile may NOT declare ``mechanism: true``.
  - subprofile referencing a role that is not bound by any state in
    the chain → registry merge raises orphan-role error.
  - the shipped ``engine/profiles/base.json`` shell passes lint.

The actual mechanism literal scans over ``m1_success_audit.py`` etc.
also activate once those modules migrate; here the scan is exercised
against a synthetic fixture so the wiring is tested without depending
on step-2 work.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from engine.profile.lint import (
    lint_base_mechanism_source,
    lint_base_profile,
    lint_domain_profile,
    lint_kernel_source,
)
from engine.profile.loader import ProfileLoadError, load_profile_file
from engine.profile.registry import (
    BASE_PROFILE_NAME,
    PROFILES_DIR,
    ProfileMergeError,
    ProfileRegistry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(profiles_dir: Path, name: str, body: dict) -> Path:
    path = profiles_dir / f"{name}.json"
    path.write_text(json.dumps(body))
    return path


def _empty_base(profiles_dir: Path) -> Path:
    """A minimal valid base — empty mechanism set, no orphan roles."""
    return _write(profiles_dir, BASE_PROFILE_NAME, {"profile": BASE_PROFILE_NAME})


@pytest.fixture()
def profiles_dir(tmp_path: Path) -> Path:
    """An isolated profiles directory for each test."""
    d = tmp_path / "profiles"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Shipped base.json
# ---------------------------------------------------------------------------


def test_shipped_base_json_passes_lint():
    """The repo's actual engine/profiles/base.json must pass lint as-is."""
    base = load_profile_file(PROFILES_DIR / "base.json", is_base=True)
    assert lint_base_profile(base) == []


def test_shipped_base_json_loads_via_registry():
    """Registry can load the shipped base profile. Mechanism set grows
    as probes migrate; this assertion just verifies the chain shape
    and that every probe declared in base.json carries mechanism: true."""
    reg = ProfileRegistry()
    merged = reg.load_chain(BASE_PROFILE_NAME)
    assert merged.name == BASE_PROFILE_NAME
    assert merged.chain == (BASE_PROFILE_NAME,)
    # Every probe in the shipped base.json must be mechanism-locked.
    assert all(p.mechanism for p in merged.probes), (
        f"non-mechanism probes leaked into base.json: "
        f"{[p.name for p in merged.probes if not p.mechanism]}"
    )
    assert merged.mechanism_probe_names == {p.name for p in merged.probes}


# ---------------------------------------------------------------------------
# Lint: base profile may only contain mechanism entries
# ---------------------------------------------------------------------------


def test_base_profile_with_non_mechanism_probe_fails_lint(profiles_dir):
    _write(
        profiles_dir,
        BASE_PROFILE_NAME,
        {
            "profile": BASE_PROFILE_NAME,
            "probes": [{"name": "casual_probe", "mechanism": False}],
        },
    )
    base = load_profile_file(profiles_dir / "base.json", is_base=True)
    violations = lint_base_profile(base)
    assert any("casual_probe" in v and "mechanism: true" in v for v in violations)


def test_base_profile_with_mechanism_probe_passes_lint(profiles_dir):
    _write(
        profiles_dir,
        BASE_PROFILE_NAME,
        {
            "profile": BASE_PROFILE_NAME,
            "probes": [{"name": "m1_success_audit", "mechanism": True}],
        },
    )
    base = load_profile_file(profiles_dir / "base.json", is_base=True)
    assert lint_base_profile(base) == []


def test_base_profile_with_domain_field_fails_lint(profiles_dir):
    _write(
        profiles_dir,
        BASE_PROFILE_NAME,
        {
            "profile": BASE_PROFILE_NAME,
            "evidence_classes": ["A", "B", "C"],
        },
    )
    base = load_profile_file(profiles_dir / "base.json", is_base=True)
    violations = lint_base_profile(base)
    assert any("evidence_classes" in v for v in violations)


def test_base_profile_with_node_states_fails_lint(profiles_dir):
    """base.json must not enumerate concrete states — base references roles."""
    _write(
        profiles_dir,
        BASE_PROFILE_NAME,
        {
            "profile": BASE_PROFILE_NAME,
            "node_states": [{"name": "closed_form", "roles": ["closure_state"]}],
        },
    )
    base = load_profile_file(profiles_dir / "base.json", is_base=True)
    violations = lint_base_profile(base)
    assert any("node_states" in v for v in violations)


# ---------------------------------------------------------------------------
# Lint: domain profile may NOT declare mechanism: true
# ---------------------------------------------------------------------------


def test_domain_profile_with_mechanism_probe_fails_lint(profiles_dir):
    _write(
        profiles_dir,
        "evil_domain",
        {
            "profile": "evil_domain",
            "probes": [{"name": "fake_baseline", "mechanism": True}],
        },
    )
    domain = load_profile_file(profiles_dir / "evil_domain.json")
    violations = lint_domain_profile(domain)
    assert any("fake_baseline" in v and "mechanism" in v for v in violations)


def test_domain_profile_with_mechanism_gate_fails_lint(profiles_dir):
    _write(
        profiles_dir,
        "evil_domain",
        {
            "profile": "evil_domain",
            "gates": [{"id": "sneak_gate", "mechanism": True}],
        },
    )
    domain = load_profile_file(profiles_dir / "evil_domain.json")
    violations = lint_domain_profile(domain)
    assert any("sneak_gate" in v and "mechanism" in v for v in violations)


def test_legitimate_domain_profile_passes_lint(profiles_dir):
    _write(
        profiles_dir,
        "vmp_algorithm_extraction",
        {
            "profile": "vmp_algorithm_extraction",
            "evidence_classes": [
                {"id": "A", "desc": "closed-form + cross-env"},
                {"id": "B", "desc": "env-fixed observed"},
                {"id": "C", "desc": "speculative"},
            ],
            "node_states": [
                {"name": "closed_form", "roles": ["closure_state"]},
                {"name": "env_fixed_observed", "roles": []},
            ],
            "probes": [{"name": "length_chain_check"}],
            "gates": [{"id": "closure_check", "requires_verdicts": ["length_chain_check"]}],
        },
    )
    domain = load_profile_file(profiles_dir / "vmp_algorithm_extraction.json")
    assert lint_domain_profile(domain) == []


# ---------------------------------------------------------------------------
# Registry: subprofile cannot redeclare or claim mechanism (Lock A preview)
# ---------------------------------------------------------------------------


def test_subprofile_redeclaring_mechanism_probe_fails_merge(profiles_dir):
    _write(
        profiles_dir,
        BASE_PROFILE_NAME,
        {
            "profile": BASE_PROFILE_NAME,
            "probes": [{"name": "m1_success_audit", "mechanism": True}],
        },
    )
    _write(
        profiles_dir,
        "attacker",
        {
            "profile": "attacker",
            "probes": [{"name": "m1_success_audit", "mechanism": False}],
        },
    )
    reg = ProfileRegistry(profiles_dir)
    with pytest.raises(ProfileMergeError, match="mechanism-locked"):
        reg.load_chain("attacker")


def test_subprofile_claiming_mechanism_fails_merge(profiles_dir):
    _empty_base(profiles_dir)
    _write(
        profiles_dir,
        "attacker",
        {
            "profile": "attacker",
            "probes": [{"name": "fake_baseline", "mechanism": True}],
        },
    )
    reg = ProfileRegistry(profiles_dir)
    with pytest.raises(ProfileMergeError, match="only base may declare mechanism"):
        reg.load_chain("attacker")


# ---------------------------------------------------------------------------
# Registry: orphan role detection (Lock A · arm 5)
# ---------------------------------------------------------------------------


def test_orphan_role_referenced_by_base_fails_merge(profiles_dir):
    _write(
        profiles_dir,
        BASE_PROFILE_NAME,
        {
            "profile": BASE_PROFILE_NAME,
            "gates": [
                {
                    "id": "m1_success_closure",
                    "mechanism": True,
                    "rule": "claim_success requires verdict(closure_state) = pass",
                }
            ],
        },
    )
    _write(
        profiles_dir,
        "forgetful_domain",
        {
            "profile": "forgetful_domain",
            "node_states": [
                # Note: no role binding for closure_state — that's the bug.
                {"name": "closed_form", "roles": []},
            ],
        },
    )
    reg = ProfileRegistry(profiles_dir)
    with pytest.raises(ProfileMergeError, match="closure_state"):
        reg.load_chain("forgetful_domain")


def test_role_bound_by_domain_passes_merge(profiles_dir):
    _write(
        profiles_dir,
        BASE_PROFILE_NAME,
        {
            "profile": BASE_PROFILE_NAME,
            "gates": [
                {
                    "id": "m1_success_closure",
                    "mechanism": True,
                    "rule": "claim_success requires verdict(closure_state) = pass",
                }
            ],
        },
    )
    _write(
        profiles_dir,
        "good_domain",
        {
            "profile": "good_domain",
            "node_states": [{"name": "closed_form", "roles": ["closure_state"]}],
        },
    )
    reg = ProfileRegistry(profiles_dir)
    merged = reg.load_chain("good_domain")
    assert merged.chain == (BASE_PROFILE_NAME, "good_domain")
    assert "m1_success_closure" in merged.mechanism_gate_ids


# ---------------------------------------------------------------------------
# Registry: subprofile forgetting to inherit base still gets it forced in
# ---------------------------------------------------------------------------


def test_subprofile_without_inherits_still_gets_base(profiles_dir):
    _empty_base(profiles_dir)
    _write(profiles_dir, "leaf", {"profile": "leaf"})
    reg = ProfileRegistry(profiles_dir)
    merged = reg.load_chain("leaf")
    assert merged.chain == (BASE_PROFILE_NAME, "leaf")


def test_explicit_inherits_chain_resolves_in_order(profiles_dir):
    _empty_base(profiles_dir)
    _write(profiles_dir, "mid", {"profile": "mid", "inherits": BASE_PROFILE_NAME})
    _write(profiles_dir, "leaf", {"profile": "leaf", "inherits": "mid"})
    reg = ProfileRegistry(profiles_dir)
    merged = reg.load_chain("leaf")
    assert merged.chain == (BASE_PROFILE_NAME, "mid", "leaf")


def test_inheritance_cycle_is_rejected(profiles_dir):
    _write(profiles_dir, "a", {"profile": "a", "inherits": "b"})
    _write(profiles_dir, "b", {"profile": "b", "inherits": "a"})
    reg = ProfileRegistry(profiles_dir)
    with pytest.raises(ProfileMergeError, match="cycle"):
        reg.load_chain("a")


# ---------------------------------------------------------------------------
# Loader: bad files are rejected with clear errors
# ---------------------------------------------------------------------------


def test_loader_rejects_malformed_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json")
    with pytest.raises(ProfileLoadError, match="not valid JSON"):
        load_profile_file(bad)


def test_loader_rejects_missing_profile_field(tmp_path):
    bad = tmp_path / "nameless.json"
    bad.write_text("{}")
    with pytest.raises(ProfileLoadError, match="'profile'"):
        load_profile_file(bad)


def test_loader_rejects_filename_mismatch(profiles_dir):
    _write(profiles_dir, "actual_name", {"profile": "declared_name"})
    reg = ProfileRegistry(profiles_dir)
    with pytest.raises(ProfileLoadError, match="filename implies"):
        reg.load_chain("actual_name")


# ---------------------------------------------------------------------------
# Source-level lint: base mechanism modules cannot embed domain literals
# ---------------------------------------------------------------------------


def test_base_mechanism_source_with_state_literal_fails(tmp_path):
    """A mechanism module that compares against a literal state name
    bypasses the role indirection — must be flagged."""
    bad = tmp_path / "fake_mechanism.py"
    bad.write_text(
        textwrap.dedent(
            '''
            def check(ctx):
                state = ctx.state
                if state == "closed_form":   # ← role-bypass; must be flagged
                    return True
                return False
            '''
        ).strip()
    )
    violations = lint_base_mechanism_source([bad], ["closed_form", "env_fixed_observed"])
    assert any("closed_form" in v and "state_for_role" in v for v in violations)


def test_base_mechanism_source_with_role_access_passes(tmp_path):
    good = tmp_path / "clean_mechanism.py"
    good.write_text(
        textwrap.dedent(
            """
            def check(ctx):
                closure = ctx.state_for_role("closure_state")
                return closure is not None and closure.passed
            """
        ).strip()
    )
    violations = lint_base_mechanism_source([good], ["closed_form", "env_fixed_observed"])
    assert violations == []


def test_kernel_source_with_evidence_class_literal_fails(tmp_path):
    bad = tmp_path / "fake_core.py"
    bad.write_text(
        textwrap.dedent(
            '''
            def cap(node):
                if node.evidence_class == "B":   # ← kernel referencing domain literal
                    return "downgrade"
                return None
            '''
        ).strip()
    )
    violations = lint_kernel_source([bad], ["A", "B", "C"])
    assert any('"B"' in v.split(":")[-1] or "'B'" in v for v in violations) or any(
        "'B'" in v for v in violations
    )


def test_empty_forbidden_list_is_a_noop(tmp_path):
    """Step-1: no domain loaded → no literals to forbid → no violations."""
    src = tmp_path / "anything.py"
    src.write_text('x = "closed_form"\n')
    assert lint_base_mechanism_source([src], []) == []


def test_comments_do_not_trigger_lint(tmp_path):
    src = tmp_path / "with_comment.py"
    src.write_text('# closed_form is a domain state — discussed in §19.1\n')
    assert lint_base_mechanism_source([src], ["closed_form"]) == []
