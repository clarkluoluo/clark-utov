"""v0.4.0-dev VMP-domain profile extensions (§19.9 vmp #2–#4).

Covers:

  * V1 — ``task_level_fixed`` state binding the
    ``closure_state_within_task`` role + ``task_bound`` scope rule.
  * V2 — full scope vocabulary ordering (narrowest → widest:
    ``task_bound``, ``env_bound``, ``single_identity_bound``, ``cross_env``).
  * V3 — declarative ``cap_mapping`` section moving the
    ``constant_provenance`` 5-way category → evidence-class id table
    out of the kernel module into the domain profile.

These exercise the *shipped* on-disk profile so a domain author who
reorders ``evidence_classes`` or rewires the cap mapping sees the
synth follow without a kernel edit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.profile import BASE_PROFILE_NAME, MergedProfile
from engine.profile.probe_runtime import EvidenceClassCap, ProbeContext, Verdict
from engine.profile.probes.constant_provenance import ConstantProvenanceProbe
from engine.profile.registry import ProfileRegistry


VMP_PROFILE = "vmp_algorithm_extraction"


@pytest.fixture()
def shipped_registry() -> ProfileRegistry:
    return ProfileRegistry()


@pytest.fixture()
def vmp_profile(shipped_registry: ProfileRegistry) -> MergedProfile:
    return shipped_registry.load_chain(VMP_PROFILE)


# ---------------------------------------------------------------------------
# V1 — task_level_fixed state + closure_state_within_task role + task_bound
# ---------------------------------------------------------------------------


def test_task_level_fixed_state_declared(vmp_profile):
    names = {s.name for s in vmp_profile.node_states}
    assert "task_level_fixed" in names


def test_task_level_fixed_binds_closure_state_within_task_role(vmp_profile):
    state = next(s for s in vmp_profile.node_states if s.name == "task_level_fixed")
    assert "closure_state_within_task" in state.roles


def test_task_level_fixed_maps_to_task_bound_scope(vmp_profile):
    rules = {sr.when_state: sr.tag_scope for sr in vmp_profile.scope_semantics}
    assert rules.get("task_level_fixed") == "task_bound"


def test_closure_state_full_and_within_task_are_distinct_roles(vmp_profile):
    """``closed_form`` binds the full-strength ``closure_state``,
    ``task_level_fixed`` binds the narrower ``closure_state_within_task``.
    Same conceptual job (closure) at two strengths."""
    full = next(s for s in vmp_profile.node_states if s.name == "closed_form")
    task = next(s for s in vmp_profile.node_states if s.name == "task_level_fixed")
    assert "closure_state" in full.roles
    assert "closure_state_within_task" in task.roles
    assert "closure_state" not in task.roles
    assert "closure_state_within_task" not in full.roles


# ---------------------------------------------------------------------------
# V2 — scope vocabulary ordering
# ---------------------------------------------------------------------------


def test_scope_order_declares_full_vocabulary(vmp_profile):
    assert vmp_profile.scope_order == (
        "task_bound", "env_bound", "single_identity_bound", "cross_env",
    )


def test_scope_rank_returns_narrowest_first(vmp_profile):
    assert vmp_profile.scope_rank("task_bound") == 0
    assert vmp_profile.scope_rank("env_bound") == 1
    assert vmp_profile.scope_rank("single_identity_bound") == 2
    assert vmp_profile.scope_rank("cross_env") == 3


def test_scope_rank_unknown_scope_returns_none(vmp_profile):
    assert vmp_profile.scope_rank("interstellar") is None


def test_env_fixed_observed_still_tags_env_bound(vmp_profile):
    """V2 must not regress the existing v0.3.0 scope rule."""
    rules = {sr.when_state: sr.tag_scope for sr in vmp_profile.scope_semantics}
    assert rules.get("env_fixed_observed") == "env_bound"


# ---------------------------------------------------------------------------
# V3 — cap_mapping declarative section
# ---------------------------------------------------------------------------


def test_cap_mapping_loaded_with_default_categories(vmp_profile):
    by_cat = {e.category: e.class_id for e in vmp_profile.cap_mapping}
    assert by_cat.get("hardcoded_fixed") == "A"
    assert by_cat.get("appkey_fixed_function") == "A"
    assert by_cat.get("session_level_derived") == "B"
    assert by_cat.get("task_level_fixed") == "B"
    assert by_cat.get("undetermined") == "B"


def test_cap_for_category_lookup(vmp_profile):
    assert vmp_profile.cap_for_category("hardcoded_fixed") == "A"
    assert vmp_profile.cap_for_category("session_level_derived") == "B"
    assert vmp_profile.cap_for_category("unknown_category") is None


# ---------------------------------------------------------------------------
# V3 acceptance — kernel-free reorder follows profile (§19.7 #5 generalised)
# ---------------------------------------------------------------------------


def _archival_params(*, category: str) -> dict:
    """Build a params shape that drives the CP probe down a deterministic
    branch for a given category, without rerun/dataflow synthesis."""
    if category == "hardcoded_fixed":
        return {
            "report": {
                "values": [
                    {
                        "value_name": "flag",
                        "rerun_observations": [
                            {"dimension": "same_session",  "value_hex": "DEAD"},
                            {"dimension": "same_session",  "value_hex": "DEAD"},
                            {"dimension": "new_session",   "value_hex": "DEAD"},
                            {"dimension": "new_appkey",    "value_hex": "DEAD"},
                            {"dimension": "new_per_input", "value_hex": "DEAD"},
                        ],
                        "producer_dataflow": {"producer_reads": ["static"]},
                    }
                ]
            }
        }
    raise AssertionError(f"unhandled category fixture: {category}")


def test_cp_probe_uses_profile_cap_mapping_when_available(vmp_profile):
    """Profile-driven path: the CP probe consults
    ``profile.cap_for_category`` to resolve the ceiling, not the
    module's hardcoded ``_CATEGORY_TABLE``."""
    ctx = ProbeContext(
        method="record_observation",
        params=_archival_params(category="hardcoded_fixed"),
        profile=vmp_profile,
    )
    verdict = ConstantProvenanceProbe().run(ctx)
    assert verdict.affects_evidence_class is not None
    # vmp profile pins hardcoded_fixed → A; matches the legacy default
    # but goes through the profile path.
    assert verdict.affects_evidence_class.class_id == "A"


def test_cp_probe_follows_profile_reorder_without_kernel_change(tmp_path: Path):
    """A custom domain reorders evidence_classes (``S > A > B > C``)
    AND moves ``hardcoded_fixed`` to ``S``. The CP probe's ceiling
    reflects the new mapping with zero engine edit."""
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "base.json").write_text(json.dumps({
        "profile": "base",
        "probes": [
            {"name": "m1_success_audit",    "mechanism": True},
            {"name": "m3_bypass_block",     "mechanism": True},
            {"name": "constant_provenance", "mechanism": True},
            {"name": "value_provenance",    "mechanism": True},
            {"name": "watch_first_write",   "mechanism": True},
        ],
    }))
    (profiles_dir / "weird_domain.json").write_text(json.dumps({
        "profile": "weird_domain",
        "inherits": "base",
        "evidence_classes": [
            {"id": "S"}, {"id": "A"}, {"id": "B"}, {"id": "C"},
        ],
        "node_states": [
            {"name": "closed_form", "roles": ["closure_state"]},
        ],
        "cap_mapping": {
            "hardcoded_fixed":       "S",
            "appkey_fixed_function": "A",
            "session_level_derived": "B",
            "per_input_variable":    "",
            "undetermined":          "C",
        },
    }))
    reg = ProfileRegistry(profiles_dir)
    merged = reg.load_chain("weird_domain")
    assert merged.cap_for_category("hardcoded_fixed") == "S"

    ctx = ProbeContext(
        method="record_observation",
        params=_archival_params(category="hardcoded_fixed"),
        profile=merged,
    )
    verdict = ConstantProvenanceProbe().run(ctx)
    assert verdict.affects_evidence_class is not None
    assert verdict.affects_evidence_class.class_id == "S"


def test_cp_probe_falls_back_to_module_table_without_profile():
    """No profile = legacy v0.3.0 ceiling, unchanged."""
    ctx = ProbeContext(
        method="record_observation",
        params=_archival_params(category="hardcoded_fixed"),
        profile=None,
    )
    verdict = ConstantProvenanceProbe().run(ctx)
    assert verdict.affects_evidence_class is not None
    assert verdict.affects_evidence_class.class_id == "A"


# ---------------------------------------------------------------------------
# V3 lint — cap_mapping is a domain field, base profile may not declare it
# ---------------------------------------------------------------------------


def test_cap_mapping_in_base_profile_rejected_by_lint(tmp_path: Path):
    """Base may carry only mechanism; cap_mapping is a domain table."""
    from engine.profile.lint import lint_base_profile
    from engine.profile.loader import load_profile_file

    p = tmp_path / "base.json"
    p.write_text(json.dumps({
        "profile": "base",
        "probes": [{"name": "m1_success_audit", "mechanism": True}],
        "cap_mapping": {"hardcoded_fixed": "A"},
    }))
    profile = load_profile_file(p, is_base=True)
    violations = lint_base_profile(profile)
    assert any("cap_mapping" in v for v in violations)


def test_scope_order_in_base_profile_rejected_by_lint(tmp_path: Path):
    from engine.profile.lint import lint_base_profile
    from engine.profile.loader import load_profile_file

    p = tmp_path / "base.json"
    p.write_text(json.dumps({
        "profile": "base",
        "probes": [{"name": "m1_success_audit", "mechanism": True}],
        "scope_order": ["env_bound", "cross_env"],
    }))
    profile = load_profile_file(p, is_base=True)
    violations = lint_base_profile(profile)
    assert any("scope_order" in v for v in violations)
