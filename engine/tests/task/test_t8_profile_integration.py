"""T8 — Profile layer integration (PLAN §20.1.2 task ↔ profile).

Profile gains a ``task_templates`` section: the intent-class
declaration of recommended reusable task specs.  Base profile rejects
the field (it's domain semantics); domain profiles may declare any
number.  ``MergedProfile.task_template_for(name)`` returns the raw
spec dict for downstream parsing via
:func:`engine.task.parse_task_spec`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.profile import (
    BASE_PROFILE_NAME,
    ProfileRegistry,
)
from engine.profile.lint import lint_base_profile
from engine.profile.loader import load_profile_file
from engine.task import parse_task_spec


def _write(d: Path, name: str, body: dict) -> Path:
    p = d / f"{name}.json"
    p.write_text(json.dumps(body))
    return p


@pytest.fixture()
def profiles_dir(tmp_path: Path) -> Path:
    d = tmp_path / "profiles"
    d.mkdir()
    return d


def _base_with_floor(d: Path) -> None:
    _write(d, BASE_PROFILE_NAME, {
        "profile": BASE_PROFILE_NAME,
        "probes": [
            {"name": "m1_success_audit",    "mechanism": True},
            {"name": "m3_bypass_block",     "mechanism": True},
            {"name": "constant_provenance", "mechanism": True},
            {"name": "value_provenance",    "mechanism": True},
            {"name": "watch_first_write",   "mechanism": True},
            {"name": "scope_boundary_gate", "mechanism": True},
            {"name": "scope_upscale_gate",  "mechanism": True},
        ],
    })


# ---------------------------------------------------------------------------
# Base lint rejects task_templates
# ---------------------------------------------------------------------------


def test_base_with_task_templates_lint_rejects(tmp_path: Path):
    p = tmp_path / "base.json"
    p.write_text(json.dumps({
        "profile": "base",
        "probes": [{"name": "m1_success_audit", "mechanism": True}],
        "task_templates": [
            {"name": "tmpl", "spec": {"id": "t", "goal": "g",
                                       "done_criterion": {"kind": "node_closed",
                                                          "node": "n"},
                                       "nodes": [{"id": "n"}]}},
        ],
    }))
    profile = load_profile_file(p, is_base=True)
    violations = lint_base_profile(profile)
    assert any("task_templates" in v for v in violations)


# ---------------------------------------------------------------------------
# Domain profile carries templates; merge exposes them
# ---------------------------------------------------------------------------


def _make_domain_with_templates(profiles_dir: Path) -> None:
    _base_with_floor(profiles_dir)
    _write(profiles_dir, "vmp_algorithm_extraction", {
        "profile": "vmp_algorithm_extraction",
        "inherits": "base",
        "evidence_classes": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
        "node_states": [
            {"name": "closed_form", "roles": ["closure_state"]},
        ],
        "task_templates": [
            {
                "name": "constant_provenance_probe",
                "spec": {
                    "id": "cp_probe_call",
                    "goal": "run CP probe on a value record",
                    "done_criterion": {
                        "kind": "node_closed", "node": "classified",
                    },
                    "nodes": [{"id": "classified"}],
                    "input_contract": {
                        "accepts": ["value_record"],
                        "produces": ["category_verdict"],
                    },
                },
            },
            {
                "name": "merge_cross_check",
                "spec": {
                    "id": "merge_check",
                    "goal": "end-to-end byte-equal cross-check",
                    "done_criterion": {
                        "kind": "named_artefact", "name": "byte_equal_pass",
                    },
                    "input_contract": {
                        "accepts": ["front_half_impl", "back_half_impl"],
                        "capabilities": ["re_execute"],
                        "produces": ["byte_equal_pass"],
                    },
                },
            },
        ],
    })


def test_merged_profile_carries_task_templates(profiles_dir: Path):
    _make_domain_with_templates(profiles_dir)
    reg = ProfileRegistry(profiles_dir)
    merged = reg.load_chain("vmp_algorithm_extraction")
    names = merged.task_template_names()
    assert "constant_provenance_probe" in names
    assert "merge_cross_check" in names


def test_task_template_for_returns_raw_spec(profiles_dir: Path):
    _make_domain_with_templates(profiles_dir)
    reg = ProfileRegistry(profiles_dir)
    merged = reg.load_chain("vmp_algorithm_extraction")
    spec = merged.task_template_for("merge_cross_check")
    assert spec is not None
    assert spec["id"] == "merge_check"


def test_task_template_for_unknown_returns_none(profiles_dir: Path):
    _make_domain_with_templates(profiles_dir)
    reg = ProfileRegistry(profiles_dir)
    merged = reg.load_chain("vmp_algorithm_extraction")
    assert merged.task_template_for("nonexistent") is None


def test_template_spec_materialises_via_parse_task_spec(profiles_dir: Path):
    """End-to-end: profile-declared template flows through into a real
    TaskSpec via the task loader."""
    _make_domain_with_templates(profiles_dir)
    reg = ProfileRegistry(profiles_dir)
    merged = reg.load_chain("vmp_algorithm_extraction")
    raw = merged.task_template_for("merge_cross_check")
    spec = parse_task_spec(raw)
    assert spec.id == "merge_check"
    assert spec.is_reusable is True
    assert spec.input_contract.accepts == ("front_half_impl", "back_half_impl")
    assert spec.input_contract.capabilities == ("re_execute",)


# ---------------------------------------------------------------------------
# Inheritance: child profile may override or add templates
# ---------------------------------------------------------------------------


def test_child_profile_adds_template(profiles_dir: Path):
    _make_domain_with_templates(profiles_dir)
    _write(profiles_dir, "vmp_with_extra", {
        "profile": "vmp_with_extra",
        "inherits": "vmp_algorithm_extraction",
        "task_templates": [
            {
                "name": "scratch_layout_locator",
                "spec": {
                    "id": "locator",
                    "goal": "locate scratch21 canonical slot via scan",
                    "done_criterion": {"kind": "node_closed", "node": "loc"},
                    "nodes": [{"id": "loc"}],
                    "input_contract": {},
                },
            },
        ],
    })
    reg = ProfileRegistry(profiles_dir)
    merged = reg.load_chain("vmp_with_extra")
    names = set(merged.task_template_names())
    assert "scratch_layout_locator" in names      # added
    assert "constant_provenance_probe" in names   # inherited
    assert "merge_cross_check" in names           # inherited


def test_child_profile_overrides_template(profiles_dir: Path):
    _make_domain_with_templates(profiles_dir)
    _write(profiles_dir, "vmp_with_override", {
        "profile": "vmp_with_override",
        "inherits": "vmp_algorithm_extraction",
        "task_templates": [
            {
                "name": "merge_cross_check",
                "spec": {
                    "id": "merge_v2",
                    "goal": "merge with relaxed acceptance",
                    "done_criterion": {
                        "kind": "named_artefact", "name": "byte_equal_pass",
                    },
                    "input_contract": {},
                },
            },
        ],
    })
    reg = ProfileRegistry(profiles_dir)
    merged = reg.load_chain("vmp_with_override")
    spec = merged.task_template_for("merge_cross_check")
    assert spec["id"] == "merge_v2"
