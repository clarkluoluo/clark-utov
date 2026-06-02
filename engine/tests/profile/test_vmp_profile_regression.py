"""vmp_algorithm_extraction profile regression (PLAN §19.7 #2).

The first domain profile. Two acceptance goals:

  1. **Structural** — loading ``vmp_algorithm_extraction`` produces a
     merged profile with the expected evidence-class ordering A/B/C,
     the expected node-state vocabulary with the ``closure_state``
     role correctly bound to ``closed_form``, the
     ``length_chain_check`` probe registered as **non-mechanism**, and
     the ``env_fixed_observed → env_bound`` scope rule. Mechanism
     probes from base remain present and mechanism-flagged.

  2. **Open-layer demonstration** — a subprofile that inherits
     ``vmp_algorithm_extraction`` can freely:

       * override ``length_chain_check`` with a custom non-mechanism
         probe of the same name (the merge takes the child's),
       * ``disable: ["length_chain_check"]`` to drop the probe entirely
         from the merged set.

     In both cases, base mechanism probes stay locked in place — the
     open layer is the domain, not the baseline.

The probe-behaviour tests for length_chain_check itself (pass / fail
/ undetermined paths) live in this file too — it's the canonical
domain probe and its acceptance is naturally co-located with the
profile that owns it.

The wider §19.7 #2 acceptance — "every v0.2.0-dev acceptance test
stays green" — is enforced by the engine-wide pytest run; the
adapter pattern preserves underlying module behaviour, so
``test_value_provenance.py``, ``test_watch_first_write.py``, and
``test_length_chain.py`` continue to exercise the same paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.profile import BASE_PROFILE_NAME, ProbeContext, ProfileMergeError
from engine.profile.probe_runtime import get_builtin_probe_class
from engine.profile.probes.length_chain_check import LengthChainCheckProbe
from engine.profile.registry import ProfileRegistry


VMP_PROFILE = "vmp_algorithm_extraction"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(profiles_dir: Path, name: str, body: dict) -> Path:
    path = profiles_dir / f"{name}.json"
    path.write_text(json.dumps(body))
    return path


@pytest.fixture()
def shipped_registry() -> ProfileRegistry:
    """Registry pointed at the engine's actual on-disk profiles dir."""
    return ProfileRegistry()


@pytest.fixture()
def profiles_dir(tmp_path: Path) -> Path:
    """Isolated profiles dir for open-layer override / disable tests."""
    d = tmp_path / "profiles"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Structural — vmp_algorithm_extraction loads cleanly
# ---------------------------------------------------------------------------


def test_vmp_profile_loads_with_base_chain(shipped_registry):
    merged = shipped_registry.load_chain(VMP_PROFILE)
    assert merged.name == VMP_PROFILE
    assert merged.chain == (BASE_PROFILE_NAME, VMP_PROFILE)


def test_vmp_profile_evidence_classes_ordered_a_b_c(shipped_registry):
    merged = shipped_registry.load_chain(VMP_PROFILE)
    ids = [ec.id for ec in merged.evidence_classes]
    assert ids == ["A", "B", "C"]


def test_vmp_profile_node_states_include_canonical_five(shipped_registry):
    merged = shipped_registry.load_chain(VMP_PROFILE)
    names = {s.name for s in merged.node_states}
    expected = {
        "closed_form",
        "env_fixed_observed",
        "observed_stable",
        "stuck",
        "capability_gap",
    }
    assert expected.issubset(names)


def test_vmp_profile_binds_closure_state_role_to_closed_form(shipped_registry):
    """The role indirection point — base mechanism gates that reference
    `closure_state` will resolve to `closed_form` in this domain.
    Other domains bind their own state to the same role."""
    merged = shipped_registry.load_chain(VMP_PROFILE)
    closed_form = next(s for s in merged.node_states if s.name == "closed_form")
    assert "closure_state" in closed_form.roles


def test_vmp_profile_length_chain_check_registered_non_mechanism(shipped_registry):
    merged = shipped_registry.load_chain(VMP_PROFILE)
    lcc = next(p for p in merged.probes if p.name == "length_chain_check")
    assert lcc.mechanism is False, (
        "length_chain_check must NOT be mechanism — it's a VMP-specific "
        "invariant, and putting it in base would break the open-layer "
        "principle for other domains that don't have length chains."
    )
    assert "length_chain_check" not in merged.mechanism_probe_names


def test_vmp_profile_keeps_base_mechanism_probes_intact(shipped_registry):
    """Base mechanism probes survive the domain inheritance — the
    domain layer adds, it doesn't remove from baseline."""
    merged = shipped_registry.load_chain(VMP_PROFILE)
    expected_mechanism = {
        "m1_success_audit",
        "m3_bypass_block",
        "constant_provenance",
        "value_provenance",
        "watch_first_write",
    }
    assert expected_mechanism.issubset(merged.mechanism_probe_names)
    assert expected_mechanism.issubset({p.name for p in merged.probes})


def test_vmp_profile_scope_rule_env_fixed_observed_env_bound(shipped_registry):
    merged = shipped_registry.load_chain(VMP_PROFILE)
    rule = next(
        s for s in merged.scope_semantics if s.when_state == "env_fixed_observed"
    )
    assert rule.tag_scope == "env_bound"


def test_vmp_profile_passes_domain_lint(shipped_registry):
    """The shipped vmp profile must not contain mechanism: true entries."""
    from engine.profile.lint import lint_domain_profile

    vmp_raw = shipped_registry._load_raw(VMP_PROFILE)  # internal; fine in tests
    assert lint_domain_profile(vmp_raw) == []


# ---------------------------------------------------------------------------
# Open-layer demonstration — domain probes are freely overridable / disable-able
# ---------------------------------------------------------------------------


def _mirror_base(profiles_dir: Path) -> None:
    """Construct a base + vmp pair in the temp dir mirroring the shipped
    shape closely enough for inheritance tests."""
    _write(
        profiles_dir,
        BASE_PROFILE_NAME,
        {
            "profile": BASE_PROFILE_NAME,
            "probes": [
                {"name": "m1_success_audit",   "mechanism": True},
                {"name": "m3_bypass_block",    "mechanism": True},
                {"name": "constant_provenance", "mechanism": True},
                {"name": "value_provenance",   "mechanism": True},
                {"name": "watch_first_write",  "mechanism": True},
            ],
        },
    )
    _write(
        profiles_dir,
        VMP_PROFILE,
        {
            "profile": VMP_PROFILE,
            "inherits": BASE_PROFILE_NAME,
            "evidence_classes": [
                {"id": "A"}, {"id": "B"}, {"id": "C"},
            ],
            "node_states": [
                {"name": "closed_form", "roles": ["closure_state"]},
                {"name": "env_fixed_observed", "roles": []},
            ],
            "probes": [{"name": "length_chain_check"}],
            "scope_semantics": [
                {"when_state": "env_fixed_observed", "tag_scope": "env_bound"}
            ],
        },
    )


def test_subprofile_can_override_length_chain_check(profiles_dir):
    """A user subprofile that re-declares length_chain_check with their
    own module pointer must merge successfully — domain layer is open."""
    _mirror_base(profiles_dir)
    _write(
        profiles_dir,
        "vmp_with_custom_lcc",
        {
            "profile": "vmp_with_custom_lcc",
            "inherits": VMP_PROFILE,
            "probes": [
                {
                    "name": "length_chain_check",
                    "module": "my_pkg.my_custom_lcc",
                }
            ],
        },
    )
    reg = ProfileRegistry(profiles_dir)
    merged = reg.load_chain("vmp_with_custom_lcc")
    lcc = next(p for p in merged.probes if p.name == "length_chain_check")
    assert lcc.module == "my_pkg.my_custom_lcc"
    # Base mechanism probes still intact:
    assert "m1_success_audit" in merged.mechanism_probe_names


def test_subprofile_can_disable_length_chain_check(profiles_dir):
    """`disable:` on a domain probe removes it from the merged set,
    while leaving every base mechanism intact."""
    _mirror_base(profiles_dir)
    _write(
        profiles_dir,
        "vmp_without_lcc",
        {
            "profile": "vmp_without_lcc",
            "inherits": VMP_PROFILE,
            "disable": ["length_chain_check"],
        },
    )
    reg = ProfileRegistry(profiles_dir)
    merged = reg.load_chain("vmp_without_lcc")

    assert "length_chain_check" not in {p.name for p in merged.probes}
    # Mechanism probes still all present:
    expected_mech = {
        "m1_success_audit", "m3_bypass_block", "constant_provenance",
        "value_provenance", "watch_first_write",
    }
    assert expected_mech.issubset(merged.mechanism_probe_names)


def test_disabling_unknown_target_is_silent_noop(profiles_dir):
    """Typo-tolerant: an unknown disable target just doesn't match
    anything on the chain; merge succeeds."""
    _mirror_base(profiles_dir)
    _write(
        profiles_dir,
        "vmp_typo",
        {
            "profile": "vmp_typo",
            "inherits": VMP_PROFILE,
            "disable": ["lenght_chain_chek"],  # nb. spelling
        },
    )
    reg = ProfileRegistry(profiles_dir)
    merged = reg.load_chain("vmp_typo")
    # Original probe still there, since disable target didn't match:
    assert "length_chain_check" in {p.name for p in merged.probes}


# ---------------------------------------------------------------------------
# length_chain_check probe behavior
# ---------------------------------------------------------------------------


def test_length_chain_check_probe_is_registered_as_builtin():
    assert get_builtin_probe_class("length_chain_check") is LengthChainCheckProbe


def _consistent_chain_params() -> dict:
    """4:3 base64 ratio on every adjacent pair — fully explainable."""
    return {
        "report": {
            "length_chain": [
                {"name": "raw",        "length": 21},
                {"name": "b64",        "length": 28},
            ]
        }
    }


def _unexplained_chain_params() -> dict:
    """22 → 21 is neither equal, multiple, nor any recognised ratio."""
    return {
        "report": {
            "length_chain": [
                {"name": "a", "length": 22},
                {"name": "b", "length": 21},
            ]
        }
    }


def test_length_chain_probe_consistent_chain_returns_pass():
    verdict = LengthChainCheckProbe().run(
        ProbeContext(method="record_chain", params=_consistent_chain_params())
    )
    assert verdict.result == "pass"
    assert verdict.evidence["unexplained_edge_count"] == 0


def test_length_chain_probe_unexplained_edge_returns_fail():
    verdict = LengthChainCheckProbe().run(
        ProbeContext(method="record_chain", params=_unexplained_chain_params())
    )
    assert verdict.result == "fail"
    assert verdict.evidence["unexplained_edge_count"] >= 1


def test_length_chain_probe_no_chain_returns_undetermined():
    verdict = LengthChainCheckProbe().run(
        ProbeContext(method="record_chain", params={"unrelated": 1})
    )
    assert verdict.result == "undetermined"
