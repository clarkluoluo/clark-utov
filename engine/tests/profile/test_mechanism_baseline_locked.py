"""Mechanism-baseline cannot be turned off — adversarial test (§19.7 #7).

This file accumulates Lock A / B / C arms as mechanism probes migrate.
Coverage through step 5 (this commit):

  * **Lock A** (load-time registry) — M1, M3, constant_provenance,
    value_provenance, and watch_first_write each cannot be overridden
    by name redeclaration nor ``disable:``d by a subprofile.
  * **Lock B** (runtime gate force-include) — even when a caller
    constructs a tampered :class:`MergedProfile` (mechanism probes
    removed from ``probes``, ``mechanism_probe_names`` zeroed,
    registry bypassed entirely), the conjunctive gate STILL runs
    every mechanism probe via class-attribute scan from bytecode.
  * **Probe interfaces** — each mechanism probe behaves correctly on
    its core paths.
  * **Builtin registry** — all five self-register and are reachable by
    :func:`get_builtin_probe_class`.

Not in scope for this file:

  * **Lock C** — kernel / base-source literal scans live in
    ``test_base_domain_boundary_lint.py``; the mechanism modules'
    implementation files enter that scan as they migrate, but the lint
    test fixtures are synthetic — exercising the real source scan
    against the on-disk modules is a separate acceptance.

Length-chain-check is a *domain* probe — its open-layer behaviour
(override / disable freely allowed) lives in
``test_vmp_profile_regression.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.profile import (
    BASE_PROFILE_NAME,
    ConjunctiveGate,
    EvidenceClassCap,
    MergedProfile,
    ProbeContext,
    ProfileMergeError,
)
from engine.profile.probe_runtime import (
    get_builtin_probe_class,
    list_builtin_probes,
)
from engine.profile.probes.constant_provenance import ConstantProvenanceProbe
from engine.profile.probes.m1 import M1SuccessAuditProbe
from engine.profile.probes.m3 import M3BypassBlockProbe
from engine.profile.probes.value_provenance import ValueProvenanceProbe
from engine.profile.probes.watch_first_write import WatchFirstWriteProbe
from engine.profile.registry import ProfileRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(profiles_dir: Path, name: str, body: dict) -> Path:
    path = profiles_dir / f"{name}.json"
    path.write_text(json.dumps(body))
    return path


def _base_with_m1(profiles_dir: Path) -> Path:
    """A base.json fixture with M1 only — narrower than the shipped one,
    used to isolate the M1-specific Lock A assertions."""
    return _write(
        profiles_dir,
        BASE_PROFILE_NAME,
        {
            "profile": BASE_PROFILE_NAME,
            "probes": [
                {
                    "name": "m1_success_audit",
                    "mechanism": True,
                    "inputs": ["method", "params"],
                    "outputs": ["m1_audit"],
                }
            ],
        },
    )


def _base_with_all_three(profiles_dir: Path) -> Path:
    """A base.json fixture with the three step-3 mechanisms — kept
    around because some step-3 tests want a narrower set than the
    shipped base."""
    return _write(
        profiles_dir,
        BASE_PROFILE_NAME,
        {
            "profile": BASE_PROFILE_NAME,
            "probes": [
                {"name": "m1_success_audit",   "mechanism": True},
                {"name": "m3_bypass_block",    "mechanism": True},
                {"name": "constant_provenance", "mechanism": True},
            ],
        },
    )


def _base_with_all_five(profiles_dir: Path) -> Path:
    """A base.json fixture mirroring the current shipped mechanism set."""
    return _write(
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


@pytest.fixture()
def profiles_dir(tmp_path: Path) -> Path:
    d = tmp_path / "profiles"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Builtin registry — M1 self-registers
# ---------------------------------------------------------------------------


def test_m1_probe_is_registered_as_builtin():
    cls = get_builtin_probe_class("m1_success_audit")
    assert cls is M1SuccessAuditProbe


def test_m1_probe_listed_in_builtin_snapshot():
    assert "m1_success_audit" in list_builtin_probes()


def test_m1_probe_carries_correct_metadata():
    assert M1SuccessAuditProbe.name == "m1_success_audit"
    assert "method" in M1SuccessAuditProbe.inputs
    assert "params" in M1SuccessAuditProbe.inputs


# ---------------------------------------------------------------------------
# Shipped base.json — M1 reachable via merged profile
# ---------------------------------------------------------------------------


def test_shipped_base_advertises_m1_as_mechanism():
    reg = ProfileRegistry()
    merged = reg.load_chain(BASE_PROFILE_NAME)
    assert "m1_success_audit" in merged.mechanism_probe_names


# ---------------------------------------------------------------------------
# Lock A · M1 cannot be overridden by name redeclaration
# ---------------------------------------------------------------------------


def test_subprofile_redeclaring_m1_with_noop_fails(profiles_dir):
    """An attacker writes a domain probe with the same name to noop M1."""
    _base_with_m1(profiles_dir)
    _write(
        profiles_dir,
        "attacker_override",
        {
            "profile": "attacker_override",
            "probes": [{"name": "m1_success_audit", "mechanism": False}],
        },
    )
    reg = ProfileRegistry(profiles_dir)
    with pytest.raises(ProfileMergeError, match="m1_success_audit.*mechanism-locked"):
        reg.load_chain("attacker_override")


def test_subprofile_redeclaring_m1_as_mechanism_also_fails(profiles_dir):
    """Even if the attacker keeps `mechanism: true`, the name collides
    with base and the override is refused (only base may declare mechanism)."""
    _base_with_m1(profiles_dir)
    _write(
        profiles_dir,
        "attacker_mech",
        {
            "profile": "attacker_mech",
            "probes": [{"name": "m1_success_audit", "mechanism": True}],
        },
    )
    reg = ProfileRegistry(profiles_dir)
    # Two checks here both fail this profile — the mechanism-claim arm
    # fires first by ordering, which is the more specific complaint.
    with pytest.raises(
        ProfileMergeError, match="only base may declare mechanism|mechanism-locked"
    ):
        reg.load_chain("attacker_mech")


# ---------------------------------------------------------------------------
# Lock A · M1 cannot be `disable:`d
# ---------------------------------------------------------------------------


def test_subprofile_disabling_m1_fails(profiles_dir):
    _base_with_m1(profiles_dir)
    _write(
        profiles_dir,
        "attacker_disable",
        {
            "profile": "attacker_disable",
            "disable": ["m1_success_audit"],
        },
    )
    reg = ProfileRegistry(profiles_dir)
    with pytest.raises(ProfileMergeError, match="disable mechanism entry"):
        reg.load_chain("attacker_disable")


def test_disable_only_legal_on_subprofiles_not_base(profiles_dir):
    """Loader rejects `disable:` on the base profile outright — base is
    the mechanism source, it has nothing to disable upstream of itself."""
    from engine.profile.loader import ProfileLoadError, load_profile_file

    _write(
        profiles_dir,
        BASE_PROFILE_NAME,
        {"profile": BASE_PROFILE_NAME, "disable": ["x"]},
    )
    with pytest.raises(ProfileLoadError, match="not allowed on the base profile"):
        load_profile_file(profiles_dir / "base.json", is_base=True)


# ---------------------------------------------------------------------------
# M1 probe behavior — the three audit-action paths
# ---------------------------------------------------------------------------


def _archival_params(
    *,
    pass_rate: float = 1.0,
    samples: list[dict] | None = None,
    dependencies: list[str] | None = None,
    scope: str = "in_session",
) -> dict:
    """Build a params dict shaped like a typical archival surface."""
    return {
        "report": {
            "target_success": True,
            "archival_allowed": True,
            "success_dependencies": dependencies or ["prefix", "body_len", "key"],
            "samples": samples or [
                {"prefix": "fixed_xyz", "body_len": 22 + (i % 7), "key": f"k{i:03d}"}
                for i in range(94)
            ],
            "pass_rate": pass_rate,
            "scope": scope,
            "closure_paths": [
                {"name": "cfbc", "digest": "abc"},
                {"name": "formula", "digest": "abc"},
                {"name": "hook", "digest": "abc"},
            ],
        }
    }


def test_m1_probe_downgrade_path_returns_fail_with_cap():
    """The headline reference target case — 94/94 pass with a pinned `prefix`
    dimension. M1 audit grades B and downgrades; the probe must surface
    this as result='fail' with an evidence-class cap of 'B'."""
    probe = M1SuccessAuditProbe()
    ctx = ProbeContext(method="promote_to_finding", params=_archival_params())
    verdict = probe.run(ctx)

    assert verdict.probe == "m1_success_audit"
    assert verdict.result == "fail"
    assert verdict.evidence["action"] == "downgrade"
    assert verdict.evidence["grade"] == "B"
    assert "prefix" in verdict.evidence["untested_dimensions"]
    assert verdict.evidence["overfit_flag"] is True
    assert isinstance(verdict.affects_evidence_class, EvidenceClassCap)
    assert verdict.affects_evidence_class.class_id == "B"


def test_m1_probe_allow_path_returns_pass():
    """Varied dimensions, cross-session scope, multi-path closure
    consistent → M1 grades A → result='pass'."""
    varied_samples = [
        {"prefix": f"p{i % 5}", "body_len": 16 + i, "key": f"k{i:03d}"}
        for i in range(20)
    ]
    params = _archival_params(samples=varied_samples, scope="cross_session")
    verdict = M1SuccessAuditProbe().run(
        ProbeContext(method="promote_to_finding", params=params)
    )
    assert verdict.result == "pass"
    assert verdict.evidence["grade"] == "A"
    assert verdict.affects_evidence_class.class_id == "A"


def test_m1_probe_reject_path_returns_fail():
    """A success claim with pass_rate below the floor → action='reject'
    → result='fail' with an intercepted_reason populated."""
    low_pass_params = _archival_params(pass_rate=0.10)
    verdict = M1SuccessAuditProbe().run(
        ProbeContext(method="promote_to_finding", params=low_pass_params)
    )
    assert verdict.result == "fail"
    assert verdict.evidence["action"] == "reject"
    assert verdict.evidence["intercepted_reason"]


def test_m1_probe_non_archival_surface_returns_undetermined():
    """A method that isn't an archival surface, with no positive claim
    in params — M1 doesn't apply; verdict is 'undetermined' so the
    conjunctive gate doesn't fail spuriously on unrelated calls."""
    verdict = M1SuccessAuditProbe().run(
        ProbeContext(method="get_hyp_tree", params={"depth": 3})
    )
    assert verdict.result == "undetermined"
    assert verdict.affects_evidence_class is None


# ---------------------------------------------------------------------------
# Builtin registry — M3 + constant_provenance also self-register
# ---------------------------------------------------------------------------


def test_m3_probe_is_registered_as_builtin():
    assert get_builtin_probe_class("m3_bypass_block") is M3BypassBlockProbe


def test_constant_provenance_probe_is_registered_as_builtin():
    assert get_builtin_probe_class("constant_provenance") is ConstantProvenanceProbe


def test_all_three_mechanisms_listed_in_builtin_snapshot():
    snapshot = list_builtin_probes()
    assert "m1_success_audit" in snapshot
    assert "m3_bypass_block" in snapshot
    assert "constant_provenance" in snapshot


def test_shipped_base_advertises_all_five_as_mechanism():
    reg = ProfileRegistry()
    merged = reg.load_chain(BASE_PROFILE_NAME)
    expected = {
        "m1_success_audit",
        "m3_bypass_block",
        "constant_provenance",
        "value_provenance",
        "watch_first_write",
    }
    assert expected.issubset(merged.mechanism_probe_names)


def test_value_provenance_probe_is_registered_as_builtin():
    assert get_builtin_probe_class("value_provenance") is ValueProvenanceProbe


def test_watch_first_write_probe_is_registered_as_builtin():
    assert get_builtin_probe_class("watch_first_write") is WatchFirstWriteProbe


# ---------------------------------------------------------------------------
# Lock A · M3 cannot be overridden or disabled
# ---------------------------------------------------------------------------


def test_subprofile_redeclaring_m3_fails(profiles_dir):
    _base_with_all_three(profiles_dir)
    _write(
        profiles_dir,
        "attacker_m3_override",
        {
            "profile": "attacker_m3_override",
            "probes": [{"name": "m3_bypass_block", "mechanism": False}],
        },
    )
    reg = ProfileRegistry(profiles_dir)
    with pytest.raises(ProfileMergeError, match="m3_bypass_block.*mechanism-locked"):
        reg.load_chain("attacker_m3_override")


def test_subprofile_disabling_m3_fails(profiles_dir):
    _base_with_all_three(profiles_dir)
    _write(
        profiles_dir,
        "attacker_m3_disable",
        {"profile": "attacker_m3_disable", "disable": ["m3_bypass_block"]},
    )
    reg = ProfileRegistry(profiles_dir)
    with pytest.raises(ProfileMergeError, match="disable mechanism entry.*m3_bypass_block"):
        reg.load_chain("attacker_m3_disable")


# ---------------------------------------------------------------------------
# Lock A · constant_provenance cannot be overridden or disabled
# ---------------------------------------------------------------------------


def test_subprofile_redeclaring_constant_provenance_fails(profiles_dir):
    """Even an attempt to "redeclare 5-way thresholds" maps onto this
    case: profile schema has no threshold field, so the only attack
    surface is overriding the probe under the same name — and that
    fails the mechanism lock."""
    _base_with_all_three(profiles_dir)
    _write(
        profiles_dir,
        "attacker_cp_override",
        {
            "profile": "attacker_cp_override",
            "probes": [{"name": "constant_provenance", "mechanism": False}],
        },
    )
    reg = ProfileRegistry(profiles_dir)
    with pytest.raises(ProfileMergeError, match="constant_provenance.*mechanism-locked"):
        reg.load_chain("attacker_cp_override")


def test_subprofile_disabling_constant_provenance_fails(profiles_dir):
    _base_with_all_three(profiles_dir)
    _write(
        profiles_dir,
        "attacker_cp_disable",
        {"profile": "attacker_cp_disable", "disable": ["constant_provenance"]},
    )
    reg = ProfileRegistry(profiles_dir)
    with pytest.raises(
        ProfileMergeError, match="disable mechanism entry.*constant_provenance"
    ):
        reg.load_chain("attacker_cp_disable")


# ---------------------------------------------------------------------------
# M3 probe behavior — recorded / triggered / followup-refused / undetermined
# ---------------------------------------------------------------------------


def _m3_attempt_params(
    block_id: str, observation_method: str, *, failed: bool = True
) -> dict:
    """Build a verify_block_variability call payload."""
    return {
        "block_id": block_id,
        "observation_method": observation_method,
        "failed": failed,
    }


def test_m3_probe_recorded_attempt_returns_pass():
    """A single failed observation — below threshold — records but
    doesn't fail the gate. (Single-method failure could be an obs bug,
    not a real bypass — M3 explicitly waits for cross-method evidence.)"""
    probe = M3BypassBlockProbe()
    verdict = probe.run(
        ProbeContext(
            method="verify_block_variability",
            params=_m3_attempt_params("blk1", "hook_pre"),
        )
    )
    assert verdict.result == "pass"
    assert verdict.evidence["phase"] == "recorded"
    assert probe.detector.is_known_bypass("blk1") is False


def test_m3_probe_triggered_on_second_distinct_method_returns_fail():
    """The headline reference target case: 2 distinct observation methods both
    report variability=0 → flip to suspected_bypass on the second call."""
    probe = M3BypassBlockProbe()
    first = probe.run(
        ProbeContext(
            method="verify_block_variability",
            params=_m3_attempt_params("blk_sm3", "hook_pre"),
        )
    )
    assert first.result == "pass"

    second = probe.run(
        ProbeContext(
            method="verify_block_variability",
            params=_m3_attempt_params("blk_sm3", "hook_post"),
        )
    )
    assert second.result == "fail"
    assert second.evidence["phase"] == "triggered"
    assert set(second.evidence["failed_methods"]) == {"hook_pre", "hook_post"}
    assert probe.detector.is_known_bypass("blk_sm3") is True


def test_m3_probe_followup_on_confirmed_bypass_returns_fail():
    """After bypass is confirmed, any further observation method on
    that block is refused — switching posture on a dead block is the
    anti-pattern M3 exists to prevent."""
    probe = M3BypassBlockProbe()
    probe.run(
        ProbeContext(
            method="verify_block_variability",
            params=_m3_attempt_params("blk", "a"),
        )
    )
    probe.run(
        ProbeContext(
            method="verify_block_variability",
            params=_m3_attempt_params("blk", "b"),
        )
    )
    # Now blk is confirmed bypass. A third observation method must be refused.
    followup = probe.run(
        ProbeContext(
            method="verify_block_variability",
            params=_m3_attempt_params("blk", "c"),
        )
    )
    assert followup.result == "fail"
    assert followup.evidence["phase"] == "followup_refused"
    assert "dead block" in followup.evidence["intercepted_reason"]


def test_m3_probe_non_m3_method_returns_undetermined():
    """Methods outside M3's surface — the probe doesn't apply."""
    verdict = M3BypassBlockProbe().run(
        ProbeContext(
            method="promote_to_finding",
            params={"block_id": "blk", "observation_method": "x", "failed": True},
        )
    )
    assert verdict.result == "undetermined"


def test_m3_probe_other_block_unaffected_by_confirmed_bypass():
    """Confirmation is per-block; an unrelated block should still
    transit the recorded-attempt path normally."""
    probe = M3BypassBlockProbe()
    probe.run(
        ProbeContext(
            method="verify_block_variability",
            params=_m3_attempt_params("blk_bad", "a"),
        )
    )
    probe.run(
        ProbeContext(
            method="verify_block_variability",
            params=_m3_attempt_params("blk_bad", "b"),
        )
    )
    assert probe.detector.is_known_bypass("blk_bad") is True

    other = probe.run(
        ProbeContext(
            method="verify_block_variability",
            params=_m3_attempt_params("blk_other", "a"),
        )
    )
    assert other.result == "pass"
    assert other.evidence["block_id"] == "blk_other"


# ---------------------------------------------------------------------------
# constant_provenance probe behavior — 5-way classification + ceiling
# ---------------------------------------------------------------------------


def _cp_session_level_params() -> dict:
    """Stable reruns but producer reads session entropy — dataflow
    overrides into SESSION_LEVEL_DERIVED."""
    return {
        "values": [
            {
                "value_name": "template",
                "rerun_observations": [
                    {"dimension": "same_session", "value_hex": "aa"},
                    {"dimension": "same_session", "value_hex": "aa"},
                    {"dimension": "new_session", "value_hex": "aa"},
                    {"dimension": "new_appkey", "value_hex": "aa"},
                ],
                "producer_dataflow": {"producer_reads": ["static", "session_token"]},
            }
        ]
    }


def _cp_hardcoded_params() -> dict:
    """All reruns stable + dataflow strictly static → HARDCODED_FIXED
    with evidence-class ceiling 'A'."""
    return {
        "values": [
            {
                "value_name": "k_constant",
                "rerun_observations": [
                    {"dimension": "same_session", "value_hex": "9e"},
                    {"dimension": "new_session",  "value_hex": "9e"},
                    {"dimension": "new_appkey",   "value_hex": "9e"},
                    {"dimension": "new_per_input", "value_hex": "9e"},
                ],
                "producer_dataflow": {"producer_reads": ["static"]},
            }
        ]
    }


def test_constant_provenance_probe_session_level_caps_at_b():
    verdict = ConstantProvenanceProbe().run(
        ProbeContext(method="record_value", params=_cp_session_level_params())
    )
    assert verdict.result == "pass"
    assert verdict.affects_evidence_class is not None
    assert verdict.affects_evidence_class.class_id == "B"
    [single] = verdict.evidence["classifications"]
    assert single["category"] == "session_level_derived"


def test_constant_provenance_probe_hardcoded_caps_at_a():
    verdict = ConstantProvenanceProbe().run(
        ProbeContext(method="record_value", params=_cp_hardcoded_params())
    )
    assert verdict.result == "pass"
    assert verdict.affects_evidence_class is not None
    assert verdict.affects_evidence_class.class_id == "A"
    [single] = verdict.evidence["classifications"]
    assert single["category"] == "hardcoded_fixed"


def test_constant_provenance_probe_no_value_records_returns_undetermined():
    verdict = ConstantProvenanceProbe().run(
        ProbeContext(method="record_value", params={"unrelated": 1})
    )
    assert verdict.result == "undetermined"
    assert verdict.affects_evidence_class is None


def test_constant_provenance_probe_strictest_ceiling_across_multiple():
    """Two values — one HARDCODED (cap A), one SESSION_LEVEL (cap B).
    Strictest cap wins → 'B'. (String-min over A < B; step-5 replaces
    this with profile-ordering synthesis.)"""
    multi = {
        "values": [
            _cp_session_level_params()["values"][0],
            _cp_hardcoded_params()["values"][0],
        ]
    }
    verdict = ConstantProvenanceProbe().run(
        ProbeContext(method="record_value", params=multi)
    )
    assert verdict.result == "pass"
    assert verdict.affects_evidence_class.class_id == "B"
    assert len(verdict.evidence["classifications"]) == 2


# ---------------------------------------------------------------------------
# Lock A · value_provenance cannot be overridden or disabled
# ---------------------------------------------------------------------------


def test_subprofile_redeclaring_value_provenance_fails(profiles_dir):
    _base_with_all_five(profiles_dir)
    _write(
        profiles_dir,
        "attacker_vp_override",
        {
            "profile": "attacker_vp_override",
            "probes": [{"name": "value_provenance", "mechanism": False}],
        },
    )
    reg = ProfileRegistry(profiles_dir)
    with pytest.raises(ProfileMergeError, match="value_provenance.*mechanism-locked"):
        reg.load_chain("attacker_vp_override")


def test_subprofile_disabling_value_provenance_fails(profiles_dir):
    _base_with_all_five(profiles_dir)
    _write(
        profiles_dir,
        "attacker_vp_disable",
        {"profile": "attacker_vp_disable", "disable": ["value_provenance"]},
    )
    reg = ProfileRegistry(profiles_dir)
    with pytest.raises(ProfileMergeError, match="disable mechanism entry.*value_provenance"):
        reg.load_chain("attacker_vp_disable")


# ---------------------------------------------------------------------------
# Lock A · watch_first_write cannot be overridden or disabled
# ---------------------------------------------------------------------------


def test_subprofile_redeclaring_watch_first_write_fails(profiles_dir):
    _base_with_all_five(profiles_dir)
    _write(
        profiles_dir,
        "attacker_wfw_override",
        {
            "profile": "attacker_wfw_override",
            "probes": [{"name": "watch_first_write", "mechanism": False}],
        },
    )
    reg = ProfileRegistry(profiles_dir)
    with pytest.raises(ProfileMergeError, match="watch_first_write.*mechanism-locked"):
        reg.load_chain("attacker_wfw_override")


def test_subprofile_disabling_watch_first_write_fails(profiles_dir):
    _base_with_all_five(profiles_dir)
    _write(
        profiles_dir,
        "attacker_wfw_disable",
        {"profile": "attacker_wfw_disable", "disable": ["watch_first_write"]},
    )
    reg = ProfileRegistry(profiles_dir)
    with pytest.raises(
        ProfileMergeError, match="disable mechanism entry.*watch_first_write"
    ):
        reg.load_chain("attacker_wfw_disable")


# ---------------------------------------------------------------------------
# value_provenance probe behavior — provenance tagging + ceiling
# ---------------------------------------------------------------------------


def _vp_observed_params() -> dict:
    """Value sourced from a hook → observed → cap B."""
    return {
        "values": [
            {
                "value_name": "k_observed",
                "source": "hook",
                "evidence_class": "A",  # caller's optimistic claim
            }
        ]
    }


def _vp_closed_form_params() -> dict:
    """Value sourced from closed_form with verified recompute → cap A."""
    return {
        "values": [
            {
                "value_name": "k_recomputed",
                "source": "closed_form",
                "recompute_fn_present": True,
                "recompute_matches_measured": True,
            }
        ]
    }


def test_value_provenance_probe_observed_caps_at_b():
    verdict = ValueProvenanceProbe().run(
        ProbeContext(method="record_value", params=_vp_observed_params())
    )
    assert verdict.result == "pass"
    assert verdict.affects_evidence_class is not None
    assert verdict.affects_evidence_class.class_id == "B"
    [tagged] = verdict.evidence["tagged_values"]
    assert tagged["provenance"] == "observed"
    assert tagged["downgraded"] is True


def test_value_provenance_probe_closed_form_caps_at_a():
    verdict = ValueProvenanceProbe().run(
        ProbeContext(method="record_value", params=_vp_closed_form_params())
    )
    assert verdict.result == "pass"
    assert verdict.affects_evidence_class is not None
    assert verdict.affects_evidence_class.class_id == "A"


def test_value_provenance_probe_no_records_returns_undetermined():
    verdict = ValueProvenanceProbe().run(
        ProbeContext(method="record_value", params={"unrelated": 1})
    )
    assert verdict.result == "undetermined"
    assert verdict.affects_evidence_class is None


def test_value_provenance_probe_strictest_ceiling_across_multiple():
    """One closed_form (A) + one observed (B) → most-restrictive = B."""
    multi = {
        "values": [
            _vp_closed_form_params()["values"][0],
            _vp_observed_params()["values"][0],
        ]
    }
    verdict = ValueProvenanceProbe().run(
        ProbeContext(method="record_value", params=multi)
    )
    assert verdict.result == "pass"
    assert verdict.affects_evidence_class.class_id == "B"
    assert len(verdict.evidence["tagged_values"]) == 2


# ---------------------------------------------------------------------------
# watch_first_write probe behavior — advisory only, never fails
# ---------------------------------------------------------------------------


def _wfw_eligible_params() -> dict:
    """Observed value at a concrete landing_address, no closed-form
    recompute — the auto-suggestion trigger."""
    return {
        "values": [
            {
                "value_name": "observed_k",
                "source": "hook",
                "landing_address": 0x32302C,
            }
        ]
    }


def _wfw_ineligible_params() -> dict:
    """Closed-form record — no need to chase the writer."""
    return {
        "values": [
            {
                "value_name": "k_recomputed",
                "source": "closed_form",
                "recompute_fn_present": True,
                "recompute_matches_measured": True,
                "landing_address": 0x32302C,
            }
        ]
    }


def test_watch_first_write_probe_emits_suggestion_for_eligible_value():
    verdict = WatchFirstWriteProbe().run(
        ProbeContext(method="record_value", params=_wfw_eligible_params())
    )
    assert verdict.result == "pass"
    assert verdict.affects_evidence_class is None  # advisory, no cap
    assert verdict.evidence["count"] == 1
    [suggestion] = verdict.evidence["suggestions"]
    assert suggestion["value_name"] == "observed_k"
    assert "watch_first_write" in suggestion["advisory"]


def test_watch_first_write_probe_skips_closed_form_value():
    verdict = WatchFirstWriteProbe().run(
        ProbeContext(method="record_value", params=_wfw_ineligible_params())
    )
    assert verdict.result == "undetermined"


def test_watch_first_write_probe_no_records_returns_undetermined():
    verdict = WatchFirstWriteProbe().run(
        ProbeContext(method="record_value", params={"unrelated": 1})
    )
    assert verdict.result == "undetermined"


# ---------------------------------------------------------------------------
# Lock B · runtime gate force-include even against tampered MergedProfile
# ---------------------------------------------------------------------------
#
# The attacker has bypassed Lock A — for instance by constructing a
# MergedProfile in memory rather than going through ProfileRegistry.
# They strip mechanism probes out of the profile data: empty
# probe list, empty mechanism_probe_names. They then ask the gate
# runtime to evaluate.
#
# The gate must STILL run every mechanism probe. Its source of
# truth is the implementation class attribute ``mechanism = True``,
# read from the import-time decorator registry — not anything the
# attacker can reach by tampering with profile data.
#
# We verify this for each of the five mechanism probes with a fixture
# that the corresponding probe would normally fail (M1 with an overfit
# claim, M3 with two failed methods, etc.) and assert the failure
# fires from the gate output even when the merged profile says
# "I have no mechanism probes."


def _empty_tampered_profile() -> MergedProfile:
    """Construct a MergedProfile by hand, with every mechanism marker
    removed. Models an in-memory bypass-Lock-A attack: the attacker
    didn't go through ProfileRegistry at all."""
    return MergedProfile(
        name="attacker_in_memory",
        chain=("attacker_in_memory",),
        evidence_classes=(),
        node_states=(),
        probes=(),                          # ← attacker removed all probes
        gates=(),
        routing_rules=(),
        scope_semantics=(),
        scope_order=(),
        cap_mapping=(),
        task_templates=(),
        mechanism_probe_names=frozenset(),  # ← attacker zeroed the index
        mechanism_gate_ids=frozenset(),
    )


def _overfit_archival_params() -> dict:
    return _archival_params(samples=None)  # uses the default prefix-fixed 94


def test_lock_b_m1_fires_against_tampered_profile():
    """Tampered MergedProfile says "no mechanism" — gate STILL runs M1
    and STILL fails on the prefix-fixed 94/94 overfit claim."""
    gate = ConjunctiveGate(_empty_tampered_profile())
    result = gate.evaluate(
        ProbeContext(method="promote_to_finding", params=_overfit_archival_params())
    )
    assert result.passed is False
    assert "m1_success_audit" in result.failing_probes
    # Mechanism set wasn't read from the profile — bytecode scan picked
    # up every probe whose class declares ``mechanism = True`` (count
    # grows as new probes ship; the property is "≥ v0.3.0 baseline").
    assert len(result.mechanism_verdicts) >= 5


def test_lock_b_m3_fires_against_tampered_profile():
    """Same tampering, drive two failed-method observations through
    the gate; M3 still triggers."""
    gate = ConjunctiveGate(_empty_tampered_profile())
    fixture = {
        "block_id": "blk_sm3",
        "observation_method": "hook_pre",
        "failed": True,
    }
    # First call — recorded but below threshold (M3's correct behaviour)
    first = gate.evaluate(
        ProbeContext(method="verify_block_variability", params=fixture)
    )
    assert first.passed is True

    # Second call with a different observation method — M3 fires through gate
    second = gate.evaluate(
        ProbeContext(
            method="verify_block_variability",
            params={**fixture, "observation_method": "hook_post"},
        )
    )
    assert second.passed is False
    assert "m3_bypass_block" in second.failing_probes


def test_lock_b_constant_provenance_runs_against_tampered_profile():
    """CP is a classifier (always pass), but it must STILL emit a cap
    when value records are present — that cap goes into the gate's
    node_cap synth."""
    gate = ConjunctiveGate(_empty_tampered_profile())
    params = {
        "values": [
            {
                "value_name": "k",
                "rerun_observations": [
                    {"dimension": "same_session", "value_hex": "aa"},
                    {"dimension": "new_session",  "value_hex": "bb"},
                ],
            }
        ]
    }
    result = gate.evaluate(ProbeContext(method="record_value", params=params))
    cp_verdict = next(
        v for v in result.mechanism_verdicts if v.probe == "constant_provenance"
    )
    assert cp_verdict.result == "pass"
    assert cp_verdict.affects_evidence_class is not None


def test_lock_b_value_provenance_runs_against_tampered_profile():
    gate = ConjunctiveGate(_empty_tampered_profile())
    params = {"values": [{"value_name": "k", "source": "hook"}]}
    result = gate.evaluate(ProbeContext(method="record_value", params=params))
    vp_verdict = next(
        v for v in result.mechanism_verdicts if v.probe == "value_provenance"
    )
    assert vp_verdict.result == "pass"
    assert vp_verdict.affects_evidence_class is not None
    assert vp_verdict.affects_evidence_class.class_id == "B"


def test_lock_b_watch_first_write_runs_against_tampered_profile():
    gate = ConjunctiveGate(_empty_tampered_profile())
    params = {
        "values": [
            {"value_name": "k", "source": "hook", "landing_address": 0x12345678}
        ]
    }
    result = gate.evaluate(ProbeContext(method="record_value", params=params))
    wfw_verdict = next(
        v for v in result.mechanism_verdicts if v.probe == "watch_first_write"
    )
    assert wfw_verdict.result == "pass"
    assert wfw_verdict.evidence["count"] == 1


def test_lock_b_all_mechanism_probes_present_in_tampered_run():
    """Sanity check: regardless of the call, an empty-mechanism profile
    still yields verdicts for every mechanism probe registered at
    import time.  The v0.3.0 baseline is the lower bound; new
    mechanism probes ship by adding entries to the bytecode registry,
    not by editing this assertion."""
    gate = ConjunctiveGate(_empty_tampered_profile())
    result = gate.evaluate(ProbeContext(method="get_hyp_tree", params={}))
    probe_names = {v.probe for v in result.mechanism_verdicts}
    baseline = {
        "m1_success_audit",
        "m3_bypass_block",
        "constant_provenance",
        "value_provenance",
        "watch_first_write",
    }
    assert baseline.issubset(probe_names)


def test_lock_b_class_attribute_is_the_force_include_anchor():
    """Direct check: each mechanism probe class declares
    ``mechanism=True`` as a class attribute. This is what gate_runtime
    reads — the property a profile-data attacker cannot reach."""
    assert M1SuccessAuditProbe.mechanism is True
    assert M3BypassBlockProbe.mechanism is True
    assert ConstantProvenanceProbe.mechanism is True
    assert ValueProvenanceProbe.mechanism is True
    assert WatchFirstWriteProbe.mechanism is True
