"""M1 success-audit framework gate — acceptance tests.

Mirrors the manual reference target ``success_audit.md`` round: a 94/94
prefix-fixed sample set must be intercepted and downgraded by the
framework rather than relying on agent self-discipline.

Coverage:
  - target_success=true with a pinned dimension → evidence_class=B,
    action=downgrade, params rewritten to strong_partial.
  - hard-evidence claim (varied dims, cross-session, multi-path
    closure) → A / allow.
  - low pass_rate → C / reject (intercept, no dispatch).
  - independent toggle UTOV_M1_AUDIT=off bypasses the gate entirely.
  - DisciplineWrapper exposes the audit block on the envelope.
"""

from __future__ import annotations

from engine.discipline_wrapper import DisciplineWrapper
from engine.m1_success_audit import (
    M1AuditConfig,
    apply_audit_to_params,
    audit_success_claim,
)
from engine.methodology import MethodologyConfig


# ---------------------------------------------------------------------------
# Module-level audit grading
# ---------------------------------------------------------------------------


def _prefix_fixed_samples(n: int = 94) -> list[dict[str, object]]:
    """N samples with prefix held constant — the reference target trap."""
    return [
        {"prefix": "fixed_xyz",
         "body_len": 22 + (i % 7),
         "key":      f"k{i:03d}"}
        for i in range(n)
    ]


def _varied_samples(n: int = 20) -> list[dict[str, object]]:
    return [
        {"prefix":   f"p{i % 5}",
         "body_len": 16 + i,
         "key":      f"k{i:03d}"}
        for i in range(n)
    ]


def test_b_class_downgrade_when_dimension_is_pinned():
    """The headline reference target case: 94/94 pass, but `prefix` is fixed.
    The gate must downgrade to strong_partial and rewrite the params."""
    cfg = M1AuditConfig()
    params = {
        "report": {
            "target_success":       True,
            "archival_allowed":     True,
            "success_dependencies": ["prefix", "body_len", "key"],
            "samples":              _prefix_fixed_samples(94),
            "pass_rate":            1.0,
            "scope":                "in_session",
            "closure_paths":        [
                {"name": "cfbc",    "digest": "abc"},
                {"name": "formula", "digest": "abc"},
                {"name": "hook",    "digest": "abc"},
            ],
        },
    }
    audit = audit_success_claim("promote_to_finding", params, cfg=cfg)
    assert audit is not None
    assert audit.evidence_class == "B"
    assert audit.action == "downgrade"
    assert audit.downgraded_to == "strong_partial"
    assert "prefix" in audit.untested_dimensions
    assert audit.overfit_flag is True
    assert audit.closure_consistent is True

    # apply_audit_to_params must rewrite the truthy claims to False
    apply_audit_to_params(params, audit)
    assert params["report"]["target_success"]   is False
    assert params["report"]["archival_allowed"] is False
    assert params["evidence_class"] == "B"
    assert params["downgraded_to"]  == "strong_partial"
    assert "m1_audit" in params


def test_a_class_passes_when_dimensions_varied_and_closure_holds():
    cfg = M1AuditConfig()
    params = {
        "target_success":       True,
        "success_dependencies": ["prefix", "body_len", "key"],
        "samples":              _varied_samples(20),
        "pass_rate":            1.0,
        "scope":                "cross_session",
        "closure_paths": [
            {"name": "cfbc",    "digest": "xyz"},
            {"name": "formula", "digest": "xyz"},
        ],
    }
    audit = audit_success_claim("promote_to_finding", params, cfg=cfg)
    assert audit is not None
    assert audit.evidence_class == "A"
    assert audit.action == "allow"
    assert audit.untested_dimensions == ()
    assert audit.overfit_flag is False


def test_c_class_rejects_when_pass_rate_below_floor():
    cfg = M1AuditConfig(min_pass_rate=0.95)
    params = {
        "target_success":       True,
        "success_dependencies": ["prefix"],
        "samples":              _varied_samples(10),
        "pass_rate":            0.30,
        "scope":                "cross_session",
    }
    audit = audit_success_claim("promote_to_finding", params, cfg=cfg)
    assert audit is not None
    assert audit.evidence_class == "C"
    assert audit.action == "reject"
    assert audit.intercepted_reason is not None
    assert "pass_rate" in audit.intercepted_reason


def test_c_class_rejects_when_closure_paths_disagree():
    cfg = M1AuditConfig()
    params = {
        "target_success":       True,
        "success_dependencies": ["prefix"],
        "samples":              _varied_samples(10),
        "pass_rate":            1.0,
        "scope":                "cross_session",
        "closure_paths": [
            {"name": "cfbc",    "digest": "aaa"},
            {"name": "formula", "digest": "bbb"},  # disagreement
        ],
    }
    audit = audit_success_claim("promote_to_finding", params, cfg=cfg)
    assert audit is not None
    assert audit.action == "reject"
    assert "closure" in (audit.intercepted_reason or "")


def test_audit_skipped_when_no_claim_present():
    """A method with no positive target_success / archival_allowed and
    not on the archival_methods whitelist must not trigger the gate."""
    audit = audit_success_claim(
        "verify_handler_binops",
        {"checked": 100, "passed": 100},
        cfg=M1AuditConfig(),
    )
    assert audit is None


def test_toggle_disables_module():
    cfg = M1AuditConfig(enabled=False)
    params = {"target_success": True, "success_dependencies": ["x"], "samples": []}
    assert audit_success_claim("promote_to_finding", params, cfg=cfg) is None


def test_env_toggle_off():
    cfg = M1AuditConfig.from_env({"UTOV_M1_AUDIT": "off"})
    assert cfg.enabled is False


# ---------------------------------------------------------------------------
# Discipline-wrapper integration
# ---------------------------------------------------------------------------


def test_wrapper_intercepts_c_class_archival():
    """An archival call whose evidence grades C must not dispatch.
    Verifies the wrapper turns reject into a JSON-RPC-style refusal."""
    wrapper = DisciplineWrapper(
        config=MethodologyConfig(),
        m1_audit_config=M1AuditConfig(min_pass_rate=0.95),
    )
    dispatched: list[str] = []

    def dispatch(method, params):
        dispatched.append(method)
        return {"ok": True}

    params = {
        "target_success":       True,
        "success_dependencies": ["prefix"],
        "samples":              _varied_samples(10),
        "pass_rate":            0.20,
        "scope":                "cross_session",
    }
    result, env = wrapper.step("promote_to_finding", params, dispatch)
    assert result is None, "rejected archival must not be dispatched"
    assert dispatched == [], "dispatch_fn must never be called"
    assert env.intercepted is True
    assert env.m1_audit is not None
    assert env.m1_audit["evidence_class"] == "C"


def test_wrapper_downgrades_b_class_and_dispatches():
    """The acceptance scenario from the user request: simulate a
    target_success=true archival with a pinned dimension; confirm the
    wrapper intercepts the unaudited claim, grades B, and downgrades
    to strong_partial before letting dispatch proceed."""
    wrapper = DisciplineWrapper(
        config=MethodologyConfig(),
        m1_audit_config=M1AuditConfig(),
    )
    captured: dict[str, object] = {}

    def dispatch(method, params):
        # Snapshot what the downstream archival actually receives,
        # so we can assert it was rewritten to strong_partial.
        captured["method"] = method
        captured["target_success_after"] = _walk_first(params, "target_success")
        captured["archival_allowed_after"] = _walk_first(params, "archival_allowed")
        captured["evidence_class"] = params.get("evidence_class")
        captured["downgraded_to"] = params.get("downgraded_to")
        return {"archived": "as_strong_partial"}

    params = {
        "report": {
            "target_success":       True,
            "archival_allowed":     True,
            "success_dependencies": ["prefix", "body_len", "key"],
            "samples":              _prefix_fixed_samples(94),
            "pass_rate":            1.0,
            "scope":                "in_session",
            "closure_paths": [
                {"name": "cfbc",    "digest": "X"},
                {"name": "formula", "digest": "X"},
                {"name": "hook",    "digest": "X"},
            ],
        },
    }
    result, env = wrapper.step("promote_to_finding", params, dispatch)

    # 1. Dispatch happened (B is "gate passes with annotation").
    assert result == {"archived": "as_strong_partial"}
    assert env.intercepted is False

    # 2. Audit block is on the envelope, graded B / downgrade.
    assert env.m1_audit is not None
    assert env.m1_audit["evidence_class"] == "B"
    assert env.m1_audit["action"] == "downgrade"
    assert env.m1_audit["downgraded_to"] == "strong_partial"
    assert "prefix" in env.m1_audit["untested_dimensions"]

    # 3. The dispatched params were rewritten before dispatch saw them.
    assert captured["target_success_after"]   is False
    assert captured["archival_allowed_after"] is False
    assert captured["evidence_class"] == "B"
    assert captured["downgraded_to"]  == "strong_partial"

    # 4. An alert string was added so the agent reads the downgrade.
    assert any("DOWNGRADE" in a for a in env.alerts)


def test_wrapper_passes_a_class_unchanged():
    wrapper = DisciplineWrapper(
        config=MethodologyConfig(),
        m1_audit_config=M1AuditConfig(),
    )
    seen: list[bool] = []

    def dispatch(method, params):
        seen.append(_walk_first(params, "target_success"))
        return {"ok": True}

    params = {
        "target_success":       True,
        "success_dependencies": ["prefix", "body_len", "key"],
        "samples":              _varied_samples(20),
        "pass_rate":            1.0,
        "scope":                "cross_session",
        "closure_paths": [
            {"name": "cfbc",    "digest": "z"},
            {"name": "formula", "digest": "z"},
        ],
    }
    result, env = wrapper.step("promote_to_finding", params, dispatch)
    assert result == {"ok": True}
    assert env.intercepted is False
    assert env.m1_audit is not None
    assert env.m1_audit["evidence_class"] == "A"
    # A-class leaves the claim intact.
    assert seen == [True]


def test_wrapper_skipped_on_non_archival_methods():
    """The gate must not run on regular tool calls; the existing
    methodology checks must keep working unchanged."""
    wrapper = DisciplineWrapper(
        config=MethodologyConfig(),
        m1_audit_config=M1AuditConfig(),
    )
    result, env = wrapper.step(
        "verify_handler_binops",
        {"checked": 100, "passed": 100},
        lambda m, p: {"x": 1},
    )
    assert result == {"x": 1}
    assert env.intercepted is False
    assert env.m1_audit is None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _walk_first(node, key):
    """Return the first value for ``key`` found anywhere in ``node``."""
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for v in node.values():
            r = _walk_first(v, key)
            if r is not None:
                return r
    elif isinstance(node, list):
        for v in node:
            r = _walk_first(v, key)
            if r is not None:
                return r
    return None
