"""Methodology / anti-drift wrapper unit tests + 5-point acceptance.

Acceptance set (from the user spec):
  1. footer present on every call
  2. periodic card injected around step 15-20
  3. contradiction triggers context-sensitive prompt
  4. 3 same-type failures triggers checkpoint-suggest prompt
  5. un-ledgered data reference triggers interception (refusal)

We exercise the wrapper directly against a stub dispatch so the test
is independent of the live agent_mode wire loop. The wire-level
plumbing is covered separately by the existing agent_mode tests.
"""

from __future__ import annotations

import os

import pytest

from engine.discipline_wrapper import DisciplineRaise, DisciplineWrapper
from engine.methodology import MethodologyConfig, MethodologyState


def _ok_dispatch(method: str, params: dict) -> dict:
    """Stub dispatch — returns a simple OK envelope."""
    return {"ok": True, "method": method}


def _failing_dispatch(method: str, params: dict) -> dict:
    raise RuntimeError(f"{method} crashed")


def test_footer_is_present_on_every_call():
    w = DisciplineWrapper(config=MethodologyConfig(periodic_interval=99))
    for _ in range(5):
        _, env = w.step("get_findings", {}, _ok_dispatch)
        assert env.footer, "footer must be set on every result"
        assert "方法论自检" in env.footer
        assert "[继续]" in env.footer


def test_periodic_card_appears_within_interval_window():
    """Acceptance #2: a 20-step session must produce a full card
    around step 15-20."""
    cfg = MethodologyConfig(periodic_interval=15)
    w = DisciplineWrapper(config=cfg)
    cards_at: list[int] = []
    for i in range(20):
        _, env = w.step("get_findings", {}, _ok_dispatch)
        if env.card is not None:
            cards_at.append(w.state.step_count)
    assert cards_at, "at least one periodic card must fire in 20 steps"
    assert any(15 <= s <= 20 for s in cards_at), \
        f"periodic card must appear in step 15-20 window; got {cards_at}"


def test_disabled_wrapper_is_no_op():
    """UTOV_METHODOLOGY=off path: enabled=False short-circuits to empty
    envelope (no footer text), and the underlying dispatch still runs."""
    cfg = MethodologyConfig(enabled=False)
    w = DisciplineWrapper(config=cfg)
    res, env = w.step("get_findings", {}, _ok_dispatch)
    assert res == {"ok": True, "method": "get_findings"}
    assert env.footer == ""
    assert env.card is None
    assert env.prompts == []
    assert env.alerts == []


def test_env_off_disables():
    cfg = MethodologyConfig.from_env({"UTOV_METHODOLOGY": "off"})
    assert cfg.enabled is False
    cfg = MethodologyConfig.from_env({"UTOV_METHODOLOGY": "0"})
    assert cfg.enabled is False
    cfg = MethodologyConfig.from_env({"UTOV_METHODOLOGY": "false"})
    assert cfg.enabled is False
    cfg = MethodologyConfig.from_env({})
    assert cfg.enabled is True


def test_env_interval_override():
    cfg = MethodologyConfig.from_env({"UTOV_METHODOLOGY_INTERVAL": "7"})
    assert cfg.periodic_interval == 7


def test_contradiction_in_result_triggers_prompt():
    """Acceptance #3. The result carries `invariants_failed` (M8 module
    output shape); wrapper raises the contradiction prompt."""
    def contradicting_dispatch(method, params):
        return {
            "ok": True,
            "invariants_failed": [{"name": "hook_valid_but_no_match",
                                   "message": "..."}],
        }
    w = DisciplineWrapper(config=MethodologyConfig())
    _, env = w.step("verify_plugin_findings", {}, contradicting_dispatch)
    assert any("矛盾" in p for p in env.prompts), \
        f"contradiction prompt must fire; got prompts={env.prompts}"


def test_high_number_success_triggers_evidence_class_prompt():
    """Acceptance-adjacent: pass_rate >= 0.99 attaches the
    evidence_class / pending_review prompt."""
    def dispatch(method, params):
        return {"checked": 100, "passed": 100, "pass_rate": 1.0}
    w = DisciplineWrapper(config=MethodologyConfig())
    _, env = w.step("verify_plugin_findings", {}, dispatch)
    assert any("evidence_class" in p for p in env.prompts)


def test_repeated_failures_trigger_checkpoint_prompt():
    """Acceptance #4: 3 same-type failures in a row → repeated_failures
    prompt fires."""
    w = DisciplineWrapper(config=MethodologyConfig(failure_streak_threshold=3))
    for _ in range(3):
        with pytest.raises(DisciplineRaise):
            w.step("verify_handler_binops", {}, _failing_dispatch)
    # next call (not a failure) — should still surface the streak prompt
    _, env = w.step("get_findings", {}, _ok_dispatch)
    assert any("checkpoint" in p.lower() or "Γ" in p
               for p in env.prompts), \
        f"streak prompt must fire after 3 same-type failures; got {env.prompts}"


def test_unledgered_payload_intercepted():
    """Acceptance #5: a promote_to_finding/inject_finding carrying a
    payload marked `ledger_status=experiment` is refused."""
    w = DisciplineWrapper(config=MethodologyConfig())
    res, env = w.step(
        "inject_finding",
        {
            "kind": "algo_signature", "subject": "x",
            "payload": {"ledger_status": "experiment", "value": 1},
            "reason": "trying",
        },
        _ok_dispatch,
    )
    assert env.intercepted is True
    assert res is None
    assert env.intercepted_reason and "未入账本" in env.intercepted_reason


def test_unledgered_payload_passes_with_allow_unpromoted_flag():
    """An explicit `--allow-unpromoted` token in `reason` lets it
    through (the user authorized the exception)."""
    w = DisciplineWrapper(config=MethodologyConfig())
    res, env = w.step(
        "inject_finding",
        {
            "kind": "algo_signature", "subject": "x",
            "payload": {"ledger_status": "experiment", "value": 1},
            "reason": "trying --allow-unpromoted",
        },
        _ok_dispatch,
    )
    assert env.intercepted is False
    assert res == {"ok": True, "method": "inject_finding"}


def test_verifier_bypass_alert_after_threshold():
    """Three or more override_verdict calls escalate to a verifier-
    bypass alert."""
    cfg = MethodologyConfig(bypass_alert_threshold=3)
    w = DisciplineWrapper(config=cfg)
    last_env = None
    for _ in range(3):
        _, last_env = w.step("override_verdict", {"hyp_id": 1,
                                                  "new_verdict": "fail",
                                                  "reason": "x"},
                             _ok_dispatch)
    assert last_env is not None
    assert any("绕过 verifier" in a for a in last_env.alerts), \
        f"bypass alert must fire on 3rd call; got {last_env.alerts}"


def test_forbidden_keyword_in_reason_intercepted():
    """Reason text matching a forbidden keyword refuses the call."""
    w = DisciplineWrapper(config=MethodologyConfig())
    _, env = w.step(
        "override_verdict",
        {"hyp_id": 1, "new_verdict": "fail", "reason": "凭印象判定"},
        _ok_dispatch,
    )
    assert env.intercepted is True
    assert "凭印象" in (env.intercepted_reason or "")


def test_rerun_from_stage_resets_failure_streak():
    """A backtrack via rerun_from_stage clears the failure streak."""
    w = DisciplineWrapper(config=MethodologyConfig())
    # accumulate two failures
    for _ in range(2):
        with pytest.raises(DisciplineRaise):
            w.step("verify_handler_binops", {}, _failing_dispatch)
    assert w.state.failures_since_checkpoint == 2
    # checkpoint via rerun_from_stage
    w.step("rerun_from_stage", {"stage": "s1", "reason": "test"}, _ok_dispatch)
    assert w.state.failures_since_checkpoint == 0
    assert w.state.backtrack_count == 1


def test_no_recent_checkpoint_alert_combines_with_failures():
    """Steps_since_checkpoint AND failures_since_checkpoint both must
    cross threshold for the no_recent_checkpoint alert."""
    cfg = MethodologyConfig(
        steps_since_checkpoint_warn=5,
        failure_streak_threshold=2,
        bypass_alert_threshold=999,    # disable bypass alert noise
    )
    w = DisciplineWrapper(config=cfg)
    for _ in range(2):
        with pytest.raises(DisciplineRaise):
            w.step("verify_handler_binops", {}, _failing_dispatch)
    # bring steps_since_checkpoint above warn threshold
    last_env = None
    for _ in range(5):
        _, last_env = w.step("get_findings", {}, _ok_dispatch)
    assert last_env is not None
    assert any("距上次盘整" in a for a in last_env.alerts)


# ---------------------------------------------------------------------------
# Acceptance #1+#2 combined: 20-step session simulation
# ---------------------------------------------------------------------------


def test_full_20_step_session_meets_acceptance_1_and_2():
    """End-to-end smoke test: 20 successful calls must each carry a
    footer (acceptance #1), and at least one full card must fire
    inside the 15-20 step window (acceptance #2)."""
    w = DisciplineWrapper(config=MethodologyConfig(periodic_interval=15))
    footers = 0
    cards = 0
    for _ in range(20):
        _, env = w.step("get_findings", {}, _ok_dispatch)
        if env.footer:
            footers += 1
        if env.card:
            cards += 1
    assert footers == 20, "every call must carry a footer"
    assert cards >= 1, "at least one card in 20 steps"


# ---------------------------------------------------------------------------
# Combined acceptance check — single test that asserts the spec's
# bulleted criteria in one place. The individual tests above narrow the
# failure mode.
# ---------------------------------------------------------------------------


def test_spec_acceptance_all_five_pass():
    """Aggregated 5-point check mirroring the user's acceptance list."""
    # 1. footer always present
    w1 = DisciplineWrapper(config=MethodologyConfig(periodic_interval=99))
    for _ in range(20):
        _, env = w1.step("get_findings", {}, _ok_dispatch)
        assert env.footer

    # 2. periodic card in 20-step window
    w2 = DisciplineWrapper(config=MethodologyConfig(periodic_interval=15))
    fired = False
    for _ in range(20):
        _, env = w2.step("get_findings", {}, _ok_dispatch)
        fired = fired or env.card is not None
    assert fired

    # 3. contradiction prompt fires
    w3 = DisciplineWrapper(config=MethodologyConfig())
    _, env3 = w3.step(
        "verify_plugin_findings", {},
        lambda m, p: {"ok": True, "invariants_failed": [{"name": "x", "message": "y"}]},
    )
    assert any("矛盾" in p for p in env3.prompts)

    # 4. checkpoint prompt fires after 3 same-type failures
    w4 = DisciplineWrapper(config=MethodologyConfig(failure_streak_threshold=3))
    for _ in range(3):
        with pytest.raises(DisciplineRaise):
            w4.step("verify_handler_binops", {}, _failing_dispatch)
    _, env4 = w4.step("get_findings", {}, _ok_dispatch)
    assert any("checkpoint" in p.lower() or "Γ" in p for p in env4.prompts)

    # 5. un-ledgered data ref intercepted
    w5 = DisciplineWrapper(config=MethodologyConfig())
    _, env5 = w5.step(
        "inject_finding",
        {"kind": "k", "subject": "s",
         "payload": {"ledger_status": "experiment"},
         "reason": "trying"},
        _ok_dispatch,
    )
    assert env5.intercepted is True
