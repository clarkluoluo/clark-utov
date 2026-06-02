"""M3 bypass-block auto-detector — acceptance tests.

Mirrors the reference target retro: SM3 candidate block probed under three
distinct observation methods, all reporting variability=0 across
distinct inputs. The framework gate should:

  1. Not fire on a single failed observation (could be obs bug).
  2. Flip the block to suspected_bypass when ≥ N (default 2) distinct
     observation methods all fail.
  3. Refuse follow-up observation attempts on the same block
     ("don't change posture on a dead block").
  4. Leave OTHER blocks unaffected.
  5. Be independently toggleable (UTOV_M3_BYPASS=off).
"""

from __future__ import annotations

import pytest

from engine.discipline_wrapper import DisciplineWrapper
from engine.m3_bypass_block import (
    BypassBlockDetector,
    M3BypassConfig,
    extract_attempt,
)
from engine.methodology import MethodologyConfig


# ---------------------------------------------------------------------------
# Module-level detector behaviour
# ---------------------------------------------------------------------------


def test_single_failed_observation_does_not_trigger():
    """Single-method failure must be treated as a potential obs bug,
    not a bypass — the criterion is *cross-method* invariance."""
    det = BypassBlockDetector(M3BypassConfig(min_failed_observations=2))
    out = det.record_attempt("block_sm3", "hook_pre", failed=True)
    assert out is None
    assert det.is_known_bypass("block_sm3") is False


def test_two_distinct_methods_failed_triggers_bypass():
    """The headline case: 2 distinct observation methods both report
    variability=0 → flip to suspected_bypass on the second call."""
    det = BypassBlockDetector(M3BypassConfig(min_failed_observations=2))
    assert det.record_attempt("block_sm3", "hook_pre",  failed=True) is None
    detection = det.record_attempt("block_sm3", "hook_post", failed=True)
    assert detection is not None
    assert detection.triggered is True
    assert detection.suspected_bypass is True
    assert set(detection.failed_methods) == {"hook_pre", "hook_post"}
    assert "upstream" in detection.recommendation or "parallel" in detection.recommendation
    assert det.is_known_bypass("block_sm3") is True


def test_three_methods_required_when_n_is_3():
    """Configurable threshold. The reference target actually had 3 failed obs;
    the spec says default N=2 with N>=2 configurable."""
    det = BypassBlockDetector(M3BypassConfig(min_failed_observations=3))
    assert det.record_attempt("blk", "a", failed=True) is None
    assert det.record_attempt("blk", "b", failed=True) is None
    out = det.record_attempt("blk", "c", failed=True)
    assert out is not None
    assert out.triggered is True


def test_same_method_repeated_does_not_inflate_count():
    """Repeated retries with the *same* observation method are one
    method, not many — otherwise an observation-bug retry loop would
    self-trigger the bypass flag."""
    det = BypassBlockDetector(M3BypassConfig(min_failed_observations=2))
    for _ in range(5):
        assert det.record_attempt("blk", "hook_pre", failed=True) is None
    assert det.is_known_bypass("blk") is False


def test_once_triggered_intercept_followup_returns_detection():
    det = BypassBlockDetector(M3BypassConfig(min_failed_observations=2))
    det.record_attempt("blk", "a", failed=True)
    det.record_attempt("blk", "b", failed=True)
    fu = det.intercept_followup("blk", "c")
    assert fu is not None
    assert fu.triggered is False             # not the original trigger
    assert fu.suspected_bypass is True
    assert "refused c" in (fu.intercepted_reason or "")


def test_other_blocks_unaffected():
    det = BypassBlockDetector(M3BypassConfig(min_failed_observations=2))
    det.record_attempt("blk_A", "a", failed=True)
    det.record_attempt("blk_A", "b", failed=True)
    assert det.is_known_bypass("blk_A") is True
    assert det.is_known_bypass("blk_B") is False
    assert det.intercept_followup("blk_B", "x") is None


def test_passing_observation_does_not_trigger():
    det = BypassBlockDetector(M3BypassConfig(min_failed_observations=2))
    det.record_attempt("blk", "a", failed=False)
    det.record_attempt("blk", "b", failed=False)
    assert det.is_known_bypass("blk") is False


def test_mixed_pass_then_fail_one_pass_does_not_count_as_failed_method():
    """A method that PASSED variability is not counted in the failed
    set, even if other methods failed."""
    det = BypassBlockDetector(M3BypassConfig(min_failed_observations=2))
    det.record_attempt("blk", "a", failed=True)
    det.record_attempt("blk", "b", failed=False)   # b passes
    assert det.is_known_bypass("blk") is False
    det.record_attempt("blk", "c", failed=True)
    # Now {a, c} are failed → threshold hit.
    assert det.is_known_bypass("blk") is True


def test_env_toggle_off_disables_module():
    cfg = M3BypassConfig.from_env({"UTOV_M3_BYPASS": "off"})
    assert cfg.enabled is False
    det = BypassBlockDetector(cfg)
    out = det.record_attempt("blk", "a", failed=True)
    out2 = det.record_attempt("blk", "b", failed=True)
    assert out is None and out2 is None
    assert det.is_known_bypass("blk") is False


def test_env_configurable_n():
    cfg = M3BypassConfig.from_env({"UTOV_M3_BYPASS_N": "3"})
    assert cfg.min_failed_observations == 3


# ---------------------------------------------------------------------------
# Input extraction
# ---------------------------------------------------------------------------


def test_extract_attempt_reads_explicit_failed():
    cfg = M3BypassConfig()
    out = extract_attempt(
        "verify_block_variability",
        {"block_id": "blk", "observation_method": "m", "failed": True},
        None, cfg=cfg,
    )
    assert out == ("blk", "m", True)


def test_extract_attempt_reads_unique_count_below_floor():
    cfg = M3BypassConfig(min_unique_for_pass=2)
    out = extract_attempt(
        "verify_block_variability",
        {"block_id": "blk", "observation_method": "m"},
        {"unique_count": 1},
        cfg=cfg,
    )
    assert out == ("blk", "m", True)


def test_extract_attempt_reads_eq_sign_zero():
    cfg = M3BypassConfig()
    out = extract_attempt(
        "verify_block_variability",
        {"block_id": "blk", "observation_method": "m"},
        {"hook_digest_eq_sign": 0.0},
        cfg=cfg,
    )
    assert out == ("blk", "m", True)


def test_extract_attempt_returns_none_for_non_m3_method():
    cfg = M3BypassConfig()
    out = extract_attempt(
        "promote_to_finding",
        {"block_id": "blk", "observation_method": "m", "failed": True},
        None, cfg=cfg,
    )
    assert out is None


def test_extract_attempt_returns_none_without_block_id():
    cfg = M3BypassConfig()
    out = extract_attempt(
        "verify_block_variability",
        {"observation_method": "m", "failed": True},
        None, cfg=cfg,
    )
    assert out is None


# ---------------------------------------------------------------------------
# DisciplineWrapper integration — the canonical reference target scenario
# ---------------------------------------------------------------------------


def _wrapper_for_tests() -> DisciplineWrapper:
    return DisciplineWrapper(
        config=MethodologyConfig(),
        m3_bypass_config=M3BypassConfig(min_failed_observations=2),
    )


def test_wrapper_triggers_bypass_on_second_distinct_failed_observation():
    """First M3 fail: nothing fires. Second distinct method also
    fails: wrapper flips the block and attaches the m3_bypass block
    to the envelope. Original dispatch still ran (we don't
    retroactively refuse the call that crossed the threshold; the
    detection is the *output* of that call)."""
    wrapper = _wrapper_for_tests()

    def dispatch(method, params):
        return {"unique_count": 1}    # variability=0

    _, env1 = wrapper.step(
        "verify_block_variability",
        {"block_id": "blk_sm3", "observation_method": "hook_pre"},
        dispatch,
    )
    assert env1.m3_bypass is None        # one failure, no trigger

    _, env2 = wrapper.step(
        "verify_block_variability",
        {"block_id": "blk_sm3", "observation_method": "hook_post"},
        dispatch,
    )
    assert env2.m3_bypass is not None
    assert env2.m3_bypass["triggered"] is True
    assert env2.m3_bypass["suspected_bypass"] is True
    assert set(env2.m3_bypass["failed_methods"]) == {"hook_pre", "hook_post"}
    assert any("M3-BYPASS/TRIGGERED" in a for a in env2.alerts)


def test_wrapper_intercepts_followup_on_confirmed_bypass_block():
    """Once a block is flipped to suspected_bypass, ANY further
    observation attempt on it (even with a brand-new method name) is
    refused before dispatch — the "stop changing posture" rule."""
    wrapper = _wrapper_for_tests()
    dispatched: list[str] = []

    def dispatch(method, params):
        dispatched.append((params or {}).get("observation_method", "?"))
        return {"unique_count": 1}

    # Two distinct failures → trigger.
    wrapper.step("verify_block_variability",
                 {"block_id": "blk", "observation_method": "a"}, dispatch)
    wrapper.step("verify_block_variability",
                 {"block_id": "blk", "observation_method": "b"}, dispatch)
    assert dispatched == ["a", "b"]

    # Third attempt with a NEW method must be refused — never reaches dispatch.
    result, env = wrapper.step(
        "verify_block_variability",
        {"block_id": "blk", "observation_method": "c"},
        dispatch,
    )
    assert result is None
    assert dispatched == ["a", "b"], "intercepted attempt must NOT dispatch"
    assert env.intercepted is True
    assert env.m3_bypass is not None
    assert env.m3_bypass["triggered"] is False  # follow-up, not the trigger
    assert any("M3-BYPASS" in a for a in env.alerts)


def test_wrapper_intercepts_followup_under_any_observation_method_name():
    """The intercept-method whitelist (hook installation, block
    tracing, dumping…) all count once the block is dead. A switch
    from 'hook_pre' to 'install_hook' is still refused."""
    wrapper = _wrapper_for_tests()

    def dispatch(method, params):
        return {"unique_count": 1}

    wrapper.step("verify_block_variability",
                 {"block_id": "blk", "observation_method": "a"}, dispatch)
    wrapper.step("verify_block_variability",
                 {"block_id": "blk", "observation_method": "b"}, dispatch)

    # Now try install_hook on the same block.
    result, env = wrapper.step(
        "install_hook",
        {"block_id": "blk", "observation_method": "alt_capture"},
        lambda m, p: pytest.fail("dispatch should not run"),
    )
    assert result is None
    assert env.intercepted is True


def test_wrapper_does_not_block_other_blocks():
    wrapper = _wrapper_for_tests()
    wrapper.step("verify_block_variability",
                 {"block_id": "blk_A", "observation_method": "a"},
                 lambda m, p: {"unique_count": 1})
    wrapper.step("verify_block_variability",
                 {"block_id": "blk_A", "observation_method": "b"},
                 lambda m, p: {"unique_count": 1})
    # blk_A is now suspected_bypass; blk_B must remain free.
    seen = []
    result, env = wrapper.step(
        "verify_block_variability",
        {"block_id": "blk_B", "observation_method": "a"},
        lambda m, p: (seen.append(1) or {"unique_count": 5}),
    )
    assert result == {"unique_count": 5}
    assert env.intercepted is False
    assert seen == [1]


def test_wrapper_disabled_module_does_not_intercept():
    wrapper = DisciplineWrapper(
        config=MethodologyConfig(),
        m3_bypass_config=M3BypassConfig(enabled=False),
    )
    for obs in ("a", "b", "c"):
        result, env = wrapper.step(
            "verify_block_variability",
            {"block_id": "blk", "observation_method": obs},
            lambda m, p: {"unique_count": 1},
        )
        assert result == {"unique_count": 1}
        assert env.intercepted is False
        assert env.m3_bypass is None
