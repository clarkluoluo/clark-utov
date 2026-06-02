"""DisciplineWrapper profile wire-in (PLAN §19 / IMPL_PLAN §P1.0 step 8).

When the wrapper is constructed with a :class:`MergedProfile`, every
envelope advertises ``{name, chain}`` so agents / loggers know which
judgement-semantics profile the engine is running under. When the
wrapper is constructed without one (legacy callers), the envelope
shape is identical to v0.2.0-dev — that's how the wire-in stays
backwards-compatible.

Step 8's job is to make the wire reachable. The wrapper's existing
M1/M3/etc. dispatch logic is NOT migrated to consult the profile;
that's a future step. These tests verify only the metadata
plumbing — they don't assert on dispatch behaviour, since dispatch
is unchanged.
"""

from __future__ import annotations

import pytest

from engine.discipline_wrapper import DisciplineEnvelope, DisciplineWrapper
from engine.methodology import MethodologyConfig
from engine.profile import ProfileRegistry


@pytest.fixture()
def vmp_profile():
    return ProfileRegistry().load_chain("vmp_algorithm_extraction")


# ---------------------------------------------------------------------------
# Constructor signature
# ---------------------------------------------------------------------------


def test_wrapper_init_accepts_profile_keyword():
    import inspect

    sig = inspect.signature(DisciplineWrapper.__init__)
    param = sig.parameters.get("profile")
    assert param is not None
    assert param.default is None
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_wrapper_without_profile_stores_none():
    wrapper = DisciplineWrapper(config=MethodologyConfig())
    assert wrapper.profile is None


def test_wrapper_with_profile_stores_it(vmp_profile):
    wrapper = DisciplineWrapper(config=MethodologyConfig(), profile=vmp_profile)
    assert wrapper.profile is vmp_profile


# ---------------------------------------------------------------------------
# Envelope shape — profile sibling field
# ---------------------------------------------------------------------------


def test_envelope_default_omits_profile_key():
    """A wrapper that wasn't given a profile produces an envelope
    whose ``to_dict()`` has no ``profile`` key (preserves the
    v0.2.0-dev envelope shape verbatim)."""
    env = DisciplineEnvelope(footer="x")
    out = env.to_dict()
    assert "profile" not in out


def test_envelope_with_profile_round_trips_name_and_chain():
    env = DisciplineEnvelope(
        footer="x",
        profile={"name": "vmp_algorithm_extraction", "chain": ["base", "vmp_algorithm_extraction"]},
    )
    out = env.to_dict()
    assert out["profile"]["name"] == "vmp_algorithm_extraction"
    assert out["profile"]["chain"] == ["base", "vmp_algorithm_extraction"]


# ---------------------------------------------------------------------------
# End-to-end through step()
# ---------------------------------------------------------------------------


def _noop_dispatch(method: str, params: dict) -> dict:
    return {"ok": True}


def test_wrapper_step_adds_profile_to_envelope_when_set(vmp_profile):
    """When the wrapper has a profile, every envelope through step()
    advertises ``{name, chain}``."""
    wrapper = DisciplineWrapper(config=MethodologyConfig(), profile=vmp_profile)
    _result, env = wrapper.step("get_hyp_tree", {"depth": 3}, _noop_dispatch)
    out = env.to_dict()
    assert out["profile"]["name"] == "vmp_algorithm_extraction"
    assert out["profile"]["chain"] == ["base", "vmp_algorithm_extraction"]


def test_wrapper_step_omits_profile_when_not_set():
    """Legacy callers — no profile passed — get the v0.2.0-dev envelope
    shape with no ``profile`` key. This is the backward-compat
    guarantee that keeps the 539 existing tests green."""
    wrapper = DisciplineWrapper(config=MethodologyConfig())
    _result, env = wrapper.step("get_hyp_tree", {"depth": 3}, _noop_dispatch)
    out = env.to_dict()
    assert "profile" not in out


def test_wrapper_step_advertises_inherited_chain(vmp_profile):
    """A subprofile chain advertises its full inheritance — agents
    can branch on either ``name`` or ``chain``."""
    weird = ProfileRegistry().load_chain("weird_target_x")
    wrapper = DisciplineWrapper(config=MethodologyConfig(), profile=weird)
    _result, env = wrapper.step("get_hyp_tree", {}, _noop_dispatch)
    out = env.to_dict()
    assert out["profile"]["name"] == "weird_target_x"
    assert out["profile"]["chain"] == [
        "base", "vmp_algorithm_extraction", "weird_target_x"
    ]
