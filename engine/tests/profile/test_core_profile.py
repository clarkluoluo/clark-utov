"""Core.profile wire-in (PLAN §19 / IMPL_PLAN §P1.0 step 8).

Acceptance: ``Core(profile_name=...)`` loads the profile lazily via
:class:`ProfileRegistry`, and ``Core.profile`` returns the resolved
:class:`MergedProfile`. Default (``profile_name=None``) keeps Core
profile-agnostic — that's how the v0.3.0 wire-in stays backward
compatible with the ~500 existing tests that construct Core without
a profile.

These tests construct Core via a lightweight stub so they don't drag
in the full conformance / trace-loading pipeline. The wire-in itself
is just a property — once we know it loads under controlled
conditions, real callers picking it up at runtime is straightforward.
"""

from __future__ import annotations

import pytest

from engine.profile import MergedProfile


# ---------------------------------------------------------------------------
# Direct property tests (no actual Core construction needed)
# ---------------------------------------------------------------------------
#
# Core's __init__ does a lot besides accepting profile_name — it
# materialises a trace, runs conformance, etc. Driving it through
# fixtures here would pull in TraceReader / RunnerAdapter mocks for
# every test. Cheaper: import the Core class, exercise the property's
# code path against a synthetic instance.


def _bare_core_with_profile_name(profile_name: str | None):
    """Build a Core-shaped object that only carries the two fields
    Core.profile reads. This is enough to test the property in
    isolation."""
    from engine.core import Core

    inst = Core.__new__(Core)
    inst._profile_name = profile_name
    inst._profile = None
    return inst


def test_core_with_no_profile_name_returns_none():
    """The default — Core didn't get a profile_name. The property is
    inert and the registry never loads."""
    inst = _bare_core_with_profile_name(None)
    assert inst.profile is None


def test_core_with_profile_name_loads_via_registry():
    inst = _bare_core_with_profile_name("vmp_algorithm_extraction")
    merged = inst.profile
    assert isinstance(merged, MergedProfile)
    assert merged.name == "vmp_algorithm_extraction"
    assert "base" in merged.chain


def test_core_profile_property_is_cached():
    """Second access returns the same instance — the registry is
    consulted once, not on every property read."""
    inst = _bare_core_with_profile_name("vmp_algorithm_extraction")
    first = inst.profile
    second = inst.profile
    assert first is second


def test_core_profile_resolves_inherited_chains():
    """When ``profile_name`` points to a subprofile, the property
    returns the resolved merge — ``chain`` reflects the inheritance."""
    inst = _bare_core_with_profile_name("weird_target_x")
    merged = inst.profile
    assert merged.chain == ("base", "vmp_algorithm_extraction", "weird_target_x")


def test_core_with_unknown_profile_name_raises_on_access():
    """A typo in profile_name → loud failure on first access, not at
    construction time (lazy). That keeps Core() cheap when the caller
    never reads .profile."""
    from engine.profile import ProfileLoadError

    inst = _bare_core_with_profile_name("definitely_not_a_real_profile")
    with pytest.raises(ProfileLoadError):
        _ = inst.profile


# ---------------------------------------------------------------------------
# Constructor signature exposes the new keyword without breaking callers
# ---------------------------------------------------------------------------


def test_core_init_signature_accepts_profile_name():
    """The new kwarg has a default — callers that didn't migrate keep
    working unchanged. (This is what the 500+ existing tests rely on.)"""
    import inspect

    from engine.core import Core

    sig = inspect.signature(Core.__init__)
    param = sig.parameters.get("profile_name")
    assert param is not None
    assert param.default is None
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
