"""resolve_probe_class — external probe loading via ``module:`` (§19.7 #6).

The profile layer's promise: a user adds a probe by writing a Python
file in their own package and pointing the profile at it. No engine
source change required. This test file exercises that loop end-to-end
against a stub probe living in ``tests/profile/fixtures/``.
"""

from __future__ import annotations

import pytest

from engine.profile.probe_runtime import (
    Probe,
    ProbeContext,
    ProbeResolveError,
    get_builtin_probe_class,
    resolve_probe_class,
)
from engine.profile.types import ProbeSpec


FIXTURE_MODULE = "tests.profile.fixtures.stub_external_probes"


# ---------------------------------------------------------------------------
# Builtin path — when module: is None, fall back to the decorator registry
# ---------------------------------------------------------------------------


def test_resolve_returns_builtin_when_module_is_none():
    """No module → look up by name in builtin registry. This is the
    path mechanism probes (M1 etc.) take."""
    spec = ProbeSpec(name="m1_success_audit")
    cls = resolve_probe_class(spec)
    assert cls is not None
    assert cls is get_builtin_probe_class("m1_success_audit")


def test_resolve_returns_none_for_unknown_builtin():
    """No module + unknown name → None (a profile typo).  Loader
    catches obvious typos earlier; this is the "nothing to import"
    case."""
    spec = ProbeSpec(name="absolutely_not_a_real_probe")
    assert resolve_probe_class(spec) is None


# ---------------------------------------------------------------------------
# External path — module: "pkg.mod" (no class pin)
# ---------------------------------------------------------------------------


def test_resolve_loads_external_probe_via_snake_to_pascal_convention():
    """``key_entropy_check`` + ``module: "pkg.mod"`` (no class pin) →
    look up ``KeyEntropyCheckProbe`` (snake_case → PascalCase +
    Probe). This is the convenience form for typical probe authors."""
    spec = ProbeSpec(name="key_entropy_check", module=FIXTURE_MODULE)
    cls = resolve_probe_class(spec)
    assert cls is not None
    assert issubclass(cls, Probe)
    assert cls.__name__ == "KeyEntropyCheckProbe"


def test_external_probe_can_be_instantiated_and_run():
    """End-to-end: resolve → instantiate → run against a real
    ProbeContext. The engine never sees the probe's source — it just
    loaded a class via importlib."""
    spec = ProbeSpec(name="key_entropy_check", module=FIXTURE_MODULE)
    cls = resolve_probe_class(spec)
    assert cls is not None
    probe = cls()
    verdict = probe.run(ProbeContext(params={"key_bytes": "deadbeef"}))
    assert verdict.probe == "key_entropy_check"
    assert verdict.result == "pass"
    assert verdict.evidence["length_hex_chars"] == 8


def test_external_probe_with_no_params_returns_undetermined():
    spec = ProbeSpec(name="key_entropy_check", module=FIXTURE_MODULE)
    cls = resolve_probe_class(spec)
    probe = cls()
    verdict = probe.run(ProbeContext(params={}))
    assert verdict.result == "undetermined"


# ---------------------------------------------------------------------------
# External path — module: "pkg.mod:ClassName" (explicit class pin)
# ---------------------------------------------------------------------------


def test_resolve_loads_explicit_class_via_colon_form():
    """``module: pkg.mod:HardlyExisting`` lets the profile pin a class
    that doesn't match the snake_case-to-PascalCase convention. Useful
    when the class name is dictated by an external library."""
    spec = ProbeSpec(
        name="explicit_class_probe",
        module=f"{FIXTURE_MODULE}:HardlyExisting",
    )
    cls = resolve_probe_class(spec)
    assert cls is not None
    assert cls.__name__ == "HardlyExisting"


# ---------------------------------------------------------------------------
# Error paths — bad module / missing class / not a Probe subclass
# ---------------------------------------------------------------------------


def test_resolve_raises_on_unimportable_module():
    """A nonexistent module should fail loudly — silent None would
    hide a typo in the profile."""
    spec = ProbeSpec(name="x", module="absolutely.not.a.real.module")
    with pytest.raises(ProbeResolveError, match="cannot import"):
        resolve_probe_class(spec)


def test_resolve_raises_when_class_not_found():
    """Module loads OK, but neither the PascalCase-convention class
    nor the literal name attribute exists in it."""
    spec = ProbeSpec(name="never_defined", module=FIXTURE_MODULE)
    with pytest.raises(ProbeResolveError, match="no class"):
        resolve_probe_class(spec)


def test_resolve_raises_when_explicit_class_missing():
    spec = ProbeSpec(name="x", module=f"{FIXTURE_MODULE}:DoesNotExist")
    with pytest.raises(ProbeResolveError, match="no class"):
        resolve_probe_class(spec)


def test_resolve_raises_when_target_not_probe_subclass():
    """Module path resolves to something that isn't a Probe class.
    Importing succeeds; the type check catches it."""
    # Use a stdlib helper attribute that exists but isn't a Probe.
    spec = ProbeSpec(name="x", module="json:JSONDecoder")
    with pytest.raises(ProbeResolveError, match="not a Probe subclass"):
        resolve_probe_class(spec)
