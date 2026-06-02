"""External probe fixtures for the ``module:``-import path
(acceptance §19.7 #6).

This module simulates a user-supplied probe package. The profile
layer's :func:`resolve_probe_class` loads probes from here via the
``module:`` field on a ``ProbeSpec`` — proving that adding a new
probe requires zero engine source change.

Two stubs:

  * :class:`KeyEntropyCheckProbe` — discovered via the snake-case →
    PascalCase + ``Probe`` convention (``key_entropy_check`` →
    ``KeyEntropyCheckProbe``).
  * :class:`HardlyExisting` — discovered only when the profile pins
    the explicit class name via ``module: …:HardlyExisting``.

Both are domain probes (``mechanism = False``) — only base may
declare mechanism, and a fixture file proving otherwise would itself
be a §19.7 #8 lint violation.
"""

from __future__ import annotations

from engine.profile.probe_runtime import Probe, ProbeContext, Verdict


class KeyEntropyCheckProbe(Probe):
    """Stub: pretends to measure key entropy. Returns ``pass`` when
    the params carry a ``key_bytes`` field, ``undetermined`` otherwise."""

    name = "key_entropy_check"
    mechanism = False
    inputs = ("params",)
    outputs = ("key_entropy",)

    def run(self, ctx: ProbeContext) -> Verdict:
        key_bytes = ctx.params.get("key_bytes")
        if not isinstance(key_bytes, str):
            return Verdict(probe=self.name, result="undetermined")
        return Verdict(
            probe=self.name,
            result="pass",
            evidence={"length_hex_chars": len(key_bytes)},
        )


class HardlyExisting(Probe):
    """Stub with a deliberately non-canonical class name — proves the
    ``module: pkg.mod:ClassName`` explicit-class-pin form works."""

    name = "explicit_class_probe"
    mechanism = False

    def run(self, ctx: ProbeContext) -> Verdict:
        return Verdict(probe=self.name, result="pass")
