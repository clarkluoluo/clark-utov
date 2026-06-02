"""Probe runtime — interface, context, and builtin registry (PLAN §19.2).

The kernel never imports specific probes directly. Instead:

  * A probe is a subclass of :class:`Probe` registered under its
    declarative name (via :func:`register_builtin_probe` for in-engine
    probes, or via the ``module:`` field on the profile entry for
    user-provided ones).
  * The profile registry returns a :class:`MergedProfile` with the set
    of probe names; runtime asks ``get_builtin_probe_class(name)``
    or imports ``module`` to materialise the class.
  * The probe's ``run`` method receives a :class:`ProbeContext` and
    returns a :class:`Verdict`. Mechanism probes that need to reach
    domain states use ``ctx.state_for_role("<role>")`` — never compare
    state literals directly (lint enforces this, §19.7 #1c).

Step 2 lands the interface plus the first mechanism probe (M1).
``affects_state`` / ``affects_scope`` fields are reserved for step 5
when the state machine + scope runtime arrive.
"""

from __future__ import annotations

import abc
import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.profile.types import ProbeSpec


ProbeResult = Literal["pass", "fail", "undetermined"]


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceClassCap:
    """A verdict may cap the node's evidence class.

    The conjunctive gate composes caps by taking the **lowest** (per
    §19.3) — strength order comes from the active profile's
    ``evidence_classes`` tuple.
    """

    class_id: str
    reason: str = ""


@dataclass(frozen=True)
class Verdict:
    """Result of running one probe."""

    probe: str
    result: ProbeResult
    evidence: dict[str, Any] = field(default_factory=dict)
    affects_evidence_class: Optional[EvidenceClassCap] = None


@dataclass(frozen=True)
class StateView:
    """Read-only view of a domain state binding handed to mechanism code."""

    name: str
    roles: tuple[str, ...]


# ---------------------------------------------------------------------------
# Probe context
# ---------------------------------------------------------------------------


class ProbeContext:
    """Per-call context for one probe run.

    Mechanism probes must access domain states through
    :meth:`state_for_role` only — looking up by literal state name
    breaks the role indirection (§19.1) and is forbidden by lint.

    ``profile`` is the active :class:`MergedProfile` (passed as
    ``Any`` here to avoid an import cycle with ``registry``). When
    present, probes that synthesise evidence-class caps consult its
    ``evidence_classes`` ordering — that's how the same probe code
    produces the right cap for ``vmp_algorithm_extraction`` (A/B/C)
    vs a future profile that uses ``S > A > B > C``. When absent,
    probes fall back to the step-4 alphabetic heuristic so isolated
    probe tests remain valid.
    """

    def __init__(
        self,
        *,
        method: str = "",
        params: Optional[dict[str, Any]] = None,
        result: Any = None,
        state_bindings: Optional[dict[str, StateView]] = None,
        extras: Optional[dict[str, Any]] = None,
        profile: Any = None,
    ) -> None:
        self.method = method
        self.params = params or {}
        self.result = result
        self._state_by_role: dict[str, StateView] = state_bindings or {}
        self.extras = extras or {}
        self.profile = profile

    def state_for_role(self, role: str) -> Optional[StateView]:
        """Return the state bound to ``role`` in the active profile, or None."""
        return self._state_by_role.get(role)


# ---------------------------------------------------------------------------
# Probe interface
# ---------------------------------------------------------------------------


class Probe(abc.ABC):
    """Abstract base for all probes — both mechanism and domain.

    The ``mechanism`` class attribute is the **runtime** truth used by
    ``gate_runtime`` to force-include the probe at conjunctive-gate
    evaluation time (Lock B, §19.7 #7). It is intentionally a class
    attribute on the implementation, not a profile-data field — that
    way a tampered :class:`MergedProfile` cannot silently demote a
    mechanism probe by editing the profile's probe list. The
    base.json ``mechanism: true`` flag still drives Lock A and the
    mechanism-name index; the two must stay consistent (a lint check
    verifies this in a later step).
    """

    name: str = ""
    mechanism: bool = False
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()

    @abc.abstractmethod
    def run(self, ctx: ProbeContext) -> Verdict:
        ...


# ---------------------------------------------------------------------------
# Builtin probe registry
# ---------------------------------------------------------------------------


_BUILTIN_PROBES: dict[str, type[Probe]] = {}
_BUILTINS_LOADED = False


def register_builtin_probe(name: str) -> Callable[[type[Probe]], type[Probe]]:
    """Decorator: register a :class:`Probe` subclass under ``name``.

    Used by in-engine probes (base mechanism + first-party domain
    probes). User-provided probes register via the profile entry's
    ``module:`` field instead — they don't need this decorator.
    """

    def decorator(cls: type[Probe]) -> type[Probe]:
        existing = _BUILTIN_PROBES.get(name)
        if existing is not None and existing is not cls:
            raise RuntimeError(
                f"builtin probe '{name}' already registered as "
                f"{existing.__module__}.{existing.__name__}"
            )
        if not cls.name:
            cls.name = name
        _BUILTIN_PROBES[name] = cls
        return cls

    return decorator


def get_builtin_probe_class(name: str) -> Optional[type[Probe]]:
    """Look up a registered builtin probe class by name."""
    _ensure_builtins_loaded()
    return _BUILTIN_PROBES.get(name)


def list_builtin_probes() -> dict[str, type[Probe]]:
    """Snapshot of all currently registered builtin probes."""
    _ensure_builtins_loaded()
    return dict(_BUILTIN_PROBES)


def _ensure_builtins_loaded() -> None:
    """Import all base mechanism + first-party probe modules so their
    decorator-side effects fire. Idempotent."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    # Side-effect import — each module registers via @register_builtin_probe.
    from engine.profile import probes  # noqa: F401


# ---------------------------------------------------------------------------
# Dynamic probe resolution — §19.7 #6
# ---------------------------------------------------------------------------


class ProbeResolveError(Exception):
    """Raised when a probe spec carries a ``module:`` path that cannot
    be imported, the named class cannot be located, or the resolved
    object is not a Probe subclass."""


def resolve_probe_class(spec: "ProbeSpec") -> Optional[type[Probe]]:
    """Resolve a :class:`ProbeSpec` to its concrete :class:`Probe` class.

    Two paths:

      * **builtin** — ``spec.module is None``. Look up ``spec.name`` in
        the import-time decorator registry. This is the path mechanism
        probes (M1 / M3 / …) and first-party domain probes
        (length_chain_check) take.

      * **external** — ``spec.module`` is set. The string is either
        ``"pkg.mod:ClassName"`` (explicit class) or ``"pkg.mod"``
        (looks up an attribute matching ``spec.name`` in
        ``snake_case`` form converted to ``PascalCase + "Probe"``,
        with a fallback to the literal ``spec.name``). This is the
        path users take when supplying their own probes via profile
        — see §19.7 #6.

    Returns ``None`` only for builtins that aren't registered (typo
    in the profile, or the probe module hasn't been imported and
    isn't reachable through the decorator chain). External imports
    that fail raise :class:`ProbeResolveError` — silent ``None``
    would hide configuration errors.
    """
    if spec.module:
        return _resolve_external(spec)
    return get_builtin_probe_class(spec.name)


def _resolve_external(spec: "ProbeSpec") -> type[Probe]:
    module_path = spec.module or ""
    if ":" in module_path:
        mod_name, _, class_name = module_path.partition(":")
    else:
        mod_name = module_path
        class_name = _snake_to_class(spec.name)

    try:
        module = importlib.import_module(mod_name)
    except ImportError as exc:
        raise ProbeResolveError(
            f"probe '{spec.name}': cannot import module '{mod_name}': {exc}"
        ) from exc

    cls = getattr(module, class_name, None)
    if cls is None and ":" not in module_path:
        # Fallback: try the literal probe name verbatim.
        cls = getattr(module, spec.name, None)

    if cls is None:
        raise ProbeResolveError(
            f"probe '{spec.name}': module '{mod_name}' has no class "
            f"'{class_name}' (and no attribute named '{spec.name}')"
        )
    if not (isinstance(cls, type) and issubclass(cls, Probe)):
        raise ProbeResolveError(
            f"probe '{spec.name}': '{mod_name}.{class_name}' is not a "
            f"Probe subclass"
        )
    return cls


def _snake_to_class(name: str) -> str:
    """``key_entropy_check`` → ``KeyEntropyCheckProbe``.

    Convention for external probes that don't pin a ``:ClassName`` in
    the profile: the class is named ``<PascalCase(probe_name)>Probe``.
    Probe authors that want a different class name pin it explicitly
    via ``module: pkg.mod:OtherClassName``.
    """
    parts = [p for p in name.split("_") if p]
    return "".join(p.capitalize() for p in parts) + "Probe"
