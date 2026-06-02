"""Declarative ``related_helpers`` metadata — one source of truth, two consumers.

Lowers the "the feature exists but only insiders know it" cost (spec C): key
entry points (CLI commands / Python API functions) carry a *machine-queryable*
list of next-layer helpers you might want after them. Insiders know
``trace_provenance ↔ run_recapture_loop`` and ``import_map ↔ extern model``;
this map makes those links discoverable.

Design is **general-first**: the listed relations are seeds, the mechanism is
the deliverable. Adding a relation = one entry in ``RELATED_HELPERS``. Both the
CLI (``--verbose`` printer) and the API (``related_helpers(name)`` lookup) read
this same map, so there is no duplication.

Surfacing policy (spec A8 §3/§4):
  * Silent by default — only ``--verbose``/debug surfaces it.
  * An entry point with no relations ⇒ nothing printed (never a fake "no
    related helpers" claim).
  * A relation naming a symbol that exists nowhere in the engine is a
    *lint-catchable* error (:func:`lint_related_helpers`), never silently shown.

Target strings are human-readable display names. The last dotted component is
the actual symbol; an optional ``module.`` prefix is just a hint for readers.
The linter resolves a target by scanning the ``engine`` package for that
symbol, so adding a relation needs no import-path bookkeeping.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# THE relation map — single source of truth. Adding a relation = one entry.
# --------------------------------------------------------------------------
RELATED_HELPERS: dict[str, list[str]] = {
    "trace_provenance":     ["run_recapture_loop", "suggest_observations"],
    "check_parity_vectors": ["real_gold.collect_real_gold", "check_seed_independence"],
    "import_map":           ["extern_model.resolve_extern_model",
                             "libc_boundary.synthesize_boundary_edge"],
    "run_recovery":         ["check_emit_self_consistency", "derive_mem_sink_interval"],
}

# Targets that are intentionally planned-but-not-yet-landed. The linter treats
# these as known-OK so a wave's map passes before the owning modules exist.
# Remove a name from here once its symbol lands (then the linter enforces it for
# real). Keyed by the *symbol* (last dotted component). Currently empty: this
# wave's planned targets — `resolve_extern_model` (#1 extern_model) and
# `suggest_observations` (#4 observation_planner) — have both landed, so the
# linter now enforces them for real.
PLANNED_TARGETS: frozenset[str] = frozenset()


def related_helpers(name: str) -> list[str]:
    """Return the related-helper display names for an entry point.

    Unmapped entry point ⇒ empty list (the caller surfaces nothing — never a
    fake "no related helpers" line). The returned list is a fresh copy, safe
    for the caller to mutate.
    """
    return list(RELATED_HELPERS.get(name, ()))


def _symbol_of(target: str) -> str:
    """Last dotted component of a target display string (the real symbol)."""
    return target.rsplit(".", 1)[-1]


def _engine_symbols() -> set[str]:
    """All public symbol names defined/re-exported anywhere under ``engine``.

    Walks the ``engine`` package once; modules that fail to import (optional
    deps, other in-flight waves) are skipped — a target only needs to resolve
    *somewhere*. Used by the linter to catch bad links generically, without
    per-target import paths.
    """
    import importlib
    import pkgutil

    import engine as _engine_pkg

    names: set[str] = set()
    # The top-level package itself.
    names.update(n for n in dir(_engine_pkg) if not n.startswith("_"))
    for modinfo in pkgutil.walk_packages(_engine_pkg.__path__, "engine."):
        try:
            mod = importlib.import_module(modinfo.name)
        except Exception:
            continue
        names.update(n for n in dir(mod) if not n.startswith("_"))
    return names


def lint_related_helpers() -> list[str]:
    """Return a list of error strings for relations naming unknown symbols.

    A target resolves if its symbol is defined/re-exported anywhere under the
    ``engine`` package, OR it is in :data:`PLANNED_TARGETS`. Anything else is a
    bad link. Empty list ⇒ map is clean.
    """
    known = _engine_symbols()
    errors: list[str] = []
    for entry, targets in RELATED_HELPERS.items():
        for target in targets:
            sym = _symbol_of(target)
            if sym in PLANNED_TARGETS or sym in known:
                continue
            errors.append(
                f"related_helpers[{entry!r}] -> {target!r}: symbol {sym!r} is "
                f"not importable anywhere under 'engine' and is not in "
                f"PLANNED_TARGETS (bad link)."
            )
    return errors


def format_related_line(name: str) -> str | None:
    """Render the ``ℹ related: X, Y, Z`` line for an entry point, or ``None``.

    Returns ``None`` when the entry point has no relations, so callers print
    nothing in that case (degenerate ⇒ surfaced as silence, not a fake line).
    """
    targets = related_helpers(name)
    if not targets:
        return None
    return "ℹ related: " + ", ".join(targets)
