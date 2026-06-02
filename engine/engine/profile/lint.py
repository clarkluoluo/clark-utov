"""Profile-layer lint (PLAN §19.7 #1 + #8).

Three sources of structural drift, each with its own check:

1. **base.json non-mechanism entry** — anything in base.json that is
   not ``mechanism: true`` is a category error; the base profile is
   the mechanism baseline, semantic content belongs in domain.
2. **domain profile claiming mechanism** — ``mechanism: true`` outside
   the base profile is a category error; it would let downstream
   profiles silently elevate arbitrary probes to "baseline."
3. **role-binding bypass in implementation code** — base mechanism
   modules (``m1_success_audit.py`` etc., as they migrate to the Probe
   interface) must access state through
   ``ctx.state_for_role("<role>")``. Direct comparison with a domain
   state literal (``if state == "closed_form":``) bypasses the role
   indirection layer and breaks cross-domain reuse. The check is a
   regex scan over the listed implementation files for any of the
   declared domain state names.

Step-1 surface: the structural checks ((1) + (2)) are the primary
acceptance for this commit. The implementation-code scan ((3)) is
wired but starts with an empty file list — it activates in step 2
when the first base mechanism probe (M1) migrates and the implementation
file enters the scan set.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from engine.profile.types import Profile


# ---------------------------------------------------------------------------
# Structural lint (profile JSON shape)
# ---------------------------------------------------------------------------


def lint_base_profile(profile: Profile) -> list[str]:
    """The base profile may contain ONLY mechanism entries.

    Returns a list of human-readable violation strings. Empty list ==
    pass. Caller decides how to surface (CI fail / pre-commit / etc.).
    """
    if not profile.is_base:
        return [
            f"lint_base_profile called on non-base profile '{profile.name}' "
            f"(loader did not flag is_base=True)"
        ]

    violations: list[str] = []
    for probe in profile.probes:
        if not probe.mechanism:
            violations.append(
                f"base profile '{profile.name}': probe '{probe.name}' is not "
                f"mechanism: true (base may only contain mechanism entries)"
            )
    for gate in profile.gates:
        if not gate.mechanism:
            violations.append(
                f"base profile '{profile.name}': gate '{gate.id}' is not "
                f"mechanism: true (base may only contain mechanism entries)"
            )
    # Evidence classes, node states, routing rules and scope semantics are
    # domain-level concerns and must not appear in base — base only knows
    # roles and mechanisms.
    if profile.evidence_classes:
        violations.append(
            f"base profile '{profile.name}': evidence_classes is a domain field "
            f"and must not appear in base"
        )
    if profile.node_states:
        violations.append(
            f"base profile '{profile.name}': node_states is a domain field "
            f"(base references roles, domain binds states); must not appear in base"
        )
    if profile.routing_rules:
        violations.append(
            f"base profile '{profile.name}': routing_rules is a domain field "
            f"and must not appear in base"
        )
    if profile.scope_semantics:
        violations.append(
            f"base profile '{profile.name}': scope_semantics is a domain field "
            f"and must not appear in base"
        )
    if profile.scope_order:
        violations.append(
            f"base profile '{profile.name}': scope_order is a domain field "
            f"(scope vocabulary + ordering is domain semantics, base only "
            f"consumes via ScopeBoundaryGate); must not appear in base"
        )
    if profile.cap_mapping:
        violations.append(
            f"base profile '{profile.name}': cap_mapping is a domain field "
            f"(category → evidence_class id is domain semantics, §19.9 vmp #4); "
            f"must not appear in base"
        )
    if profile.task_templates:
        violations.append(
            f"base profile '{profile.name}': task_templates is a domain field "
            f"(recommended task procedures are intent-class semantics, "
            f"PLAN §20.1.2); must not appear in base"
        )
    return violations


def lint_domain_profile(profile: Profile) -> list[str]:
    """A domain profile may NOT contain ``mechanism: true`` entries.

    Prevents downstream from silently elevating an arbitrary probe to
    the mechanism baseline. (Registry merge also catches this with
    ProfileMergeError; lint is the file-level pre-merge check so CI
    can fail before the registry even loads.)
    """
    if profile.is_base:
        return [
            f"lint_domain_profile called on base profile '{profile.name}' "
            f"(use lint_base_profile instead)"
        ]

    violations: list[str] = []
    for probe in profile.probes:
        if probe.mechanism:
            violations.append(
                f"domain profile '{profile.name}': probe '{probe.name}' declares "
                f"'mechanism: true' — only base may declare mechanism entries"
            )
    for gate in profile.gates:
        if gate.mechanism:
            violations.append(
                f"domain profile '{profile.name}': gate '{gate.id}' declares "
                f"'mechanism: true' — only base may declare mechanism entries"
            )
    return violations


# ---------------------------------------------------------------------------
# Source-level lint (Python implementation files)
# ---------------------------------------------------------------------------


# Match double-quoted, single-quoted, or f-string-prefixed literals containing
# the target state name. Comments use ``#`` and won't appear inside string
# literals, so a literal-substring match within the source is adequate for
# step-1 scope. ast-based check is a v2 refinement.
def _state_literal_patterns(state_name: str) -> list[re.Pattern[str]]:
    escaped = re.escape(state_name)
    return [
        re.compile(rf'"{escaped}"'),
        re.compile(rf"'{escaped}'"),
    ]


def lint_base_mechanism_source(
    source_paths: Iterable[Path],
    forbidden_state_names: Iterable[str],
) -> list[str]:
    """Scan base mechanism Python files for domain-state literals.

    The role indirection layer (§19.1) is structurally broken if a
    mechanism module compares state names directly. Mechanism modules
    must reach domain states through
    ``ctx.state_for_role("<role>")`` only.

    ``forbidden_state_names`` is sourced from the merged profile's
    ``node_states`` so the check stays in lockstep with whatever
    domain profile is in play. Pass an empty list while no domain is
    loaded — the function then returns an empty violation list.
    """
    forbidden = [name for name in forbidden_state_names if name]
    if not forbidden:
        return []

    patterns: list[tuple[str, re.Pattern[str]]] = []
    for name in forbidden:
        for pat in _state_literal_patterns(name):
            patterns.append((name, pat))

    violations: list[str] = []
    for src_path in source_paths:
        path = Path(src_path)
        if not path.exists():
            violations.append(f"lint_base_mechanism_source: file not found: {path}")
            continue
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.lstrip()
            # Skip comments and docstring-shaped lines — the lint cares about
            # runtime behaviour, not commentary. (Module docstrings still pass
            # this filter because they start with `"""` not `#`; if a literal
            # appears inside a docstring it counts as documentation drift
            # and is intentionally flagged.)
            if stripped.startswith("#"):
                continue
            for state_name, pat in patterns:
                if pat.search(line):
                    violations.append(
                        f"{path}:{lineno}: base mechanism module references "
                        f"domain state literal '{state_name}' — use "
                        f"ctx.state_for_role(<role>) instead (§19.1)"
                    )
    return violations


def lint_kernel_source(
    source_paths: Iterable[Path],
    forbidden_literals: Iterable[str],
) -> list[str]:
    """Scan engine kernel framework files for any domain literal.

    Same shape as :func:`lint_base_mechanism_source` but conceptually
    distinct: kernel framework code (``core.py``, state-machine /
    probe / gate / routing runtimes) must remain entirely
    profile-agnostic. ``forbidden_literals`` should include both
    evidence-class IDs (``"A"`` / ``"B"`` / ``"C"`` — quoted match
    only, not the bare identifier) and node-state names.
    """
    return lint_base_mechanism_source(source_paths, forbidden_literals)


# ---------------------------------------------------------------------------
# Dynamic-criterion lint — v0.4.0 B6 / §19.9 base #6
# ---------------------------------------------------------------------------


# Static-offset / pinned-label fingerprints.  We deliberately keep the set
# small and high-signal — false positives push profile authors to add
# elaborate exception lists, defeating the lint.  Patterns matched:
#   * ``0x1234``        — hex offsets / addresses
#   * ``offset 24``     — decimal-offset phrase
#   * ``at offset N``   — "at offset" phrase (decimal or hex)
#   * ``+0x18`` / ``+24`` — relative-offset shorthand
_STATIC_OFFSET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Relative-offset shorthand first so the matched literal carries the
    # leading sign — `+0x18` is more diagnostic than the bare `0x18`.
    re.compile(r"[+\-]0x[0-9a-fA-F]+\b"),
    re.compile(r"\b0x[0-9a-fA-F]+\b"),
    re.compile(r"\bat\s+offset\s+\d+\b", re.IGNORECASE),
    re.compile(r"\boffset\s+\d+\b", re.IGNORECASE),
)


# Scan-style language signals the criterion is dynamic — e.g. "located
# by scan over <pattern>", "match handler by signature".  When a criterion
# mentions an offset literal but also includes one of these keywords the
# lint treats the offset as *example-only* and lets it through.
_DYNAMIC_KEYWORDS: tuple[str, ...] = (
    "scan", "locate", "located", "search", "match", "find",
    "pattern", "signature", "shape",
)


def _criterion_violates_dynamic_rule(text: str) -> tuple[bool, str | None]:
    """Return ``(violates, matched_literal)`` for one criterion string."""
    if not text:
        return False, None
    for pat in _STATIC_OFFSET_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        lowered = text.lower()
        if any(k in lowered for k in _DYNAMIC_KEYWORDS):
            # The criterion mentions a concrete offset but is framed
            # around a dynamic descriptor (e.g. "scan for sig X at
            # +0x18 of the match"); offset is illustrative, accept.
            return False, None
        return True, m.group(0)
    return False, None


def lint_dynamic_criteria(profile: Profile) -> list[str]:
    """Lint profile gate rules + scope rules for static-offset criteria
    (§19.9 base #6 / v0.4.0 B6).

    A review-checklist criterion that pins a concrete byte offset turns
    into a false seam the moment the implementation drifts.  The lint
    flags criteria that mention an offset literal without any
    accompanying scan-style descriptor.  Both base and domain profiles
    are checked — the principle is general.

    Returns human-readable violation strings; empty list = pass.
    """
    violations: list[str] = []
    for gate in profile.gates:
        bad, matched = _criterion_violates_dynamic_rule(gate.rule)
        if bad:
            violations.append(
                f"profile '{profile.name}': gate '{gate.id}' criterion pins "
                f"a concrete offset/literal ('{matched}') — frame as a scan "
                f"over a pattern instead so the gate survives implementation "
                f"drift (§19.9 base #6)"
            )
    for sc in profile.scope_semantics:
        bad, matched = _criterion_violates_dynamic_rule(sc.tag_scope)
        if bad:
            violations.append(
                f"profile '{profile.name}': scope rule "
                f"'{sc.when_state} → {sc.tag_scope}' pins a concrete literal "
                f"('{matched}') — use a named scope value, not an offset"
            )
    return violations
