"""Routing-runtime — cause → action lookup driven by profile data.

The kernel-side companion to ``MergedProfile.routing_rules``. Block-cause
classification (recognition_gap / strategy_gap / true_boundary /
collection_gap) is mechanism — every domain reaches stuck nodes the
same way and uses the same cause vocabulary. But what *action* a
domain takes when a cause fires is a policy question:

  * vmp_algorithm_extraction routes ``recognition_gap`` to
    ``escalate_l2`` (small-LLM pattern matcher).
  * A domain without an L2 layer could route the same cause straight
    to the user (``escalate_user``), or to a different probe.

This module provides :class:`RoutingTable`, a read-only view onto the
active profile's routing rules. Existing call sites (currently
``engine.block_cause.BlockCauseRouter``) consult it when present and
fall back to their legacy hardcoded mapping when not — that's how the
migration stays backwards-compatible (§19.7 #2).
"""

from __future__ import annotations

from typing import Iterable, Optional

from engine.profile.registry import MergedProfile


class RoutingTable:
    """Read-only cause → ordered-actions lookup over a profile's rules.

    Construction is cheap (just builds a dict from the profile's
    ``routing_rules`` tuple). Callers can hold one instance for the
    life of a session.
    """

    def __init__(self, profile: MergedProfile) -> None:
        self._profile = profile
        self._by_cause: dict[str, tuple[str, ...]] = {
            rule.cause: tuple(rule.actions) for rule in profile.routing_rules
        }

    @property
    def profile(self) -> MergedProfile:
        return self._profile

    @property
    def causes(self) -> tuple[str, ...]:
        """All causes the profile declares routing for (alphabetised
        for stable ordering)."""
        return tuple(sorted(self._by_cause.keys()))

    def lookup(self, cause: str) -> tuple[str, ...]:
        """Return the ordered tuple of action ids the profile maps
        ``cause`` to. Empty tuple if the profile has no rule for it —
        callers can fall back to their own default.
        """
        return self._by_cause.get(cause, ())

    def primary_action(self, cause: str) -> Optional[str]:
        """First-preference action id for ``cause``, or ``None`` if
        the profile has no rule.

        The single-action case (``recognition_gap → [escalate_l2]``)
        is the dominant pattern; callers that don't care about
        fallback chains read this and treat None as "use my legacy
        default."
        """
        actions = self._by_cause.get(cause)
        if not actions:
            return None
        return actions[0]

    def has(self, cause: str) -> bool:
        return cause in self._by_cause

    def all_action_ids(self) -> frozenset[str]:
        """The union of every action id mentioned by any rule. Used
        by lint helpers to verify a profile's vocabulary against a
        runtime's known action enum."""
        ids: set[str] = set()
        for actions in self._by_cause.values():
            ids.update(actions)
        return frozenset(ids)


def lint_actions_against_known(
    table: RoutingTable, known_action_ids: Iterable[str]
) -> list[str]:
    """Verify every action id declared in the profile is recognised
    by the consuming runtime. Returns human-readable violations.

    The consuming runtime (e.g. ``BlockCauseRouter``) supplies the
    set of action ids it can actually execute; this helper catches
    a profile that names ``escalate_l4`` or similar typos before the
    router gets a chance to silently fall back to its legacy default.
    """
    known = set(known_action_ids)
    violations: list[str] = []
    for cause in table.causes:
        for action_id in table.lookup(cause):
            if action_id not in known:
                violations.append(
                    f"routing rule '{cause}' → '{action_id}' references an "
                    f"action the runtime does not recognise "
                    f"(known: {sorted(known)})"
                )
    return violations
