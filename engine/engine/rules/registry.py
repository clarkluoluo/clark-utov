"""Registered plugins (PLAN §14, DECISIONS D-025).

A registered rule is a deterministic predicate that:
  - matches a (kind, applicability_tags) signature
  - returns either a conclusion (with confidence) or `abstain`
  - is still subject to verifier per PLAN §1.1

Storage: JSON files under work_root/_rules/ + index in a sqlite table for
fast lookup. Rules are global across targets (matched by applicability_tags).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class RuleStatus(str, Enum):
    ACTIVE = "active"
    QUARANTINED = "quarantined"
    REVOKED = "revoked"


@dataclass
class Rule:
    rule_id: str
    kind: str                                  # the hyp.kind it replaces
    applicability_tags: list[str]              # MUST be non-empty (D-025)
    matcher: Callable[[dict[str, Any]], bool]  # condition: input → fire?
    conclude: Callable[[dict[str, Any]], dict[str, Any] | None]
                                               # returns conclusion or None for abstain
    confidence: float
    origin_hyp_ids: list[int] = field(default_factory=list)   # account trail
    status: RuleStatus = RuleStatus.ACTIVE


class Registry:
    def __init__(self) -> None:
        self._rules: dict[str, Rule] = {}

    def register(self, rule: Rule) -> None:
        if not rule.applicability_tags:
            raise ValueError("rule must declare at least one applicability tag (D-025)")
        if rule.rule_id in self._rules:
            raise ValueError(f"rule {rule.rule_id} already registered")
        self._rules[rule.rule_id] = rule

    def match(self, kind: str, target_tags: set[str]) -> list[Rule]:
        """Return active rules whose kind matches and whose applicability_tags
        intersect target_tags."""
        out = []
        for r in self._rules.values():
            if r.status != RuleStatus.ACTIVE:
                continue
            if r.kind != kind:
                continue
            if any(tag in target_tags for tag in r.applicability_tags):
                out.append(r)
        return out

    def quarantine(self, rule_id: str) -> None:
        self._rules[rule_id].status = RuleStatus.QUARANTINED

    def revoke(self, rule_id: str) -> None:
        self._rules[rule_id].status = RuleStatus.REVOKED

    def get(self, rule_id: str) -> Rule:
        return self._rules[rule_id]
