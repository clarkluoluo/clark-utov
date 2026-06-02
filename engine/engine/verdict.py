"""Structured verdict emitter (capability_request.md §P2-3, M10).

A natural-language verdict like *"确认（分层）"* hides four heterogeneous
conclusions (``io_oracle``, ``implementation_sm3_gmt``, ``triton_symbolic``,
``sm3_body_binary``) behind the single word "确认"; readers walk away
with "confirmed" and forget the gap. M10 forces every final-report
verdict to be a YAML block with four exclusive lists:

```yaml
confirmed:        [io_oracle, implementation_sm3_gmt]
not_confirmed:    [sm3_body_binary]
invalidated:      []
known_gaps:       [vmp_register_capture]
```

This module produces the YAML deterministically (no PyYAML dependency —
the schema is tight enough that we can render by hand). Any layer name
must appear in exactly one list; a bare word like "通过" without a layer
subject is rejected at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_BANNED_BARE_WORDS = ("通过", "确认", "success", "passed", "ok")


class VerdictError(ValueError):
    """Raised when a verdict construction violates M10 (bare positive
    word, duplicate layer, or empty subject)."""


@dataclass(frozen=True, slots=True)
class Verdict:
    """Conjunctive verdict over four exclusive lists of layer ids."""

    confirmed:     tuple[str, ...] = field(default_factory=tuple)
    not_confirmed: tuple[str, ...] = field(default_factory=tuple)
    invalidated:   tuple[str, ...] = field(default_factory=tuple)
    known_gaps:    tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirmed":     list(self.confirmed),
            "not_confirmed": list(self.not_confirmed),
            "invalidated":   list(self.invalidated),
            "known_gaps":    list(self.known_gaps),
        }

    @property
    def archival_allowed(self) -> bool:
        """``true`` iff nothing is in ``not_confirmed`` or
        ``invalidated`` (M6 conjunctive rule). ``known_gaps`` doesn't
        block archival; that's what it's for."""
        return not self.not_confirmed and not self.invalidated

    def to_yaml(self) -> str:
        lines = [
            "verdict:",
            f"  confirmed:     {_yaml_list(self.confirmed)}",
            f"  not_confirmed: {_yaml_list(self.not_confirmed)}",
            f"  invalidated:   {_yaml_list(self.invalidated)}",
            f"  known_gaps:    {_yaml_list(self.known_gaps)}",
            f"  archival_allowed: {'true' if self.archival_allowed else 'false'}",
        ]
        return "\n".join(lines) + "\n"


def _yaml_list(items: tuple[str, ...]) -> str:
    if not items:
        return "[]"
    quoted = ", ".join(items)
    return f"[{quoted}]"


def build_verdict(
    *,
    confirmed:     list[str] | tuple[str, ...] = (),
    not_confirmed: list[str] | tuple[str, ...] = (),
    invalidated:   list[str] | tuple[str, ...] = (),
    known_gaps:    list[str] | tuple[str, ...] = (),
) -> Verdict:
    """Validate then construct. Rejects:
      - empty / banned bare words
      - same layer in multiple lists
    """
    buckets = {
        "confirmed":     tuple(confirmed),
        "not_confirmed": tuple(not_confirmed),
        "invalidated":   tuple(invalidated),
        "known_gaps":    tuple(known_gaps),
    }
    seen: dict[str, str] = {}
    for bucket, items in buckets.items():
        for layer in items:
            if not isinstance(layer, str) or not layer.strip():
                raise VerdictError(f"empty / non-string layer in {bucket}: {layer!r}")
            if layer.strip().lower() in _BANNED_BARE_WORDS:
                raise VerdictError(
                    f"bare verdict word {layer!r} in {bucket} — supply a "
                    f"layer subject (e.g. 'sm3_body_binary'), not a bare "
                    f"verb."
                )
            if layer in seen:
                raise VerdictError(
                    f"layer {layer!r} appears in both {seen[layer]} and "
                    f"{bucket} — lists must be disjoint."
                )
            seen[layer] = bucket
    return Verdict(
        confirmed=buckets["confirmed"],
        not_confirmed=buckets["not_confirmed"],
        invalidated=buckets["invalidated"],
        known_gaps=buckets["known_gaps"],
    )


def lint_markdown(text: str) -> list[str]:
    """Return a list of complaints if a markdown report uses banned bare
    verdict words outside fenced code blocks. Empty list = clean."""
    complaints: list[str] = []
    in_fence = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        # only check lines inside or below a verdict section heuristically
        for word in _BANNED_BARE_WORDS:
            # bare presence — no following ':' (which is the structured form)
            if f" {word}" in line and not (line.strip().lower() == word):
                # Allow when followed by a subject (i.e. ': layer_name')
                if f"{word}:" in line or f"{word} layer" in line.lower():
                    continue
                complaints.append(
                    f"L{lineno}: bare verdict word {word!r} — use structured "
                    f"YAML or 'confirmed: <layer>'."
                )
                break
    return complaints
