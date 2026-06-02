"""Implementation path — an OPTIONAL task-spec field (roadmap §8.11/§9.4).

the reference case's three-layer spoof had a secondary cause: the brief gave no
implementation path, so the task read as "a mountain with no steps" and the
agent, under the pressure to "complete", reached for a formal shortcut. The
cure is to let a brief that *wants* a rhythm spell out its light-to-heavy path
(staged plan + upgrade condition + a compliant wall-hit exit per stage).

This field is a **tool, not a mandate** (roadmap §9.4 "implementation_path 降级
为可选工具"). Hypotask/utov offers it; a brief may omit it and still load. The
runtime enforcement of light-to-heavy lives in :mod:`engine.vmp_phase_api`;
this is the spec-level *declaration* of the same intent, available to renderers
and agents.

SCOPE NOTE — this module validates SHAPE ONLY (types). It deliberately does NOT
judge path quality: it does not require staging, does not grade whether the
gradient is "reasonable", and does not require a compliant_exit on every stage.
Those are brief-quality checks (the "brief 三审") held for a separate decision —
brief review must stay on "form honesty" and never reach into "task content"
(roadmap §9: 伸手过去就越界了).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ImplementationPathError(Exception):
    """Raised when an implementation_path is mis-shaped (wrong types)."""


@dataclass(frozen=True)
class ImplementationStage:
    """One stage of a light-to-heavy plan.

    Only ``name`` is required. ``upgrade_when`` is the condition under which
    the agent may move to a heavier stage ("轻闭不住" proof); ``compliant_exit``
    is the honest wall-hit route (declare capability_blocked / escalate /
    downgrade) so the agent is never trapped between formal-spoof and grinding.
    Both are free text and OPTIONAL — unfilled means "not specified", not
    "invalid"."""

    name:           str
    intent:         str = ""
    upgrade_when:   str = ""
    compliant_exit: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name":           self.name,
            "intent":         self.intent,
            "upgrade_when":   self.upgrade_when,
            "compliant_exit": self.compliant_exit,
        }


@dataclass(frozen=True)
class ImplementationPath:
    """An optional staged plan attached to a task spec."""

    stages: tuple[ImplementationStage, ...] = ()
    note:   str = ""

    @classmethod
    def parse(cls, raw: Any, *, source: str = "<dict>") -> "ImplementationPath":
        """Parse a JSON implementation_path object. Shape-only validation.

        ``raw`` is expected to be a dict (the loader handles the absent /
        null case before calling here).
        """
        if not isinstance(raw, dict):
            raise ImplementationPathError(
                f"{source}: implementation_path must be a JSON object"
            )
        stages_raw = raw.get("stages", []) or []
        if not isinstance(stages_raw, list):
            raise ImplementationPathError(
                f"{source}: implementation_path.stages must be a list"
            )
        stages: list[ImplementationStage] = []
        seen: set[str] = set()
        for i, s in enumerate(stages_raw):
            if not isinstance(s, dict):
                raise ImplementationPathError(
                    f"{source}: stages[{i}] must be an object"
                )
            name = s.get("name")
            if not isinstance(name, str) or not name:
                raise ImplementationPathError(
                    f"{source}: stages[{i}] missing non-empty string 'name'"
                )
            if name in seen:
                raise ImplementationPathError(
                    f"{source}: duplicate stage name '{name}'"
                )
            seen.add(name)

            def _opt_str(field_name: str, idx: int = i) -> str:
                v = s.get(field_name, "") or ""
                if not isinstance(v, str):
                    raise ImplementationPathError(
                        f"{source}: stages[{idx}].{field_name} must be a string"
                    )
                return v

            stages.append(ImplementationStage(
                name=name,
                intent=_opt_str("intent"),
                upgrade_when=_opt_str("upgrade_when"),
                compliant_exit=_opt_str("compliant_exit"),
            ))

        note = raw.get("note", "") or ""
        if not isinstance(note, str):
            raise ImplementationPathError(
                f"{source}: implementation_path.note must be a string"
            )
        return cls(stages=tuple(stages), note=note)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [s.to_dict() for s in self.stages],
            "note":   self.note,
        }
