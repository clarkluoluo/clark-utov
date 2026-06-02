"""CVD mount policy + T-intake readers + escalation rules + stall_pressure.

Implements CVD_MOUNT_POLICY.md on top of the frozen CVD/CVD-Plus base:
  - MountPolicy (§7): a first-class, agent-editable, checkpoint-serialized object —
    stall threshold, budgets, armed heavy tools, disabled tools.
  - RunManifest (§6): the pre-drive announcement (mounted / armed triggers / budget).
  - T-intake Readers (§3): wrap the obs_readers (calltrace, hook-dump) as Reader
    plugins, default-ON, self-gating on artifact presence.
  - EscalationRules (§5 / CVD_DESIGN §11.2): E2 opaque, E3 scrub preempt, E6 widen.
  - stall_pressure + the T2 level-jump cost gate (§5 / §11.4).

Concrete plugins import the cvd interfaces; cvd imports THIS module lazily (in
default_registry / default_policy / the driver) to avoid an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cvd import (
    Artifact,
    Candidate,
    EscalationRule,
    ReadResult,
    Reader,
)

# Names of the planned heavy (T2) tools — armed, not built by default (§3).
HEAVY_TOOLS = ("triton_symex", "vmtrace", "localize_divergence", "recapture")


# --- MountPolicy (§7) -------------------------------------------------------

@dataclass
class MountPolicy:
    stall_theta: float = 3.0
    expected_tries: int = 4              # tries-to-confirm baseline for stall_pressure
    max_candidates: int = 64
    max_widen: int = 4
    # T2 / side-effect budget gate (§10.2) — conservative: heavy exceeds by default
    time_budget_s: float = 60.0
    disk_budget_mb: float = 256.0
    token_budget: int = 50_000
    heavy_armed: bool = False            # arm the T2 tier
    heavy_tools: list[str] = field(default_factory=lambda: list(HEAVY_TOOLS))
    disabled_tools: set[str] = field(default_factory=set)   # tools the agent removed

    def to_dict(self) -> dict[str, Any]:
        return {"stall_theta": self.stall_theta, "expected_tries": self.expected_tries,
                "max_candidates": self.max_candidates, "max_widen": self.max_widen,
                "time_budget_s": self.time_budget_s, "disk_budget_mb": self.disk_budget_mb,
                "token_budget": self.token_budget, "heavy_armed": self.heavy_armed,
                "heavy_tools": list(self.heavy_tools),
                "disabled_tools": sorted(self.disabled_tools)}

    @classmethod
    def from_dict(cls, d: dict) -> "MountPolicy":
        return cls(stall_theta=d.get("stall_theta", 3.0),
                   expected_tries=d.get("expected_tries", 4),
                   max_candidates=d.get("max_candidates", 64),
                   max_widen=d.get("max_widen", 4),
                   time_budget_s=d.get("time_budget_s", 60.0),
                   disk_budget_mb=d.get("disk_budget_mb", 256.0),
                   token_budget=d.get("token_budget", 50_000),
                   heavy_armed=d.get("heavy_armed", False),
                   heavy_tools=list(d.get("heavy_tools", HEAVY_TOOLS)),
                   disabled_tools=set(d.get("disabled_tools", [])))


def default_policy() -> MountPolicy:
    return MountPolicy()


# --- T-intake Readers (§3) --------------------------------------------------

class CalltraceReader(Reader):
    name = "calltrace_reader"; version = "1"; owner = "core"

    def detect(self, artifact: Artifact) -> bool:
        return artifact.kind == "calltrace"

    def read(self, artifact: Artifact) -> ReadResult:
        from .obs_readers import parse_calltrace
        return ReadResult(call_events=parse_calltrace(artifact.text))


class HookDumpReader(Reader):
    name = "hook_dump_reader"; version = "1"; owner = "core"

    def detect(self, artifact: Artifact) -> bool:
        return artifact.kind == "hook_dump"

    def read(self, artifact: Artifact) -> ReadResult:
        from .obs_readers import parse_hook_snapshots
        return ReadResult(snapshots=parse_hook_snapshots(artifact.text))


# --- EscalationRules (§5 / CVD_DESIGN §11.2) --------------------------------

class ScrubPreemptRule(EscalationRule):
    """E3: a recovered-transient (write-then-wipe) candidate is high-information —
    preempt it to the front instead of relying on soft base_value ordering."""
    name = "E3_scrub_preempt"; version = "1"; owner = "core"

    def trigger(self, c: Candidate, state, history) -> bool:
        return c.kind == "recovered_transient"

    def escalate(self, c: Candidate, state) -> dict:
        return {"action": "preempt", "why": "rare high-entropy stack-wipe secret"}


class OpaqueBoundaryRule(EscalationRule):
    """E2: a provenance candidate (spawned from a confirmed sink) may resolve to
    OPAQUE_CALLEE — preempt it so the boundary call / BLR target gets resolved
    promptly by #3 rather than waiting behind the cheap frontier."""
    name = "E2_opaque_boundary"; version = "1"; owner = "core"

    def trigger(self, c: Candidate, state, history) -> bool:
        return c.kind == "provenance"

    def escalate(self, c: Candidate, state) -> dict:
        return {"action": "preempt", "why": "resolve producer / call boundary (#3, BLR target)"}


class WidenRule(EscalationRule):
    """E6: frontier-exhausted widening. The driver already widens on an empty
    frontier; this row makes the trigger explicit in the policy/manifest."""
    name = "E6_stall_widen"; version = "1"; owner = "core"

    def trigger(self, c: Candidate, state, history) -> bool:
        return False   # frontier-level, handled by the driver's empty-frontier widen

    def escalate(self, c: Candidate, state) -> dict:
        return {"action": "widen"}


def default_rules() -> list[EscalationRule]:
    return [ScrubPreemptRule(), OpaqueBoundaryRule(), WidenRule()]


def default_readers() -> list[Reader]:
    return [CalltraceReader(), HookDumpReader()]


# --- stall_pressure + T2 cost gate (§5 / §11.4) -----------------------------

def stall_pressure(*, tried: int, frontier_size: int, confirms: int,
                   policy: MountPolicy) -> float:
    """(budget_spent/expected) × frontier_size / progress.  Diverging =
    much spend, large frontier, no confirms → high; a converging run stays low."""
    spent_ratio = tried / max(1, policy.expected_tries)
    progress = max(1, confirms)
    return spent_ratio * max(1, frontier_size) / progress


def estimate_t2(state, action: str) -> dict:
    """Coarse (time, disk, tokens) for a heavy T2 action — the cost oracle behind
    the budget gate (§10.2). Heavy by construction (re-trace / symbolic exec), so
    it exceeds a conservative budget unless the agent raised it. Deterministic;
    does NOT run the heavy pipeline."""
    n = max(1, len(state.items))
    if action in ("vmtrace", "recapture"):
        return {"action": action, "time_s": 30.0 + 0.01 * n, "disk_mb": 64.0 + 0.05 * n,
                "tokens": 0}
    # triton_symex / localize_divergence: symbolic / LLM heavy
    return {"action": action, "time_s": 20.0 + 0.005 * n, "disk_mb": 16.0,
            "tokens": 2000 + 5 * n}


def heavy_over_budget(est: dict, policy: MountPolicy) -> bool:
    return (est.get("time_s", 0) > policy.time_budget_s
            or est.get("disk_mb", 0) > policy.disk_budget_mb
            or est.get("tokens", 0) > policy.token_budget)


# --- RunManifest (§6) -------------------------------------------------------

def build_manifest(state, registry, policy: MountPolicy) -> dict:
    def _live(plugins):
        return [f"{p.name}@{p.version}" for p in plugins if p.name not in policy.disabled_tools]

    mounted = (_live(registry.readers) + _live(registry.generators)
               + _live(registry.verifiers) + _live(registry.terminals))
    armed = [
        {"trigger": "signal-present", "condition": "a T1 detector's signature appears",
         "mounts": _live(registry.generators)},
        {"trigger": "sink-confirmed", "condition": "SinkValidator CONFIRMED",
         "mounts": ["oracle_provenance@3"]},
        {"trigger": "frontier-exhausted", "condition": "no candidates in scope",
         "mounts": ["in-memory widen"]},
        {"trigger": "stall_pressure>θ", "condition": f"> {policy.stall_theta}",
         "mounts": (list(policy.heavy_tools) if policy.heavy_armed
                    else ["(heavy tier not armed/built → EXTENSION_REQUEST)"])},
        {"trigger": "budget-over", "condition": "T2 estimate over budget",
         "mounts": ["BUDGET_PAUSE"]},
    ]
    return {
        "goal": {"expected_len": len(state.expected), "obs_scope": state.obs_scope},
        "obs_scope": state.obs_scope,
        "mounted": mounted,
        "armed_triggers": armed,
        "budget": {"max_candidates": policy.max_candidates, "max_widen": policy.max_widen,
                   "time_s": policy.time_budget_s, "disk_mb": policy.disk_budget_mb,
                   "tokens": policy.token_budget},
        "rationale": ("default sweep; T1 signal tools self-gate; heavy tier "
                      + ("armed" if policy.heavy_armed else "NOT armed")
                      + f" behind stall_pressure>{policy.stall_theta} + budget"),
    }


__all__ = [
    "MountPolicy", "default_policy", "HEAVY_TOOLS",
    "CalltraceReader", "HookDumpReader", "default_readers",
    "ScrubPreemptRule", "OpaqueBoundaryRule", "WidenRule", "default_rules",
    "stall_pressure", "estimate_t2", "heavy_over_budget", "build_manifest",
]
