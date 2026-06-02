"""Conjunctive gate runtime (PLAN §19.7 #7 Lock B).

The runtime side of the mechanism baseline lock. ``Lock A`` (load-time
registry rejection) prevents subprofiles from declaring "I don't want
M1" through profile-data tampering. ``Lock B`` is the bypass-Lock-A
defence: even when a caller hands the gate a :class:`MergedProfile`
that has been tampered with — mechanism probes removed from
``probes``, ``mechanism_probe_names`` zeroed out, the registry skipped
entirely — the gate **still** runs every mechanism probe at
evaluation time.

The defence is structural: the gate does not consult the merged
profile to learn which probes are mechanism. It iterates
:func:`engine.profile.probe_runtime.list_builtin_probes` and filters
on ``cls.mechanism is True`` — a *class attribute on the
implementation*, which a profile-data attacker cannot reach. (An
attacker with code-execution access can monkey-patch the class, but
at that point Lock B is no longer the right defence layer — the
threat model has shifted from "tampered profile data" to "tampered
engine bytecode," which is the host system's problem.)

The gate is conjunctive: any verdict with ``result == "fail"`` fails
the overall gate. ``undetermined`` verdicts neither pass nor fail —
they are recorded but don't move the verdict. Domain (non-mechanism)
verdicts are mixed in via ``extra_verdicts``: the wrapper runs domain
probes and hands the resulting Verdicts to ``evaluate``; mechanism
probes are run inside the gate itself, so a caller cannot accidentally
forget to include them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from engine.profile.evidence_class_synth import synth_node_cap
from engine.profile.probe_runtime import (
    EvidenceClassCap,
    Probe,
    ProbeContext,
    Verdict,
    list_builtin_probes,
)
from engine.profile.registry import MergedProfile


@dataclass(frozen=True)
class GateResult:
    """Outcome of one conjunctive-gate evaluation."""

    passed: bool
    verdicts: tuple[Verdict, ...]
    mechanism_verdicts: tuple[Verdict, ...]
    node_cap: EvidenceClassCap | None = None
    failing_probes: tuple[str, ...] = field(default_factory=tuple)


class ConjunctiveGate:
    """Force-include mechanism probes + combine domain verdicts.

    Construct once per active session/profile. Mechanism probe
    instances are cached on the gate so M3 keeps its per-session
    detector state across calls.
    """

    def __init__(self, profile: MergedProfile) -> None:
        self._profile = profile
        self._mechanism_instances: dict[str, Probe] = {}

    @property
    def profile(self) -> MergedProfile:
        return self._profile

    def mechanism_probe_classes(self) -> dict[str, type[Probe]]:
        """Return ``{name: class}`` for every builtin probe whose
        implementation class declares ``mechanism = True``.

        The merged profile is **not** consulted — Lock B's whole
        point. The source of truth is the import-time decorator
        registry plus the implementation class's ``mechanism``
        attribute.
        """
        return {
            name: cls
            for name, cls in list_builtin_probes().items()
            if getattr(cls, "mechanism", False)
        }

    def _instance_for(self, name: str, cls: type[Probe]) -> Probe:
        cached = self._mechanism_instances.get(name)
        if cached is not None:
            return cached
        probe = cls()
        self._mechanism_instances[name] = probe
        return probe

    def run_mechanism_verdicts(self, ctx: ProbeContext) -> tuple[Verdict, ...]:
        """Run every base mechanism probe under ``ctx``."""
        verdicts: list[Verdict] = []
        for name, cls in sorted(self.mechanism_probe_classes().items()):
            probe = self._instance_for(name, cls)
            verdicts.append(probe.run(ctx))
        return tuple(verdicts)

    def evaluate(
        self,
        ctx: ProbeContext,
        *,
        extra_verdicts: Iterable[Verdict] = (),
    ) -> GateResult:
        """Conjunctive evaluation.

        ``extra_verdicts`` carries pre-computed domain probe results —
        the wrapper has already run them since domain probes can be
        disabled by profile (open layer) and their selection IS
        profile-driven. Mechanism verdicts are computed here so they
        cannot be silently omitted.
        """
        mechanism_verdicts = self.run_mechanism_verdicts(ctx)
        extras = tuple(extra_verdicts)
        all_verdicts = mechanism_verdicts + extras

        failing = tuple(v.probe for v in all_verdicts if v.result == "fail")
        node_cap = synth_node_cap(
            all_verdicts,
            evidence_classes=self._profile.evidence_classes,
        )

        return GateResult(
            passed=len(failing) == 0,
            verdicts=all_verdicts,
            mechanism_verdicts=mechanism_verdicts,
            node_cap=node_cap,
            failing_probes=failing,
        )
