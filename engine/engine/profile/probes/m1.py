"""M1 success-audit probe adapter (PLAN §19 / IMPL_PLAN §P1.0 step 2).

Wraps :mod:`engine.m1_success_audit` — the underlying audit logic is
unchanged. This module is the Probe-interface entry point so the
profile registry can dispatch M1 by name as a base mechanism probe.

Mapping :class:`SuccessAuditResult` → :class:`Verdict`:

  * ``action=='allow'``     → ``result='pass'``
  * ``action=='downgrade'`` → ``result='fail'`` (the wrapper still
    proceeds with rewritten params; ``fail`` flips the conjunctive
    gate, which is the intent — a downgraded claim must not slip
    through as a passing success)
  * ``action=='reject'``    → ``result='fail'``
  * ``audit is None``       → ``result='undetermined'`` (the call is
    not an archival surface; M1 simply doesn't apply)

The audit grade always populates :class:`EvidenceClassCap`, so the
node's evidence-class ceiling is capped accordingly regardless of
which action is taken.
"""

from __future__ import annotations

from engine.m1_success_audit import M1AuditConfig, audit_success_claim
from engine.profile.probe_runtime import (
    EvidenceClassCap,
    Probe,
    ProbeContext,
    Verdict,
    register_builtin_probe,
)


@register_builtin_probe("m1_success_audit")
class M1SuccessAuditProbe(Probe):
    """Mechanism probe: a success claim must be backed by closure-state
    evidence, dimension coverage, scope, and overfit absence.

    M1 references the domain via the ``closure_state`` role (see
    PLAN §19.1); the role binding plumbing activates in step 5 when
    the conjunctive gate runtime arrives. Step 2 just produces a
    verdict directly from the existing audit logic.
    """

    name = "m1_success_audit"
    mechanism = True
    inputs = ("method", "params")
    outputs = ("m1_audit",)

    def __init__(self, config: M1AuditConfig | None = None) -> None:
        self._config = config

    def run(self, ctx: ProbeContext) -> Verdict:
        audit = audit_success_claim(ctx.method, ctx.params, cfg=self._config)
        if audit is None:
            return Verdict(probe=self.name, result="undetermined")

        if audit.action == "allow":
            result: str = "pass"
        else:
            # downgrade and reject both flip the gate — neither permits a
            # bare success claim to be archived without intervention.
            result = "fail"

        cap = EvidenceClassCap(
            class_id=audit.evidence_class,
            reason=f"M1 grade {audit.evidence_class} ({audit.action})",
        )

        return Verdict(
            probe=self.name,
            result=result,  # type: ignore[arg-type]
            evidence={
                "grade": audit.evidence_class,
                "action": audit.action,
                "downgraded_to": audit.downgraded_to,
                "untested_dimensions": list(audit.untested_dimensions),
                "overfit_flag": audit.overfit_flag,
                "scope": audit.scope,
                "closure_consistent": audit.closure_consistent,
                "pass_rate": audit.pass_rate,
                "sample_count": audit.sample_count,
                "intercepted_reason": audit.intercepted_reason,
            },
            affects_evidence_class=cap,
        )
