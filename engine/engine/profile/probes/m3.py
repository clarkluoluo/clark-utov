"""M3 bypass-block probe adapter (PLAN §19 / IMPL_PLAN §P1.0 step 3).

Wraps :mod:`engine.m3_bypass_block` — the underlying
:class:`BypassBlockDetector` retains all its session-state semantics
(history per block, sticky confirmation, follow-up interception).
This module is the Probe-interface entry point so the profile registry
can dispatch M3 by name as a base mechanism probe.

The probe is stateful at the *instance* level: it owns a
:class:`BypassBlockDetector`. The wrapper / loop layer is responsible
for keeping the same probe instance alive across calls in one session
so M3's cross-call detection works. Tests typically construct a fresh
probe per scenario.

Mapping :class:`BypassDetection` → :class:`Verdict`:

  * ``call doesn't qualify as an M3 attempt``    → ``undetermined``
  * ``recorded, threshold not yet crossed``      → ``pass``
  * ``triggered (threshold crossed on this call)`` → ``fail``
  * ``follow-up attempt on a confirmed bypass``  → ``fail``
"""

from __future__ import annotations

from engine.m3_bypass_block import (
    BypassBlockDetector,
    M3BypassConfig,
    extract_attempt,
)
from engine.profile.probe_runtime import (
    Probe,
    ProbeContext,
    Verdict,
    register_builtin_probe,
)


@register_builtin_probe("m3_bypass_block")
class M3BypassBlockProbe(Probe):
    """Mechanism probe: when ≥N distinct observation methods all fail
    variability on the same block, mark the block ``suspected_bypass``
    and refuse follow-up observation attempts on it.

    The detector defaults to ``min_failed_observations=2``; callers
    that need a different N construct the probe with a custom config.
    """

    name = "m3_bypass_block"
    mechanism = True
    inputs = ("method", "params", "result")
    outputs = ("m3_bypass",)

    def __init__(
        self,
        config: M3BypassConfig | None = None,
        detector: BypassBlockDetector | None = None,
    ) -> None:
        if detector is not None:
            self._detector = detector
            self._config = detector.cfg
        else:
            self._config = config or M3BypassConfig()
            self._detector = BypassBlockDetector(self._config)

    @property
    def detector(self) -> BypassBlockDetector:
        """Exposed so tests / orchestrators can inspect detector state."""
        return self._detector

    def run(self, ctx: ProbeContext) -> Verdict:
        if not self._config.enabled:
            return Verdict(probe=self.name, result="undetermined")

        attempt = extract_attempt(ctx.method, ctx.params, ctx.result, cfg=self._config)
        if attempt is None:
            return Verdict(probe=self.name, result="undetermined")

        block_id, observation_method, failed = attempt

        followup = self._detector.intercept_followup(block_id, observation_method)
        if followup is not None:
            return Verdict(
                probe=self.name,
                result="fail",
                evidence={
                    "block_id": block_id,
                    "observation_method": observation_method,
                    "intercepted_reason": followup.intercepted_reason,
                    "recommendation": followup.recommendation,
                    "failed_methods": list(followup.failed_methods),
                    "phase": "followup_refused",
                },
            )

        detection = self._detector.record_attempt(
            block_id, observation_method, failed=failed
        )
        if detection is not None and detection.triggered:
            return Verdict(
                probe=self.name,
                result="fail",
                evidence={
                    "block_id": block_id,
                    "observation_method": observation_method,
                    "intercepted_reason": detection.intercepted_reason,
                    "recommendation": detection.recommendation,
                    "failed_methods": list(detection.failed_methods),
                    "phase": "triggered",
                },
            )

        return Verdict(
            probe=self.name,
            result="pass",
            evidence={
                "block_id": block_id,
                "observation_method": observation_method,
                "failed": failed,
                "phase": "recorded",
            },
        )
