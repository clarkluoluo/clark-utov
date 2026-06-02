"""Watch-first-write probe adapter (PLAN §19 / IMPL_PLAN §P1.0 step 4).

Wraps :mod:`engine.watch_first_write`. The auto-suggestion rule —
"value observed at a concrete landing address with no verified
closed-form recompute → recommend a memory watchpoint to capture the
producing PC" — generalises beyond VMP algorithm extraction (key
extraction and any observation-based investigation benefits identically),
so this lives in the base profile as ``mechanism: true``.

The probe is advisory: it never fails a call. It emits zero or more
suggestions, each carrying either an installable spec (when
``UTOV_WATCH_FIRST_WRITE_AUTO_TRIGGER`` is on) or just an advisory
string the wrapper surfaces to the agent.

Mapping → :class:`Verdict`:

  * ``no eligible records``       → ``undetermined``
  * ``≥ 1 suggestion``            → ``pass`` (advisory; no cap, no fail)
"""

from __future__ import annotations

from engine.profile.probe_runtime import (
    Probe,
    ProbeContext,
    Verdict,
    register_builtin_probe,
)
from engine.watch_first_write import WatchFirstWriteConfig, suggest_watches_in_params


@register_builtin_probe("watch_first_write")
class WatchFirstWriteProbe(Probe):
    """Mechanism probe: suggest memory watchpoints for observed values
    whose producer isn't yet known (no closed-form recompute available).
    Pure advisory — the probe doesn't cap evidence or fail calls; it
    proposes a next observation step.
    """

    name = "watch_first_write"
    mechanism = True
    inputs = ("method", "params")
    outputs = ("watch_first_write",)

    def __init__(self, config: WatchFirstWriteConfig | None = None) -> None:
        self._config = config

    def run(self, ctx: ProbeContext) -> Verdict:
        suggestions = suggest_watches_in_params(ctx.params, cfg=self._config)
        if not suggestions:
            return Verdict(probe=self.name, result="undetermined")

        return Verdict(
            probe=self.name,
            result="pass",
            evidence={
                "suggestions": [s.to_dict() for s in suggestions],
                "count": len(suggestions),
            },
        )
