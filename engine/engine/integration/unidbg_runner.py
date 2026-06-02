"""UnidbgSignRunner: a Hypotask `Runner` over utov's existing unidbg adapter.

Hypotask defines the neutral-workbench `Runner` abstraction
(`hypotask.runner.interface.Runner`) but leaves the implementation to the
application. This wraps utov's existing `engine.runner_client.RunnerAdapter`
(the JSON-RPC unidbg bridge) so a Hypotask task can declare it via
`uses_runner` + `runner_capabilities_required` and the engine can keep driving
the real RE work through the same adapter.

The runner is a neutral workbench: it does NOT know about tasks, goals, or
done_criteria. It only answers capability invocations. The capability names map
onto `hypotask.runner.interface.RunnerCapability` constants; the actual work is
delegated to the underlying `RunnerAdapter` methods.
"""

from __future__ import annotations

import logging
from typing import Any

from hypotask.runner.interface import Runner, RunnerCapability

from ..runner_client import (
    ObservePoint,
    RegRelWatch,
    RunnerAdapter,
    mem_snapshots_from_rerun,
)

_log = logging.getLogger(__name__)


from dataclasses import dataclass, field  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    from ..runner_client import RerunResult
    from ..types import MemSnapshot


@dataclass(frozen=True)
class MemRegionWatchResult:
    """The MEMREGION_WATCH invocation result — the raw rerun PLUS the canonical
    :class:`MemSnapshot` objects the engine's oracle_sink consumes (task 6).

    Carries BOTH so a caller wanting the raw output still has ``rerun``, while the
    sink-facing path reads ``mem_snapshots`` directly (no per-caller re-conversion).
    ``mem_snapshots`` is empty when the adapter captured no mem at the observe point
    (the WARN already fired) — never fabricated."""

    rerun: "RerunResult"
    mem_snapshots: tuple["MemSnapshot", ...] = field(default_factory=tuple)


class UnidbgSignRunner(Runner):
    """Hypotask Runner backed by a utov `RunnerAdapter` (unidbg JSON-RPC).

    Construct with an already-built adapter (e.g. a `SubprocessRunnerAdapter`
    talking to the Java unidbg runner, or a `NullRunnerAdapter` for File-mode
    static-trace samples). Capability set is derived from what the adapter
    actually implements, so File-mode adapters advertise a reduced set.
    """

    name = "unidbg_signrunner"

    # Map Hypotask capability constants → the adapter method that fulfils them.
    # An adapter "supports" a capability iff it overrides the corresponding
    # RunnerAdapter method (File-mode adapters leave them raising
    # NotImplementedError).
    _CAPABILITY_METHODS = {
        RunnerCapability.TRACE: "get_trace",
        RunnerCapability.RE_EXECUTE: "rerun",
        RunnerCapability.MEMREGION_WATCH: "rerun",        # via observe_points mem
        RunnerCapability.PHASE_INSTALL: "code_hook_range",
    }

    def __init__(self, adapter: RunnerAdapter, name: str | None = None):
        self._adapter = adapter
        if name:
            self.name = name

    def capabilities(self) -> list[str]:
        """Capabilities this runner actually supports.

        A capability is advertised only if the backing adapter overrides the
        method that fulfils it — so a File-mode (static-trace) adapter that
        only implements `metadata()` advertises nothing here, and a Live-mode
        adapter advertises trace/re-execute/etc.
        """
        caps: list[str] = []
        for cap, method_name in self._CAPABILITY_METHODS.items():
            if self._adapter_overrides(method_name):
                if cap not in caps:
                    caps.append(cap)
        return caps

    def invoke(self, capability: str, **kwargs) -> Any:
        """Execute one capability invocation, delegating to the adapter."""
        if capability == RunnerCapability.TRACE:
            return self._adapter.get_trace(
                kwargs["input_bytes"], kwargs["start"], kwargs["end"]
            )
        if capability == RunnerCapability.RE_EXECUTE:
            return self._adapter.rerun(
                kwargs["input_bytes"], kwargs.get("observe_points")
            )
        if capability == RunnerCapability.MEMREGION_WATCH:
            point, desc = self._memregion_watch_point(kwargs)
            result = self._adapter.rerun(kwargs["input_bytes"], [point])
            # capture/mem → MemSnapshot end-to-end (task 6): this capability DECLARES
            # mem-capture (it asked the adapter to observe a mem region), so it must
            # CARRY the captured region forward as canonical MemSnapshots the engine's
            # oracle_sink consumes — not just hand back a raw RerunResult the sink
            # cannot read. Attach the converted snapshots; when the adapter claimed
            # mem-capture but produced NONE, WARN (construct-symmetry: the capability
            # advertised mem-capture, so a silent empty is a degradation that must be
            # surfaced, not a silent write/read fallback at the sink).
            snapshots = mem_snapshots_from_rerun(result)
            if not snapshots:
                _log.warning(
                    "MEMREGION_WATCH invoked (%s) but the adapter produced NO mem "
                    "snapshots — the capability advertised mem-capture yet the "
                    "observe point captured nothing. oracle_sink will fall back to "
                    "write/read location (located_via != 'snapshot'); check the "
                    "adapter actually captures mem at observe points "
                    "(runner_interface §3.7 MemSnapshot).",
                    desc)
            return MemRegionWatchResult(rerun=result, mem_snapshots=tuple(snapshots))
        if capability == RunnerCapability.PHASE_INSTALL:
            return self._adapter.code_hook_range(
                kwargs["input_bytes"], kwargs["hooks"]
            )
        raise NotImplementedError(f"unidbg_signrunner does not handle '{capability}'")

    @staticmethod
    def _memregion_watch_point(kwargs: dict) -> tuple[ObservePoint, str]:
        """Build the ObservePoint for a MEMREGION_WATCH invocation + a log desc.

        Two MUTUALLY-EXCLUSIVE addressing forms (the address is either known up
        front, or only computable from a live register at a PC):

        * **reg-relative** (``base_reg`` given) — a PC-gated single-point watch
          ``[base_reg+offset]@pc`` of ``width`` bytes (contracts §3.2.1). The
          runner resolves the address from the live register value at ``pc``;
          no wide-region scan, so no noise / cap hazard. Carried on
          ``ObservePoint.mem_regrel``. ``offset`` defaults 0, ``width`` 8,
          ``kind`` ``"read"`` — and ``pc`` defaults to the watch ``pc`` (the arm
          site) when not separately given.
        * **concrete** (``base``/``size`` given) — the original fixed
          ``(addr,size)`` range on ``ObservePoint.mem``.

        Refuses an ambiguous mix (both forms) rather than silently picking one
        — symmetry/correctness is a construct guarantee, not a caller habit."""
        has_regrel = "base_reg" in kwargs
        has_concrete = "base" in kwargs or "size" in kwargs
        if has_regrel and has_concrete:
            raise ValueError(
                "MEMREGION_WATCH given BOTH reg-relative (base_reg) and concrete "
                "(base/size) addressing — these are mutually exclusive; pass one "
                "form (contracts/runner_interface.md §3.2 / §3.2.1)."
            )
        when = kwargs.get("when", "after")
        if has_regrel:
            base_reg = kwargs["base_reg"]
            offset = int(kwargs.get("offset", 0))
            width = int(kwargs.get("width", 8))
            kind = kwargs.get("kind", "read")
            # The arm/access PC: an explicit ``point_watch_pc`` wins, else the
            # watch ``pc`` (the directive's arm site) is reused.
            pc = int(kwargs.get("point_watch_pc", kwargs["pc"]))
            watch = RegRelWatch(
                base_reg=base_reg, offset=offset, width=width, pc=pc, kind=kind
            )
            point = ObservePoint(
                pc=int(kwargs["pc"]),
                when=when,
                capture=("mem",),
                mem_regrel=(watch,),
            )
            sign = "+" if offset >= 0 else "-"
            desc = (f"reg-relative [{base_reg} {sign} 0x{abs(offset):x}]@0x{pc:x} "
                    f"width={width} kind={kind}")
            return point, desc
        # Concrete (original) form.
        base = int(kwargs["base"])
        size = int(kwargs["size"])
        point = ObservePoint(
            pc=int(kwargs["pc"]),
            when=when,
            capture=("mem",),
            mem=((base, size),),
        )
        desc = f"concrete pc=0x{int(kwargs['pc']):x} base=0x{base:x} size={size}"
        return point, desc

    @property
    def adapter(self) -> RunnerAdapter:
        """The underlying utov RunnerAdapter (engine drives RE work through it)."""
        return self._adapter

    def _adapter_overrides(self, method_name: str) -> bool:
        """True iff the adapter's concrete class overrides RunnerAdapter.method.

        RunnerAdapter's File-mode default raises NotImplementedError; an adapter
        that overrides the method is declaring real support. We also honour an
        explicit static `CAPABILITIES` frozenset on the adapter when present.
        """
        base_method = getattr(RunnerAdapter, method_name, None)
        cls_method = getattr(type(self._adapter), method_name, None)
        return cls_method is not None and cls_method is not base_method
