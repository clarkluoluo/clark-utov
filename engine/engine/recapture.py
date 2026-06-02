"""Recapture — turn a provenance gap into a parameterised re-collection request.

The front half of the `recapture` heavy tool (the named T2 placeholder in
``cvd_mount.HEAVY_TOOLS``). Its whole reason to exist is the boundary-dissolving
insight: a "target-specific collection" needs NO target-specific code — the only
target-specific things are the *parameter values* (which PC, which address), and
those come from the caller / the agent / CVD's own diagnosis. The runner already
exposes a fully generic capture primitive (``rerun(input, observe_points)`` —
contracts/runner_interface.md §3.2); this module just produces a valid, prefilled
``observe_points`` list from a #3 provenance verdict and validates it.

What lives here (NO side effects — safe to run with no runner):
  - :class:`RecaptureSpec`     the agent-facing parameter struct.
  - :func:`observe_points_from_provenance`  prefill the capture list from a
                               :class:`engine.oracle_provenance.ProvenanceResult`.
                               NEEDS_OBSERVATION → watch the un-captured native
                               addresses at their reading PC; OPAQUE_CALLEE →
                               capture the call-site argument registers ("its
                               call-time inputs") at the boundary.
  - :func:`plan_recapture`     spec + prefill + validate, ready to hand off.
  - :func:`validate_spec`      pure precondition check → a diagnosis, never a run.
  - :func:`observations_to_snapshots`  fold a runner ``RerunResult`` back into
                               canonical :class:`MemSnapshot` so CVD can re-drive.

What is DEFERRED (the side-effecting T2 back half — see
todo/recapture-squeeze-tool.md): actually dispatching ``adapter.rerun`` behind the
MountPolicy heavy budget gate. :func:`dispatch_recapture` marks that seam.

Generic — no target address, no runner format, no concrete enum is baked in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .aarch64_mem import AddrDecomposition, decompose_addressing
from .oracle_provenance import ProvenanceResult, ProvenanceVerdict
from .runner_client import (
    ObservePoint,
    RegRelWatch,
    RerunResult,
    _regrel_from_wire,
    _truncated_detail_suffix,
)
from .types import Instruction, MemSnapshot

_log = logging.getLogger(__name__)

# AArch64 ABI: x0..x7 carry arguments / return values. A call's INPUTS are these
# registers at the call site. Parameterised — pass arch-specific regs to override.
AARCH64_ARG_REGS: tuple[str, ...] = ("x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7")


# --- agent-facing parameter struct ------------------------------------------

@dataclass
class RecaptureSpec:
    """The squeeze request. ``observe_points`` is usually prefilled from
    provenance (:func:`plan_recapture`); the agent only confirms / edits it and
    supplies the triggering ``input``."""
    input: bytes                                  # input that drives this path (agent)
    window: tuple[int, int] | None = None         # (front_pc, rear_pc) squeeze interval
    focus_pcs: tuple[int, ...] = ()               # call sites / callees to drill (0x221068)
    observe_points: list[ObservePoint] = field(default_factory=list)
    expected_repr: str = "raw"                    # raw|base64|... transform hint (carried)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input.hex(),
            "window": (None if self.window is None
                       else [f"0x{self.window[0]:x}", f"0x{self.window[1]:x}"]),
            "focus_pcs": [f"0x{p:x}" for p in self.focus_pcs],
            "observe_points": [self._observe_point_to_dict(op)
                               for op in self.observe_points],
            "expected_repr": self.expected_repr,
            "note": self.note,
        }

    @staticmethod
    def _observe_point_to_dict(op: ObservePoint) -> dict[str, Any]:
        """Serialize one ObservePoint, carrying the FULL point shape.

        ``mem_regrel`` (reg-relative PC-gated single-point watches) was being
        dropped here even when the ObservePoint carried it — a serialization
        completeness bug that left replay / handoff / auto-resume with an
        incomplete watch shape. We reuse the SAME per-watch wire shape via
        ``runner_client._regrel_to_wire`` (``base_reg/offset/width/pc/kind`` plus
        ``index/scale`` for a register-offset form) and the SAME convention: emit
        ``mem_regrel`` ONLY when non-empty so a plain concrete point serializes
        exactly as before. Symmetry is by construction — one shape, two sinks."""
        pt: dict[str, Any] = {
            "pc": f"0x{op.pc:x}",
            "when": op.when,
            "capture": list(op.capture),
            "regs": list(op.regs),
            "mem": [[f"0x{a:x}", s] for (a, s) in op.mem],
        }
        if op.mem_regrel:
            from .runner_client import _regrel_to_wire
            pt["mem_regrel"] = [_regrel_to_wire(w) for w in op.mem_regrel]
        return pt


# --- engine→runner dynamic-watch prescription (B3 — CONTRACT only) ----------
#
# engine 开处方, runner 抓药. This is the *shape* of the dynamic_watch_batch
# directive ONLY — engine defines/serializes/validates it and (where needed)
# GENERATES it from a provenance verdict. engine does NOT attach observe points,
# does NOT batch-capture MEM, does NOT rerun, and assumes NOTHING about the
# runner's max-watch ceiling (that comes back in the runner's RESPONSE; see
# :func:`warn_if_over_runner_capacity`). All of that lives runner-side, in a
# separate repo. (dev-dynamic-watch-directive-contract-spec.md)

# Fixed legal set for ``snapshot_policy`` (contract §契约要点). ``same_execution_only``
# encodes G1: every snapshot in a batch must come from the SAME execution as that
# batch's ``rr.output`` (no cross-rerun accumulation — a nonce-bearing output would
# otherwise produce a FALSE producer chain). The set is fixed here, not a free
# string: an unknown policy is REJECTED, never silently accepted (no quiet
# downgrade of the nonce-honesty guarantee).
DYNAMIC_WATCH_SNAPSHOT_POLICIES: frozenset[str] = frozenset({"same_execution_only"})

DYNAMIC_WATCH_BATCH_KIND = "dynamic_watch_batch"


@dataclass(frozen=True)
class DynamicWatchBatch:
    """An engine→runner *prescription* for a one-rerun batch of observe points.

    The contract shape clark pinned (dev-dynamic-watch-directive-contract-spec.md):
    a ``kind``-tagged batch of :class:`~engine.runner_client.ObservePoint` (the SAME
    observe-point shape the rerun wire path uses — reused by construction, never a
    second shape) plus three fields that encode G1 (nonce-honesty) INTO the contract
    itself rather than leaving it as a driver-side comment:

      * ``must_capture_output_same_rerun`` — the runner MUST capture this batch's
        ``output`` in the SAME rerun that produced the watch snapshots (one nonce).
      * ``expected_source`` — where the round's ``expected`` is taken from:
        ``"rr.output"`` (the default) = THIS rerun's output.
      * ``snapshot_policy`` — a fixed-set value (``same_execution_only``): no
        snapshot may be carried across reruns.

    This is a ONE-WAY prescription: engine validates the shape it EMITS; it does
    not implement how the runner fulfils it, nor does it know the runner's max-watch
    ceiling (the runner reports that back; :func:`warn_if_over_runner_capacity`
    WARNs LOUD when a response says the batch exceeded capacity — a capability wall
    is reported in shape, never silently truncated).

    Round-trip: :meth:`to_dict` / :meth:`from_dict` are lossless (incl. each
    observe point's ``mem_regrel`` with B6 ``index``/``scale``) so a directive can
    be replayed / handed off / resumed without losing watch shape (A4 lesson: a
    serializer that drops a field breaks the replay chain).
    """
    observe_points: tuple[ObservePoint, ...]
    must_capture_output_same_rerun: bool = True
    expected_source: str = "rr.output"
    snapshot_policy: str = "same_execution_only"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Field validation on the shape engine EMITS (contract §契约要点).

        - ``observe_points`` non-empty (an empty batch captures nothing but the
          endpoint output — a prescription with no watches is a mistake here).
        - ``snapshot_policy`` ∈ the fixed legal set — an unknown value is REJECTED
          (``ValueError``), never silently accepted (would quietly drop the G1
          nonce-honesty guarantee).
        - ``must_capture_output_same_rerun`` must be ``True`` for the only legal
          policy (``same_execution_only`` is meaningless if the output is allowed
          to come from a different rerun than the snapshots — G1 by construction).
        Raises ``ValueError`` on any violation (loud, never a silent downgrade)."""
        if not self.observe_points:
            raise ValueError(
                "dynamic_watch_batch needs at least one observe point — an empty "
                "batch is a prescription with nothing to watch")
        if self.snapshot_policy not in DYNAMIC_WATCH_SNAPSHOT_POLICIES:
            raise ValueError(
                f"snapshot_policy={self.snapshot_policy!r} is not a legal value; "
                f"allowed: {sorted(DYNAMIC_WATCH_SNAPSHOT_POLICIES)} — an unknown "
                "policy is rejected, never silently accepted (it would drop the G1 "
                "same-execution nonce-honesty guarantee)")
        if (self.snapshot_policy == "same_execution_only"
                and not self.must_capture_output_same_rerun):
            raise ValueError(
                "snapshot_policy='same_execution_only' requires "
                "must_capture_output_same_rerun=True — otherwise the batch's output "
                "(its nonce) could come from a different rerun than the snapshots, "
                "breaking G1 by construction")

    def to_dict(self) -> dict[str, Any]:
        """Lossless wire form of the directive. ``observe_points`` reuse the SAME
        per-point shape as the rerun/recapture wire path (via
        :meth:`RecaptureSpec._observe_point_to_dict`) so there is exactly ONE
        observe-point shape across the engine (construct symmetry)."""
        return {
            "kind": DYNAMIC_WATCH_BATCH_KIND,
            "observe_points": [RecaptureSpec._observe_point_to_dict(op)
                               for op in self.observe_points],
            "must_capture_output_same_rerun": self.must_capture_output_same_rerun,
            "expected_source": self.expected_source,
            "snapshot_policy": self.snapshot_policy,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DynamicWatchBatch":
        """Inverse of :meth:`to_dict` — lossless round-trip (incl. each observe
        point's ``mem_regrel`` with ``index``/``scale``). Validates ``kind`` and
        runs full field validation on the reconstructed directive (an inbound
        directive with a bad ``kind`` or illegal ``snapshot_policy`` is rejected at
        the boundary, never silently coerced)."""
        kind = d.get("kind")
        if kind != DYNAMIC_WATCH_BATCH_KIND:
            raise ValueError(
                f"not a {DYNAMIC_WATCH_BATCH_KIND} directive: kind={kind!r}")
        ops = tuple(_observe_point_from_dict(o)
                    for o in d.get("observe_points", []))
        return cls(
            observe_points=ops,
            must_capture_output_same_rerun=bool(
                d.get("must_capture_output_same_rerun", True)),
            expected_source=d.get("expected_source", "rr.output"),
            snapshot_policy=d.get("snapshot_policy", "same_execution_only"),
        )


def _observe_point_from_dict(d: dict[str, Any]) -> ObservePoint:
    """Inverse of :meth:`RecaptureSpec._observe_point_to_dict` — reconstruct one
    :class:`ObservePoint` from its wire form, lossless incl. ``mem``,
    ``mem_regrel`` (with B6 ``index``/``scale`` via
    :func:`runner_client._regrel_from_wire`). ``pc`` accepts the hex string the
    serializer emits OR an int; ``mem`` accepts the ``[addr_hex, size]`` pairs the
    recapture serializer emits."""
    pc = d["pc"]
    mem: list[tuple[int, int]] = []
    for entry in d.get("mem", ()):
        addr, size = entry
        mem.append((int(addr, 16) if isinstance(addr, str) else int(addr),
                    int(size)))
    mem_regrel = tuple(_regrel_from_wire(w) for w in d.get("mem_regrel", ()))
    return ObservePoint(
        pc=int(pc, 16) if isinstance(pc, str) else int(pc),
        when=d.get("when", "before"),
        capture=tuple(d.get("capture", ())),
        regs=tuple(d.get("regs", ())),
        mem=tuple(mem),
        mem_regrel=mem_regrel,
    )


def dynamic_watch_batch_from_spec(
    spec: RecaptureSpec,
    *,
    must_capture_output_same_rerun: bool = True,
    expected_source: str = "rr.output",
    snapshot_policy: str = "same_execution_only",
) -> DynamicWatchBatch:
    """GENERATE a :class:`DynamicWatchBatch` directive from a (provenance-prefilled)
    :class:`RecaptureSpec` — the engine-side "write the prescription" step. The
    observe points come straight from the spec (which provenance already filled in
    via :func:`plan_recapture` / :func:`observe_points_from_provenance`); the G1
    fields default to the only legal same-execution policy. Pure / no side effects
    (no attach, no rerun — that is the runner's job). Validation runs in
    ``__post_init__`` so a malformed prescription fails loudly at generation time."""
    return DynamicWatchBatch(
        observe_points=tuple(spec.observe_points),
        must_capture_output_same_rerun=must_capture_output_same_rerun,
        expected_source=expected_source,
        snapshot_policy=snapshot_policy,
    )


def warn_if_over_runner_capacity(
    directive: DynamicWatchBatch,
    response: dict[str, Any] | None,
) -> bool:
    """Inspect a runner RESPONSE for a "batch exceeded my watch capacity" signal and
    WARN LOUD if so. Returns ``True`` iff the response reported the directive over
    capacity.

    engine does NOT know the runner's max-watch ceiling up front (that is runner
    state); the runner reports it back. A response may carry ``max_watch_points``
    (the ceiling) and/or ``watch_capacity_exceeded`` (an explicit boolean) and/or
    ``accepted_watch_points`` (how many it actually armed). Hitting the wall is a
    capability-墙 condition: we report it in SHAPE (counts + ceiling) and never
    silently let the runner truncate the batch (a partial watch set would yield an
    INCOMPLETE — and silently so — provenance ledger). Best-effort: a ``None`` /
    shapeless response reports nothing (``False``)."""
    if not response:
        return False
    requested = len(directive.observe_points)
    ceiling = response.get("max_watch_points")
    accepted = response.get("accepted_watch_points")
    explicit = bool(response.get("watch_capacity_exceeded", False))
    over = explicit
    if isinstance(ceiling, int) and requested > ceiling:
        over = True
    if isinstance(accepted, int) and accepted < requested:
        over = True
    if over:
        _log.warning(
            "dynamic_watch_batch EXCEEDED runner watch capacity: requested=%d, "
            "runner ceiling=%s, accepted=%s — the runner CANNOT arm the full batch. "
            "The captured ledger would be INCOMPLETE; do NOT consume it as complete "
            "provenance. Split the batch / narrow the watch plan, or raise the "
            "runner ceiling (engine 开处方 but the runner's max-watch is a runner "
            "capability wall — reported in shape, never silently truncated).",
            requested,
            ceiling if ceiling is not None else "(unreported)",
            accepted if accepted is not None else "(unreported)")
    return over


@dataclass(frozen=True)
class UnstableMemPoint:
    """A gap that could NOT be upgraded to a reg-relative watch (B6 tier ③).

    Carries the PC + the (concrete, RUN-LOCAL) addr ONLY as a diagnostic, with an
    explicit ``reason`` — it is an EXPLICIT unstable status, never a silent watch.
    The concrete addr here MUST NOT be reused as a cross-run watch (invariant 1):
    consumers read ``reason`` and either widen the trace or skip the point."""
    pc: int
    addr: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"pc": f"0x{self.pc:x}", "addr": f"0x{self.addr:x}",
                "status": "UNSTABLE_CONCRETE", "reason": self.reason}


@dataclass(frozen=True)
class RegSnapshotDirective:
    """A gap whose addressing form is decomposable but whose addressing register
    value(s) are MISSING at this PC in the current trace (B6 tier ②).

    The fix is to recapture a register snapshot at ``pc`` (the ``needed_regs``),
    then re-decompose → ①. This rides the SAME recapture mechanism (B2): it is a
    register-observe ObservePoint, surfaced here so the caller arms it without
    manual intervention. Never a bare concrete mem watch."""
    pc: int
    addr: int
    needed_regs: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"pc": f"0x{self.pc:x}", "addr": f"0x{self.addr:x}",
                "status": "NEEDS_REG_SNAPSHOT",
                "needed_regs": list(self.needed_regs), "reason": self.reason}


@dataclass(frozen=True)
class PrefillResult:
    observe_points: list[ObservePoint]
    unplaceable_addrs: tuple[int, ...] = ()   # gaps with no reading PC → cannot hook
    # B6 three-tier classification of NEEDS_OBSERVATION gaps (only populated when
    # ``items`` is supplied so the gap PC's mnemonic + reg row can be inspected):
    #   ② NEEDS_REG_SNAPSHOT directives — re-capture regs at the PC, then upgrade.
    #   ③ UNSTABLE_CONCRETE points — explicit, never silently emitted as a watch.
    reg_snapshot_directives: tuple[RegSnapshotDirective, ...] = ()
    unstable_points: tuple[UnstableMemPoint, ...] = ()

    def assert_no_bare_concrete(self) -> None:
        """Global invariant (A8④ / invariant 5): no observe point may carry a
        BARE concrete ``mem`` range — every memory point must be either a
        reg-relative ``mem_regrel`` watch, a register-snapshot directive, or an
        explicit ``UNSTABLE_CONCRETE`` status. Raises ``AssertionError`` if any
        ObservePoint still leaks a concrete ``mem`` (the false-closure shape
        clark forbids). Call this on the OUTPUT of a B6 (items-supplied) prefill."""
        for op in self.observe_points:
            assert not op.mem, (
                f"observe plan leaks a BARE concrete mem watch at pc=0x{op.pc:x} "
                f"({op.mem!r}) — invariant 5: every mem point must be mem_regrel, "
                "a reg-snapshot directive, or an explicit UNSTABLE_CONCRETE status")


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    findings: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "findings": self.findings}


@dataclass(frozen=True)
class RecapturePlan:
    spec: RecaptureSpec
    validation: ValidationReport | None
    unplaceable_addrs: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"spec": self.spec.to_dict(),
                "validation": None if self.validation is None else self.validation.to_dict(),
                "unplaceable_addrs": [f"0x{a:x}" for a in self.unplaceable_addrs]}


# --- prefill: provenance verdict -> observe_points --------------------------

def _coalesce(addrs: list[int], min_size: int = 1) -> list[tuple[int, int]]:
    """Sorted unique addresses -> list of (start, length) contiguous runs, each
    padded up to ``min_size``."""
    runs: list[list[int]] = []
    for a in addrs:
        if runs and a == runs[-1][0] + runs[-1][1]:
            runs[-1][1] += 1
        else:
            runs.append([a, 1])
    return [(a, max(n, min_size)) for a, n in runs]


def observe_points_from_provenance(
    prov: ProvenanceResult,
    *,
    items: list[Instruction] | None = None,
    arg_regs: tuple[str, ...] = AARCH64_ARG_REGS,
    callee_capture: tuple[str, ...] = ("regs", "mem"),
    min_size: int = 1,
) -> PrefillResult:
    """Map a provenance verdict to a prefilled capture list.

    NEEDS_OBSERVATION  -> per-gap. **B6 reg-relative upgrade** (when ``items`` is
                          supplied so the gap PC's mnemonic + reg row are visible):
                          each gap is classified into three tiers
                          (:func:`engine.aarch64_mem.decompose_addressing`) —
                          ① ``REGREL_UPGRADED``  → a ``mem_regrel`` reg-relative
                            point watch (``base_reg+offset[,index,scale]``), with
                            NO concrete addr (the runner resolves the live EA at
                            hook time; the stale run-local addr never crosses runs
                            — invariant 1/2);
                          ② ``NEEDS_REG_SNAPSHOT`` → a register-observe directive
                            (capture the addressing regs at this PC, then
                            re-decompose) — recorded on ``reg_snapshot_directives``
                            AND armed as a ``regs`` ObservePoint (rides the B2
                            recapture mechanism, no caller hand-work);
                          ③ ``UNSTABLE_CONCRETE`` → recorded on ``unstable_points``
                            with an explicit reason; NEVER emitted as a bare watch.
                          A gap with no reading PC is ``unplaceable`` (can't hook).
                          With ``items`` supplied, the OUTPUT carries NO bare
                          concrete ``mem`` watch (invariant 5 / A8④ —
                          :meth:`PrefillResult.assert_no_bare_concrete`).

                          **Legacy path** (``items`` omitted): unchanged — the
                          gap addresses are coalesced into concrete ``mem``
                          ObservePoints at their reading PC. Used by the WITHIN-run
                          iterative recapture loop (B2), where the address is
                          watched in the SAME execution (not a cross-run watch);
                          additive / byte-for-byte (invariant 7).
    OPAQUE_CALLEE      -> for each boundary PC, a ``regs`` ObservePoint capturing
                          the ABI argument registers BEFORE the call — i.e. the
                          callee's call-time inputs ("其调用前输入").
    Other verdicts     -> no points (production already visible; nothing to add).
    """
    pts: list[ObservePoint] = []
    unplaceable: list[int] = []
    directives: list[RegSnapshotDirective] = []
    unstable: list[UnstableMemPoint] = []

    if prov.verdict is ProvenanceVerdict.NEEDS_OBSERVATION:
        if items is None:
            # Legacy within-run path (B2 loop): coalesce gaps into concrete mem
            # points at their reading PC. NOT a cross-run watch.
            by_pc: dict[int, set[int]] = {}
            for w in prov.next_watch:
                addr = int(w["addr"], 16)
                pc = w.get("pc")
                if pc is None:
                    unplaceable.append(addr)
                    continue
                by_pc.setdefault(int(pc, 16), set()).add(addr)
            for pc, addrs in sorted(by_pc.items()):
                mem = tuple(_coalesce(sorted(addrs), min_size))
                pts.append(ObservePoint(pc=pc, when="before", capture=("mem",), mem=mem))
        else:
            # B6 reg-relative upgrade: classify each gap against its PC's
            # mnemonic + same-run reg row. A multi-byte access produces one gap
            # PER BYTE at the SAME PC (addr, addr+1, …); they belong to ONE
            # access, so coalesce contiguous bytes per PC into access windows and
            # decompose the window's BASE address (its EA) — never byte-by-byte
            # (which would mis-decompose addr+1 as a different base+offset).
            by_idx_pc: dict[int, "Instruction"] = {}
            for ins in items:
                by_idx_pc.setdefault(ins.pc, ins)
            gaps_by_pc: dict[int, list[int]] = {}
            for w in prov.next_watch:
                addr = int(w["addr"], 16)
                pc = w.get("pc")
                if pc is None:
                    unplaceable.append(addr)
                    continue
                gaps_by_pc.setdefault(int(pc, 16), []).append(addr)
            # ① collect per-PC reg-relative watches; ② directives; ③ unstable.
            regrel_by_pc: dict[int, list[RegRelWatch]] = {}
            snap_regs_by_pc: dict[int, set[str]] = {}
            for pc, addrs in sorted(gaps_by_pc.items()):
                ins = by_idx_pc.get(pc)
                # one access window = the start of each contiguous byte-run.
                windows = _coalesce(sorted(set(addrs)), min_size)
                if ins is None:
                    # PC named by the gap is not in the trace items — we cannot
                    # inspect its mnemonic/regs. Treat as needing a snapshot
                    # there (decomposable only once observed), explicit not bare.
                    for (a, _n) in windows:
                        directives.append(RegSnapshotDirective(
                            pc=pc, addr=a, needed_regs=(),
                            reason="gap PC not present in supplied trace items — "
                                   "capture its instruction + register state, then re-decompose"))
                    snap_regs_by_pc.setdefault(pc, set())
                    continue
                for (addr, _n) in windows:
                    dec = decompose_addressing(ins.mnemonic, addr, ins.regs_read)
                    if dec.verdict is AddrDecomposition.REGREL_UPGRADED:
                        # base+offset, or the register-offset form (index live)
                        # carried structurally as base+(index<<scale)+offset — the
                        # runner adds index_val<<scale at hook time. No addr leaks.
                        regrel_by_pc.setdefault(pc, []).append(RegRelWatch(
                            base_reg=dec.base_reg, offset=dec.offset,
                            width=dec.width, pc=pc, kind="read",
                            index=dec.index, scale=dec.scale))
                    elif dec.verdict is AddrDecomposition.NEEDS_REG_SNAPSHOT:
                        directives.append(RegSnapshotDirective(
                            pc=pc, addr=addr, needed_regs=dec.needed_regs,
                            reason=dec.reason))
                        snap_regs_by_pc.setdefault(pc, set()).update(dec.needed_regs)
                    else:
                        unstable.append(UnstableMemPoint(pc=pc, addr=addr, reason=dec.reason))
            # Emit ① reg-relative ObservePoints (no concrete mem).
            for pc, watches in sorted(regrel_by_pc.items()):
                pts.append(ObservePoint(
                    pc=pc, when="before", capture=("mem",),
                    mem_regrel=tuple(watches)))
            # Emit ② register-observe directives as real ObservePoints so the B2
            # recapture mechanism arms them without caller hand-work.
            for pc, regs in sorted(snap_regs_by_pc.items()):
                if not regs:
                    continue
                pts.append(ObservePoint(
                    pc=pc, when="before", capture=("regs",),
                    regs=tuple(sorted(regs))))

    elif prov.verdict is ProvenanceVerdict.OPAQUE_CALLEE:
        for pc in prov.boundary_pcs:
            pts.append(ObservePoint(pc=pc, when="before",
                                    capture=tuple(callee_capture), regs=tuple(arg_regs)))

    return PrefillResult(
        observe_points=pts,
        unplaceable_addrs=tuple(sorted(set(unplaceable))),
        reg_snapshot_directives=tuple(directives),
        unstable_points=tuple(unstable))


def plan_recapture(
    prov: ProvenanceResult,
    input_bytes: bytes,
    *,
    window: tuple[int, int] | None = None,
    focus_pcs: tuple[int, ...] = (),
    expected_repr: str = "raw",
    items: list[Instruction] | None = None,
    arg_regs: tuple[str, ...] = AARCH64_ARG_REGS,
    callee_capture: tuple[str, ...] = ("regs", "mem"),
    min_size: int = 1,
) -> RecapturePlan:
    """Build a validated, prefilled :class:`RecapturePlan` from a provenance gap.

    ``focus_pcs`` defaults to the verdict's boundary + callee targets (the call to
    drill). Pass ``items`` to also run :func:`validate_spec`; omit it to just
    build the spec. NO side effects either way.
    """
    pre = observe_points_from_provenance(
        prov, arg_regs=arg_regs, callee_capture=callee_capture, min_size=min_size)
    fp = tuple(focus_pcs) or (prov.boundary_pcs + prov.callee_targets)
    spec = RecaptureSpec(
        input=bytes(input_bytes), window=window, focus_pcs=fp,
        observe_points=list(pre.observe_points), expected_repr=expected_repr,
        note=prov.detail)
    report = validate_spec(spec, items) if items is not None else None
    return RecapturePlan(spec=spec, validation=report,
                         unplaceable_addrs=pre.unplaceable_addrs)


# --- validation: a diagnosis, never a run -----------------------------------

def _finding(severity: str, code: str, msg: str) -> dict:
    return {"severity": severity, "code": code, "detail": msg}


def validate_spec(spec: RecaptureSpec, items: list[Instruction]) -> ValidationReport:
    """Pure precondition check (CVD_MOUNT_POLICY: garbage params get a diagnosis,
    not a wasted rerun). Returns errors (block dispatch) + warnings (proceed)."""
    findings: list[dict] = []
    pcset = {ins.pc for ins in items}

    if not spec.input:
        findings.append(_finding("error", "no_input",
                                 "spec.input is empty — rerun needs the triggering input"))

    if spec.window is not None:
        lo, hi = spec.window
        if lo >= hi:
            findings.append(_finding("error", "bad_window",
                                     f"front 0x{lo:x} >= rear 0x{hi:x}"))

    if not spec.observe_points:
        findings.append(_finding("warning", "no_observe_points",
                                 "no observation points — rerun would capture only "
                                 "the endpoint output"))

    for op in spec.observe_points:
        if op.pc not in pcset:
            findings.append(_finding("error", "pc_not_in_trace",
                                     f"observe pc 0x{op.pc:x} not seen in the trace"))
        if spec.window is not None:
            lo, hi = spec.window
            if not (lo <= op.pc <= hi):
                findings.append(_finding("warning", "pc_outside_window",
                                         f"observe pc 0x{op.pc:x} outside the squeeze "
                                         f"window [0x{lo:x}, 0x{hi:x}]"))
        if op.when not in ("before", "after"):
            findings.append(_finding("error", "bad_when",
                                     f"observe pc 0x{op.pc:x}: when={op.when!r} "
                                     f"(must be 'before' or 'after')"))
        for (addr, size) in op.mem:
            if size <= 0:
                findings.append(_finding("error", "bad_mem_size",
                                         f"mem capture at 0x{addr:x} has size {size}"))

    for pc in spec.focus_pcs:
        if pc not in pcset:
            # OPAQUE_CALLEE's callee target is, by definition, NOT in the trace —
            # that is exactly why we recapture. A warning, never an error.
            findings.append(_finding("warning", "focus_not_in_trace",
                                     f"focus 0x{pc:x} not in trace (expected for an "
                                     f"un-traced OPAQUE_CALLEE target)"))

    ok = not any(f["severity"] == "error" for f in findings)
    return ValidationReport(ok=ok, findings=findings)


# --- ingest: runner RerunResult -> canonical snapshots ----------------------

def observations_to_snapshots(result: RerunResult) -> list[MemSnapshot]:
    """Fold a runner ``RerunResult`` into canonical :class:`MemSnapshot` so CVD /
    #3 provenance can re-drive on the narrowed window. Pure transform — the loop's
    ingest half, usable the moment a real (or fake) adapter hands back a result.

    Truncation propagation (construct-symmetry): when the runner hit a record cap
    (``result.truncated``), WARN to the top-level logger and stamp every derived
    snapshot ``truncated=True`` so the re-driven CVD / provenance treats this
    narrowed window as INCOMPLETE, not complete/clean provenance
    (contracts/runner_interface.md §rerun)."""
    if result.truncated:
        _log.warning(
            "recapture observations TRUNCATED: the runner hit a record cap%s — the "
            "re-driven window is INCOMPLETE and MUST NOT be consumed as complete/"
            "clean provenance. Derived snapshots are stamped truncated=True "
            "(contracts/runner_interface.md §rerun).",
            _truncated_detail_suffix(result.truncated_detail))
    snaps: list[MemSnapshot] = []
    for obs in result.observations:
        for addr, data in obs.mem.items():
            if not data:
                continue
            snaps.append(MemSnapshot(
                addr=addr, data=bytes(data),
                label=f"recapture@0x{obs.pc:x}:{obs.when}", source="recapture",
                truncated=bool(result.truncated)))
    return snaps


# --- dispatch seam (DEFERRED — the side-effecting T2 back half) --------------

def dispatch_recapture(plan: RecapturePlan, adapter, *, policy=None):
    """The back half is deliberately not built here. It runs the target
    (side-effecting, budget-gated) and must sit behind the MountPolicy heavy gate.

    To complete the loop later:
        result = adapter.rerun(plan.spec.input, list(plan.spec.observe_points))
        snaps  = observations_to_snapshots(result)
        # fold snaps into CvdState and re-drive CVD on the narrowed window.

    See todo/recapture-squeeze-tool.md (§back-half) and cvd_mount.HEAVY_TOOLS.
    """
    raise NotImplementedError(
        "recapture dispatch is the deferred T2 back-half (heavy, side-effecting); "
        "only the front half — prefill + validate + ingest — is live. "
        "See todo/recapture-squeeze-tool.md.")
