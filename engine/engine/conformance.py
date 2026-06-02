"""Runner conformance test — mandatory pre-flight gate (PLAN §17, DECISIONS D-026).

Run this BEFORE entering the main analysis pipeline. Failure → main pipeline
refuses to start. A clean runner protects the verifier from environment-side
pollution that would otherwise pass through silently and poison every finding.

Checks:
  C1 DETERMINISM        same input × N reruns → bit-identical output
  C2 INPUT_SENSITIVITY  flipped-byte inputs produce at least some different outputs
  C3 OBSERVE_POINT      observation at algo_entry_pc returns non-empty state
                        (regs round-trip + MEM round-trip when mem-capture declared)
  C4 TRACE_INTEGRITY    trace start/end PCs match metadata anchors
  C5 CROSS_CALL_INDEP   get_trace line-count holds after intervening reruns
  C6 OBSERVER_NEUTRAL   turning an observer set ON does not perturb observed I/O
  C7 CAPABILITY_RT      every DECLARED capability round-trips request→response
                        (declared-but-not-wired = LOUD FAIL; the Bug1 class)

Mode handling:
  - Live   (full 3-method runner): run all applicable checks
  - File   (static trace only):    skip rerun-based checks, run C4 only; set verifier_degraded
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

from .runner_client import (
    ObservePoint,
    RunnerAdapter,
    TraceReader,
)
from .types import TargetMeta

# --- thresholds (DECISIONS D-026, tunable) ---
DETERMINISM_TRIALS = 5
SENSITIVITY_BYTE_FLIPS = 3
SENSITIVITY_MIN_DIFFERENT = 2
RERUN_WARN_SECONDS = 1.0

# C5 cross-call independence thresholds
INDEPENDENCE_MIN_BASELINE_LINES = 10        # below this we can't tell — SKIP
INDEPENDENCE_FAIL_RATIO = 0.5               # second-trace < 50% baseline → FAIL
INDEPENDENCE_WARN_RATIO = 0.9               # < 90% but >= 50% → PASS with warning

# Capability name a runner declares (TargetMeta.capabilities or an adapter's
# static CAPABILITIES) when it can fulfil rerun() observe_points. C3 uses it to
# tell "PC was wrong" (now reconciled by relocation, see below) apart from
# "runner simply has no observe ability" (a capability gap → degrade, not FAIL).
OBSERVE_CAPABILITY = "observe_point"

# Capability a runner declares when its rerun() observe points can capture
# MEMORY ranges (not just registers). C3-mem (mem round-trip) and the generic
# C7 capability round-trip use it: a runner that does NOT declare it degrades
# (SKIP) rather than FAILs; one that DOES declare it yet returns no requested
# bytes is a real broken mem-capture hook (FAIL) — the recurring "contract said
# it, wire never wired it" class (Bug1 was exactly that drop).
MEM_CAPTURE_CAPABILITY = "mem_capture"

# Generic capability round-trip registry (C7). Each entry probes ONE declared
# capability by issuing a request that exercises it and asserting the response
# carries the requested thing back. Maps capability name → probe fn.
#   probe(runner, meta) -> (ok: bool, detail: dict)
# A capability the runner does NOT declare is skipped for that runner (not its
# job to fulfil); a DECLARED capability whose probe round-trip fails is a LOUD
# FAIL — never a silent empty. Extend by adding (capability, probe) pairs.
def _probe_observe_point(runner: "RunnerAdapter", meta: "TargetMeta") -> tuple[bool, dict]:
    op = ObservePoint(pc=meta.algo_entry_pc, when="before", capture=("regs",))
    r = runner.rerun(b"\x00" * (meta.input_length or 8), observe_points=[op])
    ok = bool(r.observations) and bool(r.observations[0].regs)
    return ok, {"n_observations": len(r.observations),
                "regs_captured": len(r.observations[0].regs) if r.observations else 0}


def _probe_mem_capture(runner: "RunnerAdapter", meta: "TargetMeta") -> tuple[bool, dict]:
    ok, detail, _ = _mem_round_trip_probe(runner, meta)
    return ok, detail


# capability name → (request-side probe, human label). Registry, extend freely.
CAPABILITY_PROBES: dict[str, Any] = {
    OBSERVE_CAPABILITY: _probe_observe_point,
    MEM_CAPTURE_CAPABILITY: _probe_mem_capture,
}

# Module relocation: anchors are emitted by the runner's static metadata as
# absolute PCs, but unidbg (and any loader) rebases the module to a run-specific
# base. A clean rebase shifts EVERY anchor by ONE page-aligned base delta. We
# detect that, reconcile anchors to the run's actual base, and proceed — instead
# of FAILing C4 on a difference that is just relocation (the recurring class the
# red line forbids hacking the runner for). Page granularity = 4 KiB.
_PAGE_BITS = 12
_PAGE_MASK = (1 << _PAGE_BITS) - 1


@dataclass(frozen=True)
class Relocation:
    """A detected clean module rebase: one page-aligned ``base_delta`` that maps
    BOTH the entry and exit metadata anchors onto the trace's observed PCs.

    The two-anchor cross-validation is the anti-masking guard: only when a single
    delta lines up entry AND exit (and the page offsets are preserved) do we trust
    it as a whole-module rebase. A mismatch means a genuinely wrong anchor — that
    must still FAIL, never be reconciled away."""

    base_delta: int
    entry_from:  int
    entry_to:    int
    exit_from:   int
    exit_to:     int

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_delta": f"0x{self.base_delta:x}",
            "entry": {"from": f"0x{self.entry_from:x}", "to": f"0x{self.entry_to:x}"},
            "exit":  {"from": f"0x{self.exit_from:x}",  "to": f"0x{self.exit_to:x}"},
        }


def detect_relocation(
    meta: TargetMeta, first_pc: int | None, last_pc: int | None,
) -> Relocation | None:
    """Return a :class:`Relocation` iff the trace bounds are a clean page-aligned
    rebase of the metadata anchors, else None.

    Clean rebase ⇔ (1) ``entry_delta == exit_delta`` and non-zero — ONE base delta
    explains both anchors (the cross-validation), and (2) the page offset of each
    anchor is preserved (``anchor & PAGE_MASK == observed & PAGE_MASK``), i.e. the
    delta is page-aligned. If entry/exit need different deltas, or a page offset
    shifts, it is NOT a clean relocation → return None so C4 still FAILs (a real
    wrong anchor must not be reconciled away)."""
    if first_pc is None or last_pc is None:
        return None
    entry_delta = first_pc - meta.algo_entry_pc
    exit_delta = last_pc - meta.algo_exit_pc
    if entry_delta == 0:                       # no shift → nothing to reconcile
        return None
    if entry_delta != exit_delta:              # anchors disagree → not a rebase
        return None
    if (meta.algo_entry_pc & _PAGE_MASK) != (first_pc & _PAGE_MASK):
        return None
    if (meta.algo_exit_pc & _PAGE_MASK) != (last_pc & _PAGE_MASK):
        return None
    return Relocation(
        base_delta=entry_delta,
        entry_from=meta.algo_entry_pc, entry_to=first_pc,
        exit_from=meta.algo_exit_pc, exit_to=last_pc,
    )


def rebase_pc(pc: int, reloc: Relocation | None) -> int:
    """Apply a detected relocation's base delta to any module-relative PC.

    The single primitive every downstream consumer (C3 observe PC, drive window
    anchor, ``manifest.hook_rvas``-resolved PCs) uses to rebase against the run's
    actual base — so reconcile happens ONCE and everyone reads the same value."""
    return pc + reloc.base_delta if reloc is not None else pc


def rebase_meta(meta: TargetMeta, reloc: Relocation | None) -> TargetMeta:
    """Return ``meta`` with its entry/exit anchors rebased by ``reloc`` (or as-is
    when there is no relocation — idempotent)."""
    if reloc is None:
        return meta
    return replace(meta, algo_entry_pc=reloc.entry_to, algo_exit_pc=reloc.exit_to)


class CheckId(str, Enum):
    C1 = "C1_DETERMINISM"
    C2 = "C2_INPUT_SENSITIVITY"
    C3 = "C3_OBSERVE_POINT"
    C4 = "C4_TRACE_INTEGRITY"
    C5 = "C5_CROSS_CALL_INDEPENDENCE"
    C6 = "C6_OBSERVER_NEUTRALITY"
    C7 = "C7_CAPABILITY_ROUND_TRIP"


class CheckResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class CheckRecord:
    check: CheckId
    result: CheckResult
    detail: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0


@dataclass
class ConformanceReport:
    target_name: str
    mode: str                                          # "live" | "file"
    overall: CheckResult                               # PASS | FAIL
    verifier_degraded: bool
    checks: list[CheckRecord] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""
    # LOUD, non-silent record of a reconciled module rebase (None when no
    # relocation). C4 PASSes as relocation-reconciled, not by pretending the
    # anchors matched — this field is where the verdict says so.
    relocation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["checks"] = [
            {"check": c.check.value, "result": c.result.value, "detail": c.detail, "duration_s": c.duration_s}
            for c in self.checks
        ]
        d["overall"] = self.overall.value
        return d


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# --- individual checks ---

def _c1_determinism(runner: RunnerAdapter, probe_input: bytes) -> CheckRecord:
    t0 = time.monotonic()
    outputs: list[bytes] = []
    try:
        for _ in range(DETERMINISM_TRIALS):
            r = runner.rerun(probe_input, observe_points=[])
            outputs.append(r.output)
    except NotImplementedError:
        return CheckRecord(CheckId.C1, CheckResult.SKIP, {"reason": "rerun not implemented (File mode)"}, time.monotonic() - t0)
    unique = {hashlib.sha256(o).hexdigest()[:16] for o in outputs}
    duration = time.monotonic() - t0
    if len(unique) == 1:
        return CheckRecord(
            CheckId.C1, CheckResult.PASS,
            {"trials": DETERMINISM_TRIALS, "output_sha16": next(iter(unique)),
             "avg_s": duration / DETERMINISM_TRIALS},
            duration,
        )
    return CheckRecord(
        CheckId.C1, CheckResult.FAIL,
        {"trials": DETERMINISM_TRIALS, "distinct_output_count": len(unique),
         "output_sha16_set": sorted(unique),
         "diagnosis": "runner non-deterministic — fix fake/random/time-source on environment side"},
        duration,
    )


def _c2_input_sensitivity(runner: RunnerAdapter, probe_input: bytes, baseline_output: bytes) -> CheckRecord:
    t0 = time.monotonic()
    if not probe_input:
        return CheckRecord(CheckId.C2, CheckResult.SKIP, {"reason": "empty probe_input"}, time.monotonic() - t0)
    rng = random.Random(0xDEADBEEF)  # deterministic probe selection for reproducibility
    positions = rng.sample(range(len(probe_input)), min(SENSITIVITY_BYTE_FLIPS, len(probe_input)))
    differing = 0
    details: list[dict] = []
    try:
        for pos in positions:
            flipped = bytearray(probe_input)
            flipped[pos] ^= 0xFF
            r = runner.rerun(bytes(flipped), observe_points=[])
            same = r.output == baseline_output
            differing += 0 if same else 1
            details.append({"flipped_byte": pos, "same_as_baseline": same})
    except NotImplementedError:
        return CheckRecord(CheckId.C2, CheckResult.SKIP, {"reason": "rerun not implemented (File mode)"}, time.monotonic() - t0)
    duration = time.monotonic() - t0
    passed = differing >= SENSITIVITY_MIN_DIFFERENT
    return CheckRecord(
        CheckId.C2, CheckResult.PASS if passed else CheckResult.FAIL,
        {"flips_tested": len(positions), "differing": differing,
         "required": SENSITIVITY_MIN_DIFFERENT, "per_flip": details,
         "diagnosis": None if passed else "runner may be a stub returning constant output, OR input wired wrong"},
        duration,
    )


def _c3_observe_point(runner: RunnerAdapter, probe_input: bytes, meta: TargetMeta,
                      *, reloc: "Relocation | None" = None) -> CheckRecord:
    """C3 — observe at the (reconciled) entry anchor returns non-empty state.

    ``meta`` is already relocation-reconciled by the orchestrator, so the observe
    PC is correct. An empty result is therefore NOT a wrong-PC problem; it is
    capability-aware: a runner that does not declare the observe-point capability
    is DEGRADED (SKIP), not failed — only a runner that declares it yet returns
    empty is a real broken hook (FAIL)."""
    t0 = time.monotonic()
    op = ObservePoint(pc=meta.algo_entry_pc, when="before", capture=("regs",))
    try:
        r = runner.rerun(probe_input, observe_points=[op])
    except NotImplementedError:
        return CheckRecord(CheckId.C3, CheckResult.SKIP, {"reason": "rerun not implemented (File mode)"}, time.monotonic() - t0)
    duration = time.monotonic() - t0
    # A reconciled (or already-correct) PC means the observe address is not the
    # suspect — every C3 diagnosis says so, to rule out the relocation class.
    pc_note = (
        f"observe PC relocation-reconciled to 0x{meta.algo_entry_pc:x} (base "
        f"delta {reloc.to_dict()['base_delta']}, see C4) — PC is correct, not the cause"
        if reloc is not None else
        f"observe PC 0x{meta.algo_entry_pc:x} matches the metadata entry anchor"
    )
    if not r.observations:
        # PC is correct (A) → distinguish "no observe capability" from "broken hook".
        from .block_cause import oracle_from_adapter  # local: avoids heavy import at module load
        oracle = oracle_from_adapter(runner, metadata=meta)
        if not oracle.has(OBSERVE_CAPABILITY):
            return CheckRecord(CheckId.C3, CheckResult.SKIP,
                {"capability": OBSERVE_CAPABILITY, "declared": False,
                 "verifier_degraded": True,
                 "diagnosis": (
                     "runner does not declare the observe-point capability — "
                     "observations unavailable; DEGRADING C3 (verifier_degraded) "
                     "rather than failing. " + pc_note + ". Declare "
                     f"capabilities=['{OBSERVE_CAPABILITY}'] (TargetMeta) once the "
                     "runner supports observe capture.")}, duration)
        return CheckRecord(CheckId.C3, CheckResult.FAIL,
            {"capability": OBSERVE_CAPABILITY, "declared": True,
             "diagnosis": (
                 "runner DECLARES the observe-point capability but returned an "
                 "empty observations list — the observe hook implementation is "
                 "broken. " + pc_note)}, duration)
    obs = r.observations[0]
    if not obs.regs:
        return CheckRecord(CheckId.C3, CheckResult.FAIL,
            {"observation_at_pc": f"0x{obs.pc:x}",
             "diagnosis": "observation has no register values — runner observe hook misconfigured. " + pc_note}, duration)
    detail: dict[str, Any] = {
        "observation_at_pc": f"0x{obs.pc:x}", "regs_captured": len(obs.regs),
        "relocation_reconciled": reloc is not None}
    # MEM round-trip (tc2 upstream): regs alone proved the observe hook fires;
    # now prove a requested MEM range actually round-trips back. The criterion is
    # the PROBE (what actually round-trips), NOT the declaration: MEM observe that
    # actually works must not be silently dropped just because the runner forgot to
    # declare mem-capture (spec MEM-cap; feedback_construct_symmetry_not_caller_
    # obligation — a usable capability never silent-degrades). The four states:
    #   declared + OK        → run, PASS (unchanged).
    #   declared + FAILS     → LOUD FAIL (the Bug1 class, unchanged).
    #   NOT declared + OK    → run anyway + a LOUD capability_undeclared_but_working
    #                          record (never a silent skip).
    #   NOT declared + unavailable → degrade WITH a surfaced WARN +
    #                          mem_skip_reason=probe_unavailable (never silent empty).
    # Every skip/degrade carries mem_skip_reason ∈ {file_mode, probe_unavailable,
    # undeclared_working} so "MEM was skipped" can never read as "MEM was fine".
    from .block_cause import oracle_from_adapter  # local: heavy import deferred
    oracle = oracle_from_adapter(runner, metadata=meta)
    declared = oracle.has(MEM_CAPTURE_CAPABILITY)
    try:
        ok, mdetail, status = _mem_round_trip_probe(runner, meta, reloc=reloc)
    except NotImplementedError:
        # rerun is File-mode/stub — the mem probe cannot run at all. Label it.
        detail["mem_round_trip"] = {
            "capability": MEM_CAPTURE_CAPABILITY, "declared": declared,
            "status": "degraded", "mem_skip_reason": "file_mode",
            "verifier_degraded": True,
            "note": ("MEM round-trip skipped: rerun() not implemented (File mode) — "
                     "labeled file_mode, NOT silently treated as checked.")}
        detail["verifier_degraded"] = True
        return CheckRecord(CheckId.C3, CheckResult.PASS, detail, duration)
    mdetail["capability"] = MEM_CAPTURE_CAPABILITY
    mdetail["declared"] = declared
    mdetail["status"] = status
    detail["mem_round_trip"] = mdetail
    duration = time.monotonic() - t0
    if status == "broken":
        # Probe ran and the wire is broken. Declared → the classic LOUD FAIL.
        # Undeclared → still a broken wire we exercised, but the runner never
        # promised it; surface it as a degrade WITH a WARN (probe_unavailable), not
        # a FAIL against an undeclared capability (and not a silent empty).
        if declared:
            return CheckRecord(CheckId.C3, CheckResult.FAIL,
                {**detail, "diagnosis": mdetail["diagnosis"] + " " + pc_note}, duration)
        mdetail["status"] = "degraded"
        mdetail["mem_skip_reason"] = "probe_unavailable"
        mdetail["capability_state"] = "undeclared_unavailable"
        mdetail.setdefault("warn",
            "MEM round-trip probe returned no bytes and the runner does not declare "
            f"{MEM_CAPTURE_CAPABILITY!r} — genuine capability gap; DEGRADING with a "
            "surfaced WARN (mem_skip_reason=probe_unavailable), never a silent skip.")
        detail["verifier_degraded"] = True
        return CheckRecord(CheckId.C3, CheckResult.PASS, detail, duration)
    if status == "degraded":
        # The probe could not be formed (no register to anchor an address). Label
        # the reason so it never reads as "checked and fine".
        mdetail["mem_skip_reason"] = "probe_unavailable"
        if not declared:
            mdetail["capability_state"] = "undeclared_unavailable"
        detail["verifier_degraded"] = True
        return CheckRecord(CheckId.C3, CheckResult.PASS, detail, duration)
    # status == "ok": MEM round-trip actually works. If the runner did NOT declare
    # it, that is the bug the spec targets — a working capability silently dropped.
    # Run it (we just did) AND surface a LOUD capability_undeclared_but_working
    # inconsistency record + treat it as available; never a silent skip.
    if not declared:
        mdetail["capability_state"] = "undeclared_working"
        mdetail["mem_skip_reason"] = "undeclared_working"
        mdetail["inconsistency"] = "capability_undeclared_but_working"
        mdetail["warn"] = (
            f"MEM round-trip ACTUALLY ROUND-TRIPS but the runner does not declare "
            f"{MEM_CAPTURE_CAPABILITY!r} — a working capability was being silently "
            "dropped (capability_undeclared_but_working). Treating MEM as AVAILABLE "
            f"and running C3-mem; declare capabilities=['{MEM_CAPTURE_CAPABILITY}'] "
            "(TargetMeta) to make the clean declared state the norm.")
    return CheckRecord(CheckId.C3, CheckResult.PASS, detail, duration)


def _pick_mem_probe_addr(regs: dict[str, int]) -> int | None:
    """Pick a generically-valid memory address to round-trip from observed regs.

    The stack pointer always points at mapped, readable memory at any PC, so it
    is a target-agnostic probe site (no curve-fit to a specific address). Falls
    back to the frame pointer, then any non-null register value, so the probe
    works even on runners that name sp differently. Returns None if no register
    carries a usable address (then the mem round-trip degrades, never FAILs)."""
    for name in ("sp", "x31", "wsp", "x29", "fp"):
        v = regs.get(name)
        if v:
            return v
    for v in regs.values():
        if v and v > 0x1000:   # skip tiny scalars unlikely to be addresses
            return v
    return None


# Number of bytes the mem round-trip requests at the probe address. Small,
# stack-resident, alignment-safe — enough to prove the wire carries mem back.
_MEM_PROBE_SIZE = 8


def _mem_round_trip_probe(runner: "RunnerAdapter", meta: "TargetMeta",
                          *, reloc: "Relocation | None" = None) -> tuple[bool, dict, str]:
    """Core mem round-trip: request a non-empty MEM range and assert the
    response observation actually carries bytes at the requested address.

    Returns ``(ok, detail, status)`` where ``status`` ∈ {"ok", "degraded",
    "broken"}:
      - "degraded": runner gave us no register to anchor a probe address, or no
        regs at all — we cannot form a valid probe; caller SKIPs (not a FAIL).
      - "broken": we requested mem at a valid addr but the runner returned an
        observation WITHOUT those bytes — the mem-capture wire is broken (FAIL).
        This is the Bug1 class: request dropped capture/mem → snapshot empty.
      - "ok": requested address present in the returned mem map → round-trip
        proven end-to-end.
    Pure of any target-specific address (probe addr derived from live regs)."""
    entry = meta.algo_entry_pc
    probe_in = b"\x00" * (meta.input_length or 8)
    # Step 1: learn a live register to anchor a valid probe address.
    seed = runner.rerun(probe_in, observe_points=[
        ObservePoint(pc=entry, when="before", capture=("regs",))])
    if not seed.observations or not seed.observations[0].regs:
        return (False, {"reason": "no register observation to anchor a mem probe address",
                        "verifier_degraded": True}, "degraded")
    addr = _pick_mem_probe_addr(seed.observations[0].regs)
    if addr is None:
        return (False, {"reason": "no register carried a usable probe address",
                        "verifier_degraded": True}, "degraded")
    # Step 2: request that exact MEM range and demand it round-trips back.
    op = ObservePoint(pc=entry, when="before", capture=("mem",),
                      mem=((addr, _MEM_PROBE_SIZE),))
    r = runner.rerun(probe_in, observe_points=[op])
    got = {}
    for o in r.observations:
        if o.pc == entry:
            got = o.mem
            break
    # The contract: a byte from [addr, addr+size) MUST appear. We accept any addr
    # in the requested range (runner may key by region start or per-byte).
    present = any(addr <= a < addr + _MEM_PROBE_SIZE for a in got) or addr in got
    detail = {"probe_addr": f"0x{addr:x}", "probe_size": _MEM_PROBE_SIZE,
              "addrs_returned": [f"0x{a:x}" for a in sorted(got)][:8],
              "n_mem_keys": len(got)}
    if reloc is not None:
        detail["relocation_reconciled"] = True
    if present:
        return (True, detail, "ok")
    detail["diagnosis"] = (
        "requested a non-empty MEM range but the runner returned NO bytes at the "
        "requested address — mem-capture wire is broken (request dropped "
        "capture/mem, or runner ignores the mem field). This is the same-execution "
        "snapshot path the Bug1 class kills end-to-end.")
    return (False, detail, "broken")


@dataclass(frozen=True)
class _TraceBounds:
    first_pc: int | None
    last_pc:  int | None
    last_mnem: str
    count:    int
    duration: float


def _peek_trace_bounds(trace_reader: TraceReader) -> _TraceBounds:
    """One streaming pass to recover first/last PC, terminal mnemonic, and count.

    Done ONCE up front so relocation reconcile (which needs the trace's real
    bounds) and C4 share a single pass — readers re-open / re-yield on each
    ``iter()``, but we never rely on a second iteration of the same reader."""
    t0 = time.monotonic()
    first_pc = last_pc = None
    last_mnem = ""
    count = 0
    for ins in trace_reader:
        if first_pc is None:
            first_pc = ins.pc
        last_pc = ins.pc
        last_mnem = ins.mnemonic
        count += 1
    return _TraceBounds(first_pc, last_pc, last_mnem, count, time.monotonic() - t0)


def _c4_trace_integrity(bounds: _TraceBounds, meta: TargetMeta,
                        *, reloc: "Relocation | None" = None) -> CheckRecord:
    """Checks first PC == algo_entry_pc and last PC == algo_exit_pc (or ret-like).

    ``meta`` is the relocation-reconciled metadata, so a clean module rebase no
    longer reads as a bounds mismatch — when ``reloc`` is set the PASS is recorded
    as relocation-reconciled (LOUD), not as a silent match."""
    first_pc, last_pc = bounds.first_pc, bounds.last_pc
    last_mnem, count, duration = bounds.last_mnem, bounds.count, bounds.duration
    if first_pc is None:
        return CheckRecord(CheckId.C4, CheckResult.FAIL, {"diagnosis": "empty trace"}, duration)
    start_ok = first_pc == meta.algo_entry_pc
    end_ok = last_pc == meta.algo_exit_pc or last_mnem.strip().startswith("ret")
    if start_ok and end_ok:
        detail: dict[str, Any] = {
            "first_pc": f"0x{first_pc:x}", "last_pc": f"0x{last_pc:x}",
            "last_mnem": last_mnem, "instr_count": count}
        if reloc is not None:
            detail["relocation_detected"] = reloc.to_dict()
            detail["note"] = (
                "trace bounds match the RECONCILED anchors (module rebased by "
                f"{reloc.to_dict()['base_delta']}); PASS is relocation-reconciled, "
                "not a raw metadata match")
        return CheckRecord(CheckId.C4, CheckResult.PASS, detail, duration)
    return CheckRecord(CheckId.C4, CheckResult.FAIL,
        {"first_pc": f"0x{first_pc:x}", "expected_start": f"0x{meta.algo_entry_pc:x}",
         "last_pc": f"0x{last_pc:x}", "expected_end": f"0x{meta.algo_exit_pc:x}",
         "last_mnem": last_mnem,
         "diagnosis": "trace bounds don't match metadata; check start/end anchors and trace range. "
                      "Not a clean relocation either (entry/exit deltas disagree or the page offset "
                      "shifted) — a genuinely wrong anchor, not reconciled away"}, duration)


def _c5_cross_call_independence(runner: RunnerAdapter, meta: TargetMeta,
                                  probe_input: bytes) -> CheckRecord:
    """Verify get_trace() works correctly after a sequence of rerun() calls.
    This is the canonical pattern that fails when a runner leaks state across
    calls (e.g. observation CodeHooks not detached). Contract §3.2 requires
    no cross-call side effects.

    Methodology (order matters — rerun MUST come before get_trace):
      1. rerun with an ObservePoint (this is what typically installs the hook)
      2. rerun × 2 without observation (any stale hook keeps firing)
      3. get_trace — if any rerun leaked state, this trace collapses
      4. additionally a 2nd get_trace to detect drift between calls

    Pass conditions (BOTH required):
      - both traces have >= INDEPENDENCE_MIN_BASELINE_LINES (10) instructions
      - ratio (after / first) >= INDEPENDENCE_FAIL_RATIO (0.5)
    """
    t0 = time.monotonic()
    op = ObservePoint(pc=meta.algo_entry_pc, when="after", capture=("regs",))

    # 1-2. Exercise rerun BEFORE any get_trace. This is the demo-like sequence.
    try:
        runner.rerun(probe_input, observe_points=[op])
        runner.rerun(probe_input, observe_points=[])
        runner.rerun(probe_input, observe_points=[])
    except NotImplementedError:
        return CheckRecord(CheckId.C5, CheckResult.SKIP,
            {"reason": "rerun not implemented (File mode)"}, time.monotonic() - t0)
    except Exception as e:
        return CheckRecord(CheckId.C5, CheckResult.SKIP,
            {"reason": f"rerun under exercise raised: {type(e).__name__}: {e}"},
            time.monotonic() - t0)

    # 3. First get_trace AFTER rerun-contamination — this is where leaky
    #    runners collapse.
    try:
        path1 = runner.get_trace(probe_input, meta.algo_entry_pc, meta.algo_exit_pc)
    except NotImplementedError:
        return CheckRecord(CheckId.C5, CheckResult.SKIP,
            {"reason": "get_trace not implemented (File mode)"}, time.monotonic() - t0)
    except Exception as e:
        return CheckRecord(CheckId.C5, CheckResult.FAIL,
            {"diagnosis": f"get_trace after rerun raised: {type(e).__name__}: {e}"},
            time.monotonic() - t0)
    first_lines = _count_lines(path1)

    # 4. Another observed-rerun + plain rerun + second get_trace, to detect drift.
    try:
        runner.rerun(probe_input, observe_points=[op])
        runner.rerun(probe_input, observe_points=[])
        path2 = runner.get_trace(probe_input, meta.algo_entry_pc, meta.algo_exit_pc)
    except Exception as e:
        return CheckRecord(CheckId.C5, CheckResult.FAIL,
            {"first_lines": first_lines,
             "diagnosis": f"second cycle raised: {type(e).__name__}: {e}"},
            time.monotonic() - t0)
    second_lines = _count_lines(path2)
    duration = time.monotonic() - t0

    detail = {
        "first_trace_lines": first_lines,
        "second_trace_lines": second_lines,
        "min_required": INDEPENDENCE_MIN_BASELINE_LINES,
        "first_path": str(path1),
        "second_path": str(path2),
    }

    # FAIL pattern 1: trace collapsed to nearly nothing — strong leak
    if first_lines < INDEPENDENCE_MIN_BASELINE_LINES:
        detail["diagnosis"] = (
            f"first get_trace after rerun produced only {first_lines} line(s). "
            "Healthy runners produce hundreds. Likely cause: rerun() installed "
            "an observation hook and never removed it, contaminating subsequent "
            "trace emission. See contracts §3.2 (no cross-call side effects)."
        )
        return CheckRecord(CheckId.C5, CheckResult.FAIL, detail, duration)

    # FAIL pattern 2: drift between trace #1 and trace #2
    ratio = second_lines / first_lines if first_lines else 0
    detail["ratio"] = round(ratio, 4)
    if ratio < INDEPENDENCE_FAIL_RATIO:
        detail["diagnosis"] = (
            f"second trace is {ratio:.0%} of first — state accumulates "
            "across rerun cycles. Runner is non-idempotent."
        )
        return CheckRecord(CheckId.C5, CheckResult.FAIL, detail, duration)

    if ratio < INDEPENDENCE_WARN_RATIO:
        detail["diagnosis"] = f"minor drift (ratio={ratio:.2f}) — investigate if reproducible"
    return CheckRecord(CheckId.C5, CheckResult.PASS, detail, duration)


def _first_diff_offset(a: bytes, b: bytes) -> int:
    """First byte offset where a and b differ, or the shorter length if one is
    a prefix of the other. -1 only if equal."""
    if a == b:
        return -1
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def check_observer_neutrality(
    runner_baseline: RunnerAdapter,
    runner_instrumented: RunnerAdapter,
    probe_input: bytes,
    *,
    label_baseline: str = "observers_off",
    label_instrumented: str = "observers_on",
) -> CheckRecord:
    """C6 — does turning an observer-set ON change the I/O it observes?

    Tier-1 detection for the "observer perturbs the observed path" class
    (the color-oracle vs guard-init-hook regression: clean run PASSes, but the
    guard observers turn the same path into WRITE_UNMAPPED → NULL). An observer
    that alters the result it observes violates observation neutrality
    (contracts §3.2; roadmap §2.2 "让缝被看见").

    Runs the SAME ``probe_input`` through two regimes — a baseline adapter
    (observer-set off / minimal) and an instrumented adapter (observer-set on)
    — and compares. The engine does not toggle the observer flag itself
    (that is the harness's job: hand in two adapters); it only runs both and
    reports the divergence. A FAIL fires when:

      * one regime faults while the other succeeds (the WRITE_UNMAPPED case), or
      * both succeed but the output bytes differ (first-diff offset reported).

    SKIP when both regimes are File-mode (no rerun). The point is to surface
    the perturbation the moment it appears, not weeks later as a silent
    regression an agent has to bisect by hand.
    """
    t0 = time.monotonic()

    def _run(adapter: RunnerAdapter):
        try:
            return ("ok", adapter.rerun(probe_input, observe_points=[]).output, None)
        except NotImplementedError:
            return ("file", None, None)
        except Exception as e:  # a fault under one regime is itself the signal
            return ("error", None, f"{type(e).__name__}: {e}")

    state_b, out_b, err_b = _run(runner_baseline)
    state_i, out_i, err_i = _run(runner_instrumented)
    duration = time.monotonic() - t0

    detail = {"baseline": label_baseline, "instrumented": label_instrumented}

    if state_b == "file" and state_i == "file":
        detail["reason"] = "both regimes File-mode (rerun not implemented)"
        return CheckRecord(CheckId.C6, CheckResult.SKIP, detail, duration)
    if state_b == "file" or state_i == "file":
        detail["reason"] = "only one regime supports rerun — cannot compare"
        return CheckRecord(CheckId.C6, CheckResult.SKIP, detail, duration)

    # One side faulted while the other ran — the regime changed the path.
    if (state_b == "error") != (state_i == "error"):
        faulting = label_instrumented if state_i == "error" else label_baseline
        clean = label_baseline if state_i == "error" else label_instrumented
        detail.update({
            "faulting_regime": faulting,
            "clean_regime": clean,
            "error": err_i if state_i == "error" else err_b,
            "diagnosis": (
                f"{faulting} faulted while {clean} succeeded — the observer-set "
                f"perturbs the observed path. Scope the observer to its owned "
                f"band (pass-through elsewhere) or run the path under {clean}."
            ),
        })
        return CheckRecord(CheckId.C6, CheckResult.FAIL, detail, duration)

    if state_b == "error" and state_i == "error":
        detail.update({"baseline_error": err_b, "instrumented_error": err_i,
                       "reason": "both regimes faulted — neutrality undecidable"})
        return CheckRecord(CheckId.C6, CheckResult.SKIP, detail, duration)

    # Both ran — compare outputs.
    if out_b == out_i:
        detail.update({"output_sha16": hashlib.sha256(out_b).hexdigest()[:16],
                       "output_len": len(out_b)})
        return CheckRecord(CheckId.C6, CheckResult.PASS, detail, duration)

    off = _first_diff_offset(out_b, out_i)
    detail.update({
        "baseline_len": len(out_b),
        "instrumented_len": len(out_i),
        "first_diff_offset": off,
        "baseline_sha16": hashlib.sha256(out_b).hexdigest()[:16],
        "instrumented_sha16": hashlib.sha256(out_i).hexdigest()[:16],
        "diagnosis": (
            f"output diverges at byte {off} when the observer-set is on — the "
            f"observer is not neutral. Scope it to its owned band, or treat the "
            f"two regimes as separate runs."
        ),
    })
    return CheckRecord(CheckId.C6, CheckResult.FAIL, detail, duration)


def check_observer_set_matrix(
    baseline: RunnerAdapter,
    regimes: dict[str, RunnerAdapter],
    probe_input: bytes,
    *,
    baseline_label: str = "clean",
) -> list[CheckRecord]:
    """Tier-2 helper: run C6 for each named observer-set against the baseline.

    Lets a caller validate a whole set of named regimes (runner_interface §3.6
    rule 2 — ``observer_sets``) against one path in a single sweep and see which
    sets are neutral (PASS) and which perturb it (FAIL), rather than discovering
    a clash one regression at a time. Returns one :class:`CheckRecord` per
    regime, each labelled with that regime's name.
    """
    return [
        check_observer_neutrality(
            baseline, adapter, probe_input,
            label_baseline=baseline_label, label_instrumented=name,
        )
        for name, adapter in regimes.items()
    ]


def _c7_capability_round_trip(runner: RunnerAdapter, meta: TargetMeta,
                              *, reloc: "Relocation | None" = None) -> list[CheckRecord]:
    """C7 — for EVERY declared capability, prove request→response round-trips.

    Generalises C3-for-mem: walks :data:`CAPABILITY_PROBES`, and for each
    capability the runner DECLARES (TargetMeta.capabilities or adapter
    CAPABILITIES) issues a request that exercises it and asserts the response
    actually carried the requested thing back. A declared-but-not-round-tripping
    capability is a LOUD FAIL — the recurring "contract declared it, the wire
    never wired it" class (Bug1 dropped capture/mem; this gate catches the next
    one). The criterion is the PROBE, not the declaration: a capability the runner
    does NOT declare but that ACTUALLY round-trips is surfaced as a LOUD
    ``capability_undeclared_but_working`` PASS (spec MEM-cap — a usable capability
    never silently dropped), while an undeclared one that does not round-trip is
    genuinely not this runner's job and is not asserted here (C3 labels the gap).
    Returns one CheckRecord per probeable capability that either declared or
    actually round-tripped; empty list (→ caller adds a single SKIP) otherwise."""
    from .block_cause import oracle_from_adapter
    oracle = oracle_from_adapter(runner, metadata=meta)
    records: list[CheckRecord] = []
    for cap, probe in sorted(CAPABILITY_PROBES.items()):
        declared = oracle.has(cap)
        # The probe-not-declaration criterion (spec MEM-cap) is scoped to the
        # capability the silent-degrade bug actually hit: mem_capture. For it we
        # probe + surface undeclared-but-working even when undeclared. Other
        # capabilities keep the declared-only contract here (C3 carries observe's
        # own degrade-with-WARN path), so an undeclared observe runner is not newly
        # asserted against — only the declared+broken class FAILs, as before.
        probe_when_undeclared = (cap == MEM_CAPTURE_CAPABILITY)
        if not declared and not probe_when_undeclared:
            continue
        t0 = time.monotonic()
        try:
            ok, detail = probe(runner, meta)
        except NotImplementedError:
            # rerun is unimplemented. Declared → a contradiction (LOUD FAIL).
            # Undeclared → the capability genuinely is not wired; not its job here,
            # skip silently (C3 already labels the file_mode/probe gap).
            if declared:
                records.append(CheckRecord(CheckId.C7, CheckResult.FAIL,
                    {"capability": cap, "declared": True,
                     "diagnosis": (f"runner DECLARES {cap!r} but rerun() is not "
                                   "implemented — declaration contradicts the wire")},
                    time.monotonic() - t0))
            continue
        dur = time.monotonic() - t0
        rec_detail = {"capability": cap, "declared": declared, **detail}
        if reloc is not None:
            rec_detail.setdefault("relocation_reconciled", True)
        if ok:
            # Round-trips. Undeclared-but-working is the silent-degrade bug (spec
            # MEM-cap): surface a LOUD capability_undeclared_but_working WARN and
            # still PASS (the capability IS usable) — never drop a working one.
            if not declared:
                rec_detail["inconsistency"] = "capability_undeclared_but_working"
                rec_detail["warn"] = (
                    f"capability {cap!r} ACTUALLY ROUND-TRIPS but the runner does not "
                    f"declare it — surfacing capability_undeclared_but_working and "
                    f"treating it as available (declare capabilities=[{cap!r}] to make "
                    "the clean declared state the norm).")
            records.append(CheckRecord(CheckId.C7, CheckResult.PASS, rec_detail, dur))
        elif declared:
            rec_detail.setdefault("diagnosis",
                f"runner DECLARES {cap!r} but the request→response round-trip "
                "returned nothing — capability wire is broken (declared, not wired)")
            records.append(CheckRecord(CheckId.C7, CheckResult.FAIL, rec_detail, dur))
        # undeclared + not round-tripping → genuinely not this runner's capability;
        # nothing to assert here (C3 degrades + labels the gap). No silent PASS/FAIL.
    return records


def _count_lines(path) -> int:
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return 0
    n = 0
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.strip():
                n += 1
    return n


# --- orchestrator ---

def run_conformance(
    runner: RunnerAdapter,
    trace_reader: TraceReader | None,
    probe_input: bytes,
    mode: str | None = None,
) -> ConformanceReport:
    """Run all applicable checks. Returns a report; caller decides what to do.

    Args:
        runner: full RunnerAdapter (Live mode) or NullRunnerAdapter (File mode)
        trace_reader: required for C4. None → C4 skipped, overall = FAIL (incomplete).
        probe_input: a representative test input; for File mode this is the
                     input the static trace was recorded against.
        mode: "live" | "file"; if None, auto-detect by trying runner.rerun().
    """
    meta = runner.metadata()
    started = _now()

    if mode is None:
        try:
            runner.rerun(probe_input, observe_points=[])
            mode = "live"
        except NotImplementedError:
            mode = "file"

    # --- relocation-aware anchor reconcile (single point; everyone uses rebased) ---
    # Peek the trace's real bounds ONCE, detect a clean module rebase, and rebase
    # ALL anchors to the run's actual base. C3/C5 (observe + get_trace) and C4 then
    # all read the reconciled ``eff_meta`` — no per-step rebase, no basis mismatch.
    bounds: _TraceBounds | None = None
    reloc: Relocation | None = None
    if trace_reader is not None:
        bounds = _peek_trace_bounds(trace_reader)
        reloc = detect_relocation(meta, bounds.first_pc, bounds.last_pc)
    eff_meta = rebase_meta(meta, reloc)

    checks: list[CheckRecord] = []

    if mode == "live":
        # C5 MUST go first. It detects cross-call state leaks that only
        # manifest when get_trace is the VERY FIRST trace call after some
        # reruns. If C1-C4 ran first, they'd "prime" the emulator and the
        # specific leak pattern would hide.
        checks.append(_c5_cross_call_independence(runner, eff_meta, probe_input))

        c1 = _c1_determinism(runner, probe_input)
        checks.append(c1)
        # baseline_output reuse: take the deterministic output we just produced
        baseline_out = b""
        if c1.result == CheckResult.PASS:
            baseline_out = runner.rerun(probe_input, observe_points=[]).output
            checks.append(_c2_input_sensitivity(runner, probe_input, baseline_out))
            checks.append(_c3_observe_point(runner, probe_input, eff_meta, reloc=reloc))
            # C7 — generic capability round-trip for every declared capability.
            c7 = _c7_capability_round_trip(runner, eff_meta, reloc=reloc)
            if c7:
                checks.extend(c7)
            else:
                checks.append(CheckRecord(CheckId.C7, CheckResult.SKIP,
                    {"reason": "runner declares no round-trip-probeable capability",
                     "verifier_degraded": True}, 0.0))
        else:
            # C1 fail / skip → C2/C3/C7 add SKIP records so report is complete.
            checks.append(CheckRecord(CheckId.C2, CheckResult.SKIP,
                {"reason": "C1 did not PASS"}, 0.0))
            checks.append(CheckRecord(CheckId.C3, CheckResult.SKIP,
                {"reason": "C1 did not PASS"}, 0.0))
            checks.append(CheckRecord(CheckId.C7, CheckResult.SKIP,
                {"reason": "C1 did not PASS"}, 0.0))
    else:
        for cid in (CheckId.C5, CheckId.C1, CheckId.C2, CheckId.C3, CheckId.C7):
            checks.append(CheckRecord(cid, CheckResult.SKIP,
                {"reason": "File mode — runner has no rerun"}, 0.0))

    if bounds is not None:
        checks.append(_c4_trace_integrity(bounds, eff_meta, reloc=reloc))
    else:
        checks.append(CheckRecord(CheckId.C4, CheckResult.SKIP,
            {"reason": "no trace_reader supplied"}, 0.0))

    completed = _now()
    # A check may DEGRADE the verifier (e.g. C3 SKIP because the runner declares
    # no observe capability) without failing the gate — a capability gap is not a
    # failure. File mode is degraded by definition.
    degraded = (mode == "file") or any(c.detail.get("verifier_degraded") for c in checks)
    # Overall verdict:
    #  - any FAIL → overall FAIL
    #  - Live mode: PASS when every check either PASSed or only degraded the
    #    verifier (a capability-gap SKIP); any other SKIP still fails (we never
    #    relaxed the strict "ran everything" bar except for declared gaps).
    #  - File mode: C4 PASS suffices; verifier_degraded = True
    any_fail = any(c.result == CheckResult.FAIL for c in checks)
    if any_fail:
        overall = CheckResult.FAIL
    elif mode == "live":
        ok = all(c.result == CheckResult.PASS or c.detail.get("verifier_degraded")
                 for c in checks)
        overall = CheckResult.PASS if ok else CheckResult.FAIL
    else:  # file
        c4 = next((c for c in checks if c.check == CheckId.C4), None)
        overall = CheckResult.PASS if c4 and c4.result == CheckResult.PASS else CheckResult.FAIL

    return ConformanceReport(
        target_name=meta.target_name,
        mode=mode,
        overall=overall,
        verifier_degraded=degraded,
        checks=checks,
        started_at=started,
        completed_at=completed,
        relocation=reloc.to_dict() if reloc is not None else None,
    )


def write_report(report: ConformanceReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))


def require_pass_or_die(report: ConformanceReport) -> None:
    """Caller helper — block main pipeline unless overall PASS."""
    if report.overall != CheckResult.PASS:
        details = "\n".join(
            f"  {c.check.value}: {c.result.value}" +
            (f"  // {c.detail.get('diagnosis')}" if c.detail.get("diagnosis") else "")
            for c in report.checks
        )
        raise RuntimeError(
            f"Runner conformance check FAILED (mode={report.mode}, overall={report.overall.value}):\n"
            f"{details}\n"
            "Fix the runner / environment before re-running. See conformance_report.json."
        )
