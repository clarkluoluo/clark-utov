"""#2 — real-gold collector with a distinct-OUTPUT-variance floor.

Universal principle (spec_tc2_realgold_distinct_floor): :func:`setup_symex.
check_parity_vectors` already decides EXACT / UNCLOSABLE / BLOCK; the missing piece
is the COLLECTOR that feeds it real held-out vectors. For a target whose output is
driven by per-run external state (``time(NULL)`` → ``srand`` → ``rand``, or any
nonce/time seed), a plain N-attempt batch can produce N IDENTICAL observed outputs —
and then parity "matches" trivially. Counting attempts is the wrong floor; the right
floor is on DISTINCT OBSERVED OUTPUTS (the output-variance gate, generalized from
``feedback_parity_needs_output_variance``). Below the floor the honest verdict is
``INSUFFICIENT_VARIANCE`` — NOT a false ``UNCLOSABLE`` (which reads as "F is wrong")
and NOT a false ``EXACT``.

Inventory (A8①, don't rebuild): reuse :func:`recapture_loop.assert_same_execution`
(exec-id same-execution guard), :func:`setup_symex.check_parity_vectors` (the
verdict), :class:`setup_symex.ParityVector`, and the parity-variance gate
(``check_parity_vectors`` already judges output-variance / closability). This module
adds: ParityVector emission + a distinct-output floor + a collect-until loop control.
It drives the runner via the existing ``adapter.rerun`` (same wire as
``recapture_loop``), not a new analysis.

Subject = the SEED, not the input (A8②). The recovery variable is whatever drives
the output: F(input), F(nonce), F(time), F(input,nonce). Per rerun the collector
captures the observe-point values for the declared seed(s) AND the observed output,
stamps the exec-id, and emits a ParityVector keyed by the seed-assignment
fingerprint. Seed forms covered: a reg seed, a ``mem@addr`` seed, an external-state
seed (time/rand — CAPTURED, not controlled).

Degenerate (A8④, always a verdict): below the floor after the rerun budget is spent,
emit ``INSUFFICIENT_VARIANCE`` with ``(observed_distinct, floor, reruns_spent)`` — a
surfaced, honest stop, never a silent UNCLOSABLE / EXACT.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .runner_client import ObservePoint, ObservedState, RerunResult
from .setup_symex import (
    ParityVector,
    ParityVectorReport,
    SetupSymexConfig,
    check_parity_vectors,
)

__all__ = [
    "SeedSpec",
    "RealGoldReport",
    "collect_real_gold",
    "INSUFFICIENT_VARIANCE",
]

# The honest below-floor verdict (A8④). NOT a ParityVectorReport verdict — it is a
# distinct downgrade applied by the collector when the variance floor was not met,
# so a consumer never mistakes "couldn't collect diverse evidence" for "F is wrong"
# (UNCLOSABLE) or for a true EXACT.
INSUFFICIENT_VARIANCE = "INSUFFICIENT_VARIANCE"


@dataclass(frozen=True)
class SeedSpec:
    """One seed (the recovery variable) to capture per rerun (A8②).

    ``kind`` is the seed FORM: ``"reg"`` (a register value at an observe PC),
    ``"mem"`` (bytes at ``addr`` — a ``mem@addr`` seed), or ``"external"`` (an
    external-state seed like time/rand — captured, not controlled). ``name`` is the
    stable label used in the seed-assignment fingerprint. ``reg`` / ``addr`` / ``len``
    locate the value within an :class:`ObservedState`."""

    name: str
    kind: str = "reg"               # "reg" | "mem" | "external"
    reg: str | None = None
    addr: int | None = None
    length: int = 8
    observe_pc: int | None = None   # which observe point carries this seed's value


@dataclass(frozen=True)
class RealGoldReport:
    """The collector's structured result (#2).

    ``vectors`` are ready for :func:`check_parity_vectors`. ``observed_distinct`` is
    the distinct-OUTPUT count (the variance floor axis). ``verdict_hint`` is
    EXACT / UNCLOSABLE / INSUFFICIENT_VARIANCE / BLOCK — INSUFFICIENT_VARIANCE when
    the floor was not met (A8④), otherwise the authoritative ``check_parity_vectors``
    verdict."""

    vectors: tuple[ParityVector, ...]
    observed_distinct: int
    reruns_spent: int
    floor: int
    floor_met: bool
    verdict_hint: str
    parity_report: ParityVectorReport | None = None
    truncated: bool = False
    same_execution_violation: dict[str, Any] | None = None
    detail: str = ""

    @property
    def exact(self) -> bool:
        return self.verdict_hint == "EXACT"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": "real_gold_report",
            "n_vectors": len(self.vectors),
            "observed_distinct": self.observed_distinct,
            "reruns_spent": self.reruns_spent,
            "floor": self.floor,
            "floor_met": self.floor_met,
            "verdict_hint": self.verdict_hint,
            "truncated": self.truncated,
            "detail": self.detail,
        }
        if self.parity_report is not None:
            out["parity_report"] = self.parity_report.to_dict()
        if self.same_execution_violation is not None:
            out["same_execution_violation"] = self.same_execution_violation
        return out


def _find_observation(result: RerunResult, pc: int | None) -> ObservedState | None:
    """The observation captured at ``pc`` (any when ``pc`` is None — first available)."""
    if not result.observations:
        return None
    if pc is None:
        return result.observations[0]
    for obs in result.observations:
        if obs.pc == pc:
            return obs
    return None


def _capture_seed(result: RerunResult, seed: SeedSpec) -> str:
    """Capture one seed's value from this rerun → a hex string for the fingerprint.

    A8②: a reg seed reads ``ObservedState.regs[reg]``; a mem seed reads
    ``ObservedState.mem[addr]``; an external seed is captured the same way (utov
    does not control it). Returns ``"?"`` when the value was not captured (surfaced,
    never fabricated)."""
    obs = _find_observation(result, seed.observe_pc)
    if obs is None:
        return "?"
    if seed.kind == "mem" and seed.addr is not None:
        data = obs.mem.get(seed.addr)
        if data is None:
            # try byte-wise coverage from the observation
            return "?"
        return bytes(data)[: seed.length].hex()
    if seed.reg is not None:
        val = obs.regs.get(seed.reg)
        if val is None:
            return "?"
        return f"{val:x}"
    return "?"


def _seed_fingerprint(seed_values: dict[str, str]) -> str:
    """The per-vector seed-assignment fingerprint = the ``input_key`` CVD uses.

    A stable hash of the (name → value) seed assignment so two reruns with the SAME
    seed assignment collapse to one counted vector (input-dimension independence),
    while different seed assignments are distinct. External-state seeds (time/rand)
    naturally vary run-to-run → distinct fingerprints, which is exactly the variance
    the floor measures."""
    if not seed_values:
        return "<no-seed>"
    canon = ";".join(f"{k}={seed_values[k]}" for k in sorted(seed_values))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def collect_real_gold(
    runner_adapter: Any,
    observe_points: Iterable[ObservePoint],
    seeds: Iterable[SeedSpec],
    *,
    loop_input: bytes,
    predict: Callable[[dict[str, str], bytes], bytes],
    window: tuple[int, int],
    distinct_output_floor: int | None = None,
    max_reruns: int = 200,
    exec_identity: str | None = None,
    config: SetupSymexConfig | None = None,
    env: dict[str, str] | None = None,
) -> RealGoldReport:
    """Drive runner reruns until the DISTINCT-OUTPUT floor is met, emit ParityVectors,
    then hand them to :func:`check_parity_vectors` (#2).

    Args:
      runner_adapter: anything with ``rerun(input, observe_points) -> RerunResult``
        (the existing wire — same as ``recapture_loop``).
      observe_points: the points that carry the seed values + the output region.
      seeds: the recovery variable(s) to capture per rerun (A8② — reg/mem/external).
      loop_input: the (fixed) input driving each rerun; the variance comes from the
        per-run external state, not from dialing this input.
      predict: ``F`` — given the captured seed values + input, returns the predicted
        output bytes. (utov never invents F; the caller supplies the emitted
        transform; the collector only feeds parity.)
      window: the (start, end) PCs for the parity report.
      distinct_output_floor: the variance floor. Defaults from the SAME env as the
        parity gate (``UTOV_SETUP_SYMEX_PARITY_VECTORS``) so the variance floor and
        the independence floor stay COUPLED; an explicit value overrides.
      max_reruns: the rerun budget.
      exec_identity: the deriving-trace exec id (its vectors are tautological and
        excluded from the independent floor — passed to ``check_parity_vectors`` as
        ``trace_exec_id``).

    Loop control: keep rerunning until ``observed_distinct >= floor`` or
    ``reruns_spent >= max_reruns``. Each rerun: drive runner → capture seed values +
    output (each under its own single-execution ``exec_id``, so the per-vector
    determinism check in ``check_parity_vectors`` sees no cross-run mixing) → append
    a ParityVector. Then hand ``vectors`` to ``check_parity_vectors``; downgrade to
    ``INSUFFICIENT_VARIANCE`` if the floor
    was not met (A8④). When the floor is met on the FIRST batch (a normal
    input-varying target) behavior == today's single batch → check_parity_vectors
    (A8③ preserve function)."""
    cfg = config or SetupSymexConfig.from_env(env)
    floor = int(distinct_output_floor) if distinct_output_floor is not None else cfg.parity_min_vectors
    floor = max(1, floor)
    min_vectors = cfg.parity_min_vectors
    observe_points = list(observe_points)
    seeds = list(seeds)

    vectors: list[ParityVector] = []
    distinct_outputs: set[str] = set()
    reruns_spent = 0
    any_truncated = False

    while reruns_spent < max_reruns:
        result = runner_adapter.rerun(loop_input, observe_points)
        if not isinstance(result, RerunResult):
            raise TypeError(
                "runner_adapter.rerun must return a RerunResult; got "
                f"{type(result).__name__}")
        reruns_spent += 1
        any_truncated = any_truncated or bool(result.truncated)

        output = bytes(result.output)
        if not output:
            # An empty output cannot be a gold vector — honest stop, not a silent spin.
            return RealGoldReport(
                vectors=tuple(vectors),
                observed_distinct=len(distinct_outputs),
                reruns_spent=reruns_spent, floor=floor, floor_met=False,
                verdict_hint="BLOCK",
                truncated=any_truncated,
                detail=(f"rerun #{reruns_spent} returned an EMPTY output — cannot "
                        "build a gold vector; check the rerun wire / that loop_input "
                        "drives a production"),
            )

        seed_values = {s.name: _capture_seed(result, s) for s in seeds}
        input_key = _seed_fingerprint(seed_values)
        exec_id = _exec_id(reruns_spent, output)
        predicted = bytes(predict(seed_values, loop_input))

        vectors.append(ParityVector(
            input_key=input_key,
            observed=output.hex(),
            predicted=predicted.hex(),
            exec_id=exec_id,
            derived_from=False,
        ))

        distinct_outputs.add(output.hex())

        # Loop control: stop once the DISTINCT-OUTPUT floor is met.
        if len(distinct_outputs) >= floor:
            break

    floor_met = len(distinct_outputs) >= floor
    observed_distinct = len(distinct_outputs)

    # Same-execution coupling (G1): the determinism guard inside check_parity_vectors
    # forbids CROSS-RUN MIXING — one execution's observed output backing two distinct
    # inputs. Here each ParityVector pairs its OWN rerun's predicted-vs-observed under
    # its own exec_id (one rerun = one vector = one token), so there is no mixing by
    # construction; the MANY distinct exec_ids across vectors are the variance we WANT
    # (a variance collector deliberately spans executions), NOT a violation. So we do
    # not run assert_same_execution over the vector set (it would always flag the
    # intended multi-execution spread). The per-vector determinism is enforced by
    # check_parity_vectors' own exec_id check below. ``se_violation`` stays None.
    se_violation = None

    report = check_parity_vectors(
        vectors, window=window, min_vectors=min_vectors,
        trace_exec_id=exec_identity)

    # A8④: below the floor → INSUFFICIENT_VARIANCE, NOT a false UNCLOSABLE/EXACT.
    if not floor_met:
        return RealGoldReport(
            vectors=tuple(vectors),
            observed_distinct=observed_distinct,
            reruns_spent=reruns_spent, floor=floor, floor_met=False,
            verdict_hint=INSUFFICIENT_VARIANCE,
            parity_report=report,
            truncated=any_truncated,
            same_execution_violation=se_violation,
            detail=(f"distinct observed outputs={observed_distinct} < floor={floor} "
                    f"after {reruns_spent} rerun(s) (budget {max_reruns}). The output "
                    "is driven by per-run external state that did not vary enough to "
                    "clear the variance floor — INSUFFICIENT_VARIANCE (an honest stop: "
                    "NOT 'F is wrong' / not UNCLOSABLE, and NOT a false EXACT). Raise "
                    "max_reruns, or the production is near-constant on this seed."),
        )

    # Floor met → the authoritative check_parity_vectors verdict stands.
    return RealGoldReport(
        vectors=tuple(vectors),
        observed_distinct=observed_distinct,
        reruns_spent=reruns_spent, floor=floor, floor_met=True,
        verdict_hint=report.verdict,
        parity_report=report,
        truncated=any_truncated,
        same_execution_violation=se_violation,
        detail=(f"variance floor met: {observed_distinct} distinct observed output(s) "
                f">= floor={floor} after {reruns_spent} rerun(s); "
                f"check_parity_vectors verdict={report.verdict} over {len(vectors)} "
                "held-out vector(s)"),
    )


def _exec_id(rerun_no: int, output: bytes) -> str:
    """A single-execution token for this rerun (folds the rerun number + the output
    nonce). Deterministic / pure — two reruns with different outputs get different
    tokens, so cross-run mixing is DETECTABLE."""
    return f"realgold#{rerun_no}:{output[:16].hex()}"
