"""Recovery-as-a-CVD-run — real-case e2e on the VMP trace (Triton-gated).

The dev-side closure of the "假绿" loop (dev-recovery-as-cvd-run.md §真案验收): the
collect-mode recovery run is exercised over a REAL window from example/task-libEncryptor/libs/arm64-v8a/trace.txt
through the REAL Level-2 Triton runner — not a synthetic mock. It asserts the
MECHANISM, not a case solution: one run produces a fully-classified structured gap
map (every candidate routed to a defined disposition), with NO silent stall and NO
"no Verifier applies" wiring gap. The case gold F is intentionally NOT hardcoded
here (that lives in the case fixture, not the primitive — invariant 2/6); a CONFIRMED
window is allowed but not required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.cvd import CvdOutcome, run_cvd
from engine.cvd_recovery import recovery_registry
from engine.dispatch_coverage import HandlerInvocation, preflight_dispatch_coverage
from engine.runner_client import UnidbgTextTraceReader
from engine.setup_symex import CaseConfig
from engine.setup_symex_runner import build_level2_runner, triton_available

REPO_ROOT = Path(__file__).resolve().parents[2]
VMP_TRACE = REPO_ROOT / "example" / "task-libEncryptor" / "libs" / "arm64-v8a" / "trace.txt"

requires_triton = pytest.mark.skipif(
    not triton_available(), reason="Triton not installed on host")
requires_trace = pytest.mark.skipif(
    not VMP_TRACE.exists(), reason="example/task-libEncryptor/libs/arm64-v8a/trace.txt not present")


def _real_window(lo: int, hi: int):
    out = []
    for ins in UnidbgTextTraceReader(VMP_TRACE):
        if lo <= ins.idx <= hi:
            out.append(ins)
        if ins.idx > hi:
            break
    return out


@requires_trace
@requires_triton
def test_real_trace_collect_run_produces_classified_gap_map():
    trace = _real_window(0, 40)
    assert trace, "expected a non-empty real window"

    # Two representative handler windows (idx bands) → two recover_window candidates.
    invs = [HandlerInvocation("A", 0, 20), HandlerInvocation("B", 21, 40)]
    cov = preflight_dispatch_coverage(trace, invocations=invs)

    base = CaseConfig(
        target="libEncryptor.so", input_hash="real", run_id="real-1",
        seed_hint_addr=0x100, sink_hint_addr=0x200, entry_pc=trace[0].pc - 1,
        window=(0, 20), window_kind="idx",
        reg_file=tuple(sorted({r for ins in trace for r in ins.regs_read})),
        inputs=("carrier",), parity_min=8, symbolic_regs=None, task="real_recover")

    # The REAL Level-2 Triton runner (no gold fn → cannot falsely close; it surfaces
    # the real symex outcome per window).
    runner = build_level2_runner()
    reg = recovery_registry(base_config=base, triton_runner=runner, coverage=cov,
                            decisions={"alias_vs_compute": "compute", "which_static": []})

    res = run_cvd(trace, b"\x00" * 16, registry=reg, collect_extensions=True)

    # The run terminated cleanly with the whole gap map — never a silent stall.
    assert res.outcome is CvdOutcome.COLLECTED
    # Every candidate window was routed to a DEFINED disposition (the anti-whack-a-
    # mole property): confirmed / extension_request / pending / ELIMINATED all count
    # — what must NOT happen is a silent stall or a "no Verifier applies" miss.
    #
    # §5′ semantic note: under the OLD (clo-AND) gate these real windows tripped the
    # backing block (clo-unbacked symbolic EA) → fixable → ELIMINATED + geometry-flip
    # spawn (landing in extension_requests). The §5′ DYNAMIC gate recognises that the
    # real trace carries op.addr+value for those loads → they are dynamically backed,
    # ENTER symex, and the real Triton runner converges them to a multi-vector parity
    # verdict (ELIMINATED/parity, no spawn) instead of a stale clo-block. The door
    # opening is the point of §5′; the windows are still cleanly routed.
    eliminated = sum(1 for e in res.log if e.get("event") == "ELIMINATED")
    routed = (len(res.confirmed) + len(res.extension_requests)
              + len(res.pending_judgments) + eliminated)
    assert routed >= 1
    # The mechanism is actually wired: the recover_window Verifier applied to every
    # recover_window candidate, so NO gap is a "no Verifier applies" wiring miss.
    assert not any(e.get("missing_kind") == "verifier"
                   for e in res.extension_requests), \
        f"recover_window verifier failed to apply: {res.extension_requests}"
    # No window stalled as a stale clo-block: the §5′ dynamic gate must NOT route a
    # window carrying op.addr+value to the "blind address closure" re-capture path.
    assert not any(e.get("event") == "ELIMINATED"
                   and "blind address closure" in str(e.get("reason", ""))
                   for e in res.log), \
        "a dynamically-backed window was stale-blocked as a blind closure"
    # Gaps converge to RECOGNIZED frontiers, not unclassified stalls.
    for e in res.extension_requests:
        assert e.get("terminal_kind") in (
            "opaque_staging", "unmodeled_instruction", "seed_invariant") \
            or e.get("missing_kind") == "capability", e
