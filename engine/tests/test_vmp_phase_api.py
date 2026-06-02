"""Pin the VMP phase API: the sign5 light-to-heavy path, forced.

These tests prove (1) the canonical chain runs in order and wires the real
primitives, (2) there is no "guess a candidate set" entry, and (3) the heavy
vmtrace is unreachable without a proof that the light phases were exhausted.
"""

from __future__ import annotations

import inspect

import pytest

from engine.constant_provenance import ConstantProvenanceResult
from engine.phase import ANCHOR_FUNC_ENTRY, Anchor
from engine.phase_instrument import GRAN_FULL, PhaseInstrumentSpec
from engine.phase_sequence import (
    EscalationConfirmation,
    EscalationProof,
    PhaseGateError,
    PhaseStatus,
)
from engine.vmp_phase_api import (
    PHASE_HEAVY_VMTRACE,
    PHASE_IO_OBSERVE,
    PHASE_MATERIALIZATION,
    PHASE_PROVENANCE,
    VMP_PHASE_SEQUENCE,
    FormulaInduction,
    ParityIntent,
    VmpPhaseApi,
    VmtraceBudget,
)
from engine.watch_first_write import WatchFirstWriteSpec


def test_canonical_sequence_is_valid_and_ordered():
    ordered = [s.name for s in VMP_PHASE_SEQUENCE.ordered]
    assert ordered == [
        PHASE_IO_OBSERVE,
        PHASE_MATERIALIZATION,
        PHASE_PROVENANCE,
        "phase_4_formula_induction",
        "phase_5_parity",
    ]
    # heavy is an escalation, not part of the linear chain
    assert [s.name for s in VMP_PHASE_SEQUENCE.escalations] == [PHASE_HEAVY_VMTRACE]


def test_happy_path_runs_in_order_and_builds_specs():
    api = VmpPhaseApi()

    io = api.phase_1_io_observe(entry_pc=0x1000)
    assert isinstance(io, PhaseInstrumentSpec)
    api.record(PHASE_IO_OBSERVE, PhaseStatus.RAN, summary="i/o shape captured")

    mat = api.phase_2_materialization_trace(output_base=0x4000, output_len=32)
    assert isinstance(mat, PhaseInstrumentSpec)
    api.record(PHASE_MATERIALIZATION, PhaseStatus.RAN)

    watch = api.phase_3_watch_producer(addr=0x4000, value_name="prefix")
    assert isinstance(watch, WatchFirstWriteSpec)
    result = api.phase_3_classify("prefix")
    assert isinstance(result, ConstantProvenanceResult)
    api.record(PHASE_PROVENANCE, PhaseStatus.CLOSED, summary="provenance traced")

    intent = api.phase_4_formula_induction(
        FormulaInduction(expression="out = sm3(appkey || template)",
                         derived_from=(PHASE_MATERIALIZATION, PHASE_PROVENANCE))
    )
    assert isinstance(intent, ParityIntent)
    api.record("phase_4_formula_induction", PhaseStatus.RAN, summary="formula induced")

    final = api.phase_5_parity(intent)
    assert isinstance(final, ParityIntent)
    api.record("phase_5_parity", PhaseStatus.CLOSED, summary="158/158 bytewise")
    assert api.run.is_closed()


def test_out_of_order_is_refused():
    api = VmpPhaseApi()
    # jump straight to materialization without phase_1
    with pytest.raises(PhaseGateError):
        api.phase_2_materialization_trace(output_base=0x4000, output_len=32)


def test_no_candidate_set_entry_exists():
    """Structural: the API offers no way to submit a *set* of guesses. Phase 4
    takes exactly one induced formula; no method name hints at guessing."""
    methods = {n for n, _ in inspect.getmembers(VmpPhaseApi, inspect.isfunction)}
    for banned in ("guess", "candidate", "candidates", "try_algorithms", "brute"):
        assert not any(banned in m for m in methods), f"unexpected guess-verb: {banned}"
    # FormulaInduction carries a single expression, not a collection
    sig = inspect.signature(FormulaInduction.__init__)
    assert "expression" in sig.parameters


# --- heavy vmtrace gate -----------------------------------------------------

def _light_phases_walled(api: VmpPhaseApi) -> None:
    api.phase_1_io_observe(entry_pc=0x1000)
    api.record(PHASE_IO_OBSERVE, PhaseStatus.RAN)
    api.phase_2_materialization_trace(output_base=0x4000, output_len=32)
    api.record(PHASE_MATERIALIZATION, PhaseStatus.RAN)
    api.phase_3_watch_producer(addr=0x4000, value_name="prefix")
    api.record(
        PHASE_PROVENANCE,
        PhaseStatus.COULD_NOT_CLOSE,
        could_not_close_reason="producer PC outside observable range (true boundary)",
    )


_BUDGET = VmtraceBudget(runtime_s=1800, disk_mb=4096, note="full vmtrace est")


def test_heavy_vmtrace_refused_without_proof_or_confirmation():
    api = VmpPhaseApi()
    _light_phases_walled(api)
    with pytest.raises(PhaseGateError) as ei:
        api.phase_heavy_vmtrace(
            anchor=Anchor(ANCHOR_FUNC_ENTRY, {"pc": 0x1000}), budget=_BUDGET)
    # the refusal points the caller at the confirmation prompt
    assert "escalation_prompt" in str(ei.value)


def test_heavy_vmtrace_requires_a_budget():
    api = VmpPhaseApi()
    _light_phases_walled(api)
    with pytest.raises(TypeError):  # budget is a required kw-only arg
        api.phase_heavy_vmtrace(  # type: ignore[call-arg]
            confirmation=EscalationConfirmation(who="user"),
            anchor=Anchor(ANCHOR_FUNC_ENTRY, {"pc": 0x1000}),
        )


def test_budget_must_have_positive_runtime_and_disk():
    with pytest.raises(ValueError):
        VmtraceBudget(runtime_s=0, disk_mb=10)
    with pytest.raises(ValueError):
        VmtraceBudget(runtime_s=10, disk_mb=0)


def test_heavy_vmtrace_via_human_confirmation_after_light_phases():
    api = VmpPhaseApi()
    _light_phases_walled(api)
    # driver would render this question to the user, WITH the agent's budget
    prompt = api.heavy_vmtrace_prompt(_BUDGET)
    assert prompt.phase == PHASE_HEAVY_VMTRACE
    assert "vmtrace" in prompt.question
    assert prompt.walled == (PHASE_PROVENANCE,)
    assert "4096MB" in prompt.question  # cost shown to the human
    # user answers yes → confirmation unlocks the gate
    spec = api.phase_heavy_vmtrace(
        confirmation=EscalationConfirmation(who="user", note="tried 1-3, go"),
        anchor=Anchor(ANCHOR_FUNC_ENTRY, {"pc": 0x1000}),
        budget=_BUDGET,
    )
    assert isinstance(spec, PhaseInstrumentSpec)
    assert spec.granularity == GRAN_FULL
    assert api.run.confirmation(PHASE_HEAVY_VMTRACE).who == "user"
    assert api.heavy_budget.disk_mb == 4096


def test_prompt_warns_when_light_phases_were_skipped():
    api = VmpPhaseApi()
    # jump toward vmtrace having run nothing — the case most worth stopping
    prompt = api.heavy_vmtrace_prompt()
    assert prompt.untried_required  # phase_1/2/3 all listed as not tried
    assert "尚未尝试" in prompt.question


def test_confirmation_is_first_time_only():
    api = VmpPhaseApi()
    _light_phases_walled(api)
    api.phase_heavy_vmtrace(
        confirmation=EscalationConfirmation(who="user"),
        anchor=Anchor(ANCHOR_FUNC_ENTRY, {"pc": 0x1000}),
        budget=_BUDGET,
    )
    # second call needs neither proof nor confirmation — already unlocked
    spec2 = api.phase_heavy_vmtrace(
        anchor=Anchor(ANCHOR_FUNC_ENTRY, {"pc": 0x2000}), budget=_BUDGET)
    assert isinstance(spec2, PhaseInstrumentSpec)


def test_proof_and_confirmation_together_rejected():
    api = VmpPhaseApi()
    _light_phases_walled(api)
    with pytest.raises(PhaseGateError):
        api.phase_heavy_vmtrace(
            proof=EscalationProof(cites=(PHASE_PROVENANCE,), reason="walled"),
            confirmation=EscalationConfirmation(who="user"),
            anchor=Anchor(ANCHOR_FUNC_ENTRY, {"pc": 0x1000}),
            budget=_BUDGET,
        )


def test_heavy_vmtrace_refused_when_light_phases_did_not_wall():
    api = VmpPhaseApi()
    api.phase_1_io_observe(entry_pc=0x1000)
    api.record(PHASE_IO_OBSERVE, PhaseStatus.RAN)
    api.phase_2_materialization_trace(output_base=0x4000, output_len=32)
    api.record(PHASE_MATERIALIZATION, PhaseStatus.RAN)
    api.phase_3_watch_producer(addr=0x4000, value_name="prefix")
    api.record(PHASE_PROVENANCE, PhaseStatus.RAN)  # ran, did NOT wall
    with pytest.raises(PhaseGateError) as ei:
        api.phase_heavy_vmtrace(
            proof=EscalationProof(cites=(PHASE_PROVENANCE,), reason="want more"),
            anchor=Anchor(ANCHOR_FUNC_ENTRY, {"pc": 0x1000}),
            budget=_BUDGET,
        )
    assert "could_not_close" in str(ei.value)


def test_heavy_vmtrace_allowed_with_valid_proof_and_is_full_grain():
    api = VmpPhaseApi()
    _light_phases_walled(api)
    spec = api.phase_heavy_vmtrace(
        proof=EscalationProof(
            cites=(PHASE_PROVENANCE,),
            reason="phase_3 hit a true boundary: producer outside observable range",
        ),
        anchor=Anchor(ANCHOR_FUNC_ENTRY, {"pc": 0x1000}),
        budget=_BUDGET,
    )
    assert isinstance(spec, PhaseInstrumentSpec)
    assert spec.granularity == GRAN_FULL
