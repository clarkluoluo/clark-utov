"""The phase API is reachable from the agent's actual tool surface (agent-serve
RPC dispatch), order is enforced there, vmtrace is gated there, and there is NO
"enumerate standard crypto" method — the candidate-guessing path does not exist
at the interface (roadmap §8.13).

_dispatch only uses `core` as an attribute holder for the per-session
VmpPhaseApi, so a SimpleNamespace stands in for a full Core.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine.orchestrators.agent_mode import _dispatch
from engine.phase_sequence import PhaseGateError


def _core():
    return SimpleNamespace()


def test_phase_state_lists_the_sequence_including_heavy():
    core = _core()
    st = _dispatch(core, "phase_state", {})
    assert st["sequence"][0] == "phase_1_io_observe"
    assert "phase_heavy_vmtrace" in st["sequence"]
    assert st["closed"] is False


def test_phases_run_in_order_via_rpc():
    core = _core()
    spec1 = _dispatch(core, "phase_1_io_observe", {"entry_pc": "0x706d0"})
    assert spec1["kind"] == "phase_instrument"
    assert spec1["granularity"] == "pc_band"
    _dispatch(core, "phase_record", {"phase": "phase_1_io_observe", "status": "ran"})
    spec2 = _dispatch(core, "phase_2_materialization_trace",
                      {"output_base": "0x70f84", "output_len": 128})
    assert spec2["kind"] == "phase_instrument"


def test_out_of_order_is_refused_at_the_rpc_layer():
    core = _core()
    # jump to phase_2 without phase_1 → the gate fires through dispatch
    with pytest.raises(PhaseGateError):
        _dispatch(core, "phase_2_materialization_trace",
                  {"output_base": 0x4000, "output_len": 32})


def test_phase_3_classify_is_the_only_crypto_source_move():
    core = _core()
    # advance to phase_3
    _dispatch(core, "phase_1_io_observe", {"entry_pc": 0x1000})
    _dispatch(core, "phase_record", {"phase": "phase_1_io_observe", "status": "ran"})
    _dispatch(core, "phase_2_materialization_trace", {"output_base": 0x4000, "output_len": 32})
    _dispatch(core, "phase_record", {"phase": "phase_2_materialization_trace", "status": "ran"})
    out = _dispatch(core, "phase_3_classify", {"value_name": "cipher", "producer_dataflow": {}})
    assert "classifications" in out and isinstance(out["classifications"], list)


def test_no_enumerate_standard_crypto_method_exists():
    """Structural: the candidate-guessing detour has no RPC method. An agent
    that wants to spray SM4/AES/HMAC/CTR finds no interface for it."""
    core = _core()
    for banned in ("enumerate_algorithms", "try_standard_crypto",
                   "guess_cipher", "bruteforce_algorithm"):
        with pytest.raises(ValueError, match="unknown method"):
            _dispatch(core, banned, {})


# --- vmtrace gate at the RPC layer -----------------------------------------

def _advance_to_walled_phase3(core):
    _dispatch(core, "phase_1_io_observe", {"entry_pc": 0x1000})
    _dispatch(core, "phase_record", {"phase": "phase_1_io_observe", "status": "ran"})
    _dispatch(core, "phase_2_materialization_trace", {"output_base": 0x4000, "output_len": 32})
    _dispatch(core, "phase_record", {"phase": "phase_2_materialization_trace", "status": "ran"})
    _dispatch(core, "phase_3_watch_producer", {"addr": 0x4000, "value_name": "cipher"})
    _dispatch(core, "phase_record", {
        "phase": "phase_3_provenance", "status": "could_not_close",
        "could_not_close_reason": "FUN_0016fdf0 is non-standard VMP crypto; producer opaque"})


_ANCHOR = {"anchor_type": "func_entry", "params": {"pc": 0x1000}}
_BUDGET = {"runtime_s": 1800, "disk_mb": 4096, "note": "full vmtrace"}


def test_vmtrace_refused_without_proof_or_confirmation():
    core = _core()
    _advance_to_walled_phase3(core)
    with pytest.raises(PhaseGateError):
        _dispatch(core, "phase_heavy_vmtrace", {"anchor": _ANCHOR, "budget": _BUDGET})


def test_vmtrace_prompt_shows_budget():
    core = _core()
    _advance_to_walled_phase3(core)
    p = _dispatch(core, "phase_heavy_vmtrace_prompt", {"budget": _BUDGET})
    assert "4096MB" in p["question"]


def test_vmtrace_allowed_with_confirmation_and_budget():
    core = _core()
    _advance_to_walled_phase3(core)
    spec = _dispatch(core, "phase_heavy_vmtrace", {
        "anchor": _ANCHOR, "budget": _BUDGET,
        "confirmation": {"who": "user", "note": "tried 1-3, F is non-standard, go"}})
    assert spec["granularity"] == "full_instruction"
    st = _dispatch(core, "phase_state", {})
    assert st["heavy_budget"]["disk_mb"] == 4096


def test_vmtrace_requires_budget():
    core = _core()
    _advance_to_walled_phase3(core)
    with pytest.raises(KeyError):  # budget is a required param
        _dispatch(core, "phase_heavy_vmtrace", {
            "anchor": _ANCHOR, "confirmation": {"who": "user"}})
