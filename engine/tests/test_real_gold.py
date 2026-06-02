"""#2 — real-gold collector with a distinct-OUTPUT-variance floor.

Regression fixtures map 1:1 to spec_tc2_realgold_distinct_floor "Regression
fixtures": distinct parity (EXACT), nonce closability (INSUFFICIENT_VARIANCE not
UNCLOSABLE), negative input-varying (floor met first batch). Drives a fake adapter
through the existing rerun wire.
"""

from __future__ import annotations

from engine.real_gold import (
    INSUFFICIENT_VARIANCE,
    RealGoldReport,
    SeedSpec,
    collect_real_gold,
)
from engine.runner_client import ObservedState, ObservePoint, RerunResult
from engine.setup_symex import SetupSymexConfig


class _FakeAdapter:
    """Drives a scripted sequence of (output, seed_value) reruns through the rerun
    wire. ``seed_reg`` is the register carrying the seed at observe PC ``seed_pc``."""

    def __init__(self, scripted, *, seed_pc=0x1000, seed_reg="x0"):
        self._scripted = list(scripted)
        self._i = 0
        self._seed_pc = seed_pc
        self._seed_reg = seed_reg

    def rerun(self, input_bytes, observe_points=None):
        out, seed_val = self._scripted[min(self._i, len(self._scripted) - 1)]
        self._i += 1
        obs = ObservedState(
            pc=self._seed_pc, when="after",
            regs={self._seed_reg: seed_val}, mem={})
        return RerunResult(output=out, observations=(obs,))


_OBS = [ObservePoint(pc=0x1000, when="after", capture=("regs",), regs=("x0",))]
_SEEDS = [SeedSpec(name="seed", kind="reg", reg="x0", observe_pc=0x1000)]
_WINDOW = (0x2000, 0x2100)


def test_distinct_parity_100_distinct_outputs_yields_exact():
    # 100 reruns, each a DISTINCT seed → DISTINCT output; F predicts exactly.
    scripted = [(bytes([i, i, i, i]), i) for i in range(100)]
    adapter = _FakeAdapter(scripted)

    def predict(seed_values, _input):
        s = int(seed_values["seed"], 16)
        return bytes([s, s, s, s])

    rep = collect_real_gold(
        adapter, _OBS, _SEEDS, loop_input=b"x", predict=predict, window=_WINDOW,
        distinct_output_floor=100, max_reruns=200)
    assert rep.floor_met is True
    assert rep.observed_distinct == 100
    assert rep.verdict_hint == "EXACT"
    assert rep.parity_report.independent_pass == 100


def test_nonce_repeating_output_below_floor_is_insufficient_variance_not_unclosable():
    # time(NULL)-style: every rerun produces the SAME output (the nonce did not move
    # within the budget), yet every prediction matches. The honest verdict is
    # INSUFFICIENT_VARIANCE (NOT UNCLOSABLE — F isn't wrong; the cohort is thin).
    scripted = [(b"\xAA\xBB\xCC\xDD", 7)] * 50    # identical output, identical seed
    adapter = _FakeAdapter(scripted)

    def predict(seed_values, _input):
        return b"\xAA\xBB\xCC\xDD"

    rep = collect_real_gold(
        adapter, _OBS, _SEEDS, loop_input=b"x", predict=predict, window=_WINDOW,
        distinct_output_floor=100, max_reruns=50)
    assert rep.floor_met is False
    assert rep.observed_distinct == 1
    assert rep.verdict_hint == INSUFFICIENT_VARIANCE
    assert rep.reruns_spent == 50               # budget spent
    # Crucially NOT a false UNCLOSABLE / EXACT.
    assert rep.verdict_hint != "UNCLOSABLE"
    assert rep.verdict_hint != "EXACT"


def test_collector_keeps_going_until_distinct_floor_met():
    # The output repeats for a while, then starts varying → the collector keeps
    # rerunning past the repeats until the distinct floor is reached.
    scripted = ([(b"\x00\x00", 0)] * 5
                + [(bytes([i, i]), i) for i in range(1, 6)])
    adapter = _FakeAdapter(scripted)

    def predict(seed_values, _input):
        s = int(seed_values["seed"], 16)
        return bytes([s, s])

    rep = collect_real_gold(
        adapter, _OBS, _SEEDS, loop_input=b"x", predict=predict, window=_WINDOW,
        distinct_output_floor=5, max_reruns=20)
    assert rep.floor_met is True
    assert rep.observed_distinct >= 5
    # spent more than 5 reruns because of the leading repeats
    assert rep.reruns_spent > 5


def test_negative_input_varying_floor_met_first_batch_no_extra_reruns():
    # A normal input-varying target: every rerun already distinct → floor met without
    # extra collect-until iterations. Behaviour == today's single batch.
    scripted = [(bytes([i, i, i]), i) for i in range(3)]
    adapter = _FakeAdapter(scripted)

    def predict(seed_values, _input):
        s = int(seed_values["seed"], 16)
        return bytes([s, s, s])

    rep = collect_real_gold(
        adapter, _OBS, _SEEDS, loop_input=b"x", predict=predict, window=_WINDOW,
        distinct_output_floor=3, max_reruns=200)
    assert rep.floor_met is True
    assert rep.reruns_spent == 3
    assert rep.verdict_hint == "EXACT"


def test_floor_defaults_from_parity_env_when_not_given():
    # No explicit floor → defaults from the parity gate's env
    # (UTOV_SETUP_SYMEX_PARITY_VECTORS), keeping the variance floor and the
    # independence floor coupled.
    scripted = [(bytes([i]), i) for i in range(10)]
    adapter = _FakeAdapter(scripted)

    def predict(seed_values, _input):
        return bytes([int(seed_values["seed"], 16)])

    rep = collect_real_gold(
        adapter, _OBS, _SEEDS, loop_input=b"x", predict=predict, window=_WINDOW,
        env={"UTOV_SETUP_SYMEX_PARITY_VECTORS": "4"})
    assert rep.floor == 4
    assert rep.floor_met is True


def test_wrong_F_below_floor_is_still_insufficient_variance():
    # Even when F is wrong (mismatches), below the variance floor the collector
    # reports INSUFFICIENT_VARIANCE (cohort thin) — it does not pre-empt with BLOCK
    # off too-few distinct outputs (the variance gate is the honest stop here).
    scripted = [(b"\x01\x02", 7)] * 3
    adapter = _FakeAdapter(scripted)

    def predict(seed_values, _input):
        return b"\xFF\xFF"          # wrong

    rep = collect_real_gold(
        adapter, _OBS, _SEEDS, loop_input=b"x", predict=predict, window=_WINDOW,
        distinct_output_floor=10, max_reruns=3)
    assert rep.floor_met is False
    assert rep.verdict_hint == INSUFFICIENT_VARIANCE


def test_floor_met_but_F_wrong_is_block_not_exact():
    # Floor met (output-diverse), but F mismatches → check_parity_vectors decides
    # BLOCK (the transform is wrong), NOT a false EXACT.
    scripted = [(bytes([i, i]), i) for i in range(5)]
    adapter = _FakeAdapter(scripted)

    def predict(seed_values, _input):
        return b"\xFF\xFF"          # always wrong

    rep = collect_real_gold(
        adapter, _OBS, _SEEDS, loop_input=b"x", predict=predict, window=_WINDOW,
        distinct_output_floor=5, max_reruns=20)
    assert rep.floor_met is True
    assert rep.verdict_hint == "BLOCK"


def test_empty_output_is_honest_block_not_a_spin():
    adapter = _FakeAdapter([(b"", 0)])

    def predict(seed_values, _input):
        return b""

    rep = collect_real_gold(
        adapter, _OBS, _SEEDS, loop_input=b"x", predict=predict, window=_WINDOW,
        distinct_output_floor=5, max_reruns=10)
    assert rep.verdict_hint == "BLOCK"
    assert "EMPTY output" in rep.detail


def test_mem_seed_form_captured():
    # A mem@addr seed form: the seed value comes from ObservedState.mem.
    class _MemAdapter:
        def __init__(self):
            self._i = 0

        def rerun(self, input_bytes, observe_points=None):
            i = self._i
            self._i += 1
            obs = ObservedState(pc=0x1000, when="after", regs={},
                                mem={0x5000: bytes([i, i])})
            return RerunResult(output=bytes([i, i]), observations=(obs,))

    seeds = [SeedSpec(name="s", kind="mem", addr=0x5000, length=2, observe_pc=0x1000)]

    def predict(seed_values, _input):
        return bytes.fromhex(seed_values["s"])

    rep = collect_real_gold(
        _MemAdapter(), _OBS, seeds, loop_input=b"x", predict=predict,
        window=_WINDOW, distinct_output_floor=4, max_reruns=10)
    assert rep.floor_met is True
    assert rep.verdict_hint == "EXACT"


def test_report_to_dict_shapes():
    scripted = [(bytes([i]), i) for i in range(3)]
    adapter = _FakeAdapter(scripted)

    def predict(seed_values, _input):
        return bytes([int(seed_values["seed"], 16)])

    rep = collect_real_gold(
        adapter, _OBS, _SEEDS, loop_input=b"x", predict=predict, window=_WINDOW,
        distinct_output_floor=3, max_reruns=10)
    d = rep.to_dict()
    assert d["kind"] == "real_gold_report"
    assert d["observed_distinct"] == 3
    assert d["floor_met"] is True
