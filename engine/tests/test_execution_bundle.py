"""Tests for spec #5 — execution bundle exporter (engine.execution_bundle).

Fixtures (spec §Fixtures), all with a STUB RunnerAdapter (no real runner):
  (a) multi-point capture → bundle with all named observations under one exec_id;
  (b) a declarative derived field computed + a failing extractor surfaced in
      derived_errors;
  (c) same-execution violation → same_execution=False (loud);
  (d) write_json carries the export stamp.

Plus the GENERAL-FIRST proof-point: a TC2-shaped derived map (time_seed /
rand_words / src32 / sink32 / output) is just ONE instantiation — the design
target is the generic mechanism, not those fields.
"""

from __future__ import annotations

import struct
from pathlib import Path

from engine.execution_bundle import DERIVED_NONE, CaptureBundle, capture_bundle
from engine.export_stamp import is_utov_export, load_stamped_json
from engine.runner_client import (
    ObservedState,
    ObservePoint,
    RerunResult,
    RunnerAdapter,
)


class _StubRunner(RunnerAdapter):
    """A fake runner that returns a fixed RerunResult — no real target.

    ``observations`` is the exact tuple of ObservedState to return; ``output`` is
    the produced bytes; ``truncated`` mirrors a cap-hit ledger. The stub does NOT
    look at observe_points (the bundle matches by (pc, when) on whatever we emit),
    so a test fully controls which spec names resolve."""

    def __init__(self, *, output, observations=(), truncated=False):
        self._output = bytes(output)
        self._observations = tuple(observations)
        self._truncated = truncated

    def metadata(self):  # pragma: no cover - not exercised by capture_bundle
        raise NotImplementedError

    def rerun(self, input_bytes, observe_points=None):
        return RerunResult(
            output=self._output,
            observations=self._observations,
            truncated=self._truncated,
        )


def _spec():
    """A multi-point named observation spec (generic, not TC2-specific)."""
    return {
        "entry": ObservePoint(pc=0x1000, when="before", capture=("regs",)),
        "sink": ObservePoint(pc=0x2000, when="after", capture=("regs", "mem")),
    }


# (a) ---------------------------------------------------------------------------
def test_multi_point_capture_under_one_exec_id():
    obs = (
        ObservedState(pc=0x1000, when="before", regs={"x0": 0xAA}, mem={}),
        ObservedState(pc=0x2000, when="after", regs={"x1": 0xBB},
                      mem={0x40000: b"\xde\xad\xbe\xef"}),
    )
    runner = _StubRunner(output=b"\x01\x02\x03\x04", observations=obs)

    b = capture_bundle(runner, b"in", _spec())

    # both named points resolved, keyed by MEANING not pc
    assert set(b.observations) == {"entry", "sink"}
    assert b.observations["entry"].regs == {"x0": 0xAA}
    assert b.observations["sink"].mem == {0x40000: b"\xde\xad\xbe\xef"}
    # one execution, same-execution true by construction
    assert b.same_execution is True
    assert b.same_execution_detail is None
    assert b.exec_identity["exec_id"]  # a single execution token
    d = b.to_dict()
    assert d["input"] == b"in".hex()
    assert d["output"] == b"\x01\x02\x03\x04".hex()
    assert d["observations"]["sink"]["mem"] == {"0x40000": "deadbeef"}
    # clean bundle carries no degenerate noise
    assert "derived_errors" not in d
    assert "same_execution_detail" not in d


def test_spec_point_without_observation_is_absent_not_forged():
    # runner emits only the entry point; the 'sink' spec point produced nothing
    obs = (ObservedState(pc=0x1000, when="before", regs={"x0": 1}, mem={}),)
    runner = _StubRunner(output=b"\x00", observations=obs)
    b = capture_bundle(runner, b"x", _spec())
    assert "entry" in b.observations
    assert "sink" not in b.observations  # absence, not a fabricated state


# (b) ---------------------------------------------------------------------------
def test_declarative_derived_computed_and_failing_surfaced():
    obs = (
        ObservedState(pc=0x2000, when="after", regs={"x0": 0x11223344}, mem={}),
    )
    runner = _StubRunner(output=b"\xaa\xbb", observations=obs)

    def good(bundle: CaptureBundle):
        return bundle.observations["sink"].regs["x0"]

    def bad(bundle: CaptureBundle):
        # references a name that does not exist → KeyError → surfaced
        return bundle.observations["missing"].regs["x9"]

    b = capture_bundle(runner, b"i", _spec(),
                       derived={"x0_val": good, "broken": bad})

    assert b.derived["x0_val"] == 0x11223344
    # failing extractor: field is null AND the reason is surfaced (never dropped)
    assert b.derived["broken"] is None
    assert "broken" in b.derived_errors
    assert "KeyError" in b.derived_errors["broken"]
    d = b.to_dict()
    assert d["derived"]["x0_val"] == 0x11223344
    assert d["derived"]["broken"] is None
    assert d["derived_errors"]["broken"]


def test_derived_none_default_is_empty():
    runner = _StubRunner(output=b"\x00")
    b = capture_bundle(runner, b"x", {}, derived=DERIVED_NONE)
    assert b.derived == {}
    assert b.derived_errors == {}


# (c) ---------------------------------------------------------------------------
def test_same_execution_violation_is_loud():
    # An adapter that forged mem from TWO distinct executions: the snapshots carry
    # two distinct execution_id tokens already, so capture_bundle does NOT overwrite
    # them and assert_same_execution fires → same_execution=False + loud detail.
    # We hand a stub whose mem-snapshots already disagree by injecting pre-stamped
    # snapshots via a runner that returns observations whose mem maps to >=2 tokens.
    # Simplest construction: monkeypatch mem_snapshots_from_rerun to yield two
    # cross-rerun snapshots — but per A8① we instead use the real path: two Observed
    # states whose mem produce snapshots, then force distinct tokens by patching.
    from dataclasses import replace as _replace

    from engine import execution_bundle as eb
    from engine.types import MemSnapshot

    obs = (
        ObservedState(pc=0x2000, when="after", regs={}, mem={0x10: b"\x01"}),
        ObservedState(pc=0x2000, when="after", regs={}, mem={0x20: b"\x02"}),
    )
    runner = _StubRunner(output=b"\xff", observations=obs)

    real = eb.mem_snapshots_from_rerun

    def cross_rerun(result):
        snaps = real(result)
        # stamp the two snapshots with DISTINCT pre-existing tokens (cross-rerun);
        # capture_bundle must leave already-tokened snapshots untouched (mask guard)
        out = []
        for i, s in enumerate(snaps):
            out.append(_replace(s, execution_id=f"rerun#{i}"))
        return out

    eb.mem_snapshots_from_rerun = cross_rerun
    try:
        b = capture_bundle(runner, b"i", _spec())
    finally:
        eb.mem_snapshots_from_rerun = real

    assert b.same_execution is False
    assert b.same_execution_detail is not None
    assert b.same_execution_detail["violation"] == "cross_rerun"
    d = b.to_dict()
    assert d["same_execution"] is False
    assert "same_execution_detail" in d
    # sanity: MemSnapshot is what flowed through
    assert isinstance(real(RerunResult(output=b"", observations=obs))[0], MemSnapshot)


# (d) ---------------------------------------------------------------------------
def test_write_json_carries_export_stamp(tmp_path: Path):
    runner = _StubRunner(output=b"\x01\x02",
                         observations=(ObservedState(pc=0x1000, when="before",
                                                     regs={"x0": 7}, mem={}),))
    b = capture_bundle(runner, b"ab", _spec(),
                       exec_identity={"case": "tc-demo"})
    out = tmp_path / "bundle.json"
    text = b.write_json(str(out), ts="2026-06-03T00:00:00Z")

    assert is_utov_export(text)
    assert out.read_text() == text
    header, payload = load_stamped_json(text)
    assert header is not None
    assert header["exported_by"] == "engine.execution_bundle.capture_bundle"
    assert header["ts"] == "2026-06-03T00:00:00Z"
    assert header["exec_identity"] == {"case": "tc-demo"}
    assert payload["output"] == b"\x01\x02".hex()
    assert payload["observations"]["entry"]["regs"]["x0"] == "0x7"


# proof-point (TC2-shaped derived map is ONE instantiation, not the design) -----
def test_tc2_shaped_derived_map_is_just_one_instantiation():
    # observed: time_seed reg + 32 rand words in mem + src32 + sink snapshot
    rand_words = list(range(32))
    rand_blob = b"".join(struct.pack("<I", w) for w in rand_words)
    obs = (
        ObservedState(pc=0x1000, when="before", regs={"x0": 0xC0FFEE}, mem={}),
        ObservedState(pc=0x2000, when="after", regs={"x1": 0xABCD},
                      mem={0x50000: rand_blob, 0x60000: b"SRC_32_BYTES____padpadpadpadpad!"}),
    )
    runner = _StubRunner(output=b"\xde\xad", observations=obs)

    spec = {
        "seed_pt": ObservePoint(pc=0x1000, when="before", capture=("regs",)),
        "sink_pt": ObservePoint(pc=0x2000, when="after", capture=("regs", "mem")),
    }

    def time_seed(b: CaptureBundle):
        return b.observations["seed_pt"].regs["x0"]

    def rand(b: CaptureBundle):
        blob = b.observations["sink_pt"].mem[0x50000]
        return list(struct.unpack("<32I", blob))

    def src32(b: CaptureBundle):
        return b.observations["sink_pt"].mem[0x60000][:32].hex()

    def sink32(b: CaptureBundle):
        return b.output.hex()

    b = capture_bundle(runner, b"seed-input", spec, derived={
        "time_seed": time_seed,
        "rand_words": rand,
        "src32": src32,
        "sink32": sink32,
        "output": lambda bn: bn.output.hex(),
    })

    assert b.same_execution is True
    assert b.derived["time_seed"] == 0xC0FFEE
    assert b.derived["rand_words"] == rand_words
    assert b.derived["src32"] == b"SRC_32_BYTES____padpadpadpadpad!"[:32].hex()
    assert b.derived["sink32"] == b"\xde\xad".hex()
    assert not b.derived_errors  # all extractors computed under one execution stamp
