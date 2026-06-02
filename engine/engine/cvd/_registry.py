"""CVD built-in plugins, ROI scoring, and the default Registry."""
from __future__ import annotations

import math

from ..oracle_provenance import ProvenanceVerdict, trace_provenance
from ..oracle_sink import SinkVerdict, validate_sink
from ._model import (
    BASE_VALUE, _SURPRISE_CAP, _STALL_GAIN,
    Candidate, CvdState, Verdict, VStatus, Terminal,
    Verifier, CandidateGenerator, TerminalClassifier, Registry,
)


# --- ROI (CVD_DESIGN §11.1) -------------------------------------------------

def _surprise(signal: str, history: dict[str, int]) -> float:
    total = sum(history.values())
    cnt = history.get(signal, 0)
    freq = (cnt + 1.0) / (total + len(history) + 1.0)
    return 1.0 + min(_SURPRISE_CAP, -math.log(freq))


def _roi(c: Candidate, history: dict[str, int], stall: int) -> float:
    base = c.base_value or BASE_VALUE.get(c.signal, 1.0)
    # credibility (§C) modulates ORDER only — a low-credibility candidate is still
    # verified, just later; a high-credibility lead is checked early. Never trust.
    credibility = 1.0 + max(0.0, c.credibility)
    return base * _surprise(c.signal, history) * (1.0 + _STALL_GAIN * stall) * credibility


# --- built-in plugins (wrap the existing primitives) ------------------------

def _clusters(addrs: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a in sorted(set(addrs)):
        if out and a == out[-1][1]:
            out[-1] = (out[-1][0], a + 1)
        else:
            out.append((a, a + 1))
    return out


class SinkGenerator(CandidateGenerator):
    name = "sink_gen"; version = "1"; owner = "core"; kind = "sink"

    def generate(self, state: CvdState) -> list[Candidate]:
        items = state.scoped_items()
        L = len(state.expected)
        cands: list[Candidate] = []
        write_addrs = [op.addr + k for ins in items for op in ins.mem
                       if op.rw == "w" and op.size > 0 for k in range(op.size)]
        for lo, hi in _clusters(write_addrs):
            cands.append(Candidate("sink", lo, "write_cluster",
                                   f"contiguous write cluster [0x{lo:x},0x{hi:x})",
                                   base_value=BASE_VALUE["write_cluster"]))
        for snap in state.scoped_snapshots():
            data = bytes(snap.data)
            for off in range(len(data) - L + 1):
                if data[off:off + L] == state.expected:
                    base = snap.addr + off
                    cands.append(Candidate("sink", base, "snapshot_eq_expected",
                                           f"snapshot region == expected @0x{base:x}",
                                           base_value=BASE_VALUE["snapshot_eq_expected"]))
        return cands


class SinkValidatorVerifier(Verifier):
    """#1 sink-validator as a Verifier (applies to kind=sink)."""
    name = "sink_validator"; version = "1"; owner = "core"

    def applies(self, c, state) -> bool:
        return c.kind == "sink"

    def verify(self, c, state) -> Verdict:
        cand_base = None if c.signal == "snapshot_eq_expected" else c.locus
        sv = validate_sink(state.scoped_items(), state.expected,
                           candidate_base=cand_base, snapshots=state.scoped_snapshots())
        if sv.verdict is SinkVerdict.SINK_CONFIRMED:
            prov_cand = Candidate("provenance", sv.base, "provenance",
                                  f"trace producer of confirmed sink @0x{sv.base:x}",
                                  base_value=BASE_VALUE["provenance"])
            return Verdict(VStatus.CONFIRMED, located_base=sv.base, spawn=[prov_cand],
                           evidence={"located_via": sv.located_via})
        if sv.verdict is SinkVerdict.WRONG_SINK and sv.base is not None:
            sig = ("snapshot_eq_expected" if sv.located_via == "snapshot"
                   else "located_real_sink")
            return Verdict(VStatus.ELIMINATED, reason="wrong_sink", spawn=[Candidate(
                "sink", sv.base, sig, f"validator-located real sink @0x{sv.base:x}",
                base_value=BASE_VALUE.get(sig, 4.0))])
        return Verdict(VStatus.ELIMINATED, reason="no_match")


def _capability_for(prov) -> str:
    if prov.verdict is ProvenanceVerdict.OPAQUE_CALLEE:
        tgt = (f" target(s) {[hex(t) for t in prov.callee_targets]}"
               if prov.callee_targets else "")
        return (f"intra-callee / bridge observation at boundary "
                f"{[hex(p) for p in prov.boundary_pcs]}{tgt} — trace into the callee")
    if prov.verdict is ProvenanceVerdict.NEEDS_OBSERVATION:
        return f"watch native addresses: {prov.next_watch}"
    return prov.detail


class ProvenanceVerifier(Verifier):
    """#3 oracle_provenance as a Verifier (applies to kind=provenance, the
    expansion spawned when a sink is confirmed). Returns a TERMINAL verdict."""
    name = "oracle_provenance"; version = "3"; owner = "core"

    def applies(self, c, state) -> bool:
        return c.kind == "provenance"

    def verify(self, c, state) -> Verdict:
        prov = trace_provenance(state.scoped_items(), state.expected,
                                sink_base=c.locus, snapshots=state.scoped_snapshots())
        success = prov.verdict in (ProvenanceVerdict.CONTINUOUS_BUFFER,
                                   ProvenanceVerdict.STREAMING)
        return Verdict(VStatus.TERMINAL, terminal_kind=prov.verdict.value, success=success,
                       evidence=prov.to_dict(), located_base=c.locus,
                       capability_request=("" if success else _capability_for(prov)))


class DefaultTerminalClassifier(TerminalClassifier):
    name = "default_terminal"; version = "1"; owner = "core"

    def classify(self, state) -> Terminal | None:
        sv = validate_sink(state.scoped_items(), state.expected,
                           snapshots=state.scoped_snapshots())
        if sv.verdict is SinkVerdict.SINK_CONFIRMED:
            prov = trace_provenance(state.scoped_items(), state.expected,
                                    sink_base=sv.base, snapshots=state.scoped_snapshots())
            success = prov.verdict in (ProvenanceVerdict.CONTINUOUS_BUFFER,
                                       ProvenanceVerdict.STREAMING)
            return Terminal(prov.verdict.value, prov.to_dict(), sv.base,
                            "" if success else _capability_for(prov), success=success)
        if sv.verdict is SinkVerdict.OUTPUT_NOT_OBSERVABLE:
            return Terminal("OUTPUT_NOT_OBSERVABLE", sv.to_dict(), None, sv.detail)
        return None   # cannot classify -> the driver emits an EXTENSION_REQUEST


def default_registry() -> Registry:
    """CVD_MOUNT_POLICY §3 default set: T-intake readers + T0 + the default-ON T1
    (scrub is ON — the one MOUNT_POLICY §4 bug to fix) + the escalation rules."""
    from ..scrub_capture import ScrubGenerator, ScrubVerifier
    from ..cvd_mount import default_readers, default_rules
    from ..framing_transform import default_transforms
    reg = (Registry()
           .register(SinkGenerator())
           .register(SinkValidatorVerifier())
           .register(ProvenanceVerifier())
           .register(DefaultTerminalClassifier())
           .register(ScrubGenerator())          # §4: scrub default-ON
           .register(ScrubVerifier()))
    for r in default_readers():
        reg.register(r)
    for rule in default_rules():
        reg.register(rule)
    for tf in default_transforms():             # §10.6: encoding edge now has instances
        reg.register(tf)
    return reg
