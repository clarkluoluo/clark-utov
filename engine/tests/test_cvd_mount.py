"""CVD mount policy + escalation triggers (CVD_MOUNT_POLICY.md). Acceptance for
A (default set + T-intake readers + RunManifest), B (MountPolicy editable +
checkpoint/resume + heavy budget gate), C (stall_pressure → level-jump), D
(EscalationRule preempt+escalate). Synthetic traces only.
"""

from __future__ import annotations

import json

from engine.cvd import (
    Artifact,
    CvdOutcome,
    Registry,
    Verdict,
    Verifier,
    VStatus,
    default_registry,
    resume,
    run_cvd,
)
from engine.cvd_mount import MountPolicy, default_policy
from engine.types import Instruction, MemOp


def _ins(idx, mem=()):
    return Instruction(idx=idx, pc=0x70000 + idx * 4, bytes_=b"\x00\x00\x00\x00",
                       mnemonic="str", regs_read={}, regs_write={}, mem=mem)


def _le(b):
    return int.from_bytes(b, "little")


EXPECTED = bytes([0x34, 0x15, 0x5f, 0xe9])


def _events(res):
    return [e["event"] for e in res.log]


# === A. default set + T-intake readers + RunManifest ========================

def test_scrub_in_default_registry():
    reg = default_registry()
    assert any(v.name == "scrub_verifier" for v in reg.verifiers)
    assert any(g.name == "scrub_gen" for g in reg.generators)
    assert any(r.name == "calltrace_reader" for r in reg.readers)
    assert any(r.name == "hook_dump_reader" for r in reg.readers)


def test_readers_parse_artifacts_and_manifest_is_announced():
    hook = json.dumps({"tag": "out", "pc_rva": "0x72b18", "x0": "0x72b18",
                       "mem_x0": "34155fe9"})
    calltrace = "BLR\t0x70a90\t0x72ecc\n"
    res = run_cvd([_ins(0)], EXPECTED,
                  artifacts=[Artifact("hook_dump", hook), Artifact("calltrace", calltrace)])
    reads = [e for e in res.log if e["event"] == "READ"]
    assert any(e["reader"].startswith("hook_dump_reader") and e["snapshots"] >= 1 for e in reads)
    assert any(e["reader"].startswith("calltrace_reader") and e["call_events"] >= 1 for e in reads)
    # RunManifest announced before drive
    m = res.manifest
    assert "sink_validator@1" in m["mounted"]
    assert "scrub_gen@1" in m["mounted"]
    assert any(t["trigger"] == "stall_pressure>θ" for t in m["armed_triggers"])


# === B. MountPolicy editable + checkpoint/resume + heavy budget gate ========

def test_agent_can_disable_a_tool():
    policy = default_policy()
    policy.disabled_tools.add("scrub_gen")
    res = run_cvd([_ins(0)], EXPECTED, policy=policy)
    assert "scrub_gen@1" not in res.manifest["mounted"]


def test_policy_persists_through_checkpoint_and_resume():
    trace = [_ins(0, mem=(MemOp("w", 0x1000 + i * 0x100, _le(bytes([i, i, i, i])), 4),))
             for i in range(1, 4)]
    policy = default_policy()
    policy.max_candidates = 1
    policy.stall_theta = 999.0           # an edited value we expect to survive resume
    paused = run_cvd(trace, EXPECTED, policy=policy)
    assert paused.outcome is CvdOutcome.BUDGET_EXHAUSTED
    cp = paused.checkpoint
    json.dumps(cp)                       # serializable
    assert cp["policy"]["stall_theta"] == 999.0     # the edit is in the checkpoint
    res = resume(cp, trace)              # policy restored from checkpoint
    assert res.outcome in (CvdOutcome.TERMINAL, CvdOutcome.SUCCESS,
                           CvdOutcome.EXTENSION_REQUEST, CvdOutcome.BUDGET_EXHAUSTED)


# === C. stall_pressure → level-jump =========================================

def _diverging_trace(n=12):
    # n distinct scratch write clusters, none == expected -> all eliminated, no
    # confirm, large frontier -> stall_pressure climbs.
    return [_ins(i, mem=(MemOp("w", 0x1000 + i * 0x100,
                               _le(bytes([i & 0xFF, (i + 1) & 0xFF, 2, 3])), 4),))
            for i in range(n)]


def test_stall_pressure_level_jumps_and_requests_heavy_when_unarmed():
    res = run_cvd(_diverging_trace(), EXPECTED)   # default policy: heavy NOT armed
    ev = _events(res)
    assert "LEVEL_JUMP" in ev and "PRUNE" in ev    # the level-jump fired
    assert res.outcome is CvdOutcome.EXTENSION_REQUEST
    assert res.extension_request["missing_kind"] == "verifier"
    assert "heavy" in res.extension_request["why"].lower()


class _HeavyStub(Verifier):
    name = "triton_symex"; version = "1"

    def applies(self, c, state):
        return c.kind == "heavy_probe"

    def verify(self, c, state):
        return Verdict(VStatus.TERMINAL, terminal_kind="X")


def test_stall_armed_heavy_hits_budget_gate():
    policy = default_policy()
    policy.heavy_armed = True
    policy.token_budget = 100            # tiny -> the heavy estimate exceeds it
    reg = default_registry().register(_HeavyStub())
    res = run_cvd(_diverging_trace(), EXPECTED, policy=policy, registry=reg)
    assert res.outcome is CvdOutcome.BUDGET_PAUSE
    assert res.budget_estimate is not None and res.budget_estimate["tokens"] > 100
    assert "LEVEL_JUMP" in _events(res)


# === D. signal-triggered EscalationRules preempt + escalate =================

def test_scrub_signal_rule_preempts():
    secret = bytes([0x9f, 0x3c, 0xe1, 0x42])
    trace = [
        _ins(0, mem=(MemOp("w", 0xbef00, _le(secret), 4),)),    # secret on "stack"
        _ins(1, mem=(MemOp("w", 0xbef00, 0, 4),)),              # wiped
    ]
    res = run_cvd(trace, secret)         # secret IS the oracle -> scrub recovers it
    esc = [e for e in res.log if e["event"] == "ESCALATE"]
    assert any(e["rule"].startswith("E3_scrub_preempt") for e in esc)


def test_opaque_boundary_rule_preempts_provenance_candidate():
    # a real sink -> spawns a provenance candidate -> E2 rule preempts it.
    trace = [_ins(0, mem=(MemOp("w", 0x2000, _le(EXPECTED), 4),))]
    res = run_cvd(trace, EXPECTED)
    esc = [e for e in res.log if e["event"] == "ESCALATE"]
    assert any(e["rule"].startswith("E2_opaque_boundary") for e in esc)
