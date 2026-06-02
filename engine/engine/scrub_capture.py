"""Scrub Capture — recover a write-then-wiped transient secret (TOOL_SCRUB_CAPTURE.md).

A secret (key / IV / intermediate) is written to the stack, used, then wiped
(zeroed / constant-filled) so it does not persist. last-write-wins sees only the
wipe; a point-in-time snapshot is taken after the secret is gone. But the value
WAS in the trace for an instant — if writes are recorded as a stream, that
instant is in the write history.

This is the UNKNOWN-target capability (the missing piece): detect write-then-wipe
over the write-event stream and CAPTURE the pre-wipe value as a ranked candidate,
without knowing what it is. It plugs into CVD via the Registry (CVD_PLUS §2-§5):
a ScrubGenerator emits `recovered_transient` candidates that ride E3's surprise
spike; a ScrubVerifier confirms one ONLY when it matches the run's oracle —
otherwise it stays a ranked, reported candidate, never asserted as "the key".

Limits (honest, TOOL_SCRUB_CAPTURE §5): cannot recover what was never observed
(wipe inside an un-traced callee → OPAQUE_CALLEE); the entropy heuristic can miss
a structured (low-entropy) secret and can false-positive on random-looking
non-secrets — hence "ranked candidate, verify before trust".
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .cvd import (
    BASE_VALUE,
    Candidate,
    CandidateGenerator,
    Verdict,
    Verifier,
    VStatus,
)
from .types import Instruction

LIFETIME_WINDOW = 64          # max instrs between write and wipe to count as transient
MIN_DISTINCT_RATIO = 0.75     # "high-entropy / random-looking" heuristic for V_old
MIN_SECRET_LEN = 4


@dataclass(frozen=True)
class ScrubCandidate:
    value: bytes               # the recovered pre-wipe bytes
    region: tuple[int, int]    # (addr, size)
    wrote_at: tuple[int, int]  # (idx, pc) — producer of the secret
    wiped_at: tuple[int, int]  # (idx, pc) — the scrub
    entropy: float

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value.hex(), "region": [f"0x{self.region[0]:x}", self.region[1]],
                "wrote_at": [self.wrote_at[0], f"0x{self.wrote_at[1]:x}"],
                "wiped_at": [self.wiped_at[0], f"0x{self.wiped_at[1]:x}"],
                "entropy": round(self.entropy, 3)}


def _entropy(b: bytes) -> float:
    if not b:
        return 0.0
    from collections import Counter
    n = len(b)
    return -sum((c / n) * math.log2(c / n) for c in Counter(b).values())


def _is_ascii(b: bytes) -> bool:
    return all(0x20 <= x < 0x7F for x in b)


def _looks_secret(b: bytes) -> bool:
    return (len(b) >= MIN_SECRET_LEN
            and len(set(b)) / len(b) >= MIN_DISTINCT_RATIO
            and not _is_ascii(b))


def _is_wipe(b: bytes) -> bool:
    # zero-fill or any constant fill (memset-like)
    return len(set(b)) == 1


def scrub_capture(
    items: Iterable[Instruction],
    *,
    lifetime_window: int = LIFETIME_WINDOW,
    stack_band: tuple[int, int] | None = None,
) -> list[ScrubCandidate]:
    """Detect write-then-wipe over the write stream; recover the pre-wipe value.

    Fires only on the CONJUNCTION (high-entropy V_old, wipe V_new, short lifetime,
    and — if a stack_band is given — on-stack), so a genuine secret-wipe is rare
    and benign slot reuse does not flood the result. Ranked by entropy.
    """
    last: dict[int, tuple[bytes, int, int, int]] = {}   # addr -> (val, idx, pc, size)
    out: list[ScrubCandidate] = []
    for ins in items:
        for op in ins.mem:
            if op.rw != "w" or op.size <= 0:
                continue
            vbytes = bytes((op.val >> (8 * k)) & 0xFF for k in range(op.size))
            prev = last.get(op.addr)
            if prev is not None and prev[3] == op.size:
                v_old, j, q, _ = prev
                on_stack = stack_band is None or stack_band[0] <= op.addr < stack_band[1]
                if (_is_wipe(vbytes) and _looks_secret(v_old) and on_stack
                        and (ins.idx - j) <= lifetime_window):
                    out.append(ScrubCandidate(
                        value=v_old, region=(op.addr, op.size),
                        wrote_at=(j, q), wiped_at=(ins.idx, ins.pc),
                        entropy=_entropy(v_old)))
            last[op.addr] = (vbytes, ins.idx, ins.pc, op.size)
    out.sort(key=lambda s: -s.entropy)
    return out


class ScrubGenerator(CandidateGenerator):
    """Emits recovered-transient candidates from write-then-wipe detection."""
    name = "scrub_gen"; version = "1"; owner = "core"; kind = "recovered_transient"

    def __init__(self, stack_band: tuple[int, int] | None = None,
                 lifetime_window: int = LIFETIME_WINDOW):
        self.stack_band = stack_band
        self.lifetime_window = lifetime_window

    def generate(self, state) -> list[Candidate]:
        cands: list[Candidate] = []
        for sc in scrub_capture(state.scoped_items(), stack_band=self.stack_band,
                                lifetime_window=self.lifetime_window):
            cands.append(Candidate(
                "recovered_transient", sc.region[0], "recovered_transient_secret",
                f"write-then-wipe high-entropy secret @0x{sc.region[0]:x} "
                f"(H={sc.entropy:.2f}, wiped@idx {sc.wiped_at[0]})",
                base_value=BASE_VALUE.get("snapshot_eq_expected", 5.0),  # rides E3 spike
                provenance="observed", payload=sc.to_dict()))
        return cands


class ScrubVerifier(Verifier):
    """Confirms a recovered transient ONLY when it matches the run's oracle; else
    it stays a ranked, reported candidate (eliminated as unconfirmed, not trusted)."""
    name = "scrub_verifier"; version = "1"; owner = "core"

    def applies(self, c, state) -> bool:
        return c.kind == "recovered_transient"

    def verify(self, c, state) -> Verdict:
        value = bytes.fromhex(c.payload.get("value", ""))
        if value and value == state.expected:
            return Verdict(VStatus.TERMINAL, terminal_kind="RECOVERED_SECRET",
                           success=True, evidence=c.payload, located_base=c.locus)
        return Verdict(VStatus.ELIMINATED, reason="unconfirmed_secret", evidence=c.payload)


def register_scrub(registry, *, stack_band=None, lifetime_window=LIFETIME_WINDOW):
    """Append the scrub tool to a Registry (driver code does not change)."""
    return (registry.register(ScrubGenerator(stack_band, lifetime_window))
                    .register(ScrubVerifier()))


__all__ = ["ScrubCandidate", "scrub_capture", "ScrubGenerator", "ScrubVerifier",
           "register_scrub", "LIFETIME_WINDOW"]
