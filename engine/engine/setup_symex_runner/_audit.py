"""Systematic decode-failure guard (capstone-oracle cross-check)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from ..types import Instruction
from ._base import is_control_flow, triton_available


# ---------------------------------------------------------------------------
# Systematic decode-failure guard (§2) — capstone-oracle cross-check
# ---------------------------------------------------------------------------
#
# The escape hatch fills a FEW genuine blind spots. A *bulk* of Triton decode
# failures is not a blind-spot scenario — it is a byte-feed config bug (endianness
# / arch / slice), and papering over it with hand S-expr fills is手搓 symex over a
# broken feed. The discriminator: capstone (the oracle) decodes the SAME bytes
# Triton choked on. ``feed_mismatch`` = #(Triton-failed AND capstone-succeeded).
# Any feed_mismatch, or a fail rate over the threshold, BLOCKs and forbids the
# escape hatch — fix the byte-feed, do not S-expr fill.

# Fraction of window steps Triton may fail to decode before we call it systematic.
DECODE_FAIL_RATE_THRESHOLD = 0.05

# Opaque-staging Phase 2(i) "+ record-a-line": cap on the number of forward-site
# samples (pc, addr, size) kept per run. The count (symbolic_forwards) is exact;
# only the sample list is bounded (invariant 4: big lists carry count + capped
# sample). Purely observational — never feeds a verdict gate.
SYMBOLIC_FORWARD_SITE_CAP = 64


@dataclass(frozen=True, slots=True)
class DecodeAudit:
    """Cross-check of Triton's decode coverage against capstone over a window.

    ``feed_mismatch`` is the load-bearing signal: bytes capstone disassembles but
    Triton can't are a byte-FEED bug (端序/arch/slice), never a true un-modeled
    opcode — so even one forbids the escape hatch (``systematic`` True). ``fail_rate``
    catches a broad decode failure even where capstone is unavailable. ``sample_pcs``
    are a few mismatching PCs to make the BLOCK note actionable."""

    total:         int
    decode_failed: int
    feed_mismatch: int
    sample_pcs:    tuple[int, ...] = ()
    capstone_ok:   bool = True       # False when capstone unavailable (oracle blind)

    @property
    def fail_rate(self) -> float:
        return (self.decode_failed / self.total) if self.total else 0.0

    @property
    def systematic(self) -> bool:
        """True iff the failures look like a feed bug, not a long-tail blind spot:
        any capstone-vs-Triton feed mismatch, or a fail rate over threshold."""
        return self.feed_mismatch > 0 or self.fail_rate > DECODE_FAIL_RATE_THRESHOLD

    @property
    def note(self) -> str:
        if not self.systematic:
            return ""
        pcs = ", ".join(f"0x{p:x}" for p in self.sample_pcs)
        return (
            f"systematic decode/feed inconsistency (capstone decodes {self.feed_mismatch} "
            f"byte(s) Triton can't; decode_fail_rate={self.fail_rate:.1%} of {self.total}) "
            f"= an endianness/arch/slice byte-FEED bug, NOT an escape-hatch scenario; "
            f"fix the byte-feed (see engine.byte_order), do NOT S-expr fill / use the "
            f"semantics-table here." + (f" sample pcs: {pcs}" if pcs else ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "total":         self.total,
            "decode_failed": self.decode_failed,
            "feed_mismatch": self.feed_mismatch,
            "fail_rate":     self.fail_rate,
            "systematic":    self.systematic,
            "capstone_ok":   self.capstone_ok,
            "sample_pcs":    [f"0x{p:x}" for p in self.sample_pcs],
            "note":          self.note,
            "kind":          "setup_symex_decode_audit",
        }


def audit_window_decode(
    items: Iterable[Instruction],
    *,
    window: tuple[int, int],
    window_kind: str = "pc",
    triton_probe: Callable[[bytes], bool] | None = None,
    max_samples: int = 8,
) -> DecodeAudit:
    """Count Triton decode failures over the window and cross-check with capstone.

    Control-flow steps are skipped (the runner never feeds them to Triton, so a
    branch is not a decode failure). For each remaining step: probe Triton; if it
    fails, ask capstone — a capstone success on the same bytes is a ``feed_mismatch``
    (the diagnostic class-8 signal). ``triton_probe`` defaults to the real Triton
    probe; tests inject a fake. Pure aside from the probes; never raises."""
    from ..byte_order import capstone_available, capstone_mnemonic

    lo, hi = int(window[0]), int(window[1])
    by_idx = window_kind == "idx"
    probe = triton_probe if triton_probe is not None else _default_triton_probe()
    cap_ok = capstone_available()
    total = failed = mismatch = 0
    samples: list[int] = []
    for ins in items:
        key = ins.idx if by_idx else ins.pc
        if not (lo <= key <= hi):
            continue
        if is_control_flow(ins.mnemonic):
            continue            # never fed to Triton — not a decode failure
        total += 1
        code = bytes(ins.bytes_)
        if probe(code):
            continue
        failed += 1
        # Triton failed — is it a true blind spot, or a feed bug? capstone is oracle.
        if cap_ok and capstone_mnemonic(code) is not None:
            mismatch += 1
            if len(samples) < max_samples:
                samples.append(ins.pc)
        elif len(samples) < max_samples:
            samples.append(ins.pc)
    return DecodeAudit(
        total=total, decode_failed=failed, feed_mismatch=mismatch,
        sample_pcs=tuple(samples), capstone_ok=cap_ok)


def _default_triton_probe() -> Callable[[bytes], bool]:
    """The real Triton decode probe (reuses :func:`dispatch_coverage.triton_decode_probe`).
    When Triton is unavailable, returns a probe that reports every opcode as
    decodable — the audit then finds zero failures (it cannot accuse a feed bug it
    cannot observe; the existing Triton-required paths still surface unavailability)."""
    if not triton_available():
        return lambda _code: True
    from ..dispatch_coverage import triton_decode_probe
    return triton_decode_probe()
