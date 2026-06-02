"""setup_symex.hybrid section (split from the monolithic module)."""
from __future__ import annotations


import enum
import os
import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..dataflow import classify_semop
from ..types import Instruction, MemSnapshot
from ..watch_first_write import (
    WatchFirstWriteConfig,
    WatchFirstWriteSpec,
    request_watch_first_write,
)


@dataclass(frozen=True, slots=True)
class HybridDecision:
    """Whether a single step may be concrete-synced or MUST be symbolic.

    The soundness rule: concrete-sync is only legal for steps that do not touch
    any SymVar. Anything reading a symbolic register — including a load/store
    pair whose address or value derives from one — must be modeled symbolically,
    and a decode-ok load must symbolize at its REAL effective address. Skipping
    that silently reads back concrete 0 and drops the dependency: symex is no
    longer sound."""

    idx:            int
    must_symbolize: bool
    reason:         str
    semop:          str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "idx":            self.idx,
            "must_symbolize": self.must_symbolize,
            "reason":         self.reason,
            "semop":          self.semop,
        }


_MEM_SEMOPS = frozenset({"memory_load", "memory_store", "stack_save", "stack_restore"})


def classify_hybrid_step(
    ins: Instruction,
    *,
    symbolic_regs: Iterable[str],
    symbolic_addrs: Iterable[int] = (),
) -> HybridDecision:
    """Decide concrete-sync vs symbolic-model for one instruction.

    ``must_symbolize`` is True when the instruction reads any symbolic register,
    OR is a memory op whose effective address / loaded address is symbolic. Only
    sym-independent instructions may be concrete-synced.

    Returns the decision AND its reason so the policy is auditable (the case's
    false root cause — "Triton doesn't support STP/LDP" — was debunked by
    modeling those pairs explicitly; the rule must be explicit too)."""
    sym_regs = set(symbolic_regs)
    sym_addrs = set(symbolic_addrs)
    semop = classify_semop(ins.mnemonic)

    read_syms = sorted(r for r in ins.regs_read if r in sym_regs)
    if read_syms:
        return HybridDecision(
            idx=ins.idx, must_symbolize=True, semop=semop,
            reason=f"reads symbolic register(s) {read_syms} — concrete-sync would "
                   f"drop the dependency",
        )
    # A memory op at a symbolic address must symbolize at the real EA, even if no
    # symbolic register is read directly (the EA itself carries the dependency).
    if semop in _MEM_SEMOPS:
        touched = sorted(
            op.addr for op in ins.mem if op.addr in sym_addrs
        )
        if touched:
            return HybridDecision(
                idx=ins.idx, must_symbolize=True, semop=semop,
                reason=f"memory op at symbolic EA {[f'0x{a:x}' for a in touched]} "
                       f"— a decode-ok load must symbolize at its real EA or it "
                       f"reads back concrete 0",
            )
    return HybridDecision(
        idx=ins.idx, must_symbolize=False, semop=semop,
        reason="touches no SymVar — concrete-sync is sound here",
    )


# ---------------------------------------------------------------------------
# Contract 4 — mem[] backing (the alias spine's substrate).
# ---------------------------------------------------------------------------


