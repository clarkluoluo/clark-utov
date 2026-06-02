"""Forward register flow + backward producer search + semantic op classification.

Algorithms ported from algokiller-plugin tools/search/search.c
(Sprint 1 / regflow, producer, semop — MIT cloudza 2026, see NOTICE).

Why these belong here, not in S3/S4 stage files:
  - They are trace-format-agnostic primitives over Instruction streams
  - S4 (backward data-flow slice) uses producer_backward() as its inner loop
  - LLM hypothesis prompting (S6) uses regflow_forward() to assemble compact
    "this register evolved like {0x...→0x...→0x...}" context blocks
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from .types import Instruction


@dataclass(frozen=True)
class RegflowHit:
    idx: int
    reg: str
    value: int
    mnemonic: str


def regflow_forward(
    items: Iterable[Instruction],
    reg: str,
    idx_from: int = 0,
    idx_to: int | None = None,
    limit: int = 100,
) -> Iterator[RegflowHit]:
    """Yield every (idx, value) where the given register was written.

    Walks forward through items. Each Instruction whose regs_write contains
    `reg` produces one hit. Stops at idx_to (exclusive) or when limit reached.
    """
    emitted = 0
    for ins in items:
        if ins.idx < idx_from:
            continue
        if idx_to is not None and ins.idx >= idx_to:
            return
        if emitted >= limit:
            return
        if reg in ins.regs_write:
            yield RegflowHit(idx=ins.idx, reg=reg, value=ins.regs_write[reg], mnemonic=ins.mnemonic)
            emitted += 1


@dataclass(frozen=True)
class ProducerHit:
    idx: int
    reg: str
    value: int
    mnemonic: str


def producer_backward(
    items: list[Instruction],
    target_value: int,
    sink_idx: int,
    max_back: int = 100_000,
) -> ProducerHit | None:
    """Backward scan: find the most recent instruction whose write produced
    `target_value` into any register, starting from sink_idx-1.

    Returns the matching ProducerHit, or None if nothing within max_back.
    Items MUST be a list with stable random access (this primitive walks
    backward; pre-materialize the trace if you only have an iterator).
    """
    start = min(sink_idx - 1, len(items) - 1)
    end = max(0, start - max_back + 1)
    for i in range(start, end - 1, -1):
        ins = items[i]
        for reg, val in ins.regs_write.items():
            if val == target_value:
                return ProducerHit(idx=ins.idx, reg=reg, value=val, mnemonic=ins.mnemonic)
    return None


# -- Semantic classification of an instruction (semop). Lightweight; useful for
#    LLM context blocks ("this section is mostly memory_load + crypto_candidate"). --

_BRANCH_3 = frozenset({"cbz", "ret", "tbz"})
_ALU_3    = frozenset({"add", "sub", "and", "orr", "mul", "neg", "lsl", "lsr", "asr", "ror"})
_CMP_3    = frozenset({"cmp", "tst"})


def classify_semop(mnemonic: str) -> str:
    """Map an ARM64 mnemonic string to a high-level semantic class.

    Classes: zero | crypto_candidate | hash_loop_candidate | stack_save |
             stack_restore | memory_load | memory_store | branch | addr_calc |
             data_move | alu | compare | unknown
    """
    # The mnemonic comes pre-parsed; take the first token only.
    first = mnemonic.split(None, 1)
    if not first:
        return "unknown"
    mnem = first[0]
    rest = first[1] if len(first) > 1 else ""

    # branch family
    if mnem == "b" or mnem in ("bl", "br", "blr") or mnem.startswith("b."):
        return "branch"
    if mnem in _BRANCH_3 or mnem == "cbnz" or mnem == "tbnz":
        return "branch"

    # stp / ldp — distinguish frame save/restore from generic mem
    if mnem == "stp":
        if "x29, x30" in rest or "fp, lr" in rest or "[sp" in rest:
            return "stack_save"
        return "memory_store"
    if mnem == "ldp":
        if "x29, x30" in rest or "fp, lr" in rest or "[sp" in rest:
            return "stack_restore"
        return "memory_load"

    # madd/msub — Bernstein / DJB / FNV polynomial accumulator pattern
    if mnem in ("madd", "msub", "smaddl"):
        return "hash_loop_candidate"

    # eor/xor — distinguish "eor x0, x0, x0" (zero) from "eor xN, xM, xK" (crypto)
    if mnem in ("eor", "xor"):
        toks = [t.rstrip(",") for t in rest.split()]
        if len(toks) >= 3 and toks[0] == toks[1] == toks[2]:
            return "zero"
        return "crypto_candidate"

    # memory ops
    if mnem.startswith("ldr") or mnem in ("ldur",):
        return "memory_load"
    if mnem.startswith("str") or mnem in ("stur",):
        return "memory_store"

    # address calc
    if mnem in ("adrp", "adr"):
        return "addr_calc"

    # data movement
    if mnem.startswith("mov"):
        return "data_move"

    # ALU
    if mnem in _ALU_3:
        return "alu"

    # compare
    if mnem in _CMP_3 or mnem == "subs":
        return "compare"

    return "unknown"
