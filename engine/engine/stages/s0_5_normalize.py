"""S0.5: regs_write reconstruction (additive normalizer pass).

Some concrete traces only dump the PRE-execution register snapshot per
instruction (e.g. a gci-style runner records `regs_read` = the registers as
they were *before* the instruction ran, and leaves `regs_write` empty). The
S3 data-flow graph keys reg dependencies off `regs_write`, so on such a trace
every producer link is missing and the DFG is useless.

This pass rebuilds `regs_write` from two independent signals and merges them:

  1. Frame differencing — write-set(i) = { r | prestate(i+1)[r] != prestate(i)[r] }.
     prestate(i) is `items[i].regs_read` (the pre-execution snapshot). The
     written value is the one visible in the *next* instruction's prestate.
  2. Disassembly of the destination register — recovers writes frame
     differencing can't see (the written value equals the old value), reading
     the destination from the already-present mnemonic text.

The pass is additive and idempotent: an instruction that already carries a
non-empty `regs_write` is returned unchanged, so traces from a fuller runner
are untouched. It produces NEW frozen Instruction objects (the originals are
immutable); the stage wrapper swaps them into the shared items list in place.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from ..types import Instruction

CODE_VERSION = "s0.5-v1"


# Opcodes whose first operand is NOT a destination register write.
_STORE_OPS = frozenset({
    "str", "strb", "strh", "stur", "sturb", "sturh", "stp", "stnp",
    "sttr", "stlr", "stlrb", "stlrh", "stxr", "stxrb", "stxrh",
})
# Opcodes that read their first register operand but write no register
# (comparisons, branches, barriers, system).
_NO_REG_DEST_OPS = _STORE_OPS | frozenset({
    "cmp", "cmn", "tst", "ccmp", "ccmn",
    "b", "bl", "br", "blr", "ret", "cbz", "cbnz", "tbz", "tbnz",
    "nop", "svc", "brk", "hlt", "dmb", "dsb", "isb", "yield",
})


def _norm_reg(tok: str) -> str:
    """Normalise a register token: strip decorations, map w-> x (w8 is the
    low half of x8, which is what the snapshot keys on)."""
    tok = tok.strip().strip("[]!").split(",")[0].strip()
    if not tok:
        return ""
    if tok[0] in "wW" and tok[1:].isdigit():
        return "x" + tok[1:]
    return tok.lower()


def _is_reg(tok: str) -> bool:
    tok = tok.strip()
    return bool(tok) and tok[0] in "xXwW" and tok[1:].split(".")[0].isdigit()


def _decode_dest_regs(mnemonic: str) -> list[str]:
    """Best-effort: the destination register(s) a mnemonic writes, from text.

    Returns [] for stores / branches / compares (no register destination).
    For `ldp x0, x1, [..]` returns both. For the common ALU/load form the
    first operand is the destination.
    """
    parts = mnemonic.strip().split(None, 1)
    if not parts:
        return []
    op = parts[0].lower()
    if op in _NO_REG_DEST_OPS or op.startswith("b."):
        return []
    if len(parts) < 2:
        return []
    operands = [o.strip() for o in parts[1].split(",")]
    if not operands or not _is_reg(operands[0]):
        return []
    dests = [_norm_reg(operands[0])]
    if op == "ldp" and len(operands) >= 2 and _is_reg(operands[1]):
        dests.append(_norm_reg(operands[1]))
    return [d for d in dests if d]


def reconstruct_regs_write(items: Iterable[Instruction]) -> list[Instruction]:
    """Return a new list with `regs_write` rebuilt where it was empty.

    Idempotent: instructions that already have a non-empty `regs_write` pass
    through unchanged.
    """
    items_list = list(items)
    n = len(items_list)
    out: list[Instruction] = []
    for i, ins in enumerate(items_list):
        if ins.regs_write:
            out.append(ins)  # already populated — leave it (idempotent)
            continue
        cur_pre = ins.regs_read
        nxt_pre = items_list[i + 1].regs_read if i + 1 < n else None
        writes: dict[str, int] = {}
        # (1) frame differencing against the next prestate snapshot
        if nxt_pre is not None:
            for r, v in nxt_pre.items():
                if cur_pre.get(r) != v:
                    writes[r] = v
        # (2) disasm-decoded destination — picks up same-value writes the
        #     frame diff cannot see (value known from the next snapshot).
        for dest in _decode_dest_regs(ins.mnemonic):
            if dest in writes:
                continue
            if nxt_pre is not None and dest in nxt_pre:
                writes[dest] = nxt_pre[dest]
            # last instruction (no next snapshot) with an unknowable post-value
            # is left out: its output is never read later in the trace, so it
            # cannot matter to a backward slice.
        out.append(replace(ins, regs_write=writes) if writes else ins)
    return out


def run(ctx) -> dict:
    """Stage entry: rebuild regs_write in place on the shared items list so
    S3 (which reads regs_write) sees the reconstructed producers."""
    items = ctx["items"]
    before_nonempty = sum(1 for ins in items if ins.regs_write)
    rebuilt = reconstruct_regs_write(items)
    # ctx["items"] is the same list object Core holds (core._items); replace
    # its contents in place so downstream stages and Core observe the rebuild.
    items[:] = rebuilt
    after_nonempty = sum(1 for ins in items if ins.regs_write)
    work = ctx.get("work")
    if work is not None:
        work.mark_stage_done("s0_5", CODE_VERSION)
    return {
        "stage": "s0_5",
        "instructions": len(items),
        "regs_write_before": before_nonempty,
        "regs_write_after": after_nonempty,
        "reconstructed": after_nonempty - before_nonempty,
    }
