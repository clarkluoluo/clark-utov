"""S1: structure trace into basic blocks.

Algorithm: walk the trace once.
A new block STARTS at:
  - the first instruction
  - the instruction immediately after a control-flow transfer
  - any instruction whose PC != prev.pc + 4 (target of an unseen branch)
A block ENDS at:
  - a control-flow instruction (terminator)
  - end of trace

Per-block record:
  block_id, entry_pc, exit_pc, instr_idx_start, instr_idx_end,
  instr_count, terminator_mnem

Output: JSONL stage_outputs/s1.jsonl (one block per line).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from ..types import Instruction

CODE_VERSION = "s1-v1"

# AArch64 mnemonics that terminate a basic block.
_TERMINATORS = frozenset({
    "b", "bl", "br", "blr", "ret",
    "cbz", "cbnz", "tbz", "tbnz",
})


def _is_terminator(mnem: str) -> bool:
    first = mnem.split(None, 1)[0] if mnem else ""
    if first in _TERMINATORS:
        return True
    return first.startswith("b.")    # conditional branches: b.eq, b.ne, ...


@dataclass(frozen=True)
class BasicBlock:
    block_id: int
    entry_pc: int
    exit_pc: int
    instr_idx_start: int
    instr_idx_end: int
    instr_count: int
    terminator_mnem: str    # empty if block ended at EOF without a CF instr


def segment(items: Iterable[Instruction]) -> list[BasicBlock]:
    """Pure function: produce the list of basic blocks. Useful for tests."""
    blocks: list[BasicBlock] = []
    next_block_id = 0

    cur_start_idx: int | None = None
    cur_start_pc:  int | None = None
    prev: Instruction | None = None

    def flush(terminator: Instruction | None) -> None:
        nonlocal next_block_id, cur_start_idx, cur_start_pc
        if cur_start_idx is None:
            return
        end = terminator if terminator is not None else prev
        assert end is not None
        assert cur_start_pc is not None
        blocks.append(BasicBlock(
            block_id=next_block_id,
            entry_pc=cur_start_pc,
            exit_pc=end.pc,
            instr_idx_start=cur_start_idx,
            instr_idx_end=end.idx,
            instr_count=end.idx - cur_start_idx + 1,
            terminator_mnem=end.mnemonic if terminator is not None else "",
        ))
        next_block_id += 1
        cur_start_idx = None
        cur_start_pc = None

    for ins in items:
        if cur_start_idx is None:
            cur_start_idx = ins.idx
            cur_start_pc  = ins.pc
        else:
            # PC discontinuity → start of a new block, but only if the prev
            # instruction didn't itself terminate (otherwise we already flushed).
            assert prev is not None
            if ins.pc != prev.pc + 4:
                # Flush as if EOF, then start fresh on this instruction.
                flush(None)
                cur_start_idx = ins.idx
                cur_start_pc  = ins.pc

        if _is_terminator(ins.mnemonic):
            flush(ins)

        prev = ins

    # Trailing block (no terminator)
    flush(None)
    return blocks


def run(ctx) -> dict:
    """Stage entry point. ctx is a dict containing:
        items: list[Instruction] or Iterable[Instruction]
        work:  WorkDir
    Returns a summary dict with block_count.
    """
    items = ctx["items"]
    work = ctx["work"]
    blocks = segment(items)

    out_path = work.root / "stage_outputs" / "s1.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for b in blocks:
            row = asdict(b)
            row["entry_pc"] = f"0x{b.entry_pc:x}"
            row["exit_pc"]  = f"0x{b.exit_pc:x}"
            f.write(json.dumps(row) + "\n")

    work.mark_stage_done("s1", CODE_VERSION)
    return {"stage": "s1", "blocks": len(blocks), "out": str(out_path)}


def read_blocks(work) -> list[BasicBlock]:
    """Helper: load S1 output from disk (for downstream stages)."""
    path: Path = work.root / "stage_outputs" / "s1.jsonl"
    blocks: list[BasicBlock] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            blocks.append(BasicBlock(
                block_id=o["block_id"],
                entry_pc=int(o["entry_pc"], 16),
                exit_pc=int(o["exit_pc"], 16),
                instr_idx_start=o["instr_idx_start"],
                instr_idx_end=o["instr_idx_end"],
                instr_count=o["instr_count"],
                terminator_mnem=o["terminator_mnem"],
            ))
    return blocks
