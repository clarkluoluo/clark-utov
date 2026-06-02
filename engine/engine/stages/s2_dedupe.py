"""S2: dedupe blocks + block-aware fold (PLAN §12.4 / DECISIONS D-015).

PASS 1 — classic dedupe
  Hash key: SHA1 over the block's PC sequence.
  Many identical blocks collapse to one representative.

PASS 2 — block-aware fold
  Detect runs where consecutive blocks share the same block_hash. Collapse
  to "first block + sentinel + last block" via engine.fold.fold_runs.

Outputs (JSONL under stage_outputs/):
  s2_blocks.jsonl       — one row per unique block:
      {hash, representative_block_id, entry_pc, exit_pc, instr_count, executed_count}
  s2_executions.jsonl   — one row per surviving (post-fold) block execution:
      {kind: "block"|"sentinel",
       block_id, block_hash, instr_idx_start, instr_idx_end}    (kind=block)
      {kind: "sentinel", skipped_count, signature, first_idx, last_idx, window}
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..fold import FoldSentinel, FoldStats, fold_runs
from ..types import Instruction
from .s1_segment import BasicBlock, read_blocks as read_s1_blocks

CODE_VERSION = "s2-v1"

FOLD_THRESHOLD = 10


def _block_pcs(items_by_idx: dict[int, Instruction], block: BasicBlock) -> list[int]:
    return [items_by_idx[i].pc for i in range(block.instr_idx_start, block.instr_idx_end + 1)]


def _block_hash(pcs: list[int]) -> str:
    h = hashlib.sha1()
    for pc in pcs:
        h.update(pc.to_bytes(8, "little"))
    return h.hexdigest()


def run(ctx) -> dict:
    """ctx: {items: Iterable[Instruction], work: WorkDir}. Requires S1 to have run."""
    items = ctx["items"]
    work = ctx["work"]
    items_list = list(items) if not isinstance(items, list) else items
    by_idx = {ins.idx: ins for ins in items_list}

    blocks = read_s1_blocks(work)

    # --- Pass 1: dedupe ---
    hash_to_repr: dict[str, BasicBlock] = {}
    block_hashes: list[str] = []
    exec_counts: dict[str, int] = {}
    for b in blocks:
        pcs = _block_pcs(by_idx, b)
        bh = _block_hash(pcs)
        block_hashes.append(bh)
        exec_counts[bh] = exec_counts.get(bh, 0) + 1
        hash_to_repr.setdefault(bh, b)

    # Write blocks.jsonl
    blocks_path: Path = work.root / "stage_outputs" / "s2_blocks.jsonl"
    blocks_path.parent.mkdir(parents=True, exist_ok=True)
    with blocks_path.open("w", encoding="utf-8") as f:
        for bh, rep in hash_to_repr.items():
            f.write(json.dumps({
                "hash": bh,
                "representative_block_id": rep.block_id,
                "entry_pc": f"0x{rep.entry_pc:x}",
                "exit_pc":  f"0x{rep.exit_pc:x}",
                "instr_count": rep.instr_count,
                "executed_count": exec_counts[bh],
                "terminator_mnem": rep.terminator_mnem,
            }) + "\n")

    # --- Pass 2: block-aware fold over the block sequence ---
    # Signature = block hash. Same hash run = repeated block.
    block_stream = [(blocks[i], block_hashes[i]) for i in range(len(blocks))]
    fold_stats = FoldStats()
    folded = list(
        fold_runs(
            block_stream,
            signature_of=lambda be: be[1],
            threshold=FOLD_THRESHOLD,
            stats=fold_stats,
        )
    )

    exec_path: Path = work.root / "stage_outputs" / "s2_executions.jsonl"
    with exec_path.open("w", encoding="utf-8") as f:
        for item in folded:
            if isinstance(item, FoldSentinel):
                f.write(json.dumps({
                    "kind": "sentinel",
                    "skipped_count": item.skipped_count,
                    "signature": item.signature,
                    "first_idx": item.first_idx,
                    "last_idx": item.last_idx,
                    "window": item.window,
                }) + "\n")
            else:
                blk, bh = item
                f.write(json.dumps({
                    "kind": "block",
                    "block_id": blk.block_id,
                    "block_hash": bh,
                    "instr_idx_start": blk.instr_idx_start,
                    "instr_idx_end":   blk.instr_idx_end,
                }) + "\n")

    work.mark_stage_done("s2", CODE_VERSION)
    return {
        "stage": "s2",
        "unique_blocks": len(hash_to_repr),
        "total_block_executions": len(blocks),
        "fold_runs_applied": fold_stats.folds_applied,
        "fold_blocks_hidden": fold_stats.lines_skipped,
        "blocks_out": str(blocks_path),
        "executions_out": str(exec_path),
    }
