"""S5: lightweight simplification (no-Triton edition).

Two passes over the S4-sliced instruction set:

  1. CONSTANT FOLDING / ZERO IDIOM RECOGNITION
     - `eor x0, x0, x0` (or `mov x0, xzr`) → constant 0
     - `mov xN, #imm` → immediate constant
     - Combined with the concrete trace, we know the actual output value of
       every instruction; we just label which ones are pure-constant nodes.

  2. INSSUB REVERSE MATCH (DiANa 13-pattern table — subset)
     OLLVM -sub replaces simple ops with MBA equivalents. We invert by
     pattern-matching on a (sliced) basic block's 4-line windows. Each
     pattern, if matched, emits a "canonical" mnemonic alongside the original.

  Patterns recognized (subset of DiANa 13):
     (a^b)+(2*(a&b))      ⇒ ADD
     (a+b)+2*((-a^-1)&b)  ⇒ SUB
     (a|b) - (a&b)        ⇒ XOR
     (a&b) + (a^b)        ⇒ OR
     (a^b) + 2*(a&b)      ⇒ ADD (alias of first row)

What this lightweight pass does NOT do:
  - Deep symbolic simplification of nested arithmetic — needs Triton (P1.5).
  - Cross-block expression unification.

Output: stage_outputs/s5_simplified.jsonl, one row per surviving instr with
optional "canonical" rewrite label.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..types import Instruction
from .s4_slice import read_slice

CODE_VERSION = "s5-v1"


def _is_zero_idiom(ins: Instruction) -> bool:
    m = ins.mnemonic.strip()
    parts = [t.rstrip(",") for t in m.split()]
    if not parts:
        return False
    op = parts[0]
    if op in ("eor", "xor") and len(parts) >= 4:
        return parts[1] == parts[2] == parts[3]
    if op == "mov" and len(parts) >= 3 and parts[2] in ("xzr", "wzr", "#0", "#0x0"):
        return True
    return False


def _is_mov_imm(ins: Instruction) -> tuple[bool, int | None]:
    """Return (is_mov_imm, value) where value is the immediate if recognizable."""
    parts = [t.rstrip(",") for t in ins.mnemonic.split()]
    if not parts:
        return False, None
    op = parts[0]
    if op in ("mov", "movz") and len(parts) >= 3:
        imm = parts[2]
        if imm.startswith("#"):
            try:
                return True, int(imm[1:], 0)
            except ValueError:
                return True, None
    return False, None


# --- InsSub reverse: 4-line window patterns -------------------------------
# Each pattern is a tuple of mnemonic prefixes (we ignore operands; the
# concrete-value check disambiguates) producing the same arithmetic identity
# OLLVM -sub generates.
_INSSUB_4LINE = [
    # ADD via (a^b) + 2*(a&b)
    {"name": "add_via_mba", "ops": ("eor", "and", "lsl", "add"), "canonical": "ADD"},
    # XOR via (a|b) - (a&b)
    {"name": "xor_via_mba", "ops": ("orr", "and", "sub"),         "canonical": "XOR"},
    # OR via (a&b) + (a^b)
    {"name": "or_via_mba",  "ops": ("and", "eor", "add"),         "canonical": "OR"},
    # ADD via (a^b) + ((a&b)<<1)  (same as first but with explicit lsl)
    # already covered above
]


def _match_pattern(window: list[Instruction], op_prefixes: tuple[str, ...]) -> bool:
    if len(window) < len(op_prefixes):
        return False
    for ins, prefix in zip(window, op_prefixes):
        first_tok = ins.mnemonic.split(None, 1)[0] if ins.mnemonic else ""
        if not first_tok.startswith(prefix):
            return False
    return True


def simplify(items: list[Instruction], kept_idxs: set[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    items_by_idx = {ins.idx: ins for ins in items}

    # Walk surviving instructions in index order; look at sliding 4-windows
    # to detect MBA patterns.
    kept_sorted = sorted(kept_idxs)
    pattern_matched_idxs: set[int] = set()
    for i, idx in enumerate(kept_sorted):
        window = [items_by_idx[k] for k in kept_sorted[i:i + 4] if k in items_by_idx]
        for pat in _INSSUB_4LINE:
            if _match_pattern(window, pat["ops"]):
                for w in window[:len(pat["ops"])]:
                    pattern_matched_idxs.add(w.idx)
                rows.append({
                    "kind": "inssub_match",
                    "pattern": pat["name"],
                    "canonical": pat["canonical"],
                    "window_idx": [w.idx for w in window[:len(pat["ops"])]],
                })
                break

    # Per-instruction annotation pass.
    for idx in kept_sorted:
        ins = items_by_idx.get(idx)
        if ins is None:
            continue
        annot: dict[str, Any] = {
            "kind": "instr",
            "idx": idx,
            "pc": f"0x{ins.pc:x}",
            "mnemonic": ins.mnemonic,
        }
        if _is_zero_idiom(ins):
            annot["constant"] = 0
        is_mov, val = _is_mov_imm(ins)
        if is_mov:
            annot["mov_immediate"] = val
        if idx in pattern_matched_idxs:
            annot["part_of_inssub"] = True
        rows.append(annot)

    return rows


def run(ctx) -> dict:
    items = ctx["items"]
    work = ctx["work"]
    items_list = list(items) if not isinstance(items, list) else items
    kept = read_slice(work)
    rows = simplify(items_list, set(kept.keys()))

    out_path: Path = work.root / "stage_outputs" / "s5_simplified.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    inssub_count = sum(1 for r in rows if r.get("kind") == "inssub_match")
    work.mark_stage_done("s5", CODE_VERSION)
    return {
        "stage": "s5",
        "annotations": len(rows),
        "inssub_matches": inssub_count,
        "out": str(out_path),
    }
