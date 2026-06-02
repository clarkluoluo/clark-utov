"""S3 alternative path: Triton symbolic execution (PLAN §3, IMPL_PLAN P1.5).

The default S3 (`s3_triton.run`) builds a *concrete* data-flow graph — it
captures producer / consumer relationships using the literal register values
the runner observed. That's enough for backward slicing (S4) but it loses the
symbolic structure of each instruction's computation, which S5 needs for
deeper algebraic simplification (e.g. canonicalizing nested XOR + ROR chains
that show up in SHA-256 σ/Σ).

This module wraps Triton (https://triton-library.github.io) so the engine can
optionally generate per-instruction symbolic expressions. Triton is an
**optional** dependency — the engine's core stays pure Python deps per the
engine-vs-fixture boundary rule. If Triton isn't importable on the host, this
module degrades cleanly and the caller falls back to the concrete DFG.

Opt in via:
  - CLI: `utov pipeline ... --symex triton`
  - env: `UTOV_SYMEX_MODE=triton`
  - programmatic: `core.run_stage("s3", symex_mode="triton")`

Output: `stage_outputs/s3_symex.jsonl`, one JSON row per instruction:
    {"idx": <int>, "pc": "0x...", "mnemonic": "...",
     "reg_exprs": {"x4": "(bvxor x1 x2)", ...},
     "reg_inputs": {"x4": ["x1", "x2"], ...}}
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CODE_VERSION = "s3-symex-v1"


# Lazy import so the engine package keeps working when Triton isn't installed.
try:  # pragma: no cover — exercised only on hosts with Triton
    from triton import ARCH, MODE, TritonContext  # type: ignore
    from triton import Instruction as TritonInstr  # type: ignore
    _TRITON_OK = True
    _TRITON_IMPORT_ERR: str | None = None
except Exception as _e:  # ImportError or any binding-load error
    _TRITON_OK = False
    _TRITON_IMPORT_ERR = f"{type(_e).__name__}: {_e}"


@dataclass
class SymexNode:
    idx: int
    pc: int
    mnemonic: str
    # AArch64 register name → symbolic AST as text. Empty when Triton couldn't
    # decode the opcode (e.g. NEON / SVE / privileged ins not in the model).
    reg_exprs: dict[str, str] = field(default_factory=dict)
    # AArch64 register name → list of input registers referenced by the expr.
    # We approximate with the trace's regs_read (Triton can produce a deeper
    # free-variable list, but we keep this lightweight for S5).
    reg_inputs: dict[str, list[str]] = field(default_factory=dict)


def is_available() -> bool:
    """True iff Triton bindings imported cleanly at module load."""
    return _TRITON_OK


def unavailable_reason() -> str | None:
    """When `is_available()` is False, returns the import error text. None when
    Triton is happily loaded."""
    return _TRITON_IMPORT_ERR


def run_symex(items: list, work) -> dict[str, Any]:
    """Execute the trace symbolically with Triton, write s3_symex.jsonl.

    Caller MUST check `is_available()` first; we raise if Triton isn't
    importable rather than silently producing an empty file. The caller (the
    s3 stage entry point) is responsible for falling back to concrete DFG.
    """
    if not _TRITON_OK:
        raise RuntimeError(
            f"Triton unavailable: {_TRITON_IMPORT_ERR}. "
            f"Install via `pip install triton-library` or build from source "
            f"(https://triton-library.github.io). The engine's pure-Python "
            f"path is the concrete DFG — switch back with `--symex concrete`."
        )

    ctx_t = TritonContext()
    # The whole engine is AArch64-focused (see TargetMeta.arch). Future work
    # could route per-arch — for now we hard-pin.
    ctx_t.setArchitecture(ARCH.AARCH64)
    ctx_t.setMode(MODE.ALIGNED_MEMORY, True)

    nodes: list[SymexNode] = []
    decode_failed = 0
    for ins in items:
        node = SymexNode(idx=ins.idx, pc=ins.pc, mnemonic=ins.mnemonic)

        # Decode & symbolically execute the single instruction. Triton needs
        # the raw bytes; we don't have to seed memory because every read is
        # against a register that the previous instructions already constrained
        # symbolically (the trace is contiguous within the algo entry/exit
        # window — there are no untracked jumps in our slice).
        try:
            t_inst = TritonInstr()
            t_inst.setAddress(ins.pc)
            t_inst.setOpcode(bytes(ins.bytes_))
            ctx_t.processing(t_inst)
        except Exception:
            decode_failed += 1
            nodes.append(node)
            continue

        # Extract symbolic expression for each written register. Triton's
        # `getSymbolicRegisterValue` returns the canonical AST.
        for wreg in ins.regs_write:
            try:
                reg_obj = ctx_t.getRegister(wreg)
                ast = ctx_t.getSymbolicRegisterValue(reg_obj)
                node.reg_exprs[wreg] = str(ast)
                node.reg_inputs[wreg] = sorted(set(ins.regs_read.keys()))
            except Exception:
                # Specific register name might not be a Triton register on
                # AArch64 (e.g. an aliased "wzr"). Skip silently.
                continue
        nodes.append(node)

    out_path: Path = work.root / "stage_outputs" / "s3_symex.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for n in nodes:
            row = asdict(n)
            row["pc"] = f"0x{n.pc:x}"
            f.write(json.dumps(row) + "\n")

    return {
        "stage":         "s3",
        "symex":         "triton",
        "nodes":         len(nodes),
        "decoded":       len(nodes) - decode_failed,
        "decode_failed": decode_failed,
        "out":           str(out_path),
    }


def env_mode() -> str:
    """Read the symex-mode env var; default 'concrete'."""
    return os.environ.get("UTOV_SYMEX_MODE", "concrete").strip().lower() or "concrete"


def warn_fallback(reason: str) -> None:
    """Emit a stderr breadcrumb when the user asked for triton but we degraded.
    Centralized so callers don't reinvent the message."""
    print(
        f"warning: UTOV_SYMEX_MODE=triton requested but Triton unavailable: "
        f"{reason}; falling back to concrete DFG (BR-2 P1.5 fallback).",
        file=sys.stderr,
    )
