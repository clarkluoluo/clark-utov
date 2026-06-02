"""Shared leaf utilities for the setup-symex runner package."""
from __future__ import annotations

from ..types import Instruction


def triton_available() -> bool:
    """True iff Triton bindings import cleanly (delegates to the s3 wrapper)."""
    from ..stages import s3_triton_symex
    return s3_triton_symex.is_available()


def triton_unavailable_reason() -> str | None:
    from ..stages import s3_triton_symex
    return s3_triton_symex.unavailable_reason()


def opcode_hex(ins: Instruction) -> str:
    """The instruction's raw bytes as lower-hex — the semantics-table primary key."""
    return bytes(ins.bytes_).hex()
# Control-flow mnemonics whose data effect we recover NOTHING from: their job is
# to pick the next instruction, and on an obfuscated VM that choice is data-
# dependent — letting Triton evaluate the branch (a possibly-symbolic condition)
# makes the symbolic PC diverge. We are trace-guided: the recorded order already
# IS the taken path, so we skip the branch's processing entirely and keep walking
# the trace (concrete control flow, symbolic data).
_CONTROL_FLOW_HEADS = frozenset({
    "b", "bl", "br", "blr", "ret", "cbz", "cbnz", "tbz", "tbnz",
    "braa", "brab", "blraa", "blrab", "retaa", "retab",
})


def is_control_flow(mnemonic: str) -> bool:
    """True for a branch / call / return — an instruction whose only effect is to
    pick the next pc (no data transform we recover). ``b.<cond>`` included."""
    toks = str(mnemonic).split()
    head = toks[0].lower() if toks else ""
    return head in _CONTROL_FLOW_HEADS or head.startswith("b.")
