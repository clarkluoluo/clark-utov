"""Stdlib-only AArch64 memory-operand parser for textual traces.

Self-contained on purpose: ``runner_client`` must NOT import
setup_symex/core/dataflow (avoids an import cycle), so this helper lives here
with its own tiny regexes and only depends on :class:`engine.types.MemOp`.

Given a disassembled mnemonic plus the pre-execute register reads and the
post-execute register writes parsed off a unidbg text-trace line, recover the
load/store memory event(s) purely from AArch64 addressing syntax and the
in-line register values. No address/handler/case constants — everything is
derived from the mnemonic text and the supplied register maps.

Returns ``(ops, ea_unresolved)``:
  * ``ops`` is the tuple of recovered :class:`MemOp` (empty for non-memory
    instructions, or when the effective address cannot be computed honestly);
  * ``ea_unresolved`` is True iff this *is* a memory instruction but the EA /
    addressing form could not be resolved from the line — the caller can count
    it. We never fabricate an address or value when we cannot derive it.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Mapping
from dataclasses import dataclass

from .types import MemOp

# Bracketed addressing operand, e.g. "[x25, #0x28]" -> "x25, #0x28".
_ADDR_OPERAND_RE = re.compile(r"\[([^\]]*)\]")
# A register token (x0..x30 / w0..w30 / sp / xzr / wzr).
_REG_RE = re.compile(r"\b(x\d+|w\d+|sp|xzr|wzr)\b")
# An immediate token "#0x.." or "#123" (optionally negative).
_IMM_RE = re.compile(r"#(-?0x[0-9a-fA-F]+|-?\d+)")
# Optional shift/extend on an index register: "lsl #3", "sxtw #2", "uxtw".
_SHIFT_RE = re.compile(r"\b(lsl|lsr|asl|sxtw|uxtw|sxtx|uxtx)\b(?:\s*#(\d+))?")

# Mnemonics whose memory width differs from the destination register class.
# Loads/stores of explicit byte/halfword/word widths.
_BYTE_MNEMS = ("ldrb", "ldrsb", "strb")
_HALF_MNEMS = ("ldrh", "ldrsh", "strh")
_WORD_MNEMS = ("ldrsw",)  # sign-extending word load -> 4-byte access


def _is_mem_mnemonic(op: str) -> bool:
    """Memory access iff the base op starts with ld/st (ldr/str/ldp/stp/ldrb...).

    Excludes non-memory ops that merely start with the same letters? In AArch64
    every ``ld*``/``st*`` opcode is a load/store, so a prefix test is exact and
    case-agnostic here.
    """
    return op.startswith("ld") or op.startswith("st")


def _access_size(op: str) -> int:
    """Bytes touched by one element of this access, from the mnemonic width."""
    if op.startswith(_BYTE_MNEMS):
        return 1
    if op.startswith(_HALF_MNEMS):
        return 2
    if op in _WORD_MNEMS or op.startswith(_WORD_MNEMS):
        return 4
    # Width follows the data register class: w-reg => 4 bytes, x-reg => 8 bytes.
    # The destination/source register is decided by the caller; default here is
    # the generic 64-bit form, refined per data register below.
    return 8


def _data_regs(operands: str) -> list[str]:
    """Data (source/dest) register tokens, i.e. those *before* the '[' bracket."""
    head = operands.split("[", 1)[0]
    return [m.group(1) for m in _REG_RE.finditer(head)]


def _reg_size(reg: str) -> int:
    """Element size implied by a data register class (w=4, x/sp=8)."""
    return 4 if reg.startswith("w") else 8


def parse_mem_ops(
    mnemonic: str,
    reads: Mapping[str, int],
    writes: Mapping[str, int],
) -> tuple[tuple[MemOp, ...], bool]:
    """Recover memory ops from a disassembled AArch64 load/store line.

    See module docstring. ``reads`` = pre-execute register values,
    ``writes`` = post-execute register values (as parsed by ``_parse_state``).
    """
    text = mnemonic.strip()
    if not text:
        return (), False
    op = text.split(None, 1)[0].lower()
    if not _is_mem_mnemonic(op):
        return (), False

    is_load = op.startswith("ld")

    bracket = _ADDR_OPERAND_RE.search(text)
    if not bracket:
        # A memory mnemonic with no bracketed addressing form we recognise
        # (e.g. literal/PC-relative ldr) — honest miss.
        return (), True
    inner = bracket.group(1)

    # Split addressing tokens. Base register is the first reg inside [].
    base_match = _REG_RE.search(inner)
    if not base_match:
        return (), True
    base = base_match.group(1)
    if base not in reads:
        # Cannot resolve EA without the base register's pre-state value.
        return (), True
    base_val = reads[base]

    # Index register: a second register token inside the brackets.
    regs_in = [m.group(1) for m in _REG_RE.finditer(inner)]
    index = regs_in[1] if len(regs_in) > 1 else None

    imm_match = _IMM_RE.search(inner)
    imm = int(imm_match.group(1), 0) if imm_match else 0

    # Pre-index "[xn, #imm]!" vs post-index "[xn], #imm".
    after_bracket = text[bracket.end():]
    is_pre_index = "!" in after_bracket
    # post-index: an immediate appears AFTER the closing bracket.
    post_imm_match = _IMM_RE.search(after_bracket)
    is_post_index = post_imm_match is not None

    if index is not None:
        # Register-offset form: EA = base + (index_val << shift / extended).
        if index not in reads:
            return (), True
        shift_m = _SHIFT_RE.search(inner)
        shift = 0
        if shift_m:
            kind = shift_m.group(1)
            amt = shift_m.group(2)
            if kind in ("lsl", "asl"):
                shift = int(amt) if amt else 0
            elif kind in ("sxtw", "uxtw", "sxtx", "uxtx"):
                # Extension with optional left shift amount. Value width handling
                # (32->64) is already reflected in the trace register value; we
                # only apply the shift amount. Unknown extend semantics -> bail.
                shift = int(amt) if amt else 0
            else:
                # lsr/asr as an address shift is not a valid addressing form.
                return (), True
        ea = base_val + (reads[index] << shift)
    elif is_post_index:
        # "[xn], #imm" — EA is the original base value; imm is the writeback.
        ea = base_val
    elif is_pre_index:
        # "[xn, #imm]!" — EA is base+imm, then written back.
        ea = base_val + imm
    else:
        # "[xn]" or "[xn, #imm]".
        ea = base_val + imm

    dregs = _data_regs(text[: bracket.start()])
    if not dregs:
        return (), True

    # ldp/stp produce two element accesses at EA and EA+size.
    is_pair = op.startswith(("ldp", "stp", "ldnp", "stnp"))
    rw = "r" if is_load else "w"
    src_map = writes if is_load else reads  # value source: post for load, pre for store

    ops: list[MemOp] = []
    for i, dreg in enumerate(dregs):
        # Element size: explicit width mnemonic wins; else from register class.
        if op.startswith(_BYTE_MNEMS):
            size = 1
        elif op.startswith(_HALF_MNEMS):
            size = 2
        elif op.startswith(_WORD_MNEMS):
            size = 4
        else:
            size = _reg_size(dreg)
        if dreg not in src_map:
            # Value for this element not present on the line — honest miss for
            # the whole step (don't emit a half-resolved pair).
            return (), True
        ops.append(MemOp(rw=rw, addr=ea + i * size, val=src_map[dreg], size=size))

    # If syntactically a pair op but we only found one data reg, treat as
    # unresolved rather than guessing the second access.
    if is_pair and len(ops) < 2:
        return (), True

    return tuple(ops), False


# --- B6: reg-relative structural decomposition (NOT EA computation) ---------
#
# parse_mem_ops above *computes a live EA* (base_val + offset). B6 is the
# opposite direction: given a concrete address ``A`` observed at a PC, recover
# the *structural form* ``[base_reg + offset]`` (or ``[base + index*scale +
# offset]``) and PROVE it from the same-run register row — so the watch carries
# the STRUCTURE (stable across runs), never the concrete ``A`` (a dead stack/
# heap address that means nothing in another run = false closure, invariant 1).
#
# The reg row is used ONLY as *evidence* (verify A == base_val + offset), never
# to predict a future EA (invariant 2). The runner resolves the live EA from the
# structure at its own hook time.


class AddrDecomposition(enum.Enum):
    """Verdict tier for decomposing one observed (pc, addr) memory point."""
    REGREL_UPGRADED = "REGREL_UPGRADED"          # ① proven base(+index)+offset
    NEEDS_REG_SNAPSHOT = "NEEDS_REG_SNAPSHOT"    # ② decomposable form, regs absent
    UNSTABLE_CONCRETE = "UNSTABLE_CONCRETE"      # ③ unrepresentable / unprovable


@dataclass(frozen=True)
class DecomposedAddr:
    """Result of :func:`decompose_addressing` — a three-tier classification.

    On ① ``REGREL_UPGRADED``: ``base_reg`` + ``offset`` (and ``index``/``scale``
    for a register-offset form) carry the *structural* watch — NO concrete ``addr``
    leaks into the watch. ``width`` is the access size from the mnemonic.

    On ② ``NEEDS_REG_SNAPSHOT``: ``needed_regs`` names the addressing register(s)
    whose value is missing at this PC in the current trace; the caller emits a
    register-observe recapture directive to fill them, then re-decomposes.

    On ③ ``UNSTABLE_CONCRETE``: ``reason`` explains why the form can't be reduced
    to a stable base+offset (double-live register offset, pre/post-index
    writeback, PC-relative/literal, non-memory line, or addr-vs-decomposition
    mismatch). The point MUST NOT be reused across runs.
    """
    verdict: AddrDecomposition
    base_reg: str | None = None
    offset: int = 0
    index: str | None = None
    scale: int = 0
    width: int = 0
    needed_regs: tuple[str, ...] = ()
    reason: str = ""


def _addr_access_width(op: str, dregs: list[str]) -> int:
    """Bytes one element of this access touches (mnemonic width wins; else the
    data register class). Mirrors ``parse_mem_ops``' size logic for a single
    element."""
    if op.startswith(_BYTE_MNEMS):
        return 1
    if op.startswith(_HALF_MNEMS):
        return 2
    if op.startswith(_WORD_MNEMS):
        return 4
    return _reg_size(dregs[0]) if dregs else 8


def decompose_addressing(
    mnemonic: str,
    addr: int,
    reads: Mapping[str, int],
) -> DecomposedAddr:
    """Decompose an observed concrete ``addr`` at an AArch64 load/store into a
    *structural* reg-relative form, proving it against the in-line ``reads``.

    Three-tier verdict (see :class:`DecomposedAddr`):

    ① the mnemonic is an explicit-bracket load/store, the participating base
       (and index) register value(s) are present in ``reads``, and
       ``addr == base_val + offset`` (or ``base_val + index_val*scale + offset``)
       with a sane small ``offset`` → ``REGREL_UPGRADED`` with the structure, no
       concrete addr.
    ② the form IS a decomposable simple/register-offset bracket but ``reads`` is
       missing a participating register's value → ``NEEDS_REG_SNAPSHOT`` naming
       the absent registers (the caller arms a register-observe recapture there).
    ③ the line is not an explicit-bracket load/store, OR the form can't reduce to
       a stable base+offset (pre/post-index writeback, double-live register
       offset we can't represent, PC-relative/literal), OR the reg row is present
       but ``addr`` doesn't match ANY decomposition → ``UNSTABLE_CONCRETE``.

    Pure / deterministic; consults only the mnemonic text and the supplied reg
    row. Never fabricates a register value and never emits the concrete addr as a
    cross-run watch.
    """
    text = (mnemonic or "").strip()
    if not text:
        return DecomposedAddr(AddrDecomposition.UNSTABLE_CONCRETE,
                              reason="empty mnemonic — not a memory instruction")
    op = text.split(None, 1)[0].lower()
    if not _is_mem_mnemonic(op):
        return DecomposedAddr(AddrDecomposition.UNSTABLE_CONCRETE,
                              reason=f"{op!r} is not a load/store mnemonic")

    bracket = _ADDR_OPERAND_RE.search(text)
    if not bracket:
        # Literal / PC-relative ldr (e.g. "ldr x0, =0x..." or "ldr x0, label"):
        # no base register to rebase against — unstable as a reg-relative watch.
        return DecomposedAddr(AddrDecomposition.UNSTABLE_CONCRETE,
                              reason="no bracketed addressing form (PC-relative/literal) — no base register")
    inner = bracket.group(1)

    base_match = _REG_RE.search(inner)
    if not base_match:
        return DecomposedAddr(AddrDecomposition.UNSTABLE_CONCRETE,
                              reason="bracket has no base register")
    base = base_match.group(1)

    regs_in = [m.group(1) for m in _REG_RE.finditer(inner)]
    index = regs_in[1] if len(regs_in) > 1 else None

    imm_match = _IMM_RE.search(inner)
    offset = int(imm_match.group(1), 0) if imm_match else 0

    after_bracket = text[bracket.end():]
    is_pre_index = "!" in after_bracket
    is_post_index = _IMM_RE.search(after_bracket) is not None

    dregs = _data_regs(text[: bracket.start()])
    width = _addr_access_width(op, dregs)

    # ③ writeback forms: the base register MUTATES at this instruction, so a
    # structural [base+offset] watch is ambiguous across runs (the value the
    # runner reads at hook time depends on before/after). Refuse to rebase.
    if is_pre_index or is_post_index:
        kind = "pre-index" if is_pre_index else "post-index"
        return DecomposedAddr(AddrDecomposition.UNSTABLE_CONCRETE,
                              reason=f"{kind} writeback mutates the base register — "
                                     "ambiguous before/after, not stably rebasable")

    if index is not None:
        # Register-offset form [base, index{, shift}]. We CAN represent it as a
        # structural (base, index, scale) watch — but only if BOTH register
        # values are present to PROVE it (and the shift form is a valid address
        # shift). Otherwise ② (missing regs) or ③ (unprovable shift/mismatch).
        # AArch64 register-offset forms carry NO separate immediate offset — any
        # ``#N`` belongs to the shift/extend (parsed below), so offset is 0 here
        # (the bare ``_IMM_RE`` above would otherwise mistake the shift amount for
        # an offset).
        offset = 0
        shift_m = _SHIFT_RE.search(inner)
        scale = 0
        if shift_m:
            skind = shift_m.group(1)
            amt = shift_m.group(2)
            if skind in ("lsl", "asl", "sxtw", "uxtw", "sxtx", "uxtx"):
                scale = int(amt) if amt else 0
            else:
                # lsr/asr is not a valid address shift.
                return DecomposedAddr(AddrDecomposition.UNSTABLE_CONCRETE,
                                      reason=f"{skind!r} is not a valid address shift")
        missing = [r for r in (base, index) if r not in reads]
        if missing:
            return DecomposedAddr(
                AddrDecomposition.NEEDS_REG_SNAPSHOT,
                base_reg=base, index=index, scale=scale, width=width,
                needed_regs=tuple(missing),
                reason="register-offset form but addressing register value(s) absent "
                       "at this PC — capture a register snapshot, then re-decompose")
        if reads[base] + (reads[index] << scale) + offset == addr:
            return DecomposedAddr(
                AddrDecomposition.REGREL_UPGRADED,
                base_reg=base, offset=offset, index=index, scale=scale, width=width)
        return DecomposedAddr(
            AddrDecomposition.UNSTABLE_CONCRETE,
            reason=f"register-offset reg row does not reconstruct addr 0x{addr:x} "
                   "(base+index<<scale+offset mismatch) — not stably rebasable")

    # Simple form [base] or [base, #imm].
    if base not in reads:
        return DecomposedAddr(
            AddrDecomposition.NEEDS_REG_SNAPSHOT,
            base_reg=base, offset=offset, width=width, needed_regs=(base,),
            reason="base register value absent at this PC — capture a register "
                   "snapshot, then re-decompose")
    if reads[base] + offset == addr:
        return DecomposedAddr(
            AddrDecomposition.REGREL_UPGRADED,
            base_reg=base, offset=offset, width=width)
    return DecomposedAddr(
        AddrDecomposition.UNSTABLE_CONCRETE,
        reason=f"base reg row does not reconstruct addr 0x{addr:x} "
               "(base+offset mismatch) — not stably rebasable")
