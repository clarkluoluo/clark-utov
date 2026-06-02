"""Hand-filled semantics DSL + the persistent extension table."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


# ---------------------------------------------------------------------------
# Hand-filled semantics DSL — a small S-expression over registers + bitvectors.
#
# The escape hatch needs the agent to express ONE instruction's semantics in a
# form utov can inject into the Triton symbolic state (not free text we'd have to
# guess at). The DSL is a bounded SMT-style S-expression that maps 1:1 onto
# Triton's AstContext builders, so there is no ambiguity and a malformed fill is
# a precise error (caught + parity-backstopped), never a silent concretize.
#
#   register atom        x0  w1  sp          -> the register's current symbolic AST
#   constant             (bv <value> <size>) -> a <size>-bit bitvector constant
#   binary    (bvadd|bvsub|bvmul|bvand|bvor|bvxor|bvshl|bvlshr|bvashr|bvurem|
#              bvudiv|bvsdiv|bvsrem  <e> <e>)
#   unary                (bvnot|bvneg <e>)
#   slice                (extract <hi> <lo> <e>)   extend (zx|sx <bits> <e>)
#   concat               (concat <e> <e> ...)
#   e.g.  (bvmul x0 x1)   (bvadd (bvand x2 (bv 255 64)) x3)
# ---------------------------------------------------------------------------

SEMANTICS_BINOPS = frozenset({
    "bvadd", "bvsub", "bvmul", "bvand", "bvor", "bvxor", "bvshl", "bvlshr",
    "bvashr", "bvurem", "bvudiv", "bvsdiv", "bvsrem",
})
SEMANTICS_UNOPS = frozenset({"bvnot", "bvneg"})


class SemanticsParseError(ValueError):
    """A hand-filled semantics S-expression is malformed (structure / arity)."""


class SemanticsApplyError(RuntimeError):
    """A (well-formed) semantics expression could not be injected into the symbolic
    state — unknown register, width mismatch, or unsupported op for this backend."""


def _tokenize(text: str) -> list[str]:
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c in "()":
            out.append(c)
            i += 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in "()":
                j += 1
            out.append(text[i:j])
            i = j
    return out


def _atom(tok: str) -> "int | str":
    try:
        return int(tok, 0)          # decimal or 0x… immediate
    except ValueError:
        return tok                  # symbol: an op name or a register


def parse_sexpr(text: str) -> "int | str | list":
    """Parse the semantics DSL into nested lists (ints = immediates, strs = symbols).

    Pure — no Triton. Raises :class:`SemanticsParseError` on malformed input."""
    toks = _tokenize(str(text))
    if not toks:
        raise SemanticsParseError("empty semantics expression")
    pos = 0

    def _parse() -> "int | str | list":
        nonlocal pos
        if pos >= len(toks):
            raise SemanticsParseError("unexpected end of expression")
        t = toks[pos]
        pos += 1
        if t == "(":
            items: list = []
            while pos < len(toks) and toks[pos] != ")":
                items.append(_parse())
            if pos >= len(toks):
                raise SemanticsParseError("missing ')'")
            pos += 1                # consume ')'
            if not items:
                raise SemanticsParseError("empty '()'")
            return items
        if t == ")":
            raise SemanticsParseError("unexpected ')'")
        return _atom(t)

    node = _parse()
    if pos != len(toks):
        raise SemanticsParseError("trailing tokens after expression")
    return node


def validate_sexpr(node: "int | str | list") -> None:
    """Structural / arity check of a parsed expression. Pure — no Triton.

    Raises :class:`SemanticsParseError`. A bare immediate is rejected (wrap it as
    ``(bv value size)`` so the width is explicit — width-mismatch is a real trap)."""
    if isinstance(node, int):
        raise SemanticsParseError("bare immediate — wrap as (bv <value> <size>)")
    if isinstance(node, str):
        return                              # register atom
    op = node[0]
    if not isinstance(op, str):
        raise SemanticsParseError("operator must be a symbol")
    args = node[1:]
    if op == "bv":
        if len(args) != 2 or not all(isinstance(a, int) for a in args):
            raise SemanticsParseError("(bv <value> <size>) needs two integers")
        return
    if op == "extract":
        if len(args) != 3 or not isinstance(args[0], int) or not isinstance(args[1], int):
            raise SemanticsParseError("(extract <hi> <lo> <expr>)")
        validate_sexpr(args[2])
        return
    if op in ("zx", "sx"):
        if len(args) != 2 or not isinstance(args[0], int):
            raise SemanticsParseError(f"({op} <bits> <expr>)")
        validate_sexpr(args[1])
        return
    if op == "concat":
        if len(args) < 2:
            raise SemanticsParseError("(concat <expr> <expr> ...) needs >= 2")
        for a in args:
            validate_sexpr(a)
        return
    if op in SEMANTICS_BINOPS:
        if len(args) != 2:
            raise SemanticsParseError(f"({op} <expr> <expr>) needs two operands")
        for a in args:
            validate_sexpr(a)
        return
    if op in SEMANTICS_UNOPS:
        if len(args) != 1:
            raise SemanticsParseError(f"({op} <expr>) needs one operand")
        validate_sexpr(args[0])
        return
    raise SemanticsParseError(f"unknown op {op!r}")


# ---------------------------------------------------------------------------
# Escape hatch — the persistent instruction-semantics extension table
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InstructionSemantics:
    """Hand-filled symbolic semantics for ONE instruction the bulk decoder could
    not model — the escape-hatch entry the agent supplies and utov caches.

    ``effects`` maps each written register to a semantics-DSL S-expression (see
    :func:`parse_sexpr`) that the decoder injects into the symbolic state instead
    of concretizing — e.g. ``("x0", "(bvmul x0 x1)")``. Keyed by the exact opcode
    bytes so the same instruction is auto-handled next time; ``mnemonic`` is the
    broad fallback for callers that key by family."""

    opcode_hex: str
    mnemonic:   str
    effects:    tuple[tuple[str, str], ...]
    note:       str = ""
    filled_by:  str = "agent"

    @property
    def key(self) -> str:
        return self.opcode_hex

    def to_dict(self) -> dict[str, Any]:
        return {
            "opcode_hex": self.opcode_hex,
            "mnemonic":   self.mnemonic,
            "effects":    [list(e) for e in self.effects],
            "note":       self.note,
            "filled_by":  self.filled_by,
            "kind":       "setup_symex_insn_semantics",
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "InstructionSemantics":
        return cls(
            opcode_hex=str(d["opcode_hex"]),
            mnemonic=str(d.get("mnemonic", "")),
            effects=tuple((str(r), str(a)) for r, a in d.get("effects", [])),
            note=str(d.get("note", "")),
            filled_by=str(d.get("filled_by", "agent")),
        )


class SemanticsTable:
    """Persistent extension table: ``opcode_hex`` → :class:`InstructionSemantics`.

    Triton covers the bulk; this caches the long tail the agent hand-filled so
    the same instruction is auto-handled next round (and can seed a community
    plugin). Lookup tries the exact opcode first, then the mnemonic family."""

    def __init__(self, entries: Iterable[InstructionSemantics] | None = None,
                 path: str | Path | None = None) -> None:
        self._by_opcode: dict[str, InstructionSemantics] = {}
        self._by_mnemonic: dict[str, InstructionSemantics] = {}
        self.path = Path(path) if path is not None else None
        for sem in entries or ():
            self._index(sem)

    def _index(self, sem: InstructionSemantics) -> None:
        self._by_opcode[sem.opcode_hex] = sem
        # Last-writer-wins on the family fallback; the exact opcode is authoritative.
        self._by_mnemonic.setdefault(sem.mnemonic, sem)

    def lookup(self, opcode: str, mnemonic: str | None = None
               ) -> InstructionSemantics | None:
        sem = self._by_opcode.get(opcode)
        if sem is not None:
            return sem
        if mnemonic:
            # Family fallback: match on the leading mnemonic token (e.g. "mul").
            head = mnemonic.split()[0] if mnemonic.split() else mnemonic
            return self._by_mnemonic.get(mnemonic) or self._by_mnemonic.get(head)
        return None

    def register(self, sem: InstructionSemantics, *, persist: bool = True) -> None:
        self._index(sem)
        if persist and self.path is not None:
            self.save()

    def __len__(self) -> int:
        return len(self._by_opcode)

    def __contains__(self, opcode: str) -> bool:
        return opcode in self._by_opcode

    def to_list(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._by_opcode.values()]

    def save(self, path: str | Path | None = None) -> None:
        dst = Path(path) if path is not None else self.path
        if dst is None:
            raise ValueError("SemanticsTable.save needs a path (none configured)")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(self.to_list(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SemanticsTable":
        p = Path(path)
        entries: list[InstructionSemantics] = []
        if p.exists():
            for row in json.loads(p.read_text(encoding="utf-8")):
                entries.append(InstructionSemantics.from_dict(row))
        return cls(entries, path=p)


@dataclass(frozen=True, slots=True)
class UnmodeledInstruction:
    """The escape-hatch checkpoint: the bulk decoder hit an instruction it can't
    model and the table has no entry. NOT force-concretized, NOT silently skipped
    — surfaced so the agent fills its symbolic semantics (then cached + parity-
    backstopped). Same shape of judgment as a ``setup_symex`` Checkpoint."""

    opcode_hex: str
    mnemonic:   str
    idx:        int
    pc:         int

    @property
    def question(self) -> str:
        return (
            f"insn {self.opcode_hex} ({self.mnemonic}) @ idx {self.idx} "
            f"pc 0x{self.pc:x} is not modeled by the bulk decoder and has no "
            f"semantics-table entry — supply its symbolic semantics as "
            f"written-register → S-expression over the operand registers / "
            f"(bv <value> <size>), e.g. x0 = (bvmul x0 x1). It is NOT "
            f"force-concretized; it will be cached and still must clear the "
            f"multi-vector parity gate."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "opcode_hex": self.opcode_hex,
            "mnemonic":   self.mnemonic,
            "idx":        self.idx,
            "pc":         f"0x{self.pc:x}",
            "question":   self.question,
            "is_judgment": True,
            "kind":       "setup_symex_unmodeled_insn",
        }
