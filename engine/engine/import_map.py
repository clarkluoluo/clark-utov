"""#4+#6 — auto PLT/import map + external-function summaries.

Universal principle (spec_tc2_import_map_extern_summary): an agent that treats
ordinary libc calls as opaque native mystery wastes effort. utov should resolve
PLT/import stubs to symbol names from the binary so a ``bl 0x40000a90`` reads as
``bl rand@plt`` (#4), and expose a per-symbol summary of the call's effect
(signature + ABI arg mapping + an "introduces external state" flag) so a recovered
formula that flows through ``time``/``rand`` is stated as depending on external
state rather than silently pretending to be a pure input transform (#6).

Inventory (A8①, don't rebuild):
  * :mod:`engine.static_tools` — the whitelisted static bridge (objdump/readelf/nm)
    reads relocations + symbols. No new disassembler.
  * :mod:`engine.value_provenance` — the "external state" value tagging. An
    external-state external caps a value's evidence class (it is not a closed-form
    recompute of the input). #6 surfaces the dependency; value_provenance enforces
    the ceiling.
  * :func:`engine.oracle_provenance._resolve_call_target` — already resolves a
    direct ``bl <imm>`` and an indirect ``blr xN`` from the concrete trace. We reuse
    it for the call-target subject (A8②), never re-implement call decoding.

Degenerate (A8④, always a verdict, never fabricate): no binary / stripped / symbol
not found ⇒ the call stays ``unknown@<addr>`` (surfaced in ``unresolved``); an
external call with no summary entry is ``external_unknown`` (introduces external
state of UNKNOWN kind), never assumed pure and never given a fabricated summary.

Generic — the summary table is keyed by SYMBOL NAME, never by a TC2 address. The
concrete PLT addresses live in the per-binary :class:`ImportMap`, not the table
(the "different target?" test). No runner change.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .oracle_provenance import _is_call, _resolve_call_target
from .static_tools import is_available, run_tool

__all__ = [
    "ImportMap",
    "ExternSummary",
    "build_import_map",
    "annotate_calls",
    "extern_summary",
    "EXTERN_SUMMARIES",
]


# --------------------------------------------------------------------------- #
# #6 — external-function summary table (keyed by SYMBOL NAME, never an address)
# --------------------------------------------------------------------------- #

# ABI for aarch64 (and the generic SysV/AAPCS register-arg shape): integer/pointer
# args land in x0,x1,x2,... in order. A summary names each arg's ROLE so #5 can map
# an output sink back to a source buffer (memcpy: sink⊆dst ⇒ source = src).
_AARCH64_INT_ARG_REGS = ("x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7")


@dataclass(frozen=True)
class ExternSummary:
    """A known external's effect summary (#6).

    ``abi_args`` maps each positional arg to its register + role; #5 consumes
    ``dst``/``src``/``n`` roles to synthesize a sink→source BoundaryEdge.
    ``introduces_external_state`` flags a call whose output is NOT a closed-form
    recompute of the input (time/PRNG) — it caps the recovered formula's evidence
    class and must be surfaced on the parity/closure result (couples to the
    determinism / seed-independence gate). A pure-data mover (memcpy) is False.
    """

    name: str
    abi_args: tuple[dict[str, str], ...] = ()   # [{"reg": "x0", "role": "dst"}, ...]
    introduces_external_state: bool = False
    state_kind: str = "none"                    # "time" | "prng" | "none" | ...
    effect: str = ""

    def role_reg(self, role: str) -> str | None:
        """The ABI register holding the arg with this ROLE (e.g. role='src' → 'x1'),
        or None when the summary has no such role. Used by #5 to read dst/src/n at
        the call site."""
        for a in self.abi_args:
            if a.get("role") == role:
                return a.get("reg")
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "abi_args": [dict(a) for a in self.abi_args],
            "introduces_external_state": self.introduces_external_state,
            "state_kind": self.state_kind,
            "effect": self.effect,
        }


def _arg(role: str, idx: int) -> dict[str, str]:
    return {"reg": _AARCH64_INT_ARG_REGS[idx], "role": role}


# The general mechanism table. Extend over time. TC2's PLT addresses NEVER appear
# here — only generic symbol names.
EXTERN_SUMMARIES: dict[str, ExternSummary] = {
    "time": ExternSummary(
        name="time",
        abi_args=(_arg("tloc", 0),),
        introduces_external_state=True, state_kind="time",
        effect="returns the current wall-clock time (external time state)",
    ),
    "srand": ExternSummary(
        name="srand",
        abi_args=(_arg("seed", 0),),
        introduces_external_state=True, state_kind="prng",
        effect="seeds the C PRNG (external random state)",
    ),
    "rand": ExternSummary(
        name="rand",
        abi_args=(),
        introduces_external_state=True, state_kind="prng",
        effect=("returns the next C PRNG value (external random state; on bionic, a "
                "linear-additive-feedback / TYPE_3 generator seeded via srand)"),
    ),
    "random": ExternSummary(
        name="random",
        abi_args=(),
        introduces_external_state=True, state_kind="prng",
        effect="returns the next random() PRNG value (external random state)",
    ),
    "memcpy": ExternSummary(
        name="memcpy",
        abi_args=(_arg("dst", 0), _arg("src", 1), _arg("n", 2)),
        introduces_external_state=False, state_kind="none",
        effect="fills dst[0:n] from src[0:n] (no external state)",
    ),
    "memmove": ExternSummary(
        name="memmove",
        abi_args=(_arg("dst", 0), _arg("src", 1), _arg("n", 2)),
        introduces_external_state=False, state_kind="none",
        effect="copies n bytes src→dst with overlap-safe semantics (no external state)",
    ),
    "memset": ExternSummary(
        name="memset",
        abi_args=(_arg("dst", 0), _arg("c", 1), _arg("n", 2)),
        introduces_external_state=False, state_kind="none",
        effect="fills dst[0:n] with the constant byte c (no external state)",
    ),
    "strlen": ExternSummary(
        name="strlen",
        abi_args=(_arg("s", 0),),
        introduces_external_state=False, state_kind="none",
        effect="returns the length of the NUL-terminated string at s",
    ),
    "strcpy": ExternSummary(
        name="strcpy",
        abi_args=(_arg("dst", 0), _arg("src", 1)),
        introduces_external_state=False, state_kind="none",
        effect="copies the NUL-terminated string src→dst (no external state)",
    ),
}


def extern_summary(symbol: str) -> ExternSummary | None:
    """Return the :class:`ExternSummary` for a known external symbol, or ``None``.

    A8④: ``None`` is the honest "no summary entry" — the caller tags such a call
    ``external_unknown`` (external state of unknown kind), never a fabricated
    summary. The PLT ``@plt`` decoration is stripped before lookup so an annotated
    ``rand@plt`` resolves to the ``rand`` entry."""
    if not symbol:
        return None
    base = symbol.split("@", 1)[0]
    return EXTERN_SUMMARIES.get(base)


# --------------------------------------------------------------------------- #
# #4 — import map: resolve PLT/import stubs to symbol names from the binary
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ImportMap:
    """A resolved PLT/import map for ONE binary (per-binary DATA, not the table).

    ``by_plt_addr`` maps a concrete PLT stub address → its symbol name (the thing a
    ``bl <addr>`` actually calls). ``by_got`` maps GOT slot addresses → symbol when
    available. ``unresolved`` lists call-target addresses with NO symbol (A8④:
    surfaced as ``unknown@<addr>``, never guessed). ``binary_available`` is False
    when no binary / no static tool was usable — the honest degenerate flag."""

    by_plt_addr: dict[int, str] = field(default_factory=dict)
    by_got: dict[int, str] = field(default_factory=dict)
    unresolved: tuple[int, ...] = ()
    binary_available: bool = True
    source: str = ""          # which tool produced this map (audit)
    detail: str = ""

    def symbol_for(self, addr: int) -> str | None:
        """The symbol a call to ``addr`` resolves to (PLT stub or GOT slot), or
        None when unresolved."""
        if addr in self.by_plt_addr:
            return self.by_plt_addr[addr]
        return self.by_got.get(addr)

    def to_dict(self) -> dict[str, Any]:
        return {
            "by_plt_addr": {f"0x{a:x}": s for a, s in sorted(self.by_plt_addr.items())},
            "by_got": {f"0x{a:x}": s for a, s in sorted(self.by_got.items())},
            "unresolved": [f"0x{a:x}" for a in self.unresolved],
            "binary_available": self.binary_available,
            "source": self.source,
            "detail": self.detail,
        }


# objdump -d disassembly line:  "  400a90:\t...\tbl  400a80 <rand@plt>"
# We harvest the `<sym@plt>` annotation objdump prints on the call line, keyed by
# the stub address. radare2 / llvm-objdump share the `<sym>` annotation shape.
_DISASM_ADDR = re.compile(r"^\s*([0-9a-fA-F]+):")
_PLT_ANNOT = re.compile(r"<([A-Za-z_][\w.]*)(@plt)?>")

# `objdump -d -j .plt` header line for a stub:  "0000000000400a80 <rand@plt>:"
_PLT_LABEL = re.compile(r"^([0-9a-fA-F]+)\s+<([A-Za-z_][\w.]*)@plt>:")

# readelf -r relocation row (a libc import):
#   "0000000000411038  0000000600000402 R_AARCH64_JUMP_SLO 0000000000000000 rand + 0"
_RELOC_ROW = re.compile(
    r"^([0-9a-fA-F]+)\s+[0-9a-fA-F]+\s+\S+\s+[0-9a-fA-F]+\s+([A-Za-z_][\w.@]*)")


def _parse_plt_labels(disasm: str) -> dict[int, str]:
    """PLT stub addresses → symbol, from objdump section labels
    (``<sym@plt>:`` headers). The most reliable: each stub has its own label."""
    out: dict[int, str] = {}
    for line in disasm.splitlines():
        m = _PLT_LABEL.match(line.strip())
        if m:
            out[int(m.group(1), 16)] = m.group(2)
    return out


def _parse_got_relocs(reloc_text: str) -> dict[int, str]:
    """GOT slot addresses → symbol, from a readelf relocation dump. A JUMP_SLOT /
    GLOB_DAT row binds a GOT address to an imported symbol."""
    out: dict[int, str] = {}
    for line in reloc_text.splitlines():
        s = line.strip()
        if "R_" not in s:
            continue
        m = _RELOC_ROW.match(s)
        if m:
            sym = m.group(2).split("@", 1)[0]
            if sym:
                out[int(m.group(1), 16)] = sym
    return out


def build_import_map(
    binary_path: str | Path | None = None,
    *,
    static_artifacts: dict[str, str] | None = None,
    plt_map: dict[int, str] | None = None,
    got_map: dict[int, str] | None = None,
    timeout: float = 30.0,
) -> ImportMap:
    """Resolve PLT/import stubs → symbol names for one binary (#4).

    Resolution order (A8①, reuse the static bridge — no new disassembler):
      1. An explicit ``plt_map`` / ``got_map`` override (verbatim — the agent
         supplied a resolved map, e.g. from a trace-only run with no binary).
      2. ``static_artifacts`` — pre-captured tool output (``{"disasm": ...,
         "relocs": ...}``) so a caller can pass already-run objdump/readelf text
         (the trace-only path keeps working without a binary on disk).
      3. ``binary_path`` — run objdump (PLT labels) + readelf (GOT relocations) via
         :mod:`engine.static_tools`.

    A8④ degenerate: no binary AND no artifacts AND no tool on PATH ⇒
    ``binary_available=False`` with empty maps — every later call resolves to
    ``unknown@<addr>`` (the caller surfaces it), never a guessed name. The actual
    unresolved call addresses are filled by :func:`annotate_calls` (it knows the
    trace); ``build_import_map`` only builds the symbol tables."""
    # 1. Explicit override — verbatim.
    if plt_map is not None or got_map is not None:
        return ImportMap(
            by_plt_addr=dict(plt_map or {}),
            by_got=dict(got_map or {}),
            binary_available=True,
            source="explicit_override",
            detail="import map supplied by caller (verbatim)",
        )

    disasm = ""
    relocs = ""
    source_parts: list[str] = []

    # 2. Pre-captured static artifacts.
    if static_artifacts:
        disasm = static_artifacts.get("disasm", "") or ""
        relocs = static_artifacts.get("relocs", "") or ""
        if disasm or relocs:
            source_parts.append("static_artifacts")

    # 3. Run the static tools against the binary.
    if binary_path is not None and not (disasm or relocs):
        bp = str(binary_path)
        if not Path(bp).exists():
            return ImportMap(
                binary_available=False, source="",
                detail=f"binary not found: {bp} — calls resolve to unknown@<addr>",
            )
        if is_available("objdump"):
            r = run_tool("objdump", ["-d", "-j", ".plt", bp], timeout=timeout)
            if r.available and r.exit_code == 0 and r.stdout:
                disasm = r.stdout
                source_parts.append("objdump")
            else:
                # Some PLT layouts (e.g. .plt.sec) need a full disasm; fall back.
                r2 = run_tool("objdump", ["-d", bp], timeout=timeout)
                if r2.available and r2.exit_code == 0 and r2.stdout:
                    disasm = r2.stdout
                    source_parts.append("objdump")
        if is_available("readelf"):
            rr = run_tool("readelf", ["-r", bp], timeout=timeout)
            if rr.available and rr.exit_code == 0 and rr.stdout:
                relocs = rr.stdout
                source_parts.append("readelf")

    if not (disasm or relocs):
        return ImportMap(
            binary_available=False, source="",
            detail=("no binary / no static tool produced symbol data — calls "
                    "resolve to unknown@<addr> (supply binary_path, static_artifacts, "
                    "or an explicit plt_map)"),
        )

    by_plt = _parse_plt_labels(disasm) if disasm else {}
    by_got = _parse_got_relocs(relocs) if relocs else {}
    return ImportMap(
        by_plt_addr=by_plt, by_got=by_got, binary_available=True,
        source="+".join(dict.fromkeys(source_parts)),
        detail=(f"resolved {len(by_plt)} PLT stub(s) + {len(by_got)} GOT slot(s)"),
    )


def annotate_calls(
    trace: Iterable[Any],
    import_map: ImportMap,
) -> list[dict[str, Any]]:
    """Annotate every CALL in the trace with its resolved symbol (#4).

    For each call instruction (``bl``/``blr``/``call``), resolve the concrete
    target from the trace (reusing :func:`oracle_provenance._resolve_call_target` —
    direct imm OR indirect ``blr xN`` from regs_read, A8②) and look it up in the
    import map. Returns one annotation per call:

        {idx, pc, mnemonic, target, symbol, resolved_from, external_state, state_kind}

    ``symbol`` is ``rand`` (mapped), ``unknown@<addr>`` (no symbol — A8④, never
    guessed), or ``unknown@<unresolved>`` (target itself couldn't be read from the
    trace). ``external_state`` / ``state_kind`` come from the #6 summary; a known
    external with NO summary is tagged ``external_unknown`` (introduces external
    state of UNKNOWN kind), never assumed pure. Additive metadata — existing
    analyses that ignore it are unchanged (A8③)."""
    out: list[dict[str, Any]] = []
    for ins in trace:
        if not _is_call(ins.mnemonic):
            continue
        target = _resolve_call_target(ins)
        ann: dict[str, Any] = {
            "idx": ins.idx,
            "pc": f"0x{ins.pc:x}",
            "mnemonic": ins.mnemonic,
        }
        if target is None:
            ann.update({
                "target": None,
                "symbol": "unknown@<unresolved>",
                "resolved_from": "unresolved_target",
                "external_state": None,
                "state_kind": None,
            })
            out.append(ann)
            continue
        symbol = import_map.symbol_for(target)
        ann["target"] = f"0x{target:x}"
        if symbol is None:
            ann.update({
                "symbol": f"unknown@0x{target:x}",
                "resolved_from": "no_symbol",
                "external_state": None,        # unknown call — nothing claimed
                "state_kind": None,
            })
            out.append(ann)
            continue
        resolved_from = "plt" if target in import_map.by_plt_addr else "got"
        summary = extern_summary(symbol)
        if summary is not None:
            ann.update({
                "symbol": f"{symbol}@plt" if resolved_from == "plt" else symbol,
                "resolved_from": resolved_from,
                "external_state": summary.introduces_external_state,
                "state_kind": summary.state_kind,
            })
        else:
            # Resolved to a real symbol but no #6 summary entry — external_unknown:
            # an external call of UNKNOWN effect (introduces external state of
            # unknown kind), NOT assumed pure (A8④).
            ann.update({
                "symbol": f"{symbol}@plt" if resolved_from == "plt" else symbol,
                "resolved_from": resolved_from,
                "external_state": "external_unknown",
                "state_kind": "unknown",
            })
        out.append(ann)
    return out
