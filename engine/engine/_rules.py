"""spec #4 planner — heuristic rule registry (internal to observation_planner).

Split out of :mod:`engine.observation_planner` to keep each file ≤500 lines
(feedback_max_500_lines_per_file). This holds the SHAPE-level proposal type, the
rule-registry types, the three seed rules + their (generic aarch64 / import_map)
shape parsers, and the watch-cfg helpers. The public facade
(:mod:`engine.observation_planner`) imports + re-exports these; callers import from
the facade, not here. BEHAVIOUR is identical to the pre-split single module — this
is a pure mechanical move (no logic change).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .import_map import ImportMap, extern_summary
from .oracle_provenance import ProvenanceResult, _is_call, _resolve_call_target
from .runner_client import ObservePoint, RegRelWatch
from .types import Instruction
from .watch_first_write import WatchFirstWriteConfig, request_point_watch


# --------------------------------------------------------------------------- #
# Proposal — the heuristic's output: a SHAPE-level observe-point suggestion.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ObserveProposal:
    """One heuristic-proposed observation, carrying WHY it was proposed.

    This is the ``observation_plan`` entry shape (additive next to ``next_watch``).
    It mirrors :class:`engine.runner_client.ObservePoint` (``pc/when/capture/regs/
    mem``) plus the audit pair ``reason`` + ``heuristic`` (which rule fired). It can
    also carry a reg-relative watch (``mem_regrel``) for buffers whose address is
    only live in a register at the hook PC — the same low-noise primitive
    ``observe_points_from_provenance`` uses.
    """

    pc: int
    when: str                                   # "before" | "after"
    capture: tuple[str, ...]                    # subset of ("regs", "mem")
    heuristic: str                              # which rule produced this
    reason: str
    regs: tuple[str, ...] = ()
    mem: tuple[tuple[int, int], ...] = ()       # concrete (addr, size)
    mem_regrel: tuple[RegRelWatch, ...] = ()    # reg-relative point watch(es)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "pc": f"0x{self.pc:x}",
            "when": self.when,
            "capture": list(self.capture),
            "reason": self.reason,
            "heuristic": self.heuristic,
        }
        if self.regs:
            d["regs"] = list(self.regs)
        if self.mem:
            d["mem"] = [[f"0x{a:x}", n] for (a, n) in self.mem]
        if self.mem_regrel:
            d["mem_regrel"] = [
                {"base_reg": w.base_reg, "offset": w.offset, "width": w.width,
                 "pc": f"0x{w.pc:x}", "kind": w.kind}
                for w in self.mem_regrel
            ]
        return d

    def to_observe_point(self) -> ObservePoint:
        """Lower this proposal to a runner-fulfillable :class:`ObservePoint`."""
        return ObservePoint(
            pc=self.pc, when=self.when, capture=self.capture,
            regs=self.regs, mem=self.mem, mem_regrel=self.mem_regrel)


# --------------------------------------------------------------------------- #
# Rule registry — shape matcher + proposal builder. Adding a heuristic = one entry.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RuleContext:
    """Everything a rule may consult about a gap's surroundings (read-only).

    A rule sees the gap (``gap_addr``/``gap_pc`` from a ``next_watch`` entry, either
    may be None), the trace, a pc→instructions index, and the optional per-binary
    :class:`ImportMap` (for the ``extern_call`` rule). Target-agnostic: no rule keys
    off a literal address.
    """

    prov: ProvenanceResult
    items: tuple[Instruction, ...]
    by_pc: dict[int, list[Instruction]]
    gap_addr: int | None = None
    gap_pc: int | None = None
    import_map: ImportMap | None = None

    def ins_at_pc(self, pc: int | None) -> Instruction | None:
        if pc is None:
            return None
        seq = self.by_pc.get(pc)
        return seq[0] if seq else None


# A matcher decides whether a rule applies at this gap (cheap, pure). A builder
# emits zero or more proposals. Splitting them keeps "add a rule = add an entry".
Matcher = Callable[["RuleContext"], bool]
Builder = Callable[["RuleContext"], list[ObserveProposal]]


@dataclass(frozen=True)
class Rule:
    name: str
    matches: Matcher
    build: Builder


# --------------------------------------------------------------------------- #
# Shape parsing helpers (generic aarch64) — pure.
# --------------------------------------------------------------------------- #

# Post-indexed store:  "strb w0, [x19], #1"  /  "str x8, [x9], #8"
_POST_INDEXED_STORE = re.compile(
    r"^\s*(strb|strh|str)\s+([wx]\d+)\s*,\s*\[\s*([wx]\d+)\s*\]\s*,\s*#?"
    r"(-?(?:0x[0-9a-fA-F]+|\d+))",
    re.IGNORECASE,
)

# Plain register-indirect store:  "strb w0, [x19]"  /  "str x8, [x9, #16]"
_REG_INDIRECT_STORE = re.compile(
    r"^\s*(strb|strh|str)\s+([wx]\d+)\s*,\s*\[\s*([wx]\d+)\s*"
    r"(?:,\s*#?(-?(?:0x[0-9a-fA-F]+|\d+)))?\s*\]",
    re.IGNORECASE,
)

_STORE_WIDTH = {"strb": 1, "strh": 2, "str": 8}


def _store_shape(mnemonic: str) -> tuple[str, str, str, int] | None:
    """Parse a store into (op, value_reg, base_reg, width) for the post-indexed OR
    plain register-indirect forms; None if not a recognised reg-base store. The
    post-indexed pattern is tried first (its trailing ``, #imm`` is the distinctive
    write-chain shape)."""
    m = _POST_INDEXED_STORE.match(mnemonic) or _REG_INDIRECT_STORE.match(mnemonic)
    if not m:
        return None
    op = m.group(1).lower()
    return op, m.group(2).lower(), m.group(3).lower(), _STORE_WIDTH[op]


# --------------------------------------------------------------------------- #
# Seed rule #1 — write_chain
# --------------------------------------------------------------------------- #


def _write_chain_stores(ctx: RuleContext) -> list[Instruction]:
    """The post-indexed / reg-indirect stores on the producer backtrace (prov.chain)
    — these are the write-chain producers, found via code SHAPE regardless of which
    read happens to be the gap. Falls back to the instruction at the gap PC when the
    chain is empty (a snapshot-only path). De-duplicated, ordered by trace idx."""
    stores: list[Instruction] = []
    seen: set[int] = set()
    for step in ctx.prov.chain:
        ins = ctx.ins_at_pc(_parse_addr(step.get("pc")))
        if ins is not None and ins.pc not in seen and _store_shape(ins.mnemonic):
            seen.add(ins.pc)
            stores.append(ins)
    if not stores:
        gap_ins = ctx.ins_at_pc(ctx.gap_pc)
        if gap_ins is not None and _store_shape(gap_ins.mnemonic):
            stores.append(gap_ins)
    return stores


def _write_chain_match(ctx: RuleContext) -> bool:
    return bool(_write_chain_stores(ctx))


def _write_chain_build(ctx: RuleContext) -> list[ObserveProposal]:
    """A ``strb/strh/str wN, [xK], #imm`` post-indexed (or reg-indirect) store loop:
    the stored VALUE wN is the producer; the TARGET memory at [xK] is where it lands.
    For each such store on the producer chain, propose: capture wN BEFORE the store
    (the value being written) + the target buffer AFTER (where it landed). The target
    is captured reg-relative ([xK]) since its address is only live in the register at
    this PC — no concrete-address leak."""
    out: list[ObserveProposal] = []
    for ins in _write_chain_stores(ctx):
        op, value_reg, base_reg, width = _store_shape(ins.mnemonic)  # type: ignore[misc]
        pc = ins.pc
        # (1) the producing value, captured BEFORE the store fires.
        out.append(ObserveProposal(
            pc=pc, when="before", capture=("regs",), regs=(value_reg,),
            heuristic="write_chain",
            reason=(f"write-chain {op} {value_reg},[{base_reg}] — capture the produced "
                    f"value {value_reg} before the store"),
        ))
        # (2) the target memory at [base_reg], captured AFTER the store landed. Reg-
        #     relative: the buffer address is only live in base_reg at this PC.
        spec = request_point_watch(
            pc=pc, base_reg=base_reg, offset=0, width_bytes=width,
            value_name=f"write_chain_target@{base_reg}", kind="write",
            cfg=_planner_watch_cfg(),
        )
        out.append(ObserveProposal(
            pc=pc, when="after", capture=("mem",),
            mem_regrel=(_spec_to_regrel(spec),),
            heuristic="write_chain",
            reason=(f"write-chain {op} {value_reg},[{base_reg}] — capture the target "
                    f"memory at [{base_reg}] ({width}B) after the store"),
        ))
    return out


# --------------------------------------------------------------------------- #
# Seed rule #2 — extern_call
# --------------------------------------------------------------------------- #


def _extern_call_targets(ctx: RuleContext) -> list[tuple[Instruction, str]]:
    """The (call_ins, symbol) pairs for calls to a RESOLVED extern, found at the gap
    PC AND at every boundary PC (OPAQUE_CALLEE call sites). Requires an import_map to
    resolve a target → symbol; with no map the rule simply does not fire (the gap
    stays in next_watch — A8④). De-duplicated by PC, ordered by trace idx."""
    if ctx.import_map is None:
        return []
    candidates: list[Instruction] = []
    seen: set[int] = set()
    for pc in [ctx.gap_pc, *ctx.prov.boundary_pcs]:
        ins = ctx.ins_at_pc(pc if isinstance(pc, int) else _parse_addr(pc))
        if ins is not None and ins.pc not in seen and _is_call(ins.mnemonic):
            seen.add(ins.pc)
            candidates.append(ins)
    out: list[tuple[Instruction, str]] = []
    for ins in candidates:
        target = _resolve_call_target(ins)
        if target is None:
            continue
        symbol = ctx.import_map.symbol_for(target)
        if symbol is not None:
            out.append((ins, symbol))
    return out


def _extern_call_match(ctx: RuleContext) -> bool:
    return bool(_extern_call_targets(ctx))


def _extern_call_build(ctx: RuleContext) -> list[ObserveProposal]:
    """A ``bl/blr`` to a resolved extern (from the import_map): propose capturing
    the pre-call ABI argument registers (from :class:`ExternSummary.abi_args`, or
    the generic AAPCS arg regs when no summary entry exists) BEFORE the call, and
    the return register x0 AFTER. Generalises across externs — the symbol, not an
    address, drives it."""
    out: list[ObserveProposal] = []
    for ins, symbol in _extern_call_targets(ctx):
        summary = extern_summary(symbol)
        if summary is not None and summary.abi_args:
            arg_regs = tuple(a["reg"] for a in summary.abi_args if a.get("reg"))
            roles = ", ".join(f"{a.get('reg')}={a.get('role')}" for a in summary.abi_args)
            arg_desc = f"ABI args ({roles})"
        else:
            # No summary entry / no args declared: still propose the generic AAPCS
            # arg registers — never silently skip a resolved extern (A8④).
            arg_regs = ("x0", "x1", "x2", "x3")
            arg_desc = "generic AAPCS arg regs x0..x3 (no ABI summary for this symbol)"
        pc = ins.pc
        if arg_regs:
            out.append(ObserveProposal(
                pc=pc, when="before", capture=("regs",), regs=arg_regs,
                heuristic="extern_call",
                reason=(f"extern call {ins.mnemonic.strip()} → {symbol} — capture "
                        f"pre-call {arg_desc}"),
            ))
        out.append(ObserveProposal(
            pc=pc, when="after", capture=("regs",), regs=("x0",),
            heuristic="extern_call",
            reason=f"extern call → {symbol} — capture the return register x0 after the call",
        ))
    return out


# --------------------------------------------------------------------------- #
# Seed rule #3 — boundary_copy
# --------------------------------------------------------------------------- #


def _boundary_copy_match(ctx: RuleContext) -> bool:
    # Anchored via a declared boundary edge (a "fixed header + buffer copy"
    # boundary): the transform output has no native writer, but the pre-transform
    # SOURCE buffer does — propose observing that source buffer.
    return ctx.prov.anchored_edge is not None


def _boundary_copy_build(ctx: RuleContext) -> list[ObserveProposal]:
    """Near a declared boundary (fixed header + buffer copy): the transform output
    has no native writer, but the pre-transform SOURCE buffer does. Propose
    capturing that source buffer at the boundary's start PC so the next backtrace
    has the bytes the copy read from. Width comes from the edge's decode_meta hint
    (raw_len/n) or a conservative default."""
    edge = ctx.prov.anchored_edge
    assert edge is not None
    width = int(edge.decode_meta.get("raw_len") or edge.decode_meta.get("n") or 32)
    return [ObserveProposal(
        pc=edge.boundary_pc_from, when="before", capture=("mem",),
        mem=((edge.source_ptr, width),),
        heuristic="boundary_copy",
        reason=(f"boundary copy ({edge.transform}) at "
                f"0x{edge.boundary_pc_from:x}→0x{edge.boundary_pc_to:x} — capture the "
                f"pre-transform source buffer 0x{edge.source_ptr:x} ({width}B) the "
                f"copy reads from"),
    )]


DEFAULT_RULES: tuple[Rule, ...] = (
    Rule("write_chain", _write_chain_match, _write_chain_build),
    Rule("extern_call", _extern_call_match, _extern_call_build),
    Rule("boundary_copy", _boundary_copy_match, _boundary_copy_build),
)

# Convenience handles (re-exported) for tests / callers that want one rule.
rule_write_chain = DEFAULT_RULES[0]
rule_extern_call = DEFAULT_RULES[1]
rule_boundary_copy = DEFAULT_RULES[2]


# --------------------------------------------------------------------------- #
# watch-cfg + spec→regrel helpers — reuse the existing builder, but the planner
# must work even when UTOV_WATCH_FIRST_WRITE is disabled in the ambient env.
# --------------------------------------------------------------------------- #


def _planner_watch_cfg() -> WatchFirstWriteConfig:
    """A locally-enabled watch config so the planner's reg-relative proposals build
    regardless of the ambient env toggle. The planner only PROPOSES — it does not
    itself install a watchpoint; arming is the runner's job via run_plan."""
    base = WatchFirstWriteConfig.from_env()
    if base.enabled:
        return base
    import dataclasses as _dc
    return _dc.replace(base, enabled=True)


def _spec_to_regrel(spec: Any) -> RegRelWatch:
    """Lower a :class:`WatchFirstWriteSpec` (from request_point_watch) to the
    :class:`RegRelWatch` an ObservePoint carries — the same shape recapture.py uses."""
    return RegRelWatch(
        base_reg=spec.base_reg, offset=spec.offset, width=spec.width_bytes,
        pc=spec.pc, kind=spec.kind)


def _parse_addr(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    try:
        return int(v, 16) if isinstance(v, str) else int(v)
    except (ValueError, TypeError):
        return None
