"""S3: concrete data-flow graph (no-Triton edition).

We take advantage of a fact the original PLAN §3 didn't fully exploit: our
trace is CONCRETE — every instruction's regs_read / regs_write values are
ground truth from the live runner. So for backward slicing (S4), we don't
need symbolic execution at all. A concrete data-flow graph is sufficient and
much cheaper to build.

What we lose without Triton: full algebraic simplification in S5 (we can
only do peephole InsSub reverse, not deep symbolic simplification of nested
expressions). Symbolic execution is tracked as P1.5 work — see IMPL_PLAN.

Algorithm:
  For each instruction i:
    For each register r in i.regs_read:
      Producer p(r, i) = most recent earlier instruction j < i where r in j.regs_write.
                        If none found, mark "external" (function entry argument).
      Edge: i.dep_on[r] = j.idx  (or None)

Memory dependencies ARE tracked (additive, byte-granular): a ``last_mem_writer``
map records, per memory byte, the idx of the instruction that last wrote it;
each memory read links to the writers of the bytes it loads. This recovers the
producer link across ``ldr x8,[x10]`` that the register graph alone cannot see
(the value came from memory, not a register write). Exposed as
:attr:`DfgNode.mem_deps`; the existing ``reg_deps`` field and the
``s3_dfg.jsonl`` schema are unchanged (``mem_deps`` is a new, optional field).

Output: stage_outputs/s3_dfg.jsonl, one row per instruction with its deps.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from ..types import Instruction

CODE_VERSION = "s3-v1"


@dataclass(frozen=True)
class InferredProducer:
    """A COARSE, low-confidence producer inferred from ``regs_read`` value changes
    when ``regs_write`` did not record the write (item ② fallback).

    In a single concrete trace a register's value only changes when something
    writes it. If ``regs_write`` is absent but a register's ``regs_read`` value
    differs between two reads, SOMETHING in the interval wrote it — but we do NOT
    know which instruction precisely. So this records the INTERVAL
    ``(after_idx, at_idx]`` the write happened in, marked ``confidence="inferred"``
    — never a precise writer. It feeds the ③/④ observability gate (a chain that
    rests on inferred edges is low-confidence → leans inconclusive), never a hard
    "this idx wrote it" claim."""

    reg:        str
    at_idx:     int             # the read instruction whose value changed
    after_idx:  int | None      # the previous read of this reg (interval lower bound; None = window start)
    confidence: str = "inferred"

    def to_dict(self):
        return {"reg": self.reg, "at_idx": self.at_idx,
                "after_idx": self.after_idx, "confidence": self.confidence}


@dataclass(frozen=True)
class DfgNode:
    idx: int
    pc: int
    mnemonic: str
    # reg name → producer idx (or None if external / pre-entry value)
    reg_deps: dict[str, int | None]
    # idxs of instructions that wrote the memory bytes this instruction LOADS.
    # Additive (new field; default keeps back-compat with old s3_dfg.jsonl rows
    # that have no mem_deps key). Resolves the `ldr [addr]` producer link the
    # register graph cannot.
    mem_deps: tuple[int, ...] = ()
    # reg name → COARSE inferred-interval producer (item ② fallback). Populated
    # ONLY for a read whose value changed while regs_write recorded no writer of
    # that reg anywhere in the stream. Additive (default empty); every existing
    # reg_deps consumer is unaffected. A populated entry is LOW CONFIDENCE — it
    # marks "written somewhere in this interval", not a precise producer.
    reg_deps_inferred: dict[str, "InferredProducer"] = None  # type: ignore[assignment]

    def __post_init__(self):
        # frozen dataclass: set the mutable default without a shared-instance trap.
        if self.reg_deps_inferred is None:
            object.__setattr__(self, "reg_deps_inferred", {})


def build_dfg(items: Iterable[Instruction]) -> list[DfgNode]:
    items_list = list(items) if not isinstance(items, list) else items

    # ② fallback support (additive, computed once up-front, pure read of the
    # stream): the set of regs that are NEVER written via regs_write anywhere in
    # the stream. The fallback edge is offered ONLY for these — a reg that has any
    # regs_write writer takes the original last_writer path unchanged (invariant
    # 7: a regs_write-populated trace is byte-for-byte unchanged). For such a reg,
    # track the previous idx at which we last read it and its value, so a value
    # change marks an inferred interval producer.
    never_written: set[str] = set()
    any_written: set[str] = set()
    for ins in items_list:
        for r in ins.regs_write:
            any_written.add(r)
    for ins in items_list:
        for r in ins.regs_read:
            if r not in any_written:
                never_written.add(r)

    last_writer: dict[str, int] = {}      # reg → last writer's idx
    last_mem_writer: dict[int, int] = {}  # mem byte addr → last writer's idx
    last_read: dict[str, tuple[int, int]] = {}  # reg → (idx, value) of its last read (fallback only)
    nodes: list[DfgNode] = []
    for ins in items_list:
        reg_deps: dict[str, int | None] = {}
        reg_deps_inferred: dict[str, InferredProducer] = {}
        for r in ins.regs_read:
            reg_deps[r] = last_writer.get(r)        # None = external
            # ② fallback: only for a reg with no regs_write writer in the whole
            # stream (otherwise the original path is authoritative). If its read
            # value changed since the last read, an inferred-interval producer is
            # recorded — coarse + low confidence, additive (does not touch
            # reg_deps[r], which stays None = external for the original consumers).
            if r in never_written:
                v = ins.regs_read[r]
                prev = last_read.get(r)
                if prev is not None and prev[1] != v:
                    reg_deps_inferred[r] = InferredProducer(
                        reg=r, at_idx=ins.idx, after_idx=prev[0])
        # Concrete memory read deps: each byte loaded links to its last writer.
        # Computed BEFORE applying this instruction's own writes, so a load
        # connects to the prior store, never to itself.
        mem_deps: list[int] = []
        for op in ins.mem:
            if op.rw == "r":
                for b in range(op.addr, op.addr + op.size):
                    w = last_mem_writer.get(b)
                    if w is not None and w not in mem_deps:
                        mem_deps.append(w)
        for r in ins.regs_write:
            last_writer[r] = ins.idx
        for op in ins.mem:
            if op.rw == "w":
                for b in range(op.addr, op.addr + op.size):
                    last_mem_writer[b] = ins.idx
        # update last-read tracking AFTER computing deps (so a value change is
        # measured against the PRIOR read, never this same read).
        for r in ins.regs_read:
            if r in never_written:
                last_read[r] = (ins.idx, ins.regs_read[r])
        nodes.append(DfgNode(
            idx=ins.idx, pc=ins.pc, mnemonic=ins.mnemonic,
            reg_deps=reg_deps, mem_deps=tuple(mem_deps),
            reg_deps_inferred=reg_deps_inferred,
        ))
    return nodes


def run(ctx) -> dict:
    items = ctx["items"]
    work = ctx["work"]

    # BR-2 / IMPL_PLAN P1.5: optional Triton symbolic-execution path. Opt-in
    # via ctx["symex_mode"] or env UTOV_SYMEX_MODE=triton. We ALWAYS still
    # produce the concrete DFG (S4 currently reads s3_dfg.jsonl for slicing)
    # — the Triton run writes its own s3_symex.jsonl alongside.
    items_list = list(items)
    nodes = build_dfg(items_list)
    out_path = _write_concrete_dfg(nodes, work)
    external_count = sum(
        1 for n in nodes for v in n.reg_deps.values() if v is None
    )

    summary: dict = {
        "stage": "s3",
        "nodes": len(nodes),
        "external_reg_reads": external_count,
        "out": str(out_path),
    }

    from . import s3_triton_symex
    mode = (ctx.get("symex_mode") or s3_triton_symex.env_mode()).strip().lower()
    if mode == "triton":
        if s3_triton_symex.is_available():
            symex_summary = s3_triton_symex.run_symex(items_list, work)
            # Merge Triton-side counts into the stage's return so downstream
            # tooling (CLI summary, agent telemetry) can see the symex path
            # ran. The concrete-DFG counts stay top-level for back-compat.
            summary["symex"]               = "triton"
            summary["symex_decoded"]       = symex_summary["decoded"]
            summary["symex_decode_failed"] = symex_summary["decode_failed"]
            summary["symex_out"]           = symex_summary["out"]
            work.mark_stage_done("s3", CODE_VERSION + "+triton")
            return summary
        # Asked for Triton but it's not importable — warn, keep concrete only.
        s3_triton_symex.warn_fallback(
            s3_triton_symex.unavailable_reason() or "unknown"
        )
        summary["symex"] = "concrete-fallback"

    work.mark_stage_done("s3", CODE_VERSION)
    return summary


def _write_concrete_dfg(nodes: list[DfgNode], work) -> Path:
    out_path: Path = work.root / "stage_outputs" / "s3_dfg.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for n in nodes:
            row = asdict(n)
            row["pc"] = f"0x{n.pc:x}"
            # ② additive field: omit from the row when empty so a regs_write-
            # populated trace's s3_dfg.jsonl is byte-for-byte unchanged (no new
            # key appears unless an inferred edge actually exists).
            if not n.reg_deps_inferred:
                row.pop("reg_deps_inferred", None)
            f.write(json.dumps(row) + "\n")
    return out_path


def read_dfg(work) -> list[DfgNode]:
    path: Path = work.root / "stage_outputs" / "s3_dfg.jsonl"
    nodes: list[DfgNode] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            inferred = {
                k: InferredProducer(
                    reg=v["reg"], at_idx=int(v["at_idx"]),
                    after_idx=(None if v["after_idx"] is None else int(v["after_idx"])),
                    confidence=v.get("confidence", "inferred"))
                for k, v in o.get("reg_deps_inferred", {}).items()
            }
            nodes.append(DfgNode(
                idx=o["idx"],
                pc=int(o["pc"], 16),
                mnemonic=o["mnemonic"],
                reg_deps={k: (None if v is None else int(v)) for k, v in o["reg_deps"].items()},
                # back-compat: rows written before mem-dep tracking have no key
                mem_deps=tuple(int(x) for x in o.get("mem_deps", ())),
                # ② back-compat: rows without the inferred key → empty.
                reg_deps_inferred=inferred,
            ))
    return nodes
