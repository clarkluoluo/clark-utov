"""S6-taint: forward dynamic taint propagation (additive, target-agnostic).

Reuses the same reg + memory dependency structure S3 captures (regs_read /
regs_write + byte-granular memory ops), but propagates LABEL SETS *forward* in a
single pass instead of building backward edges. Fully parameterised: taint seeds
and sinks come from ``ctx``; NO target address is hardcoded here.

Name note: the pipeline's "S6" is already the LLM hypothesis loop
(:mod:`engine.stages.s6_hypothesis`, stage key ``"s6"``). This forward-taint
stage registers under the DISTINCT key ``"s6_taint"`` to avoid colliding with
it; its artifact is ``s6_taint.jsonl``.

Algorithm (single forward pass in idx order — S3 producers are always earlier):

    reg_labels: dict[reg, set[label]]       # current taint on each register
    mem_labels: dict[byte_addr, set[label]] # current taint on each memory byte

    for each instruction i:
        in_labels = ∪ reg_labels[r] for r in i.regs_read
                  ∪ mem_labels[b]   for each byte b of each memory READ op
        for r in i.regs_write:            reg_labels[r] = set(in_labels)
        for each byte b of each WRITE op: mem_labels[b] = set(in_labels)

Seeds (``ctx['taint_sources']``) initialise reg_labels / mem_labels first; sinks
(``ctx['sinks']``, same key S4 uses) are snapshotted as their tainted bytes pass.

Honest marking (point 4 — never silently dropped):
  - a memory READ of a byte neither seeded nor written earlier in this trace →
    a ``could_not_close`` breakpoint (the provenance has a hole — we cannot tell
    whether that external byte carried taint), recorded, not discarded.
  - a memory op that cannot be resolved to concrete bytes (size <= 0) →
    an ``unparseable_mem_op`` breakpoint.

``taint_sources`` shape (addresses are caller-supplied — none are baked in)::

    {
      "key":   {"mem": [[<seed_addr>, <size>]], "regs": [<reg>]},  # label -> seeds
      "nonce": {"regs": [<reg>]},
    }

Output ``stage_outputs/s6_taint.jsonl``: one row per tainted sink byte
``{kind:"sink_byte", sink_idx, addr, source_labels:[...], handler_pcs:[...]}``
(and per tainted sink register, ``kind:"sink_reg"``), then a final
``{kind:"summary", by_source:{label:[addr,...]}, breakpoints:[...], status}`` row.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..types import Instruction

CODE_VERSION = "s6-taint-v1"


@dataclass
class TaintResult:
    sink_bytes: list[dict] = field(default_factory=list)   # tainted sink mem bytes
    sink_regs: list[dict] = field(default_factory=list)    # tainted sink reg writes
    by_source: dict[str, list[str]] = field(default_factory=dict)  # label -> [addr]
    breakpoints: list[dict] = field(default_factory=list)  # could_not_close markers

    @property
    def status(self) -> str:
        return "could_not_close" if self.breakpoints else "closed"


def _seed(taint_sources):
    reg_labels: dict[str, set[str]] = {}
    mem_labels: dict[int, set[str]] = {}
    provenance_bytes: set[int] = set()      # bytes with known provenance (seed/stored)
    for label, spec in (taint_sources or {}).items():
        lab = str(label)
        spec = spec or {}
        for r in spec.get("regs", []) or []:
            reg_labels.setdefault(str(r), set()).add(lab)
        for entry in spec.get("mem", []) or []:
            addr, size = int(entry[0]), int(entry[1])
            for b in range(addr, addr + size):
                mem_labels.setdefault(b, set()).add(lab)
                provenance_bytes.add(b)
    return reg_labels, mem_labels, provenance_bytes


def _handlers_for(labels, label_handlers) -> list[str]:
    """Ordered-unique union of handler PCs across the given labels."""
    seen: list[int] = []
    for lab in labels:
        for pc in label_handlers.get(lab, ()):
            if pc not in seen:
                seen.append(pc)
    return [f"0x{pc:x}" for pc in seen]


def propagate_taint(
    items: Iterable[Instruction],
    taint_sources: dict | None,
    sinks: Iterable[int] | None,
) -> TaintResult:
    items_list = list(items)
    sink_set = {int(s) for s in (sinks or [])}
    reg_labels, mem_labels, provenance_bytes = _seed(taint_sources)
    label_handlers: dict[str, list[int]] = {}   # label -> ordered-unique handler PCs
    res = TaintResult()
    seen_breakpoints: set[tuple] = set()         # dedup (idx, kind, addr)

    def _mark(idx, pc, kind, addr=None, detail=""):
        key = (idx, kind, addr)
        if key in seen_breakpoints:
            return
        seen_breakpoints.add(key)
        bp = {"idx": idx, "pc": f"0x{pc:x}", "kind": kind, "detail": detail}
        if addr is not None:
            bp["addr"] = f"0x{addr:x}"
        res.breakpoints.append(bp)

    for ins in items_list:
        in_labels: set[str] = set()
        for r in ins.regs_read:
            in_labels |= reg_labels.get(r, set())
        for op in ins.mem:
            if op.rw != "r":
                continue
            if op.size <= 0:
                _mark(ins.idx, ins.pc, "unparseable_mem_op", addr=op.addr,
                      detail="memory read op with size <= 0")
                continue
            op_has_hole = False
            for b in range(op.addr, op.addr + op.size):
                if b in mem_labels:
                    in_labels |= mem_labels[b]
                elif b not in provenance_bytes:
                    op_has_hole = True
            if op_has_hole:
                # read of memory never stored in-trace and not seeded — the taint
                # provenance has a hole. Recorded, not dropped (any resolved bytes
                # in the same op still propagate their labels above).
                _mark(ins.idx, ins.pc, "unresolved_mem_read", addr=op.addr,
                      detail=(f"reads memory [0x{op.addr:x},0x{op.addr + op.size:x}) "
                              f"never written in this trace"))

        if in_labels:
            for lab in in_labels:
                lst = label_handlers.setdefault(lab, [])
                if not lst or lst[-1] != ins.pc:
                    lst.append(ins.pc)

        for r in ins.regs_write:
            reg_labels[r] = set(in_labels)
        for op in ins.mem:
            if op.rw == "w" and op.size > 0:
                for b in range(op.addr, op.addr + op.size):
                    mem_labels[b] = set(in_labels)
                    provenance_bytes.add(b)

        if ins.idx in sink_set:
            labs = sorted(in_labels)
            handlers = _handlers_for(labs, label_handlers)
            for op in ins.mem:
                if op.rw == "w" and op.size > 0:
                    for b in range(op.addr, op.addr + op.size):
                        if not labs:
                            continue
                        addr_hex = f"0x{b:x}"
                        res.sink_bytes.append({
                            "kind": "sink_byte", "sink_idx": ins.idx,
                            "addr": addr_hex, "source_labels": labs,
                            "handler_pcs": handlers,
                        })
                        for lab in labs:
                            lst = res.by_source.setdefault(lab, [])
                            if addr_hex not in lst:
                                lst.append(addr_hex)
            for r in ins.regs_write:
                if not labs:
                    continue
                res.sink_regs.append({
                    "kind": "sink_reg", "sink_idx": ins.idx, "reg": r,
                    "source_labels": labs, "handler_pcs": handlers,
                })
                for lab in labs:
                    lst = res.by_source.setdefault(lab, [])
                    tag = f"reg:{r}"
                    if tag not in lst:
                        lst.append(tag)

    return res


def run(ctx) -> dict:
    items = ctx["items"]
    sources = ctx.get("taint_sources") or {}
    sinks = ctx.get("sinks") or []
    res = propagate_taint(items, sources, sinks)

    out_path: Path | None = None
    work = ctx.get("work")
    if work is not None:
        out_path = work.root / "stage_outputs" / "s6_taint.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in res.sink_bytes:
                f.write(json.dumps(row) + "\n")
            for row in res.sink_regs:
                f.write(json.dumps(row) + "\n")
            f.write(json.dumps({
                "kind": "summary",
                "by_source": res.by_source,
                "breakpoints": res.breakpoints,
                "status": res.status,
            }) + "\n")
        work.mark_stage_done("s6_taint", CODE_VERSION)

    return {
        "stage": "s6_taint",
        "sources": sorted(str(k) for k in sources),
        "sink_bytes": len(res.sink_bytes),
        "sink_regs": len(res.sink_regs),
        "breakpoints": len(res.breakpoints),
        "status": res.status,
        "out": str(out_path) if out_path else None,
    }
