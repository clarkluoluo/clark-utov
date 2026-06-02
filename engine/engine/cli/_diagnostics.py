"""CLI diagnostics: doctor, phases, trace-info, inspect."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import click

from ..core import Core, CoreConfig, _pick_reader, open_live
from ..runner_client import NullRunnerAdapter, SubprocessRunnerAdapter, TraceReader
from ..types import TargetMeta
from ._app import (main, emit_related, _parse_runner_cmd,
                   _format_pipeline_summary, _safe_read_json, _resolve_run_dir)


@main.command("doctor")
@click.option("--sample-dir", default=None, type=click.Path(file_okay=False, path_type=Path),
              help="Optional path to a example/runner-sha256/-shaped sample fixture tree. "
                   "If unset, sample-fixture checks are skipped (BR-2 §9).")
def doctor_cmd(sample_dir: Path | None) -> None:
    """Scan host for required dependencies; report OK / warning / fail per item."""
    from ..doctor import main as doctor_main
    raise SystemExit(doctor_main(sample_dir=sample_dir))


@main.command("phases")
def phases_cmd() -> None:
    """Print the recommended light-to-heavy phase route for a VMP target.

    Discoverability: the same route is driven statefully over agent-serve
    (methods phase_1_io_observe … phase_5_parity, phase_heavy_vmtrace) where
    order is enforced and vmtrace is gated. There is no "enumerate standard
    crypto" move — the only crypto-source step is phase_3 provenance.
    """
    from ..vmp_phase_api import VMP_PHASE_SEQUENCE
    click.echo("Recommended route (light → heavy). Drive it via agent-serve:")
    for s in VMP_PHASE_SEQUENCE.ordered:
        tag = " [judgement]" if s.is_judgment else ""
        click.echo(f"  {s.order}. {s.name}{tag}")
    for s in VMP_PHASE_SEQUENCE.escalations:
        click.echo(f"  ↑ {s.name}  — ESCALATION: needs proof OR confirmation + a VmtraceBudget")
    click.echo("\nNo candidate-guessing step exists; the crypto-source move is "
               "phase_3 provenance (trace the data flow). See AGENT-WORKFLOW.md §2 Step 0.")
    # Discoverability hint (spec C): the route's crypto-source step is
    # provenance, so point at the next-layer provenance helpers. Verbose only.
    emit_related("trace_provenance")


@main.command("trace-info")
@click.argument("trace_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def trace_info(trace_path: Path) -> None:
    """Sniff format, count instructions, list hot PCs."""
    reader: TraceReader = _pick_reader(trace_path)
    n = 0
    pc_hits: dict[int, int] = {}
    first_pc: int | None = None
    last_pc: int | None = None
    for ins in reader:
        n += 1
        if first_pc is None:
            first_pc = ins.pc
        last_pc = ins.pc
        pc_hits[ins.pc] = pc_hits.get(ins.pc, 0) + 1
    click.echo(f"Format:           {type(reader).__name__}")
    click.echo(f"Instructions:     {n:,}")
    click.echo(f"Unique PCs:       {len(pc_hits):,}")
    click.echo(f"Entry PC:         0x{first_pc:x}" if first_pc else "Entry PC:         <empty>")
    click.echo(f"Exit PC:          0x{last_pc:x}" if last_pc else "Exit PC:          <empty>")
    click.echo("Top 10 hot PCs:")
    for pc, hits in sorted(pc_hits.items(), key=lambda kv: -kv[1])[:10]:
        click.echo(f"  0x{pc:x}: {hits:,}")


@main.command("inspect")
@click.option("--trace", "trace_path", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Trace file to query (JSONL or unidbg text format).")
@click.option("--fingerprint", "fingerprint_name", default=None,
              help="Filter to instructions matching this fingerprint name "
                   "(e.g. 'AES.Te0[0]'). Compares the loaded constant to "
                   "the catalog magic from engine.data.fingerprints.")
@click.option("--pc-range", "pc_range", default=None,
              help="PC range as 'LO..HI' in hex (e.g. '0x40002770..0x400028a0').")
@click.option("--reg-value", "reg_value", default=None,
              help="Match instructions where any register reads/writes this "
                   "value (hex, e.g. '0x40008960'). Useful for following "
                   "a table-base address through the trace.")
@click.option("--anchor-near", "anchor_near", type=int, default=None,
              help="Match a window around this trace_idx. Use with --window.")
@click.option("--window", type=int, default=10,
              help="±N trace lines around --anchor-near (default 10).")
@click.option("--context", "context", type=int, default=0,
              help="Show this many trace lines of context around each hit.")
@click.option("--limit", type=int, default=200,
              help="Cap output at N lines. Default 200; raise for full dumps.")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Emit one JSON object per matching instruction.")
def inspect_cmd(trace_path: Path, fingerprint_name: str | None,
                pc_range: str | None, reg_value: str | None,
                anchor_near: int | None, window: int,
                context: int, limit: int, as_json: bool) -> None:
    """0527 BUG_REPORT-7 §J.7: structured trace queries.

    Replaces the agent's grep-yak (~12 invocations per session per the
    report) with one subcommand. Supports the four query axes that
    matter for evidence-anchored navigation: fingerprint name, PC range,
    register value, trace-index window.

    Filters combine with AND.
    """
    from ..data.fingerprints import FINGERPRINTS, INSTR_PATTERNS

    # Build filters.
    fp_magic: int | None = None
    fp_pattern: str | None = None
    if fingerprint_name:
        for fp in FINGERPRINTS:
            if fp.name == fingerprint_name:
                fp_magic = fp.magic
                break
        else:
            for pat in INSTR_PATTERNS:
                if pat.name == fingerprint_name:
                    fp_pattern = pat.match_text
                    break
            else:
                raise click.UsageError(
                    f"unknown fingerprint name {fingerprint_name!r}. "
                    "List of catalog names in engine/engine/data/fingerprints.py."
                )

    pc_lo = pc_hi = None
    if pc_range:
        try:
            lo_s, hi_s = pc_range.split("..", 1)
            pc_lo, pc_hi = int(lo_s, 16), int(hi_s, 16)
        except ValueError as e:
            raise click.UsageError(f"--pc-range must be 'LO..HI' in hex: {e}")

    reg_v: int | None = None
    if reg_value:
        try:
            reg_v = int(reg_value, 16)
        except ValueError as e:
            raise click.UsageError(f"--reg-value must be hex: {e}")

    anchor_lo = anchor_hi = None
    if anchor_near is not None:
        anchor_lo = max(0, anchor_near - window)
        anchor_hi = anchor_near + window

    def _matches(ins) -> bool:
        if fp_magic is not None and fp_magic not in ins.regs_write.values():
            return False
        if fp_pattern is not None and fp_pattern not in ins.mnemonic:
            return False
        if pc_lo is not None and not (pc_lo <= ins.pc <= pc_hi):
            return False
        if reg_v is not None:
            vals = set(ins.regs_read.values()) | set(ins.regs_write.values())
            if reg_v not in vals:
                return False
        if anchor_lo is not None and not (anchor_lo <= ins.idx <= anchor_hi):
            return False
        return True

    reader: TraceReader = _pick_reader(trace_path)

    matches: list = []
    if context > 0:
        # Buffer-and-emit ±context lines. We stream once, holding a sliding
        # window of size 2*context+1 plus a "lines until quench" counter for
        # post-match emission.
        from collections import deque
        ring: deque = deque(maxlen=context)
        emit_remaining = 0
        for ins in reader:
            if _matches(ins):
                # Flush pre-context first.
                for prev in ring:
                    matches.append(prev)
                matches.append(ins)
                emit_remaining = context
            elif emit_remaining > 0:
                matches.append(ins)
                emit_remaining -= 1
            ring.append(ins)
            if len(matches) >= limit:
                break
    else:
        for ins in reader:
            if _matches(ins):
                matches.append(ins)
                if len(matches) >= limit:
                    break

    if as_json:
        out = []
        for ins in matches:
            out.append({
                "idx":        ins.idx,
                "pc":         f"0x{ins.pc:x}",
                "mnemonic":   ins.mnemonic,
                "regs_read":  {k: f"0x{v:x}" for k, v in ins.regs_read.items()},
                "regs_write": {k: f"0x{v:x}" for k, v in ins.regs_write.items()},
            })
        click.echo(json.dumps(out, indent=2))
    else:
        for ins in matches:
            r = " ".join(f"{k}=0x{v:x}" for k, v in ins.regs_read.items())
            w = " ".join(f"{k}=0x{v:x}" for k, v in ins.regs_write.items())
            click.echo(f"  [{ins.idx:>7}] 0x{ins.pc:08x}  {ins.mnemonic:<48}"
                       + (f"  read: {r}" if r else "")
                       + (f"  ⇒ {w}" if w else ""))
        if len(matches) >= limit:
            click.echo(f"(stopped at --limit {limit}; raise for more)", err=True)
