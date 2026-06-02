"""CLI pipeline drivers: pipeline, resume, rerun-from, agent-serve, pipeline-file."""
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
from ._app import (main, _parse_runner_cmd, _format_pipeline_summary,
                   _safe_read_json, _resolve_run_dir)


@main.command("pipeline")
@click.option("--runner-cmd", required=True,
              help="Shell command that spawns a runner speaking the NDJSON "
                   "protocol on its stdio. Engine does NOT need to know what "
                   "the runner is written in. Example for the bundled Java "
                   "sample: '$PWD/bin/run-runner.sh serve /path/to/lib.so'")
@click.option("--input", "input_hex", required=True,
              help="Hex-encoded input bytes (e.g. '616263' for 'abc').")
@click.option("--work-root", default="work", type=click.Path(file_okay=False, path_type=Path),
              help="Where stage outputs / SQLite ledgers go.")
@click.option("--new-run/--resume", default=True,
              help="--new-run starts a fresh runs/<id>/; --resume continues 'latest'.")
@click.option("--skip-conformance/--with-conformance", default=False,
              help="Skip the conformance gate (PLAN §17). Default: run it "
                   "(--with-conformance is the boolean-toggle complement; "
                   "passing it explicitly is equivalent to omitting the flag). "
                   "Same flag is also on `utov agent-serve` with identical "
                   "semantics. Use --skip-conformance for File-mode runners "
                   "or any runner that legitimately can't pass C1-C5.")
@click.option("--mode", type=click.Choice(["frugal", "aggressive"]), default="frugal",
              help="frugal = no LLM (default). aggressive = run S6 LLM loop.")
@click.option("--budget-tokens", type=int, default=None,
              help="Hard ceiling on total tokens. Stops on breach.")
@click.option("--budget-usd",    type=float, default=None,
              help="Hard ceiling on USD spend. Stops on breach.")
@click.option("--budget-seconds", type=float, default=None,
              help="Hard ceiling on wall-clock seconds.")
@click.option("--budget-calls",  type=int, default=None,
              help="Hard ceiling on LLM call count.")
@click.option("--estimate-only", is_flag=True, default=False,
              help="Run S1..S5 + estimator, print projected cost, exit. No LLM.")
@click.option("--emit-events", is_flag=True, default=False,
              help="Stream NDJSON events to stderr (for agent consumers).")
@click.option("--max-stuck-points", type=int, default=None,
              help="Cap S6 loop to N stuck points (default: no cap; rely on --budget-*)")
@click.option("--llm-backend", type=click.Choice(["deepseek", "mimo", "none"]), default=None,
              help="LLM backend for --mode aggressive. 'none' (the default "
                   "from 0526) returns no hypotheses — S6 runs without any "
                   "API key but produces no LLM-derived findings. 'deepseek' "
                   "and 'mimo' route through DirectBackend with the matching "
                   "env credentials. env LLM_BACKEND wins if the flag isn't passed.")
@click.option("--llm-model", default=None,
              help="LLM model id (e.g. 'deepseek-chat', 'mimo-7b-rl'). "
                   "Falls back to backend's default if unset.")
@click.option("--llm-base-url", default=None,
              help="Override base URL (only honored for --llm-backend mimo; sets MIMO_BASE_URL).")
@click.option("--symex", type=click.Choice(["concrete", "triton"]), default=None,
              help="S3 symbolic-execution mode. 'concrete' (default) builds a "
                   "data-flow graph from the trace's literal register values. "
                   "'triton' (optional dep) runs Triton symbolic execution and "
                   "writes per-instr symbolic ASTs to s3_symex.jsonl. Falls "
                   "back to concrete if Triton isn't importable.")
@click.option("--s6-concurrency", type=int, default=1, show_default=True,
              help="BR-4 §C: prefetch this many S6 stuck-point LLM calls in "
                   "parallel (DirectBackend only — DelegatedBackend stays "
                   "serial per wire protocol). 1 = original sequential path. "
                   "Recommended 4-8 for DeepSeek; profile your rate limits.")
def pipeline(runner_cmd: str, input_hex: str, work_root: Path,
             new_run: bool, skip_conformance: bool, mode: str,
             budget_tokens: int | None, budget_usd: float | None,
             budget_seconds: float | None, budget_calls: int | None,
             estimate_only: bool, emit_events: bool,
             max_stuck_points: int | None,
             llm_backend: str | None, llm_model: str | None,
             llm_base_url: str | None, symex: str | None,
             s6_concurrency: int) -> None:
    """Live mode: spawn runner, run script-mode full pipeline.

    Flow:
      1. Conformance gate (unless --skip-conformance)
      2. Deterministic stages S1..S5 (always)
      3. S6 LLM loop only if --mode=aggressive (default frugal = no LLM)
      4. Budget breach stops cleanly with partial results
      5. --estimate-only stops after S5 + projects S6 cost
    """
    from ..cost import Budget
    from ..estimate import estimate as run_estimate
    from ..orchestrators.script_mode import Mode, run_full_pipeline

    # CLI flags are sugar over the env vars DirectBackend reads. We export them
    # so the rest of the call chain (run_full_pipeline → LLMClient()) picks
    # them up unchanged. CLI wins when both are present.
    if llm_backend is not None:
        os.environ["LLM_BACKEND"] = llm_backend
    # 0526Plan D2: deprecation warning when --mode aggressive runs without
    # an explicit backend choice. Pre-0526 the implicit default was
    # "deepseek"; from 0526 onward the default is "none", which silently
    # produces no LLM findings. Warn so users who genuinely want DeepSeek
    # don't burn an aggressive run to a no-op.
    if mode == "aggressive" and llm_backend is None and "LLM_BACKEND" not in os.environ:
        click.echo(
            "warning: --mode aggressive ran without --llm-backend; the "
            "default is now 'none' (no LLM calls, no LLM findings). Pass "
            "--llm-backend deepseek to keep the pre-0526 behavior.",
            err=True,
        )
    if llm_base_url is not None:
        # Only mimo honors a base URL today; warn quietly if user paired it
        # with a different backend.
        if (llm_backend or os.environ.get("LLM_BACKEND", "deepseek")) != "mimo":
            click.echo("warning: --llm-base-url is only honored for "
                       "--llm-backend mimo; setting MIMO_BASE_URL anyway",
                       err=True)
        os.environ["MIMO_BASE_URL"] = llm_base_url
    # Stash --llm-model so the pipeline can pick it up when constructing
    # LLMClient. (Most call sites construct LLMClient() with no arg; we
    # let the explicit flag override via env shim.)
    if llm_model is not None:
        os.environ["UTOV_LLM_MODEL"] = llm_model
    # Triton symex opt-in flows through an env var so the s3 stage can read
    # it without an explicit ctx kwarg from every call site.
    if symex is not None:
        os.environ["UTOV_SYMEX_MODE"] = symex

    input_bytes = bytes.fromhex(input_hex)
    runner = SubprocessRunnerAdapter(
        cmd=_parse_runner_cmd(runner_cmd),
    )
    try:
        core = open_live(
            work_root.resolve(), runner,
            input_bytes=input_bytes,
            new_run=new_run,
            skip_conformance=skip_conformance,
        )
        click.echo(f"work dir:  {core.work.root}")
        click.echo(f"run_id:    {core.work.run_id}")

        if estimate_only:
            from ..estimate import EstimateConfig
            from ..runner_client import UnidbgTextTraceReader, JsonlTraceReader
            meta = runner.metadata()
            trace_path = runner.get_trace(input_bytes, meta.algo_entry_pc, meta.algo_exit_pc)
            with open(trace_path) as f:
                first = f.readline()
            reader = (JsonlTraceReader(trace_path) if first.lstrip().startswith("{")
                      else UnidbgTextTraceReader(trace_path))
            report = run_estimate(meta, reader, runner,
                                  work_root=work_root.resolve(),
                                  config=EstimateConfig(max_stuck_points=max_stuck_points))
            click.echo("\n" + report.as_human())
            return

        budget = Budget(
            max_total_tokens=budget_tokens,
            max_usd=budget_usd,
            max_wall_seconds=budget_seconds,
            max_calls=budget_calls,
        )
        report = run_full_pipeline(
            core, mode=Mode(mode), budget=budget,
            emit_events_to=sys.stderr if emit_events else None,
            max_stuck_points=max_stuck_points,
            s6_concurrency=s6_concurrency,
        )
        click.echo("\npipeline summary:")
        for s in report.stage_summaries:
            click.echo("  " + json.dumps(s))
        click.echo(f"\nmode:                {mode}")
        click.echo(f"hypotheses_total:    {report.hypothesis_count}")
        click.echo(f"findings_promoted:   {report.findings_promoted}")
        click.echo(f"cost: tokens={report.cost.get('total_tokens', 0):,}  "
                   f"calls={report.cost.get('calls', 0)}  "
                   f"usd=${report.cost.get('usd', 0.0):.4f}  "
                   f"wall={report.cost.get('wall_seconds', 0.0):.1f}s")
        click.echo(f"pacing: {report.progress.get('pacing', '?')}  "
                   f"closures={report.progress.get('closures', 0)}  "
                   f"pending={report.progress.get('pending', 0)}")
        if report.paused:
            click.echo(f"PAUSED: {report.pause_reason}")
        if report.next_actions:
            click.echo("\nnext_actions:")
            for a in report.next_actions:
                click.echo(f"  [{a.severity}] {a.kind}: {a.reason}")
                if a.suggested_command:
                    click.echo(f"      run: {a.suggested_command}")
        hyps = core.get_hypotheses(kind="algo_signature")
        if hyps:
            click.echo(f"\nalgo_signature hypotheses ({len(hyps)}):")
            for h in hyps[:8]:
                click.echo(f"  hyp#{h.id}  conf={h.confidence}  subj={h.subject}  "
                           f"fp={h.payload.get('fingerprint')}  source={h.source}")
    finally:
        runner.shutdown()


@main.command("resume")
@click.argument("work_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--runner-cmd", required=True,
              help="Shell command spawning a runner speaking NDJSON protocol")
@click.option("--input", "input_hex", required=True, help="hex input (same as original run)")
@click.option("--raise-budget-usd",    type=float, default=None)
@click.option("--raise-budget-tokens", type=int, default=None)
@click.option("--raise-budget-seconds", type=float, default=None)
@click.option("--mode", type=click.Choice(["frugal", "aggressive"]), default=None)
@click.option("--emit-events", is_flag=True, default=False)
def resume_cmd(work_dir: Path, runner_cmd: str, input_hex: str,
               raise_budget_usd: float | None, raise_budget_tokens: int | None,
               raise_budget_seconds: float | None,
               mode: str | None, emit_events: bool) -> None:
    """Resume a paused/partial run from an existing work_dir.

    Reads stage_state.json to know what's already done; stages flagged done at
    a matching code_version are skipped. Pass --raise-budget-* to raise caps.
    """
    from ..cost import Budget
    from ..core import CoreConfig, Core, _pick_reader
    from ..orchestrators.script_mode import Mode, run_full_pipeline
    from ..types import TargetMeta

    work_dir = work_dir.resolve()
    if (work_dir / "latest").is_symlink():
        work_dir = (work_dir / "latest").resolve()
    meta = _safe_read_json(work_dir / "meta.json")
    if not meta:
        click.echo("no meta.json found in that dir — can't resume", err=True)
        raise SystemExit(2)

    # Reconstruct TargetMeta from meta.json
    tm = TargetMeta(
        target_name=meta["target_name"], arch=meta["arch"],
        algo_entry_pc=int(meta["algo_entry_pc"], 16),
        algo_exit_pc=int(meta["algo_exit_pc"], 16),
        input_length=None, output_length=32,
    )
    input_bytes = bytes.fromhex(input_hex)
    input_hash = hashlib.sha1(input_bytes).hexdigest()[:12]
    if input_hash != meta.get("input_hash"):
        click.echo(f"WARNING: input hash mismatch (meta={meta.get('input_hash')}, "
                   f"given={input_hash}) — proceeding anyway", err=True)

    runner = SubprocessRunnerAdapter(
        cmd=_parse_runner_cmd(runner_cmd),
    )
    try:
        # Resume the existing run_id
        config = CoreConfig(
            work_root=work_dir.parent.parent.parent,   # work/<target>/<input>/ — three levels up from runs/<run>
            target_meta=tm, input_hash=input_hash,
            driver_mode="script",
            run_id=work_dir.name, new_run=False,
        )
        trace_path = runner.get_trace(input_bytes, tm.algo_entry_pc, tm.algo_exit_pc)
        reader = _pick_reader(Path(trace_path))
        core = Core(config, reader, runner, skip_conformance=True)
        # Clear the paused flag.
        core.resume_run(actor="cli", reason="utov resume command")

        budget = Budget(
            max_total_tokens=raise_budget_tokens,
            max_usd=raise_budget_usd,
            max_wall_seconds=raise_budget_seconds,
        )
        chosen_mode = Mode(mode) if mode else Mode.FRUGAL
        report = run_full_pipeline(
            core, mode=chosen_mode, budget=budget,
            emit_events_to=sys.stderr if emit_events else None,
        )
        click.echo("\nresume summary:")
        for s in report.stage_summaries:
            click.echo("  " + json.dumps(s))
        click.echo(f"\nmode={chosen_mode.value}  paused={report.paused}  "
                   f"cost=${report.cost.get('usd', 0):.4f}")
    finally:
        runner.shutdown()


@main.command("rerun-from")
@click.argument("work_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("stage", type=click.Choice(["s1", "s1b", "s2", "s3", "s4", "s5", "s6"]))
@click.option("--reason", required=True, help="Why are you rerunning?")
@click.option("--actor", default="cli")
def rerun_from(work_dir: Path, stage: str, reason: str, actor: str) -> None:
    """Cascade-invalidate `stage` and everything downstream, ready for re-run.

    Marks hyps abandoned (kept for audit), deletes derived findings, wipes
    stage_state for the cascade. Subsequent pipeline / resume will re-run.
    """
    from ..core import Core, CoreConfig
    from ..runner_client import NullRunnerAdapter
    from ..types import TargetMeta

    work_dir = work_dir.resolve()
    if (work_dir / "latest").is_symlink():
        work_dir = (work_dir / "latest").resolve()
    meta = _safe_read_json(work_dir / "meta.json")
    if not meta:
        click.echo("no meta.json — can't determine target", err=True)
        raise SystemExit(2)
    tm = TargetMeta(target_name=meta["target_name"], arch=meta["arch"],
                    algo_entry_pc=int(meta["algo_entry_pc"], 16),
                    algo_exit_pc=int(meta["algo_exit_pc"], 16),
                    input_length=None, output_length=32)
    runner = NullRunnerAdapter(tm)
    # Use existing trace (any reader works for this op).
    # Build a Core just to access the cascade method; skip conformance.
    config = CoreConfig(
        work_root=work_dir.parent.parent.parent,
        target_meta=tm, input_hash=work_dir.parent.parent.name,
        driver_mode="cli", run_id=work_dir.name, new_run=False,
    )

    class _EmptyReader:
        def __iter__(self): return iter(())
    core = Core(config, _EmptyReader(), runner, skip_conformance=True)
    result = core.rerun_from_stage(stage, actor=actor, reason=reason)
    click.echo(json.dumps(result, indent=2))


@main.command("agent-serve")
@click.option("--runner-cmd", required=True,
              help="Shell command spawning a runner speaking NDJSON protocol")
@click.option("--input", "input_hex", required=True)
@click.option("--work-root", default="work", type=click.Path(file_okay=False, path_type=Path))
@click.option("--new-run/--resume", default=True)
@click.option("--skip-conformance/--with-conformance", default=False,
              help="Skip the conformance gate (PLAN §17). Default: run it "
                   "(--with-conformance is the boolean-toggle complement; "
                   "passing it explicitly is equivalent to omitting the flag). "
                   "Same flag is also on `utov pipeline` with identical "
                   "semantics. BR-2 §15: agent-serve previously had no escape "
                   "hatch when a File-mode runner (or one whose rerun raises "
                   "UnsupportedOperationException) tripped the gate.")
def agent_serve(runner_cmd: str, input_hex: str,
                work_root: Path, new_run: bool,
                skip_conformance: bool) -> None:
    """Agent mode: spawn runner, build Core, serve bidirectional NDJSON on stdio.

    The engine routes its own LLM calls (S6 / blue_team) back to the driving
    agent as `llm_request` messages (the agent answers with its own context).
    Tool calls go agent→engine. Events go engine→agent fire-and-forget.
    """
    from ..llm_client import LLMClient
    from ..orchestrators.agent_mode import serve_mcp
    from ..progress import Tracker
    input_bytes = bytes.fromhex(input_hex)
    runner = SubprocessRunnerAdapter(
        cmd=_parse_runner_cmd(runner_cmd),
    )
    try:
        core = open_live(work_root.resolve(), runner,
                         input_bytes=input_bytes, new_run=new_run,
                         skip_conformance=skip_conformance)
        # Build an LLMClient with a placeholder backend; serve_mcp swaps in
        # the DelegatedBackend tied to stdio when it boots.
        try:
            llm = LLMClient()  # may fail if no DEEPSEEK_API_KEY — fine, we'll override
        except Exception:
            from ..llm_client import DelegatedBackend
            llm = LLMClient(backend=DelegatedBackend(in_stream=None, out_stream=None))
        tracker = Tracker(meter=__import__("engine.cost", fromlist=["CostMeter"]).CostMeter())
        serve_mcp(core, llm=llm, tracker=tracker)
    finally:
        runner.shutdown()


@main.command("pipeline-file")
@click.option("--trace", "trace_path", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--target-name", required=True, help="Logical target name (e.g. libEncryptor.so).")
@click.option("--entry", "entry_pc_hex", required=True, help="algo_entry_pc as hex (e.g. 0x40007d88).")
@click.option("--exit", "exit_pc_hex", required=True, help="algo_exit_pc as hex.")
@click.option("--input-len", default=16, type=int, help="Probe input length (bytes).")
@click.option("--output-len", default=32, type=int, help="Algorithm output length (bytes).")
@click.option("--work-root", default="work", type=click.Path(file_okay=False, path_type=Path))
@click.option("--new-run/--resume", default=True)
@click.option("--so", "so_path", default=None,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="BUG_REPORT-6 ceiling 2 / BUG_REPORT-7 §J.4: path to the "
                   "target .so. Enables the static-artifact scan to look for "
                   "AES keys / IVs adjacent to identified algorithm code.")
def pipeline_file(trace_path: Path, target_name: str, entry_pc_hex: str, exit_pc_hex: str,
                  input_len: int, output_len: int, work_root: Path, new_run: bool,
                  so_path: Path | None) -> None:
    """File mode: static trace + metadata → S1..S5 + the layer-0/1/2 verify chain.

    Same orchestration as Live mode's `pipeline` command minus the LLM/S6 loop:
    after S1..S5 we run plugin-verify, handler-semantic verify (binop/unary/
    imm/ext/bfx/ch/triton), σ/Σ fold, and algorithm-template fit so the run
    actually produces findings — not just unverified hypotheses. C1-C3 SKIP,
    C4 PASS still required.
    """
    from ..orchestrators.script_mode import Mode, run_full_pipeline

    meta = TargetMeta(
        target_name=target_name, arch="arm64",
        algo_entry_pc=int(entry_pc_hex, 16),
        algo_exit_pc=int(exit_pc_hex, 16),
        input_length=input_len, output_length=output_len,
    )
    runner = NullRunnerAdapter(meta)
    reader = _pick_reader(trace_path)
    input_hash = hashlib.sha1(b"static-trace").hexdigest()[:12]
    config = CoreConfig(
        work_root=work_root.resolve(), target_meta=meta,
        input_hash=input_hash, driver_mode="script", new_run=new_run,
    )
    core = Core(config, reader, runner, skip_conformance=False)
    click.echo(f"work dir:  {core.work.root}")
    if so_path is not None:
        core.session["so_path"] = str(so_path.resolve())
        # Persist so a resume run can still find the .so.
        core.checkpoint()

    report = run_full_pipeline(core, mode=Mode.FRUGAL)
    click.echo("\npipeline summary:")
    for s in report.stage_summaries:
        click.echo("  " + json.dumps(s))
    click.echo(f"\nhypotheses_total:    {report.hypothesis_count}")
    click.echo(f"findings_promoted:   {report.findings_promoted}")
    if report.next_actions:
        click.echo("\nnext_actions:")
        for a in report.next_actions:
            click.echo(f"  [{a.severity}] {a.kind}: {a.reason}")
            if a.suggested_command:
                click.echo(f"      run: {a.suggested_command}")

    # Algorithm results: the structural matcher emits ``algorithm_hyp`` (a
    # pre-oracle-closure HYPOTHESIS carrying a local-closure trap — task 7); the
    # strong ``algorithm_identified`` is reserved for whole-case oracle closure.
    # Show both, labelled, so a reader never mistakes a hyp for a final answer.
    algo = (core.get_hypotheses(kind="algorithm_hyp")
            + core.get_hypotheses(kind="algorithm_identified"))
    if algo:
        click.echo(f"\nalgorithm results ({len(algo)}):")
        for h in algo:
            payload = h.payload or {}
            trap = (payload.get("closure") or {}).get("trap_state")
            tag = " [HYP/trap]" if h.kind == "algorithm_hyp" else " [oracle-closed]"
            if trap and h.kind == "algorithm_hyp":
                tag = f" [HYP/{trap}]"
            click.echo(f"  hyp#{h.id}  {h.kind}{tag}  conf={h.confidence}  "
                       f"subj={h.subject}  "
                       f"evidence={payload.get('evidence_score')}  "
                       f"anchors_seen={len(payload.get('anchors_seen', []))}/"
                       f"{len(payload.get('anchors_expected', []))}")

    hyps = core.get_hypotheses(kind="algo_signature")
    if hyps:
        click.echo(f"\nalgo_signature hypotheses ({len(hyps)}):")
        for h in hyps[:8]:
            click.echo(f"  hyp#{h.id}  conf={h.confidence}  subj={h.subject}  "
                       f"fp={h.payload.get('fingerprint')}  hits={h.payload.get('hits')}")

    if report.findings_promoted == 0:
        click.echo(
            "\ntip: 0 findings promoted. Inspect with "
            "`utov status <work-dir> --json --by-source --by-kind`, "
            "or rerun with `--mode aggressive --llm-backend deepseek` "
            "to engage the LLM hypothesis loop.",
            err=True,
        )
