"""CLI queries: status, compare, audit, findings, hyps, override, emit, verify-construction."""
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


@main.command("status")
@click.argument("work_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--by-source/--no-by-source", default=False,
              help="Break the findings count down by source "
                   "(plugin / s5_deterministic / s5_triton / s5_fold_idiom / "
                   "s5_algorithm_fit / s6_llm / agent_override / etc).")
@click.option("--by-stage/--no-by-stage", default=False,
              help="Break the findings count down by promote stage.")
@click.option("--by-kind/--no-by-kind", default=False,
              help="Break the findings count down by finding kind.")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Emit one JSON object instead of human-readable lines "
                   "(agent-friendly).")
def status(work_dir: Path, by_source: bool, by_stage: bool, by_kind: bool,
           as_json: bool) -> None:
    """Inspect a work/<target>/<input>/runs/<run>/ directory and report status.

    The --by-* flags add findings-distribution breakdowns; --json emits one
    structured object on stdout (B1 — agents consume this without parsing
    human lines).
    """
    import sqlite3
    work_dir = work_dir.resolve()
    if (work_dir / "latest").is_symlink():
        work_dir = (work_dir / "latest").resolve()
    meta = _safe_read_json(work_dir / "meta.json")
    stage_state = _safe_read_json(work_dir / "stage_state.json")
    session = _safe_read_json(work_dir / "session.json")
    cf = _safe_read_json(work_dir / "conformance_report.json")

    hyp_total = 0
    hyp_counts: dict[str, int] = {}
    hyp_db = work_dir / "hypotheses.sqlite"
    if hyp_db.exists():
        try:
            conn = sqlite3.connect(hyp_db)
            try:
                hyp_counts = {r[0]: r[1] for r in conn.execute(
                    "SELECT status, COUNT(*) FROM hypotheses GROUP BY status"
                ).fetchall()}
                hyp_total = sum(hyp_counts.values())
            finally:
                conn.close()
        except sqlite3.OperationalError:
            pass

    findings_total = 0
    findings_by_source: dict[str, int] = {}
    findings_by_stage: dict[str, int] = {}
    findings_by_kind: dict[str, int] = {}
    fi_db = work_dir / "findings.sqlite"
    if fi_db.exists():
        try:
            conn = sqlite3.connect(fi_db)
            try:
                findings_total = conn.execute(
                    "SELECT COUNT(*) FROM findings"
                ).fetchone()[0]
                if by_source or as_json:
                    findings_by_source = {r[0]: r[1] for r in conn.execute(
                        "SELECT source, COUNT(*) FROM findings GROUP BY source"
                    ).fetchall()}
                if by_stage or as_json:
                    findings_by_stage = {r[0]: r[1] for r in conn.execute(
                        "SELECT stage, COUNT(*) FROM findings GROUP BY stage"
                    ).fetchall()}
                if by_kind or as_json:
                    findings_by_kind = {r[0]: r[1] for r in conn.execute(
                        "SELECT kind, COUNT(*) FROM findings GROUP BY kind"
                    ).fetchall()}
            finally:
                conn.close()
        except sqlite3.OperationalError:
            pass

    if as_json:
        click.echo(json.dumps({
            "run_dir":           str(work_dir),
            "target":            meta.get("target_name"),
            "arch":              meta.get("arch"),
            "driver_mode":       meta.get("driver_mode"),
            "run_id":            meta.get("run_id"),
            "paused":            meta.get("paused", False),
            "pause_reason":      meta.get("pause_reason"),
            "stages_done":       sorted(stage_state.keys()) if stage_state else [],
            "conformance":       cf,
            "hypotheses_total":  hyp_total,
            "hypotheses_by_status": hyp_counts,
            "findings_total":    findings_total,
            "findings_by_source": findings_by_source,
            "findings_by_stage": findings_by_stage,
            "findings_by_kind":  findings_by_kind,
        }, indent=2))
        return

    click.echo(f"run dir:        {work_dir}")
    click.echo(f"target:         {meta.get('target_name','?')}  arch={meta.get('arch','?')}")
    if meta.get("emulator_name"):
        click.echo(f"emulator:       {meta.get('emulator_name')} "
                   f"v{meta.get('emulator_version') or '?'}")
    click.echo(f"driver_mode:    {meta.get('driver_mode','?')}  run_id={meta.get('run_id','?')}")
    click.echo(f"paused:         {meta.get('paused', False)}"
               + (f"  reason={meta.get('pause_reason')}" if meta.get('paused') else ""))
    click.echo(f"stages_done:    {sorted(stage_state.keys()) if stage_state else '(none)'}")
    if cf:
        click.echo(f"conformance:    {cf.get('overall','?')}  "
                   f"mode={cf.get('mode','?')}  "
                   f"verifier_degraded={cf.get('verifier_degraded', False)}")
    if session:
        click.echo(f"session keys:   {list(session.keys())}")
    if hyp_db.exists():
        click.echo(f"hypotheses:     {hyp_total} total  {hyp_counts}")
    if fi_db.exists():
        click.echo(f"findings:       {findings_total}")
        if by_source and findings_by_source:
            click.echo("  by source:")
            for src, n in sorted(findings_by_source.items(), key=lambda kv: -kv[1]):
                click.echo(f"    {src:24s} {n:>6}")
        if by_stage and findings_by_stage:
            click.echo("  by stage:")
            for stg, n in sorted(findings_by_stage.items(), key=lambda kv: -kv[1]):
                click.echo(f"    {stg:24s} {n:>6}")
        if by_kind and findings_by_kind:
            click.echo("  by kind:")
            for k, n in sorted(findings_by_kind.items(), key=lambda kv: -kv[1]):
                click.echo(f"    {k:24s} {n:>6}")


@main.command("compare")
@click.argument("run_a", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("run_b", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, default=False)
def compare(run_a: Path, run_b: Path, as_json: bool) -> None:
    """Diff two completed runs (0526Plan B4).

    Compares findings counts (total / by source / by kind), hypothesis
    distributions, and the set of finding subjects exclusive to each side.
    Useful for before/after experiments — patch X, did frugal recover
    more findings? Did σ/Σ idioms appear at new PCs? --json emits a
    structured object for an agent to consume.
    """
    import sqlite3

    def _resolve(p: Path) -> Path:
        p = p.resolve()
        if (p / "latest").is_symlink():
            p = (p / "latest").resolve()
        return p

    def _summary(p: Path) -> dict:
        out: dict = {
            "run_dir":       str(p),
            "findings_total": 0,
            "by_source":     {},
            "by_kind":       {},
            "by_stage":      {},
            "subjects":      set(),
            "hyp_total":     0,
            "hyp_by_status": {},
        }
        fi = p / "findings.sqlite"
        if fi.exists():
            try:
                c = sqlite3.connect(fi)
                try:
                    out["findings_total"] = c.execute(
                        "SELECT COUNT(*) FROM findings"
                    ).fetchone()[0]
                    out["by_source"] = {r[0]: r[1] for r in c.execute(
                        "SELECT source, COUNT(*) FROM findings GROUP BY source"
                    ).fetchall()}
                    out["by_kind"] = {r[0]: r[1] for r in c.execute(
                        "SELECT kind, COUNT(*) FROM findings GROUP BY kind"
                    ).fetchall()}
                    out["by_stage"] = {r[0]: r[1] for r in c.execute(
                        "SELECT stage, COUNT(*) FROM findings GROUP BY stage"
                    ).fetchall()}
                    out["subjects"] = {
                        r[0] for r in c.execute(
                            "SELECT subject FROM findings"
                        ).fetchall()
                    }
                finally:
                    c.close()
            except sqlite3.OperationalError:
                pass
        hy = p / "hypotheses.sqlite"
        if hy.exists():
            try:
                c = sqlite3.connect(hy)
                try:
                    out["hyp_by_status"] = {r[0]: r[1] for r in c.execute(
                        "SELECT status, COUNT(*) FROM hypotheses GROUP BY status"
                    ).fetchall()}
                    out["hyp_total"] = sum(out["hyp_by_status"].values())
                finally:
                    c.close()
            except sqlite3.OperationalError:
                pass
        return out

    a = _summary(_resolve(run_a))
    b = _summary(_resolve(run_b))

    diff = {
        "findings_total_delta": b["findings_total"] - a["findings_total"],
        "by_source_delta":      {},
        "by_kind_delta":        {},
        "by_stage_delta":       {},
        "only_in_a":            sorted(a["subjects"] - b["subjects"])[:50],
        "only_in_b":            sorted(b["subjects"] - a["subjects"])[:50],
    }
    for key in ("by_source", "by_kind", "by_stage"):
        names = set(a[key]) | set(b[key])
        diff[f"{key}_delta"] = {
            n: b[key].get(n, 0) - a[key].get(n, 0) for n in sorted(names)
        }

    if as_json:
        a["subjects"] = sorted(a["subjects"])[:50]
        b["subjects"] = sorted(b["subjects"])[:50]
        click.echo(json.dumps({"a": a, "b": b, "diff": diff}, indent=2))
        return

    click.echo(f"a:  {a['run_dir']}")
    click.echo(f"b:  {b['run_dir']}")
    click.echo(f"\nfindings_total:  a={a['findings_total']}  "
               f"b={b['findings_total']}  Δ={diff['findings_total_delta']:+d}")
    if diff["by_source_delta"]:
        click.echo("\nby_source Δ (b − a):")
        for src, d in sorted(diff["by_source_delta"].items(),
                              key=lambda kv: -abs(kv[1])):
            if d != 0:
                click.echo(f"  {src:24s} {d:+5d}  "
                           f"(a={a['by_source'].get(src,0)}, "
                           f"b={b['by_source'].get(src,0)})")
    if diff["only_in_b"]:
        click.echo(f"\nnew subjects in b (first {len(diff['only_in_b'])}):")
        for s in diff["only_in_b"][:20]:
            click.echo(f"  + {s}")
    if diff["only_in_a"]:
        click.echo(f"\nsubjects removed in b (first {len(diff['only_in_a'])}):")
        for s in diff["only_in_a"][:20]:
            click.echo(f"  - {s}")


@main.command("audit")
@click.argument("work_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--limit", default=20, type=int)
@click.option("--action", default=None, help="filter by action name")
def audit_cmd(work_dir: Path, limit: int, action: str | None) -> None:
    """Show recent interventions logged on this work_dir."""
    import sqlite3
    work_dir = work_dir.resolve()
    if (work_dir / "latest").is_symlink():
        work_dir = (work_dir / "latest").resolve()
    db = work_dir / "hypotheses.sqlite"
    if not db.exists():
        click.echo("no hypotheses.sqlite", err=True)
        raise SystemExit(2)
    conn = sqlite3.connect(db)
    try:
        from ..store import read_interventions
        rows = read_interventions(conn, limit=limit, action=action)
    finally:
        conn.close()
    click.echo(f"interventions (most recent {len(rows)}):")
    for r in rows:
        click.echo(f"  #{r['id']:4}  {r['timestamp']}  "
                   f"{r['actor']:8} {r['action']:24} "
                   f"{r['target_table'] or '-':16} "
                   f"target_id={r['target_id'] or '-'}  "
                   f"reason={(r['reason'] or '')[:60]!r}")


@main.command("findings")
@click.argument("work_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--source", default=None)
@click.option("--stage", default=None)
@click.option("--kind", default=None)
@click.option("--subject-like", default=None)
@click.option("--limit", default=None, type=int)
@click.option("--json", "as_json", is_flag=True, default=False)
def findings_cmd(work_dir: Path, source, stage, kind, subject_like, limit, as_json) -> None:
    """Query findings.sqlite (CLI mirror of the get_findings RPC method)."""
    import sqlite3
    db = _resolve_run_dir(work_dir) / "findings.sqlite"
    if not db.exists():
        click.echo("no findings.sqlite", err=True); raise SystemExit(2)
    sql = ("SELECT id, stage, kind, subject, source, verifier_strategy,"
           " verified_at, origin_hyp_id, payload_ref FROM findings")
    clauses, args = [], []
    for col, val in (("source", source), ("stage", stage), ("kind", kind)):
        if val is not None:
            clauses.append(f"{col} = ?"); args.append(val)
    if subject_like is not None:
        clauses.append("subject LIKE ?"); args.append(subject_like)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY id ASC"
    if limit is not None:
        sql += " LIMIT ?"; args.append(int(limit))
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(sql, args).fetchall()
    finally:
        conn.close()
    keys = ("id", "stage", "kind", "subject", "source", "verifier_strategy",
            "verified_at", "origin_hyp_id", "payload_ref")
    out = [dict(zip(keys, r)) for r in rows]
    if as_json:
        click.echo(json.dumps(out)); return
    click.echo(f"findings ({len(out)}):")
    for d in out:
        click.echo(f"  #{d['id']:4} {d['stage']:4} {d['kind']:20} "
                   f"{d['source']:18} {(d['subject'] or '')[:48]}")


@main.command("hyps")
@click.argument("work_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--status", default=None)
@click.option("--kind", default=None)
@click.option("--source", default=None)
@click.option("--limit", default=None, type=int)
@click.option("--json", "as_json", is_flag=True, default=False)
def hyps_cmd(work_dir: Path, status, kind, source, limit, as_json) -> None:
    """Query hypotheses (CLI mirror of get_hypotheses; same HypTree.query path)."""
    import sqlite3
    db = _resolve_run_dir(work_dir) / "hypotheses.sqlite"
    if not db.exists():
        click.echo("no hypotheses.sqlite", err=True); raise SystemExit(2)
    from ..hyp_tree import HypTree
    conn = sqlite3.connect(db)
    try:
        hyps = HypTree(conn).query(status=status, kind=kind, source=source, limit=limit)
    finally:
        conn.close()
    out = [{"id": h.id, "parent_id": h.parent_id, "depth": h.depth,
            "status": h.status, "kind": h.kind, "source": h.source,
            "subject": h.subject, "confidence": h.confidence} for h in hyps]
    if as_json:
        click.echo(json.dumps(out)); return
    click.echo(f"hypotheses ({len(out)}):")
    for d in out:
        click.echo(f"  #{d['id']:4} {d['status']:12} {d['kind']:18} "
                   f"conf={d['confidence']}  {(d['subject'] or '')[:44]}")


@main.command("override")
@click.argument("work_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("hyp_id", type=int)
@click.argument("verdict", type=click.Choice(["pass", "fail", "inconclusive"]))
@click.option("--reason", required=True, help="Why you are overriding (logged).")
@click.option("--actor", default="cli")
def override_cmd(work_dir: Path, hyp_id: int, verdict: str, reason: str, actor: str) -> None:
    """Flip a hypothesis verdict + log an intervention row (CLI mirror of the
    override_verdict RPC method; uses the same HypTree.mark_verdict +
    log_intervention path, so behaviour matches agent-serve)."""
    import sqlite3
    db = _resolve_run_dir(work_dir) / "hypotheses.sqlite"
    if not db.exists():
        click.echo("no hypotheses.sqlite", err=True); raise SystemExit(2)
    from ..hyp_tree import HypTree
    from ..store import log_intervention
    conn = sqlite3.connect(db)
    try:
        HypTree(conn).mark_verdict(hyp_id, verdict,
                                   {"override_by": actor, "reason": reason})
        log_intervention(conn, actor=actor, action="override_verdict",
                         target_table="hypotheses", target_id=hyp_id, reason=reason)
    finally:
        conn.close()
    click.echo(f"hyp #{hyp_id} → {verdict}  (logged, actor={actor})")


@main.command("emit")
@click.argument("run_dir",
                type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", "output_path", default=None,
              type=click.Path(dir_okay=False, path_type=Path),
              help="Write rendered text here instead of stdout.")
@click.option("--format", "fmt",
              type=click.Choice(["text", "markdown"]),
              default="text",
              help="Output format. text = plain (default); markdown = "
                   "fenced code block + H1 title (consumed by README "
                   "renderers, agent UIs).")
def emit_cmd(run_dir: Path, output_path: Path | None, fmt: str) -> None:
    """FEATURE-REQUEST-1: render Tier 1 pseudocode for a finished run.

    Reads `<run_dir>/findings.sqlite` + `meta.json` + the s3 DFG, then
    renders the latest `algorithm_identified` finding as paste-and-read
    pseudocode populated with the recovered IVs, K table (when present),
    σ/Σ idiom PCs, and observed loop counts. Supported algorithms come
    from `engine/engine/data/algorithm_pseudocode.py:ALGORITHM_SPECS`.

    Examples:

        utov emit work/libsha256.so/.../runs/20260526-184227-573ab6
        utov emit <run> --output pseudocode.md --format markdown
    """
    from ..emitter import EmitterError, emit
    try:
        if output_path is not None:
            with output_path.open("w", encoding="utf-8") as f:
                emit(run_dir, out=f, fmt=fmt)
            click.echo(f"wrote {output_path}")
        else:
            emit(run_dir, out=sys.stdout, fmt=fmt)
    except EmitterError as e:
        click.echo(f"emit: {e}", err=True)
        raise SystemExit(2)


@main.command("verify-construction")
@click.option("--construction", required=True,
              help="Python lambda string. Takes input bytes as first positional "
                   "arg; additional kwargs come from --construction-args. "
                   "Examples: 'lambda x: hashlib.sha256(x).digest()' or "
                   "'lambda pt, key, iv: <AES expr>'.")
@click.option("--inputs", required=True, multiple=True,
              help="Hex-encoded input bytes. Pass --inputs multiple times for "
                   "multi-trial verification.")
@click.option("--construction-args", default="{}",
              help="JSON object of additional kwargs. Values that look like "
                   "hex get decoded to bytes; otherwise passed as-is.")
@click.option("--runner-cmd", default=None,
              help="If set, spawn this runner via NDJSON and compare its "
                   "rerun(input) bytes against the construction output. "
                   "If unset, only the construction output is computed (use "
                   "this to vet a candidate before wiring a runner).")
@click.option("--output-length", type=int, default=None,
              help="Truncate both sides to this many bytes before comparing. "
                   "Default: min(len(construction_out), len(runner_out)).")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Emit JSON instead of human lines.")
def verify_construction(construction: str, inputs: tuple[str, ...],
                        construction_args: str, runner_cmd: str | None,
                        output_length: int | None, as_json: bool) -> None:
    """0527 BUG_REPORT-7 §J.6: behavioral-match gate via candidate vs runner.

    The construction is evaluated in a restricted namespace (hashlib,
    cryptography.hazmat ciphers; no os/subprocess/import). NOT a security
    sandbox — assume the construction is agent-authored, not adversarial.
    Blacklists dunders + obvious escape patterns to catch typos, not attacks.

    Pairs naturally with --reference_impl in algorithm_identified payloads
    (BUG_REPORT-7 §J.5): copy the lambda+import, fill in the unknowns,
    pass --inputs from your test vectors.
    """
    import hashlib

    SAFE_GLOBALS: dict[str, Any] = {
        "__builtins__": {
            "int": int, "bytes": bytes, "bytearray": bytearray,
            "len": len, "range": range, "list": list, "tuple": tuple,
            "min": min, "max": max, "sum": sum, "abs": abs,
            "ord": ord, "chr": chr,
        },
        "hashlib": hashlib,
    }
    try:
        from cryptography.hazmat.primitives.ciphers import (
            Cipher, algorithms, modes,
        )
        SAFE_GLOBALS["Cipher"]     = Cipher
        SAFE_GLOBALS["algorithms"] = algorithms
        SAFE_GLOBALS["modes"]      = modes
    except ImportError:
        pass

    expr = construction.strip()
    if not expr.startswith("lambda"):
        raise click.UsageError("--construction must be a lambda expression")
    forbidden = ("__", "import ", "open(", "exec(", "eval(",
                 "compile(", "globals(", "locals(", "subprocess", "os.")
    for tok in forbidden:
        if tok in expr:
            raise click.UsageError(
                f"--construction contains forbidden token {tok!r}. "
                "Use only hashlib + cryptography.hazmat names."
            )
    try:
        fn = eval(expr, SAFE_GLOBALS, {})
    except SyntaxError as exc:
        raise click.UsageError(f"--construction parse error: {exc}")

    try:
        raw_args = json.loads(construction_args)
    except json.JSONDecodeError as exc:
        raise click.UsageError(f"--construction-args is not valid JSON: {exc}")
    if not isinstance(raw_args, dict):
        raise click.UsageError("--construction-args must be a JSON object")

    kwargs: dict[str, Any] = {}
    for k, v in raw_args.items():
        if isinstance(v, str):
            try:
                kwargs[k] = bytes.fromhex(v)
            except ValueError:
                kwargs[k] = v
        else:
            kwargs[k] = v

    construction_sha1 = hashlib.sha1(expr.encode()).hexdigest()[:12]

    runner = None
    if runner_cmd:
        runner = SubprocessRunnerAdapter(cmd=_parse_runner_cmd(runner_cmd))

    trials: list[dict] = []
    n_pass = 0
    try:
        for input_hex in inputs:
            try:
                input_bytes = bytes.fromhex(input_hex)
            except ValueError as exc:
                trials.append({
                    "input_hex":          input_hex,
                    "construction_error": f"input not hex: {exc}",
                    "match":              False,
                })
                continue
            try:
                expected = fn(input_bytes, **kwargs)
            except Exception as exc:
                trials.append({
                    "input_hex":          input_hex,
                    "construction_error": f"{type(exc).__name__}: {exc}",
                    "match":              False,
                })
                continue
            if not isinstance(expected, (bytes, bytearray)):
                trials.append({
                    "input_hex":          input_hex,
                    "construction_error": (
                        f"construction returned {type(expected).__name__}, "
                        "expected bytes"
                    ),
                    "match":              False,
                })
                continue

            if runner is None:
                trials.append({
                    "input_hex":    input_hex,
                    "expected_hex": bytes(expected).hex(),
                })
                continue

            try:
                result = runner.rerun(input_bytes, observe_points=[])
                got = result.output
            except Exception as exc:
                trials.append({
                    "input_hex":    input_hex,
                    "runner_error": f"{type(exc).__name__}: {exc}",
                    "match":        False,
                })
                continue

            cmp_len = output_length or min(len(expected), len(got))
            match = bytes(expected)[:cmp_len] == got[:cmp_len]
            trials.append({
                "input_hex":    input_hex,
                "expected_hex": bytes(expected).hex(),
                "got_hex":      got.hex(),
                "match":        match,
            })
            if match:
                n_pass += 1
    finally:
        if runner is not None:
            runner.shutdown()

    if runner_cmd:
        verdict = (
            f"PASS ({n_pass}/{len(trials)})"
            if n_pass == len(trials) and trials
            else f"FAIL ({n_pass}/{len(trials)})"
        )
    else:
        verdict = f"COMPUTED ({len(trials)} trials, no runner)"

    payload = {
        "construction_sha1": construction_sha1,
        "construction_str":  expr,
        "trials":            trials,
        "verdict":           verdict,
    }
    if as_json:
        click.echo(json.dumps(payload, indent=2))
    else:
        click.echo(f"construction: {expr}")
        click.echo(f"sha1:         {construction_sha1}")
        for t in trials:
            line = f"  input {t['input_hex']}: "
            if "construction_error" in t:
                line += f"CONSTRUCTION_ERROR {t['construction_error']}"
            elif "runner_error" in t:
                line += f"RUNNER_ERROR {t['runner_error']}"
            elif runner_cmd:
                tag = "✓" if t.get("match") else "✗"
                line += (
                    f"{tag} expected={t['expected_hex'][:32]}… "
                    f"got={t['got_hex'][:32]}…"
                )
            else:
                line += f"  expected={t['expected_hex'][:48]}…"
            click.echo(line)
        click.echo(f"\nverdict: {verdict}")
