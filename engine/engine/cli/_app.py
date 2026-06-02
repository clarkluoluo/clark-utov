"""Shared CLI app object (the click group) and cross-command helpers."""
from __future__ import annotations

import json
import os
from pathlib import Path

import click

from ..related_helpers import format_related_line


@click.group()
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="After a command's output, surface discoverability hints "
                   "(ℹ related: ...). Silent by default; changes no command "
                   "output, only appends hints.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """clark-utov engine CLI."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


def _verbose_enabled() -> bool:
    """Whether discoverability hints should be surfaced.

    True when the group ``--verbose``/``-v`` flag is set, or the ``UTOV_DEBUG``
    env var is truthy (debug-mode parity with the API surfacing). Reads the
    *current* click context so individual commands need not thread it through.
    """
    if os.environ.get("UTOV_DEBUG", "").strip() not in ("", "0", "false", "False"):
        return True
    ctx = click.get_current_context(silent=True)
    while ctx is not None:
        obj = ctx.obj
        if isinstance(obj, dict) and obj.get("verbose"):
            return True
        ctx = ctx.parent
    return False


def emit_related(name: str) -> None:
    """Print the ``ℹ related: X, Y, Z`` hint for ``name`` — verbose mode only.

    Call this AFTER a command's own output. Silent by default (spec A8 §3):
    nothing is printed unless verbose/debug is on. An entry point with no
    relations prints nothing (degenerate ⇒ silence, never a fake line).
    Written to stderr so it never contaminates a command's stdout payload.
    """
    if not _verbose_enabled():
        return
    line = format_related_line(name)
    if line is not None:
        click.echo(line, err=True)


def _parse_runner_cmd(cmd_str: str) -> list[str]:
    """Parse a user-supplied shell-style runner command into argv."""
    import shlex
    argv = shlex.split(cmd_str)
    if not argv:
        raise click.UsageError("--runner-cmd is empty")
    return argv


def _format_pipeline_summary(summaries: list[dict]) -> str:
    out = []
    for s in summaries:
        out.append(f"  {s.get('stage','?'):4}  " + ", ".join(
            f"{k}={v}" for k, v in s.items() if k != "stage" and not isinstance(v, str) or k in ("out",)
        ))
    return "\n".join(out)


def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _resolve_run_dir(work_dir: Path) -> Path:
    work_dir = work_dir.resolve()
    if (work_dir / "latest").is_symlink():
        work_dir = (work_dir / "latest").resolve()
    return work_dir
