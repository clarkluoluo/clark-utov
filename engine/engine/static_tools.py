"""Whitelisted subprocess wrappers for static analysis tools.

Per PLAN §12.6 / DECISIONS D-017:
  - Static view ("what the code looks like") complements trace view
    ("what happened"). Cross-checked at ambiguous decision points.
  - Strict argv mode — no shell, no interpolation. Safe against injection.
  - Tools missing → not a blocker. Findings derived without static evidence
    get `static_evidence="missing"` so the deliverable can flag the gap.
  - Priority (when multiple available): Binary Ninja MCP > whitelist CLI > trace-only.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# argv[0] only — fully-qualified names, no aliasing.
WHITELIST = frozenset({
    "radare2", "r2",
    "objdump", "readelf", "nm", "strings",
    "llvm-objdump", "llvm-readelf", "llvm-nm",
})


@dataclass(frozen=True)
class StaticToolResult:
    tool: str
    argv: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    available: bool   # False if tool not on PATH


class ToolNotWhitelisted(ValueError):
    pass


def is_available(tool: str) -> bool:
    return shutil.which(tool) is not None


def run_tool(tool: str, args: list[str], cwd: Path | None = None, timeout: float = 30.0) -> StaticToolResult:
    if tool not in WHITELIST:
        raise ToolNotWhitelisted(f"{tool} not in static tool whitelist")
    if not is_available(tool):
        return StaticToolResult(tool, (tool, *args), -1, "", f"{tool} not on PATH", available=False)
    proc = subprocess.run(
        [tool, *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return StaticToolResult(
        tool=tool,
        argv=(tool, *args),
        exit_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        available=True,
    )
