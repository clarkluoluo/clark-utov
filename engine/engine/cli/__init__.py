"""Top-level CLI: `utov ...`.

Subcommands:
  trace-info <path>           — sniff format, count instructions, hot PCs
  pipeline --runner-cmd '<shell command>' --input HEX [--work-root DIR]
                              — Live mode: spawn runner subprocess that speaks
                                contracts/agent_protocol.md NDJSON, run S1..S5
  pipeline-file --trace PATH --target-name NAME --entry HEX --exit HEX
                              — File mode: take a static trace, run S1..S5

The engine does NOT know what your runner is written in (Java / Python /
Rust / Go / Frida / qiling / unidbg). You pass a shell command string; we
spawn it and talk NDJSON over its stdio. For an example, see
example/runner-sha256/README.md (Java + unidbg sample runner).
"""

from ._app import main
# Importing the command modules registers their subcommands onto ``main``.
from . import _diagnostics, _pipeline, _query  # noqa: F401,E402

__all__ = ["main"]
