#!/usr/bin/env python3
"""Backward-compat shim — the actual driver moved to `engine.driver`.

BR-4 §A: in-process Python provider is a first-class implementation of the
agent protocol, so the driver belongs in the engine package proper (where it
ships in the wheel), not under `examples/` (which doesn't). Old callers that
do `python3 -m engine.examples.agent_drive ...` or
`from engine.examples.agent_drive import spawn` keep working through this
file; new code should `from engine.driver import drive, spawn, send, read_until`.
"""

from __future__ import annotations

import sys

from engine.driver import (  # noqa: F401  re-exports for back-compat
    LLM_PROVIDERS,
    _load_dotted_provider,
    drive,
    llm_file_handoff,
    llm_stub,
    main,
    read_until,
    send,
    spawn,
)

if __name__ == "__main__":
    sys.exit(main())
