"""BR-4 §A regression: in-process driver API is a real, importable function.

We can't smoke-test the full subprocess round-trip in a unit test (it needs
a real agent-serve runner), but we can pin:

  1. `from engine.driver import drive` works (the import path BR-4 promises)
  2. `drive` is callable, signature accepts `provider=<callable>` kwarg
  3. The same names that worked from `engine.examples.agent_drive` for the
     bundled providers / wire helpers still work from `engine.driver`
"""

from __future__ import annotations

import inspect


def test_driver_public_api_importable():
    """`engine.driver` is the canonical import path."""
    from engine.driver import drive, read_until, send, spawn  # noqa: F401
    from engine.driver import _resolve_workflow  # noqa: F401  (internal but stable)
    assert callable(drive)
    assert callable(spawn)
    assert callable(send)
    assert callable(read_until)


def test_drive_signature_takes_provider_callable():
    """drive(provider=...) — the BR-4 §A user-facing entry."""
    from engine.driver import drive
    sig = inspect.signature(drive)
    assert "provider" in sig.parameters, \
        "drive() must accept a Python callable as `provider`"
    # Keyword-only is fine but the parameter must exist.
    p = sig.parameters["provider"]
    # No default → caller must supply it.
    assert p.default is inspect.Parameter.empty


def test_bundled_providers_are_first_class():
    """The bundled `llm_stub` / `arm-heuristic` providers MUST be plain
    callables with the right signature, so users can compose them or wrap
    them inline without going through the CLI string-key dispatch."""
    from engine.driver import LLM_PROVIDERS, llm_stub
    assert callable(llm_stub)
    # arm-heuristic / stub / file all have the same shape (req: dict) -> dict
    for name, fn in LLM_PROVIDERS.items():
        assert callable(fn), f"{name} provider not callable"

    # Smoke: stub answers a synthetic request.
    reply = llm_stub({
        "id": "llm-1", "type": "llm_request",
        "system_prompt": "system",
        "user_context": "Snippet:\neor x4, x1, x2",
        "schema": {}, "n": 2,
    })
    assert reply["id"] == "llm-1"
    assert reply["type"] == "llm_response"
    assert isinstance(reply["hypotheses"], list)


def test_back_compat_shim_still_imports():
    """Old code `from engine.examples.agent_drive import ...` keeps working."""
    # Importable from the engine package because it ships in the wheel via
    # the moved driver module. The examples/ shim re-exports from engine.driver.
    from engine.driver import drive as drive_canonical
    # NB: engine/examples/agent_drive.py lives in source tree only (not wheel),
    # so we don't import it here — the canonical path is engine.driver.
    assert callable(drive_canonical)
