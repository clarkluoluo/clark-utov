"""Driver layer — only the orchestrators live here.

Both script_mode and agent_mode are thin layers over engine.core.Core.
Either may be running at a time, never both. Mode is recorded in meta.json
so the other can pick up — see DECISIONS D-021 mode-switch protocol.
"""
