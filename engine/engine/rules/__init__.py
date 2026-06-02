"""Rule self-evolution subsystem (PLAN §14, DECISIONS D-022..D-025).

LLM hypothesis → repeated verifier pass → LLM-extracted rule draft →
replay admission test → registered plugin → telemetry → demote/revoke.

The plugin's output is still verified per PLAN §1 rule 1. Promotion saves
the LLM call, not the verification step.
"""
