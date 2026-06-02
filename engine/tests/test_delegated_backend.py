"""Round-trip test: DelegatedBackend writes llm_request to its out_stream
and reads llm_response from its in_stream. Mock both with StringIO."""

from __future__ import annotations

import io
import json

from engine.llm_client import DelegatedBackend


def test_delegated_backend_round_trip():
    # Simulate the agent's reply landing on the engine's stdin.
    in_buf = io.StringIO(json.dumps({
        "id": "llm-1", "type": "llm_response",
        "hypotheses": [
            {"kind": "handler_semantic", "subject": "h@0x1234",
             "payload": {"op": "XOR", "dst": "x4", "src": ["x1", "x2"]},
             "confidence": 0.9, "rationale": "x1 ^ x2 == x4 matches trace"},
        ],
    }) + "\n")
    out_buf = io.StringIO()

    be = DelegatedBackend(in_stream=in_buf, out_stream=out_buf)
    hyps = be.generate_hypotheses(
        system_prompt="be a hypothesis maker",
        user_context="instruction is `eor x4, x1, x2`",
        schema={"type": "object"},
        n=5,
    )
    assert len(hyps) == 1
    h = hyps[0]
    assert h.kind == "handler_semantic"
    assert h.subject == "h@0x1234"
    assert h.payload["op"] == "XOR"
    assert h.confidence == 0.9

    # The engine wrote one llm_request to out_buf
    out_buf.seek(0)
    sent = json.loads(out_buf.read().strip())
    assert sent["type"] == "llm_request"
    assert sent["id"] == "llm-1"
    assert sent["n"] == 5
    assert "be a hypothesis maker" in sent["system_prompt"]


def test_delegated_backend_id_mismatch_is_skipped_until_match():
    """If the agent's stdin pipe has unrelated chatter (other tool replies)
    before the llm_response, we should skip past it to find ours."""
    msgs = [
        json.dumps({"id": 99, "result": "unrelated"}),         # other tool reply
        json.dumps({"type": "event", "kind": "stage_done"}),   # an event
        json.dumps({"id": "llm-1", "type": "llm_response",
                    "hypotheses": []}),
    ]
    in_buf = io.StringIO("\n".join(msgs) + "\n")
    out_buf = io.StringIO()
    be = DelegatedBackend(in_stream=in_buf, out_stream=out_buf)
    hyps = be.generate_hypotheses("s", "u", {}, n=1)
    assert hyps == []
