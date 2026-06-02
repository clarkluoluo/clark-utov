"""LLM hypothesis generator with pluggable backends.

Two operating modes (PLAN §13 + §15):

  DirectBackend     — engine maintains its own DeepSeek / MiMo key and calls
                      the model directly via OpenAI-compatible HTTP. Every
                      call charges into CostMeter; BudgetExceeded propagates.

  DelegatedBackend  — engine asks the driving AGENT (over NDJSON stdio) to
                      answer the prompt with its own LLM context. The agent
                      pays nothing on our books; CostMeter records 0. The
                      driving agent is expected to honor the same schema as
                      DirectBackend's return.

DECISIONS D-006 (model defaults), D-020 (LLM unified abstraction), D-029
(delegated backend new).
"""

from __future__ import annotations

import abc
import json
import os
from dataclasses import dataclass
from typing import IO, Any

from dotenv import load_dotenv

from .cost import CostMeter

load_dotenv()


@dataclass(frozen=True)
class Hypothesis:
    kind: str
    subject: str
    payload: dict[str, Any]
    confidence: float
    rationale: str


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class LLMBackend(abc.ABC):
    """One LLM call site, regardless of whether we call DeepSeek or ask the
    driving agent. Same in / same out — the orchestrator's S6 path doesn't
    know which one is wired."""

    @abc.abstractmethod
    def generate_hypotheses(
        self,
        system_prompt: str,
        user_context: str,
        schema: dict[str, Any],
        n: int,
    ) -> list[Hypothesis]:
        ...


class NullBackend(LLMBackend):
    """0526Plan D2: a no-op LLM backend. Returns no hypotheses on every
    call so downstream code keeps working without an API key. Selected
    by `--llm-backend none` or env `LLM_BACKEND=none` (the new default).
    """

    def generate_hypotheses(
        self,
        system_prompt: str,
        user_context: str,
        schema: dict[str, Any],
        n: int,
    ) -> list[Any]:
        return []


class DirectBackend(LLMBackend):
    """OpenAI-compatible HTTP backend (DeepSeek / MiMo)."""

    def __init__(self, backend: str | None = None, model: str | None = None,
                 meter: CostMeter | None = None):
        # Import locally so unit tests without openai installed still work
        # against DelegatedBackend.
        from openai import OpenAI
        backend = backend or os.environ.get("LLM_BACKEND", "deepseek")
        # CLI plumbs explicit --llm-model via UTOV_LLM_MODEL so we don't have to
        # thread the flag through every LLMClient() construction site.
        env_model = os.environ.get("UTOV_LLM_MODEL")
        if backend == "deepseek":
            self.client = OpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com",
            )
            self.model = model or env_model or "deepseek-chat"
        elif backend == "mimo":
            self.client = OpenAI(
                api_key=os.environ.get("MIMO_API_KEY", "EMPTY"),
                base_url=os.environ.get("MIMO_BASE_URL", "http://localhost:8000/v1"),
            )
            self.model = model or env_model or "mimo-7b-rl"
        else:
            raise ValueError(f"Unknown LLM_BACKEND: {backend}")
        self.backend = backend
        self.meter = meter

    def generate_hypotheses(
        self,
        system_prompt: str,
        user_context: str,
        schema: dict[str, Any],
        n: int,
    ) -> list[Hypothesis]:
        wrapped_schema = {
            "type": "object",
            "properties": {
                "hypotheses": {
                    "type": "array",
                    "minItems": 1, "maxItems": n,
                    "items": schema,
                }
            },
            "required": ["hypotheses"],
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",
             "content": user_context
                        + f"\n\nReturn at most {n} candidates as JSON matching this schema:\n"
                        + json.dumps(wrapped_schema)},
        ]
        for attempt in range(2):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0 if attempt == 0 else 0.7,
            )
            text = resp.choices[0].message.content or ""
            if self.meter is not None and resp.usage is not None:
                self.meter.charge(
                    model=self.model,
                    input_tokens=resp.usage.prompt_tokens or 0,
                    output_tokens=resp.usage.completion_tokens or 0,
                )
            try:
                obj = json.loads(text)
                items = obj.get("hypotheses", [])
                return [_to_hyp(h) for h in items]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        return []


class DelegatedBackend(LLMBackend):
    """Sends an `llm_request` NDJSON message to the driving agent on `out_stream`,
    blocks reading `in_stream` for the matching `llm_response`. Same in/out as
    DirectBackend — orchestrator code doesn't change.

    Wire protocol (per contracts/agent_protocol.md):
        engine → agent:  {"id": "llm-N", "type": "llm_request",
                          "system_prompt": "...", "user_context": "...",
                          "schema": {...}, "n": 5}
        agent  → engine: {"id": "llm-N", "type": "llm_response",
                          "hypotheses": [{...}, {...}, ...]}

    Errors:
        - agent sends {"id": "...", "type": "llm_error", "message": "..."} → empty list
        - timeout / mismatched id → empty list (logged to errlog)
    """

    def __init__(self, in_stream: IO[str], out_stream: IO[str],
                 model_label: str = "delegated-agent"):
        self.in_stream = in_stream
        self.out_stream = out_stream
        self.model_label = model_label   # for CostMeter / accounting (0 cost)
        self._req_seq = 0

    def generate_hypotheses(
        self,
        system_prompt: str,
        user_context: str,
        schema: dict[str, Any],
        n: int,
    ) -> list[Hypothesis]:
        self._req_seq += 1
        req_id = f"llm-{self._req_seq}"
        req = {
            "id": req_id, "type": "llm_request",
            "system_prompt": system_prompt,
            "user_context":  user_context,
            "schema":        schema,
            "n":             n,
        }
        self.out_stream.write(json.dumps(req) + "\n")
        self.out_stream.flush()

        # Read until we find the matching response. Skip messages that aren't
        # llm_response or have a different id (might be unrelated tool calls
        # the agent sent while we were waiting — return them via a queue if
        # we add multiplexing later. For now, drop with a stderr breadcrumb).
        for raw in self.in_stream:
            raw = raw.strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "llm_error" and msg.get("id") == req_id:
                return []
            if msg.get("type") != "llm_response" or msg.get("id") != req_id:
                # Push back / queue mechanism would go here. Drop for now.
                continue
            items = msg.get("hypotheses") or []
            return [_to_hyp(h) for h in items]
        return []  # EOF


def _to_hyp(h: dict) -> Hypothesis:
    return Hypothesis(
        kind=h["kind"],
        subject=h["subject"],
        payload=h.get("payload", {}),
        confidence=float(h.get("confidence", 0.5)),
        rationale=h.get("rationale", ""),
    )


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class LLMClient:
    """Stable facade for callers (s6 / blue_team / rules.promotion). Holds a
    pluggable `backend` and a `meter`. Switching backends at runtime is safe.

    Default behavior preserves the pre-D-029 API: building LLMClient() with no
    args creates a DirectBackend with env-selected credentials.
    """

    def __init__(self, backend: LLMBackend | None = None,
                 *, env_backend: str | None = None, model: str | None = None):
        self._meter: CostMeter | None = None
        if backend is not None:
            self.backend: LLMBackend = backend
        else:
            # 0526Plan D2: when the resolved backend is "none" use NullBackend
            # so downstream code keeps working without an API key. This is
            # the path callers hit when they construct LLMClient() with no
            # args and don't have LLM_BACKEND set.
            resolved = env_backend or os.environ.get("LLM_BACKEND", "none")
            if resolved == "none":
                self.backend = NullBackend()
            else:
                self.backend = DirectBackend(backend=resolved, model=model)
        # For DirectBackend forwarding of charged tokens, we re-attach meter
        # whenever attach_meter is called.

    def attach_meter(self, meter: CostMeter) -> None:
        self._meter = meter
        if isinstance(self.backend, DirectBackend):
            self.backend.meter = meter
        # DelegatedBackend doesn't bill — we leave meter unused there.

    def set_backend(self, backend: LLMBackend) -> None:
        self.backend = backend
        if isinstance(self.backend, DirectBackend) and self._meter is not None:
            self.backend.meter = self._meter

    @property
    def model(self) -> str:
        if isinstance(self.backend, DirectBackend):
            return self.backend.model
        return getattr(self.backend, "model_label", "unknown")

    @property
    def client(self):  # noqa: D401 — kept for compatibility w/ blue_team raw call
        """Underlying OpenAI client when DirectBackend, else None."""
        if isinstance(self.backend, DirectBackend):
            return self.backend.client
        return None

    def generate_hypotheses(
        self,
        system_prompt: str,
        user_context: str,
        schema: dict[str, Any],
        n: int = 5,
    ) -> list[Hypothesis]:
        return self.backend.generate_hypotheses(system_prompt, user_context, schema, n)
