"""S6: LLM hypothesis generation, each candidate immediately verified.

Wires together:
  - llm_client.LLMClient (DeepSeek default; MiMo fallback)
  - hyp_tree.HypTree (the persistent backtracking ledger)
  - verifier.Verifier (PLAN §1.1 — only source of truth)
  - discipline.wrap_messages (PLAN §12.3 — anti-drift)

Flow (PLAN §1.3 + DECISIONS D-005):
  1. Caller picks a "stuck point" — e.g. a sliced expression S5 can't reduce.
  2. We extract a small clean context (regs, expression, surrounding mnemonics).
  3. LLM produces N candidates (N=5 default; auto-shrink to 3 at depth ≥ 3).
  4. Each candidate inserted as a child HypNode under the parent.
  5. Each is immediately verified by `Verifier.check_handler_semantic` (the
     default strategy; caller may override).
  6. On PASS: status → 'passed'; on FAIL: status → 'failed'; on INCONCLUSIVE
     left 'pending' to be picked up later or by the auto-extend protocol.

This stage does NOT itself decide which next candidate to try; the
orchestrator (script or agent) drives that loop using hyp_tree.next_pending_sibling
and the verifier verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..discipline import DisciplineState, wrap_messages
from ..hyp_tree import HypTree
from ..llm_client import LLMClient
from ..verifier import Verifier

CODE_VERSION = "s6-v1"

DEFAULT_N = 5
DEPTH_SHRINK_AT = 3
SHRUNK_N = 3

_HYP_SCHEMA = {
    "type": "object",
    "properties": {
        "kind":       {"type": "string"},
        "subject":    {"type": "string"},
        "payload":    {"type": "object"},
        "confidence": {"type": "number"},
        "rationale":  {"type": "string"},
    },
    "required": ["kind", "subject", "payload", "confidence"],
}

_SYSTEM_PROMPT = (
    "You are a deterministic-pattern recognizer for reverse-engineered "
    "ARM64 trace fragments. You only NAME and EXPLAIN — never invent. "
    "Each hypothesis must be expressible as a concrete predicate the "
    "verifier can mechanically check. If you're unsure, give a low "
    "confidence — do not pad output.\n\n"
    "VERIFIER SHAPES (BR-4 §D) — your `payload` MUST fit one of these, "
    "otherwise the verifier will mark it INCONCLUSIVE and waste your work:\n"
    "  1. 2-reg binop:   {op: XOR|EOR|AND|OR|ORR|ADD|SUB|MUL|LSL|LSR|ASR|ROR|"
    "BIC|EON|ORN, dst: <reg>, src: [<reg>, <reg>]}\n"
    "  2. reg-imm binop: {op: <same>, dst: <reg>, src: [<reg>], imm: '0x...'}\n"
    "  3. shifted/extended-reg: above + {src2_ext: {kind: lsl|lsr|asr|ror|"
    "sxtw|sxth|sxtb|uxtw|uxth|uxtb, amount: <int>}}\n"
    "  4. unary:         {op: MOV|MVN|NEG|SXTW|UXTW|REV|CLZ, dst: <reg>, src: [<reg>]}\n"
    "  5. ternary (SHA-2 round funcs): {op: CH|MAJ|PARITY, dst: <reg>, "
    "src: [<reg>, <reg>, <reg>]}\n"
    "DO NOT propose memory loads, sp-relative arithmetic, multi-imm forms "
    "(movk shift chains), or floating-point — verifier has no model for them. "
    "If the snippet doesn't fit any shape above, return an empty list rather "
    "than guessing."
)


@dataclass
class StuckContext:
    """What the orchestrator hands to S6 when it can't make progress on its own."""
    parent_hyp_id: int | None
    kind_hint: str                # e.g. "handler_semantic", "algo_signature"
    summary: str                  # human-readable description of the stuck point
    snippet: str                  # the offending instruction window / expression text
    expected_output: dict[str, Any] | None = None    # for handler_semantic checks
    instr_idx: int | None = None  # trace index of the stuck instruction (if applicable);
                                  # script_mode uses this to fetch regs_read/regs_write
                                  # for verifier input/output state


def _build_prompts(ctx: StuckContext, discipline: DisciplineState, n: int
                   ) -> tuple[str, str]:
    """Returns (system_prompt, user_context) for one stuck point."""
    user_msg = (
        f"Stuck point: {ctx.summary}\n"
        f"Kind hint:    {ctx.kind_hint}\n"
        f"Snippet:\n{ctx.snippet}\n"
        f"Propose at most {n} hypotheses, each MUST be a concrete claim the "
        f"verifier can mechanically check."
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    messages = wrap_messages(discipline, messages)
    sys_text = "\n".join(m["content"] for m in messages if m["role"] == "system")
    usr_text = "\n".join(m["content"] for m in messages if m["role"] == "user")
    return sys_text, usr_text


def generate_hypotheses_only(
    ctx: StuckContext,
    tree: HypTree,
    llm: LLMClient,
    discipline: DisciplineState,
    n: int | None = None,
) -> tuple[int, list]:
    """BR-4 §C: the LLM-call half of propose_and_verify, extracted so the
    orchestrator can call it concurrently across stuck points and serialize
    only the tree/promote/verify side. Returns `(n_used, hypotheses)`."""
    depth = 0
    if ctx.parent_hyp_id is not None:
        depth = tree.get(ctx.parent_hyp_id).depth + 1
    if n is None:
        n = SHRUNK_N if depth >= DEPTH_SHRINK_AT else DEFAULT_N
    sys_text, usr_text = _build_prompts(ctx, discipline, n)
    hyps = llm.generate_hypotheses(sys_text, usr_text, _HYP_SCHEMA, n=n)
    return n, hyps


def propose_and_verify(
    ctx: StuckContext,
    tree: HypTree,
    llm: LLMClient,
    verifier: Verifier,
    discipline: DisciplineState,
    input_state: dict[str, int] | None = None,
    expected_output_state: dict[str, int] | None = None,
    n: int | None = None,
) -> list[dict[str, Any]]:
    """Generate hypotheses and verify each. Returns list of verdict records.

    Composition of `generate_hypotheses_only` (parallelizable LLM call) and
    `ingest_hypotheses_and_verify` (must stay serial). Callers wanting
    concurrency should call the two halves directly — see BR-4 §C.
    """
    _n_used, hyps = generate_hypotheses_only(ctx, tree, llm, discipline, n=n)
    return ingest_hypotheses_and_verify(
        ctx, hyps, tree, verifier,
        input_state=input_state,
        expected_output_state=expected_output_state,
    )


def ingest_hypotheses_and_verify(
    ctx: StuckContext,
    hyps: list,
    tree: HypTree,
    verifier: Verifier,
    input_state: dict[str, int] | None = None,
    expected_output_state: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """BR-4 §C: the serial half of propose_and_verify — given an
    already-fetched `hyps` list (from generate_hypotheses_only), insert each
    into the tree and verify. Tree mutations stay single-threaded; this is
    the side concurrency callers must NOT parallelize."""
    verdicts: list[dict[str, Any]] = []
    for h in hyps:
        hyp_id = tree.add(
            parent_id=ctx.parent_hyp_id,
            kind=h.kind, subject=h.subject,
            payload={**h.payload, "rationale": h.rationale},
            confidence=h.confidence,
            source="llm",
            tags=[("source", "llm"), ("verdict_bucket",
                   "strong" if h.confidence >= 0.8 else "medium" if h.confidence >= 0.5 else "weak")],
            created_in_stage="s6",
        )
        # Verify if we have enough info; otherwise leave 'pending'.
        if input_state is not None and expected_output_state is not None:
            res = verifier.check_handler_semantic(input_state, h.payload, expected_output_state)
            tree.mark_verdict(hyp_id, res.verdict.value, res.detail)
            verdicts.append({"hyp_id": hyp_id, "verdict": res.verdict.value, "detail": res.detail})
        else:
            verdicts.append({"hyp_id": hyp_id, "verdict": "pending",
                             "detail": {"reason": "no concrete state supplied to verify"}})
    return verdicts


def run(ctx) -> dict:
    """Stage entry. ctx keys:
        work: WorkDir
        stuck_context: StuckContext or dict (auto-converted)
        verifier: Verifier
        llm: LLMClient (optional — defaults to LLMClient())
        discipline: DisciplineState (optional — auto-create per work)
        input_state, expected_output_state: optional for immediate verify
    """
    from ..store import open_hypotheses_db

    work = ctx["work"]
    stuck_raw = ctx["stuck_context"]
    # Accept dict form too — agent-mode JSON-RPC can't pass dataclasses.
    if isinstance(stuck_raw, dict):
        stuck = StuckContext(
            parent_hyp_id=stuck_raw.get("parent_hyp_id"),
            kind_hint=stuck_raw["kind_hint"],
            summary=stuck_raw["summary"],
            snippet=stuck_raw["snippet"],
            expected_output=stuck_raw.get("expected_output"),
            instr_idx=stuck_raw.get("instr_idx"),
        )
    else:
        stuck = stuck_raw
    verifier = ctx["verifier"]
    llm = ctx.get("llm") or LLMClient()
    discipline = ctx.get("discipline") or DisciplineState(
        target=work.target_dir.name, run_id=work.run_id,
    )
    conn = open_hypotheses_db(work)
    tree = HypTree(conn)

    verdicts = propose_and_verify(
        stuck, tree, llm, verifier, discipline,
        input_state=ctx.get("input_state"),
        expected_output_state=ctx.get("expected_output_state"),
        n=ctx.get("n"),
    )
    conn.close()

    return {
        "stage": "s6",
        "candidates":   len(verdicts),
        "passed":       sum(1 for v in verdicts if v["verdict"] == "pass"),
        "failed":       sum(1 for v in verdicts if v["verdict"] == "fail"),
        "pending":      sum(1 for v in verdicts if v["verdict"] == "pending"),
        "inconclusive": sum(1 for v in verdicts if v["verdict"] == "inconclusive"),
        # Verdict list with hyp_ids so the agent orchestrator can promote_to_finding
        # the ones that pass (BR-2 §2 — script_mode does this in-process; agent_mode
        # only had aggregate counts and couldn't reproduce the promote step).
        "verdicts":     verdicts,
    }
