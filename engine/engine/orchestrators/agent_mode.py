"""Mode 2: bidirectional NDJSON tool server for agents (PLAN §13 + D-029).

Two message directions on stdio:

  Agent → Engine   (existing, unchanged):
    {"id": int, "method": "<name>", "params": {...}?}
    →
    {"id": int, "result": {...}}  | {"id": int, "error": {...}}

  Engine → Agent   (NEW — delegated LLM):
    {"id": "llm-N", "type": "llm_request", "system_prompt": "...",
     "user_context": "...", "schema": {...}, "n": 5}
    Agent must answer with:
    {"id": "llm-N", "type": "llm_response", "hypotheses": [{...}, ...]}

  Engine → Agent   (NEW — observability events; one-way, fire-and-forget):
    {"type": "event", "kind": "stage_done", "detail": {...}}
    {"type": "event", "kind": "ask_user.budget_overrun", "detail": {...}}
    {"type": "event", "kind": "safe_interrupt_point", "detail": {...}}

The agent receives request-and-event mixed on stdout; it should switch on the
`type` field. When the agent is waiting for a tool-call response, it should
also be ready to answer `llm_request` messages (otherwise the engine deadlocks
inside its outer call).

contracts/agent_protocol.md spells this out for external implementers.
"""

from __future__ import annotations

import json
import sys
import threading
import traceback
from queue import Empty, Queue
from typing import IO, Any

from ..core import Core
from ..discipline_wrapper import DisciplineRaise, DisciplineWrapper
from ..llm_client import DelegatedBackend, LLMClient
from ..methodology import MethodologyConfig
from ..progress import EventKind, Tracker
from ..static_tools import WHITELIST, run_tool


def serve_mcp(core: Core, *, stdin=None, stdout=None, stderr=None,
              llm: LLMClient | None = None,
              tracker: Tracker | None = None,
              discipline: DisciplineWrapper | None = None,
              methodology_config: MethodologyConfig | None = None) -> None:
    """Run the bidirectional NDJSON loop bound to `core`.

    If `llm` is provided, we replace its backend with a DelegatedBackend
    pointed at the same stdio so engine→agent LLM questions ride along.

    ``discipline`` (or ``methodology_config``) controls the
    methodology-reinforcement / anti-drift wrapper that attaches a
    footer + periodic card + context prompts + alerts to every JSON-RPC
    envelope. ``UTOV_METHODOLOGY=off`` in the environment disables it.
    """
    stdin  = stdin  or sys.stdin
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr

    if discipline is None:
        discipline = DisciplineWrapper(
            core=core,
            config=methodology_config or MethodologyConfig.from_env(),
        )

    # Demultiplexer: a single reader thread parses every stdin line and
    # routes it: either a tool request (becomes a `_RpcMessage`) or an
    # `llm_response` (goes to the llm-waiter queue), or a free-form ack.
    rpc_q: "Queue[dict[str, Any] | None]" = Queue()
    llm_q: "Queue[dict[str, Any]]"        = Queue()

    def _reader():
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("type") in ("llm_response", "llm_error"):
                llm_q.put(msg)
            else:
                rpc_q.put(msg)
        rpc_q.put(None)   # EOF sentinel
    threading.Thread(target=_reader, daemon=True).start()

    # Wire delegated LLM backend to use the SAME stdio (using llm_q for input).
    # Stash the configured LLM on `core` so s6_* dispatchers can route through
    # the same delegated backend — otherwise `s6_hypothesis.run()` falls back
    # to `LLMClient()` which tries to read DEEPSEEK_API_KEY at construction.
    if llm is not None:
        llm.set_backend(_QueueDelegatedBackend(stdout=stdout, llm_q=llm_q))
        core._llm = llm  # type: ignore[attr-defined]

    # Optionally emit Tracker events back to the agent.
    if tracker is not None:
        def _emit_event(evt):
            stdout.write(json.dumps({
                "type": "event", "kind": evt.kind.value,
                "timestamp": evt.timestamp, "detail": evt.detail,
            }) + "\n")
            stdout.flush()
        tracker.on_any(_emit_event)

    print("engine ready", file=stderr, flush=True)

    while True:
        req = rpc_q.get()
        if req is None:
            return  # stdin closed
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params") or {}
        try:
            if method == "shutdown":
                _write_ok(stdout, req_id, "ok")
                return
            result, env = discipline.step(
                method, params,
                lambda m, p: _dispatch(core, m, p),
            )
            if env.intercepted:
                _write_err(
                    stdout, req_id, -32001,
                    env.intercepted_reason or "discipline intercept",
                    data={"methodology": env.to_dict()},
                )
                continue
            _write_ok(stdout, req_id, result,
                      methodology=env.to_dict())
        except DisciplineRaise as dr:
            # Dispatch raised; the envelope was already built so the
            # agent gets methodology context with the error.
            _write_err(
                stdout, req_id, -32000, "dispatch failed during discipline step",
                data={"traceback": traceback.format_exc(),
                      "methodology": dr.envelope.to_dict()},
            )
        except Exception as e:
            # BR-2 §6: include traceback so the driving agent can locate the
            # actual failure site instead of grepping the source by exception
            # text. JSON-RPC `error.data` is the documented carrier.
            _write_err(stdout, req_id, -32000, f"{type(e).__name__}: {e}",
                       data={"traceback": traceback.format_exc()})


class _QueueDelegatedBackend(DelegatedBackend):
    """Variant of DelegatedBackend whose `in_stream` is a Queue fed by the
    serve_mcp reader thread (otherwise the reader and the backend would race
    on stdin)."""

    def __init__(self, stdout: IO[str], llm_q: "Queue[dict[str, Any]]"):
        # Skip parent __init__ for stream args; provide stub stream values.
        self.in_stream  = None   # type: ignore[assignment]
        self.out_stream = stdout
        self.model_label = "delegated-agent"
        self._req_seq = 0
        self._q = llm_q

    def generate_hypotheses(self, system_prompt, user_context, schema, n):
        self._req_seq += 1
        req_id = f"llm-{self._req_seq}"
        req = {
            "id": req_id, "type": "llm_request",
            "system_prompt": system_prompt, "user_context": user_context,
            "schema": schema, "n": n,
        }
        self.out_stream.write(json.dumps(req) + "\n")
        self.out_stream.flush()

        # Wait for a matching response. We use a long timeout because the
        # agent's own LLM may be slow.
        while True:
            try:
                msg = self._q.get(timeout=300.0)
            except Empty:
                return []
            if msg.get("id") != req_id:
                continue
            if msg.get("type") == "llm_error":
                return []
            items = msg.get("hypotheses") or []
            from ..llm_client import _to_hyp
            return [_to_hyp(h) for h in items]


# --- dispatch (unchanged from previous version) ---


def _dispatch(core: Core, method: str, params: dict[str, Any]) -> Any:
    if method == "metadata":
        m = core.config.target_meta
        return {
            "target_name":      m.target_name,
            "arch":             m.arch,
            "algo_entry_pc":    f"0x{m.algo_entry_pc:x}",
            "algo_exit_pc":     f"0x{m.algo_exit_pc:x}",
            "input_length":     m.input_length,
            "output_length":    m.output_length,
            "algo_symbol":      m.algo_symbol,
            "emulator_name":    m.emulator_name,
            "emulator_version": m.emulator_version,
        }
    if method == "list_stages":
        from ..core import _STAGES
        return sorted(_STAGES.keys())
    if method == "run_stage":
        name = params["name"]
        return core.run_stage(name, **{k: v for k, v in params.items() if k != "name"})
    if method == "run_pipeline":
        return core.run_pipeline(params.get("stages"))
    if method == "get_hypotheses":
        hyps = core.get_hypotheses(
            status=params.get("status"),
            kind=params.get("kind"),
            source=params.get("source"),
        )
        return [_hyp_to_dict(h) for h in hyps]
    if method == "submit_hypothesis":
        hid = core.submit_hypothesis(
            kind=params["kind"],
            subject=params["subject"],
            payload=params.get("payload", {}),
            confidence=params.get("confidence"),
            parent_id=params.get("parent_id"),
            source=params.get("source", "agent"),
        )
        return {"hyp_id": hid}
    if method == "promote_to_finding":
        fid = core.promote_to_finding(
            int(params["hyp_id"]),
            verifier_strategy=params.get("verifier_strategy", "manual"),
            stage=params.get("stage", "agent"),
        )
        return {"finding_id": fid}
    if method == "verify_plugin_findings":
        # Mechanical pass: run Verifier.check_fingerprint on every pending
        # plugin algo_signature hyp; promote those whose magic value really
        # appears in the trace. No LLM, no budget. Idempotent — re-running
        # after a partial promotion only touches still-pending hyps.
        return core.verify_and_promote_plugin_findings()
    if method == "verify_handler_binops":
        # BR-4 §1: deterministic handler-semantic pass — walk trace for
        # reg-reg-reg ARM binops and auto-promote each PASS. Symmetric to
        # verify_plugin_findings; takes no params; idempotent (dedupes on PC).
        return core.verify_and_promote_handler_binops()
    if method == "verify_handler_unaries":
        # 0526Plan C5.4/5: layer-0 unary discoverer (MOV/MVN/NEG/SXTW/UXTW/REV/CLZ).
        return core.verify_and_promote_handler_unaries()
    if method == "verify_handler_imm_binops":
        # 0526Plan C5.1/2/3: layer-0 reg-imm binop discoverer.
        return core.verify_and_promote_handler_imm_binops()
    if method == "verify_handler_extended_binops":
        # 0526Plan C5.7: layer-0 shifted/extended-register binop discoverer.
        return core.verify_and_promote_handler_extended_binops()
    if method == "verify_handler_bfx":
        # 0526Plan C5.6: layer-0 bit-field-extract (ubfx / sbfx) discoverer.
        return core.verify_and_promote_handler_bfx()
    if method == "verify_handler_ch_idioms":
        # 0526Plan C3: SHA-2 Ch(x,y,z) 3-insn idiom discoverer.
        return core.verify_and_promote_handler_ch_idioms()
    if method == "verify_handler_maj_idioms":
        # BR-8 #2: SHA-2 Maj(a,b,c) 3-insn idiom discoverer.
        return core.verify_and_promote_handler_maj_idioms()
    if method == "self_rescan_missing_anchors":
        # BR-8 #3: re-run σ/Σ + Ch + Maj when an algorithm_identified
        # finding has missing anchors, then recompute the fit.
        return core.self_rescan_missing_anchors()
    if method == "emit_pseudocode":
        # FEATURE-REQUEST-1: render Tier 1 pseudocode for this run.
        # Returns the rendered string; agent decides whether to print it,
        # store it, or pass it on.
        fmt = params.get("format", "text")
        try:
            return {"text": core.emit_pseudocode(fmt=fmt)}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
    if method == "dataflow_query":
        # BR-8 #4: agent-facing trace query helpers (rotations_on_input,
        # xor_chain_to, producer_chain, boolean_subgraph). Avoids the
        # agent having to grep s3_dfg.jsonl by hand.
        pc_range = params.get("within_pc_range")
        if pc_range is not None and isinstance(pc_range, list):
            pc_range = (
                int(pc_range[0], 16) if isinstance(pc_range[0], str)
                else int(pc_range[0]),
                int(pc_range[1], 16) if isinstance(pc_range[1], str)
                else int(pc_range[1]),
            )
        from_pc = params.get("from_pc")
        if isinstance(from_pc, str):
            from_pc = int(from_pc, 16)
        return core.dataflow_query(
            kind=params["kind"],
            input_reg=params.get("input_reg"),
            within_pc_range=pc_range,
            target_reg=params.get("target_reg"),
            dst_reg=params.get("dst_reg"),
            from_pc=from_pc,
            max_depth=int(params.get("max_depth", 8)),
            max_results=int(params.get("max_results", 256)),
        )
    if method == "verify_triton_simplifications":
        # 0526Plan C1: Triton symbolic-execution verifier path.
        return core.verify_and_promote_triton_simplifications()
    if method == "verify_sigma_idioms":
        # 0526Plan C4.1/2: SHA-2 σ/Σ layer-1 fold-idiom discoverer.
        return core.verify_and_promote_sigma_idioms()
    if method == "verify_algorithm_templates":
        # 0526Plan E1.4/5: layer-2 algorithm template fit.
        return core.verify_and_promote_algorithm_templates()
    if method == "preprocess_batch":
        # 0527: one-call deterministic batch. Optional `passes` filter.
        # Returns {batch_id, ran, results, totals, next_step_hints}.
        passes = params.get("passes")
        return core.preprocess_batch(passes=passes)
    if method == "discard_batch":
        # 0527: bulk-fail the hypotheses behind a preprocess batch via
        # override_verdict("fail"). Optional sources / kinds filters.
        return core.discard_batch(
            params["batch_id"],
            sources=params.get("sources"),
            kinds=params.get("kinds"),
            reason=params.get("reason", "agent discarded preprocess batch"),
            actor=params.get("actor", "agent"),
        )
    if method == "get_findings":
        # 0526Plan B1: filtered findings.sqlite query — symmetric to
        # get_hypotheses. Optional filters: source / stage / kind /
        # subject_like / limit.
        return core.get_findings(
            source=params.get("source"),
            stage=params.get("stage"),
            kind=params.get("kind"),
            subject_like=params.get("subject_like"),
            limit=params.get("limit"),
        )
    if method == "stuck_statistics":
        # 0526Plan B2: group S5 stuck points by mnemonic /
        # verifiable_shape / pc_cluster.
        return core.stuck_statistics(
            max_points=params.get("max_points"),
            cluster_gap=int(params.get("cluster_gap", 0x40)),
        )
    if method == "read_trace_window":
        items = core.read_trace_window(int(params["idx_from"]), int(params["idx_to"]))
        return [{
            "idx": i.idx, "pc": f"0x{i.pc:x}",
            "mnemonic": i.mnemonic,
            "regs_read":  {k: f"0x{v:x}" for k, v in i.regs_read.items()},
            "regs_write": {k: f"0x{v:x}" for k, v in i.regs_write.items()},
        } for i in items]
    if method == "static_tool":
        tool = params["tool"]
        args = params.get("args", [])
        if tool not in WHITELIST:
            raise ValueError(f"{tool!r} not in static-tool whitelist")
        r = run_tool(tool, list(args))
        return {"tool": r.tool, "exit_code": r.exit_code,
                "stdout": r.stdout, "stderr": r.stderr,
                "available": r.available}
    if method == "is_safe_to_interrupt":
        return {"safe": core.is_safe_to_interrupt()}

    # --- S6 LLM hypothesis methods (agent doesn't go via run_stage("s6") since
    #     run_stage can't carry a StuckContext from JSON-RPC) ---
    if method == "s6_find_stuck_points":
        from .script_mode import _find_stuck_points
        cap = params.get("max_points")
        stuck = _find_stuck_points(core, max_points=cap)
        return [{
            "parent_hyp_id":   s.parent_hyp_id,
            "kind_hint":       s.kind_hint,
            "summary":         s.summary,
            "snippet":         s.snippet,
            "expected_output": s.expected_output,
            "instr_idx":       s.instr_idx,
        } for s in stuck]
    if method == "s6_propose_and_verify":
        # Process ONE stuck point. Agent supplies the dict; we route through s6.
        from ..stages.s6_hypothesis import run as s6_run
        sc = params["stuck_context"]
        # Auto-fill input/output state from trace at instr_idx if agent didn't
        # supply them explicitly (so the verifier actually runs — same fix as
        # script_mode, otherwise everything stays in `pending`).
        in_state = params.get("input_state")
        out_state = params.get("expected_output_state")
        idx = sc.get("instr_idx") if isinstance(sc, dict) else None
        if in_state is None and idx is not None and 0 <= idx < len(core._items):
            in_state = dict(core._items[idx].regs_read)
        if out_state is None and idx is not None and 0 <= idx < len(core._items):
            out_state = dict(core._items[idx].regs_write)
        summary = s6_run({
            "items":         core._items,
            "work":          core.work,
            "session":       core.session,
            "verifier":      core.verifier,
            "llm":           getattr(core, "_llm", None),  # use delegated backend if serve_mcp wired one
            "stuck_context": sc,
            "input_state":   in_state,
            "expected_output_state": out_state,
            "n":             params.get("n"),
        })
        return summary
    if method == "s6_auto_loop":
        # Mirror script_mode's S6 phase: find stuck points, propose+verify
        # each, promote passes. Budget enforced via LLM client (caller should
        # set Budget before calling). Returns aggregate stats.
        from .script_mode import _find_stuck_points
        from ..stages.s6_hypothesis import run as s6_run
        cap = params.get("max_points")
        stuck = _find_stuck_points(core, max_points=cap)
        # Pre-fetch the wired LLM so we don't fall back to LLMClient() (which
        # would try to read DEEPSEEK_API_KEY) inside s6_hypothesis.run.
        wired_llm = getattr(core, "_llm", None)
        total_candidates = 0
        total_passed = total_failed = total_pending = total_inconclusive = 0
        for sp in stuck:
            in_state: dict | None = None
            out_state: dict | None = None
            if sp.instr_idx is not None and 0 <= sp.instr_idx < len(core._items):
                ins = core._items[sp.instr_idx]
                in_state = dict(ins.regs_read)
                out_state = dict(ins.regs_write)
            try:
                r = s6_run({
                    "items":         core._items,
                    "work":          core.work,
                    "session":       core.session,
                    "verifier":      core.verifier,
                    "llm":           wired_llm,
                    "stuck_context": sp,
                    "input_state":   in_state,
                    "expected_output_state": out_state,
                    "n":             params.get("n"),
                })
            except Exception as e:
                return {"processed_until_error": True,
                        "error": f"{type(e).__name__}: {e}",
                        "candidates": total_candidates,
                        "passed": total_passed, "failed": total_failed,
                        "pending": total_pending,
                        "inconclusive": total_inconclusive}
            total_candidates  += r.get("candidates", 0)
            total_passed      += r.get("passed", 0)
            total_failed      += r.get("failed", 0)
            total_pending     += r.get("pending", 0)
            total_inconclusive += r.get("inconclusive", 0)
            # Mirror script_mode: every verdict=pass is promoted to a finding,
            # otherwise the verifier work is thrown away and findings.sqlite
            # stays empty even though hypotheses passed (BR-2 §2).
            for v in r.get("verdicts", []):
                if v.get("verdict") == "pass":
                    try:
                        core.promote_to_finding(
                            int(v["hyp_id"]),
                            verifier_strategy="handler_semantic",
                        )
                    except Exception:
                        pass   # already promoted / race — skip silently
        return {"processed":    len(stuck),
                "candidates":   total_candidates,
                "passed":       total_passed,
                "failed":       total_failed,
                "pending":      total_pending,
                "inconclusive": total_inconclusive}

    if method == "checkpoint":
        core.checkpoint()
        return "ok"
    if method == "pause":
        core.pause(params.get("reason", "agent requested"), params.get("hint"))
        return "ok"

    # --- intervention API (PLAN §15 agent operability + 留痕) ---
    if method == "override_verdict":
        core.override_verdict(int(params["hyp_id"]), params["new_verdict"],
                              reason=params["reason"], actor=params.get("actor", "agent"))
        return "ok"
    if method == "batch_override_verdict":
        # 0526Plan B3: same as override_verdict but for a list of hyp_ids
        # in one call. agent-user-advice §3.3 — the hyp-by-hyp loop is
        # too verbose when a reviewer wants to mass-resolve similar stucks.
        hyp_ids = [int(h) for h in (params.get("hyp_ids") or [])]
        verdict = params["new_verdict"]
        reason = params["reason"]
        actor = params.get("actor", "agent")
        results: dict[str, Any] = {"total": len(hyp_ids), "ok": [], "errors": {}}
        for hid in hyp_ids:
            try:
                core.override_verdict(hid, verdict, reason=reason, actor=actor)
                results["ok"].append(hid)
            except Exception as e:
                results["errors"][str(hid)] = f"{type(e).__name__}: {e}"
        return results
    if method == "list_interventions":
        # 0526Plan B3: filtered intervention audit query.
        from ..store import open_hypotheses_db
        from ..store import read_interventions as _read
        conn = open_hypotheses_db(core.work)
        try:
            return _read(
                conn,
                limit=int(params.get("limit") or 50),
                actor=params.get("actor"),
                action=params.get("action"),
            )
        finally:
            conn.close()
    if method == "force_status":
        core.force_status(int(params["hyp_id"]), params["new_status"],
                          reason=params["reason"], actor=params.get("actor", "agent"))
        return "ok"
    if method == "inject_finding":
        fid = core.inject_finding(
            kind=params["kind"], subject=params["subject"],
            payload=params.get("payload", {}),
            reason=params["reason"],
            actor=params.get("actor", "agent"),
            verifier_strategy=params.get("verifier_strategy", "manual"),
        )
        return {"finding_id": fid}
    if method == "add_tag":
        core.add_tag_with_reason(int(params["hyp_id"]), params["axis"], params["value"],
                                 reason=params["reason"],
                                 actor=params.get("actor", "agent"))
        return "ok"
    if method == "add_dependency":
        core.add_dependency_with_reason(
            int(params["from_hyp_id"]), int(params["to_hyp_id"]),
            kind=params.get("kind", "supports"),
            reason=params["reason"], actor=params.get("actor", "agent"),
        )
        return "ok"
    if method == "rerun_from_stage":
        return core.rerun_from_stage(params["stage"],
                                     reason=params.get("reason", "agent requested"),
                                     actor=params.get("actor", "agent"))
    if method == "resume":
        core.resume_run(reason=params.get("reason", "agent resumed"),
                        actor=params.get("actor", "agent"))
        return "ok"
    if method == "list_interventions":
        return core.list_interventions(
            limit=int(params.get("limit", 100)),
            action=params.get("action"),
        )
    if method == "localize_divergence":
        # capability_request.md §P1-3 differential localiser
        return core.localize_divergence(
            bytes.fromhex(params["good_input_hex"]),
            bytes.fromhex(params["bad_input_hex"]),
            resync_look_ahead=int(params.get("resync_look_ahead", 200)),
        )
    if method == "invalidate_cluster":
        # capability_request.md §P1-4 cluster cascade
        return core.invalidate_cluster(
            int(params["parent_finding_id"]),
            reason=params["reason"],
            actor=params.get("actor", "agent"),
        )

    # --- VMP phase API (the light-to-heavy route as the agent's tool surface) ---
    # These make engine.vmp_phase_api callable from the agent, so the path is
    # enforced by the interface it actually drives — not just documented. Order
    # is enforced (a phase refuses entry until its predecessor recorded a
    # verdict), vmtrace is gated (proof OR confirmation + budget), and there is
    # deliberately NO "enumerate standard crypto" method: the only crypto-source
    # move is phase_3 provenance (追数据流，不撒候选 — roadmap §8.13).
    if method == "phase_state":
        api = _phase_api(core)
        seq = api.run.sequence
        return {
            "sequence": [s.name for s in seq.ordered]
                        + [s.name for s in seq.escalations],
            "trail":    [o.to_dict() for o in api.run.trail()],
            "closed":   api.run.is_closed(),
            "heavy_budget": api.heavy_budget.to_dict() if api.heavy_budget else None,
        }
    if method == "phase_1_io_observe":
        api = _phase_api(core)
        spec = api.phase_1_io_observe(entry_pc=_as_int(params["entry_pc"]),
                                      label=params.get("label", ""))
        return spec.to_dict()
    if method == "phase_2_materialization_trace":
        api = _phase_api(core)
        spec = api.phase_2_materialization_trace(
            output_base=_as_int(params["output_base"]),
            output_len=_as_int(params["output_len"]))
        return spec.to_dict()
    if method == "phase_3_watch_producer":
        api = _phase_api(core)
        spec = api.phase_3_watch_producer(
            addr=_as_int(params["addr"]),
            value_name=params["value_name"],
            reason=params.get("reason", "phase_3: trace producer of observed value"))
        return spec.to_dict()
    if method == "phase_3_classify":
        # The crypto-source classifier: follow the data flow. No candidate spray.
        from ..constant_provenance import classify_values_in_params
        api = _phase_api(core)
        if not api.run.entered("phase_3_provenance"):
            api.run.enter("phase_3_provenance")
        results = classify_values_in_params(params)
        return {"classifications": [r.to_dict() for r in results]}
    if method == "phase_4_formula_induction":
        from ..vmp_phase_api import FormulaInduction
        api = _phase_api(core)
        intent = api.phase_4_formula_induction(FormulaInduction(
            expression=params["expression"],
            derived_from=tuple(params.get("derived_from", []) or []),
            note=params.get("note", "")))
        return intent.to_dict()
    if method == "phase_5_parity":
        from ..vmp_phase_api import FormulaInduction, ParityIntent
        api = _phase_api(core)
        intent = ParityIntent(
            formula=FormulaInduction(
                expression=params.get("expression", ""),
                derived_from=tuple(params.get("derived_from", []) or [])),
            inputs_min=int(params.get("inputs_min", 1)))
        return api.phase_5_parity(intent).to_dict()
    if method == "phase_record":
        from ..phase_sequence import PhaseStatus
        api = _phase_api(core)
        status = PhaseStatus(params["status"])  # ran|closed|could_not_close
        outcome = api.record(
            params["phase"], status,
            summary=params.get("summary", ""),
            could_not_close_reason=params.get("could_not_close_reason", ""))
        return outcome.to_dict()
    if method == "phase_heavy_vmtrace_prompt":
        from ..vmp_phase_api import VmtraceBudget
        api = _phase_api(core)
        budget = None
        if params.get("budget"):
            b = params["budget"]
            budget = VmtraceBudget(runtime_s=float(b["runtime_s"]),
                                   disk_mb=float(b["disk_mb"]), note=b.get("note", ""))
        return api.heavy_vmtrace_prompt(budget).to_dict()
    if method == "phase_heavy_vmtrace":
        from ..phase import Anchor
        from ..vmp_phase_api import VmtraceBudget
        from ..phase_sequence import EscalationConfirmation, EscalationProof
        api = _phase_api(core)
        a = params["anchor"]
        anchor = Anchor(anchor_type=a["anchor_type"], params=a.get("params", {}),
                        label=a.get("label", ""))
        b = params["budget"]
        budget = VmtraceBudget(runtime_s=float(b["runtime_s"]),
                               disk_mb=float(b["disk_mb"]), note=b.get("note", ""))
        proof = None
        if params.get("proof"):
            p = params["proof"]
            proof = EscalationProof(cites=tuple(p.get("cites", [])), reason=p.get("reason", ""))
        confirmation = None
        if params.get("confirmation"):
            c = params["confirmation"]
            confirmation = EscalationConfirmation(who=c.get("who", "agent"), note=c.get("note", ""))
        spec = api.phase_heavy_vmtrace(anchor=anchor, budget=budget,
                                       proof=proof, confirmation=confirmation)
        return spec.to_dict()

    raise ValueError(f"unknown method: {method!r}")


def _as_int(v: Any) -> int:
    """Accept an int or a hex/decimal string (e.g. "0x706d0")."""
    if isinstance(v, str):
        return int(v, 0)
    return int(v)


def _phase_api(core: Core):
    """The per-session VmpPhaseApi — holds the phase state machine so order +
    the escalation gate persist across RPC calls in one agent-serve session."""
    api = getattr(core, "_vmp_phase_api", None)
    if api is None:
        from ..vmp_phase_api import VmpPhaseApi
        api = VmpPhaseApi()
        core._vmp_phase_api = api
    return api


def _hyp_to_dict(h) -> dict[str, Any]:
    return {
        "id": h.id, "parent_id": h.parent_id, "depth": h.depth,
        "status": h.status, "kind": h.kind, "source": h.source,
        "subject": h.subject, "payload": h.payload, "confidence": h.confidence,
    }


def _write_ok(stdout, req_id, result, methodology: dict | None = None) -> None:
    env: dict[str, Any] = {"id": req_id, "result": result}
    if methodology:
        env["methodology"] = methodology
    stdout.write(json.dumps(env) + "\n")
    stdout.flush()


def _write_err(stdout, req_id, code, message, data: dict | None = None) -> None:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    stdout.write(json.dumps({"id": req_id, "error": err}) + "\n")
    stdout.flush()


_ = EventKind   # re-export silencer (used by tracker callback above)
