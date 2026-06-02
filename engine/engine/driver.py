#!/usr/bin/env python3
"""In-process driver for `utov agent-serve` — public API (BR-4 §A).

`from engine.driver import drive` gives you a single function entry to spawn
the engine, hook a Python callable as the LLM provider, and run any of the
bundled workflows. No file-based handoff, no CLI / PYTHONPATH dance — just
import and call. See `contracts/agent_protocol.md` for why a Python callable
is a first-class implementation of the wire protocol.

Historical:

Implements the NDJSON wire protocol in `contracts/agent_protocol.md`:

  - Spawn `utov agent-serve --runner-cmd ... --input ...`
  - Wait for "engine ready" on stderr
  - Send tool requests on stdin
  - Read replies / events / llm_requests on stdout
  - Service `llm_request` inline (engine deadlocks if you don't)
  - Graceful shutdown

This is the canonical reference implementation: single-threaded, no
threading/Queue gymnastics. The "agent LLM" is hooked via a pluggable
function — pick one of:

  --llm file        (default) — write llm_request to /tmp/utov_llm/in/<id>.json,
                                block until /tmp/utov_llm/out/<id>.json appears.
                                Lets a HUMAN or external agent / model service
                                answer LLM calls without any Python integration.

  --llm stub                  — in-process rule-based stub (no real LLM).
                                Useful for offline smoke-testing of the wire
                                protocol. Will give garbage hypotheses, but
                                proves the round-trip works.

  --llm arm-heuristic         — in-process ARM-disassembly-aware provider.
                                Only emits a hypothesis when the stuck snippet
                                is a plain 3-register binop (eor/and/orr/add/
                                sub/mul/lsl/lsr/ror) the verifier can check
                                mechanically. BR-2 §10b replacement for the
                                random stub — runs `--workflow s6-loop` on
                                libsha256.so end-to-end producing real findings
                                at $0.

  --llm <dotted.path:fn>      — BR-2 §10d: import any python callable
                                `def provider(req: dict) -> dict`. Lets you
                                inject custom inference without going through
                                the /tmp file inbox.

Workflows:

  --workflow demo             — metadata + list_stages + get_hypotheses, exit.
                                No LLM needed.

  --workflow pipeline         — run_pipeline (s1..s5). With --mode=aggressive
                                the engine drives S6 internally; LLM calls
                                come at you via llm_request.

  --workflow promote-plugin   — run_pipeline, then for every PENDING
                                algo_signature hyp (from S1.5 plugin),
                                override_verdict=pass + promote_to_finding.
                                This is the practical "agent acts as judge
                                for fingerprint evidence" pattern — no LLM.

  --workflow s6-list          — run_pipeline + s6_find_stuck_points; print a
                                mnemonic histogram of stuck snippets. Useful
                                for sizing how much verifier coverage you'd
                                need to promote them all. (BR-2 §10)

  --workflow s6-loop          — run_pipeline + s6_auto_loop driven by the
                                chosen `--llm`. Each `llm_request` answered
                                in-process. Findings table is populated
                                automatically by the engine's s6_auto_loop
                                + verify_plugin_findings paths.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# LLM provider implementations
# ---------------------------------------------------------------------------

INBOX  = Path("/tmp/utov_llm/in")
OUTBOX = Path("/tmp/utov_llm/out")


def llm_file_handoff(req: dict) -> dict:
    """Stash `req` under INBOX/<id>.json; poll OUTBOX/<id>.json for the reply.
    Lets a human or external agent answer without any Python integration."""
    n = req["id"]
    INBOX.mkdir(parents=True, exist_ok=True)
    OUTBOX.mkdir(parents=True, exist_ok=True)
    inp = INBOX  / f"{n}.json"
    out = OUTBOX / f"{n}.json"
    inp.write_text(json.dumps(req, indent=2))
    sys.stderr.write(f"\n[llm_request {n}] waiting for {out} ...\n")
    sys.stderr.flush()
    while not out.exists():
        time.sleep(0.25)
    reply = json.loads(out.read_text())
    reply.setdefault("id", n)
    reply.setdefault("type", "llm_response")
    return reply


def llm_stub(req: dict) -> dict:
    """Rule-based stub. Always proposes ADD/XOR/OR/AND for handler_semantic
    asks, using the first few register names seen in the user_context. Real
    integrations should hit an actual LLM here."""
    ctx = (req.get("user_context") or "")
    regs = list(dict.fromkeys(re.findall(r"\b(x\d{1,2}|sp|lr|fp)\b", ctx)))[:3]
    if len(regs) < 3:
        regs += ["x0", "x1", "x2"][len(regs):]
    n = int(req.get("n", 5))
    out = [
        {"kind": "handler_semantic", "subject": f"handler@{op}",
         "payload": {"op": op, "dst": regs[0], "src": [regs[1], regs[2]]},
         "confidence": conf, "rationale": f"stub guess: {op} on {regs}"}
        for op, conf in [("XOR", 0.7), ("ADD", 0.6), ("OR", 0.4),
                          ("AND", 0.3), ("SUB", 0.3)][:n]
    ]
    return {"id": req["id"], "type": "llm_response", "hypotheses": out}


_ARM_TO_OP = {
    "eor": "XOR", "and": "AND", "orr": "OR", "orn": "ORN",
    "add": "ADD", "sub": "SUB", "mul": "MUL",
    "lsl": "LSL", "lsr": "LSR", "asr": "ASR", "ror": "ROR",
    "bic": "BIC", "eon": "EON",
}
_ARM_UNARY = {
    "mov":  "MOV",  "mvn": "MVN",  "neg": "NEG",
    "sxtw": "SXTW", "uxtw": "UXTW", "rev": "REV",
}
_REG = re.compile(r"^[wx]\d+$|^wzr$|^xzr$|^sp$|^lr$|^fp$")
_EXT_KIND = {"lsl", "lsr", "asr", "ror", "sxtw", "sxth", "sxtb",
             "uxtw", "uxth", "uxtb"}
_IMM = re.compile(r"^#(?:0x[0-9a-fA-F]+|-?\d+)$")


def _llm_arm_heuristic(req: dict) -> dict:
    """ARM-disassembly aware provider (BR-2 §10b).

    Emits a high-confidence handler_semantic claim when the stuck snippet is a
    form the engine's verifier knows how to check mechanically:
      - 3-register binop  (eor/and/orr/add/sub/mul/lsl/lsr/asr/ror/bic/eon/orn)
      - reg-reg-#imm binop with literal immediate
      - 3-register binop with src2 shift/extend tail
      - unary form (mov/mvn/neg/sxtw/uxtw/rev) where src is a single register

    Anything else (memory loads, adrp, fmov, movk, multi-imm, sp-relative
    arithmetic, …) gets an empty reply so it doesn't pollute the ledger.
    """
    req_id = req.get("id", "llm-?")
    ctx = req.get("user_context") or ""
    m = re.search(r"Snippet:\s*\n(.+)", ctx)
    if not m:
        return {"id": req_id, "type": "llm_response", "hypotheses": []}
    insn = m.group(1).strip()
    toks = [t for t in re.split(r"[,\s]+", insn) if t]
    if not toks:
        return {"id": req_id, "type": "llm_response", "hypotheses": []}
    mnem = toks[0].lower()
    ops = toks[1:]

    # Reject any operand using sp directly — those are prologue/frame setup
    # and not interesting for handler semantics on libsha256.
    def _has_sp(xs): return any(o == "sp" for o in xs)

    # --- Unary form: mov/mvn/neg/sxtw/uxtw/rev Rd, Rs ---
    if mnem in _ARM_UNARY and len(ops) >= 2 and not _has_sp(ops[:2]) \
            and _REG.match(ops[0]) and _REG.match(ops[1]):
        op = _ARM_UNARY[mnem]
        dst, src = ops[0], ops[1]
        return _emit_hyp(req_id, mnem, op, dst=dst, src=[src])

    if mnem not in _ARM_TO_OP:
        return {"id": req_id, "type": "llm_response", "hypotheses": []}
    op = _ARM_TO_OP[mnem]

    # --- 3-register binop, optionally with a shift/extend tail ---
    if len(ops) >= 3 and all(_REG.match(o) for o in ops[:3]) and not _has_sp(ops[:3]):
        dst, s1, s2 = ops[0], ops[1], ops[2]
        if len(ops) == 3:
            return _emit_hyp(req_id, mnem, op, dst=dst, src=[s1, s2])
        # Shift/extend tail: "lsl #3", "sxtw", "sxtw #2"
        tail = ops[3].lower() if len(ops) > 3 else ""
        if tail in _EXT_KIND:
            amount = 0
            if len(ops) >= 5 and _IMM.match(ops[4]):
                amount = _parse_imm(ops[4])
            return _emit_hyp(req_id, mnem, op, dst=dst, src=[s1, s2],
                              src2_ext={"kind": tail, "amount": amount})

    # --- 2-register + #imm binop ---
    if len(ops) >= 3 and _REG.match(ops[0]) and _REG.match(ops[1]) \
            and _IMM.match(ops[2]) and not _has_sp(ops[:2]):
        dst, s1 = ops[0], ops[1]
        imm_val = _parse_imm(ops[2])
        return _emit_hyp(req_id, mnem, op, dst=dst, src=[s1],
                          imm=f"0x{imm_val:x}")

    return {"id": req_id, "type": "llm_response", "hypotheses": []}


def _parse_imm(token: str) -> int:
    raw = token.lstrip("#")
    return int(raw, 16) if raw.lower().startswith("0x") else int(raw)


def _emit_hyp(req_id: str, mnem: str, op: str, *, dst: str,
              src: list[str], imm: str | None = None,
              src2_ext: dict | None = None) -> dict:
    payload: dict[str, Any] = {"op": op, "dst": dst, "src": src}
    if imm is not None:
        payload["imm"] = imm
    if src2_ext is not None:
        payload["src2_ext"] = src2_ext
    subj_tail = "_".join([dst, *src] + ([imm] if imm else []))
    rationale = (
        f"ARM64 '{mnem}' resolved as {op}; payload={payload}. "
        f"verifier can check mechanically."
    )
    return {
        "id": req_id, "type": "llm_response",
        "hypotheses": [{
            "kind":       "handler_semantic",
            "subject":    f"{mnem}_{subj_tail}",
            "payload":    payload,
            "confidence": 0.9,
            "rationale":  rationale,
        }],
    }


LLM_PROVIDERS: dict[str, Callable[[dict], dict]] = {
    "file":          llm_file_handoff,
    "stub":          llm_stub,
    "arm-heuristic": _llm_arm_heuristic,
}


def _load_dotted_provider(spec: str) -> Callable[[dict], dict]:
    """BR-2 §10d: `--llm pkg.mod:callable` dynamic-loads any python callable.

    Signature: `(req: dict) -> dict` — same as the bundled providers.
    """
    if ":" not in spec:
        raise SystemExit(
            f"--llm {spec!r} not recognized. Expected one of "
            f"{sorted(LLM_PROVIDERS)} or a dotted spec 'pkg.mod:callable'."
        )
    mod_path, fn_name = spec.rsplit(":", 1)
    mod = importlib.import_module(mod_path)
    fn = getattr(mod, fn_name, None)
    if not callable(fn):
        raise SystemExit(f"{mod_path}:{fn_name} is not a callable")
    return fn   # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Wire-protocol driver (single-threaded, reads stdout sequentially)
# ---------------------------------------------------------------------------

def spawn(runner_cmd: str, input_hex: str, work_root: str = "work",
          *, new_run: bool = True) -> subprocess.Popen:
    """Spawn `utov agent-serve`; block on stderr for 'engine ready'.

    BR-2 §10a: use `sys.executable` so we inherit the parent's interpreter and
    don't fall over the "homebrew python3 -> 3.14 but wheel installed under
    3.11" trap. `--new-run / --resume` is the BR-2 §11 fix.
    """
    cmd = [
        sys.executable, "-u", "-m", "engine.cli", "agent-serve",
        "--runner-cmd", runner_cmd,
        "--input",      input_hex,
        "--work-root",  work_root,
        "--new-run" if new_run else "--resume",
    ]
    p = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, env=os.environ.copy(),
    )
    # Block on the "engine ready" sentinel on stderr (≤ 30 s).
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        line = p.stderr.readline()                    # type: ignore[union-attr]
        if not line:
            if p.poll() is not None:
                raise RuntimeError(f"engine died during startup (rc={p.returncode})")
            time.sleep(0.05)
            continue
        sys.stderr.write(f"[engine stderr] {line.rstrip()}\n")
        if "engine ready" in line:
            return p
    raise TimeoutError("engine did not say 'engine ready' within 30 s")


def send(p: subprocess.Popen, obj: dict) -> None:
    p.stdin.write(json.dumps(obj) + "\n")             # type: ignore[union-attr]
    p.stdin.flush()                                    # type: ignore[union-attr]


def read_until(p: subprocess.Popen, want_id: int,
               llm: Callable[[dict], dict]) -> dict:
    """Read stdout until a tool reply with matching id arrives. Inline-services
    llm_requests so the engine doesn't deadlock waiting for its own LLM."""
    while True:
        line = p.stdout.readline()                    # type: ignore[union-attr]
        if not line:
            raise RuntimeError("engine stdout EOF before tool reply arrived")
        line = line.rstrip("\n")
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = msg.get("type")
        if t == "event":
            # fire-and-forget; you can hook display logic here if you want
            continue
        if t == "llm_request":
            send(p, llm(msg))
            continue
        if t in ("llm_response", "llm_error"):
            # Stray (shouldn't happen on stdout in this direction); ignore.
            continue
        # Tool reply.
        if msg.get("id") == want_id:
            return msg


# ---------------------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------------------

def workflow_demo(p, llm) -> None:
    for rid, (method, params) in enumerate([
        ("metadata", None),
        ("list_stages", None),
        ("get_hypotheses", {}),
    ], 1):
        req: dict[str, Any] = {"id": rid, "method": method}
        if params is not None:
            req["params"] = params
        send(p, req)
        resp = read_until(p, rid, llm)
        print(f"\n=== {method} ===")
        print(json.dumps(resp.get("result", resp.get("error")), indent=2)[:1500])


def workflow_pipeline(p, llm, *, mode: str | None) -> None:
    params: dict[str, Any] = {}
    if mode is not None:
        params["mode"] = mode    # forwarded if your server supports it
    send(p, {"id": 1, "method": "run_pipeline", "params": params})
    resp = read_until(p, 1, llm)
    print("=== run_pipeline ===")
    print(json.dumps(resp.get("result", resp.get("error")), indent=2)[:2500])


def workflow_promote_plugin(p, llm) -> None:
    """Run S1..S5, then promote every pending algo_signature hyp to a finding
    via override_verdict + promote_to_finding. This is the 'agent acts as
    judge' pattern — no LLM call needed; you trust the plugin evidence."""
    send(p, {"id": 1, "method": "run_pipeline"})
    read_until(p, 1, llm)
    send(p, {"id": 2, "method": "get_hypotheses",
             "params": {"status": "pending", "kind": "algo_signature"}})
    hyps = read_until(p, 2, llm).get("result", [])
    print(f"=== {len(hyps)} pending algo_signature hypotheses ===")
    for h in hyps:
        print(f"  hyp#{h['id']:<3} subj={h['subject']:<20} source={h.get('source')}")

    promoted, failed = 0, []
    req_id = 10
    for h in hyps:
        send(p, {"id": req_id, "method": "override_verdict",
                 "params": {"hyp_id": h["id"], "new_verdict": "pass",
                            "actor": "ref-agent",
                            "reason": "plugin fingerprint match; agent judged "
                                      "evidence sufficient given expected hits "
                                      "and S4 closed slice."}})
        read_until(p, req_id, llm)
        req_id += 1
        send(p, {"id": req_id, "method": "promote_to_finding",
                 "params": {"hyp_id": h["id"],
                            "verifier_strategy": "agent_override",
                            "stage": "agent"}})
        resp = read_until(p, req_id, llm)
        req_id += 1
        if resp.get("error"):
            failed.append((h["id"], resp["error"]["message"]))
        else:
            promoted += 1
    print(f"\n=== promote result: {promoted} ok, {len(failed)} failed ===")
    for hid, msg in failed[:5]:
        print(f"  hyp#{hid} → {msg[:200]}")


def workflow_s6_list(p, llm, *, max_points: int | None) -> None:
    """BR-2 §10: print a mnemonic histogram of S6 stuck points so callers can
    size verifier coverage. No LLM needed (but `llm` is wired so we can still
    service stray llm_requests during run_pipeline)."""
    from collections import Counter
    send(p, {"id": 1, "method": "run_pipeline"})
    read_until(p, 1, llm)
    params: dict[str, Any] = {}
    if max_points is not None:
        params["max_points"] = max_points
    send(p, {"id": 2, "method": "s6_find_stuck_points", "params": params})
    stuck = read_until(p, 2, llm).get("result", [])
    print(f"=== {len(stuck)} stuck points ===")
    mnems: Counter[str] = Counter()
    for s in stuck:
        snip = (s.get("snippet") or "").strip()
        mnems[snip.split()[0] if snip else "?"] += 1
    print("\nmnemonic distribution:")
    for mnem, count in mnems.most_common():
        print(f"  {mnem:<10} {count}")
    print("\nfirst 20 snippets:")
    for i, s in enumerate(stuck[:20]):
        print(f"  [{i:3d}] idx={s.get('instr_idx'):>5}  "
              f"{(s.get('snippet') or '').strip()}")


def workflow_s6_loop(p, llm, *, max_points: int | None, n: int | None,
                     promote_plugin_findings: bool) -> None:
    """BR-2 §10c: bundled "do everything" S6 driver.

    1. run_pipeline (s1..s5)
    2. verify_plugin_findings — promotes algo_signature hyps whose fingerprint
       magic actually appears in the trace. Same mechanical pass that
       script_mode runs after S5 in frugal mode.
    3. s6_auto_loop — engine drives every stuck point through `llm` and
       promotes each verdict=pass to a finding.
    4. Pull a summary of hypothesis/finding tables and print.

    `--llm arm-heuristic` makes this end-to-end on libsha256.so in seconds at
    $0 spend.
    """
    send(p, {"id": 1, "method": "run_pipeline"})
    read_until(p, 1, llm)

    # Mechanical plugin verify (no LLM, deterministic).
    send(p, {"id": 2, "method": "verify_plugin_findings"})
    plug = read_until(p, 2, llm).get("result", {}) or {}
    print(f"=== verify_plugin_findings: "
          f"checked={plug.get('checked', 0)} "
          f"passed={plug.get('passed', 0)} "
          f"promoted={plug.get('promoted', 0)} ===")

    # BR-4 §1: deterministic handler-semantic pass — scoops up reg-reg-reg
    # binops verifier can check without LLM. On TC2 VMP baseline this
    # promotes ~5,500 findings at $0.
    send(p, {"id": 4, "method": "verify_handler_binops"})
    hnd = read_until(p, 4, llm).get("result", {}) or {}
    print(f"=== verify_handler_binops: "
          f"checked={hnd.get('checked', 0)} "
          f"passed={hnd.get('passed', 0)} "
          f"promoted={hnd.get('promoted', 0)} ===")

    if promote_plugin_findings:
        # Optional "agent-as-judge" override for plugin hyps that did NOT
        # auto-verify (e.g. fingerprint anchor outside the captured trace).
        send(p, {"id": 3, "method": "get_hypotheses",
                 "params": {"status": "pending", "kind": "algo_signature"}})
        leftovers = read_until(p, 3, llm).get("result", [])
        rid = 100
        promoted = 0
        for h in leftovers:
            send(p, {"id": rid, "method": "override_verdict",
                     "params": {"hyp_id": h["id"], "new_verdict": "pass",
                                "actor": "ref-agent",
                                "reason": "plugin evidence accepted by driver."}})
            read_until(p, rid, llm)
            rid += 1
            send(p, {"id": rid, "method": "promote_to_finding",
                     "params": {"hyp_id": h["id"],
                                "verifier_strategy": "agent_override",
                                "stage": "agent"}})
            r = read_until(p, rid, llm)
            rid += 1
            if not r.get("error"):
                promoted += 1
        print(f"=== agent-judge plugin promotion: {promoted}/{len(leftovers)} ===")

    s6_params: dict[str, Any] = {}
    if max_points is not None:
        s6_params["max_points"] = max_points
    if n is not None:
        s6_params["n"] = n
    send(p, {"id": 50, "method": "s6_auto_loop", "params": s6_params})
    r = read_until(p, 50, llm).get("result", {}) or {}
    print(f"\n=== s6_auto_loop: {json.dumps(r)} ===")

    send(p, {"id": 60, "method": "get_hypotheses"})
    hyps = read_until(p, 60, llm).get("result", [])
    by_kind: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for h in hyps:
        by_kind[h["kind"]] = by_kind.get(h["kind"], 0) + 1
        by_status[h["status"]] = by_status.get(h["status"], 0) + 1
    print(f"\nhypotheses total: {len(hyps)}")
    print(f"  by kind:   {by_kind}")
    print(f"  by status: {by_status}")


# ---------------------------------------------------------------------------
# Programmatic API (BR-4: in-process Python provider is a first-class user)
# ---------------------------------------------------------------------------

_WORKFLOW_NAMES = ["demo", "pipeline", "promote-plugin", "s6-list", "s6-loop"]


def _resolve_workflow(name: str, *,
                      mode: str | None = None,
                      max_points: int | None = None,
                      n: int | None = None,
                      promote_plugin_findings: bool = False
                      ) -> Callable[[Any, Callable[[dict], dict]], None]:
    """Return a `(p, provider) -> None` runner for the chosen workflow."""
    if name == "demo":
        return lambda p, prov: workflow_demo(p, prov)
    if name == "pipeline":
        return lambda p, prov: workflow_pipeline(p, prov, mode=mode)
    if name == "promote-plugin":
        return lambda p, prov: workflow_promote_plugin(p, prov)
    if name == "s6-list":
        return lambda p, prov: workflow_s6_list(p, prov, max_points=max_points)
    if name == "s6-loop":
        return lambda p, prov: workflow_s6_loop(
            p, prov,
            max_points=max_points, n=n,
            promote_plugin_findings=promote_plugin_findings,
        )
    raise SystemExit(f"unknown workflow {name!r}")


def drive(*,
          runner_cmd: str,
          input_hex: str,
          provider: Callable[[dict], dict],
          workflow: str = "s6-loop",
          work_root: str = "work",
          new_run: bool = True,
          mode: str | None = None,
          max_points: int | None = None,
          n: int | None = None,
          promote_plugin_findings: bool = False,
          ) -> None:
    """In-process driver entry — BR-4 §A.

    Spawn `utov agent-serve`, run the chosen workflow with `provider` as the
    LLM provider, gracefully shut down. `provider` is any Python callable
    matching `(req: dict) -> dict` — the same in-process shape the bundled
    `llm_stub` / `arm-heuristic` providers use; agent_protocol.md treats this
    as a first-class implementation of `llm_request` (no different from a
    DeepSeek-backed agent — the wire-side contract is structured JSON, not
    natural-language QA).

    Example:

        from engine.driver import drive

        def my_provider(req: dict) -> dict:
            # inspect req["user_context"], emit hypotheses
            return {"id": req["id"], "type": "llm_response",
                    "hypotheses": [...]}

        drive(runner_cmd="java -jar runner.jar serve lib.so",
              input_hex="616263",
              provider=my_provider,
              workflow="s6-loop")
    """
    runner = _resolve_workflow(
        workflow,
        mode=mode, max_points=max_points, n=n,
        promote_plugin_findings=promote_plugin_findings,
    )
    p = spawn(runner_cmd, input_hex, work_root=work_root, new_run=new_run)
    try:
        runner(p, provider)
        send(p, {"id": 999, "method": "shutdown"})
        try:
            read_until(p, 999, provider)
        except Exception:
            pass
    finally:
        try:
            p.stdin.close()                            # type: ignore[union-attr]
        except Exception:
            pass
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


# ---------------------------------------------------------------------------
# Main (CLI thin wrapper over drive())
# ---------------------------------------------------------------------------

def _build_workflow_runner(name: str, args) -> Callable[[Any, Callable], None]:
    """Back-compat shim: CLI used to build the runner here. drive() now uses
    _resolve_workflow directly. Kept so any third-party caller that imported
    this name keeps working."""
    return _resolve_workflow(
        name,
        mode=getattr(args, "mode", None),
        max_points=getattr(args, "max_points", None),
        n=getattr(args, "n", None),
        promote_plugin_findings=getattr(args, "promote_plugin_findings", False),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runner-cmd", required=True,
                    help="shell command that spawns the runner")
    ap.add_argument("--input",      required=True, help="hex-encoded input")
    ap.add_argument("--work-root",  default="work")
    ap.add_argument("--llm", default="file",
                    help="how to answer llm_request: 'file' (default, /tmp inbox/outbox), "
                         "'stub' (random), 'arm-heuristic' (in-process ARM-aware), "
                         "or a dotted python.path:callable (BR-2 §10d).")
    ap.add_argument("--workflow", choices=_WORKFLOW_NAMES, default="demo")
    ap.add_argument("--mode", default=None, help="forwarded to run_pipeline")
    ap.add_argument("--max-points", type=int, default=None,
                    help="forwarded to s6_find_stuck_points / s6_auto_loop")
    ap.add_argument("--n", type=int, default=None,
                    help="forwarded to s6_auto_loop: hypotheses per stuck point")
    ap.add_argument("--promote-plugin-findings", action="store_true",
                    help="in --workflow s6-loop: also agent-override-promote "
                         "plugin algo_signature hyps that didn't auto-verify")
    ap.add_argument("--resume", action="store_true",
                    help="BR-2 §11: drive an existing run instead of --new-run")
    args = ap.parse_args()

    # Resolve LLM provider.
    if args.llm in LLM_PROVIDERS:
        llm: Callable[[dict], dict] = LLM_PROVIDERS[args.llm]
    else:
        llm = _load_dotted_provider(args.llm)

    drive(
        runner_cmd=args.runner_cmd,
        input_hex=args.input,
        provider=llm,
        workflow=args.workflow,
        work_root=args.work_root,
        new_run=not args.resume,
        mode=args.mode,
        max_points=args.max_points,
        n=args.n,
        promote_plugin_findings=args.promote_plugin_findings,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
