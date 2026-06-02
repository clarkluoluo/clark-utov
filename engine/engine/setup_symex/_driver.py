"""setup_symex.driver section (split from the monolithic module)."""
from __future__ import annotations


import enum
import os
import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..dataflow import classify_semop
from ..types import Instruction, MemSnapshot
from ..watch_first_write import (
    WatchFirstWriteConfig,
    WatchFirstWriteSpec,
    request_watch_first_write,
)
from ._boundary import BoundaryEnd, BoundaryRole, LOCATED_SINK_VALIDATION, LOCATED_WATCH, bind_boundary, locate_boundary
from ._config import SetupSymexConfig, _require_enabled
from ._emit import emit_python
from ._emit_selfcheck import _eval_emitted_on_seed, check_emit_self_consistency
from ._entry_state import derive_window_mem_live_in, derive_window_symbolic_regs, seed_entry_state
from ._hybrid import classify_hybrid_step
from ._lint import lint_case_config
from ._mem_backing import audit_address_closure, check_mem_backing
from ._mode import estimate_opacity, pick_mode
from ._parity import ParityVector, check_parity_vectors
from ._plan import Checkpoint, build_setup_symex_plan


@dataclass(frozen=True, slots=True)
class CaseConfig:
    """The ONLY thing the agent fills per case — target-specific, never in engine."""

    target:           str
    input_hash:       str
    run_id:           str
    seed_hint_addr:   int
    sink_hint_addr:   int
    entry_pc:         int
    window:           tuple[int, int]
    reg_file:         tuple[str, ...]
    inputs:           tuple[str, ...]          # emit input names
    parity_min:       int = 8
    # How ``window`` is bounded across ALL window-consuming steps in :func:`drive`.
    # "pc" (default, back-compat) — inclusive ``(pc_lo, pc_hi)`` address band.
    # "idx" — inclusive ``(idx_lo, idx_hi)`` trace-execution-order band. A VMP
    # handler window is an execution segment, NOT an address range (the handler
    # jumps out of its address band and the PC band would pull in other
    # occurrences → empty/wrong window). drive threads this basis to every window
    # step so they share ONE basis (mixing pc/idx re-creates the empty-window bug).
    window_kind:      str = "pc"
    nonce:            str | None = None
    symbolic_regs:    tuple[str, ...] | None = None
    pointed_buffers:  tuple[tuple[int, int], ...] = ()
    concrete_backing: ConcreteBacking | None = None
    # The memory arm of the input: external memory regions the agent decided to
    # SYMBOLIZE (vs back) — each (addr, size, concrete_shadow). The symbolize-vs-
    # back judgment is the agent's (a checkpoint); drive surfaces an unpinned
    # mem live-in as an advisory note rather than guessing.
    symbolic_mem:     tuple[tuple[int, int, int], ...] = ()
    seed_located_via: str = LOCATED_WATCH
    sink_located_via: str = LOCATED_SINK_VALIDATION
    task:             str = ""                 # label used as the ledger subject


@dataclass(frozen=True, slots=True)
class DrivePause:
    """drive paused at a Checkpoint needing the agent's judgment.

    Re-invoke :func:`drive` with the decision filled into ``decisions`` (or pass
    an ``on_checkpoint`` resolver) to continue. Checkpoints are NOT auto-decided —
    that is the whole point of surfacing them."""

    checkpoint:      Checkpoint
    pending:         tuple[str, ...]
    completed_steps: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "paused_at":       self.checkpoint.name,
            "checkpoint":      self.checkpoint.to_dict(),
            "pending":         list(self.pending),
            "completed_steps": list(self.completed_steps),
            "kind":            "setup_symex_drive_pause",
        }


@dataclass(frozen=True, slots=True)
class DriveResult:
    """The outcome of a completed drive pass (the agent reads this / the view)."""

    closed:          bool
    mode:            str
    parity:          str | None
    emitted_F:       str | None
    backing_ok:      bool
    address_closure: dict[str, Any]
    mem_backing:     dict[str, Any]
    per_step:        tuple[dict[str, Any], ...]
    entry_keys:      tuple[str, ...]
    view_path:       str | None
    checkpoints:     dict[str, Any]
    parity_report:   dict[str, Any] | None = None
    unmodeled:       dict[str, Any] | None = None
    decode_audit:    dict[str, Any] | None = None
    self_check:      dict[str, Any] | None = None
    note:            str = ""
    # Opaque-staging Phase 2(i) "+ record-a-line": how many loads the runner
    # forwarded (left symbolic on a staging hit) this run. 0 when no symex ran or no
    # staging interval was hit. Purely observational — never feeds a verdict gate.
    symbolic_forwards: int = 0
    # Issue 7 — mem-sink readability. When the runner ran in EXPLICIT mem-sink mode
    # and could NOT read the sink bytes back symbolically (EA never symbolic / read
    # failed / input-invariant store), it surfaces a structured reason here so the
    # recovery layer raises MEM_SINK_UNPLACEABLE / routes an input-invariant store
    # to the seed-independence exclusion — never a silent register/constant fallback.
    # None on the register path (regression guard) / when the sink read succeeded.
    mem_sink_unreadable: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "closed":          self.closed,
            "mode":            self.mode,
            "parity":          self.parity,
            "emitted_F":       self.emitted_F,
            "backing_ok":      self.backing_ok,
            "address_closure": self.address_closure,
            "mem_backing":     self.mem_backing,
            "per_step":        list(self.per_step),
            "entry_keys":      list(self.entry_keys),
            "view_path":       self.view_path,
            "checkpoints":     self.checkpoints,
            "parity_report":   self.parity_report,
            "unmodeled":       self.unmodeled,
            "decode_audit":    self.decode_audit,
            "self_check":      self.self_check,
            "note":            self.note,
            "symbolic_forwards": self.symbolic_forwards,
            "mem_sink_unreadable": self.mem_sink_unreadable,
            "kind":            "setup_symex_drive_result",
        }


def _parse_parity(text: Any, default_n: int) -> tuple[int, int]:
    """Parse a ``"m/n"`` gold-parity string → (m, n). Tolerant of None / junk."""
    try:
        m, n = (int(x) for x in str(text).split("/", 1))
        return m, n
    except (ValueError, AttributeError):
        return 0, int(default_n)


def _parity_vectors_from_run(
    run: Mapping[str, Any], parity: Any, default_n: int,
) -> list[ParityVector]:
    """Build the cross-run parity vectors the multi-vector gate scores.

    Prefers the runner's explicit ``parity_vectors`` (each carrying its own
    ``exec_id`` / observed / predicted — full determinism provenance). Falls back
    to synthesizing distinct vectors from the scalar ``"m/n"`` gold parity: ``m``
    matching + ``n-m`` mismatching, no exec_id (determinism unverifiable, but a
    1/1 still fails the independent floor — which is the actual gap)."""
    raw = run.get("parity_vectors")
    if raw:
        out: list[ParityVector] = []
        for i, v in enumerate(raw):
            out.append(ParityVector(
                input_key=str(v.get("input_key", v.get("input", i))),
                observed=str(v.get("observed", "")),
                predicted=str(v.get("predicted", "")),
                exec_id=(str(v["exec_id"]) if v.get("exec_id") is not None else None),
                derived_from=bool(v.get("derived_from", False))))
        return out
    m, n = _parse_parity(parity, default_n=default_n)
    return [ParityVector(input_key=f"gold-{i}", observed="1",
                         predicted=("1" if i < m else "0"))
            for i in range(max(n, 0))]


def _cohort_parity_vectors(
    *,
    cohort_traces: "Sequence[Sequence[Instruction]] | None",
    cohort_keys: "Sequence[str] | None",
    triton_runner: Callable[[dict[str, Any]], Mapping[str, Any]],
    entry: "EntryStateSpec",
    mode_value: str,
    window: tuple[int, int],
    window_kind: str,
    decisions: Mapping[str, Any],
    emitted_F: str,
    inputs: Sequence[str],
    mem_sink: "Mapping[str, Any] | None" = None,
) -> list[ParityVector]:
    """Build one REAL cross-run ParityVector per cohort vector (the feed leg).

    The cohort→parity feed (todo/dev-cohort-parity-feed-leg.md). For each cohort
    trace the runner is run ONCE more (same entry / window / window_kind /
    decisions, only the ``items`` swapped to that cohort trace); the resulting
    ``trace_self_check`` carries the vector's *real* facts:

      * ``observed`` = ``run_v["trace_self_check"]["sink_value"]`` — the window's
        true exit output for THAT input, computed by the runner's live oracle
        (invariant 8: a real oracle run, never synthesized).
      * ``predicted`` = ``emitted_F`` evaluated on that vector's
        ``trace_self_check["seed_values"]`` via :func:`_eval_emitted_on_seed`
        (invariant 8: a real eval of the recovered F on the vector's own seed).

    Each vector carries its OWN ``exec_id`` (the runner's, else a stable
    ``cohort-k:key`` synthesised id) so the determinism gate verifies no two
    distinct inputs share one execution (mixing). A cohort vector whose runner
    produced no usable sink/seed facts (no ``trace_self_check`` sink, or an
    un-evaluable F) is SKIPPED — it cannot be made up (invariant 8); the feed is
    best-effort and a runner error on one vector never aborts the others.

    Invariant 7: an empty / missing ``cohort_traces`` returns ``[]`` — the caller
    then behaves byte-for-byte as today (main trace + fallback only)."""
    traces = list(cohort_traces or ())
    if not traces:
        return []
    keys = list(cohort_keys or ())
    out: list[ParityVector] = []
    for k, ct in enumerate(traces):
        items_v = list(ct or ())
        if not items_v:
            continue
        input_key = str(keys[k]) if k < len(keys) and keys[k] is not None \
            else f"cohort-{k}"
        try:
            _ctx_v = {
                "entry": entry.to_dict(), "mode": mode_value,
                "window": list(window), "window_kind": window_kind,
                "items": items_v, "decisions": dict(decisions)}
            # Issue 7 — same mem-sink descriptor on each cohort run so observed is
            # the store's bytes for THAT input (bytewise cross-vector parity).
            if mem_sink is not None:
                _ctx_v["output_mem"] = dict(mem_sink)
            run_v = dict(triton_runner(_ctx_v))
        except Exception:
            # A runner failure on ONE cohort vector must not lose the others or the
            # main verdict — skip this vector (best-effort feed), never fabricate.
            continue
        tsc = run_v.get("trace_self_check")
        tsc = dict(tsc) if isinstance(tsc, Mapping) else {}
        observed_v = tsc.get("sink_value")
        seed_v = tsc.get("seed_values")
        if observed_v is None or seed_v is None:
            # No real oracle facts for this vector → cannot build a TRUE vector;
            # skip rather than invent (invariant 8).
            continue
        ok, value, _why = _eval_emitted_on_seed(emitted_F, inputs, dict(seed_v))
        if not ok:
            # emitted_F did not evaluate on this vector's own seed → no honest
            # predicted; skip (the self-check / main-trace path still reports F).
            continue
        # Per-vector exec_id: the runner's if it surfaced one, else a stable id
        # fingerprinting this cohort vector's input — distinct per input so the
        # determinism gate sees no two distinct inputs sharing one execution.
        exec_id_v = run_v.get("exec_id")
        exec_id_v = str(exec_id_v) if exec_id_v is not None \
            else f"cohort-{k}:{input_key}"
        out.append(ParityVector(
            input_key=input_key,
            observed=str(observed_v),
            predicted=str(value),
            exec_id=exec_id_v))
    return out


def _collect_staging_intervals(
    staging_diag: Any,
    items: Sequence[Instruction],
    window: tuple[int, int],
    window_is_idx: bool,
    pointer_chain: Any | None,
) -> list[tuple[int, int]]:
    """Collect the staging ``(addr, size)`` intervals to forward for a
    ``symbolic_address`` diagnosis (Phase 2(i) injection set).

    Shared by drive's FIRST-run ``clo_deferred`` injection and the opaque-staging
    re-續 fallback re-run, so both paths build the SAME set from one place.
    Backbone (case-agnostic, always present): each ``symbolic_address`` target
    load's actual trace ``(op.addr, op.size)`` (rw=="r", size>0, de-duped).
    Enhancement (only when a ``pointer_chain`` is supplied): the chain's STORE-side
    landing via :func:`resolve_staging_address`, merged + de-duped.

    Returns ``[]`` when the verdict is NOT ``symbolic_address`` or no target load
    has a read MemOp — the empty set keeps the entry unchanged (invariant 7)."""
    from ..opaque_staging import resolve_staging_address, VERDICT_SYMBOLIC_ADDRESS
    injected: list[tuple[int, int]] = []
    if staging_diag is None or staging_diag.verdict != VERDICT_SYMBOLIC_ADDRESS:
        return injected
    seen_iv: set[tuple[int, int]] = set()
    target_idxs = {bl.idx for bl in staging_diag.blind_loads}
    for ins in items:
        if ins.idx not in target_idxs:
            continue
        for op in ins.mem:
            if op.rw != "r":
                continue
            iv = (int(op.addr), int(op.size))
            if iv[1] > 0 and iv not in seen_iv:
                seen_iv.add(iv)
                injected.append(iv)
    # optional pointer-chain enhancement (store-side landing), merged + de-duped.
    if pointer_chain is not None:
        for iv in resolve_staging_address(
                items, pointer_chain, window=window, window_is_idx=window_is_idx):
            iv = (int(iv[0]), int(iv[1]))
            if iv[1] > 0 and iv not in seen_iv:
                seen_iv.add(iv)
                injected.append(iv)
    return injected


@dataclass(frozen=True, slots=True)
class _SymexRunResult:
    """The locals one symex+emit+G4+parity pass produces (re-entrant helper out).

    Holds exactly the values the inline symex block used to leave behind, so a
    caller assigns them to the same names and the downstream note/close logic is
    byte-for-byte unchanged. The opaque-fallback re-run (Phase 2(i) re-續) calls
    the same helper a second time with an injected entry and overwrites these."""

    parity_ok:         bool
    emitted_F:         str | None
    parity:            str | None
    parity_vreport:    ParityVectorReport | None
    self_check:        EmitSelfCheckReport | None
    unmodeled:         dict[str, Any] | None
    symbolic_forwards: int
    ran:               bool   # did the gated symex block actually fire this pass
    mem_sink_unreadable: str | None = None   # Issue 7 — runner could not read the sink bytes


def drive(
    *,
    trace: Iterable[Instruction],
    case_config: CaseConfig,
    triton_runner: Callable[[dict[str, Any]], Mapping[str, Any]],
    ledger: Any = None,
    decisions: Mapping[str, Any] | None = None,
    on_checkpoint: Callable[["Checkpoint"], Any] | None = None,
    cfg: SetupSymexConfig | None = None,
    ts: str | None = None,
    pointer_chain: "Any | None" = None,
    cohort_traces: "Sequence[Sequence[Instruction]] | None" = None,
    cohort_keys: "Sequence[str] | None" = None,
    mem_sink: "Mapping[str, Any] | None" = None,
) -> "DriveResult | DrivePause":
    """Execute the set-up symex plan end to end (Level 1 thin orchestrator).

    Returns a :class:`DriveResult` on completion, or a :class:`DrivePause` when a
    Checkpoint needs the agent's judgment and neither ``decisions`` nor
    ``on_checkpoint`` supplied it. ``triton_runner`` is called for the symbolic
    step (Level 1 = agent-provided) and returns ``{propagated, expr_source,
    gold_parity}``. ``ledger`` is a :mod:`engine.cvd_ledger` connection; when
    given, drive lands the durable findings (emit + one run-summary roll-up,
    per the recording policy) and refreshes the stamped view.

    ``cohort_traces`` (optional, the verifier transparently forwards its
    ``self.cohort_traces``) feeds the cross-run parity gate: for each cohort
    vector the runner is run once more (same entry/window, items swapped) and the
    result becomes one REAL :class:`ParityVector` (observed = the vector's true
    window-exit sink, predicted = ``emitted_F`` on the vector's own seed). Empty /
    missing → byte-for-byte today's behaviour (main trace + fallback only;
    invariant 7). ``cohort_keys`` (parallel) labels each vector's ``input_key``.

    ``mem_sink`` (Issue 7, optional — spec_f0_mem_write_window_sink.md) is the
    EXPLICIT mem-write recovery sink descriptor ``{"sink_addr", "sink_size",
    "sink_idx"}``: the window's OUTPUT is a memory write (a store), not a register.
    When present, drive forwards it to the runner ctx as ``output_mem`` (the
    Level-2 ``TritonStepDecoder`` then reads the symbolic bytes of ``[sink_addr,
    sink_addr+sink_size)`` after the window and emits a byte-list F instead of an
    x8 value) on BOTH the main and the per-cohort parity runs, so observed/
    predicted are compared bytewise. ``None`` → the register path, byte-for-byte
    today (the regression guard)."""
    cfg = cfg or SetupSymexConfig.from_env()
    _require_enabled(cfg)
    from .. import cvd_ledger as _cl

    cc = case_config
    items = list(trace)
    exec_identity = _cl.ExecIdentity(
        target=cc.target, input_hash=cc.input_hash, run_id=cc.run_id, nonce=cc.nonce)
    trace_exec_id = exec_identity.ref
    backing = (replace(cc.concrete_backing, exec_id=trace_exec_id)
               if cc.concrete_backing is not None else None)
    decisions = dict(decisions or {})
    plan = build_setup_symex_plan()
    per_step: list[dict[str, Any]] = []
    completed: list[str] = []

    def _pending() -> tuple[str, ...]:
        return tuple(c.name for c in plan.checkpoints if c.name not in decisions)

    def _resolve(cp: Checkpoint) -> bool:
        if cp.name in decisions:
            return True
        if on_checkpoint is not None:
            decisions[cp.name] = on_checkpoint(cp)
            return True
        return False

    # 1 — locate_boundary (provenance, no typed addr); transient, not recorded.
    locate_boundary(seed_hint_addr=cc.seed_hint_addr, sink_hint_addr=cc.sink_hint_addr, cfg=cfg)
    bind_boundary(BoundaryEnd(BoundaryRole.SEED, cc.seed_hint_addr, cc.seed_located_via))
    bind_boundary(BoundaryEnd(BoundaryRole.SINK, cc.sink_hint_addr, cc.sink_located_via))
    completed.append("locate_boundary")
    per_step.append({"step": "locate_boundary", "ok": True})

    # 1.25 — PARITY/SELF-CHECK WIRING LINT (spec #3): a pure pre-flight,
    # NON-BLOCKING lint of the case-config / parity inputs (declared inputs vs
    # window, sink source, observed semantics, sink-mask width) over the SAME
    # descriptor fields the self-check / parity gates consume — caught by the
    # tool, not by a human reading the script. Loud (WARN/ERROR) findings ride
    # above the run in ``per_step``; an OK / INFO-only report (the degenerate
    # "runner facts not in yet" pre-flight case) appends NOTHING, so the lint-OK
    # path leaves drive() byte-for-byte unchanged. Never aborts the run.
    _lint_report = lint_case_config(cc)
    if _lint_report.max_level in ("WARN", "ERROR"):
        per_step.append({"step": "parity_wiring_lint",
                         "max_level": _lint_report.max_level,
                         "findings": [f.to_dict() for f in _lint_report.findings
                                      if f.level in ("WARN", "ERROR")]})

    # 1.5 — DECODE-FEED GUARD (§2): cross-check Triton's decode coverage over the
    # window against capstone (the oracle), BEFORE any analysis checkpoint. A bulk
    # of Triton decode failures — especially any that capstone DOES decode — is a
    # byte-feed config bug (endianness/arch/slice), NOT an escape-hatch
    # (un-modeled-opcode) scenario. When systematic the window's bytes are not
    # verified, so we MUST NOT ask the agent analysis judgments (mem disposition /
    # alias / static) about it — verify the foundation (bytes) before asking the
    # upper layers. The three analysis checkpoints below are guarded on
    # ``not decode_systematic``; a systematic feed falls straight through to the
    # BLOCK + fix-the-feed note path (no symex/escape-hatch). Inputs ``items`` /
    # ``cc.window`` / ``cc.window_kind`` are ready at the drive entry.
    from ..setup_symex_runner import audit_window_decode
    _audit = audit_window_decode(items, window=cc.window, window_kind=cc.window_kind)
    decode_audit: dict[str, Any] | None = _audit.to_dict() if _audit.total else None
    decode_systematic = _audit.systematic
    completed.append("decode_feed_guard")
    per_step.append({"step": "decode_feed_guard", "systematic": decode_systematic,
                     "feed_mismatch": _audit.feed_mismatch,
                     "fail_rate": _audit.fail_rate})

    # 2 — seed_entry_state (full reg_file + concrete backing arm).
    # Auto-derive per-handler symbolic inputs (window live-in) when the agent
    # did not hand-config them — utov configures, the agent does not
    # run-once-look-once. A non-empty hand-supplied set is an explicit override
    # and is honoured as-is; empty/None → auto-seed from the window's live-in.
    seed_regs = cc.symbolic_regs
    auto_seed_info: dict[str, Any] | None = None
    seed_block_note: str | None = None
    if not seed_regs:
        derived, auto_seed_info = derive_window_symbolic_regs(
            items, window=cc.window, reg_file=cc.reg_file,
            window_is_idx=(cc.window_kind == "idx"))
        # The C2 split: a live-in register that is concretely BACKED is a pinned
        # pointer base, not a symbolic input — exclude it (seed_entry_state would
        # otherwise reject the symbolize∩back overlap). An empty live-in window
        # falls to the full reg_file under the same split (still C2-complete).
        backed = set(backing.backed_regs) if backing is not None else set()
        basis = derived or tuple(cc.reg_file)
        seed = tuple(r for r in basis if r not in backed)
        if not seed and backed:
            # Every live-in register is concretely backed — the live-in is fully
            # consumed by the backing split. Do NOT fall back to the full reg_file
            # (passing symbolic_regs=None would re-symbolize the backed regs and
            # clash with the backing — the auto-seed edge bug). Re-derive the seed
            # from the full reg_file minus the backed set instead.
            seed = tuple(r for r in dict.fromkeys(cc.reg_file) if r not in backed)
            if seed:
                auto_seed_info["seed_from_reg_file_minus_backed"] = True
        if not seed:
            # reg_file − backed is empty too: nothing symbolic is left to seed. This
            # is a real config gap (the whole entry state is pinned concrete) — BLOCK
            # with a note, never silently fall back to the full reg_file (that would
            # symbolize the very registers the backing pins, contradicting C2).
            seed_block_note = (
                "auto-seed found no symbolic input: every live-in register is "
                f"concretely backed ({sorted(backed)}) and reg_file - backed is "
                "empty — the entry state is fully pinned, so the symbolic chain "
                "cannot start. NOT falling back to the full reg_file (that would "
                "symbolize backed pointer bases and clash with the backing). Widen "
                "reg_file, or move a real input register out of concrete_backing.")
            auto_seed_info["seed_blocked"] = True
        # When blocked, seed nothing EXPLICITLY (empty tuple, not None): None would
        # make seed_entry_state symbolize the full reg_file and clash with the
        # backing — the very edge bug. An empty seed symbolizes nothing and the
        # drive short-circuits on seed_block_note below.
        seed_regs = seed if seed else (() if seed_block_note is not None else None)
        if backed:
            auto_seed_info["backed_excluded"] = [r for r in basis if r in backed]
    # Memory arm of the input (the F0 handler11 symbolic=0 root cause): a value
    # entering through ``ldr`` is invisible to register live-in. Detect external
    # memory inputs (A1, mechanical); the disposition of each — symbolize (a new
    # input that ARRIVED) vs back (a state carrier / pointer-table base that PASSES
    # THROUGH) — is the agent's judgment (A2), surfaced as the named checkpoint
    # ``mem_input_symbolize_vs_back`` and NEVER auto-guessed.
    mem_live_in, mem_info = derive_window_mem_live_in(
        items, window=cc.window, window_is_idx=(cc.window_kind == "idx"))
    backed_addrs = backing.backed_addrs if backing is not None else frozenset()
    pinned_addrs = {a for (a, _s, _v) in cc.symbolic_mem} | set(backed_addrs)

    def _mem_disp(addr: int) -> Any:
        md = decisions.get("mem_input_symbolize_vs_back") or {}
        return md.get(addr, md.get(f"0x{addr:x}"))

    undecided = [m for m in mem_live_in
                 if m.addr not in pinned_addrs and _mem_disp(m.addr) is None]
    if undecided:
        mem_cp = Checkpoint(
            name="mem_input_symbolize_vs_back",
            question=(
                "un-pinned external memory input(s) — for each, decide SYMBOLIZE "
                "(a new input that arrived: carrier byte / plaintext / key material) "
                "vs BACK (a state carrier threaded from the previous handler, or a "
                "constant-table base). Items: "
                + "; ".join(
                    f"0x{m.addr:x}+{m.size}@idx{m.src_idx} via {list(m.base_regs)}"
                    for m in undecided)),
            why=("register live-in cannot see a value that enters through ldr; left "
                 "un-symbolized it stays concrete 0 and the window exit collapses "
                 "(the handler11 symbolic=0 trap). Cache the decision per handler "
                 "type once made (same type + offset → auto-apply, not re-asked)."))
        # Feed guard: only ask the agent's mem-disposition judgment when the bytes
        # are verified. A systematic feed bug means this window should not be
        # analysed at all — fall through (no pause) to the fix-the-feed BLOCK below.
        if not decode_systematic:
            if on_checkpoint is not None:
                decisions["mem_input_symbolize_vs_back"] = on_checkpoint(mem_cp)
            elif "mem_input_symbolize_vs_back" not in decisions:
                return DrivePause(checkpoint=mem_cp, pending=_pending(),
                                  completed_steps=tuple(completed))
    # Apply the disposition: symbolize entries extend the effective symbolic_mem;
    # back entries are pinned via concrete_backing (the agent supplies it).
    eff_sym_mem: list[tuple[int, int, int]] = list(cc.symbolic_mem)
    decided_back: list[int] = []
    for m in mem_live_in:
        if m.addr in pinned_addrs:
            continue
        d = _mem_disp(m.addr)
        if d == "back":
            decided_back.append(m.addr)
        elif d is not None:
            val = int(d["symbolize"]) if isinstance(d, Mapping) else int(d)
            eff_sym_mem.append((m.addr, m.size, val))
    sym_mem_addrs = {a for (a, _s, _v) in eff_sym_mem}
    unpinned_mem = [m for m in mem_live_in
                    if m.addr not in sym_mem_addrs and m.addr not in backed_addrs
                    and m.addr not in decided_back]
    mem_info["symbolized"] = sorted(sym_mem_addrs)
    mem_info["decided_back"] = sorted(decided_back)
    mem_info["unpinned"] = [m.to_dict() for m in unpinned_mem]
    entry = seed_entry_state(
        entry_pc=cc.entry_pc, reg_file=cc.reg_file, pointed_buffers=cc.pointed_buffers,
        symbolic_regs=seed_regs, concrete_backing=backing,
        symbolic_mem=tuple(eff_sym_mem), cfg=cfg)
    completed.append("seed_entry_state")
    _seed_step: dict[str, Any] = {
        "step": "seed_entry_state", "symbolic_regs": list(entry.symbolic_regs)}
    if auto_seed_info is not None:
        _seed_step["auto_seed"] = auto_seed_info
    if not mem_info["empty"]:
        _seed_step["mem_live_in"] = mem_info
    per_step.append(_seed_step)

    # 3 — pick_mode (encoded switch; sym not yet propagated pre-run).
    mode = pick_mode(estimate_opacity(items, sym_propagated=False), cfg=cfg)
    completed.append("pick_mode")
    per_step.append({"step": "pick_mode", "mode": mode.mode.value})

    # 4 — backing GATE: check_mem_backing + audit_address_closure (unified, guarded).
    _win_idx = cc.window_kind == "idx"
    back = check_mem_backing(items, window=cc.window, window_is_idx=_win_idx,
                             backing=backing, trace_exec_id=trace_exec_id)
    clo = audit_address_closure(items, window=cc.window, window_is_idx=_win_idx,
                                backing=backing, trace_exec_id=trace_exec_id)
    # §5′ DYNAMIC backing gate: the enter-symex BLOCK criterion is the DYNAMIC
    # (concolic) view — ``back.sufficient`` alone (every mem-class step in the
    # window has its EA + value in the trace, via a mem[] operand, a snapshot/hook
    # closure, OR the register side). The static address-closure ``clo`` is NO
    # LONGER hard-AND'd into the gate: a symbolic / computed EA whose op.addr+value
    # ARE in the trace is exactly the P2(i) forwarding frontier (level3 landed) —
    # concolic symex walks the concrete address, so a symbolic static closure is
    # irrelevant to "can it enter symex". Only a TRULY blind leg (op.addr AND value
    # both absent) leaves ``back.sufficient`` False → real missing backing → BLOCK.
    backing_ok = back.sufficient
    completed.append("check_mem_backing")
    per_step.append({"step": "check_mem_backing", "sufficient": back.sufficient,
                     "closure_sufficient": clo.sufficient, "blind_count": len(back.blind_pcs)})

    # 4b — symbolic-EA DIAGNOSIS (§5′): for a dynamically-backed window whose STATIC
    # closure is un-backed (``clo.sufficient`` False — a symbolic / computed EA),
    # annotate WHY (symbolic_address / known_addr / staging) for the gap map and the
    # agent. This is INFORMATIONAL ONLY — it does NOT gate "does it enter symex"
    # (``backing_ok`` already decided that on the dynamic view above). It tells the
    # agent the EA is staged behind the symbol and is deferred to P2(i) forwarding —
    # do NOT re-capture. A truly blind window (back.sufficient False) is handled by
    # the BLOCK + re-capture note path, not here.
    staging_diag = None
    clo_deferred = back.sufficient and not clo.sufficient
    if clo_deferred:
        # local import: opaque_staging imports audit_address_closure from this
        # module (circular at module top), so defer the import to here.
        from ..opaque_staging import diagnose_opaque_staging, derive_pointer_chain
        staging_diag = diagnose_opaque_staging(
            items, window=cc.window, window_is_idx=_win_idx,
            pointer_chain=pointer_chain,
            symbolic_inputs=tuple(entry.symbolic_regs), cohort_traces=())
        # Self-produced pointer-chain SHAPE: when the caller did NOT supply one,
        # derive the store/load base-register shape from THIS diagnosis (坎3) so the
        # store-side narrow (resolve_staging_address) is not永远 None. The caller's
        # explicit pointer_chain always wins (optional override); derive→None keeps
        # the backbone-only forward unchanged (invariant 7/8).
        pc_eff = pointer_chain or derive_pointer_chain(
            staging_diag, items, window=cc.window, window_is_idx=_win_idx)
        # P2(i) FEED LINE: a symbolic-address window's staging load is BACKED (the
        # trace has a concrete value) yet should carry the symbol — forwarding only
        # happens if the runner's _symbolic_staging covers the load's landing bytes.
        # Inject those intervals into the entry BEFORE the symex call below, so the
        # FIRST run forwards instead of collapsing to an opaque frontier. The interval
        # set is collected by the shared ``_collect_staging_intervals`` helper (reused
        # by the opaque-staging re-續 fallback below). verdict != symbolic_address / no
        # target load → empty set → entry unchanged (invariant 7: byte-for-byte). This
        # injection touches NO close/parity/G4 gate — it only gives the symbolic load a
        # chance to forward; the existing gates still decide whether the window closes
        # (a truly input-varying EA still BLOCKs → opaque frontier, see the degrade).
        injected = _collect_staging_intervals(
            staging_diag, items, cc.window, _win_idx, pc_eff)
        if injected:
            entry = replace(entry, symbolic_staging=tuple(injected))
        per_step.append({"step": "symbolic_addressing_gate",
                         "deferred": True,
                         "staging_verdict": staging_diag.verdict,
                         "symbolic_staging_injected": len(injected)})

    # 5 — CHECKPOINT alias_vs_compute (agent's judgment). Feed guard: only pause
    # for this judgment when the bytes are verified; a systematic feed bug falls
    # through (no pause) to the fix-the-feed BLOCK.
    cp1 = next(c for c in plan.checkpoints if c.name == "alias_vs_compute")
    if not decode_systematic and not _resolve(cp1):
        return DrivePause(checkpoint=cp1, pending=_pending(), completed_steps=tuple(completed))
    completed.append("alias_vs_compute")

    # 6 — classify_hybrid_steps (symbol-preserving); transient, not recorded.
    sym = set(entry.symbolic_regs)
    _wlo, _whi = cc.window
    hybrids = [classify_hybrid_step(ins, symbolic_regs=sym)
               for ins in items
               if _wlo <= (ins.idx if _win_idx else ins.pc) <= _whi and ins.mem][:16]
    completed.append("classify_hybrid_steps")
    per_step.append({"step": "classify_hybrid_steps", "n": len(hybrids),
                     "must_symbolize": sum(1 for h in hybrids if h.must_symbolize)})

    # 7 — CHECKPOINT which_static (agent's judgment). Feed guard: only pause for
    # this judgment when the bytes are verified; a systematic feed bug falls
    # through (no pause) to the fix-the-feed BLOCK.
    cp2 = next(c for c in plan.checkpoints if c.name == "which_static")
    if not decode_systematic and not _resolve(cp2):
        return DrivePause(checkpoint=cp2, pending=_pending(), completed_steps=tuple(completed))
    completed.append("which_static")

    # symex + 8 emit — ONLY if the backing gate held AND the decode feed is clean
    # AND the auto-seed found a symbolic input. No bypass: a blind closure / a
    # systematic feed bug / a fully-pinned seed would make forward symex emit a
    # passthrough stub, so short-circuit honestly.
    #
    # RE-ENTRANT (Phase 2(i) re-續): the symex+emit+G4+parity pass is a closure
    # ``_run_symex_once(entry)`` so the opaque-staging fallback below can re-run it
    # ONCE with an injected entry. The FIRST call below is byte-for-byte identical to
    # the old inline block (it reads the same locals from the enclosing scope, appends
    # the same per_step/completed entries, and returns the same values, which are
    # assigned back to the same names). It touches NO close/parity/G4 gate.
    def _run_symex_once(entry: EntryStateSpec) -> _SymexRunResult:
        emitted_F: str | None = None
        parity: str | None = None
        parity_ok = False
        parity_vreport: ParityVectorReport | None = None
        self_check: EmitSelfCheckReport | None = None
        unmodeled: dict[str, Any] | None = None
        # Opaque-staging Phase 2(i) "+ record-a-line": how many loads the runner
        # actually forwarded (left symbolic on a staging hit) this run. Defaults 0
        # (no symex / no staging hit) — purely observational, never read by a gate.
        symbolic_forwards = 0
        mem_sink_unreadable: str | None = None
        ran = False
        if backing_ok and not decode_systematic \
                and seed_block_note is None:
            ran = True
            _main_ctx = {
                "entry": entry.to_dict(), "mode": mode.mode.value,
                "window": list(cc.window), "window_kind": cc.window_kind,
                "items": items, "decisions": dict(decisions)}
            # Issue 7 — forward the EXPLICIT mem-sink descriptor so the runner reads
            # the store's symbolic bytes (mem output), not x8. Absent → unchanged.
            if mem_sink is not None:
                _main_ctx["output_mem"] = dict(mem_sink)
            run = dict(triton_runner(_main_ctx))
            # Opaque-staging Phase 2(i) "+ record-a-line": surface how many loads the
            # runner actually FORWARDED (left symbolic on a staging hit) this run, next
            # to ``propagated``. With per_step's symbolic_staging_injected this gives
            # the full "injected N intervals → forwarded M loads" picture (a tc4
            # half-wired signal is injected > 0 with forwarded == 0). Observational.
            symbolic_forwards = int(run.get("symbolic_forwards", 0) or 0)
            # Issue 7 — mem-sink readability: the runner ran in EXPLICIT mem-sink
            # mode but could not read the sink bytes back symbolically (EA never
            # symbolic / read failed / input-invariant store). Carry the structured
            # reason so the recovery verifier raises MEM_SINK_UNPLACEABLE / routes an
            # input-invariant store to the seed-independence exclusion — never a
            # silent register/constant fallback (A8④). None on the register path.
            mem_sink_unreadable = run.get("mem_sink_unreadable") or None
            per_step.append({"step": "symex", "propagated": run.get("propagated"),
                             "symbolic_forwards": symbolic_forwards})
            completed.append("symex")
            # Level-2 escape hatch: the runner hit an un-modeled instruction and
            # surfaced a precise checkpoint instead of force-concretizing. Carry it.
            u = run.get("unmodeled")
            unmodeled = dict(u) if isinstance(u, Mapping) else None
            parity = run.get("gold_parity")
            expr_source = str(run.get("expr_source", "")).strip()
            if expr_source:
                emit = emit_python(mode=mode.mode, expr_source=expr_source,
                                   inputs=cc.inputs, parity_min=cc.parity_min)
                emitted_F = emit.expr_source
                m, _n = _parse_parity(parity, default_n=cc.parity_min)
                # G4 SELF-CHECK (necessary, BEFORE the cross-vector parity gate): the
                # recovered F, evaluated on the trace's OWN concrete seed values, must
                # equal the trace's concrete sink at the window exit. The runner
                # surfaces those trace facts (concrete seed per input + exit sink) in
                # ``trace_self_check`` — reliable even when its propagation is buggy.
                # No facts / un-evaluable F → INCONCLUSIVE (surfaced, never silent).
                tsc = run.get("trace_self_check")
                tsc = dict(tsc) if isinstance(tsc, Mapping) else {}
                self_check = check_emit_self_consistency(
                    expr_source=emitted_F, inputs=cc.inputs,
                    seed_values=(tsc.get("seed_values")
                                 if tsc.get("seed_values") is not None else None),
                    trace_sink=tsc.get("sink_value"),
                    sink_mask=tsc.get("sink_mask"),
                    sink_form=str(tsc.get("sink_form", "reg")))
                completed.append("emit_self_check")
                per_step.append({"step": "emit_self_check",
                                 "status": self_check.status,
                                 "f_on_trace": self_check.f_on_trace,
                                 "trace_sink": self_check.trace_sink})
                # Multi-vector gate: EXACT needs >= parity_min_vectors INDEPENDENT
                # cross-run vectors (never the deriving trace), each self-consistent
                # within its own execution. A tautological 1/1 → BLOCK. The scalar
                # match floor (m >= parity_min) still applies on top.
                # cohort→parity FEED LEG (todo/dev-cohort-parity-feed-leg.md): the
                # main trace alone yields only the runner's own parity_vectors (often
                # just the tautological 1/1) → supplied < need → a feed pit, not an
                # F-error. Run the runner once per cohort vector (Triton-gated, same
                # entry/window) to build REAL cross-run vectors (observed = the
                # vector's true exit sink, predicted = emitted_F on its own seed).
                # No cohort / empty → [] → supplied byte-for-byte today (invariant 7).
                cohort_vecs = _cohort_parity_vectors(
                    cohort_traces=cohort_traces, cohort_keys=cohort_keys,
                    triton_runner=triton_runner, entry=entry,
                    mode_value=mode.mode.value, window=cc.window,
                    window_kind=cc.window_kind, decisions=decisions,
                    emitted_F=emitted_F, inputs=cc.inputs, mem_sink=mem_sink)
                parity_vreport = check_parity_vectors(
                    _parity_vectors_from_run(run, parity, default_n=cc.parity_min)
                    + cohort_vecs,
                    window=cc.window, min_vectors=cfg.parity_min_vectors,
                    trace_exec_id=trace_exec_id)
                # The self-check is necessary: a BLOCK here forbids close regardless
                # of parity (F that fails to reproduce its own trace cannot be EXACT).
                parity_ok = ((m >= emit.parity_min) and parity_vreport.sufficient
                             and not self_check.blocked)
                completed.append("emit_python")
                per_step.append({"step": "emit_python", "parity": parity,
                                 "parity_ok": parity_ok,
                                 "parity_vectors": parity_vreport.verdict})
        return _SymexRunResult(
            parity_ok=parity_ok, emitted_F=emitted_F, parity=parity,
            parity_vreport=parity_vreport, self_check=self_check,
            unmodeled=unmodeled, symbolic_forwards=symbolic_forwards, ran=ran,
            mem_sink_unreadable=mem_sink_unreadable)

    _sx = _run_symex_once(entry)
    emitted_F = _sx.emitted_F
    parity = _sx.parity
    parity_ok = _sx.parity_ok
    parity_vreport = _sx.parity_vreport
    self_check = _sx.self_check
    unmodeled = _sx.unmodeled
    symbolic_forwards = _sx.symbolic_forwards
    mem_sink_unreadable = _sx.mem_sink_unreadable
    # CLOSE soundness (§5′): a dynamically-backed window whose static closure is
    # un-backed (symbolic / computed EA) may CLOSE only by the SAME chain every
    # other window uses — ``parity_ok`` (which itself requires P2(i) forwarding to
    # succeed + G4 emit self-check not BLOCKED + cross-vector parity). The dynamic
    # gate above only let it ENTER symex; the close gate is UNCHANGED. ``parity_ok``
    # defaults False and is set True only inside the symex/emit block after the
    # self-check + parity-vector gates, so a window whose P2(i) did not resolve has
    # parity_ok=False → not closed → falls through to the opaque branch. There is no
    # ``clo``-based relaxation here: closing rides parity_ok exactly as before.
    closed = (backing_ok and parity_ok
              and not decode_systematic and seed_block_note is None)

    # OPAQUE-STAGING re-續 (Phase 2(i) re-續): "hit opaque → forward FIRST, then go
    # on". The FIRST-run injection above only covers the ``clo_deferred`` window
    # (back sufficient + static closure symbolic). A window whose static closure IS
    # backed (``clo_deferred`` False — e.g. a pointer-indirect staging load) never
    # got the first-run injection, so it ran symex blind and collapsed to opaque.
    # When that window TRULY lands opaque (backed, no close, no systematic feed bug,
    # no seed block) AND the FIRST run FORWARDED NOTHING (``symbolic_forwards == 0``,
    # so it is NOT an already-tried real frontier), re-diagnose via the Phase 0b
    # DFG path (does NOT depend on ``clo_deferred``), inject the staging interval(s),
    # and re-run ``_run_symex_once`` ONCE with the injected entry. Four conjuncts must
    # ALL hold (invariant 7) — any miss → no re-run → byte-for-byte unchanged:
    #   (a) not closed AND truly opaque (backed, not systematic, no seed block)
    #   (b) ``symbolic_forwards == 0`` — the first run did NOT forward (not redundant,
    #       not a real frontier we already tried)
    #   (c) diagnose verdict == symbolic_address
    #   (d) the collected interval set is non-empty AND adds bytes the entry does not
    #       already carry (a clo_deferred entry that already holds them is not re-run)
    # The re-run rides the EXACT SAME parity_ok / G4 / cross-vector / seed chain — the
    # close criterion is UNCHANGED; a still-input-varying EA still BLOCKs → frontier
    # (degrade still yields a verdict), now carrying both forward counts as evidence.
    opaque_now = (not closed and backing_ok and not parity_ok
                  and not decode_systematic and seed_block_note is None)
    if opaque_now and symbolic_forwards == 0:
        from ..opaque_staging import diagnose_opaque_staging, derive_pointer_chain
        fb_diag = diagnose_opaque_staging(
            items, window=cc.window, window_is_idx=_win_idx,
            pointer_chain=pointer_chain,
            symbolic_inputs=tuple(entry.symbolic_regs), cohort_traces=())
        # 坎3: caller did not supply a shape → self-derive from THIS diagnosis so the
        # opaque fallback narrows the store side itself (not永远 backbone-only because
        # pointer_chain is None). Caller override wins; derive→None ⇒ backbone-only.
        fb_pc = pointer_chain or derive_pointer_chain(
            fb_diag, items, window=cc.window, window_is_idx=_win_idx)
        fb_intervals = _collect_staging_intervals(
            fb_diag, items, cc.window, _win_idx, fb_pc)
        # only re-run when the diagnosis adds staging bytes the entry does not already
        # carry — a clo_deferred entry already holding them must NOT double-run.
        already = set(entry.symbolic_staging)
        new_iv = [iv for iv in fb_intervals if iv not in already]
        if new_iv:
            entry = replace(entry, symbolic_staging=tuple(
                list(entry.symbolic_staging) + new_iv))
            _rx = _run_symex_once(entry)
            emitted_F = _rx.emitted_F
            parity = _rx.parity
            parity_ok = _rx.parity_ok
            parity_vreport = _rx.parity_vreport
            self_check = _rx.self_check
            unmodeled = _rx.unmodeled
            symbolic_forwards = _rx.symbolic_forwards
            mem_sink_unreadable = _rx.mem_sink_unreadable
            closed = (backing_ok and parity_ok
                      and not decode_systematic and seed_block_note is None)
            per_step.append({"step": "opaque_forward_retry",
                             "injected": len(new_iv),
                             "retry_parity_ok": parity_ok,
                             "retry_symbolic_forwards": symbolic_forwards})

    if decode_systematic:
        # §2: a systematic decode/feed inconsistency is the ROOT cause — it outranks
        # the backing / un-modeled-opcode notes because feeding the symex (and the
        # backing analysis) garbage bytes is what makes them look broken downstream.
        # Fix the byte-feed, not the symex / escape hatch.
        note = _audit.note
    elif not backing_ok:
        # Real missing backing (§5′ TRULY blind: a mem-class step whose op.addr AND
        # value are both absent — no mem[] operand, no snapshot/closure resolve, no
        # register-side value). A symbolic-EA window with op.addr+value present is
        # NOT this — it is dynamically backed (backing_ok True), entered symex, and
        # is judged downstream; only a real blind leg lands here (and in a mixed
        # window a single blind leg keeps the WHOLE window here, by design).
        note = ("backing gate not satisfied (blind address closure) — NOT bypassed; "
                "provide same-execution backing or re-capture before emit")
    elif seed_block_note is not None:
        note = seed_block_note
    elif cc.window_kind == "idx" and mem_info.get("n_window_items", -1) == 0:
        # window_kind=='idx' but the idx band matched no trace step: every window
        # step ran on an EMPTY window (auto-seed/derive/backing/symex all see 0
        # items → emit "0"). This is the root cause, surfaced loudly rather than
        # run silently — almost always an out-of-range / mis-specified idx window.
        note = ("window_kind='idx' but the window matched 0 trace items (idx band "
                f"{list(cc.window)} selected nothing) — likely an out-of-range or "
                "mis-specified idx window. Every window step ran empty; this is NOT "
                "a seeded run. Re-check the window bounds against the trace idx range.")
    elif unmodeled is not None:
        note = ("Level-2 symex hit an un-modeled instruction — NOT force-concretized; "
                + str(unmodeled.get("question", "supply its symbolic semantics")))
    elif unpinned_mem:
        note = ("external memory input(s) in the window are neither symbolized nor "
                "backed — the symbolic chain cannot start there, so the exit "
                "collapses to a concrete 0 (the handler11 symbolic=0 trap). Decide "
                "symbolize-vs-back for each and set CaseConfig.symbolic_mem / "
                "concrete_backing: "
                + ", ".join(f"0x{m.addr:x}+{m.size}@idx{m.src_idx}" for m in unpinned_mem))
    elif self_check is not None and self_check.blocked:
        # G4: F does not reproduce its own trace — symex unsound, ranks above the
        # cross-vector parity note (this is the more basic, necessary failure).
        note = self_check.note
    elif parity_vreport is not None and not parity_vreport.sufficient:
        note = parity_vreport.advisory
    elif auto_seed_info is not None and auto_seed_info["empty"]:
        note = ("auto-seed found no live-in registers in the window — fell back "
                "to the full reg_file; verify the window bounds locate a real "
                "handler body (a degenerate window seeds nothing useful)")
    elif auto_seed_info is not None and auto_seed_info["dropped_not_in_reg_file"]:
        note = ("auto-seed dropped live-in regs absent from reg_file: "
                f"{auto_seed_info['dropped_not_in_reg_file']} — widen reg_file "
                "to cover them or re-check the window bounds")
    elif self_check is not None and self_check.status == "INCONCLUSIVE":
        # Surfaced, not silent (lowest priority — only when no other note): the
        # necessary self-check could not run (no trace facts / un-evaluable F). It
        # does not flip an otherwise-passing close, but the agent/runner must know
        # the recovered F was never checked against its own trace.
        note = (f"G4 emit self-check INCONCLUSIVE — {self_check.note}. The recovered "
                "F was NOT checked against its own trace; supply "
                "trace_self_check.seed_values + sink_value from the runner.")
    elif clo_deferred and not closed:
        # §5′ honest deferral note: this window is DYNAMICALLY backed (op.addr+value
        # in the trace) but its STATIC closure is un-backed — the EA is symbolic /
        # staged behind the symbol (NOT missing backing — do NOT re-capture). The
        # dynamic gate let it ENTER symex P2(i) forwarding, but P2(i) did not resolve
        # it to a closing F (no higher-priority note fired). The window stays open
        # for the opaque branch (Phase 0/0b diagnosis), never a false CLOSE.
        note = ("symbolic memory-addressing window: EA is symbolic / staged behind "
                "the symbol (NOT missing backing — do NOT re-capture). Deferred to "
                "symex P2(i) forwarding, which did not resolve a closing F here; "
                "route to the opaque-staging branch (Phase 0/0b) to localize the "
                "staging address.")
    else:
        note = ""

    # --- record (recording policy: durable findings + ONE roll-up, no per-step) ---
    entry_keys: list[str] = []
    view_path: str | None = None
    if ledger is not None:
        if emitted_F is not None:
            entry_keys.append(_cl.record_verdict(
                ledger, call_fn="emit_python",
                inputs={"window": list(cc.window), "parity_min": cc.parity_min},
                exec_identity=exec_identity, verdict=("PASS" if parity_ok else "FAIL"),
                kind=_cl.LedgerKind.EMIT, subject=cc.task or "emit",
                payload={"F": emitted_F, "parity": parity, "parity_ok": parity_ok},
                ts=ts, auto_view=False))
        entry_keys.append(_cl.record_call(
            ledger, call_fn="drive", inputs={"task": cc.task, "window": list(cc.window)},
            exec_identity=exec_identity, kind=_cl.LedgerKind.RUN_SUMMARY,
            subject=cc.task or "drive",
            result={
                "task": cc.task, "mode": mode.mode.value, "closed": closed,
                "backing": {"sufficient": back.sufficient, "blind_count": len(back.blind_pcs)},
                "closure": {"sufficient": clo.sufficient,
                            "unbacked_root_count": len(clo.unbacked_roots)},
                "gold_parity": parity,
                "symbolic_forwards": symbolic_forwards,
                "parity_vectors": (parity_vreport.to_dict()
                                   if parity_vreport is not None else None),
                "checkpoints": dict(decisions),
                "plan": [s.name for s in plan.steps],
            },
            ts=ts, auto_view=True))
        d = _cl.ledger_dir(ledger)
        if d is not None:
            view_path = str(d / "cvd_ledger_view.md")

    return DriveResult(
        closed=closed, mode=mode.mode.value, parity=parity, emitted_F=emitted_F,
        backing_ok=backing_ok, address_closure=clo.to_dict(), mem_backing=back.to_dict(),
        per_step=tuple(per_step), entry_keys=tuple(entry_keys), view_path=view_path,
        checkpoints=dict(decisions),
        parity_report=(parity_vreport.to_dict() if parity_vreport is not None else None),
        unmodeled=unmodeled,
        decode_audit=decode_audit,
        self_check=(self_check.to_dict() if self_check is not None else None),
        note=note, symbolic_forwards=symbolic_forwards,
        mem_sink_unreadable=mem_sink_unreadable)


