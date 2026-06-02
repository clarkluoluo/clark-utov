"""run_recovery — the one-call consumer entry that closes the hand-wired half.

The reference case evidence: a test-agent hand-rolled 80 lines to wire a recovery run and
made 3 config errors (no decisions → instant PENDING; mixed default_registry →
OUTPUT_NOT_OBSERVABLE noise; pinned a pre-relocation sink). This pins that the
one-call entry makes those 3 unrepeatable while still escalating the GENUINE
judgment. Synthetic shapes only — zero case-specific knowledge.
"""

from __future__ import annotations

from engine.cvd import CvdOutcome
from engine.cvd_recovery import _reconcile_anchors, run_recovery
from engine.dispatch_coverage import HandlerInvocation, preflight_dispatch_coverage
from engine.export_stamp import is_utov_export
from engine.setup_symex import CaseConfig, build_concrete_backing
from engine.types import Instruction, MemOp


def _ins(idx, pc, mnem, *, reads=None, writes=None, mem=()):
    return Instruction(idx=idx, pc=pc, bytes_=b"\x00\x00\x00\x00", mnemonic=mnem,
                       regs_read=reads or {}, regs_write=writes or {}, mem=tuple(mem))


def _closed_runner(_ctx):
    return {"propagated": True, "gold_parity": "8/8",
            "expr_source": "def f(carrier):\n    return carrier & 0xff\n",
            "parity_vectors": [{"input_key": f"v{i}", "observed": f"o{i}",
                                "predicted": f"o{i}", "exec_id": f"e{i}"} for i in range(3)],
            "trace_self_check": {"seed_values": {"carrier": 0x10},
                                 "sink_value": 0x10, "sink_mask": 0xFF}}


def _base(**kw):
    d = dict(
        target="synthetic.so", input_hash="ab", run_id="r", seed_hint_addr=0x100,
        sink_hint_addr=0x200, entry_pc=0x0FFF, window=(0, 1), window_kind="idx",
        reg_file=("w0", "w1", "x16"), inputs=("carrier",), parity_min=8,
        symbolic_regs=("w1",),
        concrete_backing=build_concrete_backing(reg_values={"x16": 0x9000}),
        task="recover")
    d.update(kw)
    return CaseConfig(**d)


# --- one call → stamped JSON, no sqlite, no default-sink noise --------------- #

def test_run_recovery_one_call_clean_gap_map(tmp_path):
    trace = [_ins(0, 0x1000, "ldr w0, [x16]", reads={"x16": 0x9000}, writes={"w0": 1}),
             _ins(1, 0x1004, "mul w0, w0, w1", reads={"w0": 1, "w1": 2}, writes={"w0": 2})]
    cov = preflight_dispatch_coverage(
        trace, invocations=[HandlerInvocation("A", 0, 1)], reg_file=("w0", "w1", "x16"))
    res, path = run_recovery(
        trace, base_config=_base(), triton_runner=_closed_runner, coverage=cov,
        expected=b"\x00\x00\x00\x00", work_root=tmp_path / "out",
        ts="2026-06-01T00:00:00Z", exec_identity={"target": "synthetic.so"})

    assert res.outcome is CvdOutcome.COLLECTED
    assert len(res.confirmed) == 1
    # stamped JSON actively persisted, no sqlite dead ledger.
    assert path.name == "cvd_gap_map.json" and is_utov_export(path.read_text())
    assert list((tmp_path / "out").glob("*.sqlite*")) == []
    # ② no default_registry sink chain → no OUTPUT_NOT_OBSERVABLE noise.
    kinds = {e.get("terminal_kind") for e in res.extension_requests}
    assert "OUTPUT_NOT_OBSERVABLE" not in kinds


# --- ① defaults let it reach symex; ②real judgment still escalates ---------- #

def test_run_recovery_defaults_alias_but_escalates_mem_judgment(tmp_path):
    # A window with an EXTERNAL memory load (mem read, no in-window writer, not
    # pinned) → drive surfaces mem_input_symbolize_vs_back, a GENUINE judgment the
    # entry must NOT auto-decide. alias_vs_compute / which_static ARE defaulted, so
    # they cause no spurious PENDING.
    trace = [_ins(0, 0x1000, "ldr w0, [x5]", reads={"x5": 0x8000},
                  mem=(MemOp("r", 0x8000, 0x11, 4),), writes={"w0": 0x11}),
             _ins(1, 0x1004, "mul w0, w0, w1", reads={"w0": 0x11, "w1": 2}, writes={"w0": 1})]
    cov = preflight_dispatch_coverage(
        trace, invocations=[HandlerInvocation("A", 0, 1)],
        reg_file=("w0", "w1", "x5"))
    base = _base(reg_file=("w0", "w1", "x5"), symbolic_regs=("w1",),
                 concrete_backing=None)
    res, _ = run_recovery(
        trace, base_config=base, triton_runner=_closed_runner, coverage=cov,
        expected=b"\x00\x00\x00\x00", work_root=tmp_path / "out2",
        ts="2026-06-01T00:00:00Z")
    assert res.outcome is CvdOutcome.COLLECTED
    # the genuine judgment is escalated (PENDING), not auto-decided.
    assert any("mem_input_symbolize_vs_back" in p.get("reason", "")
               for p in res.pending_judgments)
    # the defaulted checkpoints did NOT cause a PENDING.
    assert not any("alias_vs_compute" in p.get("reason", "")
                   for p in res.pending_judgments)


def test_caller_decisions_override_defaults(tmp_path):
    # A caller-supplied decision for the mem judgment is honoured (no PENDING then).
    trace = [_ins(0, 0x1000, "ldr w0, [x5]", reads={"x5": 0x8000},
                  mem=(MemOp("r", 0x8000, 0x11, 4),), writes={"w0": 0x11}),
             _ins(1, 0x1004, "mul w0, w0, w1", reads={"w0": 0x11, "w1": 2}, writes={"w0": 1})]
    cov = preflight_dispatch_coverage(
        trace, invocations=[HandlerInvocation("A", 0, 1)], reg_file=("w0", "w1", "x5"))
    base = _base(reg_file=("w0", "w1", "x5"), symbolic_regs=("w1",), concrete_backing=None)
    res, _ = run_recovery(
        trace, base_config=base, triton_runner=_closed_runner, coverage=cov,
        expected=b"\x00\x00\x00\x00", work_root=tmp_path / "out3",
        ts="2026-06-01T00:00:00Z",
        decisions={"mem_input_symbolize_vs_back": {0x8000: {"symbolize": 0x11}}})
    assert not any("mem_input_symbolize_vs_back" in p.get("reason", "")
                   for p in res.pending_judgments)


# --- ③ anchor reconcile: rebase pre-relocation anchors onto the run's base --- #

def test_reconcile_anchors_rebases_pc_anchors():
    cc = _base(entry_pc=0x40358000, sink_hint_addr=0x40358100, seed_hint_addr=0x40358050,
               window=(0x40358000, 0x40358100), window_kind="pc")
    # trace observed at a different module base, ONE page-aligned delta on both.
    delta = 0x12000000 - 0x40358000
    items = [_ins(0, 0x40358000 + delta, "x"), _ins(1, 0x40358100 + delta, "x")]
    out = _reconcile_anchors(cc, items)
    assert out.entry_pc == 0x40358000 + delta
    assert out.sink_hint_addr == 0x40358100 + delta
    assert out.seed_hint_addr == 0x40358050 + delta
    assert out.window == (0x40358000 + delta, 0x40358100 + delta)   # pc window rebased


def test_reconcile_anchors_no_shift_is_untouched():
    cc = _base(entry_pc=0x1000, sink_hint_addr=0x1100)
    items = [_ins(0, 0x1000, "x"), _ins(1, 0x1100, "x")]   # entry_delta == 0
    out = _reconcile_anchors(cc, items)
    assert out.entry_pc == 0x1000 and out.sink_hint_addr == 0x1100


def test_reconcile_anchors_disagreeing_anchors_not_reconciled():
    # entry and exit need DIFFERENT deltas → not a clean rebase → leave untouched
    # (a genuinely wrong anchor must not be masked away).
    cc = _base(entry_pc=0x40358000, sink_hint_addr=0x40358100, window_kind="pc")
    items = [_ins(0, 0x12000000, "x"), _ins(1, 0x99999100, "x")]
    out = _reconcile_anchors(cc, items)
    assert out.entry_pc == 0x40358000 and out.sink_hint_addr == 0x40358100


# --- 坎1 重改: cohort_trace_paths → symmetry BY CONSTRUCTION (no sidecar param) --- #

import json   # noqa: E402  (kept local to this section's fixture writers)


def _write_cohort_vector(tmp_path, name, loaded, *, with_sibling=True):
    """A bare cohort main trace JSONL (mem in the ``_mem.jsonl`` sibling, like a
    captured vector) + its conventional ``<stem>_mem.jsonl`` sibling. Returns the
    main trace path — the ONLY thing run_recovery's caller passes."""
    trace = tmp_path / f"{name}.jsonl"
    trace.write_text(
        json.dumps({"idx": 0, "pc": "0x1000", "bytes": "00000000",
                    "mnemonic": "ldr w0, [x5]"}) + "\n"
        + json.dumps({"idx": 1, "pc": "0x1004", "bytes": "00000000",
                      "mnemonic": "mul w0, w0, w1"}) + "\n",
        encoding="utf-8")
    if with_sibling:
        (tmp_path / f"{name}_mem.jsonl").write_text(
            json.dumps({"idx": 0, "rw": "r", "addr": "0x8000",
                        "val": loaded, "size": 4}) + "\n",
            encoding="utf-8")
    return str(trace)


def test_run_recovery_cohort_paths_symmetric_no_sidecar_param(tmp_path):
    # THE proof: run_recovery given ONLY cohort_trace_paths (each with a sibling),
    # NO cohort_mem_sidecars at all → every vector is loaded via the SAME
    # JsonlTraceReader(p).merged() the main trace uses → the cohort is symmetric by
    # construction. A VARYING cohort value → auto-symbolize prefill → the mem
    # judgment is NOT escalated (it was decided from evidence, not punted as PENDING).
    trace = [_ins(0, 0x1000, "ldr w0, [x5]", reads={"x5": 0x8000},
                  mem=(MemOp("r", 0x8000, 0x11, 4),), writes={"w0": 0x11}),
             _ins(1, 0x1004, "mul w0, w0, w1", reads={"w0": 0x11, "w1": 2}, writes={"w0": 1})]
    cov = preflight_dispatch_coverage(
        trace, invocations=[HandlerInvocation("A", 0, 1)], reg_file=("w0", "w1", "x5"))
    base = _base(reg_file=("w0", "w1", "x5"), symbolic_regs=("w1",), concrete_backing=None)
    paths = [_write_cohort_vector(tmp_path, "v0", 0x11),
             _write_cohort_vector(tmp_path, "v1", 0x22)]   # value VARIES across cohort
    res, _ = run_recovery(
        trace, base_config=base, triton_runner=_closed_runner, coverage=cov,
        expected=b"\x00\x00\x00\x00", work_root=tmp_path / "cp1",
        ts="2026-06-01T00:00:00Z",
        cohort_trace_paths=paths, input_keys=["a", "b"])   # <-- NO sidecar arg
    assert res.outcome is CvdOutcome.COLLECTED
    # symmetric cohort merged from paths → varying value → auto-symbolize prefilled
    # → the judgment is no longer punted to the agent (contrast: bare cohort = PENDING).
    assert not any("mem_input_symbolize_vs_back" in p.get("reason", "")
                   for p in res.pending_judgments)


def test_run_recovery_cohort_paths_missing_sibling_warns_in_gap_map(tmp_path):
    # A CONSTANT cohort value → back/recommend (not auto-prefilled) → mem judgment
    # PENDINGs; one vector lacks its sibling → the load WARN rides into the gap-map
    # mem_disposition_diagnostics (boundary, not a silent batch degradation).
    trace = [_ins(0, 0x1000, "ldr w0, [x5]", reads={"x5": 0x8000},
                  mem=(MemOp("r", 0x8000, 0x11, 4),), writes={"w0": 0x11}),
             _ins(1, 0x1004, "mul w0, w0, w1", reads={"w0": 0x11, "w1": 2}, writes={"w0": 1})]
    cov = preflight_dispatch_coverage(
        trace, invocations=[HandlerInvocation("A", 0, 1)], reg_file=("w0", "w1", "x5"))
    base = _base(reg_file=("w0", "w1", "x5"), symbolic_regs=("w1",), concrete_backing=None)
    paths = [_write_cohort_vector(tmp_path, "v0", 0x11),
             _write_cohort_vector(tmp_path, "v1", 0x11),
             _write_cohort_vector(tmp_path, "v2", 0x11, with_sibling=False)]
    res, gap_path = run_recovery(
        trace, base_config=base, triton_runner=_closed_runner, coverage=cov,
        expected=b"\x00\x00\x00\x00", work_root=tmp_path / "cp2",
        ts="2026-06-01T00:00:00Z",
        cohort_trace_paths=paths, input_keys=["a", "b", "c"])
    assert res.outcome is CvdOutcome.COLLECTED
    blob = gap_path.read_text()
    # the missing-sibling vector's WARN reached the persisted gap map (not silent).
    assert "no mem sidecar" in blob and "v2_mem.jsonl" in blob


# --- run-level top-level cohort_load bubble: visible on EVERY window outcome --- #
# The per-window mem_disposition_diagnostics channel only fires on a PENDING window.
# An opaque TERMINAL / all-CONFIRMED run emits NO PENDING evidence → a missing-
# sibling WARN was previously SILENT there (the tc4 opaque-TERMINAL blind spot).
# The run-level cohort_load field surfaces it at the TOP of the persisted gap map
# regardless of outcome — degradation allowed, silence not.

from engine.export_stamp import load_stamped_json   # noqa: E402


def _no_cohort_trace_and_cov():
    """A minimal trace with NO external mem input → the recovery window reaches a
    non-PENDING terminal (no mem judgment to punt), so the per-window diagnostics
    channel stays silent. The cohort load WARN must still surface run-level."""
    trace = [_ins(0, 0x1000, "ldr w0, [x16]", reads={"x16": 0x9000}, writes={"w0": 1}),
             _ins(1, 0x1004, "mul w0, w0, w1", reads={"w0": 1, "w1": 2}, writes={"w0": 2})]
    cov = preflight_dispatch_coverage(
        trace, invocations=[HandlerInvocation("A", 0, 1)], reg_file=("w0", "w1", "x16"))
    return trace, cov


def test_cohort_load_warn_surfaces_at_gap_map_top_level_on_non_pending_run(tmp_path):
    # An opaque-TERMINAL / all-CONFIRMED run (no PENDING mem judgment) + one cohort
    # vector lacking its _mem.jsonl sibling. PROOF: the no-sidecar WARN is visible at
    # the JSON TOP LEVEL (run-level cohort_load.no_mem_sidecar), NOT buried in a
    # per-window PENDING evidence block that this run never emits.
    trace, cov = _no_cohort_trace_and_cov()
    paths = [_write_cohort_vector(tmp_path, "v0", 0x11),
             _write_cohort_vector(tmp_path, "v1", 0x11, with_sibling=False)]
    res, gap_path = run_recovery(
        trace, base_config=_base(), triton_runner=_closed_runner, coverage=cov,
        expected=b"\x00\x00\x00\x00", work_root=tmp_path / "topwarn",
        ts="2026-06-01T00:00:00Z",
        cohort_trace_paths=paths, input_keys=["a", "b"])
    assert res.outcome is CvdOutcome.COLLECTED
    # this run emits NO PENDING mem-disposition evidence (the old-only channel).
    assert not any("mem_input_symbolize_vs_back" in p.get("reason", "")
                   for p in res.pending_judgments)
    # …yet the WARN is at the TOP of the persisted gap map (parsed, not substring).
    _hdr, doc = load_stamped_json(gap_path.read_text())
    assert "cohort_load" in doc
    warns = doc["cohort_load"]["no_mem_sidecar"]
    assert [w["vector"] for w in warns] == [1]
    assert "v1_mem.jsonl" in warns[0]["warn"]
    # and the in-memory result mirrors it (same field, OUT-layer only).
    assert res.cohort_load["no_mem_sidecar"][0]["vector"] == 1


def test_cohort_load_top_level_clean_when_all_vectors_have_sibling(tmp_path):
    # invariant 7: every vector has its sibling (no bare feed) → the top-level
    # cohort_load is CLEAN (no_mem_sidecar == []) — present but quiet, no new noise.
    trace, cov = _no_cohort_trace_and_cov()
    paths = [_write_cohort_vector(tmp_path, "v0", 0x11),
             _write_cohort_vector(tmp_path, "v1", 0x22)]   # both have siblings
    res, gap_path = run_recovery(
        trace, base_config=_base(), triton_runner=_closed_runner, coverage=cov,
        expected=b"\x00\x00\x00\x00", work_root=tmp_path / "topclean",
        ts="2026-06-01T00:00:00Z",
        cohort_trace_paths=paths, input_keys=["a", "b"])
    assert res.outcome is CvdOutcome.COLLECTED
    _hdr, doc = load_stamped_json(gap_path.read_text())
    assert doc["cohort_load"]["no_mem_sidecar"] == []
    assert doc["cohort_load"]["errors"] == []


def test_cohort_load_absent_when_no_load_layer_ran(tmp_path):
    # No cohort_trace_paths → no load layer → the top-level cohort_load field is
    # OMITTED entirely (today's behaviour, zero new noise on non-recovery runs).
    trace, cov = _no_cohort_trace_and_cov()
    res, gap_path = run_recovery(
        trace, base_config=_base(), triton_runner=_closed_runner, coverage=cov,
        expected=b"\x00\x00\x00\x00", work_root=tmp_path / "noload",
        ts="2026-06-01T00:00:00Z")
    assert res.cohort_load is None
    _hdr, doc = load_stamped_json(gap_path.read_text())
    assert "cohort_load" not in doc
