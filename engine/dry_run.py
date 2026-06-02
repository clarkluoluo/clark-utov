"""Dry-run: exercise every currently-implemented capability against the vmp sample.

This is a sanity check that the plumbing works end-to-end with what we have today.
Stages S1..S6 themselves are still NotImplementedError stubs — those will get
filled in P1+. What we can exercise NOW:

  - Trace reader (UnidbgTextTraceReader on vmp/trace.txt)
  - Conformance gate (File mode → C1/C2/C3 SKIP, C4 PASS)
  - Fingerprint catalog scan (regs_write magic match)
  - Fold primitives (block-aware repetition on PC sequences)
  - Dataflow primitives (regflow, producer_backward, classify_semop)
  - HypTree CRUD (in a throwaway SQLite)
  - Store (work dir layout creation)
"""

from __future__ import annotations

import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.conformance import run_conformance, write_report                     # noqa: E402
from engine.data.fingerprints import FINGERPRINTS, INSTR_PATTERNS                # noqa: E402
from engine.dataflow import classify_semop, producer_backward, regflow_forward   # noqa: E402
from engine.fold import FoldStats, fold_block_repeats, fold_runs                 # noqa: E402
from engine.hyp_tree import HypTree                                              # noqa: E402
from engine.runner_client import NullRunnerAdapter, UnidbgTextTraceReader        # noqa: E402
from engine.store import WorkDir, open_hypotheses_db                             # noqa: E402
from engine.types import TargetMeta                                              # noqa: E402

TRACE_PATH = REPO_ROOT / "testTarget" / "vmp" / "trace.txt"


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def step_conformance(meta: TargetMeta) -> None:
    banner("STEP 1 — Conformance gate (PLAN §17)")
    runner = NullRunnerAdapter(meta)
    reader = UnidbgTextTraceReader(TRACE_PATH)
    report = run_conformance(runner=runner, trace_reader=reader, probe_input=b"\x00" * 16)
    print(f"  mode={report.mode}  overall={report.overall.value}  verifier_degraded={report.verifier_degraded}")
    for c in report.checks:
        diag = c.detail.get("diagnosis")
        diag_s = f"  // {diag}" if diag else ""
        print(f"    {c.check.value:24} {c.result.value:5}  {diag_s}")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        write_report(report, Path(f.name))
        print(f"  report written → {f.name}")


def step_fingerprint_scan(items: list) -> None:
    banner("STEP 2 — Fingerprint scan (PLAN §12.5)")
    by_magic = {fp.magic: fp for fp in FINGERPRINTS}
    hit_counts: Counter[str] = Counter()
    for ins in items:
        for val in ins.regs_write.values():
            fp = by_magic.get(val)
            if fp is not None:
                hit_counts[fp.name] += 1
    if not hit_counts:
        print("  no constant fingerprint hits — expected for VMP (per §12.5 caveat)")
    else:
        for name, n in hit_counts.most_common(10):
            print(f"    {name:25}  hits={n}")
    print(f"  scanned {len(items):,} instr against {len(FINGERPRINTS)} fingerprints + {len(INSTR_PATTERNS)} SIMD patterns")


def step_fold(items: list) -> None:
    banner("STEP 3 — Fold (PLAN §12.4)")
    print("  simple-run fold by mnemonic (threshold=10):")
    s1 = FoldStats()
    _ = list(fold_runs(items, signature_of=lambda i: i.mnemonic, threshold=10, stats=s1))
    print(f"    folds_applied={s1.folds_applied}  lines_skipped={s1.lines_skipped}")
    print("  block-aware fold (PC signature):")
    for W in (4, 6, 14):
        s = FoldStats()
        _ = list(fold_block_repeats(items, signature_of=lambda i: f"0x{i.pc:x}", window=W, threshold=3, stats=s))
        print(f"    W={W:2}  folds_applied={s.folds_applied}  lines_skipped={s.lines_skipped}")


def step_dataflow(items: list) -> None:
    banner("STEP 4 — Dataflow primitives")
    sp_writes = list(regflow_forward(items, reg="sp", limit=5))
    print("  regflow_forward(sp, limit=5):")
    for h in sp_writes:
        print(f"    idx={h.idx:5}  sp=0x{h.value:x}  '{h.mnemonic}'")
    print("\n  producer_backward — pick last write, find prior producer:")
    last_with_write = next(ins for ins in reversed(items) if ins.regs_write)
    reg, val = next(iter(last_with_write.regs_write.items()))
    print(f"    target: idx={last_with_write.idx} wrote {reg}=0x{val:x}")
    p = producer_backward(items, val, sink_idx=last_with_write.idx, max_back=2000)
    if p:
        print(f"    earliest match: idx={p.idx} reg={p.reg} value=0x{p.value:x}")
    classes = Counter(classify_semop(ins.mnemonic) for ins in items)
    print("\n  classify_semop distribution (top 6):")
    for k, n in classes.most_common(6):
        print(f"    {k:24}  {n:6}")


def step_hyp_tree() -> None:
    banner("STEP 5 — Hypothesis tree CRUD")
    with tempfile.TemporaryDirectory() as td:
        work = WorkDir(td, "dryrun_target", "deadbeef", new_run=True)
        conn = open_hypotheses_db(work)
        tree = HypTree(conn)
        root_id = tree.add(None, kind="algo_signature", subject="SHA256?",
                           payload={"evidence": ["SHA256.K[0] @ idx 1234"]}, confidence=0.85)
        child_a = tree.add(root_id, kind="handler_semantic", subject="handler@0x40006cc4",
                           payload={"op": "XOR"}, confidence=0.7)
        child_b = tree.add(root_id, kind="handler_semantic", subject="handler@0x40006cc4",
                           payload={"op": "ADD"}, confidence=0.4)
        print(f"  inserted: root={root_id}, child_a={child_a}, child_b={child_b}")
        print(f"  next pending sibling of {child_a}: {tree.next_pending_sibling(child_a)}")
        tree.mark_verdict(child_a, verdict="fail", verifier_result={"reason": "delta mismatch"})
        sib = tree.next_pending_sibling(child_a)
        print(f"  after marking child_a failed, sibling iterator returns: id={sib.id if sib else None}")
        print(f"  work dir created: {work.root.name}")
        print(f"    latest symlink → {(work.target_dir / 'latest').readlink()}")


def main() -> int:
    if not TRACE_PATH.exists():
        print(f"ERROR: trace not found at {TRACE_PATH}", file=sys.stderr)
        return 1
    meta = TargetMeta(
        target_name="libEncryptor.so",
        arch="arm64",
        algo_entry_pc=0x40007D88,
        algo_exit_pc=0x40007ED8,
        input_length=None,
        output_length=32,
    )
    step_conformance(meta)
    print("\nLoading full trace into memory for analysis…")
    items = list(UnidbgTextTraceReader(TRACE_PATH))
    print(f"  loaded {len(items):,} instructions")
    step_fingerprint_scan(items)
    step_fold(items)
    step_dataflow(items)
    step_hyp_tree()
    banner("DRY RUN COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
