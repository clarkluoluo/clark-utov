#!/usr/bin/env python3
"""File-mode full pipeline: S1..S5 + the layer-0/1/2 verify chain.

Equivalent to `utov pipeline-file` (post BUG_REPORT-6 #1 fix) — kept here as
a self-contained reference template and as a smoke test for new static
traces. NullRunner means C1/C2/C3 conformance and the algorithm-fit IO test
are SKIPPED; algorithm-fit's `io_test` field self-documents the skip.

Usage:
    python3 file_mode_full.py \\
        --trace path/to/trace.txt \\
        --target-name libEncryptor.so \\
        --entry 0x40007d88 --exit 0x40007ed8 \\
        --work-root ./work
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from engine.core import Core, CoreConfig, _pick_reader  # noqa: PLC2701
from engine.orchestrators.script_mode import Mode, run_full_pipeline
from engine.runner_client import NullRunnerAdapter
from engine.types import TargetMeta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--target-name", required=True)
    ap.add_argument("--entry",     required=True, help="algo_entry_pc hex")
    ap.add_argument("--exit",      required=True, dest="exit_pc",
                                   help="algo_exit_pc hex")
    ap.add_argument("--input-len",  type=int, default=16)
    ap.add_argument("--output-len", type=int, default=32)
    ap.add_argument("--work-root",  type=Path, default=Path("work"))
    args = ap.parse_args()

    meta = TargetMeta(
        target_name=args.target_name, arch="arm64",
        algo_entry_pc=int(args.entry, 16),
        algo_exit_pc=int(args.exit_pc, 16),
        input_length=args.input_len, output_length=args.output_len,
    )
    runner = NullRunnerAdapter(meta)
    reader = _pick_reader(args.trace)
    input_hash = hashlib.sha1(b"static-trace").hexdigest()[:12]
    config = CoreConfig(
        work_root=args.work_root.resolve(), target_meta=meta,
        input_hash=input_hash, driver_mode="script", new_run=True,
    )
    core = Core(config, reader, runner, skip_conformance=False)
    print(f"work dir:  {core.work.root}")

    report = run_full_pipeline(core, mode=Mode.FRUGAL)

    print("\npipeline summary:")
    for s in report.stage_summaries:
        print("  " + json.dumps(s))
    print(f"\nfindings_promoted: {report.findings_promoted}")
    print(f"hypotheses_total:  {report.hypothesis_count}")

    algo = core.get_hypotheses(kind="algorithm_identified")
    if algo:
        print("\nalgorithm_identified:")
        for h in algo:
            payload = h.payload or {}
            print(f"  {h.subject}  conf={h.confidence}  "
                  f"evidence={payload.get('evidence_score')}  "
                  f"anchors={len(payload.get('anchors_seen', []))}/"
                  f"{len(payload.get('anchors_expected', []))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
