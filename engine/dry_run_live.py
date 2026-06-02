"""Live-mode dry-run: clark-utov Python engine drives the Java unidbg runner.

This is the symmetric counterpart of dry_run.py:
  - dry_run.py      uses NullRunnerAdapter (File mode) + static vmp/trace.txt
                    → conformance C1/C2/C3 SKIP, only C4 runs
  - dry_run_live.py uses SubprocessRunnerAdapter (Live mode) driving
                    example/runner-sha256/runner (Java unidbg + libsha256.so)
                    → conformance C1-C4 all run, full verifier capability

If C1-C4 all PASS here, the system's runner integration path is fully verified.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.conformance import run_conformance, write_report                       # noqa: E402
from engine.runner_client import SubprocessRunnerAdapter, UnidbgTextTraceReader   # noqa: E402

RUNNER_JAR = REPO_ROOT / "example" / "runner-sha256" / "runner" / "target" / "sha256-test-runner-0.1.0.jar"
SO_PATH    = REPO_ROOT / "example" / "runner-sha256" / "libs" / "arm64-v8a" / "libsha256.so"


def banner(t: str) -> None:
    print(f"\n{'=' * 70}\n  {t}\n{'=' * 70}")


def main() -> int:
    if not RUNNER_JAR.exists():
        print(f"ERROR: runner jar not built — run 'mvn -DskipTests package' in"
              f" example/runner-sha256/runner/ first.\n  expected: {RUNNER_JAR}", file=sys.stderr)
        return 2
    if not SO_PATH.exists():
        print(f"ERROR: target .so missing: {SO_PATH}", file=sys.stderr)
        return 2

    banner("STEP 1 — Spawn Java runner via SubprocessRunnerAdapter")
    runner = SubprocessRunnerAdapter(
        cmd=["java", "-jar", str(RUNNER_JAR), "serve", str(SO_PATH)],
        cwd=RUNNER_JAR.parent,
    )
    print(f"  spawned: java -jar {RUNNER_JAR.name} serve")

    try:
        banner("STEP 2 — Pull metadata")
        meta = runner.metadata()
        print(f"  target={meta.target_name}  arch={meta.arch}")
        print(f"  entry=0x{meta.algo_entry_pc:x}  exit=0x{meta.algo_exit_pc:x}")
        print(f"  output_length={meta.output_length}  symbol={meta.algo_symbol}")

        banner("STEP 3 — Pull a trace via get_trace (\"abc\")")
        trace_path = runner.get_trace(b"abc", meta.algo_entry_pc, meta.algo_exit_pc)
        trace_p = Path(trace_path)
        line_count = sum(1 for _ in open(trace_p, "r", encoding="utf-8", errors="replace"))
        print(f"  trace written → {trace_path}")
        print(f"  lines: {line_count:,}")

        banner("STEP 4 — Conformance gate (PLAN §17) in LIVE mode")
        # Build a TraceReader pointed at the live-generated trace for C4
        trace_reader = UnidbgTextTraceReader(trace_p)
        report = run_conformance(
            runner=runner,
            trace_reader=trace_reader,
            probe_input=b"abc",
            mode="live",
        )
        print(f"  mode={report.mode}  overall={report.overall.value}  verifier_degraded={report.verifier_degraded}")
        for c in report.checks:
            diag = c.detail.get("diagnosis")
            tail = f"  // {diag}" if diag else ""
            print(f"    {c.check.value:24} {c.result.value:5} ({c.duration_s*1000:.1f} ms){tail}")

        # Write report to a temp location for inspection
        report_path = Path("/tmp/conformance_live_report.json")
        write_report(report, report_path)
        print(f"  report → {report_path}")

        banner("DRY RUN COMPLETE")
        return 0 if report.overall.value == "pass" else 1
    finally:
        runner.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
