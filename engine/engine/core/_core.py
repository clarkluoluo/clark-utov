"""The Core facade class, assembled from method-group mixins."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..conformance import (
    ConformanceReport,
    require_pass_or_die,
    run_conformance,
    write_report,
)
from ..hyp_tree import HypNode, HypTree
from ..runner_client import (
    JsonlTraceReader,
    RunnerAdapter,
    TraceReader,
    UnidbgTextTraceReader,
)
from ..store import (
    WorkDir, _now_iso,
    archive_subtree as _archive_subtree,
    log_intervention as _log_intervention,
    open_findings_db, open_hypotheses_db,
    read_interventions as _read_interventions,
    read_payload,
)
from ..stages import (
    s0_5_normalize, s1_segment, s1b_fingerprint, s2_dedupe,
    s3_triton, s4_slice, s5_simplify, s6_taint,
)
from ..types import Instruction, TargetMeta
from ..verifier import Verifier

from ._base import *  # noqa: F401,F403
from ._base import _STAGES, _algorithm_hyp_trap, _run_io_equivalence  # noqa: F401
from ._lifecycle import _CoreLifecycleMixin
from ._hyp import _CoreHypMixin
from ._handler_verify import _CoreHandlerVerifyMixin
from ._batch import _CoreBatchMixin
from ._algorithm import _CoreAlgorithmMixin


class Core(_CoreLifecycleMixin, _CoreHypMixin, _CoreHandlerVerifyMixin, _CoreBatchMixin, _CoreAlgorithmMixin):
    """Driver-agnostic system facade."""

    def __init__(
        self,
        config: CoreConfig,
        trace_reader: TraceReader,
        rerun: RunnerAdapter,
        *,
        skip_conformance: bool = False,
        profile_name: str | None = None,
    ):
        self.config = config
        self.work = WorkDir(
            root=config.work_root,
            target=config.target_meta.target_name,
            input_hash=config.input_hash,
            run_id=config.run_id,
            new_run=config.new_run,
        )
        self.trace_reader = trace_reader
        self.rerun = rerun
        self.verifier = Verifier(rerun)

        # v0.3.0 profile layer wire-in (PLAN §19 / IMPL_PLAN §P1.0 step 8).
        # Lazy + opt-in: ``profile_name=None`` (the default) means the
        # ProfileRegistry is not loaded at all — existing callers that
        # haven't migrated keep working unchanged. Pass an explicit
        # ``profile_name`` (typically ``"vmp_algorithm_extraction"``)
        # to make ``Core.profile`` resolve to a :class:`MergedProfile`
        # that wrappers / loop processors / lints can consume.
        self._profile_name: str | None = profile_name
        self._profile: Any | None = None  # cached MergedProfile after first access

        # Materialize trace once — stages need random access for many passes.
        # For very large traces, switch to a paginated cache instead.
        self._items: list[Instruction] = list(trace_reader)

        # SessionState: cross-stage feedback bag. Each stage may stash hints
        # for downstream stages here (e.g. S1.5 publishes fingerprint anchor
        # idxs that S4 picks up as additional sinks). Persisted as session.json
        # so a mode switch can resume.
        self.session: dict[str, Any] = self._load_or_init_session()

        # Critical-section counter: stages bump on entry/decrement on exit.
        # is_safe_to_interrupt() = (counter == 0).
        self._critical_section_depth: int = 0

        # Persist meta.json so the other driver can pick up state (D-021).
        self._write_meta()

        if not skip_conformance:
            self._gate_conformance()




# --- convenience constructor for the common Live-mode usage ---

def open_live(
    work_root: Path,
    runner: RunnerAdapter,
    *,
    input_bytes: bytes,
    run_id: str | None = None,
    new_run: bool = False,
    skip_conformance: bool = False,
) -> Core:
    """Convenience: ask runner for trace + meta, build a Core, gate conformance.

    Important: C5 (cross-call independence) MUST run BEFORE the first
    get_trace, because the bug it detects only manifests as the first ever
    get_trace after some reruns. If get_trace runs first, the leak gets
    "primed away" and C5 can't see it. We run C5 standalone here, then let
    Core's _gate_conformance run the full C1-C5 suite normally.
    """
    meta = runner.metadata()

    if not skip_conformance:
        # Lazy import to avoid circular: core ↔ conformance
        from ..conformance import (
            _c5_cross_call_independence as _c5,
            CheckResult,
        )
        c5 = _c5(runner, meta, input_bytes)
        if c5.result == CheckResult.FAIL:
            raise RuntimeError(
                "Pre-trace conformance C5 (cross-call independence) FAILED. "
                "The runner leaks state across calls; subsequent get_trace would "
                "produce a contaminated trace. Fix the runner before re-running. "
                f"Detail: {c5.detail}"
            )

    trace_path = runner.get_trace(input_bytes, meta.algo_entry_pc, meta.algo_exit_pc)
    reader = _pick_reader(Path(trace_path))
    input_hash = hashlib.sha1(input_bytes).hexdigest()[:12]
    config = CoreConfig(
        work_root=work_root,
        target_meta=meta,
        input_hash=input_hash,
        driver_mode="script",
        run_id=run_id,
        new_run=new_run,
    )
    return Core(config, reader, runner, skip_conformance=skip_conformance)


def _pick_reader(path: Path) -> TraceReader:
    """Sniff: JSONL starts with '{' on the first non-empty line."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.lstrip()
            if not s:
                continue
            if s.startswith("{"):
                return JsonlTraceReader(path)
            return UnidbgTextTraceReader(path)
    raise ValueError(f"{path}: empty trace file")


# Convenience: also expose findings DB
def open_findings(core: Core):
    return open_findings_db(core.work)
