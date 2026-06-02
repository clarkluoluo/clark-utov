"""Core mixin: session/meta lifecycle + pipeline stages."""
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


class _CoreLifecycleMixin:
    """Core methods: session/meta lifecycle + pipeline stages (split from the monolithic Core)."""
    @property
    def profile(self):
        """Active :class:`engine.profile.MergedProfile` if a profile
        name was supplied at construction; ``None`` otherwise.

        Lazy: the :class:`ProfileRegistry` is only touched on first
        access. Callers that don't care about the profile layer never
        pay the cost.
        """
        if self._profile_name is None:
            return None
        if self._profile is None:
            from engine.profile import ProfileRegistry
            self._profile = ProfileRegistry().load_chain(self._profile_name)
        return self._profile

    def _write_meta(self) -> None:
        tm = self.config.target_meta
        (self.work.root / "meta.json").write_text(json.dumps({
            "target_name":      tm.target_name,
            "arch":             tm.arch,
            "algo_entry_pc":    f"0x{tm.algo_entry_pc:x}",
            "algo_exit_pc":     f"0x{tm.algo_exit_pc:x}",
            "emulator_name":    tm.emulator_name,
            "emulator_version": tm.emulator_version,
            "driver_mode":      self.config.driver_mode,
            "input_hash":       self.config.input_hash,
            "run_id":           self.work.run_id,
        }, indent=2))

    def _load_or_init_session(self) -> dict[str, Any]:
        path = self.work.root / "session.json"
        if path.exists():
            s = json.loads(path.read_text())
            # Idempotent migration: older sessions had no
            # extra_trace_windows. Refresh from current CoreConfig so a
            # mode-switch can override the band list.
            s["extra_trace_windows"] = self._serialize_windows()
            path.write_text(json.dumps(s, indent=2))
            return s
        s = {
            "fingerprint_anchor_idxs": [],
            "algo_hints": [],
            "stuck_points": [],
            # capability_request.md §P0-2: list of [start_hex, end_hex]
            # pairs the runner / S4 slice should treat as in-band.
            "extra_trace_windows": self._serialize_windows(),
        }
        path.write_text(json.dumps(s, indent=2))
        return s

    def _serialize_windows(self) -> list[list[str]]:
        return [
            [f"0x{s:x}", f"0x{e:x}"]
            for (s, e) in (self.config.extra_trace_windows or ())
        ]

    def _save_session(self) -> None:
        (self.work.root / "session.json").write_text(json.dumps(self.session, indent=2))

    # --- conformance gate (PLAN §17) ---

    def _gate_conformance(self) -> ConformanceReport:
        # Use the trace we already loaded — wrap it in a reader-like.
        class _ListReader:
            def __init__(self, items):
                self._items = items
            def __iter__(self):
                return iter(self._items)
        probe = b"\x00" * 16
        if self.config.target_meta.input_length is not None:
            probe = b"\x00" * self.config.target_meta.input_length
        report = run_conformance(self.rerun, _ListReader(self._items), probe_input=probe)
        write_report(report, self.work.root / "conformance_report.json")
        require_pass_or_die(report)
        return report

    # --- capability surface ---

    def run_stage(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Execute one stage by name; persist state; return summary.

        Each stage receives `ctx["session"]` — a mutable dict that survives
        between stages. Stages publish hints there; downstream stages read
        them (e.g. S1.5 → fingerprint_anchor_idxs → S4 sinks).
        """
        if name not in _STAGES:
            raise ValueError(f"unknown stage: {name!r}; known={list(_STAGES)}")
        stage = _STAGES[name]
        code_version = getattr(stage, "CODE_VERSION", name)
        if self.work.is_stage_done(name, code_version):
            return {"stage": name, "skipped": True, "reason": "already done at this code_version"}

        ctx = {
            "items": self._items,
            "work": self.work,
            "session": self.session,
            "verifier": self.verifier,
            **kwargs,
        }
        self._critical_section_depth += 1
        try:
            summary = stage.run(ctx)
        finally:
            self._critical_section_depth -= 1
        # Persist session after every stage so a crash mid-pipeline doesn't
        # lose feedback context.
        self._save_session()
        return summary

    def is_safe_to_interrupt(self) -> bool:
        """True if no stage / critical section is currently executing.
        Agents should query this before sending shutdown / interrupt."""
        return self._critical_section_depth == 0

    def run_pipeline(self, stages: list[str] | None = None,
                     step_mode: bool = False,
                     on_step_break=None) -> list[dict[str, Any]]:
        """Run stages in order; default = canonical S1..S5 chain.

        step_mode: if True, between stages call on_step_break(stage_just_done)
        and if it returns False, halt and return what we have. Lets an agent
        intervene between every stage.
        """
        if stages is None:
            # s0_5 (regs_write reconstruction) runs first so S3's data-flow graph
            # sees producers even on snapshot-only traces. Idempotent on traces
            # that already carry regs_write — additive, never destructive.
            stages = ["s0_5", "s1", "s1b", "s2", "s3", "s4", "s5"]
        summaries: list[dict[str, Any]] = []
        for s in stages:
            summaries.append(self.run_stage(s))
            if step_mode and on_step_break is not None:
                if not on_step_break(s):
                    break
        return summaries

    def read_trace_window(self, idx_from: int, idx_to: int) -> list[Instruction]:
        return [ins for ins in self._items if idx_from <= ins.idx < idx_to]

    # --- hypothesis ledger access ---

