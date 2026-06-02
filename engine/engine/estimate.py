"""Dry-run / estimate mode (PLAN §15.3).

Runs the cheap deterministic stages on the trace WITHOUT any LLM calls, then
projects how much LLM spend the full pipeline would likely cost.

Heuristic (deliberately conservative — we'd rather over-quote than surprise):

  estimated_stuck_points = S5 surviving instructions that are NOT constants
                           and NOT InsSub-canonicalized
  estimated_llm_calls    = estimated_stuck_points * N (default 5)
                           + optional blue_team_calls (high-impact findings)
  estimated_tokens_per_call = 1200 input + 250 output (typical S6 prompt)
  estimated_usd         = sum over calls @ model pricing
  estimated_wall_seconds = calls * 1.5s   (DeepSeek-V3 latency ballpark)

The output explicitly carries `uncertainty: "high"` because:
  - "stuck points" only loosely correlates with actual S6 invocations
  - blue-team trigger depth is data-dependent
  - LLM token counts vary with how many surrounding mnemonics S6 includes

Honest about the imprecision is the design goal — don't quote a fake-precise
number.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cost import DEFAULT_PRICING, usd_cost
from .runner_client import RunnerAdapter, TraceReader
from .stages import s1_segment, s1b_fingerprint, s2_dedupe, s3_triton, s4_slice, s5_simplify
from .store import WorkDir
from .types import TargetMeta


# --- knobs (user-tunable via EstimateConfig) ---

TYPICAL_INPUT_TOKENS_PER_CALL  = 1200
TYPICAL_OUTPUT_TOKENS_PER_CALL = 250
TYPICAL_LATENCY_S_PER_CALL     = 1.5

DEFAULT_N_CANDIDATES = 5


@dataclass
class EstimateConfig:
    n_candidates_per_stuck_point: int = DEFAULT_N_CANDIDATES
    model: str = "deepseek-chat"
    include_blue_team: bool = False        # blue-team multiplies cost
    blue_team_rate: float = 0.1            # fraction of findings that trigger it
    max_stuck_points: int | None = None    # cap S6 loop; matches --max-stuck-points


@dataclass
class EstimateReport:
    target_name: str
    trace_instructions: int
    blocks: int
    unique_blocks: int
    fingerprint_hits: int
    estimated_stuck_points: int
    estimated_llm_calls: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_usd: float
    estimated_wall_seconds: float
    uncertainty: str = "high"
    notes: list[str] = None  # type: ignore[assignment]

    def as_human(self) -> str:
        lines = [
            f"Estimate for {self.target_name}  (uncertainty: {self.uncertainty})",
            f"  trace instructions:     {self.trace_instructions:,}",
            f"  basic blocks:           {self.blocks:,}  (unique {self.unique_blocks})",
            f"  fingerprint hits:       {self.fingerprint_hits}",
            f"  expected stuck points:  {self.estimated_stuck_points}",
            f"  expected LLM calls:     {self.estimated_llm_calls}",
            f"  expected tokens:        {self.estimated_input_tokens:,} in / "
            f"{self.estimated_output_tokens:,} out",
            f"  expected USD:           ${self.estimated_usd:.4f}",
            f"  expected wall time:     ~{self.estimated_wall_seconds:.0f}s",
        ]
        if self.notes:
            lines.append("  notes:")
            lines.extend(f"    - {n}" for n in self.notes)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def estimate(
    target_meta: TargetMeta,
    trace_reader: TraceReader,
    runner: RunnerAdapter,
    *,
    work_root: Path,
    config: EstimateConfig | None = None,
) -> EstimateReport:
    """Run S1..S5 (deterministic, no LLM) and project S6 cost.

    Creates a throwaway WorkDir suffixed with `_estimate` so it doesn't
    interfere with a parallel real run.
    """
    cfg = config or EstimateConfig()

    work = WorkDir(
        root=work_root,
        target=f"{target_meta.target_name}_estimate",
        input_hash="estimate",
        new_run=True,
    )
    items = list(trace_reader)
    session: dict[str, Any] = {"fingerprint_anchor_idxs": [], "algo_hints": []}
    ctx = {"items": items, "work": work, "session": session, "verifier": None}

    s1_summary = s1_segment.run(ctx)
    s1b_summary = s1b_fingerprint.run(ctx)
    s2_summary = s2_dedupe.run(ctx)
    s3_triton.run(ctx)
    s4_slice.run(ctx)
    s5_simplify.run(ctx)

    # Count S5 entries that look like unrecognised instructions — i.e. survived
    # the slice, weren't constants, weren't InsSub-canonical.
    import json
    s5_path: Path = work.root / "stage_outputs" / "s5_simplified.jsonl"
    stuck = 0
    if s5_path.exists():
        for line in s5_path.open("r", encoding="utf-8"):
            row = json.loads(line)
            if row.get("kind") != "instr":
                continue
            if "constant" in row or "mov_immediate" in row or row.get("part_of_inssub"):
                continue
            stuck += 1

    # Respect the same max_stuck_points cap that script_mode.run_full_pipeline
    # enforces — without this, estimate over-quotes the cost by reporting all
    # stuck points × N, while the actual run would stop earlier.
    effective_stuck = stuck if cfg.max_stuck_points is None \
                      else min(stuck, cfg.max_stuck_points)
    llm_calls = effective_stuck * cfg.n_candidates_per_stuck_point
    if cfg.include_blue_team:
        llm_calls += int(s1b_summary["fingerprint_hits"] * cfg.blue_team_rate)

    in_tokens  = llm_calls * TYPICAL_INPUT_TOKENS_PER_CALL
    out_tokens = llm_calls * TYPICAL_OUTPUT_TOKENS_PER_CALL
    cost_usd   = usd_cost(cfg.model, in_tokens, out_tokens, DEFAULT_PRICING)
    wall_s     = llm_calls * TYPICAL_LATENCY_S_PER_CALL

    notes: list[str] = []
    if stuck == 0:
        notes.append("S5 found 0 stuck points — pipeline might converge without any LLM call. "
                     "Real cost likely ~$0.")
    if cfg.max_stuck_points is not None and stuck > cfg.max_stuck_points:
        notes.append(
            f"Trace has {stuck} stuck points but max_stuck_points={cfg.max_stuck_points} "
            f"caps the loop at {cfg.max_stuck_points}; estimate reflects the cap. "
            "Drop --max-stuck-points to estimate full coverage."
        )
    if not cfg.include_blue_team:
        notes.append("Blue-team review is OFF in this estimate. With it on, expect "
                     "~10–30% more calls.")
    notes.append("Token-per-call is a typical-case figure; long handler-semantic "
                 "prompts can be 2× this. Treat estimate as ceiling, not exact.")

    return EstimateReport(
        target_name=target_meta.target_name,
        trace_instructions=len(items),
        blocks=int(s1_summary.get("blocks", 0)),
        unique_blocks=int(s2_summary.get("unique_blocks", 0)),
        fingerprint_hits=int(s1b_summary.get("fingerprint_hits", 0)),
        estimated_stuck_points=stuck,
        estimated_llm_calls=llm_calls,
        estimated_input_tokens=in_tokens,
        estimated_output_tokens=out_tokens,
        estimated_usd=cost_usd,
        estimated_wall_seconds=wall_s,
        uncertainty="high",
        notes=notes,
    )
