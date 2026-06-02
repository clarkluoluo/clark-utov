"""setup_symex.plan section (split from the monolithic module)."""
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


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A genuine judgment the agent must make — surfaced, never auto-decided.

    The template rides the agent's "fill the skeleton" instinct, but the real
    decisions (is this byte an alias or a computation? which bytes are static?)
    are handed BACK to the agent as explicit checkpoints. Auto-deciding them
    would be a blank generator filling in the wrong answer fast."""

    name:     str
    question: str
    why:      str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "question": self.question, "why": self.why,
                "is_judgment": True}


@dataclass(frozen=True, slots=True)
class SetupStep:
    """One ordered step of the template — a guardrail (mechanical, contract-
    enforced) or a checkpoint (the agent's judgment)."""

    order:       int
    name:        str
    contract:    str               # which of the four / dual-mode it enforces
    guardrail:   str               # what is auto-enforced (empty for judgment steps)
    checkpoint:  Checkpoint | None = None

    @property
    def is_judgment(self) -> bool:
        return self.checkpoint is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "order":       self.order,
            "name":        self.name,
            "contract":    self.contract,
            "guardrail":   self.guardrail,
            "is_judgment": self.is_judgment,
            "checkpoint":  self.checkpoint.to_dict() if self.checkpoint else None,
        }


@dataclass(frozen=True, slots=True)
class SetupSymexPlan:
    """The assembled template: ordered steps + the determinism guard.

    NOT a blank code generator. It enforces (anchors via locate_boundary, mode
    via pick_mode, same-execution capture) and surfaces (alias-vs-compute,
    which-static) — the consumer fills target-specific config into the slots."""

    steps:           tuple[SetupStep, ...]
    determinism_note: str

    @property
    def checkpoints(self) -> tuple[Checkpoint, ...]:
        return tuple(s.checkpoint for s in self.steps if s.checkpoint is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps":            [s.to_dict() for s in self.steps],
            "checkpoints":      [c.to_dict() for c in self.checkpoints],
            "determinism_note": self.determinism_note,
            "kind":             "setup_symex_plan",
        }


# The determinism guard: trace + staging substrate must come from the SAME
# execution, or pinning a value from run A onto a trace from run B fabricates a
# nonce artifact. (The case lost a pass to exactly this until the determinism
# gate caught it — the gate working is the point, not a bug.)
_DETERMINISM_NOTE = (
    "trace, staging values and output MUST be captured in the SAME execution "
    "(same nonce / same run). Merging cross-run captures fabricates artifacts; "
    "if you must merge, pass the determinism gate (EA+value match per byte) "
    "first. Also: do not assume an 'input' is pinnable — verify its generation "
    "path (RNG vs passed-in) before treating it as a fixable constant."
)


def build_setup_symex_plan() -> SetupSymexPlan:
    """Build the guard-railed set-up symex template.

    The ordered skeleton: locate boundary (guardrail: provenance, no typed
    addr) → seed complete entry state (guardrail: full reg_file) → pick mode
    (guardrail: encoded switch) → check mem[] backing (guardrail: blind-leg
    detection) → [CHECKPOINT alias-vs-compute] → classify hybrid steps
    (guardrail: symbol-preserving) → [CHECKPOINT which-static] → emit with
    parity gate (guardrail: hard parity, no critic-only close).

    The two checkpoints are the agent's real judgments; everything else is
    contract-enforced. Target-specific addresses live in case config, never
    here."""
    steps = (
        SetupStep(
            order=1, name="locate_boundary", contract="C1 boundary-via-provenance",
            guardrail="seed & sink located via watch_first_write / DFG / "
                      "sink_validation; bind_boundary rejects assumed addresses",
        ),
        SetupStep(
            order=2, name="seed_entry_state", contract="C2 entry-completeness",
            guardrail="symbolize the full reg_file + pointed buffers; an empty / "
                      "single-address seed is rejected",
        ),
        SetupStep(
            order=3, name="pick_mode", contract="dual-mode switch",
            guardrail="forward vs backward-alias decided by pick_mode's encoded "
                      "criterion (sym_propagated + opacity), not a hunch",
        ),
        SetupStep(
            order=4, name="check_mem_backing", contract="C4 mem[]-backing",
            guardrail="staging window audited for mem[]; a blind memory leg is "
                      "flagged before backtracing (re-capture, don't guess)",
        ),
        SetupStep(
            order=5, name="alias_vs_compute", contract="C4 / judgment",
            guardrail="",
            checkpoint=Checkpoint(
                name="alias_vs_compute",
                question="For each unresolved byte: is it an ALIAS of a seed byte "
                         "(materialized copy) or a real COMPUTATION (non-linear "
                         "transform)? Diff templates exhausted → it is a computation.",
                why="alias bytes close via the backward spine; computation bytes "
                    "need the transform recovered. Mislabeling a computation as an "
                    "alias degrades it to a cross-run-diff placeholder (false close).",
            ),
        ),
        SetupStep(
            order=6, name="classify_hybrid_steps", contract="C3 symbol-preserving",
            guardrail="every step reading a SymVar (incl load/store pairs at the "
                      "real EA) is modeled symbolically; only sym-independent "
                      "steps are concrete-synced",
        ),
        SetupStep(
            order=7, name="which_static", contract="judgment",
            guardrail="",
            checkpoint=Checkpoint(
                name="which_static",
                question="Which bytes are STATIC (constant across all inputs) vs "
                         "input-dependent? Confirm with the gold inputs, not one run.",
                why="a byte static in one run may vary across inputs; baking a "
                    "per-run constant into the formula passes one parity and fails "
                    "the rest.",
            ),
        ),
        SetupStep(
            order=8, name="emit_python", contract="emit + parity gate",
            guardrail="emit plain Python; parity_min is a HARD gate — gold N/N or "
                      "the recovery is not closed (no critic-only close)",
        ),
    )
    return SetupSymexPlan(steps=steps, determinism_note=_DETERMINISM_NOTE)


# ---------------------------------------------------------------------------
# Executing driver (Level 1 thin) — run the plan instead of returning a checklist.
#
# build_setup_symex_plan() returns the 8-step checklist but does NOT run it, so
# every case re-hand-wrote the same orchestration glue (run_hash2236.py,
# run_body_closure_fill_emit.py, …): load trace → run the steps → fill backing
# from the audit → record → emit. drive() executes that glue ONCE, so the agent
# supplies only a CaseConfig + a triton_runner and never rewrites the driver.
#
# What is utov-enforced (no longer the agent's to redo or fudge each round):
#   - the step ORDER (build_setup_symex_plan)
#   - the backing GATE is check_mem_backing's unified criterion — drive never
#     bypasses it (the blind_pcs==0 hand-bypass anti-pattern cannot recur here);
#     a blind closure short-circuits to an honest report, it does NOT emit a stub
#   - same-execution determinism (exec_identity.ref threaded as trace_exec_id and
#     stamped onto the backing)
#   - recording follows the recording policy: durable findings + ONE run-summary
#     roll-up, never a per-step trail; the stamped view is refreshed
# What stays the agent's (surfaced, never auto-decided):
#   - the two Checkpoints (alias_vs_compute / which_static) — resolved via
#     `decisions` / `on_checkpoint`, else drive returns a DrivePause
#   - symex execution (`triton_runner`, Level 1) and the recovered expression
#   - target-specific addresses / window / backing source (CaseConfig)
# ---------------------------------------------------------------------------


