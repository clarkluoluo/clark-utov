"""CVD data model: candidate/state/budget, verdict, and plugin interfaces."""
from __future__ import annotations

import abc
import traceback
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..types import Instruction, MemSnapshot


def _truncated_traceback(limit_frames: int = 12, max_chars: int = 4000) -> str:
    """Capture the current exception's traceback, keeping the top frames + the
    raising line (invariant 4: truncate but preserve the locating frames). The
    tail (which carries the deepest frame where the exception was raised) is
    always retained; only the middle is elided if the trace is very deep."""
    tb = traceback.format_exc()
    if not tb:
        return ""
    lines = tb.splitlines(keepends=True)
    if len(lines) > limit_frames * 2 + 4:
        head = lines[: limit_frames + 1]            # header + top frames
        tail = lines[-(limit_frames + 1):]          # raising frame + exception line
        lines = head + ["  ...<traceback truncated>...\n"] + tail
        tb = "".join(lines)
    if len(tb) > max_chars:
        # keep the head (call origin) and the tail (the actual raise site).
        keep = max_chars // 2
        tb = tb[:keep] + "\n...<traceback truncated>...\n" + tb[-keep:]
    return tb


# --- ROI weights (CVD_DESIGN §2 base_value; §11 makes the rest dynamic) ------
BASE_VALUE = {
    "snapshot_eq_expected": 5.0,
    "located_real_sink":    4.0,
    "provenance":           4.0,
    "abi_return":           3.0,
    "output_buffer":        3.0,
    "agent_submission":     1.0,   # neutral base; credibility (§B/§C) does the ordering
    "write_cluster":        1.0,
    "scratch_like":         0.3,
}
_SURPRISE_CAP = 4.0
_STALL_GAIN = 0.5


class CvdOutcome(str, Enum):
    SUCCESS = "SUCCESS"
    TERMINAL = "TERMINAL"
    EXTENSION_REQUEST = "EXTENSION_REQUEST"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"   # max_candidates hit (verify loop)
    GENERATION_BUDGET_EXHAUSTED = "GENERATION_BUDGET_EXHAUSTED"  # generation/backtrace
                                            # phase hit a budget (backtrace depth/breadth
                                            # or candidate cap) — same model as
                                            # BUDGET_EXHAUSTED, applied UPSTREAM of verify.
    BUDGET_PAUSE = "BUDGET_PAUSE"           # a T2/side-effect estimate exceeds budget (§10.2)
    PENDING_JUDGMENT = "PENDING_JUDGMENT"   # a Verifier surfaced an agent-judgment checkpoint
    COLLECTED = "COLLECTED"                 # collect mode: one run enumerated the whole gap map


# --- Candidate / state / budget ---------------------------------------------

@dataclass
class Candidate:
    kind: str
    locus: int
    signal: str
    entry_reason: str = ""
    representation: str = "raw"
    base_value: float = 1.0
    # CVD_ADDENDUM §D: provenance + a credibility WE computed from evidence (§B).
    # credibility orders verification (§C); it never grants trust — still verified.
    provenance: str = "observed"     # observed|agent_evidence|agent_finding|agent_hypothesis
    credibility: float = 1.0
    payload: dict = field(default_factory=dict)   # tool-specific data (e.g. recovered bytes)
    score: float = 0.0
    elim_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "locus": f"0x{self.locus:x}", "signal": self.signal,
                "entry_reason": self.entry_reason, "representation": self.representation,
                "base_value": self.base_value, "provenance": self.provenance,
                "credibility": self.credibility, "payload": self.payload,
                "score": round(self.score, 3), "elim_reason": self.elim_reason}

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate":
        return cls(kind=d["kind"], locus=int(d["locus"], 16), signal=d["signal"],
                   entry_reason=d.get("entry_reason", ""),
                   representation=d.get("representation", "raw"),
                   base_value=d.get("base_value", 1.0),
                   provenance=d.get("provenance", "observed"),
                   credibility=d.get("credibility", 1.0),
                   payload=dict(d.get("payload", {})))


@dataclass
class CvdState:
    items: list[Instruction]
    expected: bytes
    snapshots: list = field(default_factory=list)
    window: tuple[int, int] | None = None
    obs_scope: int = 0
    call_events: list = field(default_factory=list)   # from T-intake calltrace reader

    def scoped_items(self) -> list[Instruction]:
        if self.window is None:
            return self.items
        lo, hi = self.window
        return [ins for ins in self.items if lo <= ins.idx <= hi]

    def scoped_snapshots(self) -> list:
        return self.snapshots if self.obs_scope >= 1 else []


@dataclass
class CvdBudget:
    # verify loop (existing).
    max_candidates: int = 64
    max_widen: int = 4
    # generation / provenance-backtrace phase (dev-recovery-generation-budget-spec).
    # Same budget MODEL as the verify loop, extended UPSTREAM to the phase that
    # 8fad88f added (provenance backtrace + on-path candidate generation), which was
    # previously unbounded. Defaults are intentionally generous: a budget-internal
    # ("small") case never trips them — only a genuinely long trace / a very wide
    # producer fan-out / an explosion of on-path candidates does (invariant 7). No
    # F0 number lives here; these are universal engineering ceilings, parameterised.
    #   * max_backtrace_depth   — provenance backtrace hop/step ceiling (how far back
    #                             the producer chain is followed). Plumbed into
    #                             oracle_provenance.trace_provenance's max_steps.
    #   * max_backtrace_breadth — provenance backtrace frontier (BFS stack) ceiling:
    #                             how many pending producers may be queued at once
    #                             (the branch fan-out). Caps O(huge) producer trees.
    #   * max_gen_candidates    — on-path candidate window ceiling: keep the top-N by
    #                             ROI (base_value), drop the long tail, report it.
    #   * band_gap_threshold    — on-path BAND coalescing (dev-recovery-bands spec
    #                             Req4): consecutive producer-chain idxs whose gap is
    #                             <= this are merged into ONE band candidate
    #                             (window=[start,end]) instead of one per idx — so a
    #                             7191-idx chain becomes ~2000 bands, not 7191 single-
    #                             idx windows that exhaust the cap. Universal (a small
    #                             gap = same contiguous slice); no case idx baked here.
    #   * max_composite_symex_items — composite recovery (dev-recovery-bands spec
    #                             Req6): the symex item-budget for COMBINING adjacent
    #                             on-path bands. A composite whose estimated symex
    #                             work (total band span × items) exceeds this gets a
    #                             COMPOSITE_TOO_EXPENSIVE terminal + a cost estimate
    #                             (a comfortable exit) instead of a >90s hang. Generous
    #                             default; only a genuinely huge combined window trips
    #                             it (the tc2 idx28444..41354 巨窗). No case idx baked.
    #   * composite_aggregation_min — composite-aggregation trigger (dev-recovery-
    #                             evidence-integration-spec A3): the minimum count of
    #                             SAME-chain_id BAND_PARITY_FAIL bands that makes the
    #                             collect layer stop enumerating more single bands and
    #                             plan a COMPOSITE over the whole same-chain group. Data-
    #                             driven default 2 — identical to the planner's own
    #                             "≥2 adjacent bands → COMPOSITE_REQUIRED" rule (≥2 is
    #                             the universal "a group, not a brick" floor); NO case
    #                             (F0/tc2) number is baked here. Parameterised so a run
    #                             can tighten/loosen the floor.
    max_backtrace_depth: int = 100_000
    max_backtrace_breadth: int = 4_096
    max_gen_candidates: int = 256
    band_gap_threshold: int = 1
    max_composite_symex_items: int = 8_192
    composite_aggregation_min: int = 2
    #   * max_recapture_reentries — verifier-internal recapture closure (B2,
    #                             dev-recovery-verifier-internal-recapture-spec): the
    #                             number of times run_recovery may drive the recapture
    #                             loop and RE-ENTER collect with the freshly-captured
    #                             same-run snapshots before declaring the closure
    #                             budget spent. Each re-entry runs ONE run_recapture_loop
    #                             (which itself runs max_rounds reruns) + one collect.
    #                             Generous default; a budget-internal case closes in 1-2
    #                             re-entries. Hitting the cap WARNs (never silent) and
    #                             returns the last collect result. No case number baked.
    #   * max_output_determinism_reruns — output-determinism probe (P6, dev-output-
    #                             determinism-evidence-spec): the number K of SAME-input
    #                             reruns run_recovery performs (empty observe_points) to
    #                             OBSERVE whether the runner's output is stable across K.
    #                             A bounded empirical observation (observed-stable-across-K),
    #                             NOT a determinism proof. Default small (3). No case number
    #                             baked; parameterised so a run can probe deeper. Hitting a
    #                             runner record cap during the probe propagates truncated +
    #                             WARN (never silent, same discipline as B2).
    max_recapture_reentries: int = 4
    max_output_determinism_reruns: int = 3


# --- Verdict + plugin interfaces (CVD_PLUS_DESIGN §2) ------------------------

class VStatus(str, Enum):
    CONFIRMED = "confirmed"
    ELIMINATED = "eliminated"
    TERMINAL = "terminal"
    PENDING = "pending"       # not a capability gap — an agent-judgment checkpoint
                              # (e.g. setup_symex.drive returned a DrivePause)


@dataclass
class Verdict:
    status: VStatus
    reason: str = ""
    terminal_kind: str = ""
    success: bool = False          # a TERMINAL verdict that IS a goal-reaching result
    evidence: dict = field(default_factory=dict)
    located_base: int | None = None
    capability_request: str = ""
    spawn: list = field(default_factory=list)   # list[Candidate]
    error_detail: str = ""   # traceback of a tool_error (governance §7), for routing


@dataclass
class Terminal:
    kind: str
    evidence: dict = field(default_factory=dict)
    sink_base: int | None = None
    capability_request: str = ""
    success: bool = False


class Verifier(abc.ABC):
    name = "verifier"; version = "0"; owner = "core"
    @abc.abstractmethod
    def applies(self, c: Candidate, state: CvdState) -> bool: ...
    def cost(self, c: Candidate, state: CvdState) -> float: return 1.0
    @abc.abstractmethod
    def verify(self, c: Candidate, state: CvdState) -> Verdict: ...


class CandidateGenerator(abc.ABC):
    name = "generator"; version = "0"; owner = "core"; kind = ""
    @abc.abstractmethod
    def generate(self, state: CvdState) -> list[Candidate]: ...


class Transform(abc.ABC):           # encoding edge (CVD_DESIGN §10.1) — interface only
    name = "transform"; version = "0"; owner = "core"
    @abc.abstractmethod
    def detect(self, region: bytes, state: CvdState) -> bool: ...
    @abc.abstractmethod
    def forward(self, raw: bytes) -> bytes: ...
    @abc.abstractmethod
    def inverse(self, encoded: bytes) -> bytes: ...


class EscalationRule(abc.ABC):      # an E-table row (CVD_DESIGN §11.2) — interface only
    name = "rule"; version = "0"; owner = "core"
    @abc.abstractmethod
    def trigger(self, c: Candidate, state: CvdState, history: dict) -> bool: ...
    @abc.abstractmethod
    def escalate(self, c: Candidate, state: CvdState) -> dict: ...


class TerminalClassifier(abc.ABC):
    name = "terminal"; version = "0"; owner = "core"
    @abc.abstractmethod
    def classify(self, state: CvdState) -> Terminal | None: ...


@dataclass
class Artifact:
    """A captured runner output to canonicalise at PLACE (CVD_MOUNT_POLICY §3
    T-intake). kind selects the Reader; text is the raw content."""
    kind: str            # "calltrace" | "hook_dump" | ...
    text: str = ""


@dataclass
class ReadResult:
    snapshots: list = field(default_factory=list)     # MemSnapshot[]
    call_events: list = field(default_factory=list)   # CallEvent[]
    items: list = field(default_factory=list)         # Instruction[]


class Reader(abc.ABC):
    """T-intake plugin (CVD_MOUNT_POLICY §3): turn a raw runner artifact into a
    canonical utov input BEFORE the candidate loop. Produces inputs, not
    candidates/verdicts. Default-ON, self-gating on artifact presence,
    side-effect-free (parses data we already hold)."""
    name = "reader"; version = "0"; owner = "core"
    @abc.abstractmethod
    def detect(self, artifact: Artifact) -> bool: ...
    @abc.abstractmethod
    def read(self, artifact: Artifact) -> ReadResult: ...


@dataclass
class Registry:
    verifiers: list = field(default_factory=list)
    generators: list = field(default_factory=list)
    transforms: list = field(default_factory=list)
    rules: list = field(default_factory=list)
    terminals: list = field(default_factory=list)
    readers: list = field(default_factory=list)

    def register(self, obj) -> "Registry":
        if isinstance(obj, Verifier):
            self.verifiers.append(obj)
        elif isinstance(obj, CandidateGenerator):
            self.generators.append(obj)
        elif isinstance(obj, Transform):
            self.transforms.append(obj)
        elif isinstance(obj, EscalationRule):
            self.rules.append(obj)
        elif isinstance(obj, TerminalClassifier):
            self.terminals.append(obj)
        elif isinstance(obj, Reader):
            self.readers.append(obj)
        else:
            raise TypeError(f"not a registrable CVD plugin: {type(obj).__name__}")
        return self


@dataclass
class ExtensionRequest:
    missing_kind: str          # verifier | generator | transform | rule | terminal | observation
    why: str
    suggestion: str = ""
    where: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"missing_kind": self.missing_kind, "why": self.why,
                "suggestion": self.suggestion, "where": self.where}


@dataclass
class CvdResult:
    outcome: CvdOutcome
    verdict: str = ""
    sink_base: int | None = None
    provenance: dict | None = None
    capability_request: str = ""
    # Req2 G3: ONE uniform machine-readable block-reason key on every BLOCKED
    # terminal result (TERMINAL / non-success). A consumer reads ``block_why`` to
    # know WHY the run blocked without guessing reason vs capability_request vs
    # provenance. None on a non-blocked outcome (SUCCESS / collect / EXTENSION_…),
    # then omitted from to_dict (invariant 7: no noise on the green path).
    block_why: str | None = None
    extension_request: dict | None = None
    log: list[dict] = field(default_factory=list)
    checkpoint: dict | None = None      # serialized run state (resume)
    manifest: dict | None = None        # the pre-drive RunManifest (§6)
    budget_estimate: dict | None = None # T2/side-effect (time,disk,tokens) on a BUDGET_PAUSE
    # collect mode (the "one run lists the whole gap map" view): instead of
    # returning at the first single-candidate gap, the driver accumulates them.
    extension_requests: list[dict] = field(default_factory=list)  # capability gaps, per candidate
    pending_judgments: list[dict] = field(default_factory=list)   # agent-judgment checkpoints
    confirmed: list[dict] = field(default_factory=list)           # candidates that CONFIRMED
    # run-level out-layer evidence: the cohort-load report (which cohort vector was
    # fed bare because its _mem.jsonl sibling was missing). Surfaced at the TOP of
    # the gap map so the degradation is visible regardless of window outcome
    # (PENDING / CONFIRMED / opaque TERMINAL) — degradation is allowed, silence is
    # not. None → no cohort load layer ran (pre-loaded cohort / non-recovery run);
    # the field is then omitted entirely (no noise, today's behaviour).
    cohort_load: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "outcome": self.outcome.value, "verdict": self.verdict,
            "sink_base": None if self.sink_base is None else f"0x{self.sink_base:x}",
            "provenance": self.provenance, "capability_request": self.capability_request,
            "extension_request": self.extension_request, "log": self.log,
            "checkpoint": self.checkpoint, "manifest": self.manifest,
            "budget_estimate": self.budget_estimate,
            "extension_requests": self.extension_requests,
            "pending_judgments": self.pending_judgments,
            "confirmed": self.confirmed,
        }
        if self.cohort_load is not None:
            d["cohort_load"] = self.cohort_load
        if self.block_why is not None:
            d["block_why"] = self.block_why
        return d
