"""CVD — Candidate-Verification Driver, registry-driven (CVD_DESIGN + CVD_PLUS_DESIGN).

The driver layer above the verification primitives. It enumerates typed candidates,
orders them by a dynamic surprise-weighted ROI, verifies each, and always has a
defined next move — confirm→expand, miss→pop next, scope-exhausted→widen,
all-exhausted→classify-terminal. The human (agent) enters only at a RETURN
boundary; every step is recorded (auditable).

CVD-Plus property (CVD_PLUS_DESIGN §9): the driver dispatches ONLY over plugin
interfaces held in a per-run Registry — it never hardcodes a concrete tool or a
closed enum. #1 sink-validator and #3 provenance are registered Verifiers; SinkGen
is a CandidateGenerator; the dead-end classifier is a TerminalClassifier. Adding a
tool = appending to the Registry; driver code does not change. When no Verifier
applies, or no TerminalClassifier claims a dead end, the driver emits a structured
EXTENSION_REQUEST (never a silent stall). Run state is serializable (resume).

Transform-aware retreat (CVD_DESIGN §10.6, base64 raw↔framed) is now ACTIVE: a
parameterised FramingTransform (framing_transform.py) implements raw↔framed
transcoding — including bit-level drop-N (sub-byte shift) — and is registered in
default_registry(). Still deferred: the full E1–E6 escalation table — that
interface (EscalationRule) exists and is registry-backed, but carries no
non-trivial instances yet.
"""

from ._model import *  # noqa: F401,F403
from ._registry import *  # noqa: F401,F403
from ._driver import *  # noqa: F401,F403
from ._registry import _roi  # noqa: F401

__all__ = [
    "Candidate", "CvdState", "CvdBudget", "CvdResult", "CvdOutcome",
    "Verdict", "VStatus", "Terminal", "ExtensionRequest",
    "Verifier", "CandidateGenerator", "Transform", "EscalationRule",
    "TerminalClassifier", "Registry", "default_registry", "run_cvd", "resume",
    "AgentSubmission", "PlacementError", "place", "evaluate_credibility",
    "Artifact", "Reader", "ReadResult", "export_gap_map", "run_cvd_collect_to_json",
]
