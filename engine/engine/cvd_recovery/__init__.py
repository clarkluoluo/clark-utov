"""Recovery-as-a-CVD-run — register the symbolic-recovery pipeline INTO CVD.

The set-up symex recovery was a hand-coded line inside ``setup_symex.drive`` with
its own parallel verification, debugged window-by-window: each conversation round
fixed one gap, the next round hit the next. That is whack-a-mole. CVD already is
the machine for this (``cvd.py`` docstring): it "enumerates typed candidates …
always has a defined next move … emits a structured EXTENSION_REQUEST, never a
silent stall". This module makes recovery *one CVD run*: a candidate per window,
one heavy Verifier that runs the whole ``drive()`` per window, and a
TerminalClassifier that claims the genuine dead ends — so a single ``collect``
run lists the *whole* gap map at once instead of surfacing one gap per round.

Role split (CVD_PLUS_DESIGN §9): these are plugins registered in a per-run
Registry; the driver code does not change. The recovery primitives stay pure
functions — this module only *adapts* them to the plugin interfaces.

Zero case-specific knowledge (utov-arch-index invariant 2/6): no concrete
address / value / handler-id / case name lives here. The per-window geometry
arrives via the candidate ``payload`` (built by the generator from a
``CoverageMap`` / ``InputDependenceMap``); the base case identity + the symex
runner are injected into the Verifier by whoever builds the registry — they live
in the agent's config / the fixture, never in this primitive.
"""


from ._cohort import *  # noqa: F401,F403
from ._generator import *  # noqa: F401,F403
from ._compose import *  # noqa: F401,F403
from ._verifier import *  # noqa: F401,F403
from ._registry import *  # noqa: F401,F403
from ._driver import *  # noqa: F401,F403

from ._cohort import _OnpathBandRegistry  # noqa: F401
from ._cohort import _compact  # noqa: F401
from ._cohort import _drive_evidence  # noqa: F401
from ._compose import _classify_drive_result  # noqa: F401
from ._compose import _classify_block_kind  # noqa: F401
from ._compose import _dfg_symbol_trace  # noqa: F401
from ._driver import _byte_variance_ranges  # noqa: F401
from ._driver import _reconcile_anchors  # noqa: F401
from ._driver import _wants_recapture  # noqa: F401
from ._driver import _OUTPUT_DET_FORBIDDEN_WORDS  # noqa: F401

__all__ = [
    "RECOVER_WINDOW",
    "OPAQUE_STAGING_FRONTIER",
    "SIG_PROVENANCE_ONPATH",
    "SIG_PROVENANCE_OFFPATH_VARIANCE",
    "SIG_RECAPTURE_DIRECTIVE",
    "SIG_PROVENANCE_UNANCHORED",
    "SIG_PROVENANCE_BLOCKED_UNPLACEABLE",
    "SIG_GENERATION_BUDGET_EXHAUSTED",
    "TERMINAL_BAND_PARITY_FAIL",
    "TERMINAL_COMPOSITE_REQUIRED",
    "TERMINAL_COMPOSITE_TOO_EXPENSIVE",
    "TERMINAL_MEMORY_DISPOSITION_MISSING",
    "TERMINAL_MEM_SINK_UNPLACEABLE",
    "MEM_SINK_UNPLACEABLE_NEEDED",
    "derive_mem_sink_interval",
    "SRC_OUTPUT_PROVENANCE",
    "MemDispositionRec",
    "OUTPUT_DET_NO_ADAPTER",
    "probe_output_determinism",
    "load_cohort_traces",
    "recommend_mem_disposition",
    "RecoveryWindowGenerator",
    "RecoverWindowVerifier",
    "RecoveryTerminalClassifier",
    "recovery_registry",
    "run_recovery",
    "compose_band_transforms",
    "estimate_composite_cost",
    "plan_composite_recovery",
]
