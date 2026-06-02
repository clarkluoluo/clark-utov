"""Set-up symex primitive — the four contracts of opaque symbol recovery.

Origin: the VMP cipher-body case (`case-vmp-cipher-body-163511`). The cheap
methods (alias / cross-run diff) all fell through (resolved 0/65) which *proved*
the load-bearing gap: utov had no **set-up symbolic execution**. s3_triton marks
symex as P1.5 / opt-in / unbuilt — this module is the Tier-1 scaffold that fills
it, assembled from primitives utov already ships rather than greenfielded.

What "set-up symex" actually is: not "run Triton". The case showed forward symex
walls on three repeated mistakes, all the SAME root — **the symbolic boundary was
bound to the wrong place**. The primitive encodes the four contracts that, once
held, let the agent stand up a sound symbolic execution. Each contract wires an
existing utov primitive; the target-specific detail (concrete EAs, stack layout,
where the input sits) stays in case config, never in the primitive.

The four contracts (target-agnostic — that is the whole point of investing):

  1. boundary binding via PROVENANCE, not assumed addresses
     (`locate_boundary` → watch_first_write / DFG / sink-validation specs).
  2. entry-state COMPLETENESS — full reg_file + pointed buffers, not one addr
     (`seed_entry_state`).
  3. SYMBOL-PRESERVING hybrid — anything that reads a SymVar must be modeled
     symbolically (incl. load/store pairs at their real EA); concrete-sync only
     for sym-independent steps (`classify_hybrid_step`).
  4. mem[] BACKING — the alias spine's substrate. A trace with no mem[] over the
     materialization / staging window blinds the memory leg and the backtrace
     degrades to a cross-run-diff guess (`check_mem_backing`).

Plus the dual-mode switch (`pick_mode`): transparent path → forward symbolic
propagation; opaque path (VMP dispatch / concrete-overwrite) → backward alias
materialization. The switch criterion is encoded, not left to a hunch.

Delivery form is **library + a guard-railed template** (`build_setup_symex_plan`),
NOT a blank code generator. A blank generator filled wrong just accelerates the
mistake. So anchors must come from `locate_boundary` (no hand-typed addresses),
the mode comes from `pick_mode` (not a hunch), the same-execution / determinism
guard is built in, and the genuine judgments the agent must make (alias vs
compute, which bytes are static) are surfaced as explicit Checkpoints rather than
silently decided for it — `is_judgment` stays with the agent.

See `vmp-formula-induction-stuck.md` (four contracts + dual-mode regs),
`provenance-single-taint-spine.md` (the backtrace foundation this builds on),
and `case-library-with-index.md` (the template-rides-the-agent rationale).

Independent toggle: ``UTOV_SETUP_SYMEX=off|0|false|no``.
"""


from ._config import *  # noqa: F401,F403
from ._boundary import *  # noqa: F401,F403
from ._entry_state import *  # noqa: F401,F403
from ._hybrid import *  # noqa: F401,F403
from ._mem_backing import *  # noqa: F401,F403
from ._mode import *  # noqa: F401,F403
from ._emit import *  # noqa: F401,F403
from ._parity import *  # noqa: F401,F403
from ._seed_independence import *  # noqa: F401,F403
from ._emit_selfcheck import *  # noqa: F401,F403
from ._lint import *  # noqa: F401,F403
from ._plan import *  # noqa: F401,F403
from ._driver import *  # noqa: F401,F403
from ._mem_backing import _addr_regs  # noqa: F401
from ._emit_selfcheck import _eval_emitted_on_seed  # noqa: F401

__all__ = [
    # config
    "SetupSymexConfig",
    "SetupSymexDisabled",
    # C1 boundary
    "BoundaryRole",
    "BoundaryEnd",
    "BoundaryPlan",
    "BoundaryNotProvenanceLocated",
    "LOCATED_WATCH",
    "LOCATED_DFG",
    "LOCATED_SINK_VALIDATION",
    "LOCATED_ASSUMED",
    "locate_boundary",
    "bind_boundary",
    # C2 entry state
    "EntryStateSpec",
    "IncompleteEntryState",
    "seed_entry_state",
    "derive_window_symbolic_regs",
    "MemLiveIn",
    "derive_window_mem_live_in",
    "ConcreteBacking",
    "ConcreteBackingConflict",
    "build_concrete_backing",
    # C3 hybrid
    "HybridDecision",
    "classify_hybrid_step",
    # C4 mem backing
    "MemBackingReport",
    "check_mem_backing",
    "AddressLeg",
    "AddressClosureReport",
    "audit_address_closure",
    # dual mode
    "SymexMode",
    "OpacitySignals",
    "ModeDecision",
    "estimate_opacity",
    "pick_mode",
    # emit
    "EmitIntent",
    "emit_python",
    # per-handler / window multi-vector parity
    "ParityVector",
    "ParityVectorReport",
    "check_parity_vectors",
    # pre-symex seed-independence gate
    "SeedIndependenceReport",
    "check_seed_independence",
    # G4 emit self-check (recovered F must reproduce its own trace)
    "EmitSelfCheckReport",
    "check_emit_self_consistency",
    # pre-flight parity/self-check wiring lint (spec #3)
    "LintFinding",
    "LintReport",
    "LintContext",
    "Invariant",
    "INVARIANTS",
    "register_invariant",
    "lint_parity_inputs",
    "lint_case_config",
    # template
    "Checkpoint",
    "SetupStep",
    "SetupSymexPlan",
    "build_setup_symex_plan",
    # executing driver (Level 1)
    "CaseConfig",
    "DrivePause",
    "DriveResult",
    "drive",
]
