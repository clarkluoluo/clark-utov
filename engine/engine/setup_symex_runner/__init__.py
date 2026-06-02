"""Level-2 set-up symex runner — utov's own concolic engine.

The Level-1 path made the *agent* hand-write the symbolic runner
(``make_f0_handler_triton_runner`` + a hand-rolled ``_sym_emulate_mem``). That
runner was a hybrid: Triton for some steps, but a hand-rolled symbolic-memory
emulator for the rest — perpetually missing opcodes, and (the root flaw)
**force-concretizing a symbolic load** so the symbolic chain snapped. The result
was whack-a-mole: every handler stalled on a different un-modeled instruction
(端序 → ldrh → control-flow → mul/and/load-force).

The cure is NOT "chase instruction completeness" (you never catch up). It is:

  **bulk decoder (Triton) covers the big middle · an extensible ESCAPE HATCH
  covers the long tail.**

An instruction the bulk decoder can't model is **never** force-concretized and
**never** silently skipped — it is a BLOCK + a precise checkpoint
(:class:`UnmodeledInstruction`): "insn <opcode> (<mnemonic>) @ idx <Y> is not
modeled — supply its symbolic semantics." The agent fills that *one*
instruction's semantics; utov caches it in a persistent
:class:`SemanticsTable` (auto-handled next time, and a seed for a community
plugin) and continues. The hand-filled semantics still go through the
multi-vector parity gate (``setup_symex.check_parity_vectors``), so a wrong fill
is caught on the spot. This is the "挡不住就给充足提示" guard-rail applied at the
instruction layer.

Positioning (see ``todo/level2-symex-positioning.md``): the symex engine is a
commodity — utov routes every instruction to Triton's ``processing()`` and does
NOT re-implement instruction/memory semantics (the hand-rolled ``_sym_emulate_mem``
that re-implemented Triton, badly, was the anti-pattern). The runner is
deliberately thin; utov's value is *around* symex (where to symex, state set-up,
is-it-right parity, compose, reproducibility), not in symex itself. The escape
hatch fires ONLY when Triton's own ``processing()`` can't model an opcode (a true
long-tail blind spot), never to paper over a mis-configured main path. Per the
same doc, this table caches decode-level (state-independent) instruction
semantics, NOT per-step ``processing()`` output (state-dependent → caching it is
wrong); the handler-transform cache (path-aware key) is the compose layer's (Tier2).

Design seams:
- The per-instruction **decoder** is pluggable (:class:`StepDecoder`). The default
  is Triton (:class:`TritonStepDecoder`, behind ``is_available()``); tests drive a
  deterministic fake so the *framework* is exercised without Triton on the host.
- The runner conforms to the ``setup_symex.drive`` ``triton_runner`` protocol
  (``Callable[[dict], Mapping]``), so L1→L2 is a runner swap — ``drive``'s
  signature does not change.
- Computing gold parity needs the live oracle (target-specific), so the runner is
  parameterized by a ``gold`` callable the agent supplies; the framework never
  fabricates a parity number.

Triton itself is an optional dependency (engine-vs-fixture boundary): this module
imports cleanly without it; only :class:`TritonStepDecoder` needs it.
"""

from ._base import (
    triton_available,
    triton_unavailable_reason,
    opcode_hex,
    is_control_flow,
)
from ._semantics import (
    SEMANTICS_BINOPS,
    SEMANTICS_UNOPS,
    SemanticsParseError,
    SemanticsApplyError,
    parse_sexpr,
    validate_sexpr,
    InstructionSemantics,
    SemanticsTable,
    UnmodeledInstruction,
)
from ._audit import (
    DECODE_FAIL_RATE_THRESHOLD,
    SYMBOLIC_FORWARD_SITE_CAP,
    DecodeAudit,
    audit_window_decode,
)
from ._runner import (
    StepDecoder,
    RunnerResult,
    GoldFn,
    build_level2_runner,
    run_window,
    TritonStepDecoder,
)

__all__ = [
    "triton_available",
    "triton_unavailable_reason",
    "opcode_hex",
    "SEMANTICS_BINOPS",
    "SEMANTICS_UNOPS",
    "SemanticsParseError",
    "SemanticsApplyError",
    "parse_sexpr",
    "validate_sexpr",
    "is_control_flow",
    "InstructionSemantics",
    "SemanticsTable",
    "UnmodeledInstruction",
    "StepDecoder",
    "RunnerResult",
    "DECODE_FAIL_RATE_THRESHOLD",
    "DecodeAudit",
    "audit_window_decode",
    "run_window",
    "GoldFn",
    "build_level2_runner",
    "TritonStepDecoder",
]
