"""Capability-token registry — what THIS build's source actually provides.

Origin (dev-recovery-blockkind-coverage-symbol-trace.md §需求2): a build stamp's
``commit_ok`` says only "the build is self-consistent / doctor passed" — it does
NOT say the build carries a given recovery feature. A pre-feature build (commit
behind the feature landing) stamps ``commit_ok: true`` all the same, so an agent
runs a stale build, hits the pre-fix behaviour, and burns a whole round on a
串台 (cross-build confusion) without any warning.

The fix is a *capability token*: each feature milestone declares a STABLE token
in source. The token's PRESENCE in the source IS the proof that this build has
the capability — so :func:`collect_build_capabilities` simply returns
:data:`PROVIDED_CAPABILITIES` (the token is in the source ⇔ the build provides
it; there is no git lookup, no commit math). A requirement side
(:data:`TERMINAL_REQUIRES`) maps a terminal outcome to the capabilities it needs;
:func:`check_coverage` does the universal ``required ⊆ build`` subset test and
hands back a degrade WARN when something is missing.

Discipline (utov-arch-index):
  * invariant 1 — :func:`collect_build_capabilities` / :func:`check_coverage` are
    pure functions; the WARN is surfaced at the output layer, never a gate input.
  * invariant 2/6 — a token is a CAPABILITY marker (``opaque_forward_v1``), never
    a case name / address / handler id; :data:`TERMINAL_REQUIRES` keys are the
    structural terminal kinds, not cases.
  * invariant 7 — ``capabilities`` / ``coverage_ok`` are NEW fields; when every
    required token is present (coverage_ok True, no missing) the output is
    byte-for-byte today's. Nothing here feeds close / parity / G4 / seed.
  * A7 — the declaration side + a universal subset test; no commit is hardcoded.
"""

from __future__ import annotations

from typing import Iterable

# --------------------------------------------------------------------------- #
# Provided capabilities — the tokens THIS build's source carries.
#
# A token lands here in the SAME change that lands the feature it marks (so the
# token is present iff the source provides the capability — that is the whole
# trick: no git, the source is the evidence). The commit refs are documentation
# of when each landed, NOT a runtime check.
# --------------------------------------------------------------------------- #
PROVIDED_CAPABILITIES: frozenset[str] = frozenset({
    # opaque-staging Phase 2(i): forward symbolic loads through a staging buffer
    # (the clo_deferred first-run injection wiring). Landed b8976b7.
    "opaque_forward_v1",
    # the symbolic_forwards counter — "how many loads forwarded this run", the
    # "wired but never hit" observability signal. Landed b679f8a.
    "symbolic_forward_record_v1",
    # the opaque-staging re-續 fallback re-run (a non-clo_deferred staging window
    # whose first run forwarded nothing gets re-diagnosed + re-run once). 0388e3b.
    "opaque_forward_fallback_v1",
    # recovery terminal block_kind decision tree (this change's §需求1): the
    # 5-class mutually-exclusive split of the opaque merge point. Landed here.
    "recovery_block_kind_v1",
})


# --------------------------------------------------------------------------- #
# Required capabilities — what a terminal outcome NEEDS the build to provide.
#
# Keyed on the structural terminal kind / block_kind (never a case). When a
# terminal of kind K is surfaced, the build is expected to carry every token in
# ``TERMINAL_REQUIRES[K]``; a missing token means the build pre-dates the fix the
# terminal's diagnosis relies on → likely a stale (串台) build → WARN.
# --------------------------------------------------------------------------- #
TERMINAL_REQUIRES: dict[str, frozenset[str]] = {
    # an opaque-staging terminal's forward-and-retry diagnosis only means what it
    # says on a build that actually has the forwarding wiring + the record line +
    # the fallback re-run. A build missing any of these "saw opaque" for a
    # different (pre-feature) reason → WARN.
    "opaque_staging": frozenset({
        "opaque_forward_v1",
        "symbolic_forward_record_v1",
        "opaque_forward_fallback_v1",
    }),
    # every block_kind value is produced by the §需求1 decision tree, so a build
    # that emits one must carry that capability (a build without it cannot have
    # produced a block_kind at all — this guards a hand-forged / stale stamp).
    "opaque_staging_block": frozenset({"recovery_block_kind_v1"}),
    "window_boundary_mismatch": frozenset({"recovery_block_kind_v1"}),
    "symbol_not_on_output_path": frozenset({"recovery_block_kind_v1"}),
    "emit_picked_constant": frozenset({"recovery_block_kind_v1"}),
    "undetermined_constant": frozenset({"recovery_block_kind_v1"}),
}


def collect_build_capabilities() -> frozenset[str]:
    """The capability tokens THIS build's source provides.

    The token is in the source ⇔ the build has the capability, so this is just
    :data:`PROVIDED_CAPABILITIES` (no git, no commit math — invariant per the
    spec note). A pure function (invariant 1)."""
    return PROVIDED_CAPABILITIES


def check_coverage(
    required: Iterable[str],
    *,
    build_caps: Iterable[str] | None = None,
) -> tuple[bool, frozenset[str], str | None]:
    """Universal coverage test: is ``required`` a subset of the build's tokens?

    Returns ``(coverage_ok, missing, warn)``:
      * ``coverage_ok`` — every required token is provided by the build;
      * ``missing`` — the required tokens the build lacks (empty when ok);
      * ``warn`` — ``None`` when ok, else a degrade message naming the gap (the
        consumer surfaces it; it is NEVER a gate input — invariant 7).

    ``build_caps`` defaults to :func:`collect_build_capabilities`; pass an
    explicit set to test a hypothetical / another build (fixtures use this).
    Pure / deterministic (invariant 1)."""
    req = frozenset(required)
    have = (frozenset(build_caps) if build_caps is not None
            else collect_build_capabilities())
    missing = frozenset(req - have)
    if not missing:
        return True, frozenset(), None
    miss_str = ", ".join(sorted(missing))
    warn = (
        f"build lacks capability {{{miss_str}}} required by this terminal — "
        "likely a pre-feature build; rebuild from a commit that has it (a "
        "commit_ok / self-consistent stamp does NOT imply the feature is present)."
    )
    return False, missing, warn


def coverage_for_terminal(
    terminal_kind: str,
    *,
    build_caps: Iterable[str] | None = None,
) -> tuple[bool, frozenset[str], str | None]:
    """:func:`check_coverage` for a terminal kind's declared requirements.

    A terminal_kind with no entry in :data:`TERMINAL_REQUIRES` requires nothing →
    always covered (``(True, frozenset(), None)``). Convenience for the consumer
    side so it need not look up the requirement mapping itself."""
    return check_coverage(
        TERMINAL_REQUIRES.get(terminal_kind, frozenset()),
        build_caps=build_caps)


def build_capability_stamp() -> dict[str, list[str]]:
    """A compact ``{capabilities: [...]}`` block a build-identity stamp can carry.

    Lets a stamp (export header ``exec_identity`` / doctor) advertise WHICH recovery
    features this build provides, so a downstream agent can verify a stamp's build
    actually carries a capability before trusting a terminal that needs it — the
    串台 guard the spec asks for. Additive: a caller merges this into its existing
    identity dict; existing stamps that do not are byte-for-byte unchanged."""
    return {"capabilities": sorted(collect_build_capabilities())}


__all__ = [
    "PROVIDED_CAPABILITIES",
    "TERMINAL_REQUIRES",
    "collect_build_capabilities",
    "check_coverage",
    "coverage_for_terminal",
    "build_capability_stamp",
]
