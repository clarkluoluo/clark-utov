"""Shared data types used across stages."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class MemOp:
    rw: str        # "r" or "w"
    addr: int
    val: int
    size: int      # bytes


@dataclass(frozen=True, slots=True)
class MemSnapshot:
    """A memory-region observation captured out of band, normalised to utov's
    canonical shape: a concrete address + the bytes seen there + a source marker.

    Distinct from :class:`MemOp` — a snapshot is NOT an executed read/write step
    in the instruction stream; it is a captured view of memory contents (e.g. a
    runner dumping the output buffer at a hook). Read-only / observation only.

    Adapters fill these from runner-specific captures (a register-pointed dump,
    a memcpy destination, etc.); the engine never parses runner formats — it only
    consumes this canonical shape. See contracts/runner_interface.md §3.7.
    """
    addr: int
    data: bytes
    label: str = ""              # optional tag, e.g. "output_buffer"
    source: str = "snapshot"     # provenance marker (kept read-only)
    truncated: bool = False      # True iff derived from an incomplete (cap-hit)
    #                              runner ledger — NOT complete/clean provenance.
    #                              Mirrors RerunResult.truncated; see
    #                              contracts/runner_interface.md §3.7 / §rerun.
    execution_id: object = None  # SAME-EXECUTION provenance token (Req2 G1). A
    #                              snapshot captured in one rerun carries that
    #                              rerun's token; ``None`` = execution provenance
    #                              not stamped. A closure whose backing snapshot
    #                              set carries >= 2 DISTINCT non-None tokens mixes
    #                              snapshots ACROSS reruns (a different nonce per
    #                              execution) → it is NOT one real production →
    #                              the PROVENANCE_WATCH_BATCH_G1 terminal fires
    #                              (recapture_loop.assert_same_execution). Default
    #                              None keeps every existing snapshot byte-for-byte
    #                              (invariant 7); the construct (one rerun per
    #                              round) stamps it, B2 keeps it single-valued.


@dataclass(frozen=True, slots=True)
class Instruction:
    """One trace record. Matches contracts/runner_interface.md §2.1."""
    idx: int
    pc: int
    bytes_: bytes
    mnemonic: str
    regs_read: dict[str, int]
    regs_write: dict[str, int]
    mem: tuple[MemOp, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class TargetMeta:
    """Runner-provided metadata. See contracts/runner_interface.md §4."""
    target_name: str
    arch: str
    algo_entry_pc: int
    algo_exit_pc: int
    input_length: int | None      # None = variable
    output_length: int
    algo_symbol: str | None = None
    emulator_name: str | None = None      # e.g. "unidbg" / "qiling" / "frida"
    emulator_version: str | None = None   # e.g. "0.9.9"
    # Runtime capability opt-in (runner_interface §3.3): the names of runner
    # abilities this target declares it implements (e.g. "observe_point"). Wins
    # over an adapter's static ``CAPABILITIES`` via :func:`block_cause.oracle_from_adapter`.
    # Lets a project declare an ability without an engine-side code change.
    capabilities: tuple[str, ...] = ()
