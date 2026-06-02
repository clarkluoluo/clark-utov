"""Trace readers + rerun adapters.

Consumes runner-produced output per contracts/runner_interface.md.
Does NOT implement any runner-side behavior (no unidbg, no anti-debug, etc).
"""

from __future__ import annotations

import abc
import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .aarch64_mem import parse_mem_ops
from .byte_order import (
    AARCH64_INSN_BYTES,
    CONVENTION_VOTE_SAMPLE,
    ConventionDetector,
)
from .types import Instruction, MemOp

if TYPE_CHECKING:
    from .capability import CodeHookRange, RegisterTrace
    from .types import MemSnapshot, TargetMeta

_log = logging.getLogger(__name__)


# JSON-RPC error codes that mean "the runner doesn't implement this method".
# -32601 is the standard JSON-RPC "Method not found"; -32603 is "Internal
# error" which some runners abuse for unsupported-op throws. Java runners that
# throw `UnsupportedOperationException` typically end up with one of these.
_JSONRPC_METHOD_NOT_FOUND = -32601
_JSONRPC_INTERNAL_ERROR   = -32603

# Substrings (case-insensitive) in the wire error message that also indicate
# a capability-missing condition rather than a real runtime error. Kept tight
# so we don't swallow real failures.
_CAPABILITY_MISSING_NEEDLES = (
    "unsupportedoperationexception",  # Java side default
    "not implemented",
    "notimplemented",
    "unsupported operation",
    "method not found",
    "method not supported",
    "not supported",
    "file mode",                       # explicit "I'm File mode" signal
)


def _is_capability_missing_error(code: int | None, message: str) -> bool:
    """True iff a runner's JSON-RPC error indicates 'this method isn't
    implemented' (File mode) rather than a genuine runtime failure.

    Matches BR-2 §13+ — without this translation, Live-mode SubprocessRunnerAdapter
    paired with a runner that throws UnsupportedOperationException would crash
    the conformance gate because `except NotImplementedError` doesn't catch the
    RuntimeError this method previously raised."""
    if code == _JSONRPC_METHOD_NOT_FOUND:
        return True
    msg_l = (message or "").lower()
    if any(needle in msg_l for needle in _CAPABILITY_MISSING_NEEDLES):
        return True
    return False


class TraceReader(abc.ABC):
    """Streaming trace iterator."""

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Instruction]:
        ...


# The mem-sidecar SEMANTIC FAMILY (dev-closure-evidence-layering-trap-state-spec,
# task 5): a mem sidecar may be named ``<stem>_mem.jsonl`` (the canonical name the
# main trace already uses) OR ``<stem>_mem_sidecar.jsonl`` (a common alternative a
# runner emits). Recognising the FAMILY — not one dead name — kills the
# construct-symmetry anti-pattern "name it EXACTLY _mem.jsonl or it's silently
# dropped" (feedback_construct_symmetry_not_caller_obligation). Ordered: the first
# member that exists wins; the canonical name is preferred when both exist. The two
# suffixes that make a file "mem-sidecar-looking" for the WARN scan below.
_MEM_SIDECAR_SUFFIXES = ("_mem.jsonl", "_mem_sidecar.jsonl")


def _base_stem(path: Path) -> str:
    """The trace's BASE stem with any mem-sidecar family suffix stripped.

    ``trace.jsonl`` → ``trace`` (Path.stem). But when the path IS already a
    family member (``trace_mem.jsonl`` / ``trace_mem_sidecar.jsonl`` — e.g. an
    entry that passed the ``_mem.jsonl`` file ITSELF as the trace path, the F0
    auto-merge=0 divergence), we strip that family suffix so the base stem is
    recovered. Otherwise ``Path.stem`` only drops the final ``.jsonl`` and the
    candidate doubles into ``trace_mem_mem.jsonl`` (a name that never exists →
    auto-merge silently resolved None while an explicit load found the file)."""
    name = path.name
    for suf in _MEM_SIDECAR_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return path.stem


def mem_sidecar_candidates(trace_path: str | Path) -> list[Path]:
    """The conventional mem-sidecar sibling NAMES for a main trace path (the family).

    Returns the ``<base>_mem.jsonl`` and ``<base>_mem_sidecar.jsonl`` siblings in
    the SAME directory, in preference order (canonical first), where ``<base>`` is
    the trace's stem with any family suffix already stripped (:func:`_base_stem`).
    Stripping makes the AUTO path agree with the EXPLICIT load even when the path
    handed in is itself a ``_mem.jsonl`` family member (the F0 auto-merge=0 bug:
    ``stem`` left ``..._mem`` so the candidate doubled to ``..._mem_mem.jsonl``).
    Pure / deterministic — it does NOT touch the filesystem (no ``exists`` here)."""
    p = Path(trace_path)
    base = _base_stem(p)
    cands = [p.with_name(f"{base}{suf}") for suf in _MEM_SIDECAR_SUFFIXES]
    # When the path IS itself a family member, that file is a valid sidecar to
    # fold (it carries the mem events); offer it too so auto == explicit. Keep
    # the stripped-base candidates first (a real base sibling is preferred).
    if looks_like_mem_sidecar(p) and p not in cands:
        cands.append(p)
    return cands


def mem_sidecar_sibling(trace_path: str | Path) -> Path:
    """The canonical ``<stem>_mem.jsonl`` sibling for a main trace path.

    Naming convention (the one the main trace already uses — see
    ``test_trace_merge.test_jsonl_reader_merged_sidecar`` / ``test_cvd_recovery``):
    the sidecar lives in the SAME directory and is named ``<stem>_mem.jsonl`` where
    ``<stem>`` is the trace filename with its final suffix stripped
    (``trace.jsonl`` → ``trace_mem.jsonl``). A trace with no suffix
    (``trace`` → ``trace_mem.jsonl``) follows the same rule. Pure / deterministic;
    it does NOT touch the filesystem (no ``exists`` here) so it is a usable naming
    function in isolation. See :func:`mem_sidecar_candidates` for the full family
    (canonical + ``_mem_sidecar.jsonl``) the resolver actually accepts."""
    p = Path(trace_path)
    return p.with_name(f"{p.stem}_mem.jsonl")


def base_trace_sibling(trace_path: str | Path) -> Path | None:
    """The BASE instruction trace for a path that is itself a mem-sidecar member.

    When ``trace_path`` is ``X_mem.jsonl`` / ``X_mem_sidecar.jsonl``, the
    instruction skeleton lives in the sibling ``X.jsonl`` (the family suffix
    stripped). Returns that sibling iff the path looks like a family member AND
    the base sibling differs from the path; otherwise ``None`` (the path is
    already a base trace — nothing to redirect). Pure / no filesystem touch."""
    p = Path(trace_path)
    if not looks_like_mem_sidecar(p):
        return None
    base = p.with_name(f"{_base_stem(p)}.jsonl")
    return base if base != p else None


def looks_like_mem_sidecar(path: str | Path) -> bool:
    """True iff a filename is mem-sidecar-LOOKING (ends with a family suffix).

    Used by the de-silence scan: a file in the trace's directory that LOOKS like a
    mem sidecar but is NOT the one resolved/merged must be WARN-ed, not silently
    ignored (task 5)."""
    name = Path(path).name
    return any(name.endswith(suf) for suf in _MEM_SIDECAR_SUFFIXES)


def unmerged_mem_sidecars(trace_path: str | Path, resolved: str | Path | None) -> list[Path]:
    """Mem-sidecar-LOOKING files in the trace's directory that were NOT merged.

    De-silence (task 5 / invariant 1): if the directory holds a ``*_mem.jsonl`` or
    ``*_mem_sidecar.jsonl`` that the resolver did not pick (e.g. a differently-stemmed
    sidecar, or a second family member), the caller WARNs so a silently-dropped mem
    dimension is visible at the boundary. Best-effort: a missing / unreadable
    directory yields ``[]`` (the scan must never raise)."""
    p = Path(trace_path)
    resolved_p = Path(resolved).resolve() if resolved is not None else None
    out: list[Path] = []
    try:
        for sib in sorted(p.parent.iterdir()):
            if not sib.is_file() or sib == p:
                continue
            if not looks_like_mem_sidecar(sib):
                continue
            if resolved_p is not None and sib.resolve() == resolved_p:
                continue
            out.append(sib)
    except OSError:
        return []
    return out


class JsonlTraceReader(TraceReader):
    """Standard format: one JSON object per line per contracts §2.1.

    Optional sidecars (item ①): ``mem_sidecar`` is a canonical ``_mem.jsonl``
    memory-event file and ``snapshot_sidecar`` a register-pointed hook dump. They
    are read into canonical events (via ``obs_readers``) and merged onto the main
    stream by :meth:`merged`. The default ``__iter__`` is unchanged — no sidecar,
    or simply iterating, yields the main trace exactly as before (invariant 7).

    Automatic sibling resolution (68a873e anti-pattern correction): when
    ``mem_sidecar`` is NOT explicitly given, :meth:`merged` looks for the
    conventional ``<stem>_mem.jsonl`` sibling (:func:`mem_sidecar_sibling`) and
    folds it in if it exists. This makes ``merged()`` symmetric BY CONSTRUCTION —
    the main trace and a cohort vector loaded the same way both pick up their
    sidecar without the caller passing anything. An explicit ``mem_sidecar`` (or
    ``mem_sidecar=False`` to opt out) always wins over the auto-resolution
    (invariant 7: explicit override behaviour is unchanged)."""

    def __init__(self, path: str | Path, *,
                 mem_sidecar: str | Path | None | bool = None,
                 snapshot_sidecar: str | Path | None = None):
        self.path = Path(path)
        # Tri-state: an explicit path/str → that exact file; ``False`` → opt OUT of
        # auto-resolution (force bare); ``None`` (default) → try the conventional
        # sibling in merged(). ``mem_sidecar_explicit`` records whether the caller
        # named one so an explicit override is never silently second-guessed.
        self.mem_sidecar_explicit = mem_sidecar is not None
        if mem_sidecar is None or mem_sidecar is False:
            self.mem_sidecar = None
        else:
            self.mem_sidecar = Path(mem_sidecar)
        self._mem_sidecar_optout = mem_sidecar is False
        self.snapshot_sidecar = Path(snapshot_sidecar) if snapshot_sidecar else None

    def resolve_mem_sidecar(self) -> Path | None:
        """The ``_mem.jsonl`` :meth:`merged` will fold in (or ``None``).

        An explicit ``mem_sidecar`` is returned verbatim (override always wins). An
        explicit ``mem_sidecar=False`` opts out → ``None``. Otherwise the
        conventional sibling is returned ONLY if it exists on disk (auto-resolution
        is a best-effort convenience, never a fabricated path)."""
        if self.mem_sidecar is not None:
            return self.mem_sidecar
        if self._mem_sidecar_optout:
            return None
        # Accept the whole mem-sidecar FAMILY (canonical _mem.jsonl OR the common
        # _mem_sidecar.jsonl alternative), not one dead name — the first member that
        # exists on disk wins, canonical preferred (task 5: recognise the semantic
        # family so a runner naming it _mem_sidecar.jsonl is no longer silently
        # dropped). Best-effort: never a fabricated path (returns None if none exist).
        for sib in mem_sidecar_candidates(self.path):
            if sib.exists():
                return sib
        return None

    def merged(self):
        """Return a :class:`engine.trace_merge.MergedTrace`: the main stream with
        the (optional) canonical sidecars folded in. With no sidecar configured AND
        no conventional sibling on disk, the merged items are byte-for-byte the same
        as ``list(self)`` (invariant 7).

        Base-trace redirect (F0 auto-merge=0): if the path handed in is ITSELF a
        ``_mem.jsonl`` family member (an entry that passed the mem-sidecar file as
        the trace), it carries no instruction skeleton (its rows have no
        ``bytes``) → folding it against an empty main stream merges 0. We detect
        that, read the instruction skeleton from the base sibling (``X.jsonl``),
        and fold the given mem-file as the sidecar — so the AUTO path now merges
        exactly what an EXPLICIT ``mem_sidecar=`` load on the base trace would.
        The redirect is LOUD (WARN), never silent."""
        from .obs_readers import read_hook_snapshots, read_mem_events
        from .trace_merge import merge_trace_sources
        mem_path = self.resolve_mem_sidecar()
        # When self.path is a mem-family member, source instructions from the base
        # sibling instead of the (skeleton-less) mem file.
        base = base_trace_sibling(self.path)
        if base is not None and base.exists():
            _log.warning(
                "trace path %s is a mem-sidecar family member (no instruction "
                "skeleton); sourcing instructions from base sibling %s and folding "
                "%s as the mem sidecar (auto-merge now matches an explicit "
                "mem_sidecar= load on the base trace). Prefer passing the BASE "
                "trace directly. (contracts/runner_interface.md mem-sidecar family)",
                self.path.name, base.name, self.path.name)
            main = list(JsonlTraceReader(base))
            # The given mem-file is the sidecar (resolve already strips the family
            # suffix; here we also guarantee it when self.path IS the family file).
            if mem_path is None:
                mem_path = self.path
        else:
            main = list(self)
        mem_events = read_mem_events(mem_path) if mem_path else ()
        snapshots = read_hook_snapshots(self.snapshot_sidecar) if self.snapshot_sidecar else ()
        return merge_trace_sources(main, mem_events=mem_events, snapshots=snapshots)

    def _detect_convention(self) -> ConventionDetector:
        # First pass: vote the trace's byte-order CONVENTION over a bounded sample
        # of decidable instructions (see engine.byte_order). This is the streaming-
        # safe way to recover ALIAS instructions, whose individual head-match is
        # inconclusive — they inherit the window convention instead of being left
        # MSB-first. File-backed reader re-opens per pass, so memory stays bounded.
        def _samples() -> Iterator[tuple[bytes, str | None]]:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    # The convention vote is a best-effort SAMPLING pass: a row
                    # that carries no decodable "bytes" (a mem-event / annotation
                    # record interleaved in the stream) is simply not a sample —
                    # skip it, never crash the whole reader on it (the systematic
                    # recover_window KeyError('bytes') single point).
                    hexb = obj.get("bytes")
                    if not hexb:
                        continue
                    raw = bytes.fromhex(hexb)
                    if len(raw) == AARCH64_INSN_BYTES:
                        yield raw, obj.get("mnemonic")
        return ConventionDetector.from_samples(_samples(), sample=CONVENTION_VOTE_SAMPLE)

    def __iter__(self) -> Iterator[Instruction]:
        detector = self._detect_convention()
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # Symmetric with the convention pass above: a row with no
                # decodable "bytes" is a non-instruction record (mem-event /
                # annotation) interleaved in the stream — skip it rather than
                # crash on obj["bytes"]/obj["mnemonic"] (the systematic
                # recover_window KeyError('bytes') single point).
                if not obj.get("bytes"):
                    continue
                yield Instruction(
                    idx=obj["idx"],
                    pc=int(obj["pc"], 16),
                    # Canonical little-endian (memory) order so every downstream
                    # decode feed + bytes_.hex() consumer sees ONE order (see
                    # engine.byte_order): the trace's "bytes" may be LE OR the
                    # word's MSB-first hex, and a MSB-first feed mis-decodes /
                    # fails in capstone/Triton (the VMP class-8 byte-order trap).
                    # The voted convention is applied UNIFORMLY (alias insns too).
                    bytes_=detector.apply(bytes.fromhex(obj["bytes"])),
                    mnemonic=obj["mnemonic"],
                    regs_read={k: int(v, 16) for k, v in obj.get("regs_read", {}).items()},
                    regs_write={k: int(v, 16) for k, v in obj.get("regs_write", {}).items()},
                    mem=tuple(
                        MemOp(rw=m["rw"], addr=int(m["addr"], 16), val=int(m["val"], 16), size=m["size"])
                        for m in obj.get("mem", [])
                    ),
                )


# unidbg textual trace example line:
# [09:39:47 005][libEncryptor.so 0x07d88] [ff8301d1] 0x40007d88: "sub sp, sp, #0x60" sp=0xbffff700 => sp=0xbffff6a0

_UNIDBG_LINE = re.compile(
    r"\[(?P<ts>\d{2}:\d{2}:\d{2}\s+\d{3})\]"
    r"\[(?P<module>[^\s]+)\s+0x(?P<off>[0-9a-f]+)\]\s+"
    r"\[(?P<bytes>[0-9a-f]{8})\]\s+"
    r"0x(?P<pc>[0-9a-f]+):\s+"
    r"\"(?P<mnem>[^\"]*)\""
    r"(?P<state>.*)$"
)


def _parse_state(text: str) -> tuple[dict[str, int], dict[str, int]]:
    """Parse the trailing 'reg=0xVAL ... => reg=0xVAL ...' segment.

    Pre-arrow tokens are reads (pre-execute state); post-arrow are writes.
    A register appearing on both sides is read AND written.
    """
    if "=>" in text:
        pre, post = text.split("=>", 1)
    else:
        pre, post = text, ""
    reads = _kv_tokens(pre)
    writes = _kv_tokens(post)
    return reads, writes


_KV = re.compile(r"([a-z][a-z0-9]*)=0x([0-9a-fA-F]+)")


def _kv_tokens(s: str) -> dict[str, int]:
    return {m.group(1): int(m.group(2), 16) for m in _KV.finditer(s)}


class UnidbgTextTraceReader(TraceReader):
    """Compat for the unidbg default text dump (testTarget/vmp/trace.txt format)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        # Honest-miss accounting: number of memory-class steps whose effective
        # address / addressing form could not be resolved from the trace line.
        # Observable so the ③ observability assessment can reflect mem coverage
        # without silent fabrication. Reset at the start of each __iter__ pass.
        self.unresolved_mem_steps = 0

    def _detect_convention(self) -> ConventionDetector:
        # First pass — vote the dump's byte-order convention (see JsonlTraceReader
        # / engine.byte_order). Re-opens the file; memory bounded by the sample cap.
        def _samples() -> Iterator[tuple[bytes, str | None]]:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    m = _UNIDBG_LINE.match(line)
                    if not m:
                        continue
                    raw = bytes.fromhex(m.group("bytes"))
                    if len(raw) == AARCH64_INSN_BYTES:
                        yield raw, m.group("mnem")
        return ConventionDetector.from_samples(_samples(), sample=CONVENTION_VOTE_SAMPLE)

    def __iter__(self) -> Iterator[Instruction]:
        detector = self._detect_convention()
        idx = 0
        self.unresolved_mem_steps = 0
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                m = _UNIDBG_LINE.match(line)
                if not m:
                    continue
                reads, writes = _parse_state(m.group("state"))
                # Recover mem ops from the addressing expression + in-line
                # register values (EA from pre-state reads, load value from
                # post-state writes, store value from pre-state reads, size from
                # the mnemonic width). Honest miss -> empty mem + count, never
                # fabricated. See engine.aarch64_mem.parse_mem_ops.
                mem_ops, ea_unresolved = parse_mem_ops(m.group("mnem"), reads, writes)
                if ea_unresolved:
                    self.unresolved_mem_steps += 1
                yield Instruction(
                    idx=idx,
                    pc=int(m.group("pc"), 16),
                    # Canonical little-endian order (see JsonlTraceReader / the
                    # engine.byte_order note) — the voted window convention applied
                    # uniformly so an already-LE dump is untouched and an alias
                    # instruction is recovered along with its neighbours.
                    bytes_=detector.apply(bytes.fromhex(m.group("bytes"))),
                    mnemonic=m.group("mnem"),
                    regs_read=reads,
                    regs_write=writes,
                    mem=mem_ops,
                )
                idx += 1


@dataclass(frozen=True)
class RegRelWatch:
    """A PC-gated, register-relative single-point mem watch (contracts §3.2.1).

    The reg-relative twin of ``ObservePoint.mem``'s concrete ``(addr, size)``:
    a concrete watch names a fixed address known up front, but the address we
    actually want is often ``[base_reg + offset]`` computed from the register's
    LIVE value at a specific instruction (``pc``) — which can't be precomputed
    (the value moves across runs). This carries the directive so the runner
    arms at ``pc`` and, when execution reaches it, resolves ``base_reg``'s
    current value + ``offset`` and captures ``width`` bytes once, in the
    ``kind`` direction (``"read"`` = the bytes at that address at that instant;
    ``"write"`` = the bytes this instruction writes there). It does NOT expand
    into a range scan → no wide-region noise, physically can't fill a wide
    record cap (the very harms §3.2.1 names). ``kind`` matches
    ``engine.watch_first_write`` point-watch directions.
    """
    base_reg: str
    offset: int
    width: int
    pc: int
    kind: str = "read"        # "read" | "write" — capture direction
    # Register-offset addressing (B6): ``[base, index{, lsl/uxtw #scale}]`` →
    # live EA = base_val + (index_val << scale) + offset. Additive: ``index=None``
    # (default) is the plain ``[base + offset]`` form and serializes byte-for-byte
    # as before (invariant 7). When set, the runner adds ``index_val << scale`` to
    # the resolved address. ``index`` is the (run-local-VALUE, structurally-stable
    # -NAME) index register; ``scale`` the left-shift / extend amount.
    index: str | None = None
    scale: int = 0


def _regrel_to_wire(w: "RegRelWatch") -> dict[str, Any]:
    """ONE wire shape for a reg-relative watch, shared by the JSON-RPC rerun path
    (``SubprocessRunnerAdapter._serialize_observe_point``) and the recapture spec
    serializer (``recapture.RecaptureSpec._observe_point_to_dict``) — symmetry by
    construction, never two drifting copies. ``index``/``scale`` are emitted ONLY
    for a register-offset form (``index`` set), so a plain ``[base+offset]`` watch
    serializes byte-for-byte as before (invariant 7)."""
    d: dict[str, Any] = {
        "base_reg": w.base_reg,
        "offset": w.offset,
        "width": w.width,
        "pc": f"0x{w.pc:x}",
        "kind": w.kind,
    }
    if w.index is not None:
        d["index"] = w.index
        d["scale"] = w.scale
    return d


def _regrel_from_wire(d: dict[str, Any]) -> "RegRelWatch":
    """Inverse of :func:`_regrel_to_wire` — reconstruct a ``RegRelWatch`` from its
    ONE wire shape. The lossless twin needed by any from_dict round-trip (B3
    directive replay / handoff / auto-resume). ``index``/``scale`` are optional
    (present only for a register-offset form); ``kind`` defaults to ``"read"`` to
    match the dataclass default so a plain ``[base+offset]`` watch round-trips
    byte-for-byte (invariant 7). ``pc`` accepts the hex string ``_regrel_to_wire``
    emits OR an int (defensive)."""
    pc = d["pc"]
    return RegRelWatch(
        base_reg=d["base_reg"],
        offset=int(d["offset"]),
        width=int(d["width"]),
        pc=int(pc, 16) if isinstance(pc, str) else int(pc),
        kind=d.get("kind", "read"),
        index=d.get("index"),
        scale=int(d.get("scale", 0)),
    )


@dataclass(frozen=True)
class ObservePoint:
    """One observation point for rerun() — see contracts §3.2."""
    pc: int
    when: str                 # "before" | "after"
    capture: tuple[str, ...]  # subset of ("regs", "mem")
    regs: tuple[str, ...] = ()
    mem: tuple[tuple[int, int], ...] = ()   # list of (addr, size)
    # Reg-relative point watches (contracts §3.2.1) — the runtime-resolved twin
    # of concrete ``mem``. Additive: default empty so every existing concrete
    # ObservePoint serializes byte-for-byte as before (invariant 7). When set,
    # the runner resolves each ``[base_reg+offset]@pc`` from the live register
    # value and captures a single point (no wide-region noise / cap hazard).
    mem_regrel: tuple[RegRelWatch, ...] = ()


@dataclass(frozen=True)
class ObservedState:
    pc: int
    when: str
    regs: dict[str, int]
    mem: dict[int, bytes]


@dataclass(frozen=True)
class RerunResult:
    """A runner rerun outcome: the produced ``output`` plus any captured
    ``observations``.

    ``truncated`` — set ``True`` iff the runner hit a record cap (e.g.
    ``*_REGREL_CONCRETE_WRITE_MAX``) and therefore could NOT record every
    matching read/write: the observations are an INCOMPLETE ledger and MUST
    NOT be consumed as complete/clean provenance. This mirrors
    :class:`engine.phase_instrument.PhaseInstrumentResult.truncated` — the
    construct guarantees symmetry: a runner that silently dropped records but
    left ``truncated=False`` violates the contract
    (contracts/runner_interface.md §rerun). ``truncated`` is a boolean SEMANTIC,
    never the cap value itself (the cap lives runner-side).

    ``truncated_detail`` — optional free-form report (mode + counts + which
    cap) carried alongside, e.g. ``{"cap": "X25_REGREL_CONCRETE_WRITE_MAX",
    "limit": 8192, "kind": "write"}``. Advisory; for logs/exports only.
    """
    output: bytes
    observations: tuple[ObservedState, ...] = ()
    truncated: bool = False
    truncated_detail: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "output_hex": self.output.hex(),
            "n_observations": len(self.observations),
            "truncated": self.truncated,
        }
        if self.truncated_detail is not None:
            out["truncated_detail"] = dict(self.truncated_detail)
        return out


def mem_snapshots_from_rerun(result: "RerunResult") -> list["MemSnapshot"]:
    """Convert a :class:`RerunResult`'s observed mem regions → canonical
    :class:`MemSnapshot` objects the engine's oracle_sink consumes.

    The capture/mem → MemSnapshot end-to-end leg (dev-closure-evidence-layering
    -trap-state-spec, task 6): a runner that captured ``mem`` at an observe point
    returns it on ``ObservedState.mem`` (``{addr: bytes}``); ``oracle_sink`` /
    ``validate_sink`` / ``trace_provenance`` consume canonical ``MemSnapshot``
    objects, not raw observation dicts. This is the missing wire conversion — the
    contract guarantee that capture/mem reaches the sink as a snapshot, so the
    ``located_via`` becomes ``snapshot`` instead of a write/read fallback.

    Pure / deterministic; an observation with no captured mem contributes nothing
    (never a fabricated snapshot — invariant 8). Returns ``[]`` for a result with
    no mem observations (the caller then WARNs if it claimed mem-capture).

    Truncation propagation (construct-symmetry, NOT caller obligation): when the
    runner hit a record cap (``result.truncated``), this ledger is INCOMPLETE.
    We WARN to the top-level logger (visible, never silent) and stamp every
    derived snapshot ``truncated=True`` so downstream provenance / validate_sink
    knows it is not complete/clean provenance. See
    contracts/runner_interface.md §rerun."""
    from .types import MemSnapshot
    if result.truncated:
        _log.warning(
            "rerun observations TRUNCATED: the runner hit a record cap%s — the "
            "captured ledger is INCOMPLETE and MUST NOT be consumed as complete/"
            "clean provenance. Derived snapshots are stamped truncated=True; "
            "downstream located_via on this data is not authoritative. "
            "Widen/raise the runner cap or narrow the capture region "
            "(contracts/runner_interface.md §rerun).",
            _truncated_detail_suffix(result.truncated_detail))
    snaps: list[MemSnapshot] = []
    for obs in result.observations:
        for addr, data in (obs.mem or {}).items():
            if not data:
                continue
            snaps.append(MemSnapshot(
                addr=int(addr), data=bytes(data),
                label=f"observe@0x{obs.pc:x}:{obs.when}",
                source="snapshot",
                truncated=bool(result.truncated)))
    return snaps


def _truncated_detail_suffix(detail: dict[str, Any] | None) -> str:
    """Render an optional truncated_detail dict into a short ' (k=v, ...)' suffix
    for WARN messages; empty string when no detail. Pure / log-only."""
    if not detail:
        return ""
    parts = ", ".join(f"{k}={v}" for k, v in detail.items())
    return f" ({parts})"


class RunnerAdapter(abc.ABC):
    """Full 3-method interface (PLAN §17 / contracts §3).

    Implementations either fully implement Live mode or raise
    NotImplementedError from get_trace / rerun to signal File mode (in which
    case only metadata() + a separately-supplied static trace file are used).
    """

    # Engine-side static declaration of which capabilities the adapter
    # implements. Used by :func:`engine.block_cause.oracle_from_adapter`
    # to answer "does the runner have first-write capture", etc.
    # Subclasses override with the names they fulfil. ``metadata()``
    # may add to this set at runtime via a ``capabilities`` field on
    # :class:`engine.types.TargetMeta` — runtime wins over static so a
    # runner can opt into capabilities at project level without code
    # changes here. Vocabulary is shared with
    # :data:`engine.block_cause.ANCHOR_TO_CAPABILITY`.
    CAPABILITIES: frozenset[str] = frozenset()

    @abc.abstractmethod
    def metadata(self) -> "TargetMeta":  # forward ref
        ...

    def get_trace(self, input_bytes: bytes, start: int, end: int) -> str:
        """Return path to a JSONL trace file (or yield generator) for [start, end].

        File-mode adapters raise NotImplementedError here.
        """
        raise NotImplementedError

    def rerun(self, input_bytes: bytes, observe_points: list[ObservePoint] | None = None) -> RerunResult:
        """Run target on input; capture observation points; return output.

        File-mode adapters raise NotImplementedError here. Live-mode adapters
        with observe_points=None or empty return only `output`, no observations.
        """
        raise NotImplementedError

    def code_hook_range(
        self,
        input_bytes: bytes,
        hooks: list["CodeHookRange"],
    ) -> list["RegisterTrace"]:
        """Run target on ``input_bytes`` and emit a ``RegisterTrace``
        per ``CodeHookRange`` (PC-band step-callback observation).

        capability_request.md §P0-1 — needed for VMP-internal
        compress_leg / x22 / x27 tracking that PLT/BL observe_points
        can't reach.

        Default: ``NotImplementedError``. Callers can fall back to
        ``get_trace`` + ``capability.register_trace_from_instructions``
        when the runner only supports trace-level observation.
        """
        raise NotImplementedError


# Back-compat name kept for code already imported; new code use RunnerAdapter.
RerunAdapter = RunnerAdapter


class NullRunnerAdapter(RunnerAdapter):
    """Stub used by File-mode samples that only have a pre-baked static trace.

    metadata() is provided externally (e.g. from a YAML alongside the trace).
    get_trace / rerun raise — conformance.C1/C2/C3 will SKIP, only C4 runs.
    """

    def __init__(self, target_meta: "TargetMeta"):
        self._meta = target_meta

    def metadata(self) -> "TargetMeta":
        return self._meta


NullRerunAdapter = NullRunnerAdapter


class SubprocessRunnerAdapter(RunnerAdapter):
    """Live-mode adapter that spawns an external runner process and talks NDJSON
    over its stdin/stdout. Spec is symmetric with the Java {@code Main.serve()}
    loop; any subprocess implementing the same wire protocol works.

    Wire protocol (NDJSON, one object per line):
        request  : {"id": int, "method": "metadata|rerun|get_trace|shutdown",
                    "params": {...}?}
        response : {"id": int, "result": {...}} | {"id": int, "error": {...}}

    Subclass uses the same `RunnerAdapter` interface as `NullRunnerAdapter` —
    so the engine can hand-off between modes without code changes.
    """

    def __init__(self, cmd: list[str], cwd: str | Path | None = None,
                 startup_banner: str = "runner ready", startup_timeout_s: float = 30.0):
        import subprocess
        import threading

        self._cmd = cmd
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(cwd) if cwd else None,
            bufsize=1, text=True, encoding="utf-8",
        )
        self._next_id = 0
        self._stderr_lines: list[str] = []

        # Drain stderr in a daemon thread so the runner can't block on a full pipe.
        def _drain_stderr():
            for ln in self._proc.stderr:  # type: ignore[union-attr]
                self._stderr_lines.append(ln.rstrip("\n"))
        threading.Thread(target=_drain_stderr, daemon=True).start()

        # Wait for the startup banner on stderr (without consuming stdout).
        import time
        t0 = time.monotonic()
        while time.monotonic() - t0 < startup_timeout_s:
            if any(startup_banner in s for s in self._stderr_lines):
                return
            if self._proc.poll() is not None:
                raise RuntimeError("runner exited during startup; stderr:\n" + "\n".join(self._stderr_lines))
            time.sleep(0.05)
        raise TimeoutError(
            f"runner did not emit '{startup_banner}' within {startup_timeout_s}s; "
            f"stderr so far:\n" + "\n".join(self._stderr_lines)
        )

    def _call(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        req = {"id": self._next_id, "method": method}
        if params is not None:
            req["params"] = params
        self._proc.stdin.write(json.dumps(req) + "\n")  # type: ignore[union-attr]
        self._proc.stdin.flush()  # type: ignore[union-attr]
        line = self._proc.stdout.readline()  # type: ignore[union-attr]
        if not line:
            raise RuntimeError(
                "runner closed stdout unexpectedly; stderr:\n" + "\n".join(self._stderr_lines)
            )
        resp = json.loads(line)
        if "error" in resp:
            err = resp["error"]
            code = err.get("code")
            message = str(err.get("message", ""))
            # BR-2 §13+: translate "capability missing" wire errors to
            # NotImplementedError so the conformance gate's mode-detect /
            # individual checks correctly SKIP File-mode runners instead of
            # crashing with an unhandled RuntimeError. This is the exact bug
            # where a Java runner threw `UnsupportedOperationException` for
            # rerun and the engine's narrow `except NotImplementedError`
            # missed it.
            if _is_capability_missing_error(code, message):
                raise NotImplementedError(
                    f"runner does not implement {method!r} "
                    f"(wire error: {message}; code={code})"
                )
            raise RuntimeError(f"runner error: {message} (code={code})")
        return resp["result"]

    def metadata(self) -> "TargetMeta":
        from .types import TargetMeta
        r = self._call("metadata")
        return TargetMeta(
            target_name=r["target_name"],
            arch=r["arch"],
            algo_entry_pc=int(r["algo_entry_pc"], 16),
            algo_exit_pc=int(r["algo_exit_pc"], 16),
            input_length=r.get("input_length"),
            output_length=r["output_length"],
            algo_symbol=r.get("algo_symbol"),
            emulator_name=r.get("emulator_name"),
            emulator_version=r.get("emulator_version"),
            capabilities=tuple(r.get("capabilities", ()) or ()),
        )

    @staticmethod
    def _serialize_observe_point(op: ObservePoint) -> dict:
        """Wire form of one ObservePoint (contracts §3.2 / §3.2.1).

        Java side parses ``when`` with ``When.valueOf()`` and each ``capture``
        entry with ``Capture.valueOf()`` — both strict UPPER case (Java enums
        ``When{BEFORE,AFTER}`` / ``Capture{REGS,MEM}``). The engine uppercases
        both at the wire boundary; callers use the lowercase convention. (Bug2:
        ``capture`` was sent verbatim lowercase ``["mem"]`` while ``when`` was
        already upper — the asymmetry tripped ``Capture.valueOf("mem")`` →
        ``isolated once exit 1``.) ``regs`` / ``mem_regrel.base_reg`` are
        register *names* (free-form ``List<String>``), NOT enums — left as-is.
        ``mem_regrel.kind`` is contract-pinned lowercase (§3.2.1, no Java enum
        yet) — also left as-is.
        Carry the FULL point: ``capture`` selects which kinds the runner
        records, ``mem`` lists concrete ``(addr,size)`` ranges (addr hex), and
        ``mem_regrel`` lists PC-gated reg-relative single-point watches.
        Dropping ``capture``/``mem`` (historic Bug1) left the runner with
        nothing to capture for mem → snapshots came back permanently empty →
        same-execution oracle snapshot path dead end-to-end. ``mem_regrel`` is
        its reg-relative twin: emitted ONLY when non-empty so a plain concrete
        point serializes exactly as before (invariant 7)."""
        pt: dict = {
            "pc": f"0x{op.pc:x}",
            "when": op.when.upper(),
            "capture": [c.upper() for c in op.capture],
            "regs": list(op.regs),
            "mem": [{"addr": f"0x{a:x}", "size": s} for (a, s) in op.mem],
        }
        if op.mem_regrel:
            pt["mem_regrel"] = [_regrel_to_wire(w) for w in op.mem_regrel]
        return pt

    def rerun(self, input_bytes: bytes,
              observe_points: list[ObservePoint] | None = None) -> RerunResult:
        params: dict = {"input_hex": input_bytes.hex()}
        if observe_points:
            params["observe_points"] = [
                self._serialize_observe_point(op)
                for op in observe_points
            ]
        r = self._call("rerun", params)
        output = bytes.fromhex(r["output_hex"])
        observations = tuple(
            ObservedState(
                pc=int(o["pc"], 16),
                when=o["when"].lower(),  # normalize back to our convention
                regs={k: int(v, 16) for k, v in o.get("regs", {}).items()},
                mem={int(a, 16): bytes.fromhex(v) for a, v in o.get("mem", {}).items()},
            )
            for o in r.get("observations", [])
        )
        # Construct-symmetry: the runner MUST set ``truncated`` when it hit a
        # record cap (e.g. *_REGREL_CONCRETE_WRITE_MAX); we surface it verbatim
        # so the consumer side can WARN + mark derived provenance incomplete.
        truncated = bool(r.get("truncated", False))
        truncated_detail = r.get("truncated_detail")
        if truncated_detail is not None and not isinstance(truncated_detail, dict):
            truncated_detail = {"detail": truncated_detail}
        return RerunResult(
            output=output,
            observations=observations,
            truncated=truncated,
            truncated_detail=truncated_detail,
        )

    def get_trace(self, input_bytes: bytes, start: int, end: int) -> str:
        r = self._call("get_trace", {
            "input_hex": input_bytes.hex(),
            "start": f"0x{start:x}",
            "end": f"0x{end:x}",
        })
        return r["trace_path"]

    def shutdown(self) -> None:
        try:
            self._call("shutdown")
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
