"""CVD ledger — every primitive / CVD verdict lands as a stamped, queryable entry.

Origin: `cvd-lane-not-wired-to-ledger` (proven on the VMP cipher case, 163511).
The CVD self-drive lane (`cvd.py` / `cvd_mount.py` / `recapture.py`) made ZERO
store calls — `trace_provenance` / `who_wrote` / CVD verdicts are pure functions
that never land, and the `store` ledger was wired only into the full-pipeline
`core.py` path, not the lane the agent actually walks. So the agent hand-wrote
scattered json the next round could not read back, lost the global stage map
mid-case, and re-derived already-closed sub-goals — ~20 wasted rounds, half of
them rooted in "last step's result was computed but could not be queried back".

The fix is NOT "force the agent to write SQLite". It already keeps a ledger
(its markdown/json); the formats just don't connect. So utov becomes the *sole
authoritative writer*: every lane call lands one **stamped ledger entry**, and a
markdown **projection** of that ledger replaces the hand-written notes — more
complete than the hand notes, and machine-reusable next round.

Three rules (the spec's 核心三句):

  1. utov = the only authoritative writer. Every primitive/CVD call → one
     stamped LedgerEntry (:func:`record_call` / :func:`record_verdict`). The
     agent stops hand-writing json; to get a result back it queries the ledger.
  2. Stamped entries, NOT flat file-per-step. Result payloads are stored
     content-addressed (store.py's ``hyp_payloads`` — same value stored once);
     the ledger is the relational index over them.
  3. "Latest wins" is bucketed by EXECUTION IDENTITY, never by pure time. A
     later run is not a "newer version" of an earlier one — it is *another
     execution's* result (the determinism / nonce-artifact rule, lifted into the
     ledger layer). Only within one ``exec_identity`` does newest-ts win; a
     different run is a different key and can never overwrite.

Delivery is the mechanism + a read-side pull + a projection:
  - write side: :func:`record_call` (generic) and :func:`record_verdict` (the
    thin CVD-lane bridge — map a verdict string to a kind and land it).
  - query: :func:`get_latest` (same key → newest ts) and :func:`history`.
  - read-side pull: :func:`closed_subjects` / :func:`should_skip` so the next
    round skips already-closed sub-goals instead of re-deriving them (this is
    what makes "记账" pay for itself — without the pull it is just another
    unread side-product).
  - projection: :func:`project_stage_map` renders the global stage map from the
    landed producer / static-terminal / open-hypothesis entries.

Wiring point (CVD lane driver): at the place the lane finalises a verdict
(``ProvenanceVerifier`` / ``SinkValidatorVerifier`` result, a recapture close),
call :func:`record_verdict` with the run's :class:`ExecIdentity`. That single
call is the whole write-side bridge; the pure functions stay untouched.

This is a self-contained ``cvd_ledger.sqlite`` (its own WorkDir file): additive,
it never touches the hypotheses/findings schemas, so the existing suite is not
at risk. It reuses store.py's content-addressed payload helpers.
"""

from __future__ import annotations

import enum
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .export_stamp import build_export_header
from .store import WorkDir, _now_iso, content_hash, read_payload, upsert_payload


# ---------------------------------------------------------------------------
# Execution identity — the determinism bucket (rule 3).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExecIdentity:
    """Which execution a ledger entry belongs to.

    Two entries with the same ``(call_fn, inputs)`` but different
    ``exec_identity`` are NOT versions of one result — they are results from two
    different executions and must never overwrite each other. ``nonce`` is
    optional and only relevant when a run mixes a per-run nonce into the value
    under recovery (the cipher-body case): include it so an e6_capture round and
    the live round bucket apart even if target/input/run_id collide."""

    target:     str
    input_hash: str
    run_id:     str
    nonce:      str | None = None

    def canonical(self) -> dict[str, str]:
        d = {"target": self.target, "input_hash": self.input_hash,
             "run_id": self.run_id}
        if self.nonce is not None:
            d["nonce"] = self.nonce
        return d

    @property
    def ref(self) -> str:
        return content_hash(self.canonical())


# ---------------------------------------------------------------------------
# Entry kinds — how a CVD verdict maps into the ledger.
# ---------------------------------------------------------------------------


class LedgerKind(str, enum.Enum):
    # --- DURABLE findings (these LAND; they form the map / drive cross-round reuse) ---
    PRODUCER        = "producer"          # SUCCESS: the located writer (e.g. 0x223718)
    STATIC_TERMINAL = "static_terminal"   # rodata, single-ref → TERMINAL_STATIC
    OPEN_HYPOTHESIS = "open_hypothesis"   # NEEDS_OBSERVATION → open + watch anchors
    SINK            = "sink"              # validate_sink SINK_CONFIRMED + addr
    EMIT            = "emit"              # emit_python final F + parity (the deliverable)
    TRANSFORM       = "transform"         # recovered compute-point / handler transform
    RUN_SUMMARY     = "run_summary"       # drive's one-per-round roll-up (recording-policy §3)
    # --- TRANSIENT (default; dropped by the recording policy unless force=True) ---
    VERDICT         = "verdict"           # any other stamped result (transient by default)


# A closed sub-goal does not need re-deriving next round (the read-side pull).
# NEEDS_OBSERVATION is intentionally NOT closed — it is an open lead.
_CLOSED_KINDS = frozenset({LedgerKind.PRODUCER.value, LedgerKind.STATIC_TERMINAL.value})

# Recording policy (cvd-ledger-recording-policy): the ledger holds DURABLE
# findings only — those that form the stage map or drive cross-round reuse
# (should_skip). Transient orchestration (locate_boundary / seed_entry_state /
# pick_mode / classify_hybrid_step specs, every backing-fill audit iteration,
# per-plan-step gate trail) is NOT recorded — it would explode the db and bury
# the durable conclusions. ``record_call`` drops a non-durable kind unless the
# caller passes ``force=True``. The "db 爆炸" root cause was a debug trail in the
# authoritative db; keep the trail out, the db keeps one row per durable finding.
_DURABLE_KINDS = frozenset({
    LedgerKind.PRODUCER.value,
    LedgerKind.STATIC_TERMINAL.value,
    LedgerKind.OPEN_HYPOTHESIS.value,
    LedgerKind.SINK.value,
    LedgerKind.EMIT.value,
    LedgerKind.TRANSFORM.value,
    LedgerKind.RUN_SUMMARY.value,
})


# Big lists (blind_pcs can be ~15k entries) must NEVER inline into the
# authoritative db — store only a count + content hash + a small sample; the
# full list is re-derivable on demand. Applied to every recorded payload.
_MAX_INLINE_LIST = 16


def _trim_payload(value: Any, _depth: int = 0) -> Any:
    """Replace oversized lists in a payload with ``{count, sha1, sample}``.

    Recurses into dicts/lists. A list longer than ``_MAX_INLINE_LIST`` becomes a
    digest so the ledger stays small (recording-policy rule 1: no big lists in
    the db — count + hash, compute details on demand)."""
    if isinstance(value, dict):
        return {k: _trim_payload(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_INLINE_LIST:
            return {
                "_trimmed_list": True,
                "count": len(value),
                "sha1": content_hash(json.dumps(list(value), default=str,
                                                sort_keys=True)),
                "sample": [_trim_payload(v, _depth + 1)
                           for v in list(value)[:8]],
            }
        return [_trim_payload(v, _depth + 1) for v in value]
    return value


# CVD/provenance verdict strings → ledger kind. The lane passes its verdict
# string through here so the mapping lives in one auditable place.
_VERDICT_KIND = {
    "SUCCESS":           LedgerKind.PRODUCER,
    "TERMINAL_STATIC":   LedgerKind.STATIC_TERMINAL,
    "NEEDS_OBSERVATION": LedgerKind.OPEN_HYPOTHESIS,
    "SINK_CONFIRMED":    LedgerKind.SINK,
}


def kind_for_verdict(verdict: str) -> LedgerKind:
    """Map a CVD/provenance verdict string to a :class:`LedgerKind`."""
    return _VERDICT_KIND.get(verdict, LedgerKind.VERDICT)


# ---------------------------------------------------------------------------
# Key — hash(call_fn + inputs + exec_identity). Same key → latest wins;
# different exec_identity → different key → never overwrites.
# ---------------------------------------------------------------------------


def entry_key(call_fn: str, inputs: Mapping[str, Any], exec_identity: ExecIdentity) -> str:
    inputs_ref = content_hash(dict(inputs))
    return hashlib.sha1(
        f"{call_fn}\0{inputs_ref}\0{exec_identity.ref}".encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# The read-back shape.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    entry_key:     str
    call_fn:       str
    inputs:        dict[str, Any]
    exec_identity: dict[str, str]
    ts:            str
    kind:          str
    verdict:       str | None
    subject:       str
    result:        dict[str, Any]

    @property
    def is_closed(self) -> bool:
        return self.kind in _CLOSED_KINDS or self.verdict == "SUCCESS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_key":     self.entry_key,
            "call_fn":       self.call_fn,
            "inputs":        self.inputs,
            "exec_identity": self.exec_identity,
            "ts":            self.ts,
            "kind":          self.kind,
            "verdict":       self.verdict,
            "subject":       self.subject,
            "result":        self.result,
            "is_closed":     self.is_closed,
            "kind_label":    "setup_cvd_ledger_entry",
        }


# ---------------------------------------------------------------------------
# Schema + open. Self-contained DB — reuses store's content-addressed payload
# table (so store.upsert_payload / read_payload work on this connection) but
# adds nothing to the hypotheses / findings schemas.
# ---------------------------------------------------------------------------


_LEDGER_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS hyp_payloads (
    content_hash TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    bytes_len INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cvd_ledger (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_key  TEXT NOT NULL,           -- hash(call_fn + inputs + exec_identity)
    call_fn    TEXT NOT NULL,
    inputs_ref TEXT NOT NULL,           -- content_hash of the inputs json
    exec_ref   TEXT NOT NULL,           -- content_hash of the exec_identity json
    ts         TEXT NOT NULL,           -- caller-supplied runtime stamp
    result_ref TEXT NOT NULL,           -- content_hash of the result payload
    verdict    TEXT,
    kind       TEXT NOT NULL,
    subject    TEXT NOT NULL DEFAULT '',
    FOREIGN KEY(inputs_ref) REFERENCES hyp_payloads(content_hash),
    FOREIGN KEY(exec_ref)   REFERENCES hyp_payloads(content_hash),
    FOREIGN KEY(result_ref) REFERENCES hyp_payloads(content_hash)
);
CREATE INDEX IF NOT EXISTS ix_cvdledger_key  ON cvd_ledger(entry_key);
CREATE INDEX IF NOT EXISTS ix_cvdledger_exec ON cvd_ledger(exec_ref);
CREATE INDEX IF NOT EXISTS ix_cvdledger_kind ON cvd_ledger(kind);
"""


def open_ledger(work: WorkDir | str | Path) -> sqlite3.Connection:
    """Open (create if needed) the CVD ledger DB.

    Accepts a :class:`store.WorkDir` (ledger lives at
    ``<run>/cvd_ledger.sqlite``), an explicit path, or ``":memory:"``."""
    if isinstance(work, WorkDir):
        path: str = str(work.root / "cvd_ledger.sqlite")
    else:
        path = str(work)
    conn = sqlite3.connect(path)
    conn.executescript(_LEDGER_SCHEMA)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Write side — rule 1 (utov is the only authoritative writer).
# ---------------------------------------------------------------------------


def record_call(
    conn: sqlite3.Connection,
    *,
    call_fn: str,
    inputs: Mapping[str, Any],
    exec_identity: ExecIdentity,
    result: Mapping[str, Any] | str,
    kind: LedgerKind | str = LedgerKind.VERDICT,
    verdict: str | None = None,
    subject: str = "",
    ts: str | None = None,
    auto_view: bool = True,
    force: bool = False,
) -> str:
    """Land one stamped DURABLE finding. Returns its ``entry_key``.

    Recording policy: only durable kinds (:data:`_DURABLE_KINDS`) land. A
    transient ``kind`` (the default ``VERDICT`` — an intermediate spec, a
    backing-fill audit iteration, a per-step orchestration trail) is DROPPED:
    the call returns its key but writes nothing, so the authoritative db keeps
    one row per durable conclusion instead of exploding with debug trail. Pass
    ``force=True`` to record a transient kind anyway (rare). Big lists in
    ``result`` are digested to ``{count, sha1, sample}`` — never inlined.

    ``result`` is stored content-addressed (deduped by value). ``ts`` is
    caller-supplied (a workflow script cannot call ``Date.now``); when omitted
    the engine stamps it. The same ``(call_fn, inputs, exec_identity)`` may be
    recorded many times — :func:`get_latest` returns the newest-ts row, while
    a different ``exec_identity`` is a different key and never overwrites.

    ``auto_view`` (default on) keeps the readable side from being forgettable:
    after the write, the stamped ``cvd_ledger_view.md`` projection is refreshed
    in the ledger's OWN directory (for a file-backed DB). So any path that writes
    the ledger — including an agent script that only calls ``open_ledger`` +
    ``record_call`` — automatically materialises the human/agent-readable view
    next to the sqlite. Without this the sqlite is a write-only "dead ledger"
    and the agent falls back to hand-writing json (the very thing this replaces).
    In-memory DBs have no directory, so the refresh is skipped there."""
    kind_v = kind.value if isinstance(kind, LedgerKind) else str(kind)
    ek = entry_key(call_fn, inputs, exec_identity)
    # Recording policy: drop transient kinds (keep the db to durable findings).
    if kind_v not in _DURABLE_KINDS and not force:
        return ek
    stamp = ts or _now_iso()
    inputs_ref = upsert_payload(conn, _trim_payload(dict(inputs)))
    exec_ref = upsert_payload(conn, exec_identity.canonical())
    result_payload: dict[str, Any] | str = (
        result if isinstance(result, str) else
        _trim_payload(result if isinstance(result, dict) else {"value": result})
    )
    result_ref = upsert_payload(conn, result_payload)
    conn.execute(
        "INSERT INTO cvd_ledger(entry_key, call_fn, inputs_ref, exec_ref, ts,"
        " result_ref, verdict, kind, subject) VALUES (?,?,?,?,?,?,?,?,?)",
        (ek, call_fn, inputs_ref, exec_ref, stamp, result_ref,
         verdict, kind_v, subject),
    )
    conn.commit()
    if auto_view:
        _refresh_view(conn, exec_identity, ts=stamp)
    return ek


def record_verdict(
    conn: sqlite3.Connection,
    *,
    call_fn: str,
    inputs: Mapping[str, Any],
    exec_identity: ExecIdentity,
    verdict: str,
    subject: str = "",
    anchor: tuple[int, int] | None = None,
    watch: list[Any] | None = None,
    payload: Mapping[str, Any] | None = None,
    kind: LedgerKind | str | None = None,
    ts: str | None = None,
    auto_view: bool = True,
    force: bool = False,
) -> str:
    """The thin CVD-lane bridge: stamp a verdict, mapping it to a kind.

    ``verdict`` is the lane's verdict string ("SUCCESS" / "TERMINAL_STATIC" /
    "NEEDS_OBSERVATION" / "SINK_CONFIRMED" / …) → :func:`kind_for_verdict`.
    Pass ``kind`` to override when the semantic is not carried by the verdict
    string (e.g. ``LedgerKind.EMIT`` for an emit deliverable whose verdict is
    PASS/FAIL, ``LedgerKind.TRANSFORM`` for a recovered transform). ``anchor`` is
    the (trace_idx, pc) the verdict landed at; ``watch`` is the watch list for a
    NEEDS_OBSERVATION lead (recorded even when empty, so a ``watch_n=0`` lead is
    not silently lost)."""
    kind = kind if kind is not None else kind_for_verdict(verdict)
    result: dict[str, Any] = {"verdict": verdict}
    if subject:
        result["subject"] = subject
    if anchor is not None:
        result["anchor"] = {"idx": anchor[0], "pc": anchor[1],
                            "pc_hex": f"0x{anchor[1]:x}"}
    if watch is not None:
        result["watch"] = list(watch)
        result["watch_n"] = len(watch)
    if payload:
        result.update(dict(payload))
    return record_call(
        conn, call_fn=call_fn, inputs=inputs, exec_identity=exec_identity,
        result=result, kind=kind, verdict=verdict, subject=subject, ts=ts,
        auto_view=auto_view, force=force,
    )


# ---------------------------------------------------------------------------
# Query — rule 3 (same key → newest ts; cross-run never overwrites).
# ---------------------------------------------------------------------------


_SELECT = (
    "SELECT l.entry_key, l.call_fn, ip.payload, ep.payload, l.ts, l.kind,"
    " l.verdict, l.subject, rp.payload"
    " FROM cvd_ledger l"
    " JOIN hyp_payloads ip ON ip.content_hash = l.inputs_ref"
    " JOIN hyp_payloads ep ON ep.content_hash = l.exec_ref"
    " JOIN hyp_payloads rp ON rp.content_hash = l.result_ref"
)


def _row_to_entry(row: tuple) -> LedgerEntry:
    return LedgerEntry(
        entry_key=row[0], call_fn=row[1],
        inputs=json.loads(row[2]), exec_identity=json.loads(row[3]),
        ts=row[4], kind=row[5], verdict=row[6], subject=row[7],
        result=json.loads(row[8]),
    )


def get_latest(
    conn: sqlite3.Connection,
    *,
    call_fn: str,
    inputs: Mapping[str, Any],
    exec_identity: ExecIdentity,
) -> LedgerEntry | None:
    """Newest-ts entry for this exact ``(call_fn, inputs, exec_identity)`` key."""
    ek = entry_key(call_fn, inputs, exec_identity)
    row = conn.execute(
        _SELECT + " WHERE l.entry_key = ? ORDER BY l.ts DESC, l.id DESC LIMIT 1",
        (ek,),
    ).fetchone()
    return _row_to_entry(row) if row else None


def history(
    conn: sqlite3.Connection,
    *,
    call_fn: str,
    inputs: Mapping[str, Any],
    exec_identity: ExecIdentity,
) -> list[LedgerEntry]:
    """All entries for the key, newest first (audit / lineage)."""
    ek = entry_key(call_fn, inputs, exec_identity)
    rows = conn.execute(
        _SELECT + " WHERE l.entry_key = ? ORDER BY l.ts DESC, l.id DESC", (ek,),
    ).fetchall()
    return [_row_to_entry(r) for r in rows]


def is_closed(
    conn: sqlite3.Connection,
    *,
    call_fn: str,
    inputs: Mapping[str, Any],
    exec_identity: ExecIdentity,
) -> bool:
    latest = get_latest(conn, call_fn=call_fn, inputs=inputs, exec_identity=exec_identity)
    return bool(latest and latest.is_closed)


# ---------------------------------------------------------------------------
# Read-side pull — what makes the ledger pay for itself (skip closed sub-goals).
# ---------------------------------------------------------------------------


def _latest_per_key(conn: sqlite3.Connection, exec_identity: ExecIdentity) -> list[LedgerEntry]:
    """Latest entry per key for one execution (the projection / pull substrate)."""
    rows = conn.execute(
        _SELECT + " WHERE l.exec_ref = ? ORDER BY l.ts ASC, l.id ASC",
        (exec_identity.ref,),
    ).fetchall()
    latest: dict[str, LedgerEntry] = {}
    for r in rows:
        e = _row_to_entry(r)
        latest[e.entry_key] = e          # ascending order → last write wins
    return list(latest.values())


def closed_subjects(
    conn: sqlite3.Connection,
    exec_identity: ExecIdentity,
    *,
    call_fn: str | None = None,
) -> set[str]:
    """Subjects whose latest entry is closed (for THIS execution only).

    The next round filters its candidate sub-goals against this set: an already
    closed sub-goal (a located producer, a static terminal) is reused, not
    re-derived. Cross-run entries are excluded by ``exec_ref`` — a different
    execution's closes do not silently satisfy this run."""
    out: set[str] = set()
    for e in _latest_per_key(conn, exec_identity):
        if call_fn is not None and e.call_fn != call_fn:
            continue
        if e.is_closed and e.subject:
            out.add(e.subject)
    return out


def should_skip(
    conn: sqlite3.Connection,
    *,
    call_fn: str,
    inputs: Mapping[str, Any],
    exec_identity: ExecIdentity,
) -> bool:
    """True when this sub-goal is already closed in the ledger — skip re-derive."""
    return is_closed(conn, call_fn=call_fn, inputs=inputs, exec_identity=exec_identity)


# ---------------------------------------------------------------------------
# Projection — the markdown view that replaces hand-written notes.
# ---------------------------------------------------------------------------


def _render_kv(value: Any, indent: int = 0) -> list[str]:
    """Render a (possibly nested) payload as readable markdown bullet lines."""
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}- **{k}**:")
                lines.extend(_render_kv(v, indent + 1))
            else:
                lines.append(f"{pad}- **{k}**: {v}")
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, (dict, list)):
                lines.extend(_render_kv(item, indent))
            else:
                lines.append(f"{pad}- {item}")
    else:
        lines.append(f"{pad}- {value}")
    return lines


def project_stage_map(
    conn: sqlite3.Connection,
    exec_identity: ExecIdentity,
    *,
    ts: str | None = None,
    with_header: bool = True,
) -> str:
    """Render the global stage map for one execution as markdown.

    Built from the landed producer / static-terminal / open-hypothesis entries
    — the map the agent used to redraw by hand (and lose mid-case). Scoped to
    ``exec_identity`` so two runs never bleed into one map.

    With ``with_header`` (default) the output opens with the ``utov-export``
    stamp (:func:`engine.export_stamp.build_export_header`): it marks the file
    authoritative (vs an agent's hand-written notes), records the execution, and
    lists the ledger entry keys it was generated from (traceable). This is the
    agent's per-round view — write it to a fixed path with :func:`write_stage_view`
    so "query the last step" means reading this, not a hand-written json."""
    entries = _latest_per_key(conn, exec_identity)
    by_kind: dict[str, list[LedgerEntry]] = {}
    for e in entries:
        by_kind.setdefault(e.kind, []).append(e)

    eid = exec_identity.canonical()
    lines: list[str] = [
        f"# CVD stage map — {eid['target']} · input {eid['input_hash']} · "
        f"run {eid['run_id']}" + (f" · nonce {eid['nonce']}" if "nonce" in eid else ""),
        "",
        f"_{len(entries)} stamped entr{'y' if len(entries) == 1 else 'ies'} "
        f"(latest per key)_",
        "",
    ]

    def _anchor(e: LedgerEntry) -> str:
        a = e.result.get("anchor")
        return a.get("pc_hex", "") if isinstance(a, dict) else ""

    # Run summary (drive's roll-up) first — the run-level overview that replaces
    # the agent's hand-written *_report.json. Rendered as readable key/value.
    for e in by_kind.get(LedgerKind.RUN_SUMMARY.value, []):
        lines.append(f"## Run summary — {e.subject or e.call_fn}")
        lines.extend(_render_kv(e.result))
        lines.append("")

    sections = [
        (LedgerKind.PRODUCER.value,        "## Producers (SUCCESS)"),
        (LedgerKind.SINK.value,            "## Sinks (confirmed)"),
        (LedgerKind.STATIC_TERMINAL.value, "## Static terminals"),
        (LedgerKind.OPEN_HYPOTHESIS.value, "## Open hypotheses (NEEDS_OBSERVATION)"),
        (LedgerKind.TRANSFORM.value,       "## Recovered transforms"),
        (LedgerKind.EMIT.value,            "## Emitted deliverables (F + parity)"),
        (LedgerKind.VERDICT.value,         "## Other verdicts"),
    ]
    for kind_v, header in sections:
        rows = by_kind.get(kind_v, [])
        if not rows:
            continue
        lines.append(header)
        for e in sorted(rows, key=lambda x: x.subject):
            anchor = _anchor(e)
            tail = f" @ {anchor}" if anchor else ""
            if kind_v == LedgerKind.OPEN_HYPOTHESIS.value:
                tail += f" (watch_n={e.result.get('watch_n', 0)})"
            if kind_v == LedgerKind.EMIT.value and "parity" in e.result:
                tail += f" (parity={e.result.get('parity')})"
            lines.append(f"- `{e.call_fn}` · {e.subject or '(no subject)'}"
                         f"{tail} — {e.verdict or e.kind}")
        lines.append("")
    body = "\n".join(lines).rstrip() + "\n"
    if not with_header:
        return body
    header = build_export_header(
        source="utov/cvd_ledger.sqlite",
        exported_by="project_stage_map",
        exec_identity=exec_identity.canonical(),
        from_entries=sorted(e.entry_key for e in entries),
        ts=ts or _now_iso(),
    )
    return header + "\n" + body


def ledger_dir(conn: sqlite3.Connection) -> Path | None:
    """Public: the directory holding the ledger DB (None for ``:memory:``).

    The stamped ``cvd_ledger_view.md`` lives here (written by ``auto_view``)."""
    return _db_dir(conn)


def _db_dir(conn: sqlite3.Connection) -> Path | None:
    """The directory of a file-backed ledger DB (None for ``:memory:``)."""
    try:
        for _seq, name, file in conn.execute("PRAGMA database_list"):
            if name == "main" and file:
                return Path(file).parent
    except sqlite3.Error:
        pass
    return None


def _refresh_view(
    conn: sqlite3.Connection, exec_identity: ExecIdentity, *,
    ts: str | None = None, filename: str = "cvd_ledger_view.md",
) -> Path | None:
    """Rewrite the stamped projection next to the ledger DB (no-op for memory DBs).

    Called automatically on every :func:`record_call` so the readable view can
    never drift from — or be forgotten alongside — the sqlite."""
    d = _db_dir(conn)
    if d is None:
        return None
    path = d / filename
    try:
        path.write_text(
            project_stage_map(conn, exec_identity, ts=ts), encoding="utf-8")
    except OSError:
        return None
    return path


def write_stage_view(
    conn: sqlite3.Connection,
    exec_identity: ExecIdentity,
    work: WorkDir | str | Path,
    *,
    ts: str | None = None,
    filename: str = "cvd_ledger_view.md",
) -> Path:
    """Write the (stamped) stage map to a fixed path — the agent's per-round view.

    The whole point of the read-side consolidation: the agent stops hand-writing
    a ``*_report.json`` each round and reads THIS instead (more complete than the
    hand notes, scoped per execution, machine-reusable, and stamped so its
    authority is self-evident). Returns the path written. With a
    :class:`store.WorkDir` the view lands at ``<run>/cvd_ledger_view.md``."""
    md = project_stage_map(conn, exec_identity, ts=ts, with_header=True)
    if isinstance(work, WorkDir):
        path = work.root / filename
    else:
        path = Path(work)
        if path.is_dir():
            path = path / filename
    path.write_text(md, encoding="utf-8")
    return path


__all__ = [
    "ExecIdentity",
    "LedgerKind",
    "LedgerEntry",
    "kind_for_verdict",
    "entry_key",
    "open_ledger",
    "record_call",
    "record_verdict",
    "get_latest",
    "history",
    "is_closed",
    "closed_subjects",
    "should_skip",
    "project_stage_map",
    "write_stage_view",
    "ledger_dir",
    "read_payload",
]
