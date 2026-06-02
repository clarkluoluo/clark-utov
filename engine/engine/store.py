"""Persistence: 2 SQLite databases per (target, input, run), 7 tables total.

Per-run directory layout:

    work/<target>/<input_hash>/runs/<run_id>/
        ├── findings.sqlite              ← verified facts (geode layer)
        ├── hypotheses.sqlite            ← active backtracking ledger (7 tables)
        ├── archived/
        │   └── hyps_<ts>.sqlite         ← abandoned subtrees, off the hot path
        ├── stage_outputs/sN.jsonl
        ├── conformance_report.json
        ├── meta.json
        ├── session.json
        └── notes/

`hypotheses.sqlite` schema — 6 tables, separation of concerns:

    hyp_payloads        content-addressed blob store. Big JSON values live HERE
                        ONCE; everything else refs by content_hash. Solves the
                        "ledger explodes when 10k handlers each dump 4 KB of
                        state" problem.

    claim_templates     One row per distinct (kind, source, payload_ref) tuple.
                        If 10k different traces all hypothesise "this is XOR
                        x4 = x1 ^ x2", that's still ONE template row + 10k
                        hypotheses rows pointing at it.

    hypotheses          The actual N-ary tree of instances. Each row is small:
                        template ptr, parent ptr, status, subject, verifier
                        result ptr. No JSON inline.

    hyp_anchors         (hyp_id, trace_idx, pc). Many-to-many: a hyp can
                        cover N trace points; a trace point can be touched by
                        many hyps. Indexed both ways for fast position-→-hyp
                        and hyp-→-positions queries.

    hyp_tags            (hyp_id, axis, value). Multi-axis tagging (source,
                        primitive, vmp_version, confidence_bucket, ...).
                        Axes are agreed-upon strings, no schema lock-in.

    hyp_dependencies    (from_hyp_id, to_hyp_id, kind). Cross-tree semantic
                        dependence — distinct from tree structure parent_id.
                        If we refute hyp X, walk this for downstream
                        invalidation candidates.

`findings.sqlite`:

    findings(id, stage, kind, subject, payload_ref, verified_at,
             verifier_strategy, origin_hyp_id)
    + the same hyp_payloads table for blob storage

WAL is enabled on both DBs. Concurrent reads + serialised writes — agent_mode
and script_mode can both write safely.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _make_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{ts}-{short}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def content_hash(payload: dict[str, Any] | str) -> str:
    """Deterministic SHA1 of a JSON payload (or pre-serialized JSON string)."""
    if isinstance(payload, str):
        s = payload
    else:
        s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def template_hash(kind: str, source: str, payload_hash: str) -> str:
    return hashlib.sha1(f"{kind}\0{source}\0{payload_hash}".encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# WorkDir
# ---------------------------------------------------------------------------


class WorkDir:
    """Owner of a single (target, input_hash, run_id) work directory."""

    def __init__(
        self,
        root: str | Path,
        target: str,
        input_hash: str,
        run_id: str | None = None,
        new_run: bool = False,
    ):
        self.target_dir = Path(root) / target / input_hash
        self.target_dir.mkdir(parents=True, exist_ok=True)
        runs_dir = self.target_dir / "runs"
        runs_dir.mkdir(exist_ok=True)
        latest = self.target_dir / "latest"

        if new_run or (run_id is None and not latest.exists()):
            run_id = run_id or _make_run_id()
            self.root = runs_dir / run_id
            self.root.mkdir(parents=True, exist_ok=True)
            self._update_latest_symlink(latest, run_id)
        elif run_id is not None:
            self.root = runs_dir / run_id
            self.root.mkdir(parents=True, exist_ok=True)
        else:
            self.root = latest.resolve()

        self.run_id = self.root.name
        for sub in ("stage_outputs", "anomalies", "notes", "archived"):
            (self.root / sub).mkdir(exist_ok=True)

    @staticmethod
    def _update_latest_symlink(latest: Path, run_id: str) -> None:
        if latest.is_symlink() or latest.exists():
            try:
                latest.unlink()
            except OSError:
                pass
        os.symlink(Path("runs") / run_id, latest, target_is_directory=True)

    @property
    def stage_state_path(self) -> Path:
        return self.root / "stage_state.json"

    def stage_output_path(self, stage_name: str) -> Path:
        return self.root / "stage_outputs" / f"{stage_name}.parquet"

    def read_stage_state(self) -> dict[str, str]:
        if not self.stage_state_path.exists():
            return {}
        return json.loads(self.stage_state_path.read_text())

    def mark_stage_done(self, stage_name: str, code_version: str) -> None:
        state = self.read_stage_state()
        state[stage_name] = code_version
        self.stage_state_path.write_text(json.dumps(state, indent=2))

    def is_stage_done(self, stage_name: str, code_version: str) -> bool:
        return self.read_stage_state().get(stage_name) == code_version


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


_PRAGMA_WAL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;
"""


_HYPOTHESES_SCHEMA = """
CREATE TABLE IF NOT EXISTS hyp_payloads (
    content_hash TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    bytes_len INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claim_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    source TEXT NOT NULL,
    payload_ref TEXT NOT NULL,
    template_hash TEXT NOT NULL UNIQUE,
    confidence REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(payload_ref) REFERENCES hyp_payloads(content_hash)
);
CREATE INDEX IF NOT EXISTS ix_template_kind   ON claim_templates(kind);
CREATE INDEX IF NOT EXISTS ix_template_source ON claim_templates(source);

CREATE TABLE IF NOT EXISTS hypotheses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    template_id INTEGER NOT NULL,
    parent_id INTEGER,
    depth INTEGER NOT NULL,
    status TEXT NOT NULL,
    subject TEXT NOT NULL,
    verifier_result_ref TEXT,
    verdict_at TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(template_id) REFERENCES claim_templates(id),
    FOREIGN KEY(parent_id)   REFERENCES hypotheses(id),
    FOREIGN KEY(verifier_result_ref) REFERENCES hyp_payloads(content_hash)
);
CREATE INDEX IF NOT EXISTS ix_hyp_parent   ON hypotheses(parent_id);
CREATE INDEX IF NOT EXISTS ix_hyp_status   ON hypotheses(status);
CREATE INDEX IF NOT EXISTS ix_hyp_template ON hypotheses(template_id);

CREATE TABLE IF NOT EXISTS hyp_anchors (
    hyp_id    INTEGER NOT NULL,
    trace_idx INTEGER NOT NULL,
    pc        INTEGER NOT NULL,
    PRIMARY KEY (hyp_id, trace_idx),
    FOREIGN KEY(hyp_id) REFERENCES hypotheses(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_anchor_idx ON hyp_anchors(trace_idx);
CREATE INDEX IF NOT EXISTS ix_anchor_pc  ON hyp_anchors(pc);

CREATE TABLE IF NOT EXISTS hyp_tags (
    hyp_id INTEGER NOT NULL,
    axis   TEXT NOT NULL,
    value  TEXT NOT NULL,
    PRIMARY KEY (hyp_id, axis, value),
    FOREIGN KEY(hyp_id) REFERENCES hypotheses(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_tag_axis_value ON hyp_tags(axis, value);

CREATE TABLE IF NOT EXISTS hyp_dependencies (
    from_hyp_id INTEGER NOT NULL,
    to_hyp_id   INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    PRIMARY KEY (from_hyp_id, to_hyp_id, kind),
    FOREIGN KEY(from_hyp_id) REFERENCES hypotheses(id) ON DELETE CASCADE,
    FOREIGN KEY(to_hyp_id)   REFERENCES hypotheses(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_dep_to   ON hyp_dependencies(to_hyp_id);
CREATE INDEX IF NOT EXISTS ix_dep_from ON hyp_dependencies(from_hyp_id);

-- Audit trail (PLAN §15 agent intervention留痕). Every Core method that
-- mutates state writes a row. Replay-from-stage also logs a single big row.
CREATE TABLE IF NOT EXISTS interventions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    actor TEXT NOT NULL,            -- 'agent' | 'user' | 'script' | 'system'
    action TEXT NOT NULL,           -- 'override_verdict' | 'force_status' |
                                    -- 'inject_finding' | 'add_tag' |
                                    -- 'add_dependency' | 'rerun_from_stage' |
                                    -- 'pause' | 'resume' | 'submit_hyp' | ...
    target_table TEXT,              -- 'hypotheses' | 'findings' | 'stage_state' | ...
    target_id TEXT,                 -- hyp_id / stage_name / etc., stringified
    before_ref TEXT,                -- content_hash → hyp_payloads (snapshot before)
    after_ref TEXT,                 -- content_hash → hyp_payloads (snapshot after)
    reason TEXT                     -- free-text rationale supplied by actor
);
CREATE INDEX IF NOT EXISTS ix_interv_actor  ON interventions(actor);
CREATE INDEX IF NOT EXISTS ix_interv_action ON interventions(action);
CREATE INDEX IF NOT EXISTS ix_interv_ts     ON interventions(timestamp);
"""


_FINDINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS hyp_payloads (
    content_hash TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    bytes_len INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stage TEXT NOT NULL,
    kind TEXT NOT NULL,
    subject TEXT NOT NULL,
    payload_ref TEXT NOT NULL,
    verified_at TEXT NOT NULL,
    verifier_strategy TEXT NOT NULL,
    origin_hyp_id INTEGER,
    -- 0526Plan B1: which deterministic / agent / LLM path produced this
    -- finding. One of: plugin, s5_deterministic, s5_triton, s5_fold_idiom,
    -- s5_algorithm_fit, s6_llm, agent_override, manual_inject, unknown.
    source TEXT NOT NULL DEFAULT 'unknown',
    -- 0527 preprocess-batch: when a finding was promoted as part of a
    -- one-call deterministic batch (Core.preprocess_batch), the batch's
    -- short uuid lands here. NULL for ad-hoc promote_to_finding calls
    -- and historical findings. Lets agents bulk-review / bulk-discard
    -- the set of findings a single batch produced.
    batch_id TEXT,
    FOREIGN KEY(payload_ref) REFERENCES hyp_payloads(content_hash)
);
CREATE INDEX IF NOT EXISTS ix_findings_kind     ON findings(kind);
CREATE INDEX IF NOT EXISTS ix_findings_subject  ON findings(subject);
CREATE INDEX IF NOT EXISTS ix_findings_origin   ON findings(origin_hyp_id);
CREATE INDEX IF NOT EXISTS ix_findings_source   ON findings(source);
CREATE INDEX IF NOT EXISTS ix_findings_stage    ON findings(stage);
CREATE INDEX IF NOT EXISTS ix_findings_batch_id ON findings(batch_id);

-- 0526Plan C4.0: layer-1 fold idioms (e.g. SHA-2 σ/Σ) reference their
-- constituent layer-0 findings (e.g. the 3 underlying ROR/EOR steps). One
-- row per (idiom finding, constituent layer-0 finding) link. `role` is an
-- optional label like "rotr_7" or "shr_3" that lets a downstream agent
-- reconstruct the algebraic structure without re-deriving it.
CREATE TABLE IF NOT EXISTS finding_groups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    idiom_name TEXT NOT NULL,
    parent_finding_id INTEGER NOT NULL,
    member_finding_id INTEGER NOT NULL,
    role TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(parent_finding_id) REFERENCES findings(id),
    FOREIGN KEY(member_finding_id) REFERENCES findings(id)
);
CREATE INDEX IF NOT EXISTS ix_fgroups_parent ON finding_groups(parent_finding_id);
CREATE INDEX IF NOT EXISTS ix_fgroups_member ON finding_groups(member_finding_id);
CREATE INDEX IF NOT EXISTS ix_fgroups_idiom  ON finding_groups(idiom_name);
"""


def _open_db(path: Path, schema_sql: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(_PRAGMA_WAL)
    conn.executescript(schema_sql)
    conn.commit()
    return conn


def open_hypotheses_db(work: WorkDir) -> sqlite3.Connection:
    conn = _open_db(work.root / "hypotheses.sqlite", _HYPOTHESES_SCHEMA)
    # Idempotent migration: add created_in_stage column for rerun_from_stage
    # cascade. Old DBs created before this column existed get it appended.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(hypotheses)").fetchall()}
    if "created_in_stage" not in cols:
        conn.execute("ALTER TABLE hypotheses ADD COLUMN created_in_stage TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_hyp_stage ON hypotheses(created_in_stage)"
        )
        conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Audit helper
# ---------------------------------------------------------------------------


def log_intervention(
    conn: sqlite3.Connection,
    *,
    actor: str,
    action: str,
    target_table: str | None = None,
    target_id: Any = None,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    reason: str | None = None,
) -> int:
    """Append an audit row to interventions. Snapshots stored content-addressed
    in hyp_payloads (same table that holds hyp payloads — fine because
    everything is hash-keyed)."""
    before_ref = upsert_payload(conn, before) if before is not None else None
    after_ref  = upsert_payload(conn, after)  if after  is not None else None
    cur = conn.execute(
        "INSERT INTO interventions(timestamp, actor, action, target_table,"
        " target_id, before_ref, after_ref, reason)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (_now_iso(), actor, action, target_table,
         (None if target_id is None else str(target_id)),
         before_ref, after_ref, reason),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def read_interventions(
    conn: sqlite3.Connection, *, limit: int = 100,
    action: str | None = None, actor: str | None = None,
) -> list[dict[str, Any]]:
    sql = ("SELECT id, timestamp, actor, action, target_table, target_id,"
           " before_ref, after_ref, reason FROM interventions")
    args: list[Any] = []
    clauses: list[str] = []
    if action is not None:
        clauses.append("action = ?")
        args.append(action)
    if actor is not None:
        clauses.append("actor = ?")
        args.append(actor)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    rows = conn.execute(sql, args).fetchall()
    return [{
        "id": r[0], "timestamp": r[1], "actor": r[2], "action": r[3],
        "target_table": r[4], "target_id": r[5],
        "before_ref": r[6], "after_ref": r[7], "reason": r[8],
    } for r in rows]


def open_findings_db(work: WorkDir) -> sqlite3.Connection:
    conn = _open_db(work.root / "findings.sqlite", _FINDINGS_SCHEMA)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(findings)").fetchall()}
    # B1: idempotent migration — older runs were created before `source`
    # existed. Add the column with a safe default so old rows answer
    # `--by-source` queries without breaking.
    if "source" not in cols:
        conn.execute(
            "ALTER TABLE findings ADD COLUMN source TEXT NOT NULL DEFAULT 'unknown'"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_findings_source ON findings(source)"
        )
        conn.commit()
    # 0527: idempotent migration — `batch_id` for preprocess-batch tagging.
    if "batch_id" not in cols:
        conn.execute("ALTER TABLE findings ADD COLUMN batch_id TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_findings_batch_id ON findings(batch_id)"
        )
        conn.commit()
    return conn


def link_finding_group_members(
    conn: sqlite3.Connection,
    *,
    parent_finding_id: int,
    idiom_name: str,
    members: list[tuple[int, str | None]],
) -> None:
    """Bind a layer-1 idiom finding to its layer-0 constituents (0526Plan C4.0).

    `members` is a list of (member_finding_id, role) tuples — `role` lets the
    consumer reconstruct algebra (e.g. ("rotr_7", "shr_3") for SHA-256 σ0).
    Idempotent on (parent, member) — running the discoverer twice does not
    duplicate rows.
    """
    rows = [
        (idiom_name, parent_finding_id, member_id, role, _now_iso())
        for member_id, role in members
    ]
    if not rows:
        return
    conn.executemany(
        "INSERT INTO finding_groups("
        "idiom_name, parent_finding_id, member_finding_id, role, created_at"
        ") SELECT ?, ?, ?, ?, ? WHERE NOT EXISTS ("
        "  SELECT 1 FROM finding_groups "
        "  WHERE parent_finding_id = ? AND member_finding_id = ?"
        ")",
        [(*r, r[1], r[2]) for r in rows],
    )
    conn.commit()


def get_finding_group_members(
    conn: sqlite3.Connection, parent_finding_id: int,
) -> list[dict[str, Any]]:
    """Return all layer-0 constituents of a layer-1 idiom finding."""
    rows = conn.execute(
        "SELECT id, idiom_name, member_finding_id, role, created_at "
        "FROM finding_groups WHERE parent_finding_id = ? "
        "ORDER BY id ASC",
        (parent_finding_id,),
    ).fetchall()
    return [
        {"id": r[0], "idiom_name": r[1], "member_finding_id": r[2],
         "role": r[3], "created_at": r[4]}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Payload upsert helper (used by both DBs)
# ---------------------------------------------------------------------------


def upsert_payload(conn: sqlite3.Connection, payload: dict[str, Any] | str) -> str:
    """Idempotently insert payload into hyp_payloads. Returns content_hash."""
    if isinstance(payload, str):
        s = payload
    else:
        s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    ch = hashlib.sha1(s.encode("utf-8")).hexdigest()
    conn.execute(
        "INSERT OR IGNORE INTO hyp_payloads(content_hash, payload, bytes_len, created_at)"
        " VALUES (?, ?, ?, ?)",
        (ch, s, len(s), _now_iso()),
    )
    return ch


def read_payload(conn: sqlite3.Connection, content_hash_: str) -> dict[str, Any]:
    row = conn.execute(
        "SELECT payload FROM hyp_payloads WHERE content_hash = ?", (content_hash_,),
    ).fetchone()
    if row is None:
        raise KeyError(content_hash_)
    return json.loads(row[0])


# ---------------------------------------------------------------------------
# Archive helper (PLAN §1.4 — abandoned subtree off-hot-path)
# ---------------------------------------------------------------------------


def archive_subtree(
    conn: sqlite3.Connection,
    work: WorkDir,
    root_hyp_ids: Iterable[int],
) -> Path:
    """Move (DELETE + copy out) abandoned subtrees rooted at the given hyp ids
    into archived/hyps_<timestamp>.sqlite. Active connection stays slim."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    arch_path = work.root / "archived" / f"hyps_{ts}_{uuid.uuid4().hex[:4]}.sqlite"
    arch = _open_db(arch_path, _HYPOTHESES_SCHEMA)
    try:
        # Collect all descendant ids first.
        to_move: set[int] = set()
        stack = list(root_hyp_ids)
        while stack:
            nid = stack.pop()
            if nid in to_move:
                continue
            to_move.add(nid)
            stack.extend(r[0] for r in conn.execute(
                "SELECT id FROM hypotheses WHERE parent_id = ?", (nid,),
            ).fetchall())

        if not to_move:
            return arch_path

        # Copy required payload + template rows into archive.
        ids_csv = ",".join(str(i) for i in sorted(to_move))
        # template_ids the subtree references:
        tmpl_ids = {r[0] for r in conn.execute(
            f"SELECT DISTINCT template_id FROM hypotheses WHERE id IN ({ids_csv})"
        )}
        if tmpl_ids:
            tmpl_csv = ",".join(str(i) for i in sorted(tmpl_ids))
            for trow in conn.execute(
                f"SELECT id, kind, source, payload_ref, template_hash, confidence, created_at"
                f" FROM claim_templates WHERE id IN ({tmpl_csv})"
            ):
                arch.execute("INSERT OR IGNORE INTO claim_templates VALUES (?,?,?,?,?,?,?)", trow)
            # Pull payloads referenced by templates + verifier_results
            payload_hashes: set[str] = {r[0] for r in conn.execute(
                f"SELECT DISTINCT payload_ref FROM claim_templates WHERE id IN ({tmpl_csv})"
            )}
        else:
            payload_hashes = set()
        payload_hashes |= {
            r[0] for r in conn.execute(
                f"SELECT DISTINCT verifier_result_ref FROM hypotheses"
                f" WHERE id IN ({ids_csv}) AND verifier_result_ref IS NOT NULL"
            )
        }
        for ph in payload_hashes:
            row = conn.execute(
                "SELECT content_hash, payload, bytes_len, created_at FROM hyp_payloads"
                " WHERE content_hash = ?", (ph,),
            ).fetchone()
            if row:
                arch.execute("INSERT OR IGNORE INTO hyp_payloads VALUES (?,?,?,?)", row)
        for hrow in conn.execute(
            f"SELECT id, template_id, parent_id, depth, status, subject,"
            f" verifier_result_ref, verdict_at, created_at"
            f" FROM hypotheses WHERE id IN ({ids_csv})"
        ):
            arch.execute("INSERT OR IGNORE INTO hypotheses VALUES (?,?,?,?,?,?,?,?,?)", hrow)
        # anchors / tags / deps
        for arow in conn.execute(f"SELECT * FROM hyp_anchors WHERE hyp_id IN ({ids_csv})"):
            arch.execute("INSERT OR IGNORE INTO hyp_anchors VALUES (?,?,?)", arow)
        for trow in conn.execute(f"SELECT * FROM hyp_tags WHERE hyp_id IN ({ids_csv})"):
            arch.execute("INSERT OR IGNORE INTO hyp_tags VALUES (?,?,?)", trow)
        for drow in conn.execute(
            f"SELECT * FROM hyp_dependencies WHERE from_hyp_id IN ({ids_csv}) OR to_hyp_id IN ({ids_csv})"
        ):
            arch.execute("INSERT OR IGNORE INTO hyp_dependencies VALUES (?,?,?)", drow)
        arch.commit()

        # Now DELETE from active (cascades to anchors/tags/deps via FK).
        conn.execute(f"DELETE FROM hypotheses WHERE id IN ({ids_csv})")
        conn.commit()
    finally:
        arch.close()
    return arch_path
