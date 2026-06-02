"""Authority projection — utov's "where do we stand right now" single view.

Origin: dev-authority-projection-spec (B7), clark 2026-06-02. A narrow agent
re-reading 20 reports can still surface a *superseded* conclusion (e.g. read
``[x19+0x47] stack65`` as "oracle already closed" long after a later claim
demoted it to "captured but not oracle-equivalent"). This module gives the agent
one queryable view of the **current authority surface** for a case:

  - input: every claim Hypotask retrieved for a case (each carries ``supersedes``
    / ``updated_at`` / ``evidence_refs``);
  - output: ``authority_projection`` — the claims NOT superseded by any CURRENT
    claim. A superseded claim is *demoted* (kept, traceable, but off the top
    surface), never deleted.

Role boundary (locked):
  - Hypotask still only stores + retrieves; it does NOT judge. utov *judges*
    here — it computes which claims are still authoritative.
  - The projection is purely **read + project**: it never re-derives evidence,
    never decides a case's oracle, never feeds a close/parity gate, never
    computes any case formula (no SHA-512, no cipher verdict). It only pushes
    older conclusions down and surfaces the current face.
  - Additive: this module never touches Hypotask's storage/retrieval path. It is
    a read-side projection layered on top of whatever the caller retrieved.

The core (:func:`project_authority`) is source-agnostic — it takes a list of
claim dicts, not a session — so it is trivially testable and works for any
case. :func:`claims_from_findings` is the (read-only) adapter that maps Hypotask
finding rows into claim dicts; the caller may also build claims any other way.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Sequence

from .export_stamp import export_stamped_json
from .store import _now_iso

_log = logging.getLogger(__name__)

# A claim is CURRENT (live authority) unless its status says otherwise. Anything
# explicitly demoted (superseded / retired / withdrawn) is NOT current. The
# projection only treats CURRENT claims as able to supersede others — a demoted
# claim cannot push another claim down (otherwise C would "revive" when its
# superseder B is itself superseded by A; contract 2).
CURRENT_VERDICT = "CURRENT"

# Verdicts that mean "this claim is no longer the live authority". Extensible
# per spec ("可扩展(如 RETIRED)"). A claim whose verdict is none of these and is
# not CURRENT is treated conservatively as non-authoritative too (we only let an
# explicit CURRENT verdict drive supersession), but it is still surfaced in the
# demoted list so nothing is silently dropped.
_DEMOTED_VERDICTS = frozenset({"SUPERSEDED", "RETIRED", "WITHDRAWN"})


def _claim_id(claim: Mapping[str, Any]) -> str | None:
    """The stable id of a claim (``claim`` field, falling back to ``id``)."""
    cid = claim.get("claim")
    if cid is None:
        cid = claim.get("id")
    return str(cid) if cid is not None else None


def _verdict(claim: Mapping[str, Any]) -> str:
    """The claim's verdict/status (``verdict`` field, falling back to ``status``).

    Empty/missing → treated as CURRENT so a bare claim with no explicit demotion
    is live by default (a claim Hypotask stored is a standing conclusion until
    something supersedes it)."""
    v = claim.get("verdict")
    if v is None:
        v = claim.get("status")
    if v is None or str(v).strip() == "":
        return CURRENT_VERDICT
    return str(v)


def _is_current(claim: Mapping[str, Any]) -> bool:
    """Is this claim itself CURRENT (eligible to supersede others)?"""
    v = _verdict(claim)
    return v == CURRENT_VERDICT


def _supersedes(claim: Mapping[str, Any]) -> list[str]:
    """The claim ids this claim supersedes (``[]`` when none / malformed)."""
    raw = claim.get("supersedes") or []
    if isinstance(raw, str):
        raw = [raw]
    out: list[str] = []
    for x in raw:
        if x is None:
            continue
        out.append(str(x))
    return out


def _updated_at(claim: Mapping[str, Any]) -> str:
    """Recency key (``updated_at``); empty string when absent so sorts are stable."""
    u = claim.get("updated_at")
    return "" if u is None else str(u)


def _detect_cycles(
    edges: Mapping[str, set[str]],
) -> list[list[str]]:
    """Every supersede cycle in the directed graph (A→B means A supersedes B).

    Returns a list of cycles (each a list of node ids in encounter order). Used
    to WARN loudly instead of looping forever (contract 2: 环 → 显式 WARN, 不静默
    死循环). Iterative DFS with a recursion stack so we never actually recurse
    into a loop."""
    cycles: list[list[str]] = []
    seen_cycle_keys: set[frozenset[str]] = set()
    color: dict[str, int] = {}  # 0=unvisited, 1=in-stack, 2=done

    for root in edges:
        if color.get(root, 0) != 0:
            continue
        # Iterative DFS carrying the path, so we can extract the cycle slice.
        stack: list[tuple[str, list[str]]] = [(root, [root])]
        color[root] = 1
        while stack:
            node, path = stack[-1]
            advanced = False
            for nxt in sorted(edges.get(node, ())):
                c = color.get(nxt, 0)
                if c == 0:
                    color[nxt] = 1
                    stack.append((nxt, path + [nxt]))
                    advanced = True
                    break
                if c == 1:
                    # Back-edge into the active stack → cycle from nxt..node.
                    if nxt in path:
                        cyc = path[path.index(nxt):]
                        key = frozenset(cyc)
                        if key not in seen_cycle_keys:
                            seen_cycle_keys.add(key)
                            cycles.append(cyc)
            if not advanced:
                color[node] = 2
                stack.pop()
    return cycles


def project_authority(
    case: str,
    claims: Sequence[Mapping[str, Any]],
    *,
    status: str | None = None,
    next_blocker: str | None = None,
) -> dict[str, Any]:
    """Compute the authority projection for ``case`` from its ``claims``.

    Pure read + project (no session, no I/O, no judgment of the case content):

      1. A claim is *demoted* iff some **CURRENT** claim supersedes it
         (transitively along CURRENT→CURRENT supersede edges). This is the
         supersede closure: A supersedes B and B supersedes C → with A CURRENT, B
         demoted; C stays demoted (it was already superseded by B) — C does NOT
         revive just because B is no longer the top (contract 2). Only CURRENT
         claims propagate supersession, so a demoted B cannot keep pushing C down
         on its own, but C is reached directly when A also lists it OR when B
         (still a CURRENT-verdict claim that merely lost its top spot) — see note.
      2. ``authoritative_claims`` = the CURRENT claims that nobody CURRENT
         supersedes, sorted by ``updated_at`` descending (most recent first), id
         as tiebreak — all of them, never silently merged/truncated (contract 3).
      3. ``demoted_claims`` = everything pushed off the surface, each with
         ``superseded_by`` so the demotion is traceable (验收: 被取代降级且可追溯).
      4. Cycles in the supersede graph → ``warnings`` entry + ``logging.warning``;
         the closure still terminates (no infinite loop, contract 2).

    Empty ``claims`` → empty projection with an explicit status, never an error
    (A8 boundary).
    """
    # Index claims by id; later duplicates win (latest write), but record the
    # collision so nothing is silently swallowed.
    by_id: dict[str, Mapping[str, Any]] = {}
    anon: list[Mapping[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for c in claims:
        cid = _claim_id(c)
        if cid is None:
            anon.append(c)
            continue
        if cid in by_id:
            warnings.append({
                "kind": "duplicate_claim_id",
                "claim": cid,
                "note": "multiple claims share this id; keeping the last seen",
            })
        by_id[cid] = c

    # Build the CURRENT→target supersede edges. ONLY a CURRENT claim's supersedes
    # propagate (contract 2: a demoted claim must not keep something else down).
    edges: dict[str, set[str]] = {cid: set() for cid in by_id}
    for cid, c in by_id.items():
        if not _is_current(c):
            continue
        for tgt in _supersedes(c):
            edges[cid].add(tgt)

    # Cycle detection over the CURRENT supersede graph → WARN, do not loop.
    cycles = _detect_cycles(edges)
    for cyc in cycles:
        warnings.append({
            "kind": "supersede_cycle",
            "cycle": cyc,
            "note": ("supersede cycle detected; claims in the cycle are NOT "
                     "treated as authoritative (cannot resolve a single CURRENT "
                     "face) — resolve the cycle in the source claims"),
        })
        _log.warning("authority_projection: supersede cycle in case %r: %s",
                     case, " -> ".join(cyc + [cyc[0]]))
    cycle_members: set[str] = {n for cyc in cycles for n in cyc}

    # Transitive closure of "is superseded by some CURRENT claim". A target is
    # demoted if ANY CURRENT claim reaches it through CURRENT supersede edges.
    superseded_by: dict[str, set[str]] = {}
    for src, tgts in edges.items():
        # Walk forward from src; src itself is the (current) superseder of all
        # nodes reachable through CURRENT edges.
        stack = list(tgts)
        visited: set[str] = set()
        while stack:
            t = stack.pop()
            if t in visited:
                continue
            visited.add(t)
            superseded_by.setdefault(t, set()).add(src)
            # Continue the chain only through CURRENT claims (a CURRENT B that
            # supersedes C means A→B→C all collapse onto A as well).
            nxt_claim = by_id.get(t)
            if nxt_claim is not None and _is_current(nxt_claim):
                for nxt in edges.get(t, ()):
                    if nxt not in visited:
                        stack.append(nxt)

    demoted_ids = set(superseded_by) | cycle_members
    # A claim that is non-CURRENT by its own verdict is also off the top surface,
    # but it is only "demoted by supersession" if something superseded it; pure
    # status-demotion is surfaced separately so the reason is honest.

    authoritative: list[dict[str, Any]] = []
    demoted: list[dict[str, Any]] = []

    def _projected(c: Mapping[str, Any], cid: str | None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "claim": cid,
            "verdict": _verdict(c),
        }
        if _supersedes(c):
            out["supersedes"] = _supersedes(c)
        out["evidence_refs"] = list(c.get("evidence_refs") or [])
        if c.get("confidence") is not None:
            out["confidence"] = c.get("confidence")
        if _updated_at(c):
            out["updated_at"] = _updated_at(c)
        return out

    for cid, c in by_id.items():
        proj = _projected(c, cid)
        in_cycle = cid in cycle_members
        is_current = _is_current(c)
        is_superseded = cid in superseded_by
        if is_current and not is_superseded and not in_cycle:
            authoritative.append(proj)
        else:
            d = dict(proj)
            if is_superseded:
                d["superseded_by"] = sorted(superseded_by[cid])
            if in_cycle:
                d["in_supersede_cycle"] = True
            if not is_current and not is_superseded and not in_cycle:
                # Demoted purely by its own verdict (RETIRED / SUPERSEDED / …).
                d["demoted_reason"] = "non_current_verdict"
            demoted.append(d)

    # Anonymous (id-less) claims cannot participate in the supersede graph; keep
    # them visible as authoritative-if-CURRENT, never dropped.
    for c in anon:
        warnings.append({
            "kind": "claim_without_id",
            "note": "claim has no 'claim'/'id' field; cannot supersede or be "
                    "superseded — surfaced as-is",
        })
        proj = _projected(c, None)
        if _is_current(c):
            authoritative.append(proj)
        else:
            demoted.append(proj)

    # Sort by recency (most recent first), id as a stable tiebreak. Never merge
    # or truncate same-topic CURRENT claims (contract 3).
    def _sort_key(p: Mapping[str, Any]) -> tuple[str, str]:
        return (p.get("updated_at", ""), str(p.get("claim") or ""))

    authoritative.sort(key=_sort_key, reverse=True)
    demoted.sort(key=_sort_key, reverse=True)

    if status is None:
        status = "EMPTY_NO_CLAIMS" if not claims else "OPEN"

    projection: dict[str, Any] = {
        "kind": "authority_projection",
        "case": case,
        "status": status,
        "authoritative_claims": authoritative,
        "demoted_claims": demoted,
        "next_blocker": next_blocker,
    }
    if warnings:
        projection["warnings"] = warnings
    return projection


def claims_from_findings(findings: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Map Hypotask finding rows → claim dicts (read-only adapter).

    Hypotask retrieves findings (via ``integration.adapter.active_findings`` /
    ``get_active_findings``); this is the choke point that reads those rows into
    the claim shape :func:`project_authority` consumes. It NEVER writes back and
    never touches Hypotask's storage path — it only reads fields. A finding that
    already carries a ``claim`` id keeps it; otherwise the ``finding_id`` is used
    so every row maps to a stable claim id.

    Field mapping (best-effort, additive — unknown shapes degrade to a bare
    claim, never crash): ``claim``←claim/finding_id, ``verdict``←verdict/status,
    ``supersedes``←supersedes (default []), ``evidence_refs``←evidence_refs /
    [finding_id] / [], ``confidence``←confidence, ``updated_at``←
    updated_at/created_at."""
    out: list[dict[str, Any]] = []
    for f in findings:
        fid = f.get("finding_id") or f.get("id")
        cid = f.get("claim") or fid
        claim: dict[str, Any] = {"claim": str(cid) if cid is not None else None}
        v = f.get("verdict") or f.get("status")
        if v is not None:
            claim["verdict"] = v
        sup = f.get("supersedes")
        if sup:
            claim["supersedes"] = list(sup) if not isinstance(sup, str) else [sup]
        refs = f.get("evidence_refs")
        if refs:
            claim["evidence_refs"] = list(refs)
        elif fid is not None:
            claim["evidence_refs"] = [str(fid)]
        if f.get("confidence") is not None:
            claim["confidence"] = f.get("confidence")
        upd = f.get("updated_at") or f.get("created_at")
        if upd is not None:
            claim["updated_at"] = upd
        out.append(claim)
    return out


def export_authority_projection(
    projection: Mapping[str, Any],
    *,
    source: str = "hypotask (findings) · utov authority_projection",
    exported_by: str = "engine.authority_projection.project_authority",
    exec_identity: Mapping[str, Any] | None = None,
    ts: str | None = None,
) -> str:
    """Render the projection as a stamped JSON export (utov-export header).

    Reuses :func:`engine.export_stamp.export_stamped_json` so the file carries
    the authoritative ``<!-- utov-export ... -->`` discriminator (contract 4 — an
    agent's hand-written file must NOT carry it). ``from_entries`` are the claim
    ids the projection was built from (traceable)."""
    auth = projection.get("authoritative_claims") or []
    demoted = projection.get("demoted_claims") or []
    from_entries = [
        str(c.get("claim")) for c in list(auth) + list(demoted)
        if c.get("claim") is not None
    ]
    return export_stamped_json(
        projection,
        source=source,
        exported_by=exported_by,
        exec_identity=exec_identity or {"case": projection.get("case", "")},
        ts=ts or _now_iso(),
        from_entries=from_entries,
    )


__all__ = [
    "CURRENT_VERDICT",
    "project_authority",
    "claims_from_findings",
    "export_authority_projection",
]
