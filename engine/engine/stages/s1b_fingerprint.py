"""S1.5: cheap fingerprint pre-scan (PLAN §12.5, DECISIONS D-016, D-027).

For each fingerprint that fires at ≥ MIN_HITS positions, we create ONE
hypothesis (with a normalized payload describing the claim) plus N anchors
pointing at the actual instruction indices. The 95 fingerprint × 10 hits =
950 occurrences → 95 hyp rows + 950 anchor rows, NOT 950 hyp rows.

Each hyp gets tagged with:
    source = "plugin"           (D-027 axis)
    primitive = "SHA-256" / ...  (when known, for grouping)
    verdict = strong | medium | weak
    category = hash | cipher_sym | crc | mac | ecc

Also writes an inspection summary jsonl + publishes anchor idxs into session
for downstream S4 consumption.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from ..data.fingerprints import (
    FINGERPRINTS,
    INSTR_PATTERNS,
    PER_BLOCK_PRIMITIVE,
    Confidence,
)
from ..hyp_tree import HypTree
from ..store import open_hypotheses_db
from ..types import Instruction

CODE_VERSION = "s1b-v2"

MIN_HITS = 1
SAMPLE_LIMIT = 64        # how many anchors to record per fingerprint (cap)


_CONFIDENCE_SCORE = {
    Confidence.STRONG: 0.85,
    Confidence.MEDIUM: 0.65,
    Confidence.WEAK:   0.35,
}


@dataclass(frozen=True)
class FingerprintHit:
    name: str
    category: str
    confidence: str
    primitive: str | None
    magic_hex: str | None
    match_text: str | None
    total_hits: int
    # (trace_idx, pc) pairs — kept on disk for inspection; anchors go to DB.
    sample_anchors: list[tuple[int, int]]


def scan(items: Iterable[Instruction]) -> list[FingerprintHit]:
    items_list = list(items) if not isinstance(items, list) else items
    by_magic = {fp.magic: fp for fp in FINGERPRINTS}

    scalar_anchors: dict[str, list[tuple[int, int]]] = {}
    for ins in items_list:
        for val in ins.regs_write.values():
            fp = by_magic.get(val)
            if fp is not None:
                scalar_anchors.setdefault(fp.name, []).append((ins.idx, ins.pc))

    pattern_anchors: dict[str, list[tuple[int, int]]] = {}
    for ins in items_list:
        for pat in INSTR_PATTERNS:
            if pat.match_text in ins.mnemonic:
                pattern_anchors.setdefault(pat.name, []).append((ins.idx, ins.pc))

    hits: list[FingerprintHit] = []
    for fp in FINGERPRINTS:
        anchors = scalar_anchors.get(fp.name, [])
        if len(anchors) < MIN_HITS:
            continue
        hits.append(FingerprintHit(
            name=fp.name,
            category=fp.category,
            confidence=fp.confidence.value,
            primitive=PER_BLOCK_PRIMITIVE.get(fp.name),
            magic_hex=f"0x{fp.magic:x}",
            match_text=None,
            total_hits=len(anchors),
            sample_anchors=anchors[:SAMPLE_LIMIT],
        ))
    for pat in INSTR_PATTERNS:
        anchors = pattern_anchors.get(pat.name, [])
        if len(anchors) < MIN_HITS:
            continue
        hits.append(FingerprintHit(
            name=pat.name,
            category=pat.category,
            confidence=pat.confidence.value,
            primitive=pat.primitive,
            magic_hex=None,
            match_text=pat.match_text,
            total_hits=len(anchors),
            sample_anchors=anchors[:SAMPLE_LIMIT],
        ))
    return hits


def run(ctx) -> dict:
    items = ctx["items"]
    work = ctx["work"]
    hits = scan(items)

    # 1) Inspection summary (human-readable).
    out_path = work.root / "stage_outputs" / "s1b.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for h in hits:
            row = asdict(h)
            # convert tuples to lists for JSON
            row["sample_anchors"] = [[a, b] for a, b in h.sample_anchors]
            f.write(json.dumps(row) + "\n")

    # 2) Seed the ledger using the dedup-aware schema.
    conn = open_hypotheses_db(work)
    tree = HypTree(conn)
    seeded = 0
    all_anchor_idxs: list[int] = []
    try:
        for h in hits:
            score = _CONFIDENCE_SCORE[Confidence(h.confidence)]
            subject = h.primitive or h.name
            payload = {
                "fingerprint": h.name,
                "category": h.category,
                "verdict": h.confidence,
                "magic": h.magic_hex,
                "match_text": h.match_text,
                # Note: NO inline "occurrences" array here — anchors live in
                # the dedicated hyp_anchors table.
            }
            tags = [
                ("source",     "plugin"),
                ("category",   h.category),
                ("verdict",    h.confidence),
            ]
            if h.primitive:
                tags.append(("primitive", h.primitive))
            tree.add(
                parent_id=None,
                kind="algo_signature",
                subject=subject,
                payload=payload,
                confidence=score,
                source="plugin",
                anchors=h.sample_anchors,
                tags=tags,
                created_in_stage="s1b",
            )
            seeded += 1
            all_anchor_idxs.extend(idx for idx, _pc in h.sample_anchors)
    finally:
        conn.close()

    # 3) Publish into session for S4 (smart sink picker)
    session = ctx.get("session")
    if session is not None:
        session["fingerprint_anchor_idxs"] = sorted(set(all_anchor_idxs))
        strong_prims = [h.primitive for h in hits
                        if h.confidence == "strong" and h.primitive]
        if strong_prims:
            session.setdefault("algo_hints", []).extend(strong_prims)

    work.mark_stage_done("s1b", CODE_VERSION)
    return {
        "stage": "s1b",
        "fingerprint_hits": len(hits),
        "hypotheses_seeded": seeded,
        "total_anchors": len(all_anchor_idxs),
        "anchor_idx_count": len(set(all_anchor_idxs)),
        "out": str(out_path),
    }
