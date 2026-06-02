"""utov-export stamp — every file utov emits carries a provenance header.

Origin: `cvd-ledger-read-side-consolidate`. Once utov projects a ledger to a
markdown view, that view looks identical to an agent's hand-written notes — the
filename alone cannot say which is authoritative, and "newest / most accurate
lives inside utov" becomes unverifiable from the file. So every file utov
exports opens with a ``<!-- utov-export ... -->`` header that:

  1. discriminates a utov export (authoritative, machine-derived) from an agent
     hand-written file (which must NOT carry this header) — the header IS the
     discriminator;
  2. makes the file traceable to the ledger entries that generated it
     (``from_entries``);
  3. lets "the latest / most accurate is inside utov" be verified from the file
     content (``authority`` + ``source``).

Applies to ALL utov exports, not just the CVD ledger projection. The header is
an HTML comment so it renders invisibly in markdown viewers yet is trivially
machine-parseable (:func:`parse_export_header`).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable, Mapping

_log = logging.getLogger(__name__)

_MARKER = "utov-export"

# --------------------------------------------------------------------------- #
# Export-filename honesty gate — dev-export-filename-honesty-spec.md.
#
# engine does NOT name files by verdict (the caller passes the filename); a name
# like ``output_provenance_confirmed.json`` is a CONSUMER-side strong-closure
# assertion. This gate never rewrites the caller's name (that is the consumer's
# naming right); it refuses to SILENTLY emit a name that lies — WARN-loud + an
# explicit ``filename_verdict_mismatch`` field stamped INTO the file, so a reader
# knows "the name claims a closure the content does not support; trust the
# content verdict".
# --------------------------------------------------------------------------- #

# The closure-claim token set — data-driven constant, defined once (契约 1).
# A filename carrying any of these (case-insensitive, on a word boundary) is
# asserting a STRONG closure. Matched against the basename with its extension
# stripped. A NEGATED prefix (unconfirmed / unclosed / not_identified / …) is a
# disclaimer, NOT a claim — those are exempt (契约 4 / A8 negation exemption).
CLOSURE_CLAIM_TOKENS: frozenset[str] = frozenset({
    "confirmed", "closed", "identified", "oracle", "solved",
})

# Negation prefixes that turn a claim token into a disclaimer ("unconfirmed",
# "not_closed", "non-oracle"). Word-boundary aware so "reconfirmed" is NOT a
# negation (it still claims) — only a leading negation prefix immediately before
# the token disarms it.
_NEGATION_PREFIXES: tuple[str, ...] = ("un", "not", "non", "no", "in")


def _strip_ext(name: str) -> str:
    """The basename with a trailing file extension removed (``a/b.foo.json`` →
    ``b.foo``). Only the path tail matters — a parent dir named ``.../confirmed/``
    is not a filename claim (the file's own name is what asserts the verdict)."""
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    # Drop a single trailing extension (the suffix after the LAST dot), if any.
    if "." in base:
        base = base.rsplit(".", 1)[0]
    return base


def filename_closure_claims(name: str) -> list[str]:
    """The closure-claim tokens a filename asserts (``[]`` when none).

    Case-insensitive, word-boundary matched against the extension-stripped
    basename; a token immediately preceded by a negation prefix
    (``unconfirmed`` / ``not_closed`` / ``non-oracle``) is a DISCLAIMER and is
    NOT counted (契约 4). De-duplicated, in declaration order for stability."""
    stem = _strip_ext(name).lower()
    found: list[str] = []
    for token in CLOSURE_CLAIM_TOKENS:
        # Word boundary on both sides; the boundary chars (``_``, ``-``, digits,
        # other letters) are handled by \b plus an explicit separator class so
        # "output_confirmed_v2" matches but "preconfirmedness" (no boundary) does
        # not falsely claim, and "unconfirmed" is caught by the negation check.
        for m in re.finditer(rf"(?<![a-z]){re.escape(token)}(?![a-z])", stem):
            start = m.start()
            # Look at the immediately-preceding alnum run (the prefix glued to the
            # token by an underscore/hyphen or directly): if it is a negation
            # prefix, this occurrence is a disclaimer, not a claim.
            prefix_seg = re.search(r"([a-z]+)[_\-]?$", stem[:start])
            if prefix_seg and prefix_seg.group(1) in _NEGATION_PREFIXES:
                continue
            if token not in found:
                found.append(token)
            break
    # preserve declaration order for deterministic output
    return [t for t in ("confirmed", "closed", "identified", "oracle", "solved")
            if t in found]


def _walk_closure_classifications(obj: Any):
    """Yield every embedded closure-classification dict found anywhere in a
    nested result structure (lists / dicts).

    Discriminated by ``kind == "closure_classification"`` (the marker
    :meth:`ClosureClassification.to_dict` writes) — this READS the closure layer's
    own conclusion, it does not re-derive it (契约 3 / A8: no new judgment)."""
    if isinstance(obj, Mapping):
        if obj.get("kind") == "closure_classification":
            yield obj
        for v in obj.values():
            yield from _walk_closure_classifications(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_closure_classifications(v)


def result_supports_strong_closure(payload: Mapping[str, Any]) -> bool:
    """Does this run's content support a STRONG closure claim?

    REUSES the closure-evidence layering verdict (契约 3): a strong claim
    (confirmed / closed / identified / oracle) is supported ONLY when the run
    carries at least one closure classification at the ORACLE level
    (``algorithm_closed`` True — output_sink_confirmed && provenance_closed &&
    parity_exact, derived by closure_classification.classify_closure). No
    embedded oracle-closed classification → NOT supported (a NEEDS_OBSERVATION /
    BLOCKED / PENDING / non-oracle TERMINAL / window-local parity-EXACT run).

    This is purely a READ of an already-computed conclusion; it never recomputes
    sink / provenance / parity and never consults a close/parity gate (invariant
    7)."""
    for cc in _walk_closure_classifications(payload):
        if cc.get("algorithm_closed") is True or cc.get("closure_level") == "oracle":
            return True
    return False


def filename_verdict_mismatch(
    name: str, payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    """The explicit ``filename_verdict_mismatch`` record, or ``None`` if honest.

    ``None`` when the filename asserts no closure claim (regression: behaviour
    unchanged) OR the run's content actually supports the strong claim
    (oracle-closed → honest naming, let it through — 验收②). Otherwise a record
    naming what the filename CLAIMS vs what the content actually shows, so the
    reader trusts the content verdict over the name (契约 2 / never silent)."""
    claims = filename_closure_claims(name)
    if not claims:
        return None
    if result_supports_strong_closure(payload):
        return None
    actual_outcome = payload.get("outcome")
    actual_verdict = payload.get("verdict") or None
    return {
        "kind": "filename_verdict_mismatch",
        "filename": _strip_ext(name).replace("\\", "/").rsplit("/", 1)[-1],
        "filename_claim": claims,
        "actual_outcome": actual_outcome,
        "actual_verdict": actual_verdict,
        "claim_supported": False,
        "explanation": (
            "the filename asserts a strong closure "
            f"({', '.join(claims)}) but the run content does NOT support it: no "
            "oracle-closed classification (output_sink_confirmed && "
            "provenance_closed && parity_exact) is present. The filename is the "
            "CONSUMER's naming choice (utov does not rewrite it); trust this "
            "content verdict, not the name. To name it honestly, derive a name "
            "from the verdict (see safe_export_name)."),
    }


def safe_export_name(
    payload: Mapping[str, Any], *, stem: str = "cvd_gap_map", ext: str = "json",
) -> str:
    """A neutral / honest filename DERIVED from the run's verdict (契约 4 helper).

    The consumer may call this PROACTIVELY to get a name that does not over-claim:
    an oracle-closed run earns a ``..._oracle_closed`` suffix; anything else gets
    its actual lowercased outcome (``..._needs_observation`` / ``..._terminal`` /
    ``..._collected`` / …). Never emits a strong-claim token a non-oracle run
    cannot back, so the round-tripped name passes :func:`filename_verdict_mismatch`
    with no mismatch. utov does not FORCE this — it is opt-in."""
    if result_supports_strong_closure(payload):
        suffix = "oracle_closed"
    else:
        outcome = str(payload.get("outcome") or "open").lower()
        suffix = outcome
    return f"{stem}_{suffix}.{ext}"

# The standing authority line: machine-derived, the live truth is the ledger,
# do not hand-edit. (Mirrors the spec's 盖章 wording.)
DEFAULT_AUTHORITY = "machine-derived · 最新权威以 cvd_ledger.sqlite 为准 · 勿手改"

_JSON_FIELDS = ("exec_identity", "from_entries")


def build_export_header(
    *,
    source: str,
    exported_by: str,
    exec_identity: Mapping[str, Any],
    from_entries: Iterable[str],
    ts: str,
    authority: str = DEFAULT_AUTHORITY,
) -> str:
    """Build the ``<!-- utov-export ... -->`` header block (trailing newline).

    ``source`` is where the authoritative data lives (e.g.
    ``utov/cvd_ledger.sqlite``); ``exported_by`` is the utov function that
    rendered the file; ``exec_identity`` is the execution the file belongs to;
    ``from_entries`` are the ledger entry keys it was generated from (traceable);
    ``ts`` is the caller-supplied export stamp."""
    return (
        f"<!-- {_MARKER}\n"
        f"source:        {source}\n"
        f"exported_by:   {exported_by}\n"
        f"exec_identity: "
        f"{json.dumps(dict(exec_identity), sort_keys=True, separators=(',', ':'))}\n"
        f"from_entries:  {json.dumps(list(from_entries))}\n"
        f"ts:            {ts}\n"
        f"authority:     {authority}\n"
        f"-->\n"
    )


# Authority line for an OUT-layer JSON projection handed to a consumer (test-
# agent): the file itself IS the authoritative artifact — it must NOT point the
# consumer back at the sqlite ledger (that is utov's internal COLLECT layer; the
# consumer never touches it). See dev-consumer-output-stamped-json-not-sqlite.md.
CONSUMER_EXPORT_AUTHORITY = (
    "machine-derived utov export (出层投影) · 此 JSON 即权威产出 · 勿手改 · "
    "consumer 不消费内部 sqlite 账本")


def export_stamped_json(
    payload: Mapping[str, Any],
    *,
    source: str,
    exported_by: str,
    exec_identity: Mapping[str, Any],
    ts: str,
    from_entries: Iterable[str] = (),
    authority: str = CONSUMER_EXPORT_AUTHORITY,
    indent: int = 2,
) -> str:
    """Render ``payload`` as a stamped JSON document (header block + JSON body).

    The OUT-layer projection a consumer (test-agent) reads as its standard output
    / log: a ``<!-- utov-export ... -->`` header (the authority discriminator —
    hand-written files lack it) followed by the JSON body. Use for
    ``CvdResult.to_dict()`` (the collect gap map) and ``DriveResult.to_dict()``.
    The body stays a structured summary, never a trace dump (the producers already
    keep it compact — output-backtrack addendum). ``default=str`` tolerates any
    stray non-JSON scalar without crashing the export."""
    header = build_export_header(
        source=source, exported_by=exported_by, exec_identity=exec_identity,
        from_entries=from_entries, ts=ts, authority=authority)
    body = json.dumps(dict(payload), indent=indent, sort_keys=True, default=str)
    return header + body + "\n"


def load_stamped_json(text: str) -> tuple[dict[str, Any] | None, Any]:
    """Read a stamped JSON document back to ``(header_or_None, payload)``.

    Strips the ``<!-- utov-export ... -->`` header (if present) and JSON-decodes
    the body. A header of ``None`` means the text was NOT a utov export (a hand-
    written file) — the discriminator the consumer can act on."""
    s = text.lstrip()
    if not is_utov_export(s):
        return None, json.loads(s)
    end = s.find("-->")
    if end == -1:
        raise ValueError("utov-export header is not terminated (no '-->')")
    body = s[end + 3:].lstrip()
    return parse_export_header(s), json.loads(body)


def is_utov_export(text: str) -> bool:
    """True iff ``text`` opens with the utov-export header (the discriminator).

    An agent hand-written file does not carry the header, so this cleanly tells
    authoritative utov exports from hand-written notes."""
    return text.lstrip().startswith(f"<!-- {_MARKER}")


def parse_export_header(text: str) -> dict[str, Any] | None:
    """Parse the header back to a dict (``None`` if ``text`` is not a utov export).

    ``exec_identity`` and ``from_entries`` are JSON-decoded; the rest stay
    strings. Lets a consumer verify which execution / entries produced a file."""
    s = text.lstrip()
    if not s.startswith(f"<!-- {_MARKER}"):
        return None
    end = s.find("-->")
    if end == -1:
        return None
    block = s[len(f"<!-- {_MARKER}"):end]
    out: dict[str, Any] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key in _JSON_FIELDS:
            try:
                value = json.loads(value)
            except (ValueError, json.JSONDecodeError):
                pass
        out[key] = value
    return out


__all__ = [
    "DEFAULT_AUTHORITY",
    "CONSUMER_EXPORT_AUTHORITY",
    "CLOSURE_CLAIM_TOKENS",
    "build_export_header",
    "is_utov_export",
    "parse_export_header",
    "export_stamped_json",
    "load_stamped_json",
    "filename_closure_claims",
    "result_supports_strong_closure",
    "filename_verdict_mismatch",
    "safe_export_name",
]
