"""Tier 1 emitter — render `algorithm_identified` findings as
paste-and-read algorithm pseudocode.

FEATURE-REQUEST-1: utov's final deliverable used to top out at
`algorithm_identified: {algorithm, evidence_score, anchors_seen}` —
that's algorithm IDENTIFICATION (a label + a confidence score). What a
human / agent actually wants is algorithm RECONSTRUCTION: the same
label paired with the IV constants, the K table when fingerprinted,
the σ/Σ idiom PCs + register assignments, and the loop-iteration
counts the trace exhibits. All the inputs are already in
`findings.sqlite` + `s3_dfg.jsonl` + `meta.json`; this module just
projects them through an algorithm-spec template.

The emitter reads from the engine's persisted artifacts only — it
does NOT introspect Core's in-memory state. Result: it works
unchanged whether the run was driven by Live mode, File mode, or a
resume.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

from .data.algorithm_pseudocode import ALGORITHM_SPECS

# ---------------------------------------------------------------------------
# Low-level: pull a single payload by ref (uses raw SQL so this stays
# usable from outside Core, including the CLI which has no live Core).
# ---------------------------------------------------------------------------


def _payload(con: sqlite3.Connection, ref: str | None) -> dict[str, Any] | None:
    if not ref:
        return None
    row = con.execute(
        "SELECT payload FROM hyp_payloads WHERE content_hash=?", (ref,)
    ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except (TypeError, json.JSONDecodeError):
        return None


def _latest_algorithm_identified(
    con: sqlite3.Connection,
    *,
    supported: set[str] | None = None,
) -> tuple[int, dict[str, Any]] | None:
    """Return the latest algorithm finding (``algorithm_hyp`` or the reserved
    strong ``algorithm_identified``).

    The structural matcher emits ``algorithm_hyp`` (a pre-oracle-closure HYPOTHESIS
    carrying a local-closure trap — task 7); the strong ``algorithm_identified`` is
    reserved for whole-case oracle closure. Both are rendered (the emit clearly
    marks a hyp's trap — task 7④); reading only one kind would drop the matcher's
    output now that it lands as a hyp.

    If ``supported`` is given, prefer the most recent finding whose
    `algorithm` label is in the set. Falls back to "no row matches" only
    if every row is unsupported — in which case the caller surfaces an
    EmitterError listing the supported templates. Without this guard a
    run that promotes both SHA-512 and AES would error out on "no
    template for AES" while ignoring the perfectly emit-able SHA-512.
    """
    rows = con.execute(
        "SELECT id, payload_ref FROM findings "
        "WHERE kind IN ('algorithm_hyp','algorithm_identified') "
        "ORDER BY verified_at DESC, id DESC"
    ).fetchall()
    if not rows:
        return None
    if supported is None:
        fid, ref = rows[0][0], rows[0][1]
        return fid, (_payload(con, ref) or {})
    for fid, ref in rows:
        pl = _payload(con, ref) or {}
        if pl.get("algorithm") in supported:
            return fid, pl
    # Every row's algorithm is unsupported — return the latest row anyway
    # so the caller can render a useful error message naming the algo.
    fid, ref = rows[0][0], rows[0][1]
    return fid, (_payload(con, ref) or {})


def _collect_ivs(
    con: sqlite3.Connection, prefix: str, n: int,
) -> list[tuple[str, str | None, str | None]]:
    rows = con.execute(
        "SELECT subject, payload_ref, source FROM findings "
        "WHERE kind='algo_signature' AND subject LIKE ?",
        (f"{prefix}.h%",),
    ).fetchall()
    by_name: dict[str, tuple[str | None, str | None]] = {}
    for subj, ref, src in rows:
        p = _payload(con, ref) or {}
        by_name[subj] = (p.get("magic"), src)
    out: list[tuple[str, str | None, str | None]] = []
    for i in range(n):
        name = f"{prefix}.h{i}"
        magic, src = by_name.get(name, (None, None))
        out.append((name, magic, src))
    return out


def _collect_ks(
    con: sqlite3.Connection, prefix: str,
) -> list[tuple[str, str | None, str | None]]:
    rows = con.execute(
        "SELECT subject, payload_ref, source FROM findings "
        "WHERE kind='algo_signature'"
    ).fetchall()
    matches: list[tuple[str, str | None, str | None]] = []
    for _subj, ref, src in rows:
        p = _payload(con, ref) or {}
        fp = p.get("fingerprint", "")
        if isinstance(fp, str) and fp.startswith(f"{prefix}.K["):
            matches.append((fp, p.get("magic"), src))

    def _idx(name: str) -> int:
        m = re.search(r"\[(\d+)\]", name)
        return int(m.group(1)) if m else 0

    matches.sort(key=lambda x: _idx(x[0]))
    return matches


def _collect_folds(con: sqlite3.Connection) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    rows = con.execute(
        "SELECT id, subject, source, payload_ref FROM findings "
        "WHERE kind='fold_idiom' ORDER BY id"
    ).fetchall()
    for fid, subj, src, ref in rows:
        p = _payload(con, ref) or {}
        components = p.get("components") or []
        pcs = p.get("anchor_pcs")
        if pcs is None:
            pcs = []
            for c in components:
                pc_field = c.get("pc")
                if isinstance(pc_field, str):
                    try:
                        pcs.append(int(pc_field, 16))
                    except ValueError:
                        continue
                elif isinstance(pc_field, int):
                    pcs.append(pc_field)
        amounts = [(c.get("kind"), c.get("amount")) for c in components]
        out.append({
            "fid":       fid,
            "subject":   subj,
            "idiom":     p.get("idiom"),
            "pcs":       pcs,
            "amounts":   amounts,
            "input_reg": p.get("input_reg"),
            "dst_reg":   p.get("dst_reg"),
            "source":    src,
        })
    return out


def _pc_frequency(run_dir: Path) -> Counter:
    counter: Counter = Counter()
    path = run_dir / "stage_outputs" / "s3_dfg.jsonl"
    if not path.exists():
        return counter
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            counter[row.get("pc")] += 1
    return counter


def _infer_blocks(
    folds: list[dict[str, Any]],
    pc_freq: Counter,
    iters_per_block: int,
) -> int | None:
    """Best-effort nblocks: pick the median int-division of leading-PC
    execution counts that divide cleanly by iters_per_block."""
    candidates: list[int] = []
    for f in folds:
        if not f["pcs"]:
            continue
        lead = f["pcs"][0]
        if isinstance(lead, str):
            try:
                lead = int(lead, 16)
            except ValueError:
                continue
        for pc_key, count in pc_freq.items():
            try:
                if int(pc_key, 16) == lead:
                    if count >= iters_per_block and count % iters_per_block == 0:
                        candidates.append(count // iters_per_block)
                    break
            except (TypeError, ValueError):
                continue
    if not candidates:
        return None
    candidates.sort()
    return candidates[len(candidates) // 2]


# ---------------------------------------------------------------------------
# Format helpers — pure string transforms; format-flag knob for markdown vs
# plain text rendering.
# ---------------------------------------------------------------------------


def _format_constants(
    ivs: list[tuple[str, str | None, str | None]],
    ks: list[tuple[str, str | None, str | None]],
    k_spec_count: int,
) -> str:
    lines: list[str] = ["  Initial hash values (H[0..7]):"]
    for name, magic, src in ivs:
        if magic is None:
            lines.append(f"    {name:14}  [NOT FOUND]")
        else:
            tag = "(plugin)" if src == "plugin" else f"({src})"
            lines.append(f"    {name:14}  {str(magic):20} {tag}")
    if ks:
        lines.append("")
        lines.append(
            f"  Round constants K (fingerprinted: {len(ks)}; "
            f"algorithm uses {k_spec_count}):"
        )
        for name, magic, src in ks:
            tag = "(plugin)" if src == "plugin" else f"({src})"
            lines.append(f"    {name:14}  {str(magic):20} {tag}")
    else:
        lines.append("")
        lines.append(
            "  Round constants K: [no K-table fingerprints in this run]"
        )
    return "\n".join(lines)


def _format_folds(
    folds: list[dict[str, Any]],
    idiom_names: list[str],
    pc_freq: Counter,
) -> str:
    by_idiom: dict[str, list[dict[str, Any]]] = {}
    for f in folds:
        if f["idiom"]:
            by_idiom.setdefault(f["idiom"], []).append(f)
    lines: list[str] = []
    for nm in idiom_names:
        hits = by_idiom.get(nm, [])
        if not hits:
            lines.append(f"  {nm:18}  [NOT IDENTIFIED]")
            continue
        for h in hits:
            amounts_s = ",".join(f"{k}{a}" for k, a in h["amounts"] if k)
            pcs_hex = [
                p if isinstance(p, str) else f"0x{p:08x}" for p in h["pcs"]
            ]
            lead = h["pcs"][0] if h["pcs"] else None
            if isinstance(lead, str):
                try:
                    lead = int(lead, 16)
                except ValueError:
                    lead = None
            freq: int | None = None
            if lead is not None:
                for pc_key, c in pc_freq.items():
                    try:
                        if int(pc_key, 16) == lead:
                            freq = c
                            break
                    except (TypeError, ValueError):
                        continue
            freq_s = f" ×{freq}" if freq else ""
            tag = " (injected)" if h["source"] == "agent_override" else ""
            lines.append(
                f"  {nm:18}  pcs={pcs_hex}  amounts=({amounts_s})  "
                f"in={h['input_reg']}→dst={h['dst_reg']}{freq_s}{tag}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public surface — `emit(run_dir, ...)`. The function returns the rendered
# string AND optionally writes it to a file-like (so callers can both pipe
# it and capture it in one call).
# ---------------------------------------------------------------------------


class EmitterError(RuntimeError):
    """Raised on user-actionable errors (missing run, missing finding,
    unsupported algorithm). The CLI catches this and exits non-zero."""


def emit(
    run_dir: str | Path,
    *,
    out: TextIO | None = None,
    fmt: str = "text",
) -> str:
    """Render Tier 1 pseudocode for the run at ``run_dir``.

    Args:
        run_dir: a `<work_root>/<target_dir>/runs/<run_id>` path containing
                 `findings.sqlite` + `stage_outputs/s3_dfg.jsonl` +
                 `meta.json`.
        out:     optional file-like. If non-None, the rendered text is
                 written here (flushed). If None, only returned.
        fmt:     "text" (default, plain) or "markdown" (fenced).

    Returns:
        The full rendered text.

    Raises:
        EmitterError if the run dir has no findings.sqlite, no
        `algorithm_identified` finding, or an unsupported algorithm
        label. The CLI catches and downgrades to a non-zero exit.
    """
    rd = Path(run_dir).resolve()
    findings_path = rd / "findings.sqlite"
    if not findings_path.exists():
        raise EmitterError(f"{findings_path} does not exist")

    con = sqlite3.connect(findings_path)
    try:
        fit = _latest_algorithm_identified(con, supported=set(ALGORITHM_SPECS))
        if not fit:
            raise EmitterError(
                f"no `algorithm_hyp` / `algorithm_identified` finding in "
                f"{findings_path}. "
                f"Run `utov status <run_dir>` to inspect what's there."
            )
        _fit_id, fit_p = fit
        algo = fit_p.get("algorithm")
        spec = ALGORITHM_SPECS.get(algo) if isinstance(algo, str) else None
        if spec is None:
            raise EmitterError(
                f"no emitter template for algorithm {algo!r}. "
                f"Supported: {sorted(ALGORITHM_SPECS)}."
            )
        prefix = spec["prefix"]
        ivs   = _collect_ivs(con, prefix, spec["iv_count"])
        ks    = _collect_ks(con, prefix)
        folds = _collect_folds(con)
        # Ch / Maj come from `handler_semantic` findings with `ch@` /
        # `maj@` subject prefixes (BR-8 #2). We collect them here so the
        # diagnostic note below can tell the user whether the matcher
        # captured them.
        ch_hits = con.execute(
            "SELECT subject FROM findings "
            "WHERE kind='handler_semantic' AND subject LIKE 'ch@%'"
        ).fetchall()
        maj_hits = con.execute(
            "SELECT subject FROM findings "
            "WHERE kind='handler_semantic' AND subject LIKE 'maj@%'"
        ).fetchall()
    finally:
        con.close()

    pc_freq = _pc_frequency(rd)

    meta: dict[str, Any] = {}
    meta_path = rd / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
    target_name = (
        meta.get("target_name")
        or meta.get("target")
        or (meta.get("target_meta") or {}).get("target_name")
        or "?"
    )
    out_len = (
        meta.get("output_length")
        or meta.get("output_len")
        or (meta.get("target_meta") or {}).get("output_length")
    )

    sigma_full = [f"{prefix}.{n}" for n in spec["sigma_idioms"]]
    Sigma_full = [f"{prefix}.{n}" for n in spec["Sigma_idioms"]]

    n_msched = _infer_blocks(
        [f for f in folds if f["idiom"] in set(sigma_full)],
        pc_freq, spec["msched_iters_per_block"],
    )
    n_compress = _infer_blocks(
        [f for f in folds if f["idiom"] in set(Sigma_full)],
        pc_freq, spec["compress_iters_per_block"],
    )

    anchors_seen = fit_p.get("anchors_seen") or []
    anchors_expected = fit_p.get("anchors_expected") or []
    evidence = fit_p.get("evidence_score")

    bar = "=" * 64
    sections: list[str] = []
    sections.append(bar)
    sections.append(f"{algo} reconstruction  ({target_name})")
    sections.append(
        f"  evidence_score: {evidence}   "
        f"anchors: {len(anchors_seen)}/{len(anchors_expected)}"
    )
    if out_len is not None:
        sections.append(f"  output_length:  {out_len} bytes")
    # task 7④: render the local-closure trap LOUDLY at the top so a reader never
    # mistakes a pre-oracle-closure HYPOTHESIS for a final identification. A finding
    # that carries a ``closure`` trap is an ``algorithm_hyp``; one without (a
    # whole-case oracle-closed ``algorithm_identified``) prints no banner.
    closure = fit_p.get("closure") or {}
    trap = closure.get("trap_state")
    if trap and trap != "NONE":
        sections.append(f"  ** {trap} — algorithm HYPOTHESIS, NOT a closed algorithm **")
        if closure.get("next_step"):
            sections.append(f"     next: {closure.get('next_step')}")
    sections.append(bar)
    sections.append("")
    sections.append("Constants  (verified plugin fingerprints, * = injected):")
    sections.append("")
    sections.append(_format_constants(ivs, ks, spec["k_count"]))
    sections.append("")

    m_lo, m_hi = spec["msched_t_range"]
    sections.append(
        f"Message schedule  (t = {m_lo}..{m_hi}, "
        f"{spec['msched_iters_per_block']} iter/block):"
    )
    sections.append(_format_folds(folds, sigma_full, pc_freq))
    if n_msched:
        sections.append(
            f"  → observed trace covers {n_msched} block(s) "
            f"(execution count / {spec['msched_iters_per_block']})"
        )
    sections.append("")

    c_lo, c_hi = spec["compress_t_range"]
    sections.append(
        f"Compression rounds  (t = {c_lo}..{c_hi}, "
        f"{spec['compress_iters_per_block']} iter/block):"
    )
    sections.append(_format_folds(folds, Sigma_full, pc_freq))
    if n_compress:
        sections.append(f"  → observed trace covers {n_compress} block(s)")
    sections.append("")

    sections.append("Boolean round-function idioms:")
    sections.append(
        f"  Ch  sites: {len(ch_hits)}"
        + ("" if not ch_hits else "   (`subject LIKE 'ch@%'` in findings.sqlite)")
    )
    sections.append(
        f"  Maj sites: {len(maj_hits)}"
        + ("" if not maj_hits else "   (`subject LIKE 'maj@%'` in findings.sqlite)")
    )
    sections.append("")

    sections.append("Algorithm body:")
    sections.append(spec["pseudocode"])
    sections.append("")

    notes: list[str] = []
    if not ch_hits:
        notes.append(
            "Ch idiom not captured by utov matcher "
            "(no `ch@*` handler_semantic findings; BR-8 #2 matcher may have "
            "missed the compiler-chosen shape — manual `inject_finding` works)."
        )
    if not maj_hits:
        notes.append(
            "Maj idiom not captured by utov matcher "
            "(no `maj@*` handler_semantic findings; see BR-8 #2)."
        )
    if not ks:
        notes.append(
            f"No K-table fingerprints (engine catalog may not stock "
            f"{prefix}.K[*]; for {algo} the algorithm requires "
            f"{spec['k_count']} K constants)."
        )
    n_inj = sum(1 for f in folds if f["source"] == "agent_override")
    if n_inj:
        notes.append(
            f"{n_inj} fold_idiom finding(s) were injected via agent_override; "
            f"see `list_interventions` for the audit trail."
        )
    if notes:
        sections.append("Notes:")
        for n in notes:
            sections.append(f"  - {n}")
        sections.append("")

    body = "\n".join(sections)
    if fmt == "markdown":
        body = (
            f"# {algo} reconstruction — {target_name}\n\n"
            f"```\n{body}\n```\n"
        )

    if out is not None:
        out.write(body)
        if not body.endswith("\n"):
            out.write("\n")
        try:
            out.flush()
        except Exception:
            pass
    return body


def emit_to_run_dir(run_dir: str | Path, *, filename: str = "pseudocode.md") -> Path | None:
    """Best-effort: write the rendered emit output to
    ``<run_dir>/<filename>`` in markdown form. Returns the written path
    on success, or None on any error (the caller is the preprocess_batch
    auto-emit path which never wants to raise).
    """
    try:
        rd = Path(run_dir).resolve()
        target = rd / filename
        rendered = emit(rd, out=None, fmt="markdown")
        target.write_text(rendered, encoding="utf-8")
        return target
    except Exception:
        return None
