"""FEATURE-REQUEST-1 — Tier 1 emitter regression tests.

Pins:

  - happy path (SHA-256 fixture): emit renders all expected sections
    (header, constants, message-schedule, compression, body, notes).
  - graceful failure path: no algorithm_identified row → EmitterError.
  - unsupported algorithm: synthesise a 'MD5' fit → EmitterError mentioning
    the supported list.
  - auto-emit: `preprocess_batch` writes `<run_dir>/pseudocode.md` only
    when at least one algorithm_identified is in the result.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from engine.core import Core, CoreConfig
from engine.emitter import EmitterError, emit
from engine.runner_client import NullRunnerAdapter
from engine.store import _now_iso, open_findings_db, upsert_payload
from engine.types import Instruction, TargetMeta


def _build_core(instrs) -> Core:
    tm = TargetMeta(
        target_name="emitter-test", arch="arm64",
        algo_entry_pc=instrs[0].pc, algo_exit_pc=instrs[-1].pc,
        input_length=None, output_length=32,
    )
    work_root = Path(tempfile.mkdtemp(prefix="utov-test-emitter-"))
    cfg = CoreConfig(
        work_root=work_root, target_meta=tm, input_hash="testhash",
        driver_mode="script", new_run=True,
    )

    class _R:
        def __init__(self, xs): self.xs = xs
        def __iter__(self): return iter(self.xs)

    return Core(cfg, _R(instrs), NullRunnerAdapter(tm), skip_conformance=True)


def _seed_sha256_findings(core: Core, *, anchors_seen: list[str]):
    """Insert one algorithm_identified + the matching fold_idiom + 8 IV
    `algo_signature` rows so the emitter has enough to render."""
    f_conn = open_findings_db(core.work)
    try:
        fit_ref = upsert_payload(f_conn, {
            "algorithm":        "SHA-256",
            "anchors_seen":     anchors_seen,
            "anchors_expected": [
                "SHA256.Sigma0", "SHA256.Sigma1",
                "SHA256.sigma0", "SHA256.sigma1",
                "SHA256.h0", "SHA256.h1", "SHA256.h2", "SHA256.h3",
                "SHA256.h4", "SHA256.h5", "SHA256.h6", "SHA256.h7",
            ],
            "evidence_score":   round(len(anchors_seen) / 12, 3),
            "rationale":        "test fixture",
        })
        f_conn.execute(
            "INSERT INTO findings(stage, kind, subject, payload_ref, "
            "verified_at, verifier_strategy, origin_hyp_id, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("s5-algorithm-fit", "algorithm_identified", "SHA-256", fit_ref,
             _now_iso(), "structural_anchor_set_match", None,
             "s5_algorithm_fit"),
        )
        # fold_idiom rows — one per σ/Σ idiom mentioned in anchors_seen.
        for subj in anchors_seen:
            if subj.startswith("SHA256.") and not subj.startswith("SHA256.h"):
                fold_ref = upsert_payload(f_conn, {
                    "idiom":      subj,
                    "input_reg":  "w21",
                    "dst_reg":    "w22",
                    "anchor_pcs": [0x12000b00, 0x12000b08, 0x12000b0c],
                    "components": [
                        {"kind": "ror", "amount": 6, "pc": "0x12000b00"},
                        {"kind": "ror", "amount": 11, "pc": "0x12000b08"},
                        {"kind": "ror", "amount": 25, "pc": "0x12000b0c"},
                    ],
                })
                f_conn.execute(
                    "INSERT INTO findings(stage, kind, subject, payload_ref, "
                    "verified_at, verifier_strategy, origin_hyp_id, source) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    ("s5-fold", "fold_idiom", f"{subj}@0x12000b0c", fold_ref,
                     _now_iso(), "algebraic_idiom_match", None,
                     "s5_fold_idiom"),
                )
        # h0..h7 plugin fingerprints.
        for i in range(8):
            iv_ref = upsert_payload(f_conn, {
                "magic":       f"0x{0x6a09e667 + i * 0x10000:08x}",
                "fingerprint": f"SHA256.h{i}",
            })
            f_conn.execute(
                "INSERT INTO findings(stage, kind, subject, payload_ref, "
                "verified_at, verifier_strategy, origin_hyp_id, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("s1b-verify", "algo_signature", f"SHA256.h{i}", iv_ref,
                 _now_iso(), "handler_semantic", None, "plugin"),
            )
        f_conn.commit()
    finally:
        f_conn.close()


def test_emit_sha256_happy_path_contains_all_sections():
    nop = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([nop])
    _seed_sha256_findings(core, anchors_seen=[
        "SHA256.Sigma0", "SHA256.Sigma1",
        "SHA256.sigma0", "SHA256.sigma1",
        "SHA256.h0", "SHA256.h1", "SHA256.h2", "SHA256.h3",
        "SHA256.h4", "SHA256.h5", "SHA256.h6", "SHA256.h7",
    ])
    text = core.emit_pseudocode()
    # Header.
    assert "SHA-256 reconstruction" in text
    assert "evidence_score: 1.0" in text
    assert "anchors: 12/12" in text
    # Constants section + each h-row rendered.
    assert "Initial hash values (H[0..7])" in text
    for i in range(8):
        assert f"SHA256.h{i}" in text
    # Idiom sections.
    assert "Message schedule" in text
    assert "Compression rounds" in text
    assert "SHA256.sigma0" in text
    assert "SHA256.Sigma1" in text
    # Pseudocode body.
    assert "// SHA-256 (FIPS 180-4)" in text
    assert "for t in 0..63" in text
    # Boolean idiom section (Ch / Maj counts).
    assert "Boolean round-function idioms" in text
    # No K hits → notes section mentions the gap.
    assert "Notes" in text
    assert "K-table fingerprints" in text


def test_emit_renders_local_closure_trap_banner():
    """Task 7④ — an algorithm_hyp carrying a LOCAL_CLOSURE_ONLY trap renders the
    trap LOUDLY at the top so it is never read as a final identification."""
    nop = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([nop])
    f_conn = open_findings_db(core.work)
    try:
        fit_ref = upsert_payload(f_conn, {
            "algorithm": "SHA-256",
            "anchors_seen": ["SHA256.h0"],
            "anchors_expected": ["SHA256.h0"],
            "evidence_score": 1.0,
            "rationale": "test fixture (HYPOTHESIS)",
            "closure": {
                "trap_state": "LOCAL_CLOSURE_ONLY",
                "closure_level": "structural",
                "algorithm_closed": False,
                "is_primitive": True,
                "next_step": "confirm sink + provenance + multi-input parity",
            },
        })
        f_conn.execute(
            "INSERT INTO findings(stage, kind, subject, payload_ref, "
            "verified_at, verifier_strategy, origin_hyp_id, source) "
            "VALUES (?,?,?,?,?,?,?,?)",
            ("s5-algorithm-fit", "algorithm_hyp", "SHA-256", fit_ref,
             _now_iso(), "structural_anchor_set_match", None, "s5_algorithm_fit"))
        iv_ref = upsert_payload(f_conn, {"magic": "0x6a09e667",
                                         "fingerprint": "SHA256.h0"})
        f_conn.execute(
            "INSERT INTO findings(stage, kind, subject, payload_ref, "
            "verified_at, verifier_strategy, origin_hyp_id, source) "
            "VALUES (?,?,?,?,?,?,?,?)",
            ("s1b-verify", "algo_signature", "SHA256.h0", iv_ref,
             _now_iso(), "handler_semantic", None, "plugin"))
        f_conn.commit()
    finally:
        f_conn.close()
    text = core.emit_pseudocode()
    assert "LOCAL_CLOSURE_ONLY" in text
    assert "HYPOTHESIS" in text
    assert "next:" in text


def test_emit_markdown_wraps_text_in_fenced_block():
    nop = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([nop])
    _seed_sha256_findings(core, anchors_seen=[
        "SHA256.Sigma0", "SHA256.h0", "SHA256.h1", "SHA256.h2", "SHA256.h3",
    ])
    md = core.emit_pseudocode(fmt="markdown")
    assert md.startswith("# SHA-256 reconstruction")
    assert "```" in md
    # Ensure plain text emit is unchanged (no fence).
    text = core.emit_pseudocode(fmt="text")
    assert not text.startswith("# SHA-256")


def test_emit_no_algorithm_identified_raises():
    """No `algorithm_identified` row at all → EmitterError. The CLI uses
    this to exit 2 with a one-line stderr message."""
    nop = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([nop])
    # Touch findings.sqlite so the emitter walks past its existence check
    # and hits the "no algorithm_identified" branch we want to pin.
    open_findings_db(core.work).close()
    with pytest.raises(EmitterError, match="no `algorithm_hyp` / `algorithm_identified`"):
        core.emit_pseudocode()


def test_emit_unsupported_algorithm_raises():
    """A fit row carrying an unknown algorithm label must surface a
    helpful error listing the supported templates."""
    nop = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([nop])
    f_conn = open_findings_db(core.work)
    try:
        ref = upsert_payload(f_conn, {
            "algorithm":        "MD5",
            "anchors_seen":     ["MD5.K[0]"],
            "anchors_expected": ["MD5.K[0]"],
            "evidence_score":   1.0,
        })
        f_conn.execute(
            "INSERT INTO findings(stage, kind, subject, payload_ref, "
            "verified_at, verifier_strategy, origin_hyp_id, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("s5-algorithm-fit", "algorithm_identified", "MD5", ref,
             _now_iso(), "structural_anchor_set_match", None,
             "s5_algorithm_fit"),
        )
        f_conn.commit()
    finally:
        f_conn.close()
    with pytest.raises(EmitterError, match="no emitter template for algorithm"):
        core.emit_pseudocode()


def test_auto_emit_runs_when_algorithm_identified_present():
    """After preprocess_batch promotes an algorithm_identified finding,
    a `<run_dir>/pseudocode.md` artefact must exist and the batch result
    must echo its path."""
    nop = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([nop])
    # Pre-seed enough findings that the `algorithm` pass promotes SHA-256.
    f_conn = open_findings_db(core.work)
    try:
        ch = upsert_payload(f_conn, {"seed": 1})
        for subj in ("SHA256.Sigma0@0x1100", "SHA256.Sigma1@0x1104",
                     "SHA256.sigma0@0x1108", "SHA256.sigma1@0x110c"):
            f_conn.execute(
                "INSERT INTO findings(stage, kind, subject, payload_ref, "
                "verified_at, verifier_strategy, origin_hyp_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("s5-fold", "fold_idiom", subj, ch, _now_iso(),
                 "algebraic_idiom_match", None),
            )
        for i in range(8):
            iv_ref = upsert_payload(f_conn, {
                "magic":       f"0x{0x6a09e667 + i * 0x10000:08x}",
                "fingerprint": f"SHA256.h{i}",
            })
            f_conn.execute(
                "INSERT INTO findings(stage, kind, subject, payload_ref, "
                "verified_at, verifier_strategy, origin_hyp_id, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("s1b-verify", "algo_signature", f"SHA256.h{i}", iv_ref,
                 _now_iso(), "handler_semantic", None, "plugin"),
            )
        f_conn.commit()
    finally:
        f_conn.close()

    # Only run the algorithm-fit pass; the others would also work but
    # would generate noise we don't care about for this test.
    result = core.preprocess_batch(passes=["algorithm"])
    assert "SHA-256" in result["totals"]["matched_algorithms"]
    assert result["pseudocode_path"] is not None
    md_path = Path(result["pseudocode_path"])
    assert md_path.exists()
    md = md_path.read_text(encoding="utf-8")
    assert "SHA-256 reconstruction" in md
    assert "```" in md   # markdown fence


def test_auto_emit_skipped_when_no_algorithm_identified():
    """preprocess_batch must NOT write pseudocode.md when no
    algorithm_identified was promoted (avoid stale / empty file)."""
    nop = Instruction(idx=0, pc=0x1000, bytes_=b"\x00" * 4,
                      mnemonic="ret", regs_read={}, regs_write={}, mem=())
    core = _build_core([nop])
    result = core.preprocess_batch(passes=["plugin", "binop"])
    assert result["totals"]["matched_algorithms"] == []
    assert result["pseudocode_path"] is None
    assert not (core.work.root / "pseudocode.md").exists()
