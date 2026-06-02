"""capability_request.md §P1-4 — cluster cascade primitive tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

from engine.cluster import (
    collect_member_finding_ids,
    collect_member_origin_hyp_ids,
    make_cascade_id,
)
from engine.core import Core, CoreConfig
from engine.runner_client import NullRunnerAdapter
from engine.store import (
    link_finding_group_members,
    open_findings_db,
    open_hypotheses_db,
    upsert_payload,
    _now_iso,
)
from engine.types import Instruction, TargetMeta


def _build_core() -> Core:
    tm = TargetMeta(
        target_name="cluster-test", arch="arm64",
        algo_entry_pc=0x100, algo_exit_pc=0x200,
        input_length=None, output_length=4,
    )
    cfg = CoreConfig(
        work_root=Path(tempfile.mkdtemp(prefix="utov-test-cluster-")),
        target_meta=tm, input_hash="h", driver_mode="script", new_run=True,
    )

    class _R:
        def __init__(self): pass
        def __iter__(self):
            return iter([Instruction(idx=0, pc=0x100, bytes_=b"\x00\x00\x00\x00",
                                     mnemonic="", regs_read={}, regs_write={}, mem=())])

    return Core(cfg, _R(), NullRunnerAdapter(tm), skip_conformance=True)


def _seed_hyp(core: Core, kind: str, subject: str, *, status: str = "passed") -> int:
    """Insert a hypothesis row directly with the given status."""
    conn = open_hypotheses_db(core.work)
    try:
        payload_ref = upsert_payload(conn, {"k": kind, "s": subject})
        cur = conn.execute(
            "INSERT INTO claim_templates(kind, source, payload_ref, template_hash, "
            "confidence, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (kind, "test", payload_ref, f"th-{subject}-{_now_iso()}",
             0.9, _now_iso()),
        )
        template_id = cur.lastrowid
        cur = conn.execute(
            "INSERT INTO hypotheses(template_id, parent_id, depth, status, "
            "subject, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (template_id, None, 0, status, subject, _now_iso()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def _seed_finding(core: Core, kind: str, subject: str, origin_hyp_id: int | None) -> int:
    conn = open_findings_db(core.work)
    try:
        ref = upsert_payload(conn, {"kind": kind, "subject": subject})
        cur = conn.execute(
            "INSERT INTO findings(stage, kind, subject, payload_ref, verified_at,"
            " verifier_strategy, origin_hyp_id, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("s5-test", kind, subject, ref, _now_iso(), "test",
             origin_hyp_id, "s5_deterministic"),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def test_make_cascade_id_has_recognisable_prefix():
    cid = make_cascade_id()
    assert cid.startswith("cluster_cascade:")
    assert len(cid.split(":")[1]) == 8


def test_collect_member_finding_ids_empty_for_unknown_parent():
    core = _build_core()
    fc = open_findings_db(core.work)
    try:
        assert collect_member_finding_ids(fc, 9999) == []
    finally:
        fc.close()


def test_invalidate_cluster_flips_parent_and_members():
    core = _build_core()

    # Parent finding (with origin hyp at status=passed)
    parent_hyp = _seed_hyp(core, "fold_idiom", "parent-fold", status="passed")
    parent_fid = _seed_finding(core, "fold_idiom", "parent-fold", parent_hyp)

    # Three member findings, each with its own origin hyp at status=passed
    m_hyps  = [_seed_hyp(core, "handler_semantic", f"m{i}", status="passed") for i in range(3)]
    m_fids  = [_seed_finding(core, "handler_semantic", f"m{i}", m_hyps[i]) for i in range(3)]

    # Link members under parent
    fc = open_findings_db(core.work)
    try:
        link_finding_group_members(
            fc,
            parent_finding_id=parent_fid,
            idiom_name="strb-cluster",
            members=[(m_fids[i], f"role{i}") for i in range(3)],
        )
    finally:
        fc.close()

    res = core.invalidate_cluster(parent_fid, reason="constant-buffer trap detected")

    # Sanity on the report
    assert res["parent_finding_id"] == parent_fid
    assert res["parent_hyp_id"] == parent_hyp
    assert set(res["member_hyp_ids"]) == set(m_hyps)
    assert res["skipped_member_finding_ids"] == []
    assert res["parent_was_already_failed"] is False
    assert res["cascade_id"].startswith("cluster_cascade:")

    # Parent hyp now failed; members back to pending
    parent_snap = core._hyp_snapshot(parent_hyp)
    assert parent_snap is not None
    assert parent_snap["status"] == "failed"
    for h in m_hyps:
        snap = core._hyp_snapshot(h)
        assert snap is not None
        assert snap["status"] == "pending"


def test_invalidate_cluster_records_skipped_members_without_origin():
    """A member finding with origin_hyp_id=NULL can't flip — but the
    cascade must record it for the audit trail."""
    core = _build_core()
    parent_hyp = _seed_hyp(core, "fold_idiom", "parent-x", status="passed")
    parent_fid = _seed_finding(core, "fold_idiom", "parent-x", parent_hyp)
    orphan_fid = _seed_finding(core, "handler_semantic", "orphan", None)

    fc = open_findings_db(core.work)
    try:
        link_finding_group_members(
            fc,
            parent_finding_id=parent_fid,
            idiom_name="x",
            members=[(orphan_fid, None)],
        )
    finally:
        fc.close()

    res = core.invalidate_cluster(parent_fid, reason="test")
    assert res["member_hyp_ids"] == []
    assert res["skipped_member_finding_ids"] == [orphan_fid]


def test_collect_member_origin_hyp_ids_deduplicates():
    """A hyp linked twice (e.g. once per role) shows up once in the cascade."""
    core = _build_core()
    parent_hyp = _seed_hyp(core, "fold_idiom", "p", status="passed")
    parent_fid = _seed_finding(core, "fold_idiom", "p", parent_hyp)
    shared_hyp = _seed_hyp(core, "handler_semantic", "shared", status="passed")
    f1 = _seed_finding(core, "handler_semantic", "m1", shared_hyp)
    f2 = _seed_finding(core, "handler_semantic", "m2", shared_hyp)

    fc = open_findings_db(core.work)
    try:
        link_finding_group_members(
            fc, parent_finding_id=parent_fid, idiom_name="x",
            members=[(f1, "rotr_7"), (f2, "shr_3")],
        )
        hyps, skipped = collect_member_origin_hyp_ids(fc, parent_fid)
    finally:
        fc.close()

    assert hyps == [shared_hyp]
    assert skipped == []
