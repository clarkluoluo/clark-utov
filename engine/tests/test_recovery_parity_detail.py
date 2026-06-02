"""#1 — parity floor observability: parity_detail + numbers in the reason.

A parity-blocked drive used to surface a flat "insufficient independent cross-run
vectors" with every number dropped. These pins fix that the ELIMINATED disposition
now carries the floor numbers so an agent can tell, at a glance:

  * supplied < need  → a FEED pit (the cohort vectors never reached parity)
  * supplied >= need but matched < need → the emitted F is genuinely wrong

Synthetic DriveResults only (no case data); the EXACT path is byte-for-byte the
same (invariant 7): a sufficient parity_report is not the parity disposition, so no
parity_detail is attached.
"""

from __future__ import annotations

from engine.cvd_recovery import _classify_drive_result, _drive_evidence
from engine.setup_symex import DriveResult


def _parity_report(*, need, supplied, matched, vectors, sufficient):
    return {
        "min_vectors": need,
        "total": supplied,
        "independent_pass": matched,
        "vectors": vectors,
        "sufficient": sufficient,
        "verdict": "EXACT" if sufficient else "BLOCK",
    }


def _result(parity_report, *, emitted_F="def f(x): return x"):
    return DriveResult(
        closed=False, mode="encoded", parity="1/3", emitted_F=emitted_F,
        backing_ok=True, address_closure={}, mem_backing={}, per_step=(),
        entry_keys=(), view_path=None, checkpoints={},
        parity_report=parity_report)


def test_parity_reason_carries_numbers_feed_pit():
    # supplied < need: a 1/1 tautological vector → the cohort feed never arrived.
    pr = _parity_report(need=3, supplied=1, matched=0,
                        vectors=[{"input_key": "gold-0", "observed": "1",
                                  "predicted": "1", "matches": True}],
                        sufficient=False)
    disposition, reason = _classify_drive_result(_result(pr))
    assert disposition == "parity"
    assert "matched 0/supplied 1" in reason and "need >= 3" in reason


def test_parity_reason_carries_numbers_f_wrong():
    # supplied >= need but matched < need: the emitted F is wrong on real vectors.
    pr = _parity_report(need=3, supplied=3, matched=1,
                        vectors=[{"input_key": "A", "observed": "Y",
                                  "predicted": "Y", "matches": True},
                                 {"input_key": "B", "observed": "Y",
                                  "predicted": "N", "matches": False},
                                 {"input_key": "C", "observed": "Y",
                                  "predicted": "N", "matches": False}],
                        sufficient=False)
    disposition, reason = _classify_drive_result(_result(pr))
    assert disposition == "parity"
    assert "matched 1/supplied 3" in reason and "need >= 3" in reason


def _parity_detail_for(pr):
    """Mirror the verifier's parity-branch evidence assembly (cvd_recovery.verify)."""
    res = _result(pr)
    disposition, _ = _classify_drive_result(res)
    ev = _drive_evidence(res, (0x1000, 0x10FF), "pc", disposition=disposition)
    assert disposition == "parity" and res.parity_report is not None
    ev = dict(ev)
    ev["parity_detail"] = {
        "need":     res.parity_report.get("min_vectors"),
        "supplied": res.parity_report.get("total"),
        "matched":  res.parity_report.get("independent_pass"),
        "vectors":  res.parity_report.get("vectors"),
    }
    return ev


def test_parity_detail_feed_pit_shows_supplied_below_need():
    pr = _parity_report(need=3, supplied=1, matched=0,
                        vectors=[{"input_key": "gold-0", "observed": "1",
                                  "predicted": "1", "matches": True}],
                        sufficient=False)
    pd = _parity_detail_for(pr)["parity_detail"]
    assert pd["supplied"] == 1 and pd["need"] == 3      # feed pit signal
    assert pd["matched"] == 0


def test_parity_detail_f_wrong_shows_vectors():
    vectors = [{"input_key": "A", "observed": "Y", "predicted": "Y", "matches": True},
               {"input_key": "B", "observed": "Y", "predicted": "N", "matches": False},
               {"input_key": "C", "observed": "Y", "predicted": "N", "matches": False}]
    pr = _parity_report(need=3, supplied=3, matched=1, vectors=vectors,
                        sufficient=False)
    pd = _parity_detail_for(pr)["parity_detail"]
    assert pd["supplied"] == 3 and pd["matched"] == 1   # F-wrong signal
    assert pd["vectors"] == vectors


def test_exact_parity_path_has_no_parity_detail():
    # Invariant 7: a sufficient parity_report is NOT the parity disposition. An
    # emitted+closed drive never reaches _classify_drive_result; here a sufficient
    # report does not classify as "parity", so no parity_detail is attached.
    pr = _parity_report(need=3, supplied=3, matched=3,
                        vectors=[{"input_key": k, "observed": "Y",
                                  "predicted": "Y", "matches": True}
                                 for k in ("A", "B", "C")],
                        sufficient=True)
    disposition, _ = _classify_drive_result(_result(pr))
    assert disposition != "parity"
    ev = _drive_evidence(_result(pr), (0x1000, 0x10FF), "pc",
                         disposition=disposition)
    assert "parity_detail" not in ev
