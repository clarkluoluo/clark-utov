"""Tests for the authority projection (B7, dev-authority-projection-spec).

Pin the contract: the projection surfaces the CURRENT authority face, demotes
(but keeps traceable) superseded claims, resolves supersede chains correctly,
WARNs on cycles without looping, sorts same-topic CURRENT claims by recency,
exports with the utov-export stamp, and never touches Hypotask's storage path.
"""

from __future__ import annotations

from engine.authority_projection import (
    claims_from_findings,
    export_authority_projection,
    project_authority,
)
from engine.export_stamp import is_utov_export, load_stamped_json


def _ids(claims):
    return {c["claim"] for c in claims}


def test_empty_claims_is_empty_projection_not_error():
    """No claim → empty projection + explicit status, not a raised error (A8)."""
    p = project_authority("CaseX", [])
    assert p["case"] == "CaseX"
    assert p["status"] == "EMPTY_NO_CLAIMS"
    assert p["authoritative_claims"] == []
    assert p["demoted_claims"] == []
    assert p["next_blocker"] is None


def test_bare_claim_is_current_by_default():
    """A claim with no explicit verdict is CURRENT (standing conclusion)."""
    p = project_authority("C", [{"claim": "a"}])
    assert _ids(p["authoritative_claims"]) == {"a"}
    assert p["authoritative_claims"][0]["verdict"] == "CURRENT"


def test_supersede_demotes_and_is_traceable():
    """A CURRENT claim supersedes another → the latter leaves the top surface but
    is kept in demoted_claims with superseded_by (验收: 降级且可追溯)."""
    claims = [
        {"claim": "stack65_as_final_target", "verdict": "CURRENT"},
        {"claim": "stack65_surface_captured_but_not_oracle_equivalent",
         "verdict": "CURRENT",
         "supersedes": ["stack65_as_final_target"]},
    ]
    p = project_authority("ReferenceCase.F0_A_cipher", claims)
    auth = _ids(p["authoritative_claims"])
    assert "stack65_surface_captured_but_not_oracle_equivalent" in auth
    # the old "stack65 = final target" conclusion is NOT on the top surface
    assert "stack65_as_final_target" not in auth
    demoted = {c["claim"]: c for c in p["demoted_claims"]}
    assert "stack65_as_final_target" in demoted
    assert demoted["stack65_as_final_target"]["superseded_by"] == [
        "stack65_surface_captured_but_not_oracle_equivalent"]


def test_supersede_chain_converges_c_does_not_revive():
    """A supersedes B, B supersedes C (all CURRENT) → only A authoritative; C does
    NOT revive because B was itself superseded (contract 2 closure)."""
    claims = [
        {"claim": "A", "verdict": "CURRENT", "supersedes": ["B"]},
        {"claim": "B", "verdict": "CURRENT", "supersedes": ["C"]},
        {"claim": "C", "verdict": "CURRENT"},
    ]
    p = project_authority("Chain", claims)
    assert _ids(p["authoritative_claims"]) == {"A"}
    demoted = {c["claim"]: c for c in p["demoted_claims"]}
    assert set(demoted) == {"B", "C"}
    # C is demoted (reached transitively through the CURRENT chain), not revived
    assert "A" in demoted["C"]["superseded_by"] or "B" in demoted["C"]["superseded_by"]


def test_demoted_superseder_does_not_keep_its_target_down_on_its_own():
    """If A supersedes B and B (now non-CURRENT) supersedes C, but no CURRENT
    claim supersedes C → C is NOT demoted by B alone (only CURRENT claims
    propagate supersession)."""
    claims = [
        {"claim": "A", "verdict": "CURRENT", "supersedes": ["B"]},
        {"claim": "B", "verdict": "SUPERSEDED", "supersedes": ["C"]},
        {"claim": "C", "verdict": "CURRENT"},
    ]
    p = project_authority("PartialChain", claims)
    auth = _ids(p["authoritative_claims"])
    # A is current top; C is still current (B is demoted and can't push C down)
    assert "A" in auth
    assert "C" in auth
    assert "B" not in auth


def test_multiple_current_no_supersede_all_listed_sorted_by_recency():
    """Same-topic CURRENT claims with no supersede relation → all listed, sorted
    by updated_at descending, nothing merged/truncated (contract 3)."""
    claims = [
        {"claim": "c_old", "verdict": "CURRENT", "updated_at": "2026-06-01T00:00:00Z"},
        {"claim": "c_new", "verdict": "CURRENT", "updated_at": "2026-06-02T00:00:00Z"},
        {"claim": "c_mid", "verdict": "CURRENT", "updated_at": "2026-06-01T12:00:00Z"},
    ]
    p = project_authority("Multi", claims)
    order = [c["claim"] for c in p["authoritative_claims"]]
    assert order == ["c_new", "c_mid", "c_old"]
    assert len(order) == 3  # none dropped


def test_cycle_warns_and_terminates():
    """A supersede cycle → explicit WARN entry, no infinite loop; cycle members
    are flagged and not treated as authoritative (contract 2)."""
    claims = [
        {"claim": "A", "verdict": "CURRENT", "supersedes": ["B"]},
        {"claim": "B", "verdict": "CURRENT", "supersedes": ["A"]},
    ]
    p = project_authority("Cyclic", claims)  # must return, not hang
    kinds = {w["kind"] for w in p.get("warnings", [])}
    assert "supersede_cycle" in kinds
    auth = _ids(p["authoritative_claims"])
    assert "A" not in auth and "B" not in auth
    demoted = {c["claim"]: c for c in p["demoted_claims"]}
    assert demoted["A"].get("in_supersede_cycle") is True
    assert demoted["B"].get("in_supersede_cycle") is True


def test_status_and_next_blocker_passthrough():
    """Caller-supplied status / next_blocker surface verbatim (the agent's
    'where do we stand' summary)."""
    p = project_authority(
        "C", [{"claim": "x", "verdict": "CURRENT"}],
        status="OPEN_BLOCKED_ON_GENERIC_REGREL_RECAPTURE",
        next_blocker="runner_generic_regrelative_recapture",
    )
    assert p["status"] == "OPEN_BLOCKED_ON_GENERIC_REGREL_RECAPTURE"
    assert p["next_blocker"] == "runner_generic_regrelative_recapture"


def test_status_field_alias_and_evidence_refs_preserved():
    """A claim may carry 'status' instead of 'verdict'; evidence_refs survive."""
    claims = [
        {"claim": "a", "status": "CURRENT", "evidence_refs": ["finding:123"]},
        {"claim": "b", "status": "RETIRED"},
    ]
    p = project_authority("C", claims)
    auth = {c["claim"]: c for c in p["authoritative_claims"]}
    assert auth["a"]["evidence_refs"] == ["finding:123"]
    demoted = {c["claim"]: c for c in p["demoted_claims"]}
    # RETIRED with no superseder → demoted purely by its own verdict, reason honest
    assert demoted["b"]["demoted_reason"] == "non_current_verdict"


def test_export_carries_utov_export_stamp():
    """The exported projection opens with the utov-export header (contract 4)."""
    p = project_authority(
        "ReferenceCase.F0_A_cipher",
        [{"claim": "x", "verdict": "CURRENT", "evidence_refs": []}],
        status="OPEN", next_blocker="recapture",
    )
    text = export_authority_projection(p)
    assert is_utov_export(text)
    header, body = load_stamped_json(text)
    assert header is not None
    assert header["exported_by"] == "engine.authority_projection.project_authority"
    # from_entries traces the claim ids the projection was built from
    assert "x" in header["from_entries"]
    assert body["case"] == "ReferenceCase.F0_A_cipher"
    assert body["status"] == "OPEN"


def test_claims_from_findings_adapter_maps_rows():
    """The read-only adapter maps Hypotask finding rows into claim dicts without
    touching storage — finding_id becomes the claim id when no explicit claim."""
    findings = [
        {"finding_id": "f1", "verdict": "CURRENT", "content": "..."},
        {"finding_id": "f2", "claim": "named_claim", "status": "CURRENT",
         "supersedes": ["f1"], "updated_at": "2026-06-02T00:00:00Z"},
    ]
    claims = claims_from_findings(findings)
    p = project_authority("FromFindings", claims)
    auth = _ids(p["authoritative_claims"])
    assert auth == {"named_claim"}  # f1 superseded by the named claim
    demoted = {c["claim"]: c for c in p["demoted_claims"]}
    assert demoted["f1"]["superseded_by"] == ["named_claim"]


def test_does_not_touch_hypotask_storage_path():
    """Regression: importing/using the projection never imports hypotask nor
    mutates any store. The projection core is source-agnostic (pure)."""
    import sys

    before = "hypotask" in sys.modules
    project_authority("C", [{"claim": "a", "verdict": "CURRENT"}])
    claims_from_findings([{"finding_id": "f1", "verdict": "CURRENT"}])
    after = "hypotask" in sys.modules
    # the projection itself must not pull hypotask in
    assert before == after
