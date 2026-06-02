"""Integration: Hypotask used as store / link / query only — utov judges.

Architecture (locked): utov is the sole adjudicator; Hypotask is a ledger +
topology + checklist that records what utov submits and reports "is it present /
where are we". It must NOT judge correctness.

These tests exercise that contract from the utov side:
  - store a utov-decided finding and read it back,
  - finding→node→task linking + ledger trail,
  - done_criterion as a *structural checklist* (are the listed items present),
  - cross-node parity hook whose verdict is a utov callable.

They also pin the one place Hypotask STILL judges on the write path (it rewrites
a submitted evidence_class / scope) as a known overreach — see
hypotask_overreach_log.md #1. That assertion documents current reality so a
future Hypotask "store-only mode" flips it intentionally, not silently.

Skips cleanly when clark-hypotask is absent.
"""

from __future__ import annotations

import pytest

pytest.importorskip("hypotask")

from hypotask.interface import (  # noqa: E402
    get_active_findings,
    get_ledger_trail,
    get_task_state,
    write_finding,
)

from engine.integration.bootstrap import (  # noqa: E402
    bootstrap_hypotask,
    make_test_session,
)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "store.db")


def _one_node_task(session, *, done=None):
    tid = session.create_task(
        {
            "goal": "g",
            "profile_domain": "vmp_algorithm_extraction",
            "nodes": [{"name": "n1", "deps": []}],
            "done_criterion": done or {"all_of": [{"all_nodes_closed": True}]},
        }
    )
    nid = get_task_state(session, tid)["nodes"][0]["node_id"]
    return tid, nid


def test_store_and_read_back_a_finding(db_path):
    """A utov-decided finding is stored and queryable (store + query)."""
    bootstrap_hypotask(db_path)
    with make_test_session(db_path) as s:
        tid, nid = _one_node_task(s)
        r = write_finding(
            s,
            node_id=nid,
            content="utov-judged finding",
            evidence_source={"kind": "dataflow_provenance", "value_name": "x"},
            claimed_evidence_class="B",
            claimed_scope={"task_bound": True},
        )
        assert r["finding_id"]
        found = get_active_findings(s, task_id=tid)
        assert len(found) == 1
        assert found[0]["content"] == "utov-judged finding"


def test_finding_links_to_node_and_task(db_path):
    """finding→node→task linking is queryable (link)."""
    bootstrap_hypotask(db_path)
    with make_test_session(db_path) as s:
        tid, nid = _one_node_task(s)
        write_finding(
            s, node_id=nid, content="f",
            evidence_source={"kind": "dataflow_provenance"},
            claimed_evidence_class="B", claimed_scope={"task_bound": True},
        )
        st = get_task_state(s, tid)
        assert st["task"]["task_id"] == tid
        assert any(n["node_id"] == nid for n in st["nodes"])
        # finding reachable via the task it belongs to
        assert len(get_active_findings(s, task_id=tid)) == 1


def test_ledger_records_the_write(db_path):
    """Every write is auto-trailed in the ledger (store + audit)."""
    bootstrap_hypotask(db_path)
    with make_test_session(db_path) as s:
        tid, nid = _one_node_task(s)
        r = write_finding(
            s, node_id=nid, content="f",
            evidence_source={"kind": "dataflow_provenance"},
            claimed_evidence_class="B", claimed_scope={"task_bound": True},
        )
        trail = get_ledger_trail(s, entity_id=r["finding_id"])
        assert any(t["op_type"] == "write_finding" for t in trail)


def test_cross_node_parity_hook_judged_by_utov_callable(db_path):
    """The done checklist asks the named parity hook; utov's callable decides.

    Hypotask only asks "did end_to_end_bytewise report present/ok"; the answer is
    utov's parity_fn. Flip the fn → the checklist result flips. Hypotask judges
    nothing about parity itself.
    """
    from hypotask.interface import claim_task_done

    bootstrap_hypotask(db_path)
    parity = {"ok": False}

    with make_test_session(db_path, parity_fn=lambda t, c: (parity["ok"], {})) as s:
        tid = s.create_task(
            {
                "goal": "parity-gated",
                "profile_domain": "vmp_algorithm_extraction",
                "nodes": [{"name": "merge", "deps": []}],
                "done_criterion": {
                    "all_of": [{"cross_node_parity": {"name": "end_to_end_bytewise"}}]
                },
            }
        )
        # utov says parity not yet → checklist item absent → not done
        assert not claim_task_done(s, tid)["ok"]
        # utov says parity holds → checklist item present → done
        parity["ok"] = True
        assert claim_task_done(s, tid)["ok"]


def test_store_only_keeps_utov_verdict_verbatim(db_path):
    """Store-only (default guarded=False): write_finding stores utov's verdict
    verbatim — no rewrite of evidence_class / scope.

    Was the KNOWN-OVERREACH pin (hypotask_overreach_log.md #1): write_finding's
    verifier used to rewrite a submitted 'A' to 'B' and drop cross_env even with
    no probes registered. Hypotask 18bd21c made the write path store-only by
    default; utov submits a verdict, Hypotask records it as-is. utov can now
    treat the stored evidence_class as exactly what it submitted.
    """
    bootstrap_hypotask(db_path)
    with make_test_session(db_path) as s:
        tid, nid = _one_node_task(s)
        r = write_finding(
            s,
            node_id=nid,
            content="utov judged A",
            evidence_source={"kind": "observation", "value_name": "x"},
            claimed_evidence_class="A",
            claimed_scope={"cross_env": True},
        )
        v = r["verifier_verdict"]
        # store-only: submitted A stays A; cross_env preserved.
        assert v["evidence_class"] == "A", "store-only regressed? recheck overreach log #1"
        assert v["scope"].get("cross_env") is True, "store-only regressed? recheck overreach log #1"
        # and it reads back verbatim
        found = get_active_findings(s, task_id=tid)
        assert found[0]["evidence_class"] == "A"
