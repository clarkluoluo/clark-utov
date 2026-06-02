"""Adapter insulation: utov goes through engine.integration.adapter, which
returns utov-owned dataclasses, never raw Hypotask dicts. If Hypotask renames a
return key, only adapter.py breaks — these tests pin the adapter's contract.

Skips cleanly when clark-hypotask is absent.
"""

from __future__ import annotations

import pytest

pytest.importorskip("hypotask")

from engine.integration import (  # noqa: E402
    ClaimResult,
    FindingResult,
    TaskStateView,
    active_findings,
    bootstrap_hypotask,
    claim_done,
    make_test_session,
    store_finding,
    task_state,
)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "adapter.db")


def test_store_finding_returns_utov_dataclass(db_path):
    bootstrap_hypotask(db_path)
    with make_test_session(db_path) as s:
        tid = s.create_task(
            {
                "goal": "g",
                "profile_domain": "vmp_algorithm_extraction",
                "nodes": [{"name": "n1", "deps": []}],
                "done_criterion": {"all_of": [{"all_nodes_closed": True}]},
            }
        )
        st = task_state(s, tid)
        assert isinstance(st, TaskStateView)
        assert st.task_id == tid
        nid = st.nodes[0].node_id

        fr = store_finding(
            s,
            node_id=nid,
            content="utov-judged",
            evidence_source={"kind": "dataflow_provenance"},
            evidence_class="A",
            scope={"cross_env": True},
        )
        assert isinstance(fr, FindingResult)
        assert fr.finding_id
        # store-only: utov's verdict kept verbatim through the adapter
        assert fr.evidence_class == "A"
        assert fr.scope.get("cross_env") is True

        rows = active_findings(s, task_id=tid)
        assert len(rows) == 1


def test_claim_done_returns_utov_dataclass(db_path):
    bootstrap_hypotask(db_path)
    with make_test_session(db_path, parity_fn=lambda t, c: (True, {})) as s:
        tid = s.create_task(
            {
                "goal": "parity",
                "profile_domain": "vmp_algorithm_extraction",
                "nodes": [{"name": "m", "deps": []}],
                "done_criterion": {
                    "all_of": [{"cross_node_parity": {"name": "end_to_end_bytewise"}}]
                },
            }
        )
        res = claim_done(s, tid)
        assert isinstance(res, ClaimResult)
        assert res.ok is True
