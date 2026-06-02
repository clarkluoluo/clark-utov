"""Bootstrap: wire utov into a Hypotask db for STORAGE / LINKING / CHECKLIST only.

Architecture (locked): **utov judges, Hypotask stores.**
  - utov is the sole adjudicator. Every real verdict — what counts as closed,
    evidence_class, scope, source provenance, whether the task can be done — is
    decided in utov's own engine (system A + RE kernels), tuned against VMP and
    user-adjustable. utov also enforces its own bottom lines.
  - Hypotask is a ledger + topology + checklist. It stores what utov submits,
    links finding→node→task, and reports "is the thing present / where are we".
    It does NOT judge correctness. utov sends a verdict; Hypotask records it.

So this bootstrap uses ONLY Hypotask's store/link/query surface:
  - load_domain_profile: stores the domain's *vocabulary* (node_state names,
    scope class names, done_criterion atom vocab) for task specs to reference.
    This is data storage, not judgement.
  - create_task / write_finding / get_task_state / get_ledger_trail: store and
    query.
  - register_cross_node_probe: registers a *checklist hook* whose actual
    pass/fail judgement is a utov-supplied callable (parity_fn). Hypotask only
    asks "did the named check report present"; utov's fn decides the answer.

It does NOT register Hypotask's builtin mechanism probes or verifier plugins —
those make Hypotask judge (5-way classify / cap / scope gate / Triton verdict),
which is exactly the role utov keeps. See hypotask_overreach_log.md for the
points where Hypotask still tries to judge on the write path.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from hypotask.interface import open_session
from hypotask.interface.session import Role, Session

from .parity_probe import ParityFn, make_end_to_end_parity_probe

DOMAIN_NAME = "vmp_algorithm_extraction"
DOMAIN_YAML = Path(__file__).parent / "vmp_algorithm_extraction.yaml"

# Probe name referenced by a task's done_criterion `cross_node_parity` atom.
PARITY_PROBE_NAME = "end_to_end_bytewise"


def bootstrap_hypotask(
    db_path: str | os.PathLike,
    *,
    domain_yaml: str | os.PathLike = DOMAIN_YAML,
    domain_name: str = DOMAIN_NAME,
) -> None:
    """Startup: dev stores the domain profile (vocabulary) into the db. Once.

    Loads only the domain's *vocabulary* — node_state names, scope class names,
    the atoms a task's done_criterion may reference. This is storage of data
    utov defines; Hypotask does not judge with it. utov's own engine decides
    what those states/scopes actually mean and when they're reached.

    Idempotent. NOTE: we deliberately do NOT call register_builtin_probes() —
    that would make Hypotask judge evidence_class / scope / source provenance,
    which is utov's job. utov submits already-judged verdicts via write_finding.
    """
    with open_session(db_path, role=Role.DEV_AGENT) as dev:
        dev.load_domain_profile(domain_yaml, name=domain_name)


@contextmanager
def make_dev_session(db_path: str | os.PathLike) -> Iterator[Session]:
    """Dev-role session (profile/vocabulary storage ops)."""
    with open_session(db_path, role=Role.DEV_AGENT) as s:
        yield s


@contextmanager
def make_test_session(
    db_path: str | os.PathLike,
    *,
    parity_fn: ParityFn | None = None,
) -> Iterator[Session]:
    """Business session; the utov business layer drives task work through it.

    A genuine TEST_AGENT session used purely for store/link/query. utov decides
    every verdict before calling write_finding; this session just records it.

    If ``parity_fn`` is given, registers the end_to_end_bytewise cross-node
    checklist hook. The hook is a checklist entry ("did end-to-end parity get
    reported"); its actual judgement is utov's ``parity_fn`` — Hypotask does not
    decide parity, it asks utov's callable and records the answer.
    """
    with open_session(db_path, role=Role.TEST_AGENT) as s:
        if parity_fn is not None:
            probe = make_end_to_end_parity_probe(parity_fn)
            s.register_cross_node_probe(PARITY_PROBE_NAME, probe)
        yield s
