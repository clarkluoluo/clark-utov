"""end_to_end_bytewise cross-node parity probe.

This is the task-level (cross-node) check that the *whole* recovered algorithm
reproduces the runner byte-for-byte over covering inputs — the exact gate
the reference case missed when an agent declared a single node closed and called the
whole task done.

Hypotask's `TaskGate` evaluates a `{"cross_node_parity": {"name": ...}}`
done_criterion atom by looking up a probe registered under that name and calling
`probe(task_id, cfg) -> bool`. v0.1.1 has no *public* registration API, so
bootstrap registers this via the private `Session._task_gate` (see bootstrap.py;
flagged for Hypotask v0.2.0).

The probe itself is domain-neutral plumbing: it delegates the actual byte
comparison to a `parity_fn` supplied by the caller. In real reference-case use,
`parity_fn` drives the recovered implementation + the runner via the utov engine
and compares only runner-ok lines (edge_case_no_env_penalty). For tests, a
deterministic stub `parity_fn` is injected.
"""

from __future__ import annotations

from typing import Callable

# parity_fn(task_id, cfg) -> (ok, detail). detail is free-form for ledger/debug.
ParityFn = Callable[[str, dict], "tuple[bool, dict]"]


def make_end_to_end_parity_probe(parity_fn: ParityFn) -> Callable[[str, dict], bool]:
    """Build a cross-node probe from a parity comparison function.

    The returned callable matches Hypotask's `register_cross_node_probe`
    contract: `probe(task_id, cfg) -> bool`. The `cfg` is the body of the
    `cross_node_parity` done_criterion atom (e.g. `{"name": "end_to_end_bytewise"}`),
    forwarded to `parity_fn` so domains can carry extra knobs there.
    """

    def _probe(task_id: str, cfg: dict) -> bool:
        ok, _detail = parity_fn(task_id, cfg)
        return bool(ok)

    return _probe
