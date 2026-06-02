"""Execution bundle exporter — spec #5.

Turn ONE rerun's multi-point observations into a structured, stamped,
same-execution evidence bundle with **declarative** derived fields, so verifiers
stop hand-assembling field-grab / align / JSON code.

GENERAL-FIRST (A8②): the bundle schema and the ``derived`` mechanism are generic
— ``observation_spec`` is any named ``{name: ObservePoint}`` map and ``derived``
is any ``{field_name: extractor(bundle) -> value}`` map. TC2's
``time_seed / rand_words / src32 / sink32 / output`` are ONE instantiation of
that map, NOT a hardcoded schema.

A8①  Reuse, don't rebuild — every primitive is reused, never re-implemented:
  * ``adapter.rerun(input, observe_points)`` (runner_client.py:664) for the single
    execution + its multi-point observations;
  * ``mem_snapshots_from_rerun`` (runner_client.py:580) for mem-region extraction;
  * ``recapture_loop.assert_same_execution`` (recapture_loop.py:135) for the
    same-execution guard;
  * ``export_stamp.export_stamped_json`` (export_stamp.py:237) for the stamped
    ``write_json`` header.
The bundle is a *consolidation* — it adds no new capture / wire capability.

A8④  Degenerate → surfaced, never silent:
  * a derived extractor that raises / can't compute ⇒ that field is ``None`` +
    ``derived_errors[name] = reason`` (surfaced, never a dropped / forged field);
  * a same-execution violation ⇒ ``same_execution = False`` +
    ``same_execution_detail`` (loud), never silently stitched.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

from .export_stamp import export_stamped_json
from .recapture_loop import assert_same_execution
from .runner_client import (
    ObservePoint,
    ObservedState,
    RerunResult,
    RunnerAdapter,
    mem_snapshots_from_rerun,
)
from .store import _now_iso

# A derived extractor is a pure function of the (built) bundle.
DerivedExtractor = Callable[["CaptureBundle"], Any]

# The default declarative derived map: empty (a bundle with no derived fields is
# valid — the mechanism is additive / opt-in, A8③).
DERIVED_NONE: Mapping[str, DerivedExtractor] = {}

__all__ = [
    "CaptureBundle",
    "capture_bundle",
    "DerivedExtractor",
    "DERIVED_NONE",
]


def _obs_to_dict(obs: ObservedState) -> dict[str, Any]:
    """Render one ``ObservedState`` to the canonical JSON shape (hex regs / mem),
    matching ``phase.EntryState.to_dict`` — pure, no re-derivation."""
    return {
        "pc": f"0x{obs.pc:x}",
        "when": obs.when,
        "regs": {k: f"0x{v:x}" for k, v in (obs.regs or {}).items()},
        "mem": {f"0x{addr:x}": data.hex() for addr, data in (obs.mem or {}).items()},
    }


@dataclass
class CaptureBundle:
    """One execution's structured evidence (spec §Contract / §CaptureBundle).

    Built by :func:`capture_bundle` from a SINGLE ``adapter.rerun``: ``input`` →
    ``output`` plus the ``observations`` captured at the named observe points, all
    under one ``exec_identity`` stamp. ``derived`` holds the computed
    declarative-extractor values; ``derived_errors`` surfaces any extractor that
    could not compute (A8④). ``same_execution`` is the asserted same-execution
    flag; ``same_execution_detail`` carries the violation report when it is False.
    """

    exec_identity: dict[str, Any]
    input: bytes
    output: bytes
    # Named observations: {name: ObservedState}. Keyed by the observation_spec name
    # (NOT by pc) so a verifier reads a field by its MEANING, not by re-matching pc.
    observations: dict[str, ObservedState] = field(default_factory=dict)
    derived: dict[str, Any] = field(default_factory=dict)
    derived_errors: dict[str, str] = field(default_factory=dict)
    same_execution: bool = True
    same_execution_detail: dict[str, Any] | None = None
    # Advisory: the runner hit a record cap during this rerun (incomplete ledger).
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """The bundle as a JSON-ready dict (the schema in spec §CaptureBundle).

        ``input`` / ``output`` are hex; observations render via ``_obs_to_dict``;
        ``derived`` is passed through (extractors own their value shape).
        ``derived_errors`` / ``same_execution_detail`` / ``truncated`` are emitted
        only when meaningful — a clean bundle carries no degenerate noise, a
        degraded one always surfaces WHY (A8④)."""
        out: dict[str, Any] = {
            "exec_identity": dict(self.exec_identity),
            "input": self.input.hex(),
            "output": self.output.hex(),
            "observations": {
                name: _obs_to_dict(obs) for name, obs in self.observations.items()
            },
            "derived": dict(self.derived),
            "same_execution": self.same_execution,
        }
        if self.derived_errors:
            out["derived_errors"] = dict(self.derived_errors)
        if self.same_execution_detail is not None:
            out["same_execution_detail"] = dict(self.same_execution_detail)
        if self.truncated:
            out["truncated"] = True
        return out

    def write_json(
        self,
        path: str,
        *,
        source: str = "utov execution_bundle (capture_bundle)",
        exported_by: str = "engine.execution_bundle.capture_bundle",
        ts: str | None = None,
        from_entries: tuple[str, ...] = (),
    ) -> str:
        """Render the bundle as a STAMPED JSON document and write it to ``path``.

        Reuses :func:`engine.export_stamp.export_stamped_json` (A8①) so the file
        carries the authoritative ``<!-- utov-export ... -->`` header
        (``source / exported_by / exec_identity / ts / authority``) — the
        discriminator a hand-written file lacks. Returns the rendered text."""
        text = export_stamped_json(
            self.to_dict(),
            source=source,
            exported_by=exported_by,
            exec_identity=self.exec_identity,
            ts=ts or _now_iso(),
            from_entries=from_entries,
        )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return text


def _execution_token(
    result: RerunResult, exec_identity: Mapping[str, Any] | None,
) -> str:
    """The single same-execution token for THIS rerun's snapshots.

    One ``capture_bundle`` call = one ``adapter.rerun`` = one execution. The token
    folds the produced ``output`` (its nonce) so two distinct executions get
    distinct tokens — making any later cross-rerun stitch DETECTABLE by
    :func:`assert_same_execution`. Mirrors ``recapture_loop._execution_id``.
    Deterministic / pure."""
    nonce = bytes(result.output)[:16].hex()
    if exec_identity:
        case = exec_identity.get("case") or exec_identity.get("exec_id") or ""
        if case:
            return f"capture:{case}:{nonce}"
    return f"capture:{nonce}"


def capture_bundle(
    adapter: RunnerAdapter,
    input_bytes: bytes,
    observation_spec: Mapping[str, ObservePoint],
    *,
    derived: Mapping[str, DerivedExtractor] = DERIVED_NONE,
    exec_identity: Mapping[str, Any] | None = None,
) -> CaptureBundle:
    """Capture ONE execution's evidence into a :class:`CaptureBundle` (spec §Contract).

    A FREE FUNCTION over any :class:`RunnerAdapter` — it calls the adapter's public
    ``rerun`` only, never reaches into runner internals (boundary: pure engine, no
    runner / wire change).

    ``observation_spec`` — named observe points ``{name: ObservePoint}``. They are
    sent to ``adapter.rerun`` as one observe-point list; the returned observations
    are matched back to names by ``(pc, when)`` so the bundle is keyed by MEANING.

    ``derived`` — the DECLARATIVE map ``{field_name: extractor(bundle) -> value}``.
    Each extractor runs after capture, over the populated bundle. An extractor that
    raises ⇒ ``derived[name] = None`` + ``derived_errors[name] = reason`` (A8④ —
    surfaced, never dropped). The derived fields are pure functions of the bundle;
    TC2's ``time_seed / rand_words / src32 / sink32`` are ONE instantiation.

    ``exec_identity`` — the execution this bundle belongs to (carried into the
    ``write_json`` stamp). Defaults to a single-field identity derived from the
    output nonce when not supplied.

    Same-execution (A8①④): all observations come from ONE rerun → same-execution is
    true BY CONSTRUCTION. The backing snapshots are stamped with one token and run
    through :func:`assert_same_execution` as the loud 兜底 guard; a violation (only
    reachable if an adapter forged cross-rerun mem) sets ``same_execution = False``
    + ``same_execution_detail`` rather than silently stitching."""
    spec = dict(observation_spec)
    observe_points = list(spec.values())

    result = adapter.rerun(input_bytes, observe_points or None)

    # Match each returned ObservedState back to its spec name by (pc, when). A spec
    # point that produced no observation simply has no entry (never a fabricated
    # one — degenerate is absence, surfaced by the missing key, not a forged state).
    by_key: dict[tuple[int, str], ObservedState] = {
        (obs.pc, obs.when): obs for obs in result.observations
    }
    observations: dict[str, ObservedState] = {}
    for name, op in spec.items():
        obs = by_key.get((op.pc, op.when))
        if obs is not None:
            observations[name] = obs

    ident: dict[str, Any] = dict(exec_identity) if exec_identity else {}
    if not ident:
        ident = {"exec_id": _execution_token(result, None)}

    # Same-execution guard (A8①④): one rerun → one token; stamp the backing
    # snapshots and assert. assert_same_execution returns None when safe (the
    # normal path here) or a structured violation report.
    token = _execution_token(result, exec_identity)
    snaps = [
        s if s.execution_id is not None else replace(s, execution_id=token)
        for s in mem_snapshots_from_rerun(result)
    ]
    violation = assert_same_execution(snaps)
    same_execution = violation is None

    bundle = CaptureBundle(
        exec_identity=ident,
        input=bytes(input_bytes),
        output=bytes(result.output),
        observations=observations,
        same_execution=same_execution,
        same_execution_detail=violation,
        truncated=bool(result.truncated),
    )

    # Declarative derived fields (A8②④): pure functions of the bundle; a failing
    # extractor surfaces in derived_errors instead of dropping / forging the field.
    for name, extractor in dict(derived).items():
        try:
            bundle.derived[name] = extractor(bundle)
        except Exception as exc:  # noqa: BLE001 — any extractor failure must surface
            bundle.derived[name] = None
            bundle.derived_errors[name] = f"{type(exc).__name__}: {exc}"

    return bundle
