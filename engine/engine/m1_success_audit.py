"""M1 success-audit framework gate.

Background — the reference target's retro: a manual ``success_audit.md`` round
caught a 94/94 pass rate where every sample shared an identical
prefix. The success claim depended on a ``prefix`` dimension that was
never varied; the 100% pass was a fixed-input artefact, not
generalisation. M1 was a *checklist* the agent could forget. This
module turns it into a framework-layer gate: any tool call that
declares ``target_success=true`` or ``archival_allowed=true`` is
intercepted and run through four automatic checks before the
underlying archival dispatch is allowed to proceed.

The four checks (mirrors of the manual audit):

  1. **dimension_coverage** — for each declared
     ``success_dependencies`` variable, count unique values across
     ``samples``. ``unique_count == 0/1`` means the dimension was
     *fixed*, not tested. Such dimensions go into
     ``untested_dimensions``.
  2. **overfit_check** — when ``pass_rate >= overfit_pass_rate_floor``
     (default 0.99) AND ``untested_dimensions`` is non-empty, raise
     ``overfit_flag``. 100% pass with a pinned dimension is an
     overfit signature, not a confirmation.
  3. **scope** — tag the success with one of
     ``full_input_space`` / ``cross_session`` / ``in_session`` /
     ``unknown``. Honour an explicit ``scope`` field; otherwise
     downgrade to the lowest scope the sample set supports.
  4. **closure_consistency** — when the report carries
     ``closure_paths`` (e.g. cfbc / formula / hook / sign digests),
     all path digests must be byte-equal. A single path leaves it
     ``None`` (not applicable); 2+ matching paths give ``True``;
     mismatch gives ``False`` (forces reject).

Grading:

  - **A** — no untested dimensions, scope in
    {full_input_space, cross_session}, closure either consistent or
    not-applicable, no overfit flag, samples >= ``min_samples`` →
    ``action='allow'``.
  - **B** — survives basic sanity (samples present, pass_rate above
    the reject floor, closure not contradicted) but at least one
    M1 axis is weak (a dimension is fixed, scope is in-session,
    closure not multi-path verified). Forces
    ``action='downgrade'`` and rewrites the params so the dispatched
    archival sees ``target_success=False``, ``archival_allowed=False``
    and a new ``evidence_class='B'`` /
    ``downgraded_to='strong_partial'`` annotation.
  - **C** — pass_rate below ``min_pass_rate``, missing samples, or
    closure paths disagree. ``action='reject'``; the discipline
    wrapper turns this into a JSON-RPC refusal so the archival never
    dispatches.

Independent toggle: ``UTOV_M1_AUDIT=off`` (or any of ``0`` / ``false``
/ ``no``) disables the module — :func:`audit_success_claim` returns
``None`` and the wrapper proceeds as if the module weren't there.
This keeps the gate debuggable in isolation without touching the
wider ``UTOV_METHODOLOGY`` toggle.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


# Methods whose params/result count as an *archival* action — these are
# the surfaces that get audited even when no explicit
# ``target_success`` field is present. (Aligned with M6's archival
# surface; the wrapper's existing `verdict_methods` is a subset.)
DEFAULT_ARCHIVAL_METHODS: tuple[str, ...] = (
    "promote_to_finding",
    "inject_finding",
    "override_verdict",
    "finalize_verdict",
    "submit_hypothesis",
    "archive_target",
)


# Subjective-input keys (v0.4.0 B3 / §19.9 base #9). When any of these
# show up in params with a truthy value, the audit's triage flips to
# ``user_decision_required`` — the question the caller is asking is no
# longer purely deterministic, and the agent must not silently resolve
# it.  Configurable via M1AuditConfig.subjective_keys.
DEFAULT_SUBJECTIVE_INPUT_KEYS: tuple[str, ...] = (
    "user_target_intent_review_required",
    "subjective_success_review",
    "budget_decision_required",
)


@dataclass(slots=True)
class M1AuditConfig:
    """Tunable knobs for the M1 success-audit gate. Independent of
    :class:`MethodologyConfig` so the gate can be flipped without
    touching the wider methodology wrapper."""

    enabled: bool = True
    # archival-method whitelist; audit fires when method ∈ this set OR
    # the params expose ``target_success``/``archival_allowed = true``.
    archival_methods: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_ARCHIVAL_METHODS,
    )
    # Pass-rate floor below which the claim is rejected outright (C).
    min_pass_rate: float = 0.95
    # Pass-rate threshold above which overfit-with-pinned-dim is
    # flagged. 100% pass + 0-variance dim ⇒ B/overfit.
    overfit_pass_rate_floor: float = 0.99
    # Minimum sample count required for class A. Below this, even a
    # clean closure tops out at B.
    min_samples_for_class_a: int = 5
    # Minimum unique values per declared dimension to consider it
    # *tested*. ``unique_count < this`` ⇒ untested.
    min_unique_per_dim: int = 2
    # Scope tags acceptable for class A.
    class_a_scope_tags: tuple[str, ...] = ("full_input_space", "cross_session")
    # Sentinel for downgrade rewrite.
    downgraded_label: str = "strong_partial"
    # v0.4.0 B3 — triage split.
    # Param keys whose truthy presence flips the audit triage from
    # ``agent_self_resolved`` to ``user_decision_required``.  The
    # default list is fine for VMP algorithm extraction; other domains
    # may extend or replace it via env override.
    subjective_keys: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_SUBJECTIVE_INPUT_KEYS,
    )

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "M1AuditConfig":
        src = env if env is not None else os.environ
        cfg = cls()
        flag = (src.get("UTOV_M1_AUDIT") or "").strip().lower()
        if flag in ("off", "0", "false", "no"):
            cfg.enabled = False
        for env_key, attr, cast in (
            ("UTOV_M1_AUDIT_MIN_PASS_RATE",      "min_pass_rate",            float),
            ("UTOV_M1_AUDIT_OVERFIT_FLOOR",      "overfit_pass_rate_floor",  float),
            ("UTOV_M1_AUDIT_MIN_SAMPLES",        "min_samples_for_class_a",  int),
            ("UTOV_M1_AUDIT_MIN_UNIQUE_PER_DIM", "min_unique_per_dim",       int),
        ):
            v = src.get(env_key)
            if v is None:
                continue
            try:
                setattr(cfg, attr, cast(v))
            except ValueError:
                continue
        subj = src.get("UTOV_M1_AUDIT_SUBJECTIVE_KEYS")
        if subj is not None:
            cfg.subjective_keys = tuple(
                k.strip() for k in subj.split(",") if k.strip()
            )
        return cfg


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SuccessAuditResult:
    """Structured outcome of one audit pass.

    The wrapper consumes ``action`` directly:
      - ``allow``     → proceed; attach a passing-audit note to the envelope
      - ``downgrade`` → rewrite params (see :func:`apply_audit_to_params`)
                        then proceed; attach a downgrade note
      - ``reject``    → intercept; ``intercepted_reason`` is the alert text

    ``triage`` is the v0.4.0 B3 split (§19.9 base #9): when every weak
    axis is a deterministic check (dimension coverage / overfit / scope /
    closure / sample count) the result is tagged
    ``agent_self_resolved`` — the caller agent may act on the verdict
    without escalating to the user. When the caller surfaces a
    non-deterministic input (an explicit subjective-review flag, see
    :attr:`M1AuditConfig.subjective_keys`) the triage flips to
    ``user_decision_required``.
    """

    evidence_class: str          # 'A' | 'B' | 'C'
    action: str                  # 'allow' | 'downgrade' | 'reject'
    downgraded_to: str | None    # 'strong_partial' when action=='downgrade'
    dimension_coverage: dict[str, int]
    untested_dimensions: tuple[str, ...]
    overfit_flag: bool
    scope: str                   # 'full_input_space' | 'cross_session' | 'in_session' | 'unknown'
    closure_consistent: bool | None
    closure_paths: tuple[str, ...]
    pass_rate: float | None
    sample_count: int
    notes: tuple[str, ...]
    intercepted_reason: str | None  # populated when action=='reject'
    # v0.4.0 B3 — agent-vs-user delegation split.
    triage: str = "agent_self_resolved"      # 'agent_self_resolved' | 'user_decision_required'
    subjective_inputs: tuple[str, ...] = ()
    # v0.4.0 B4 — env-limit row exclusion (count of runner-imposed
    # nulls that were stripped from pass-rate math before grading).
    env_limit_rows_excluded: int = 0
    adjusted_pass_rate: float | None = None  # post-exclusion pass-rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_class":          self.evidence_class,
            "action":                  self.action,
            "downgraded_to":           self.downgraded_to,
            "dimension_coverage":      dict(self.dimension_coverage),
            "untested_dimensions":     list(self.untested_dimensions),
            "overfit_flag":            self.overfit_flag,
            "scope":                   self.scope,
            "closure_consistent":      self.closure_consistent,
            "closure_paths":           list(self.closure_paths),
            "pass_rate":               self.pass_rate,
            "sample_count":            self.sample_count,
            "notes":                   list(self.notes),
            "intercepted_reason":      self.intercepted_reason,
            "triage":                  self.triage,
            "subjective_inputs":       list(self.subjective_inputs),
            "env_limit_rows_excluded": self.env_limit_rows_excluded,
            "adjusted_pass_rate":      self.adjusted_pass_rate,
        }


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def audit_success_claim(
    method: str,
    params: dict[str, Any] | None,
    *,
    cfg: M1AuditConfig | None = None,
) -> SuccessAuditResult | None:
    """Run the M1 audit if ``method``/``params`` look like an archival
    action.  Returns ``None`` when the audit is disabled or doesn't
    apply (so the wrapper can short-circuit and dispatch normally).
    """
    cfg = cfg or M1AuditConfig.from_env()
    if not cfg.enabled:
        return None

    params = params or {}
    claim = _extract_claim(params)
    if claim is None:
        # No target_success / archival_allowed flag AND method not in
        # archival_methods → not our concern.
        if method not in cfg.archival_methods:
            return None
        # An archival method without a positive claim still gets
        # audited so we can refuse a bare boolean — but only if the
        # params actually carry a *positive* signal somewhere; if no
        # claim flag is set at all, let it pass through.
        return None

    inputs = _extract_audit_inputs(params, cfg=cfg)
    subjective = _detect_subjective_inputs(params, cfg.subjective_keys)
    return _grade(claim, inputs, cfg, subjective_inputs=subjective)


def apply_audit_to_params(
    params: dict[str, Any] | None,
    result: SuccessAuditResult,
) -> dict[str, Any]:
    """Mutate ``params`` in-place to reflect an audit outcome.

    For ``action == 'downgrade'`` this:
      - flips every ``target_success`` it finds to ``False``
      - flips every ``archival_allowed`` it finds to ``False``
      - inserts ``evidence_class`` / ``downgraded_to`` /
        ``m1_audit`` annotations at the top level so the
        downstream archival path sees a ``strong_partial`` claim
        rather than a full success.

    For ``action == 'reject'`` no mutation is needed (the wrapper
    intercepts before dispatch); we still write the audit block so
    callers that bypass the wrapper see *why* the params would have
    been rejected.

    For ``action == 'allow'`` we only annotate; the success claim
    stands.
    """
    if params is None:
        return {"m1_audit": result.to_dict()}

    if result.action == "downgrade":
        _flip_all_keys(params, "target_success", False)
        _flip_all_keys(params, "archival_allowed", False)
        params["evidence_class"] = "B"
        params["downgraded_to"]  = result.downgraded_to or "strong_partial"

    elif result.action == "reject":
        # Leave the claim words alone — the wrapper refuses dispatch
        # entirely. We just attach the audit so the rejection is
        # readable in audit trails.
        params["evidence_class"] = "C"

    else:  # allow
        params.setdefault("evidence_class", "A")

    params["m1_audit"] = result.to_dict()
    return params


def render_audit_alert(result: SuccessAuditResult) -> str:
    """Format an envelope alert string from an audit result."""
    if result.action == "reject":
        return (
            f"[M1-AUDIT/REJECT] evidence_class=C: "
            f"{result.intercepted_reason or '; '.join(result.notes) or 'insufficient evidence'}. "
            f"Archival refused — supply more samples / fix closure / raise pass_rate."
        )
    if result.action == "downgrade":
        untested = ", ".join(result.untested_dimensions) or "(none)"
        return (
            f"[M1-AUDIT/DOWNGRADE→strong_partial] evidence_class=B: "
            f"untested_dimensions={untested}; scope={result.scope}; "
            f"overfit_flag={result.overfit_flag}. "
            f"target_success/archival_allowed rewritten to false; "
            f"caller should re-test the pinned dimension before re-archiving."
        )
    return (
        f"[M1-AUDIT/ALLOW] evidence_class=A: "
        f"all {result.sample_count} samples covered "
        f"{len(result.dimension_coverage)} dimensions; scope={result.scope}; "
        f"closure_consistent={result.closure_consistent}."
    )


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


_CLAIM_KEYS: tuple[str, ...] = ("target_success", "archival_allowed")


def _extract_claim(params: dict[str, Any]) -> str | None:
    """Return the first claim key found set to a truthy value, walking
    common containers (``report``, ``payload``, ``metrics``). Returns
    ``None`` if no positive claim is present."""
    for key in _CLAIM_KEYS:
        if _truthy_anywhere(params, key):
            return key
    return None


def _truthy_anywhere(node: Any, key: str, *, depth: int = 4) -> bool:
    if depth <= 0:
        return False
    if isinstance(node, dict):
        if key in node:
            v = node[key]
            if isinstance(v, bool) and v is True:
                return True
            if isinstance(v, str) and v.strip().lower() in ("true", "1", "yes"):
                return True
        for v in node.values():
            if _truthy_anywhere(v, key, depth=depth - 1):
                return True
    elif isinstance(node, list):
        for v in node:
            if _truthy_anywhere(v, key, depth=depth - 1):
                return True
    return False


def _flip_all_keys(node: Any, key: str, new_value: Any, *, depth: int = 4) -> None:
    if depth <= 0:
        return
    if isinstance(node, dict):
        if key in node:
            node[key] = new_value
        for v in node.values():
            _flip_all_keys(v, key, new_value, depth=depth - 1)
    elif isinstance(node, list):
        for v in node:
            _flip_all_keys(v, key, new_value, depth=depth - 1)


def _first_value(node: Any, key: str, *, depth: int = 4) -> Any:
    """Walk ``node`` and return the first value found under ``key``."""
    if depth <= 0:
        return None
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for v in node.values():
            r = _first_value(v, key, depth=depth - 1)
            if r is not None:
                return r
    elif isinstance(node, list):
        for v in node:
            r = _first_value(v, key, depth=depth - 1)
            if r is not None:
                return r
    return None


@dataclass(frozen=True, slots=True)
class _AuditInputs:
    """Caller-supplied evidence pulled out of params."""
    dependencies:  tuple[str, ...]
    samples:       tuple[dict[str, Any], ...]
    pass_rate:     float | None
    scope:         str
    closure_paths: tuple[tuple[str, Any], ...]   # (name, digest)
    # v0.4.0 B4 — env-limit row carve-out.
    env_limit_rows_excluded: int = 0
    adjusted_pass_rate: float | None = None


def _extract_audit_inputs(
    params: dict[str, Any],
    *,
    cfg: M1AuditConfig | None = None,
) -> _AuditInputs:
    deps = _first_value(params, "success_dependencies")
    if not isinstance(deps, (list, tuple)):
        deps = ()
    samples = _first_value(params, "samples")
    if not isinstance(samples, (list, tuple)):
        samples = ()
    pass_rate = _first_value(params, "pass_rate")
    if isinstance(pass_rate, bool):
        pass_rate = None
    if isinstance(pass_rate, str):
        try:
            pass_rate = float(pass_rate)
        except ValueError:
            pass_rate = None
    # v0.4.0 B4 — env-limit rows opt-in. Two accepted shapes:
    #
    #   * top-level ``env_limit_rows: int`` (or ``env_limit_samples: int``) —
    #     count of runner-imposed nulls to strip from pass-rate math.
    #   * per-sample ``{"env_limit": true}`` flag — the carve-out counts
    #     them automatically.
    #
    # Both subtract from denominator AND numerator (those samples are
    # treated as "not graded").
    env_limit_count = _first_value(params, "env_limit_rows")
    if env_limit_count is None:
        env_limit_count = _first_value(params, "env_limit_samples")
    env_limit_count = env_limit_count if isinstance(env_limit_count, int) else 0
    # Plus per-sample flag (additive — if the caller passes both, the
    # explicit count wins for unflagged rows and we add flagged ones).
    per_sample_env_limit = sum(
        1 for s in samples
        if isinstance(s, dict) and bool(s.get("env_limit"))
    )
    env_limit_count = max(env_limit_count, per_sample_env_limit)

    adjusted_pass_rate: float | None = None
    if not isinstance(pass_rate, (int, float)):
        # Fall back to checked/passed pair if present.
        checked = _first_value(params, "checked")
        passed  = _first_value(params, "passed")
        if isinstance(checked, int) and isinstance(passed, int) and checked > 0:
            pass_rate = float(passed) / float(checked)
            if env_limit_count > 0 and checked > env_limit_count:
                adj_checked = checked - env_limit_count
                adj_passed  = max(0, passed)
                adjusted_pass_rate = float(adj_passed) / float(adj_checked)
        else:
            pass_rate = None
    else:
        pass_rate = float(pass_rate)
        # When pass_rate is supplied directly, the caller is responsible
        # for whether env-limit rows are already excluded. We can only
        # adjust when checked/passed are also present.
        checked = _first_value(params, "checked")
        passed  = _first_value(params, "passed")
        if (
            env_limit_count > 0
            and isinstance(checked, int) and isinstance(passed, int)
            and checked > env_limit_count
        ):
            adj_checked = checked - env_limit_count
            adjusted_pass_rate = float(passed) / float(adj_checked)

    scope = _first_value(params, "scope")
    if not isinstance(scope, str) or not scope.strip():
        scope = "unknown"
    closure = _first_value(params, "closure_paths")
    pairs: list[tuple[str, Any]] = []
    if isinstance(closure, (list, tuple)):
        for entry in closure:
            if not isinstance(entry, dict):
                continue
            name   = entry.get("name") or entry.get("path") or ""
            digest = entry.get("digest") or entry.get("value") or entry.get("bytes")
            if isinstance(name, str) and digest is not None:
                pairs.append((name, digest))
    # v0.4.0 B4 — when env-limit rows are excluded we filter them out
    # of the sample tuple used for dimension coverage / closure / count.
    if env_limit_count > 0 and per_sample_env_limit > 0:
        filtered_samples = tuple(
            s for s in samples
            if isinstance(s, dict) and not bool(s.get("env_limit"))
        )
    else:
        filtered_samples = tuple(s for s in samples if isinstance(s, dict))
    return _AuditInputs(
        dependencies=tuple(str(d) for d in deps),
        samples=filtered_samples,
        pass_rate=float(pass_rate) if isinstance(pass_rate, (int, float)) else None,
        scope=scope.strip(),
        closure_paths=tuple(pairs),
        env_limit_rows_excluded=env_limit_count,
        adjusted_pass_rate=adjusted_pass_rate,
    )


def _detect_subjective_inputs(
    params: dict[str, Any],
    subjective_keys: tuple[str, ...],
) -> tuple[str, ...]:
    """Return the subset of ``subjective_keys`` that appear truthy
    anywhere in ``params``.

    Each match means the caller has explicitly flagged a
    non-deterministic input — target intent review, budget decision,
    "is this what we mean by success?" — and the audit triage must
    surface to the user rather than letting the agent self-resolve.
    """
    return tuple(k for k in subjective_keys if _truthy_anywhere(params, k))


# ---------------------------------------------------------------------------
# The four checks + grader
# ---------------------------------------------------------------------------


def _check_dimension_coverage(
    deps: Iterable[str],
    samples: Iterable[dict[str, Any]],
) -> tuple[dict[str, int], tuple[str, ...]]:
    samples = list(samples)
    coverage: dict[str, int] = {}
    untested: list[str] = []
    for dim in deps:
        seen: set[str] = set()
        for s in samples:
            if dim not in s:
                continue
            try:
                seen.add(_normalise_for_count(s[dim]))
            except TypeError:
                seen.add(repr(s[dim]))
        coverage[dim] = len(seen)
    return coverage, tuple(untested)


def _normalise_for_count(v: Any) -> str:
    if isinstance(v, (bytes, bytearray)):
        return v.hex()
    if isinstance(v, (list, tuple)):
        return "|".join(_normalise_for_count(x) for x in v)
    return repr(v)


def _check_closure(paths: tuple[tuple[str, Any], ...]) -> tuple[bool | None, tuple[str, ...]]:
    if len(paths) < 2:
        return None, tuple(name for name, _ in paths)
    names = tuple(name for name, _ in paths)
    digests = {_normalise_for_count(d) for _, d in paths}
    return (len(digests) == 1), names


def _infer_scope(
    declared: str,
    coverage: dict[str, int],
    sample_count: int,
    cfg: M1AuditConfig,
) -> str:
    """Honour an explicit scope tag; otherwise downgrade to the lowest
    scope the evidence supports."""
    declared_low = declared.strip().lower()
    valid = {"full_input_space", "cross_session", "in_session", "unknown"}
    if declared_low in valid and declared_low != "unknown":
        return declared_low
    # Inference: if every dim has unique_count >= min_unique_per_dim AND
    # sample_count >= min_samples_for_class_a → call it cross_session;
    # else in_session; if no coverage info, unknown.
    if not coverage:
        return "unknown"
    well_covered = all(c >= cfg.min_unique_per_dim for c in coverage.values())
    if well_covered and sample_count >= cfg.min_samples_for_class_a:
        return "cross_session"
    return "in_session"


def _grade(
    claim: str,
    inputs: _AuditInputs,
    cfg: M1AuditConfig,
    *,
    subjective_inputs: tuple[str, ...] = (),
) -> SuccessAuditResult:
    notes: list[str] = []
    coverage, _ = _check_dimension_coverage(inputs.dependencies, inputs.samples)
    untested = tuple(
        dim for dim, n in coverage.items() if n < cfg.min_unique_per_dim
    )
    if inputs.dependencies and not inputs.samples:
        notes.append("no samples supplied — every declared dimension is untested")
    closure_consistent, closure_names = _check_closure(inputs.closure_paths)
    scope = _infer_scope(inputs.scope, coverage, len(inputs.samples), cfg)

    # v0.4.0 B4 — prefer the env-limit-adjusted pass-rate when present.
    # The adjusted rate strips runner-imposed nulls before grading so an
    # algorithm-correct claim isn't penalised for the runner's
    # environment ceiling.
    effective_pass_rate = (
        inputs.adjusted_pass_rate
        if inputs.adjusted_pass_rate is not None
        else inputs.pass_rate
    )
    if inputs.adjusted_pass_rate is not None and inputs.env_limit_rows_excluded:
        raw_repr = (
            f"{inputs.pass_rate:.3f}" if inputs.pass_rate is not None else "NA"
        )
        notes.append(
            f"env-limit carve-out: excluded {inputs.env_limit_rows_excluded} "
            f"runner-imposed null row(s); adjusted pass_rate="
            f"{inputs.adjusted_pass_rate:.3f} (raw {raw_repr})"
        )

    overfit_flag = bool(
        effective_pass_rate is not None
        and effective_pass_rate >= cfg.overfit_pass_rate_floor
        and untested
    )
    if overfit_flag:
        notes.append(
            f"pass_rate={effective_pass_rate:.3f} ≥ {cfg.overfit_pass_rate_floor} "
            f"with pinned dim(s) {','.join(untested)} — overfit signature"
        )

    triage = (
        "user_decision_required"
        if subjective_inputs else "agent_self_resolved"
    )
    if subjective_inputs:
        notes.append(
            "subjective input(s) flagged: " + ",".join(subjective_inputs)
            + " — triage routed to user"
        )

    # --- C: hard reject conditions ----------------------------------
    reject_reasons: list[str] = []
    if inputs.dependencies and not inputs.samples:
        reject_reasons.append("no samples")
    if effective_pass_rate is not None and effective_pass_rate < cfg.min_pass_rate:
        reject_reasons.append(f"pass_rate={effective_pass_rate:.3f} < {cfg.min_pass_rate}")
    if closure_consistent is False:
        reject_reasons.append(
            f"closure paths disagree across {','.join(closure_names) or 'paths'}"
        )

    if reject_reasons:
        reason = "; ".join(reject_reasons)
        return SuccessAuditResult(
            evidence_class="C",
            action="reject",
            downgraded_to=None,
            dimension_coverage=coverage,
            untested_dimensions=untested,
            overfit_flag=overfit_flag,
            scope=scope,
            closure_consistent=closure_consistent,
            closure_paths=closure_names,
            pass_rate=inputs.pass_rate,
            sample_count=len(inputs.samples),
            notes=tuple(notes),
            intercepted_reason=(
                f"M1 audit refused archival of {claim}=true: {reason}. "
                f"Fix the listed conditions, then resubmit."
            ),
            triage=triage,
            subjective_inputs=subjective_inputs,
            env_limit_rows_excluded=inputs.env_limit_rows_excluded,
            adjusted_pass_rate=inputs.adjusted_pass_rate,
        )

    # --- A vs B ------------------------------------------------------
    weak_axes: list[str] = []
    if untested:
        weak_axes.append(f"untested_dim={','.join(untested)}")
    if scope not in cfg.class_a_scope_tags:
        weak_axes.append(f"scope={scope}")
    if closure_consistent is None and len(closure_names) <= 1:
        # Single-path "closure" isn't a closure. Not fatal, but B.
        weak_axes.append("single_path_closure")
    if overfit_flag:
        weak_axes.append("overfit_flag")
    if (
        inputs.dependencies
        and len(inputs.samples) < cfg.min_samples_for_class_a
    ):
        weak_axes.append(
            f"sample_count={len(inputs.samples)} < {cfg.min_samples_for_class_a}"
        )

    if not weak_axes:
        return SuccessAuditResult(
            evidence_class="A",
            action="allow",
            downgraded_to=None,
            dimension_coverage=coverage,
            untested_dimensions=untested,
            overfit_flag=overfit_flag,
            scope=scope,
            closure_consistent=closure_consistent,
            closure_paths=closure_names,
            pass_rate=inputs.pass_rate,
            sample_count=len(inputs.samples),
            notes=tuple(notes),
            intercepted_reason=None,
            triage=triage,
            subjective_inputs=subjective_inputs,
            env_limit_rows_excluded=inputs.env_limit_rows_excluded,
            adjusted_pass_rate=inputs.adjusted_pass_rate,
        )

    notes.append("weak_axes=" + ",".join(weak_axes))
    return SuccessAuditResult(
        evidence_class="B",
        action="downgrade",
        downgraded_to=cfg.downgraded_label,
        dimension_coverage=coverage,
        untested_dimensions=untested,
        overfit_flag=overfit_flag,
        scope=scope,
        closure_consistent=closure_consistent,
        closure_paths=closure_names,
        pass_rate=inputs.pass_rate,
        sample_count=len(inputs.samples),
        notes=tuple(notes),
        intercepted_reason=None,
        triage=triage,
        subjective_inputs=subjective_inputs,
        env_limit_rows_excluded=inputs.env_limit_rows_excluded,
        adjusted_pass_rate=inputs.adjusted_pass_rate,
    )
