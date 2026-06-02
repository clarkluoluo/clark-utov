"""setup_symex.lint — a pre-flight wiring-invariant checker (spec #3).

A lightweight, **extensible** lint over the parity / self-check *inputs* — run
BEFORE any heavy symex/parity pass — that catches the most common wiring errors
(the seed/sink/observed/mask mis-wirings) and returns ERROR/WARN/INFO findings,
each carrying a ``code`` + ``detail`` + ``fix``. It is **general-first**: the
check set is an invariant *registry* (a list of named predicates), so adding a
new wiring invariant is ONE registry entry — the two TC2 false blockers this
round are proof-points, not the design target.

A8 four-check (spec §A8):
  1. Reuse — invariants read the SAME descriptor fields the existing
     ``check_emit_self_consistency`` (``_emit_selfcheck.py:129``) and
     ``check_parity_vectors`` (``_parity.py:145``) consume (``seed_values``,
     ``trace_sink``/``sink_value``, ``sink_mask``, ``sink_form``, ``observed``);
     no new symex.
  2. Subject = the WIRING, not the algorithm — pure pre-flight, touches no
     verdict logic.
  3. Preserve — lint is additive and NON-BLOCKING by default; an ERROR is loud
     (surfaced in ``per_step``) but ``drive()`` is unchanged when lint is OK.
  4. Degenerate → surfaced — an invariant that cannot evaluate its subject emits
     an ``INFO``/``WARN`` with the reason, never a silent pass that reads as
     "checked and clean".

Public API::

    lint_parity_inputs(case_config, *, window, declared_inputs,
                       observed_spec, sink_spec) -> LintReport
    lint_case_config(case_config, *, seed_values=None, observed_spec=None,
                     sink_spec=None) -> LintReport
    register_invariant(invariant)              # extend the registry
    INVARIANTS                                 # the seed registry (4 entries)

Descriptor shapes (all plain Mappings — declarative, runner-facing):
  * ``declared_inputs`` — the emit input set. Either a sequence of names, or a
    mapping ``name -> width_bytes`` (width optional / may be ``None``).
  * ``sink_spec``     — ``{"source": "window_exit"|<other>, "value": ...,
    "seed_values": {name: val}, "mask": int|None, "reg_width": int_bytes|None,
    "form": "reg"|"mem"}``. ``source`` says where the sink value was taken from;
    anything that is not this window's exit is the phase_D-style mis-wiring.
  * ``observed_spec`` — ``{"source": "window_local_sink"|"top_level_output"|...,
    "value": ...}``. ``source`` says what the parity ``observed`` axis measures;
    ``top_level_output`` is the phase_C-style mis-wiring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence


# --------------------------------------------------------------------------- #
# Finding / report value types
# --------------------------------------------------------------------------- #

_LEVEL_RANK = {"OK": 0, "INFO": 1, "WARN": 2, "ERROR": 3}


@dataclass(frozen=True, slots=True)
class LintFinding:
    """One wiring-invariant result. ``level`` in {ERROR, WARN, INFO}; a clean
    invariant yields NO finding (its absence is the OK). Every finding carries a
    stable ``code``, a human ``detail``, and an actionable ``fix``."""

    level:  str   # ERROR | WARN | INFO
    code:   str
    detail: str
    fix:    str

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "code": self.code,
                "detail": self.detail, "fix": self.fix}


@dataclass(frozen=True, slots=True)
class LintReport:
    """The aggregate of every invariant's findings + the worst level seen.

    ``max_level`` is ``OK`` when no finding was emitted, else the highest of
    INFO < WARN < ERROR. The report is non-blocking data: the caller (or a gate)
    decides what an ERROR means — :func:`drive` surfaces it loudly in
    ``per_step`` but does not abort on it."""

    findings:  tuple[LintFinding, ...] = ()

    @property
    def max_level(self) -> str:
        if not self.findings:
            return "OK"
        return max((f.level for f in self.findings), key=lambda lv: _LEVEL_RANK[lv])

    @property
    def ok(self) -> bool:
        return self.max_level == "OK"

    @property
    def has_error(self) -> bool:
        return any(f.level == "ERROR" for f in self.findings)

    def by_code(self, code: str) -> tuple[LintFinding, ...]:
        return tuple(f for f in self.findings if f.code == code)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind":      "setup_symex_parity_wiring_lint",
            "max_level": self.max_level,
            "findings":  [f.to_dict() for f in self.findings],
        }


# --------------------------------------------------------------------------- #
# Lint context — bundles the descriptors every invariant reads
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class LintContext:
    """What an invariant predicate sees. Plain held data — no symex, no verdict.

    The descriptor fields are normalised at construction so each invariant reads
    a stable shape (and a missing one → the degenerate INFO/WARN path, never a
    crash)."""

    case_config:     Any
    window:          tuple[int, int] | None
    declared_inputs: Mapping[str, int | None]
    observed_spec:   Mapping[str, Any] | None
    sink_spec:       Mapping[str, Any] | None


def _normalize_declared(declared: Any) -> Mapping[str, int | None]:
    """A sequence of names → {name: None}; a mapping → {name: width_or_None}."""
    if declared is None:
        return {}
    if isinstance(declared, Mapping):
        return {str(k): (int(v) if v is not None else None)
                for k, v in declared.items()}
    return {str(n): None for n in declared}


# --------------------------------------------------------------------------- #
# Invariant registry — the extensibility point. Each entry is ONE Invariant.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class Invariant:
    """A named wiring invariant: a predicate over a :class:`LintContext` that
    yields zero (clean) or more :class:`LintFinding`. Adding a new wiring check =
    constructing one of these and :func:`register_invariant`-ing it."""

    code:  str
    check: Callable[["LintContext"], Iterable[LintFinding]]
    doc:   str = ""


# ---- seed invariant 1: seed_values shape vs the declared input set ---------- #

def _inv_seed_shape(ctx: LintContext) -> Iterable[LintFinding]:
    """``seed_values`` keys/shape must match the DECLARED input set (count, names,
    and — when both are known — widths). This is the phase_D-style mis-wiring
    where the self-check was fed the wrong seed for the window's declared inputs."""
    code = "SEED_SHAPE_MISMATCH"
    sink = ctx.sink_spec or {}
    if "seed_values" not in sink:
        yield LintFinding(
            "INFO", code,
            "no seed_values supplied in sink_spec — cannot check seed shape "
            "against the declared inputs.",
            "supply sink_spec['seed_values'] = {input_name: window-entry concrete "
            "value} so the seed<->declared-input wiring is verifiable.")
        return
    seed_values = sink.get("seed_values")
    if not isinstance(seed_values, Mapping):
        yield LintFinding(
            "ERROR", code,
            f"seed_values is {type(seed_values).__name__}, not a name->value map; "
            "the self-check binds inputs by name.",
            "pass seed_values as a mapping {input_name: concrete_value}.")
        return
    declared = ctx.declared_inputs
    if not declared:
        yield LintFinding(
            "INFO", code,
            "no declared inputs supplied — cannot check seed_values shape.",
            "supply declared_inputs (case_config.inputs) so the seed wiring is "
            "verifiable.")
        return
    seed_names = set(seed_values.keys())
    declared_names = set(declared.keys())
    missing = sorted(declared_names - seed_names)
    extra = sorted(seed_names - declared_names)
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"no seed value for declared input(s) {missing}")
        if extra:
            parts.append(f"seed value(s) {extra} are not declared inputs")
        yield LintFinding(
            "ERROR", code,
            "seed_values shape does not match the declared input set: "
            + "; ".join(parts)
            + f" (declared={sorted(declared_names)}, seeded={sorted(seed_names)}).",
            "wire seed_values to EXACTLY this window's declared inputs "
            "(case_config.inputs) — one window-entry concrete value per declared "
            "input, no more, no fewer. A surplus/missing key means the self-check "
            "is reading another window's (or the top-level) seed.")
        return
    # names align — check widths where both are known.
    for name, width in declared.items():
        if width is None:
            continue
        val = seed_values.get(name)
        if isinstance(val, (bytes, bytearray)):
            actual = len(val)
        elif isinstance(val, int) and not isinstance(val, bool):
            actual = (max(val.bit_length(), 1) + 7) // 8
        else:
            continue
        if actual > width:
            yield LintFinding(
                "WARN", code,
                f"seed value for input '{name}' is {actual} byte(s) but the "
                f"declared input width is {width} byte(s) — the seed will be "
                "truncated/masked at the boundary.",
                f"declare input '{name}' at its real byte width, or supply a seed "
                "value that fits the declared width.")


# ---- seed invariant 2: sink sourced from THIS window's exit ----------------- #

_WINDOW_EXIT_SOURCES = {"window_exit", "window_local_sink", "this_window_exit"}


def _inv_sink_source(ctx: LintContext) -> Iterable[LintFinding]:
    """The ``sink`` value must be sourced from THIS window's exit, not elsewhere
    (a downstream/top-level value). This is the phase_D-style mis-wiring where the
    self-check compared F against a sink read off the wrong place."""
    code = "SINK_SOURCE"
    sink = ctx.sink_spec
    if sink is None:
        yield LintFinding(
            "INFO", code,
            "no sink_spec supplied — cannot check the sink source.",
            "supply sink_spec with a 'source' field naming where the sink value "
            "was taken from (expected: this window's exit).")
        return
    source = sink.get("source")
    if source is None:
        yield LintFinding(
            "WARN", code,
            "sink_spec has no 'source' field — the sink's provenance is "
            "unverifiable, so it cannot be confirmed to be THIS window's exit.",
            "set sink_spec['source'] = 'window_exit' and source the sink value "
            "from this window's exit (the runner's trace_self_check sink_value).")
        return
    if str(source) not in _WINDOW_EXIT_SOURCES:
        yield LintFinding(
            "ERROR", code,
            f"sink value is sourced from '{source}', not this window's exit — the "
            "self-check would compare the recovered F against a value the window "
            "does not produce.",
            "source the sink from THIS window's exit (sink_spec['source'] = "
            "'window_exit'; the runner's trace_self_check sink_value for this "
            "window), never a downstream/top-level value.")


# ---- seed invariant 3: observed = window-local sink, not top-level output --- #

def _inv_observed_semantics(ctx: LintContext) -> Iterable[LintFinding]:
    """The parity ``observed`` axis must be the WINDOW-LOCAL sink, not the
    top-level program output. This is the phase_C-style mis-wiring (parity took
    the wrong ``observed`` semantics) — a real oracle run of this window, not the
    whole program's final output."""
    code = "OBSERVED_SEMANTICS"
    obs = ctx.observed_spec
    if obs is None:
        yield LintFinding(
            "INFO", code,
            "no observed_spec supplied — cannot check the observed semantics.",
            "supply observed_spec with a 'source' field naming what the parity "
            "observed axis measures (expected: this window's local sink).")
        return
    source = obs.get("source")
    if source is None:
        yield LintFinding(
            "WARN", code,
            "observed_spec has no 'source' field — the observed axis's semantics "
            "are unverifiable, so it cannot be confirmed to be the window-local "
            "sink.",
            "set observed_spec['source'] = 'window_local_sink' — parity observed "
            "must be this window's exit output for each input, not the program's "
            "final output.")
        return
    src = str(source)
    if src in {"top_level_output", "program_output", "final_output"}:
        yield LintFinding(
            "ERROR", code,
            f"parity observed is the {src} (the whole program's final output), "
            "not this window's local sink — parity would compare the recovered "
            "window-transform F against a value that includes everything "
            "downstream of the window, so it can never match (or matches for the "
            "wrong reason).",
            "wire observed to the WINDOW-LOCAL sink (observed_spec['source'] = "
            "'window_local_sink'): for each cohort input, run THIS window and take "
            "its exit output, not the top-level program output.")
        return
    if src not in {"window_local_sink", "window_exit", "this_window_exit"}:
        yield LintFinding(
            "WARN", code,
            f"parity observed source '{src}' is unrecognised — cannot confirm it "
            "is the window-local sink.",
            "set observed_spec['source'] = 'window_local_sink' (this window's exit "
            "output per input).")


# ---- seed invariant 4: sink_mask width vs register/sink byte width ---------- #

def _mask_byte_width(mask: int) -> int | None:
    """A contiguous low-bit mask (0xff, 0xffff, 0xffffffff, …) → its byte width.
    Returns None for a non-contiguous / non-byte-aligned mask (can't size it)."""
    if mask <= 0:
        return None
    # contiguous-from-bit-0 mask: mask+1 is a power of two.
    if mask & (mask + 1) != 0:
        return None
    bits = mask.bit_length()
    if bits % 8 != 0:
        return None
    return bits // 8


def _inv_sink_mask_width(ctx: LintContext) -> Iterable[LintFinding]:
    """``sink_mask`` width must match the register/sink byte width. A mask wider
    or narrower than the sink register silently keeps/drops bytes the comparison
    should not, so F-on-trace vs trace-sink compares the wrong byte window."""
    code = "SINK_MASK_WIDTH"
    sink = ctx.sink_spec
    if sink is None:
        yield LintFinding(
            "INFO", code,
            "no sink_spec supplied — cannot check the sink mask width.",
            "supply sink_spec with 'mask' and 'reg_width' (bytes) for this sink.")
        return
    form = str(sink.get("form", "reg"))
    mask = sink.get("mask")
    reg_width = sink.get("reg_width")
    if form == "mem":
        if mask is not None:
            yield LintFinding(
                "WARN", code,
                "sink_spec declares a mem (bytes) sink but also carries an integer "
                "sink_mask — a bytes sink is compared whole, the mask is ignored "
                "and is a sign the sink form/mask are mis-wired.",
                "for a mem sink set sink_mask=None (compare the bytes whole); use "
                "sink_mask only for an integer register sink.")
        return
    if mask is None:
        yield LintFinding(
            "INFO", code,
            "no sink_mask supplied (raw / unmasked compare) — width unchecked.",
            "for an integer register sink, set sink_mask to the register width "
            "(e.g. 0xffffffff for a 32-bit reg) so the compare is width-correct.")
        return
    if not isinstance(mask, int) or isinstance(mask, bool):
        yield LintFinding(
            "ERROR", code,
            f"sink_mask is {type(mask).__name__}, not an integer width mask.",
            "set sink_mask to an integer low-bit mask matching the sink register "
            "width (e.g. 0xffffffff for 4 bytes).")
        return
    mask_w = _mask_byte_width(mask)
    if mask_w is None:
        yield LintFinding(
            "WARN", code,
            f"sink_mask {hex(mask)} is not a contiguous byte-aligned width mask — "
            "its byte width cannot be checked against the register width.",
            "use a contiguous low-bit mask (0xff / 0xffff / 0xffffffff / "
            "0xffffffffffffffff) matching the sink register byte width.")
        return
    if reg_width is None:
        yield LintFinding(
            "INFO", code,
            f"sink_mask {hex(mask)} ({mask_w} byte(s)) supplied but no reg_width "
            "to check it against.",
            "supply sink_spec['reg_width'] (the sink register's byte width) so the "
            "mask width can be verified.")
        return
    if int(reg_width) != mask_w:
        yield LintFinding(
            "ERROR", code,
            f"sink_mask {hex(mask)} is {mask_w} byte(s) but the sink register "
            f"width is {int(reg_width)} byte(s) — the comparison would mask off (or "
            "keep) the wrong byte window, so F-on-trace vs trace-sink compares "
            "mismatched widths.",
            f"set sink_mask to the {int(reg_width)}-byte register width "
            f"(e.g. {hex((1 << (int(reg_width) * 8)) - 1)}).")


# The seed registry — the four spec invariants. Extensible: append an entry via
# register_invariant() (proved by the registry test — no other code change).
INVARIANTS: list[Invariant] = [
    Invariant("SEED_SHAPE_MISMATCH", _inv_seed_shape,
              "seed_values keys/shape vs the declared input set"),
    Invariant("SINK_SOURCE", _inv_sink_source,
              "sink sourced from THIS window's exit, not elsewhere"),
    Invariant("OBSERVED_SEMANTICS", _inv_observed_semantics,
              "parity observed = window-local sink, not top-level output"),
    Invariant("SINK_MASK_WIDTH", _inv_sink_mask_width,
              "sink_mask width vs register/sink byte width"),
]


def register_invariant(invariant: Invariant) -> None:
    """Add a wiring invariant to the registry. THE extensibility point — a new
    check is one of these, no other code change (proved by the registry test)."""
    if not isinstance(invariant, Invariant):
        raise TypeError("register_invariant expects an Invariant")
    INVARIANTS.append(invariant)


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #

def lint_parity_inputs(
    case_config: Any,
    *,
    window: tuple[int, int] | None = None,
    declared_inputs: Any = None,
    observed_spec: Mapping[str, Any] | None = None,
    sink_spec: Mapping[str, Any] | None = None,
    invariants: Sequence[Invariant] | None = None,
) -> LintReport:
    """Run the wiring-invariant registry over the parity / self-check INPUTS.

    Pure pre-flight: no symex, no verdict — a declarative checker over the same
    descriptor fields ``check_emit_self_consistency`` / ``check_parity_vectors``
    consume. Returns a :class:`LintReport` (findings + max_level); NON-BLOCKING —
    the caller decides what an ERROR means.

    * ``declared_inputs`` — the emit input set (a sequence of names, or a mapping
      ``name -> width_bytes``). When omitted, falls back to ``case_config.inputs``.
    * ``sink_spec`` / ``observed_spec`` — the descriptors documented at module top.

    An invariant that cannot evaluate its subject (a missing descriptor) emits an
    INFO/WARN with the reason — never a silent pass (A8 check 4)."""
    if declared_inputs is None and case_config is not None:
        declared_inputs = getattr(case_config, "inputs", None)
    if window is None and case_config is not None:
        window = getattr(case_config, "window", None)
    ctx = LintContext(
        case_config=case_config,
        window=window,
        declared_inputs=_normalize_declared(declared_inputs),
        observed_spec=observed_spec,
        sink_spec=sink_spec,
    )
    registry = list(invariants) if invariants is not None else list(INVARIANTS)
    findings: list[LintFinding] = []
    for inv in registry:
        try:
            for f in inv.check(ctx):
                findings.append(f)
        except Exception as exc:  # an invariant bug must surface, never crash drive
            findings.append(LintFinding(
                "WARN", inv.code,
                f"invariant '{inv.code}' raised {exc!r} while checking the wiring "
                "— treated as unevaluated, not clean.",
                "report this lint invariant defect; the wiring it checks is "
                "UNVERIFIED, not confirmed."))
    return LintReport(findings=tuple(findings))


def lint_case_config(
    case_config: Any,
    *,
    seed_values: Mapping[str, Any] | None = None,
    observed_spec: Mapping[str, Any] | None = None,
    sink_spec: Mapping[str, Any] | None = None,
    invariants: Sequence[Invariant] | None = None,
) -> LintReport:
    """Convenience wrapper: lint a :class:`CaseConfig` (+ optional runner-side
    seed/observed/sink descriptors). Derives ``declared_inputs`` and ``window``
    from the config, and folds a bare ``seed_values`` into the ``sink_spec`` so a
    caller that only has the seed map need not hand-build the descriptor."""
    if seed_values is not None:
        merged = dict(sink_spec or {})
        merged.setdefault("seed_values", seed_values)
        sink_spec = merged
    return lint_parity_inputs(
        case_config,
        window=getattr(case_config, "window", None),
        declared_inputs=getattr(case_config, "inputs", None),
        observed_spec=observed_spec,
        sink_spec=sink_spec,
        invariants=invariants,
    )


__all__ = [
    "LintFinding",
    "LintReport",
    "LintContext",
    "Invariant",
    "INVARIANTS",
    "register_invariant",
    "lint_parity_inputs",
    "lint_case_config",
]
