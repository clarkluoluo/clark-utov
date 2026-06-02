"""Transformation-chain length-consistency check.

Background — the reference target's Round 5 retro: a transformation chain
``input(32 chars) → decoded(21 bytes) → compressed(16 bytes)`` was
quietly accepted even though no clean relation explains 32→21 or
21→16. The mismatch was a real structural anomaly (wrong intermediate
representation chosen); spotting it was a side-comment from the
agent, not a framework check.

This module turns the side-comment into a primitive. Given an
ordered chain of nodes with declared lengths, every adjacent pair
must satisfy one of a small set of *explainable* relations. Anything
else is flagged ``length_mismatch_unexplained`` and surfaced on the
envelope.

Explainable relations (a, b are positive lengths):

  - ``equal``                — a == b
  - ``integer_multiple``     — a == k*b or b == k*a, k integer >= 1
  - ``hex_two_to_one``       — a == 2*b or b == 2*a (hex string ↔ bytes)
  - ``base64_four_to_three`` — ratio 4/3 in either direction
  - ``explicit_ratio``       — caller declared a ``ratio`` (``num/den``)
  - ``explicit_delta``       — caller declared a ``delta`` (a - b == delta
                                or b - a == delta)
  - ``allowed_relation``     — caller's per-edge ``allowed: [...]`` list
                                contains a literal match

Sibling to :mod:`engine.m1_success_audit` (dimension coverage) and
:mod:`engine.m3_bypass_block` (cross-method invariance). All three
are "automate the human pattern-match".

Independent toggle: ``UTOV_LENGTH_CHAIN=off|0|false|no``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LengthChainConfig:
    enabled: bool = True
    # Allow at most a +/- this much absolute slack when matching
    # ``explicit_delta`` or padding-aware comparisons (0 by default —
    # padding must be declared explicitly).
    default_delta_slack: int = 0
    # Maximum integer-multiple to consider when fitting
    # ``integer_multiple``. Beyond this the relation is treated as
    # unexplained — a 1:100 jump is almost never a real transform.
    max_integer_multiple: int = 16

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "LengthChainConfig":
        src = env if env is not None else os.environ
        cfg = cls()
        flag = (src.get("UTOV_LENGTH_CHAIN") or "").strip().lower()
        if flag in ("off", "0", "false", "no"):
            cfg.enabled = False
        m = src.get("UTOV_LENGTH_CHAIN_MAX_MULTIPLE")
        if m is not None:
            try:
                cfg.max_integer_multiple = int(m)
            except ValueError:
                pass
        return cfg


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EdgeReport:
    """One adjacent-pair report.

    ``relation`` is the canonical name of the relation that matched;
    when no relation matched it is ``"unexplained"``.
    """
    from_name:   str
    to_name:     str
    from_length: int
    to_length:   int
    relation:    str
    matched:     bool
    tried:       tuple[str, ...]
    note:        str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_name":   self.from_name,
            "to_name":     self.to_name,
            "from_length": self.from_length,
            "to_length":   self.to_length,
            "relation":    self.relation,
            "matched":     self.matched,
            "tried":       list(self.tried),
            "note":        self.note,
        }


@dataclass(frozen=True, slots=True)
class LengthChainResult:
    ok: bool
    edges: tuple[EdgeReport, ...]
    unexplained_edges: tuple[EdgeReport, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok":                 self.ok,
            "edges":              [e.to_dict() for e in self.edges],
            "unexplained_edges":  [e.to_dict() for e in self.unexplained_edges],
            "notes":              list(self.notes),
        }


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------


def check_length_chain(
    chain: Iterable[dict[str, Any]],
    *,
    cfg: LengthChainConfig | None = None,
) -> LengthChainResult | None:
    """Run the consistency check on ``chain``. Returns ``None`` if
    the module is disabled or the chain has fewer than 2 nodes.

    Each node in ``chain`` may declare on the *outgoing* edge:
      - ``expected_ratio``  — string ``"num/den"`` (e.g. ``"4/3"``)
      - ``expected_delta``  — int, ``next.length == this.length + delta``
                              (negative allowed)
      - ``allowed``         — list of relation names from the
                              recognised vocabulary that the caller
                              wants to whitelist for this edge
    """
    cfg = cfg or LengthChainConfig.from_env()
    if not cfg.enabled:
        return None
    nodes = [n for n in chain if isinstance(n, dict) and "length" in n]
    if len(nodes) < 2:
        return None
    edges: list[EdgeReport] = []
    notes: list[str] = []
    for a, b in zip(nodes, nodes[1:]):
        edge = _check_edge(a, b, cfg=cfg)
        edges.append(edge)
    unexplained = tuple(e for e in edges if not e.matched)
    if unexplained:
        notes.append(
            f"{len(unexplained)} edge(s) unexplained — possible wrong "
            f"intermediate representation chosen on the chain"
        )
    return LengthChainResult(
        ok=not unexplained,
        edges=tuple(edges),
        unexplained_edges=unexplained,
        notes=tuple(notes),
    )


def _check_edge(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    cfg: LengthChainConfig,
) -> EdgeReport:
    la = int(a["length"])
    lb = int(b["length"])
    fname = str(a.get("name") or "<a>")
    tname = str(b.get("name") or "<b>")
    tried: list[str] = []
    note = ""

    expected_ratio = a.get("expected_ratio")  # string "num/den"
    expected_delta = a.get("expected_delta")
    allowed: list[str] = list(a.get("allowed") or [])

    def _hit(relation: str, n: str = "") -> EdgeReport:
        return EdgeReport(
            from_name=fname, to_name=tname,
            from_length=la, to_length=lb,
            relation=relation, matched=True,
            tried=tuple(tried), note=n,
        )

    # 1. equal
    tried.append("equal")
    if la == lb and (not allowed or "equal" in allowed):
        return _hit("equal")

    # 2. explicit_ratio (caller-declared)
    if isinstance(expected_ratio, str) and "/" in expected_ratio:
        tried.append("explicit_ratio")
        try:
            num, den = expected_ratio.split("/", 1)
            num_i, den_i = int(num), int(den)
            if num_i > 0 and den_i > 0:
                if la * den_i == lb * num_i:
                    return _hit("explicit_ratio",
                                f"a/b == {expected_ratio}")
        except ValueError:
            pass

    # 3. explicit_delta
    if isinstance(expected_delta, int):
        tried.append("explicit_delta")
        if lb == la + expected_delta:
            return _hit("explicit_delta", f"b == a + {expected_delta}")
        if abs((lb - la) - expected_delta) <= cfg.default_delta_slack:
            return _hit("explicit_delta",
                        f"|b-a - delta| <= {cfg.default_delta_slack}")

    # 4. hex_two_to_one
    tried.append("hex_two_to_one")
    if (la == 2 * lb or lb == 2 * la) and (not allowed or "hex_two_to_one" in allowed):
        return _hit("hex_two_to_one")

    # 5. base64_four_to_three (loose — pads up to +/-2)
    tried.append("base64_four_to_three")
    if _is_base64_ratio(la, lb) and (not allowed or "base64_four_to_three" in allowed):
        return _hit("base64_four_to_three")

    # 6. integer_multiple
    tried.append("integer_multiple")
    big, small = (la, lb) if la >= lb else (lb, la)
    if small > 0 and big % small == 0:
        k = big // small
        if 2 <= k <= cfg.max_integer_multiple and (not allowed or "integer_multiple" in allowed):
            return _hit("integer_multiple", f"k={k}")

    # 7. allowed_relation literal match (caller forced an unusual ratio)
    if "literal_explicit" in allowed:
        tried.append("literal_explicit")
        return _hit("literal_explicit",
                    "caller explicitly accepted any literal mismatch")

    note = (
        f"no explainable relation between {la}→{lb}; tried "
        f"{','.join(tried)}. Possible wrong intermediate representation."
    )
    return EdgeReport(
        from_name=fname, to_name=tname,
        from_length=la, to_length=lb,
        relation="unexplained", matched=False,
        tried=tuple(tried), note=note,
    )


def _is_base64_ratio(a: int, b: int) -> bool:
    """Strict base64 4-to-3 check.

    Real base64 encoding pads to a multiple of 4, so the encoded
    side must be ``big % 4 == 0`` AND exactly ``4 * ceil(raw/3)``.
    A naive ratio with arbitrary slack falsely matches mismatched
    chains like 21→16 (off by 1 under slack=2) — exactly the
    the reference target footgun this module is supposed to catch.
    """
    big, small = (a, b) if a >= b else (b, a)
    if small <= 0:
        return False
    if big % 4 != 0:
        return False
    raw_blocks = (small + 2) // 3   # ceil(small / 3)
    return big == 4 * raw_blocks


# ---------------------------------------------------------------------------
# Param extraction + rendering
# ---------------------------------------------------------------------------


def check_chains_in_params(
    params: dict[str, Any] | None,
    *,
    cfg: LengthChainConfig | None = None,
) -> list[LengthChainResult]:
    """Walk ``params`` for any ``length_chain`` field and run the
    check on each. Each found chain is a list-of-dicts with at least
    ``length`` per node."""
    cfg = cfg or LengthChainConfig.from_env()
    out: list[LengthChainResult] = []
    if not cfg.enabled or params is None:
        return out
    _walk_chains(params, cfg, out)
    return out


def _walk_chains(
    node: Any,
    cfg: LengthChainConfig,
    out: list[LengthChainResult],
    *,
    depth: int = 5,
) -> None:
    if depth <= 0 or node is None:
        return
    if isinstance(node, dict):
        if "length_chain" in node:
            chain = node["length_chain"]
            if isinstance(chain, list):
                result = check_length_chain(chain, cfg=cfg)
                if result is not None:
                    out.append(result)
                    # Annotate the chain dict so downstream consumers
                    # can read the per-edge breakdown without re-running.
                    node["length_chain_report"] = result.to_dict()
        for v in node.values():
            _walk_chains(v, cfg, out, depth=depth - 1)
    elif isinstance(node, list):
        for v in node:
            _walk_chains(v, cfg, out, depth=depth - 1)


def render_length_chain_alert(results: list[LengthChainResult]) -> str | None:
    if not results:
        return None
    bad = [r for r in results if not r.ok]
    if not bad:
        return None
    parts = []
    for r in bad:
        for e in r.unexplained_edges:
            parts.append(f"{e.from_name}({e.from_length})→{e.to_name}({e.to_length})")
    return "[LENGTH-CHAIN/UNEXPLAINED] " + "; ".join(parts)
