"""Input contract for cross-task references (PLAN §20 / v2 §4.2).

The runner-correction (PLAN §20.1, 2026-05-29) collapsed the old
"standalone task" carve-out: every task uniformly declares runner
usage, and the input_contract is now a **general reusability
declaration**.  Any task that is referenced by another task's child
list or call site must carry an :class:`InputContract`.

A contract names three things:

  * ``accepts``       — the names of inputs the task expects from the caller
                        (e.g. ``("front_half_implementation", "back_half_implementation")``).
  * ``produces``      — the names of outputs the task hands back
                        (e.g. ``("byte_equal_passing_inputs",)``).
  * ``capabilities``  — abilities the task assumes are present
                        (runner abilities like ``re_execute`` or
                        external services).

Compose-time validation: when a caller wires a contract-bearing task,
:func:`validate_contract_compose` checks that every declared ``accepts``
slot is supplied and every declared ``capabilities`` entry is present
on the caller side.  Mismatch fails at compose time — not at run time —
so a wrong-wiring shows up before any runner ticks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


class ContractMismatchError(Exception):
    """Raised by :func:`validate_contract_compose` when caller-supplied
    inputs / capabilities do not satisfy the callee task's contract."""


@dataclass(frozen=True)
class InputContract:
    """One task's reusability contract.

    All three tuples may be empty — an empty contract still
    *exists* and signals "this task is callable", but supplies nothing.
    The presence of the contract is itself the reusability marker
    (:attr:`TaskSpec.is_reusable`).
    """

    accepts: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()

    @classmethod
    def parse(cls, raw: Any, *, source: str = "<dict>") -> "InputContract":
        """Parse a JSON contract object.

        ``raw`` may be ``None`` for "no contract"; the loader handles
        that case before calling :meth:`parse`, so ``parse`` itself
        requires a dict.
        """
        if raw is None:
            raise ContractMismatchError(
                f"{source}: input_contract is null — caller should not call "
                f"parse() with None"
            )
        if not isinstance(raw, dict):
            raise ContractMismatchError(
                f"{source}: input_contract must be a JSON object"
            )

        def _str_tuple(field_name: str) -> tuple[str, ...]:
            v = raw.get(field_name, []) or []
            if not isinstance(v, list) or any(
                not isinstance(x, str) or not x for x in v
            ):
                raise ContractMismatchError(
                    f"{source}: input_contract.{field_name} must be a list of "
                    f"non-empty strings"
                )
            return tuple(v)

        return cls(
            accepts=_str_tuple("accepts"),
            produces=_str_tuple("produces"),
            capabilities=_str_tuple("capabilities"),
        )


def validate_contract_compose(
    contract: InputContract,
    *,
    supplied_inputs: Iterable[str],
    supplied_capabilities: Iterable[str],
    callee_id: str = "<callee>",
) -> None:
    """Compose-time check.

    Raises :class:`ContractMismatchError` listing every gap.  The
    explicit gap list is what makes the error useful at debug time —
    saying "contract mismatch" without naming the missing slot is
    what the v2 rule exists to prevent.
    """
    supplied_in = set(supplied_inputs)
    supplied_cap = set(supplied_capabilities)
    missing_inputs = [a for a in contract.accepts if a not in supplied_in]
    missing_caps = [c for c in contract.capabilities if c not in supplied_cap]
    if not missing_inputs and not missing_caps:
        return
    parts: list[str] = []
    if missing_inputs:
        parts.append(f"missing inputs: {missing_inputs}")
    if missing_caps:
        parts.append(f"missing capabilities: {missing_caps}")
    raise ContractMismatchError(
        f"task '{callee_id}' contract not satisfied at compose time — "
        + "; ".join(parts)
    )
