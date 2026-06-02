"""T5 — General input_contract for cross-task references (PLAN §20 v2 §4.2).

Runner-correction (PLAN §20.1, 2026-05-29): the input_contract is no
longer scoped to a "standalone" task carve-out — it is the universal
declaration that *any* reusable task must provide so cross-task
references go through the contract, not through defaults.

Compose-time validation: a caller wiring a contract-bearing task
must supply every declared ``accepts`` slot and every declared
``capabilities`` entry; mismatch fails at compose time, not run time.
"""

from __future__ import annotations

import pytest

from engine.task import (
    ContractMismatchError,
    InputContract,
    validate_contract_compose,
)


# ---------------------------------------------------------------------------
# parse() — well-formed contracts
# ---------------------------------------------------------------------------


def test_parse_full_contract():
    raw = {
        "accepts": ["front_half_impl", "back_half_impl"],
        "produces": ["byte_equal_pass"],
        "capabilities": ["re_execute"],
    }
    c = InputContract.parse(raw)
    assert c.accepts == ("front_half_impl", "back_half_impl")
    assert c.produces == ("byte_equal_pass",)
    assert c.capabilities == ("re_execute",)


def test_parse_empty_contract():
    """An empty contract still exists — its presence is the
    reusability marker, not any particular field content."""
    c = InputContract.parse({})
    assert c.accepts == ()
    assert c.produces == ()
    assert c.capabilities == ()


def test_parse_rejects_non_string_accepts():
    with pytest.raises(ContractMismatchError, match="accepts"):
        InputContract.parse({"accepts": [123]})


def test_parse_rejects_non_object():
    with pytest.raises(ContractMismatchError, match="JSON object"):
        InputContract.parse([1, 2, 3])


# ---------------------------------------------------------------------------
# validate_contract_compose — happy path
# ---------------------------------------------------------------------------


def test_compose_passes_when_every_accept_and_capability_supplied():
    c = InputContract(
        accepts=("a", "b"),
        capabilities=("re_execute",),
    )
    validate_contract_compose(
        c,
        supplied_inputs=["a", "b", "extra"],   # extras are allowed
        supplied_capabilities=["re_execute", "trace"],
    )


def test_compose_passes_for_empty_contract():
    validate_contract_compose(
        InputContract(),
        supplied_inputs=[],
        supplied_capabilities=[],
    )


# ---------------------------------------------------------------------------
# validate_contract_compose — mismatch lists every gap by name
# ---------------------------------------------------------------------------


def test_compose_missing_input_names_the_slot():
    """The whole point of compose-time validation: the error message
    must name the missing slot so the caller knows what to wire."""
    c = InputContract(accepts=("front_half_impl", "back_half_impl"))
    with pytest.raises(ContractMismatchError) as exc:
        validate_contract_compose(
            c,
            supplied_inputs=["front_half_impl"],   # back_half_impl missing
            supplied_capabilities=[],
            callee_id="merge_check",
        )
    assert "merge_check" in str(exc.value)
    assert "back_half_impl" in str(exc.value)


def test_compose_missing_capability_names_it():
    c = InputContract(capabilities=("re_execute", "memregion_watch"))
    with pytest.raises(ContractMismatchError) as exc:
        validate_contract_compose(
            c,
            supplied_inputs=[],
            supplied_capabilities=["re_execute"],
            callee_id="check",
        )
    assert "memregion_watch" in str(exc.value)


def test_compose_lists_every_gap_not_just_first():
    c = InputContract(
        accepts=("a", "b"),
        capabilities=("cap1", "cap2"),
    )
    with pytest.raises(ContractMismatchError) as exc:
        validate_contract_compose(
            c,
            supplied_inputs=[],   # a, b both missing
            supplied_capabilities=[],   # both caps missing
            callee_id="t",
        )
    msg = str(exc.value)
    assert "a" in msg and "b" in msg
    assert "cap1" in msg and "cap2" in msg
