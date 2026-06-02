"""BR-4 §E: BudgetExceeded carries a structured `axis` field, and
`NextAction(kind='raise_budget')` carries `breached_budget` so agent UIs
don't have to parse human-readable strings to know which knob to raise.
"""

from __future__ import annotations

import pytest

from engine.cost import Budget, BudgetExceeded, CostMeter


def test_budget_exceeded_carries_axis_wall_seconds():
    """A wall-time breach must label `axis="wall_seconds"`."""
    b = Budget(max_wall_seconds=0.0)   # immediate breach
    m = CostMeter(b)
    # Force a snapshot to evaluate wall_seconds > 0
    import time
    time.sleep(0.001)
    with pytest.raises(BudgetExceeded) as exc:
        # Any charge triggers _check_budget on the wall-time axis
        m.charge(model="t", input_tokens=1, output_tokens=1)
    assert exc.value.axis == "wall_seconds"


def test_budget_exceeded_carries_axis_usd():
    b = Budget(max_usd=0.0)
    m = CostMeter(b)
    with pytest.raises(BudgetExceeded) as exc:
        # Any non-zero charge crosses $0
        m.charge(model="deepseek-chat", input_tokens=100, output_tokens=100)
    assert exc.value.axis == "usd"


def test_budget_exceeded_carries_axis_tokens():
    b = Budget(max_total_tokens=10)
    m = CostMeter(b)
    with pytest.raises(BudgetExceeded) as exc:
        m.charge(model="t", input_tokens=20, output_tokens=20)
    assert exc.value.axis == "total_tokens"


def test_next_action_carries_breached_budget_field():
    """The NextAction dataclass exposes `breached_budget` for agent UI."""
    from engine.orchestrators.script_mode import NextAction
    n = NextAction(kind="raise_budget", severity="blocker",
                   reason="test", suggested_command="x",
                   breached_budget="wall_seconds")
    assert n.breached_budget == "wall_seconds"
    # And the field is optional / defaults to None for non-budget actions.
    n2 = NextAction(kind="rerun_with_different_inputs",
                    severity="warning", reason="r")
    assert n2.breached_budget is None
