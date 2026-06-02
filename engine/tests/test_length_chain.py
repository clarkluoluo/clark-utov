"""Transformation-chain length-consistency check — acceptance tests.

Headline scenario: 32 chars → 21 bytes → 16 bytes (the reference target's
Round 5 misalignment). No relation explains 32→21 or 21→16; both
edges must be flagged ``length_mismatch_unexplained``.
"""

from __future__ import annotations

from engine.discipline_wrapper import DisciplineWrapper
from engine.length_chain_check import (
    LengthChainConfig,
    check_chains_in_params,
    check_length_chain,
)
from engine.methodology import MethodologyConfig


# ---------------------------------------------------------------------------
# Module-level
# ---------------------------------------------------------------------------


def test_reference_target_32_21_16_chain_both_edges_unexplained():
    chain = [
        {"name": "input_string", "length": 32},
        {"name": "decoded",      "length": 21},
        {"name": "compressed",   "length": 16},
    ]
    r = check_length_chain(chain, cfg=LengthChainConfig())
    assert r is not None
    assert r.ok is False
    assert len(r.unexplained_edges) == 2
    names = {(e.from_name, e.to_name) for e in r.unexplained_edges}
    assert ("input_string", "decoded")    in names
    assert ("decoded",      "compressed") in names


def test_hex_two_to_one_passes():
    chain = [
        {"name": "hex_str", "length": 64},
        {"name": "bytes",   "length": 32},
    ]
    r = check_length_chain(chain, cfg=LengthChainConfig())
    assert r is not None
    assert r.ok is True
    assert r.edges[0].relation == "hex_two_to_one"


def test_integer_multiple_passes():
    chain = [
        {"name": "blocks", "length": 5},
        {"name": "bytes",  "length": 80},   # 16x
    ]
    r = check_length_chain(chain, cfg=LengthChainConfig(max_integer_multiple=16))
    assert r is not None
    assert r.ok is True
    assert r.edges[0].relation == "integer_multiple"


def test_integer_multiple_beyond_cap_unexplained():
    chain = [
        {"name": "x", "length": 1},
        {"name": "y", "length": 100},
    ]
    r = check_length_chain(chain, cfg=LengthChainConfig(max_integer_multiple=16))
    assert r is not None
    assert r.ok is False


def test_base64_4_to_3_passes():
    chain = [
        {"name": "b64", "length": 12},
        {"name": "raw", "length": 9},
    ]
    r = check_length_chain(chain, cfg=LengthChainConfig())
    assert r is not None
    assert r.ok is True
    assert r.edges[0].relation == "base64_four_to_three"


def test_explicit_ratio_unlocks_an_otherwise_unexplained_edge():
    chain = [
        {"name": "x", "length": 32, "expected_ratio": "32/21"},
        {"name": "y", "length": 21},
    ]
    r = check_length_chain(chain, cfg=LengthChainConfig())
    assert r is not None
    assert r.ok is True
    assert r.edges[0].relation == "explicit_ratio"


def test_explicit_delta_passes():
    chain = [
        {"name": "p", "length": 21, "expected_delta": -5},
        {"name": "q", "length": 16},
    ]
    r = check_length_chain(chain, cfg=LengthChainConfig())
    assert r is not None
    assert r.ok is True
    assert r.edges[0].relation == "explicit_delta"


def test_equal_passes():
    chain = [
        {"name": "a", "length": 32},
        {"name": "b", "length": 32},
    ]
    r = check_length_chain(chain, cfg=LengthChainConfig())
    assert r is not None
    assert r.ok is True
    assert r.edges[0].relation == "equal"


def test_returns_none_for_short_chain():
    assert check_length_chain([{"name": "x", "length": 1}],
                              cfg=LengthChainConfig()) is None
    assert check_length_chain([], cfg=LengthChainConfig()) is None


def test_env_toggle_off_returns_none():
    cfg = LengthChainConfig.from_env({"UTOV_LENGTH_CHAIN": "off"})
    assert cfg.enabled is False
    r = check_length_chain([{"name": "x", "length": 1},
                            {"name": "y", "length": 17}], cfg=cfg)
    assert r is None


def test_check_chains_in_params_finds_nested_chain():
    params = {
        "report": {
            "length_chain": [
                {"name": "input", "length": 32},
                {"name": "decoded", "length": 21},
                {"name": "compressed", "length": 16},
            ],
        },
    }
    out = check_chains_in_params(params, cfg=LengthChainConfig())
    assert len(out) == 1
    assert out[0].ok is False
    # Side-effect: chain dict got a report attached.
    assert "length_chain_report" in params["report"]


# ---------------------------------------------------------------------------
# Wrapper integration
# ---------------------------------------------------------------------------


def test_wrapper_emits_alert_on_unexplained_chain():
    wrapper = DisciplineWrapper(config=MethodologyConfig())
    params = {
        "length_chain": [
            {"name": "input",   "length": 32},
            {"name": "decoded", "length": 21},
            {"name": "comp",    "length": 16},
        ],
    }
    _, env = wrapper.step("submit_finding", params, lambda m, p: None)
    assert env.length_chain
    assert env.length_chain[0]["ok"] is False
    assert any("LENGTH-CHAIN/UNEXPLAINED" in a for a in env.alerts)


def test_wrapper_silent_on_clean_chain():
    wrapper = DisciplineWrapper(config=MethodologyConfig())
    params = {
        "length_chain": [
            {"name": "hex", "length": 64},
            {"name": "raw", "length": 32},
        ],
    }
    _, env = wrapper.step("submit_finding", params, lambda m, p: None)
    assert env.length_chain
    assert env.length_chain[0]["ok"] is True
    assert all("LENGTH-CHAIN/UNEXPLAINED" not in a for a in env.alerts)
