"""capability_request.md §P2-3 / M10 — structured verdict tests."""

from __future__ import annotations

import pytest

from engine.verdict import Verdict, VerdictError, build_verdict, lint_markdown


def test_archival_allowed_iff_no_not_confirmed_or_invalidated():
    v = build_verdict(confirmed=["io_oracle"], known_gaps=["vmp_capture"])
    assert v.archival_allowed is True


def test_not_confirmed_blocks_archival():
    v = build_verdict(
        confirmed=["io_oracle"],
        not_confirmed=["sm3_body_binary"],
    )
    assert v.archival_allowed is False


def test_invalidated_blocks_archival():
    v = build_verdict(invalidated=["hook_b7bb0"])
    assert v.archival_allowed is False


def test_yaml_renders_all_buckets():
    v = build_verdict(
        confirmed=["io_oracle", "implementation_sm3_gmt"],
        not_confirmed=["sm3_body_binary"],
        invalidated=[],
        known_gaps=["vmp_register_capture"],
    )
    y = v.to_yaml()
    assert "confirmed:     [io_oracle, implementation_sm3_gmt]" in y
    assert "not_confirmed: [sm3_body_binary]" in y
    assert "invalidated:   []" in y
    assert "known_gaps:    [vmp_register_capture]" in y
    assert "archival_allowed: false" in y


def test_bare_word_rejected():
    with pytest.raises(VerdictError, match="bare verdict word"):
        build_verdict(confirmed=["确认"])


def test_duplicate_layer_across_buckets_rejected():
    with pytest.raises(VerdictError, match="appears in both"):
        build_verdict(
            confirmed=["io_oracle"],
            invalidated=["io_oracle"],
        )


def test_empty_layer_rejected():
    with pytest.raises(VerdictError, match="empty"):
        build_verdict(confirmed=[""])


def test_lint_markdown_clean_text():
    clean = "## verdict\nconfirmed: io_oracle\n"
    assert lint_markdown(clean) == []


def test_lint_markdown_catches_bare_passed():
    bad = "## §5 verdict\nAll layers passed — see below.\n"
    issues = lint_markdown(bad)
    assert issues
    assert "L2" in issues[0]


def test_lint_markdown_ignores_fenced_blocks():
    text = (
        "## §5 verdict\n"
        "```\n"
        "Layer passed in offline test (this is inside a code fence)\n"
        "```\n"
        "Outside fence: nothing bare here.\n"
    )
    assert lint_markdown(text) == []


def test_to_dict_roundtrips():
    v = build_verdict(confirmed=["a"], known_gaps=["b"])
    d = v.to_dict()
    assert d == {
        "confirmed":     ["a"],
        "not_confirmed": [],
        "invalidated":   [],
        "known_gaps":    ["b"],
    }
