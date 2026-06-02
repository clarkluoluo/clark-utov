"""Export-filename honesty gate — dev-export-filename-honesty-spec.md.

A filename must not assert a strong closure (``output_provenance_confirmed.json``)
the run content does not support. utov never rewrites the consumer's filename, but
it refuses to lie SILENTLY: it WARNs loudly AND stamps an explicit
``filename_verdict_mismatch`` field into the export. Reuses the closure-evidence
layering verdict (only ORACLE / ``algorithm_closed`` supports a strong claim);
negation prefixes (unconfirmed) are exempt; a token-free name is untouched.
Synthetic shapes only — zero case-specific knowledge.
"""

from __future__ import annotations

import logging

from engine.cvd import (
    Candidate,
    CandidateGenerator,
    CvdOutcome,
    Registry,
    Verdict,
    Verifier,
    VStatus,
    export_gap_map,
    run_cvd,
    run_cvd_collect_to_json,
)
from engine.export_stamp import (
    CLOSURE_CLAIM_TOKENS,
    filename_closure_claims,
    filename_verdict_mismatch,
    load_stamped_json,
    result_supports_strong_closure,
    safe_export_name,
)

_TS = "2026-06-02T00:00:00Z"


# --- a tiny consumer-facing collect run (NOT oracle-closed) ----------------- #

class _Gen(CandidateGenerator):
    name = "g"; version = "1"; owner = "test"; kind = "x"

    def generate(self, state):
        return [Candidate("ok", 0x10, "s", "c"),
                Candidate("gap", 0x20, "s", "needs tool")]


class _Ver(Verifier):
    name = "v"; version = "1"; owner = "test"

    def applies(self, c, state):
        return c.kind in ("ok", "gap")

    def verify(self, c, state):
        if c.kind == "ok":
            return Verdict(VStatus.CONFIRMED, evidence={"ok": True})
        return Verdict(VStatus.TERMINAL, terminal_kind="needs_tool",
                       reason="gap", capability_request="register tool X")


def _collect_result():
    from engine.types import Instruction
    reg = Registry().register(_Gen()).register(_Ver())
    return run_cvd([Instruction(0, 0x1000, b"\x00\x00\x00\x00", "nop", {}, {})],
                   b"\x00", registry=reg, collect_extensions=True)


# A payload that carries an ORACLE-level closure classification (honest "confirmed").
_ORACLE_CLOSURE = {
    "kind": "closure_classification",
    "closure_level": "oracle",
    "label": "algorithm_closed_form",
    "algorithm_closed": True,
    "output_sink_confirmed": True,
    "provenance_closed": True,
    "parity_exact": True,
}
# A NON-oracle closure classification (window-local parity-EXACT, the trap shape).
_LOCAL_CLOSURE = {
    "kind": "closure_classification",
    "closure_level": "structural",
    "label": "local_formula",
    "algorithm_closed": False,
    "trap_state": "PSEUDO_CLOSURE_TRAP",
}


# --- token detection -------------------------------------------------------- #

def test_token_set_is_the_data_driven_constant():
    assert CLOSURE_CLAIM_TOKENS == frozenset(
        {"confirmed", "closed", "identified", "oracle", "solved"})


def test_filename_claims_detected_case_insensitive_word_boundary():
    assert filename_closure_claims("output_provenance_confirmed.json") == ["confirmed"]
    assert filename_closure_claims("RESULT_ORACLE_CLOSED.JSON") == ["closed", "oracle"]
    assert filename_closure_claims("foo_identified-v2.json") == ["identified"]
    assert filename_closure_claims("solved.json") == ["solved"]


def test_no_token_filename_has_no_claims():
    assert filename_closure_claims("cvd_gap_map.json") == []
    assert filename_closure_claims("run-out/gap_map.json") == []


def test_negation_prefix_is_exempt():
    # 验收④: unconfirmed / unclosed / not_identified must NOT trigger.
    assert filename_closure_claims("output_unconfirmed.json") == []
    assert filename_closure_claims("unclosed_window.json") == []
    assert filename_closure_claims("not_identified.json") == []
    assert filename_closure_claims("non-oracle.json") == []


def test_substring_inside_a_larger_word_does_not_falsely_claim():
    # word-boundary: "closedness" / "preconfirmedx" carry no boundary → no claim.
    assert filename_closure_claims("closedness.json") == []
    assert filename_closure_claims("identifiedness.json") == []


# --- support judgment reuses the closure layer ------------------------------ #

def test_only_oracle_closed_supports_a_strong_claim():
    assert result_supports_strong_closure({"confirmed": [{"closure": _ORACLE_CLOSURE}]})
    assert not result_supports_strong_closure(
        {"confirmed": [{"closure": _LOCAL_CLOSURE}]})
    assert not result_supports_strong_closure({"outcome": "NEEDS_OBSERVATION"})


# --- the mismatch record ---------------------------------------------------- #

def test_mismatch_record_built_when_name_overclaims():
    payload = {"outcome": "NEEDS_OBSERVATION", "verdict": "needs_observation"}
    rec = filename_verdict_mismatch("output_provenance_confirmed.json", payload)
    assert rec is not None
    assert rec["kind"] == "filename_verdict_mismatch"
    assert rec["filename_claim"] == ["confirmed"]
    assert rec["actual_outcome"] == "NEEDS_OBSERVATION"
    assert rec["claim_supported"] is False


def test_no_mismatch_when_oracle_closed_even_with_strong_name():
    payload = {"outcome": "COLLECTED", "confirmed": [{"closure": _ORACLE_CLOSURE}]}
    assert filename_verdict_mismatch("result_confirmed.json", payload) is None


def test_no_mismatch_for_token_free_name():
    payload = {"outcome": "NEEDS_OBSERVATION"}
    assert filename_verdict_mismatch("cvd_gap_map.json", payload) is None


def test_no_mismatch_for_negated_name():
    payload = {"outcome": "NEEDS_OBSERVATION"}
    assert filename_verdict_mismatch("output_unconfirmed.json", payload) is None


# --- integration through export_gap_map ------------------------------------- #

def test_export_with_lying_filename_stamps_mismatch_and_warns(tmp_path, caplog):
    # 验收①: confirmed-named file + a NEEDS-OBSERVATION-class run (no oracle
    # closure) → WARN-loud + filename_verdict_mismatch inside the export.
    res = _collect_result()
    out = tmp_path / "output_provenance_confirmed.json"
    with caplog.at_level(logging.WARNING, logger="engine.cvd"):
        text = export_gap_map(res, out, ts=_TS)
    assert any("EXPORT FILENAME LIES" in r.message for r in caplog.records)
    _, payload = load_stamped_json(text)
    assert "filename_verdict_mismatch" in payload
    assert payload["filename_verdict_mismatch"]["filename_claim"] == ["confirmed"]
    # not silent: the mismatch is durably written to disk.
    _, on_disk = load_stamped_json(out.read_text())
    assert "filename_verdict_mismatch" in on_disk


def test_export_with_honest_token_free_name_is_unchanged(tmp_path):
    # 验收③: cvd_gap_map.json (no claim token) → behaviour completely unchanged.
    res = _collect_result()
    out = tmp_path / "cvd_gap_map.json"
    text = export_gap_map(res, out, ts=_TS)
    _, payload = load_stamped_json(text)
    assert "filename_verdict_mismatch" not in payload
    assert payload["outcome"] == CvdOutcome.COLLECTED.value


def test_export_with_oracle_closed_run_and_confirmed_name_is_honest(tmp_path):
    # 验收②: confirmed name + a genuinely oracle-closed run → NO mismatch.
    res = _collect_result()
    # inject an oracle-closure classification onto a confirmed window (as the
    # recovery layer does) — the content now backs the strong name.
    res.confirmed = [{"evidence": {"closure": _ORACLE_CLOSURE}}]
    out = tmp_path / "result_confirmed.json"
    text = export_gap_map(res, out, ts=_TS)
    _, payload = load_stamped_json(text)
    assert "filename_verdict_mismatch" not in payload


def test_collect_to_json_filename_gate_fires_end_to_end(tmp_path, caplog):
    # the one-shot consumer entry threads its filename through the same gate.
    from engine.types import Instruction
    reg = Registry().register(_Gen()).register(_Ver())
    with caplog.at_level(logging.WARNING, logger="engine.cvd"):
        res, path = run_cvd_collect_to_json(
            [Instruction(0, 0x1000, b"\x00\x00\x00\x00", "nop", {}, {})], b"\x00",
            work_root=tmp_path / "out", ts=_TS, filename="solved.json", registry=reg)
    assert any("EXPORT FILENAME LIES" in r.message for r in caplog.records)
    _, payload = load_stamped_json(path.read_text())
    assert payload["filename_verdict_mismatch"]["filename_claim"] == ["solved"]


# --- safe_export_name helper ------------------------------------------------ #

def test_safe_export_name_derives_neutral_name_for_non_oracle_run():
    name = safe_export_name({"outcome": "NEEDS_OBSERVATION"})
    assert filename_closure_claims(name) == []          # no strong-claim token
    assert filename_verdict_mismatch(name, {"outcome": "NEEDS_OBSERVATION"}) is None
    assert "needs_observation" in name


def test_safe_export_name_marks_an_oracle_closed_run():
    payload = {"outcome": "COLLECTED", "confirmed": [{"closure": _ORACLE_CLOSURE}]}
    name = safe_export_name(payload)
    assert "oracle_closed" in name
    # round-trip: the derived name passes the gate (honest).
    assert filename_verdict_mismatch(name, payload) is None
