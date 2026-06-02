"""Core package base: CoreConfig, module constants, and helpers."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..conformance import (
    ConformanceReport,
    require_pass_or_die,
    run_conformance,
    write_report,
)
from ..hyp_tree import HypNode, HypTree
from ..runner_client import (
    JsonlTraceReader,
    RunnerAdapter,
    TraceReader,
    UnidbgTextTraceReader,
)
from ..store import (
    WorkDir, _now_iso,
    archive_subtree as _archive_subtree,
    log_intervention as _log_intervention,
    open_findings_db, open_hypotheses_db,
    read_interventions as _read_interventions,
    read_payload,
)
from ..stages import (
    s0_5_normalize, s1_segment, s1b_fingerprint, s2_dedupe,
    s3_triton, s4_slice, s5_simplify, s6_taint,
)
from ..types import Instruction, TargetMeta
from ..verifier import Verifier



@dataclass
class CoreConfig:
    work_root: Path
    target_meta: TargetMeta
    input_hash: str
    driver_mode: str
    run_id: str | None = None
    new_run: bool = False
    # capability_request.md §P0-2: extra PC bands the runner should
    # trace beyond [algo_entry_pc, algo_exit_pc]. Each entry is
    # ``(start_pc, end_pc)``; both ints. End is exclusive. Empty by
    # default — only set when the agent has identified an additional
    # window (e.g. main-VMP 0x32302c..0x325708 for the reference target).
    extra_trace_windows: tuple[tuple[int, int], ...] = ()


_STAGES = {
    "s0_5": s0_5_normalize,   # regs_write reconstruction (additive normalizer)
    "s1":  s1_segment,
    "s1b": s1b_fingerprint,
    "s2":  s2_dedupe,
    "s3":  s3_triton,
    "s4":  s4_slice,
    "s5":  s5_simplify,
    # forward taint propagation — opt-in (needs ctx['taint_sources']); NOT in the
    # default run_pipeline order. Distinct key from "s6" (the LLM hypothesis loop).
    "s6_taint": s6_taint,
}


# ── 0527 BUG_REPORT-7 §B + §J.5: algorithm-template table -----------------
# Shared by `verify_and_promote_algorithm_templates` and `recompute_algorithm_fits`.
#
# Per-entry fields:
#   anchors             : list of anchor names (idiom subjects + fingerprint names)
#   min_unique_anchors  : structural-fit threshold (default rule: 4 of N)
#   confidence_override : optional. Replaces the default `0.50 + 0.35 * coverage`
#                         formula. Set on templates whose anchors are STRONG
#                         enough that a single hit identifies the algorithm
#                         (zero false-positive rate). AES.Te0[*] constants are
#                         the canonical case — they have no natural-number
#                         collision outside an AES implementation, unlike
#                         SHA-256.h0 (`0x6a09e667`) which sometimes shows up in
#                         non-hash code. See BUG_REPORT-7 §B for the FP-rate
#                         argument.
#   reference_impl      : §J.5 — canonical Python implementation. `callable`
#                         is a lambda string the agent can `eval` to get a
#                         reference. `unknowns` lists parameters that must be
#                         recovered (key, iv, etc.) before the reference can
#                         run; SHA-* have none.
#   io_vectors          : §C — canonical (input, expected_output) pairs from
#                         NIST/FIPS test-vector annexes. Used by the duck-
#                         typed IO-equivalence runtime to verify a structural
#                         match behaviorally. Leave empty/None for keyed
#                         algorithms whose key has not been recovered yet
#                         (e.g. AES without a key-recovery step).
ALGORITHM_TEMPLATES: dict[str, dict[str, Any]] = {
    "SHA-256": {
        "anchors": [
            "SHA256.Sigma0", "SHA256.Sigma1",
            "SHA256.sigma0", "SHA256.sigma1",
            "SHA256.h0", "SHA256.h1", "SHA256.h2", "SHA256.h3",
            "SHA256.h4", "SHA256.h5", "SHA256.h6", "SHA256.h7",
        ],
        "min_unique_anchors": 4,
        "reference_impl": {
            "import":    "import hashlib",
            "callable":  "lambda data, *_: hashlib.sha256(data).digest()",
            "signature": "f(input_bytes) -> 32_bytes",
            "unknowns":  [],
        },
        "io_vectors": [
            # FIPS 180-4 Appx B.1: SHA-256("abc")
            {"input_hex":    "616263",
             "expected_hex": "ba7816bf8f01cfea414140de5dae2223"
                             "b00361a396177a9cb410ff61f20015ad"},
        ],
    },
    "SHA-512": {
        "anchors": [
            "SHA512.Sigma0", "SHA512.Sigma1",
            "SHA512.sigma0", "SHA512.sigma1",
            "SHA512.h0", "SHA512.h1", "SHA512.h2", "SHA512.h3",
            "SHA512.h4", "SHA512.h5", "SHA512.h6", "SHA512.h7",
        ],
        "min_unique_anchors": 4,
        "reference_impl": {
            "import":    "import hashlib",
            "callable":  "lambda data, *_: hashlib.sha512(data).digest()",
            "signature": "f(input_bytes) -> 64_bytes",
            "unknowns":  [],
        },
        "io_vectors": [
            # FIPS 180-4 Appx C.1: SHA-512("abc")
            {"input_hex":    "616263",
             "expected_hex": "ddaf35a193617abacc417349ae20413112e6fa4e"
                             "89a97ea20a9eeee64b55d39a2192992a274fc1a8"
                             "36ba3c23a3feebbd454d4423643ce80e2a9ac94f"
                             "a54ca49f"},
        ],
    },
    "AES": {
        # 0527 BUG_REPORT-7 §B. AES.Te0[0..3] + AES.sbox_bytes[*] cover both
        # T-table-driven and byte-array-S-box implementations. min=1 is safe
        # because Te0 constants are unique to AES.
        "anchors": [
            "AES.Te0[0]", "AES.Te0[1]", "AES.Te0[2]", "AES.Te0[3]",
            "AES.sbox_bytes[0..3]", "AES.sbox_bytes[4..7]",
            "AES.inv_sbox_bytes",
        ],
        "min_unique_anchors":  1,
        "confidence_override": 0.85,
        "reference_impl": {
            "import": (
                "from cryptography.hazmat.primitives.ciphers "
                "import Cipher, algorithms, modes"
            ),
            "callable": (
                "lambda pt, key, iv: "
                "(lambda c: c.update(pt) + c.finalize())"
                "(Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor())"
            ),
            "signature": (
                "f(plaintext_bytes, key=<16|24|32 B>, iv=<16 B>) "
                "-> ciphertext_bytes"
            ),
            "unknowns": ["key", "iv", "mode"],
        },
        # AES is keyed: no general (input → output) vector without key.
        "io_vectors": [],
    },
}


# Finding-kind vocabulary (dev-closure-evidence-layering-trap-state-spec, task 7).
# The structural-fingerprint + single-vector-IO matcher below is a PRE-CLOSURE
# ALGORITHM GUESS — a recognised primitive (SHA/AES/MD5/…) is NOT a closed algorithm
# until WHOLE-CASE oracle closure (output_sink_confirmed && provenance_closed &&
# parity_exact). So it emits ``algorithm_hyp`` (algorithm HYPOTHESIS — reads as
# "待验", not "已识别"), carrying an explicit local-closure-trap marker. The strong
# word ``algorithm_identified`` is RESERVED for whole-case oracle closure and is NOT
# emitted by this structural matcher (task 7③: "解题过程中相当长一段时间只能是
# algorithm_hyp"). Readers accept BOTH kinds (a real oracle-closed run may carry the
# strong kind); the matcher only ever produces the hyp kind.
ALGORITHM_HYP_KIND = "algorithm_hyp"
ALGORITHM_IDENTIFIED_KIND = "algorithm_identified"   # reserved: whole-case closure
# Both kinds an algorithm-result reader must accept (the hyp + the reserved strong).
ALGORITHM_RESULT_KINDS = (ALGORITHM_HYP_KIND, ALGORITHM_IDENTIFIED_KIND)


def _algorithm_hyp_trap(io_result: dict[str, Any]) -> dict[str, Any]:
    """The mandatory local-closure-trap marker for a pre-oracle-closure algorithm
    hypothesis (task 7② / task 1).

    A recognised primitive identified by structural fingerprint + a single-vector IO
    test is the SAME class of false closure as a window constant: a structural signal
    masquerading as an algorithm closure. Mark it LOCAL_CLOSURE_ONLY (primitive shape)
    so a reader sees at a glance "this is a hypothesis, not a final algorithm". Even a
    PASSED single-vector IO test does NOT clear the trap — whole-case oracle closure
    needs the output sink confirmed + provenance closed + MULTI-input parity."""
    from ..closure_classification import (
        LABEL_LOCAL_FORMULA,
        ClosureLevel,
        TrapState,
    )
    return {
        "trap_state": TrapState.LOCAL_CLOSURE_ONLY.value,
        "closure_level": ClosureLevel.STRUCTURAL.value,
        "label": LABEL_LOCAL_FORMULA,
        "algorithm_closed": False,
        "is_primitive": True,
        "reason": (
            "PRE-ORACLE-CLOSURE primitive hypothesis — identified by structural "
            "fingerprint" + (
                " + a single-vector IO match" if io_result.get("status") == "passed"
                else "") + ". NOT a closed algorithm: whole-case oracle closure needs "
            "the output sink confirmed, provenance closed, AND multi-input parity "
            "EXACT. Treat as a candidate to verify, not a final answer."),
        "next_step": (
            "confirm the output sink + anchor provenance + run multi-input parity "
            "(independent side output-diverse) before calling the algorithm identified"),
    }


def _run_io_equivalence(
    runner: RunnerAdapter,
    target_meta: TargetMeta,
    algo: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """0527 BUG_REPORT-7 §C: behavioral verification of a structural match.

    Replaces the dead class-name string match in the original implementation.
    Duck-types the runner: if it implements `rerun(bytes, observe_points)`,
    invoke it with the template's IO vector and compare bytes. Single-input
    rerun is sufficient — the multi-input "protocol not yet wired" excuse
    was self-justifying gatekeeping.

    Returns a dict with `status` (passed/failed/skipped/errored) + `detail`.
    Never raises — failures degrade to a status string.
    """
    vectors = spec.get("io_vectors") or []
    if not vectors:
        return {
            "status": "skipped",
            "detail": f"no IO vector for {algo} (keyed algorithm or no canonical test pair)",
        }

    # Confirm the runner actually implements rerun. NullRunnerAdapter / File
    # mode inherit a base method that raises NotImplementedError.
    rerun_fn = getattr(runner, "rerun", None)
    if rerun_fn is None or not callable(rerun_fn):
        return {"status": "skipped", "detail": "runner has no rerun() method"}

    vec = vectors[0]   # one canonical vector per algo is enough
    input_bytes = bytes.fromhex(vec["input_hex"])
    expected    = bytes.fromhex(vec["expected_hex"])

    try:
        result = rerun_fn(input_bytes, observe_points=[])
    except NotImplementedError:
        return {
            "status": "skipped",
            "detail": "runner.rerun raised NotImplementedError (File mode)",
        }
    except Exception as exc:
        return {
            "status": "errored",
            "detail": f"{type(exc).__name__}: {exc}",
        }

    # Truncate to runner's declared output_length so a runner that returns the
    # full output buffer (with trailing padding) still matches the canonical
    # vector. If the runner returns shorter than expected, we compare what
    # we got.
    output_length = target_meta.output_length or len(expected)
    actual = result.output[:output_length]
    expected_trunc = expected[:output_length]

    if actual == expected_trunc:
        return {
            "status": "passed",
            "detail": (
                f"input_hex={vec['input_hex']} → output matched canonical "
                f"{algo} vector ({len(actual)} bytes)"
            ),
        }
    return {
        "status": "failed",
        "detail": (
            f"input_hex={vec['input_hex']}: expected "
            f"{expected_trunc.hex()[:32]}…, got {actual.hex()[:32]}…"
        ),
    }


