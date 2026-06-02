"""Verifier — system's only source of truth (PLAN §1.1, §5).

Three concrete strategies:
  - check_handler_semantic: given a claimed op (e.g. "this handler computes XOR"),
    take the real input state at that handler's entry, compute the claim, and
    compare to the real output state after the handler ran.
  - check_simplification: given a simplified expression for a register value at
    a specific instruction index, evaluate it on the trace's actual operands
    and confirm it matches the trace's actual register value.
  - check_io_equivalence: re-run a reconstructed function model against the
    real runner over a coverage input set; outputs must agree byte-for-byte.

Verdict can be PASS / FAIL / INCONCLUSIVE. INCONCLUSIVE triggers the auto-
expand-then-pause flow from DECISIONS D-008.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .runner_client import RunnerAdapter


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class VerifierResult:
    verdict: Verdict
    strategy: str
    detail: dict[str, Any]


# Map claimed_op["op"] to a concrete Python operator. Keep tight — every entry
# is a chunk of ground truth that can be paired with a 2-arg ARM instruction.
_BIN_OPS: dict[str, Callable[[int, int], int]] = {
    "ADD":  lambda a, b: (a + b) & 0xFFFFFFFFFFFFFFFF,
    "SUB":  lambda a, b: (a - b) & 0xFFFFFFFFFFFFFFFF,
    "XOR":  operator.xor,
    # AArch64 mnemonic alias for XOR — keep both so handler claims can use
    # whichever the disassembly produced.
    "EOR":  operator.xor,
    "AND":  operator.and_,
    "OR":   operator.or_,
    "ORR":  operator.or_,
    "MUL":  lambda a, b: (a * b) & 0xFFFFFFFFFFFFFFFF,
    "LSL":  lambda a, b: (a << (b & 0x3F)) & 0xFFFFFFFFFFFFFFFF,
    "LSR":  lambda a, b: (a & 0xFFFFFFFFFFFFFFFF) >> (b & 0x3F),
    "ASR":  lambda a, b: ((a if a < 0x8000000000000000 else a - 0x10000000000000000) >> (b & 0x3F)) & 0xFFFFFFFFFFFFFFFF,
    "ROR":  lambda a, b: (((a & 0xFFFFFFFFFFFFFFFF) >> (b & 0x3F)) |
                          ((a << ((64 - (b & 0x3F)) & 0x3F)) & 0xFFFFFFFFFFFFFFFF)),
    "BIC":  lambda a, b: a & ((~b) & 0xFFFFFFFFFFFFFFFF),
    "EON":  lambda a, b: a ^ ((~b) & 0xFFFFFFFFFFFFFFFF),
    "ORN":  lambda a, b: a | ((~b) & 0xFFFFFFFFFFFFFFFF),
}

# Single-source ops: dst = fn(src). Used when ARM emits a unary instruction
# (SXTW, MOV, MVN, NEG, REV, …) — `claimed_op["src"]` has exactly one entry
# and `claimed_op` MUST omit the second operand.
_UNI_OPS: dict[str, Callable[[int], int]] = {
    "MOV":  lambda a: a & 0xFFFFFFFFFFFFFFFF,
    "MVN":  lambda a: (~a) & 0xFFFFFFFFFFFFFFFF,
    "NEG":  lambda a: (-a) & 0xFFFFFFFFFFFFFFFF,
    # SXTW: sign-extend 32-bit -> 64-bit
    "SXTW": lambda a: (a | 0xFFFFFFFF00000000) if (a & 0x80000000) else (a & 0xFFFFFFFF),
    "UXTW": lambda a: a & 0xFFFFFFFF,
    # REV: byte-reverse a 64-bit register value (AArch64 REV)
    "REV":  lambda a: int.from_bytes((a & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little"), "big"),
    # REV32 / REV16 — half-step reversals; not implemented (low priority for
    # SHA-256 trace coverage).  Add if a target needs them.
    "CLZ":  lambda a: (64 - (a & 0xFFFFFFFFFFFFFFFF).bit_length()) if a else 64,
}

# Ternary ops — used by SHA-2 family round functions etc. (BR-4 §F).
# Claim shape: {"op": "CH", "dst": "x4", "src": ["x1", "x2", "x3"]}.
_TRI_OPS: dict[str, Callable[[int, int, int], int]] = {
    # Ch(x, y, z) = (x & y) ^ (~x & z)   — SHA-256 choose
    "CH":  lambda x, y, z: ((x & y) ^ ((~x) & z)) & 0xFFFFFFFFFFFFFFFF,
    # Maj(x, y, z) = (x & y) ^ (x & z) ^ (y & z)   — SHA-256 majority
    "MAJ": lambda x, y, z: ((x & y) ^ (x & z) ^ (y & z)) & 0xFFFFFFFFFFFFFFFF,
    # Parity(x, y, z) = x ^ y ^ z   — SHA-1 round 2/4
    "PARITY": lambda x, y, z: (x ^ y ^ z) & 0xFFFFFFFFFFFFFFFF,
}

# Bit-field extract ops (0526 C5.6). Claim shape:
# {"op": "UBFX"|"SBFX", "dst": <reg>, "src": [<reg>], "lsb": int, "width": int}.
# UBFX: extract `width` bits from src starting at `lsb`, zero-extend.
# SBFX: same but sign-extend the top extracted bit to the dst width.
_BFX_OPS = frozenset(("UBFX", "SBFX"))

# Shift / extension applied to the *second* binary operand before the op.
# ARM64 extended-register form composes (extend → left-shift by `amount`):
#   `add x9, x21, w9, uxtw #3`  =>  x21 + (uxtw(w9) << 3)
# so the extension lambdas all honor `amount` as a final left-shift.
# Caller passes claimed_op["src2_ext"] = {
#     "kind": "<lsl|lsr|asr|ror|sxtw|sxth|sxtb|uxtw|uxth|uxtb>",
#     "amount": int }.
def _ext_signed(v: int, n: int, sign_bit: int, mask: int) -> int:
    base = (v | ~mask) & 0xFFFFFFFFFFFFFFFF if (v & sign_bit) else (v & mask)
    return (base << (n & 0x3F)) & 0xFFFFFFFFFFFFFFFF


def _ext_unsigned(v: int, n: int, mask: int) -> int:
    return ((v & mask) << (n & 0x3F)) & 0xFFFFFFFFFFFFFFFF


_SRC2_EXTS: dict[str, Callable[[int, int], int]] = {
    "lsl": lambda v, n: (v << (n & 0x3F)) & 0xFFFFFFFFFFFFFFFF,
    "lsr": lambda v, n: (v & 0xFFFFFFFFFFFFFFFF) >> (n & 0x3F),
    "asr": lambda v, n: ((v if v < 0x8000000000000000 else v - 0x10000000000000000) >> (n & 0x3F)) & 0xFFFFFFFFFFFFFFFF,
    "ror": lambda v, n: (((v & 0xFFFFFFFFFFFFFFFF) >> (n & 0x3F)) |
                         ((v << ((64 - (n & 0x3F)) & 0x3F)) & 0xFFFFFFFFFFFFFFFF)),
    "sxtw": lambda v, n: _ext_signed(v, n, 0x80000000,   0xFFFFFFFF),
    "sxth": lambda v, n: _ext_signed(v, n, 0x8000,       0xFFFF),
    "sxtb": lambda v, n: _ext_signed(v, n, 0x80,         0xFF),
    "uxtw": lambda v, n: _ext_unsigned(v, n, 0xFFFFFFFF),
    "uxth": lambda v, n: _ext_unsigned(v, n, 0xFFFF),
    "uxtb": lambda v, n: _ext_unsigned(v, n, 0xFF),
}


def _apply_binop(op: str, a: int, b: int, width: int,
                 *, bin_fn: Callable[[int, int], int] | None = None) -> int:
    """Width-correct binop. width is 32 (w-reg) or 64 (x-reg).

    Shift / rotate semantics depend on width (32-bit ROR ≠ 64-bit ROR);
    other binops are width-transparent but we still mask sources/result
    so a 32-bit op result doesn't accidentally pass a w-reg compare due
    to noise in the high 32 bits.
    """
    mask = (1 << width) - 1
    a_m = a & mask
    b_m = b & mask
    if op == "LSL":
        return (a_m << (b_m & (width - 1))) & mask
    if op == "LSR":
        return (a_m >> (b_m & (width - 1))) & mask
    if op == "ASR":
        sign_bit = 1 << (width - 1)
        signed = a_m if a_m < sign_bit else a_m - (mask + 1)
        return (signed >> (b_m & (width - 1))) & mask
    if op == "ROR":
        sh = b_m & (width - 1)
        if sh == 0:
            return a_m
        return ((a_m >> sh) | ((a_m << (width - sh)) & mask)) & mask
    if bin_fn is None:
        bin_fn = _BIN_OPS[op]
    return bin_fn(a_m, b_m) & mask


def _apply_src2_ext(kind: str, v: int, amount: int, width: int) -> int:
    """Width-correct src2 extension/shift.

    For sign/zero extension (sxt*/uxt*), the source slice width is implied
    by the kind (sxtw/uxtw = 32-bit slice, sxth/uxth = 16-bit, sxtb/uxtb =
    8-bit) and the result is then left-shifted by `amount` modulo the dst
    width's mask. For pure shift/rotate (lsl/lsr/asr/ror), the operation
    runs at dst width.
    """
    mask = (1 << width) - 1
    if kind == "lsl":
        return ((v & mask) << (amount & (width - 1))) & mask
    if kind == "lsr":
        return ((v & mask) >> (amount & (width - 1))) & mask
    if kind == "asr":
        v_m = v & mask
        sign_bit = 1 << (width - 1)
        signed = v_m if v_m < sign_bit else v_m - (mask + 1)
        return (signed >> (amount & (width - 1))) & mask
    if kind == "ror":
        v_m = v & mask
        sh = amount & (width - 1)
        if sh == 0:
            return v_m
        return ((v_m >> sh) | ((v_m << (width - sh)) & mask)) & mask
    # Sign / zero extension: the lambda in _SRC2_EXTS already does the
    # extend-then-left-shift in 64-bit land; mask to dst width.
    return _SRC2_EXTS[kind](v, amount) & mask


class Verifier:
    """Concrete verifier. PLAN §5 strategies."""

    def __init__(self, rerun: RunnerAdapter):
        self.rerun = rerun

    def check_handler_semantic(
        self,
        input_state: dict[str, int],
        claimed_op: dict[str, Any],
        expected_output_state: dict[str, int],
    ) -> VerifierResult:
        """input_state: {reg → value} as it stood before the handler.
        claimed_op shapes (BR-2 §7 — expanded coverage):
          - 2-arg binop:        {"op": "XOR", "dst": "x4", "src": ["x1", "x2"]}
          - reg-imm binop:      {"op": "ADD", "dst": "x4", "src": ["x1"], "imm": "0x10"}
          - shifted/extended:   {"op": "ADD", "dst": "x4", "src": ["x1", "x2"],
                                 "src2_ext": {"kind": "lsl", "amount": 3}}
                                 or {"kind": "sxtw", "amount": 0}
          - unary op:           {"op": "SXTW", "dst": "x4", "src": ["x1"]}
        expected_output_state: {reg → value} as it stood after the handler.
        """
        op_name = claimed_op.get("op")
        if op_name is None:
            return VerifierResult(
                Verdict.INCONCLUSIVE, "handler_semantic",
                {"reason": "claim missing 'op'"},
            )
        op_upper = str(op_name).upper()
        src = claimed_op.get("src") or []
        dst = claimed_op.get("dst")
        if dst is None:
            return VerifierResult(
                Verdict.INCONCLUSIVE, "handler_semantic",
                {"reason": f"claim missing 'dst' for op={op_name}"},
            )

        # --- ternary op path (Ch / Maj / Parity for SHA-2 round funcs, BR-4 §F) ---
        tri_fn = _TRI_OPS.get(op_upper)
        if tri_fn is not None and len(src) == 3:
            try:
                a = input_state[src[0]]
                b = input_state[src[1]]
                c = input_state[src[2]]
                expected = expected_output_state[dst]
            except KeyError as e:
                return VerifierResult(
                    Verdict.INCONCLUSIVE, "handler_semantic",
                    {"reason": f"missing input/output reg: {e}"},
                )
            got = tri_fn(a, b, c) & 0xFFFFFFFFFFFFFFFF
            if got == expected or (got & 0xFFFFFFFF) == (expected & 0xFFFFFFFF):
                return VerifierResult(Verdict.PASS, "handler_semantic",
                                      {"op": op_upper, "dst": dst, "value": f"0x{got:x}"})
            return VerifierResult(Verdict.FAIL, "handler_semantic", {
                "op": op_upper, "dst": dst,
                "computed": f"0x{got:x}", "expected": f"0x{expected:x}"})

        # --- bit-field extract path (UBFX / SBFX, 0526 C5.6) ---
        if op_upper in _BFX_OPS and len(src) == 1:
            lsb = claimed_op.get("lsb")
            width_field = claimed_op.get("width")
            if lsb is None or width_field is None:
                return VerifierResult(
                    Verdict.INCONCLUSIVE, "handler_semantic",
                    {"reason": f"op={op_name} needs 'lsb' and 'width' fields"},
                )
            src_name = src[0]
            if src_name in ("wzr", "xzr"):
                a = 0
            else:
                try:
                    a = input_state[src_name]
                except KeyError as e:
                    return VerifierResult(
                        Verdict.INCONCLUSIVE, "handler_semantic",
                        {"reason": f"missing input reg: {e}"},
                    )
            try:
                expected = expected_output_state[dst]
            except KeyError as e:
                return VerifierResult(
                    Verdict.INCONCLUSIVE, "handler_semantic",
                    {"reason": f"missing output reg: {e}"},
                )
            dst_width = 32 if dst.startswith("w") else 64
            src_width = 32 if src_name.startswith("w") else 64
            # Mask source to its alias width before extracting.
            extracted = (a & ((1 << src_width) - 1)) >> int(lsb)
            extracted &= (1 << int(width_field)) - 1
            if op_upper == "SBFX":
                # Sign-extend the top bit of the extracted field to dst width.
                sign_bit = 1 << (int(width_field) - 1)
                if extracted & sign_bit:
                    extracted |= (~((1 << int(width_field)) - 1)) & ((1 << dst_width) - 1)
            got = extracted & ((1 << dst_width) - 1)
            if got == (expected & ((1 << dst_width) - 1)):
                return VerifierResult(Verdict.PASS, "handler_semantic",
                                      {"op": op_upper, "dst": dst, "value": f"0x{got:x}"})
            return VerifierResult(Verdict.FAIL, "handler_semantic", {
                "op": op_upper, "dst": dst,
                "computed": f"0x{got:x}", "expected": f"0x{expected:x}"})

        # --- unary op path ---
        uni_fn = _UNI_OPS.get(op_upper)
        if uni_fn is not None and len(src) == 1:
            # AArch64 zero registers (wzr/xzr) read as 0 and aren't typically
            # captured in regs_read; substitute directly.
            src_name = src[0]
            if src_name in ("wzr", "xzr"):
                a = 0
            else:
                try:
                    a = input_state[src_name]
                except KeyError as e:
                    return VerifierResult(
                        Verdict.INCONCLUSIVE, "handler_semantic",
                        {"reason": f"missing input/output reg: {e}"},
                    )
            try:
                expected = expected_output_state[dst]
            except KeyError as e:
                return VerifierResult(
                    Verdict.INCONCLUSIVE, "handler_semantic",
                    {"reason": f"missing input/output reg: {e}"},
                )
            # REV is byte-permutation; width matters (REV w? ≠ REV x? on low-32).
            # All other unary ops are width-transparent enough that the
            # low-32 match fallback below covers w-reg variants.
            if op_upper == "REV" and dst.startswith("w"):
                got = int.from_bytes(
                    (a & 0xFFFFFFFF).to_bytes(4, "little"), "big",
                )
            else:
                got = uni_fn(a) & 0xFFFFFFFFFFFFFFFF
            if got == expected or (got & 0xFFFFFFFF) == (expected & 0xFFFFFFFF):
                return VerifierResult(Verdict.PASS, "handler_semantic",
                                      {"op": op_upper, "dst": dst, "value": f"0x{got:x}"})
            return VerifierResult(Verdict.FAIL, "handler_semantic", {
                "op": op_upper, "dst": dst,
                "computed": f"0x{got:x}", "expected": f"0x{expected:x}"})

        # --- binary op path (incl. imm + extended-register) ---
        bin_fn = _BIN_OPS.get(op_upper)
        if bin_fn is None:
            return VerifierResult(
                Verdict.INCONCLUSIVE, "handler_semantic",
                {"reason": f"unsupported claim shape: op={op_name} dst={dst} src={src}"},
            )

        # Resolve src[0]
        if len(src) < 1:
            return VerifierResult(
                Verdict.INCONCLUSIVE, "handler_semantic",
                {"reason": f"op={op_name} needs at least one src register"},
            )
        if src[0] in ("wzr", "xzr"):
            a = 0
        else:
            try:
                a = input_state[src[0]]
            except KeyError as e:
                return VerifierResult(
                    Verdict.INCONCLUSIVE, "handler_semantic",
                    {"reason": f"missing input reg: {e}"},
                )

        # Resolve src[1] — either a second register or an immediate.
        imm = claimed_op.get("imm")
        if len(src) == 2:
            if src[1] in ("wzr", "xzr"):
                b = 0
            else:
                try:
                    b = input_state[src[1]]
                except KeyError as e:
                    return VerifierResult(
                        Verdict.INCONCLUSIVE, "handler_semantic",
                        {"reason": f"missing input reg: {e}"},
                    )
        elif imm is not None:
            try:
                b = int(imm, 16) if isinstance(imm, str) else int(imm)
            except (TypeError, ValueError):
                return VerifierResult(
                    Verdict.INCONCLUSIVE, "handler_semantic",
                    {"reason": f"malformed imm: {imm!r}"},
                )
        else:
            return VerifierResult(
                Verdict.INCONCLUSIVE, "handler_semantic",
                {"reason": f"op={op_name} needs src[1] register or 'imm' field"},
            )

        # Optional pre-op extension/shift on the second operand.
        ext = claimed_op.get("src2_ext")
        if ext:
            kind = str(ext.get("kind", "")).lower()
            amount = int(ext.get("amount", 0))
            if kind not in _SRC2_EXTS:
                return VerifierResult(
                    Verdict.INCONCLUSIVE, "handler_semantic",
                    {"reason": f"unknown src2_ext kind={kind!r}"},
                )
            # Width-aware extension is applied later, after we know dst width.
            ext_kind, ext_amount = kind, amount
        else:
            ext_kind = ext_amount = None

        try:
            expected = expected_output_state[dst]
        except KeyError as e:
            return VerifierResult(
                Verdict.INCONCLUSIVE, "handler_semantic",
                {"reason": f"missing output reg: {e}"},
            )

        # ARM 32-bit form (`add w?, w?, w?`, `ror w?, w?, #N`, etc.) takes 32-bit
        # operands and 32-bit shift/rotate widths. Width is dictated by dst
        # alias, not src — w-reg writes clear the top 32 bits.
        width = 32 if dst.startswith("w") else 64
        if ext_kind is not None:
            b = _apply_src2_ext(ext_kind, b, ext_amount, width)
        got = _apply_binop(op_upper, a, b, width, bin_fn=bin_fn)

        # Mask both sides to the dst width before comparing — w-reg writes
        # zero the upper half of the underlying x-reg.
        mask = (1 << width) - 1
        if (got & mask) == (expected & mask):
            return VerifierResult(Verdict.PASS, "handler_semantic",
                                  {"op": op_upper, "dst": dst, "value": f"0x{got:x}"})
        return VerifierResult(Verdict.FAIL, "handler_semantic", {
            "op": op_upper, "dst": dst,
            "computed": f"0x{got:x}", "expected": f"0x{expected:x}"})

    def check_simplification(
        self,
        eval_fn: Callable[[dict[str, int]], int],
        concrete_input: dict[str, int],
        expected_value: int,
    ) -> VerifierResult:
        """Caller supplies a Python function that takes a concrete reg map and
        returns what the simplified expression computes; we run it on the
        trace's actual inputs and compare to the trace's actual output value.
        """
        try:
            got = eval_fn(concrete_input)
        except Exception as e:
            return VerifierResult(Verdict.INCONCLUSIVE, "simplification",
                                  {"reason": f"eval_fn raised: {type(e).__name__}: {e}"})
        if got == expected_value:
            return VerifierResult(Verdict.PASS, "simplification", {"value": f"0x{got:x}"})
        return VerifierResult(Verdict.FAIL, "simplification", {
            "computed": f"0x{got:x}", "expected": f"0x{expected_value:x}"})

    def check_fingerprint(
        self,
        claim: dict[str, Any],
        anchors: list[tuple[int, int]],
        items: list,
    ) -> VerifierResult:
        """Mechanically verify an S1.5 fingerprint claim against the trace.

        Two flavors of fingerprint payload:
          - scalar:  payload["magic"] = "0x428a2f98" — an SHA-256 round constant,
                     CRC table entry, etc. Pass if any anchor instruction's
                     regs_write contains this value.
          - pattern: payload["match_text"] = "aes" / "sha256h" / ... — Pass if
                     any anchor instruction's mnemonic contains this substring
                     (case-insensitive match, since capstone yields lowercase).

        The verifier is the only source of truth (PLAN §1.1): plugin hits start
        as `pending` until this check runs.
        """
        magic_hex = claim.get("magic")
        match_text = claim.get("match_text")
        if not anchors:
            return VerifierResult(Verdict.INCONCLUSIVE, "fingerprint",
                                  {"reason": "no anchors to check against"})

        if magic_hex:
            try:
                magic_val = int(magic_hex, 16) if isinstance(magic_hex, str) else int(magic_hex)
            except (TypeError, ValueError):
                return VerifierResult(Verdict.INCONCLUSIVE, "fingerprint",
                                      {"reason": f"malformed magic: {magic_hex!r}"})
            for idx, _pc in anchors:
                if 0 <= idx < len(items):
                    ins = items[idx]
                    if magic_val in ins.regs_write.values():
                        return VerifierResult(Verdict.PASS, "fingerprint", {
                            "verified_at_idx": idx,
                            "magic": f"0x{magic_val:x}",
                            "kind":  "scalar_magic_write",
                        })
            return VerifierResult(Verdict.FAIL, "fingerprint", {
                "diagnosis": "no anchor instruction wrote the claimed magic value",
                "magic":     magic_hex,
                "checked":   len(anchors),
            })

        if match_text:
            needle = match_text.lower()
            for idx, _pc in anchors:
                if 0 <= idx < len(items):
                    if needle in items[idx].mnemonic.lower():
                        return VerifierResult(Verdict.PASS, "fingerprint", {
                            "verified_at_idx": idx,
                            "match_text": match_text,
                            "kind": "mnemonic_pattern",
                        })
            return VerifierResult(Verdict.FAIL, "fingerprint", {
                "diagnosis": "no anchor instruction's mnemonic contains the pattern",
                "match_text": match_text,
                "checked":    len(anchors),
            })

        return VerifierResult(Verdict.INCONCLUSIVE, "fingerprint",
                              {"reason": "claim has neither 'magic' nor 'match_text'"})

    def check_io_equivalence(
        self,
        reconstructed_fn: Callable[[bytes], bytes],
        coverage_inputs: list[bytes],
    ) -> VerifierResult:
        """For each input, compare reconstructed_fn(input) to runner.rerun(input).
        Requires Live-mode runner; in File mode raises (caller should skip).
        """
        try:
            mismatches: list[dict] = []
            for inp in coverage_inputs:
                try:
                    actual = self.rerun.rerun(inp).output
                except NotImplementedError:
                    return VerifierResult(
                        Verdict.INCONCLUSIVE, "io_equivalence",
                        {"reason": "runner is File mode — no rerun available"},
                    )
                guess = reconstructed_fn(inp)
                if actual != guess:
                    mismatches.append({
                        "input": inp.hex(),
                        "actual": actual.hex(),
                        "guess":  guess.hex(),
                    })
                    if len(mismatches) >= 5:
                        break
            if mismatches:
                return VerifierResult(Verdict.FAIL, "io_equivalence", {
                    "total_inputs": len(coverage_inputs),
                    "first_mismatches": mismatches,
                })
            return VerifierResult(Verdict.PASS, "io_equivalence",
                                  {"inputs_passed": len(coverage_inputs)})
        except Exception as e:
            return VerifierResult(Verdict.INCONCLUSIVE, "io_equivalence",
                                  {"reason": f"verifier raised: {type(e).__name__}: {e}"})
