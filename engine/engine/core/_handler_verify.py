"""Core mixin: handler/triton/sigma verify-and-promote passes."""
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

from ._base import *  # noqa: F401,F403
from ._base import _STAGES, _algorithm_hyp_trap, _run_io_equivalence  # noqa: F401


class _CoreHandlerVerifyMixin:
    """Core methods: handler/triton/sigma verify-and-promote passes (split from the monolithic Core)."""
    def verify_and_promote_plugin_findings(self) -> dict[str, Any]:
        """Pure-deterministic finding pass: walk every `pending` plugin
        algo_signature hyp, run `Verifier.check_fingerprint` against its
        anchors + the trace, mark verdict, and promote on PASS.

        No LLM involvement; this is the cheap baseline that turns S1.5 hits
        into real findings when the trace genuinely contains the magic value.
        Idempotent: hyps already in passed/failed status are skipped.

        Returns: {"checked": N, "passed": K, "failed": M, "promoted": K,
                  "inconclusive": I}.
        """
        conn = open_hypotheses_db(self.work)
        promoted_ids: list[int] = []
        n_pass = n_fail = n_inc = 0
        try:
            tree = HypTree(conn)
            pending = tree.query(status="pending", kind="algo_signature",
                                 source="plugin")
            for h in pending:
                anchors = tree.anchors_of(h.id)
                result = self.verifier.check_fingerprint(
                    h.payload, anchors, self._items,
                )
                tree.mark_verdict(h.id, result.verdict.value, {
                    "strategy": result.strategy,
                    **result.detail,
                })
                if result.verdict.value == "pass":
                    n_pass += 1
                    promoted_ids.append(h.id)
                elif result.verdict.value == "fail":
                    n_fail += 1
                else:
                    n_inc += 1
        finally:
            conn.close()

        promoted = 0
        for hid in promoted_ids:
            try:
                self.promote_to_finding(hid, verifier_strategy="fingerprint",
                                        stage="s1b")
                promoted += 1
            except (KeyError, ValueError):
                # Race or already promoted in a prior partial run — best-effort.
                pass

        return {
            "checked":      len(promoted_ids) + n_fail + n_inc,
            "passed":       n_pass,
            "failed":       n_fail,
            "inconclusive": n_inc,
            "promoted":     promoted,
        }

    def verify_and_promote_handler_binops(self) -> dict[str, Any]:
        """Deterministic handler-semantic pass (BR-4 §1).

        Scans the trace for plain reg-reg-reg ARM binops the verifier can check
        mechanically (`eor/and/orr/add/sub/mul/lsl/lsr/ror` with three register
        operands and no shift/extend tail), runs
        `verifier.check_handler_semantic` against each instruction's
        regs_read/regs_write, and on PASS submits + promotes a finding tagged
        `source="s5_deterministic"`. Symmetric to
        `verify_and_promote_plugin_findings` but for handler claims; the two
        together cover the deterministic-evidence baseline.

        Idempotent: re-running only skips already-promoted PCs by hashing
        `subject=binop@<pc>` into the existing dedupe (hyp tree's UNIQUE on
        subject + payload signature handles it).
        """
        import re as _re

        # Tight mapping; expand only if verifier._BIN_OPS gains entries.
        _arm_to_op = {
            "eor": "XOR", "and": "AND", "orr": "OR",
            "add": "ADD", "sub": "SUB", "mul": "MUL",
            "lsl": "LSL", "lsr": "LSR", "ror": "ROR",
        }
        _reg_re = _re.compile(r"^[wx]\d+$|^wzr$|^xzr$")
        _tail_tokens = frozenset((
            "lsl", "lsr", "asr", "ror",
            "sxtw", "sxtb", "sxth", "uxtw", "uxtb", "uxth",
        ))

        checked = passed = failed = inconclusive = promoted = 0
        seen_pcs: set[int] = set()

        for ins in self._items:
            # Skip duplicate PCs — same instruction may appear many times in
            # a loop; one finding per PC is enough.
            if ins.pc in seen_pcs:
                continue

            toks = [t for t in _re.split(r"[,\s]+", ins.mnemonic.strip()) if t]
            if len(toks) < 4:
                continue
            mnem = toks[0].lower()
            if mnem not in _arm_to_op:
                continue
            ops = toks[1:]
            if not all(_reg_re.match(o) for o in ops[:3]):
                continue
            # Reject sp-relative arithmetic — verifier doesn't model sp
            # semantics and these are prologue/epilogue noise anyway.
            if "sp" in (ops[0], ops[1], ops[2]):
                continue
            # Reject any form with a shift/extend tail — that's a separate
            # claim shape (`src2_ext`) the verifier supports but the
            # heuristic-classified payload here doesn't carry it. Skipping
            # is safer than guessing the amount.
            if len(ops) > 3 and ops[3].lower() in _tail_tokens:
                continue

            payload = {
                "op":  _arm_to_op[mnem],
                "dst": ops[0],
                "src": [ops[1], ops[2]],
            }
            res = self.verifier.check_handler_semantic(
                ins.regs_read, payload, ins.regs_write,
            )
            checked += 1
            v = res.verdict.value
            if v == "pass":
                passed += 1
                seen_pcs.add(ins.pc)
                try:
                    hid = self.submit_hypothesis(
                        kind="handler_semantic",
                        subject=f"binop@{ins.pc:#x}",
                        payload={
                            **payload,
                            "rationale": (
                                f"ARM64 '{ins.mnemonic}' classified as "
                                f"reg-reg {payload['op']}; verifier confirmed "
                                f"against trace regs_read/regs_write at "
                                f"idx={ins.idx}."
                            ),
                        },
                        confidence=1.0,
                        source="s5_deterministic",
                        anchors=[(ins.idx, ins.idx)],
                    )
                    # promote_to_finding requires status='passed', so flip the
                    # verdict on the freshly-submitted hyp before promoting.
                    # (Mirrors what plugin pass does for pre-existing hyps via
                    # mark_verdict.)
                    conn = open_hypotheses_db(self.work)
                    try:
                        HypTree(conn).mark_verdict(hid, "pass", {
                            "strategy": res.strategy, **res.detail,
                        })
                    finally:
                        conn.close()
                    self.promote_to_finding(
                        hid, verifier_strategy="handler_semantic",
                        stage="s5-verify",
                    )
                    promoted += 1
                except (KeyError, ValueError):
                    # Already promoted by a prior partial run / race — skip.
                    pass
            elif v == "fail":
                failed += 1
            else:
                inconclusive += 1

        return {
            "stage":        "s5-verify",
            "checked":      checked,
            "passed":       passed,
            "failed":       failed,
            "inconclusive": inconclusive,
            "promoted":     promoted,
        }

    def _run_handler_pass(
        self,
        scanner,
        *,
        stage_label: str,
        subject_prefix: str,
    ) -> dict[str, Any]:
        """Shared loop for layer-0 single-instruction handler passes.

        `scanner(ins)` returns either a `payload` dict for verifier.
        check_handler_semantic, or None to skip. Dedupes by PC; on PASS,
        submits a hypothesis and promotes to a finding tagged
        source="s5_deterministic". Used by C5.1-C5.7 layer-0 discoverers.
        """
        checked = passed = failed = inconclusive = promoted = 0
        seen_pcs: set[int] = set()

        for ins in self._items:
            if ins.pc in seen_pcs:
                continue
            payload = scanner(ins)
            if payload is None:
                continue
            res = self.verifier.check_handler_semantic(
                ins.regs_read, payload, ins.regs_write,
            )
            checked += 1
            v = res.verdict.value
            if v == "pass":
                passed += 1
                seen_pcs.add(ins.pc)
                try:
                    hid = self.submit_hypothesis(
                        kind="handler_semantic",
                        subject=f"{subject_prefix}@{ins.pc:#x}",
                        payload={
                            **payload,
                            "rationale": (
                                f"ARM64 '{ins.mnemonic}' matched {subject_prefix} "
                                f"shape; verifier confirmed against trace at "
                                f"idx={ins.idx}."
                            ),
                        },
                        confidence=1.0,
                        source="s5_deterministic",
                        anchors=[(ins.idx, ins.idx)],
                    )
                    conn = open_hypotheses_db(self.work)
                    try:
                        HypTree(conn).mark_verdict(hid, "pass", {
                            "strategy": res.strategy, **res.detail,
                        })
                    finally:
                        conn.close()
                    self.promote_to_finding(
                        hid, verifier_strategy="handler_semantic",
                        stage="s5-verify",
                    )
                    promoted += 1
                except (KeyError, ValueError):
                    pass
            elif v == "fail":
                failed += 1
            else:
                inconclusive += 1

        return {
            "stage":        stage_label,
            "checked":      checked,
            "passed":       passed,
            "failed":       failed,
            "inconclusive": inconclusive,
            "promoted":     promoted,
        }

    def verify_and_promote_handler_unaries(self) -> dict[str, Any]:
        """Layer-0 unary single-instruction pass (C5.4 + C5.5).

        Scans the trace for ARM unary instructions verifier._UNI_OPS supports
        (MOV / MVN / NEG / SXTW / UXTW / REV / CLZ — 2-operand form with one
        register source and one register dest, no shift/extend tail). Same
        promote-and-dedupe-by-PC contract as verify_and_promote_handler_binops.
        """
        import re as _re

        _arm_to_unary = {
            "mov": "MOV", "mvn": "MVN", "neg": "NEG",
            "sxtw": "SXTW", "uxtw": "UXTW", "rev": "REV", "clz": "CLZ",
        }
        _reg_re = _re.compile(r"^[wx]\d+$|^wzr$|^xzr$")

        def _scan(ins):
            toks = [t for t in _re.split(r"[,\s]+", ins.mnemonic.strip()) if t]
            # MOV/MVN/NEG/SXTW/UXTW/REV/CLZ: `<op> <dst>, <src>` → 3 tokens
            if len(toks) != 3:
                return None
            mnem = toks[0].lower()
            if mnem not in _arm_to_unary:
                return None
            dst, src = toks[1], toks[2]
            if not (_reg_re.match(dst) and _reg_re.match(src)):
                return None
            if "sp" in (dst, src):
                return None
            return {
                "op":  _arm_to_unary[mnem],
                "dst": dst,
                "src": [src],
            }

        return self._run_handler_pass(
            _scan, stage_label="s5-verify-unary", subject_prefix="unary",
        )

    def verify_and_promote_handler_imm_binops(self) -> dict[str, Any]:
        """Layer-0 reg-imm binop pass (C5.1 + C5.2 + C5.3).

        Scans the trace for ARM reg-imm binops (`<op> <dst>, <src1>, #<imm>`)
        verifier accepts as shape #2: ADD/SUB/AND/ORR/EOR with literal imm,
        plus LSL/LSR/ASR/ROR with literal shift count. Skips sp-relative.
        """
        import re as _re

        # mnem → verifier op. EOR/AND/ORR mapped to canonical XOR/AND/OR.
        _arm_to_op = {
            "add": "ADD", "sub": "SUB",
            "and": "AND", "orr": "OR", "eor": "XOR",
            "lsl": "LSL", "lsr": "LSR", "asr": "ASR", "ror": "ROR",
        }
        _reg_re = _re.compile(r"^[wx]\d+$|^wzr$|^xzr$")
        _imm_re = _re.compile(r"^#(-?0x[0-9a-fA-F]+|-?\d+)$")

        def _scan(ins):
            toks = [t for t in _re.split(r"[,\s]+", ins.mnemonic.strip()) if t]
            if len(toks) != 4:
                return None
            mnem = toks[0].lower()
            if mnem not in _arm_to_op:
                return None
            dst, src1, imm_tok = toks[1], toks[2], toks[3]
            if not (_reg_re.match(dst) and _reg_re.match(src1)):
                return None
            if "sp" in (dst, src1):
                return None
            m = _imm_re.match(imm_tok)
            if not m:
                return None
            raw = m.group(1)
            try:
                imm_val = int(raw, 16) if raw.lower().startswith(("0x", "-0x")) else int(raw)
            except ValueError:
                return None
            return {
                "op":  _arm_to_op[mnem],
                "dst": dst,
                "src": [src1],
                "imm": f"0x{imm_val & 0xFFFFFFFFFFFFFFFF:x}" if imm_val >= 0 else str(imm_val),
            }

        return self._run_handler_pass(
            _scan, stage_label="s5-verify-imm", subject_prefix="imm_binop",
        )

    def verify_and_promote_handler_extended_binops(self) -> dict[str, Any]:
        """Layer-0 shifted/extended-register binop pass (C5.7).

        Scans the trace for ARM binops with a shift/extend tail
        (`add x4, x5, w6, sxtw #3` style). Maps the tail token to
        verifier's `src2_ext` claim and runs check_handler_semantic.
        """
        import re as _re

        _arm_to_op = {
            "add": "ADD", "sub": "SUB",
            "and": "AND", "orr": "OR", "eor": "XOR",
        }
        _reg_re = _re.compile(r"^[wx]\d+$|^wzr$|^xzr$")
        _ext_kinds = frozenset((
            "lsl", "lsr", "asr", "ror",
            "sxtw", "sxth", "sxtb", "uxtw", "uxth", "uxtb",
        ))
        _imm_re = _re.compile(r"^#(-?\d+|-?0x[0-9a-fA-F]+)$")

        def _scan(ins):
            toks = [t for t in _re.split(r"[,\s]+", ins.mnemonic.strip()) if t]
            # `<op> Rd, Rs1, Rs2, <kind> [#amount]` → 5 or 6 tokens
            if len(toks) not in (5, 6):
                return None
            mnem = toks[0].lower()
            if mnem not in _arm_to_op:
                return None
            dst, src1, src2 = toks[1], toks[2], toks[3]
            if not all(_reg_re.match(o) for o in (dst, src1, src2)):
                return None
            if "sp" in (dst, src1, src2):
                return None
            kind = toks[4].lower()
            if kind not in _ext_kinds:
                return None
            amount = 0
            if len(toks) == 6:
                m = _imm_re.match(toks[5])
                if not m:
                    return None
                raw = m.group(1)
                try:
                    amount = int(raw, 16) if raw.lower().startswith(("0x", "-0x")) else int(raw)
                except ValueError:
                    return None
            return {
                "op":  _arm_to_op[mnem],
                "dst": dst,
                "src": [src1, src2],
                "src2_ext": {"kind": kind, "amount": amount},
            }

        return self._run_handler_pass(
            _scan, stage_label="s5-verify-ext", subject_prefix="ext_binop",
        )

    def verify_and_promote_handler_bfx(self) -> dict[str, Any]:
        """Layer-0 bit-field-extract pass (C5.6).

        Scans the trace for ARM bitfield-extract aliases (`ubfx`, `sbfx`)
        and matches them to verifier's BFX claim shape: {"op": "UBFX|SBFX",
        "dst": <reg>, "src": [<reg>], "lsb": int, "width": int}. ARM emits
        these as `<op> Rd, Rs, #lsb, #width`.
        """
        import re as _re

        _arm_to_op = {"ubfx": "UBFX", "sbfx": "SBFX"}
        _reg_re = _re.compile(r"^[wx]\d+$|^wzr$|^xzr$")
        _imm_re = _re.compile(r"^#(-?\d+|-?0x[0-9a-fA-F]+)$")

        def _scan(ins):
            toks = [t for t in _re.split(r"[,\s]+", ins.mnemonic.strip()) if t]
            if len(toks) != 5:
                return None
            mnem = toks[0].lower()
            if mnem not in _arm_to_op:
                return None
            dst, src, lsb_tok, width_tok = toks[1], toks[2], toks[3], toks[4]
            if not (_reg_re.match(dst) and _reg_re.match(src)):
                return None
            if "sp" in (dst, src):
                return None
            m_lsb = _imm_re.match(lsb_tok)
            m_width = _imm_re.match(width_tok)
            if not (m_lsb and m_width):
                return None
            try:
                lsb = (int(m_lsb.group(1), 16) if m_lsb.group(1).lower().startswith(("0x", "-0x"))
                       else int(m_lsb.group(1)))
                width = (int(m_width.group(1), 16) if m_width.group(1).lower().startswith(("0x", "-0x"))
                         else int(m_width.group(1)))
            except ValueError:
                return None
            return {
                "op":    _arm_to_op[mnem],
                "dst":   dst,
                "src":   [src],
                "lsb":   lsb,
                "width": width,
            }

        return self._run_handler_pass(
            _scan, stage_label="s5-verify-bfx", subject_prefix="bfx",
        )

    def verify_and_promote_handler_ch_idioms(self) -> dict[str, Any]:
        """Ch idiom discoverer (0526Plan C3).

        Detects the SHA-2 Ch(x, y, z) = (x & y) ^ (~x & z) compact 3-insn ARM
        form (commonly emitted by clang/gcc for SHA-256/512 round functions):

          eor  t, y, z         ; t = y ^ z
          and  t, t, x         ; t = (y ^ z) & x = (x & y) ^ (x & z)
          eor  d, t, z         ; d = (x & y) ^ (x & z) ^ z = Ch(x, y, z)
                               ;   (or eor d, t, y → Ch(x, z, y) swap)

        Slides a 3-instruction window, checks the register-flow constraints,
        builds a ternary CH payload, runs verifier.check_handler_semantic,
        promotes on PASS. Dedupes by the final-instruction PC. Maj / Parity
        and Ch BIC/MVN variants didn't show up in TC1 / TC2 baselines so
        they're not implemented yet; add them when a target needs them.
        """
        import re as _re

        _reg_re = _re.compile(r"^[wx]\d+$|^wzr$|^xzr$")

        def _split(s: str) -> list[str]:
            return [t for t in _re.split(r"[,\s]+", s.strip()) if t]

        checked = passed = failed = inconclusive = promoted = 0
        seen_pcs: set[int] = set()
        items = self._items

        for i in range(len(items) - 2):
            ins0, ins1, ins2 = items[i], items[i + 1], items[i + 2]
            t0 = _split(ins0.mnemonic)
            t1 = _split(ins1.mnemonic)
            t2 = _split(ins2.mnemonic)
            if (len(t0) < 4 or len(t1) < 4 or len(t2) < 4
                    or t0[0].lower() != "eor"
                    or t1[0].lower() != "and"
                    or t2[0].lower() != "eor"):
                continue
            # All three are plain reg-reg-reg, no shift/extend tail
            if (len(t0) != 4 or len(t1) != 4 or len(t2) != 4):
                continue
            ops_all = t0[1:4] + t1[1:4] + t2[1:4]
            if not all(_reg_re.match(o) for o in ops_all):
                continue
            if "sp" in ops_all:
                continue

            t_tmp, y, z = t0[1], t0[2], t0[3]
            # ins1: `and t_tmp, ?, ?` where one src is t_tmp, the other is x
            if t1[1] != t_tmp:
                continue
            if t1[2] == t_tmp:
                x = t1[3]
            elif t1[3] == t_tmp:
                x = t1[2]
            else:
                continue
            # ins2: `eor d, ?, ?` where one src is t_tmp, the other is y or z
            if t2[2] == t_tmp:
                c_reg = t2[3]
            elif t2[3] == t_tmp:
                c_reg = t2[2]
            else:
                continue
            d = t2[1]
            if c_reg == z:
                ch_x, ch_y, ch_z = x, y, z
            elif c_reg == y:
                ch_x, ch_y, ch_z = x, z, y
            else:
                continue
            if ins2.pc in seen_pcs:
                continue

            input_state: dict[str, int] = {**ins0.regs_read, **ins1.regs_read}
            payload = {
                "op":  "CH",
                "dst": d,
                "src": [ch_x, ch_y, ch_z],
            }
            res = self.verifier.check_handler_semantic(
                input_state, payload, ins2.regs_write,
            )
            checked += 1
            v = res.verdict.value
            if v == "pass":
                passed += 1
                seen_pcs.add(ins2.pc)
                try:
                    hid = self.submit_hypothesis(
                        kind="handler_semantic",
                        subject=f"ch@{ins2.pc:#x}",
                        payload={
                            **payload,
                            "rationale": (
                                f"SHA-2 Ch(x,y,z) 3-insn idiom matched at "
                                f"{ins0.pc:#x}/{ins1.pc:#x}/{ins2.pc:#x}; "
                                f"verifier confirmed the ternary value."
                            ),
                            "anchor_pcs": [ins0.pc, ins1.pc, ins2.pc],
                        },
                        confidence=1.0,
                        source="s5_deterministic",
                        anchors=[(ins0.idx, ins2.idx)],
                    )
                    conn = open_hypotheses_db(self.work)
                    try:
                        HypTree(conn).mark_verdict(hid, "pass", {
                            "strategy": res.strategy, **res.detail,
                        })
                    finally:
                        conn.close()
                    self.promote_to_finding(
                        hid, verifier_strategy="handler_semantic",
                        stage="s5-verify",
                    )
                    promoted += 1
                except (KeyError, ValueError):
                    pass
            elif v == "fail":
                failed += 1
            else:
                inconclusive += 1

        # ---- Phase 2 (BR-8 #2): (x ∧ y) | (¬x ∧ z) variant ----
        # TC1 SHA-256 round-Ch uses `and / bic / orr` (gap-tolerant); since
        # (x ∧ y) and (¬x ∧ z) have disjoint supports, the final | is value-
        # equivalent to ⊕ — verifier op="CH" still matches. We index `and`
        # and `bic` ops by their dst register and, for every `orr d, ra, rb`,
        # look back at the latest-prior and / bic feeding ra/rb.
        from collections import defaultdict as _defaultdict
        ands_by_dst: dict[str, list[dict]] = _defaultdict(list)
        bics_by_dst: dict[str, list[dict]] = _defaultdict(list)
        for ins in items:
            toks = _split(ins.mnemonic)
            if len(toks) != 4 or not all(_reg_re.match(t) for t in toks[1:]):
                continue
            op = toks[0].lower()
            if op == "and":
                ands_by_dst[toks[1]].append({
                    "ins": ins, "dst": toks[1], "srcs": (toks[2], toks[3]),
                })
            elif op == "bic":
                bics_by_dst[toks[1]].append({
                    "ins": ins, "dst": toks[1],
                    "src1": toks[2], "src2": toks[3],
                })

        for ins in items:
            toks = _split(ins.mnemonic)
            if (len(toks) != 4 or toks[0].lower() != "orr"
                    or not all(_reg_re.match(t) for t in toks[1:])):
                continue
            if ins.pc in seen_pcs:
                continue
            d, ra, rb = toks[1], toks[2], toks[3]
            matched_variant = False
            for and_dst, bic_dst in ((ra, rb), (rb, ra)):
                if matched_variant:
                    break
                ac = [a for a in ands_by_dst.get(and_dst, [])
                      if a["ins"].idx < ins.idx]
                bc = [b for b in bics_by_dst.get(bic_dst, [])
                      if b["ins"].idx < ins.idx]
                if not ac or not bc:
                    continue
                and_op = max(ac, key=lambda a: a["ins"].idx)
                bic_op = max(bc, key=lambda b: b["ins"].idx)
                # bic dst, src1, src2  →  dst = src1 ∧ ¬src2  so x = src2.
                x_reg = bic_op["src2"]
                z_reg = bic_op["src1"]
                and_srcs = list(and_op["srcs"])
                if x_reg not in and_srcs:
                    continue
                and_srcs.remove(x_reg)
                y_reg = and_srcs[0]
                if y_reg == z_reg or y_reg in ("sp",) or z_reg in ("sp",):
                    continue
                x_val_a = and_op["ins"].regs_read.get(x_reg)
                x_val_b = bic_op["ins"].regs_read.get(x_reg)
                if x_val_a is None or x_val_b is None or x_val_a != x_val_b:
                    continue
                y_val = and_op["ins"].regs_read.get(y_reg)
                z_val = bic_op["ins"].regs_read.get(z_reg)
                if y_val is None or z_val is None:
                    continue
                payload = {"op": "CH", "dst": d, "src": [x_reg, y_reg, z_reg]}
                in_state = {x_reg: x_val_a, y_reg: y_val, z_reg: z_val}
                res = self.verifier.check_handler_semantic(
                    in_state, payload, ins.regs_write,
                )
                checked += 1
                v = res.verdict.value
                if v == "pass":
                    passed += 1
                    seen_pcs.add(ins.pc)
                    matched_variant = True
                    try:
                        hid = self.submit_hypothesis(
                            kind="handler_semantic",
                            subject=f"ch@{ins.pc:#x}",
                            payload={
                                **payload,
                                "rationale": (
                                    f"SHA-2 Ch via (x∧y)|(¬x∧z) (BR-8 #2): "
                                    f"{and_op['ins'].pc:#x} (and) / "
                                    f"{bic_op['ins'].pc:#x} (bic) / "
                                    f"{ins.pc:#x} (orr). Verifier confirms."
                                ),
                                "anchor_pcs": [and_op["ins"].pc,
                                               bic_op["ins"].pc, ins.pc],
                            },
                            confidence=1.0,
                            source="s5_deterministic",
                            anchors=[(and_op["ins"].idx, ins.idx)],
                        )
                        conn = open_hypotheses_db(self.work)
                        try:
                            HypTree(conn).mark_verdict(hid, "pass", {
                                "strategy": res.strategy, **res.detail,
                            })
                        finally:
                            conn.close()
                        self.promote_to_finding(
                            hid, verifier_strategy="handler_semantic",
                            stage="s5-verify",
                        )
                        promoted += 1
                    except (KeyError, ValueError):
                        pass
                elif v == "fail":
                    failed += 1
                else:
                    inconclusive += 1

        return {
            "stage":        "s5-verify-ch",
            "checked":      checked,
            "passed":       passed,
            "failed":       failed,
            "inconclusive": inconclusive,
            "promoted":     promoted,
        }

    def verify_and_promote_handler_maj_idioms(self) -> dict[str, Any]:
        """Maj idiom discoverer (BR-8 #2).

        Detects the SHA-2 Maj(a, b, c) compact ARM idiom:

          eor  t,  b, c          ; t = b ⊕ c
          and  t,  t, a          ; t = (b ⊕ c) ∧ a = (a∧b) ⊕ (a∧c)
          eor  d,  t, BC_AND     ; d = ((a∧b) ⊕ (a∧c)) ⊕ (b∧c) = Maj(a,b,c)
                                 ;   where BC_AND = b ∧ c from a prior `and`.

        The pattern is the dual of Ch: Maj is the bit-select of the
        majority of three inputs. clang/gcc emit this compact form for
        SHA-256 / SHA-512 round bodies. We accept up to a small gap
        between the three components so ILP-scheduled rounds match.

        Promotes one `handler_semantic` finding per matched Maj, dedup'd
        by the final eor PC, with op="MAJ" payload (verifier covers it
        via _TRI_OPS).
        """
        import re as _re
        from collections import defaultdict as _defaultdict
        _reg_re = _re.compile(r"^[wx]\d+$|^wzr$|^xzr$")

        def _split(s: str) -> list[str]:
            return [t for t in _re.split(r"[,\s]+", s.strip()) if t]

        checked = passed = failed = inconclusive = promoted = 0
        seen_pcs: set[int] = set()
        items = self._items

        # Index `and` ops by dst → for the BC_AND lookup. Also keep srcs.
        ands: list[dict] = []
        for ins in items:
            toks = _split(ins.mnemonic)
            if (len(toks) != 4 or toks[0].lower() != "and"
                    or not all(_reg_re.match(t) for t in toks[1:])):
                continue
            ands.append({"ins": ins, "dst": toks[1],
                         "srcs": (toks[2], toks[3])})
        ands_by_dst: dict[str, list[dict]] = _defaultdict(list)
        for a in ands:
            ands_by_dst[a["dst"]].append(a)

        # Walk for (eor, and, eor) 3-tuples within a sliding 12-insn window.
        MAJ_WIN = 12
        for i in range(len(items) - 2):
            eor0 = items[i]
            t0 = _split(eor0.mnemonic)
            if (len(t0) != 4 or t0[0].lower() != "eor"
                    or not all(_reg_re.match(t) for t in t0[1:])):
                continue
            t_tmp, b_r, c_r = t0[1], t0[2], t0[3]
            if b_r == c_r:
                continue
            # second component: and t_tmp, t_tmp, a OR and t_tmp, a, t_tmp.
            and_idx = None
            and_a = None
            for j in range(i + 1, min(len(items), i + 1 + MAJ_WIN)):
                tj = _split(items[j].mnemonic)
                if (len(tj) != 4 or tj[0].lower() != "and"
                        or not all(_reg_re.match(t) for t in tj[1:])):
                    if t_tmp in items[j].regs_write:
                        break  # t_tmp clobbered before chain closed
                    continue
                if tj[1] != t_tmp:
                    if t_tmp in items[j].regs_write:
                        break
                    continue
                if tj[2] == t_tmp:
                    and_a = tj[3]
                elif tj[3] == t_tmp:
                    and_a = tj[2]
                else:
                    if t_tmp in items[j].regs_write:
                        break
                    continue
                and_idx = j
                break
            if and_idx is None or and_a is None:
                continue
            # third component: eor d, t_tmp, bc_and  OR  eor d, bc_and, t_tmp,
            # where bc_and is the value of (b ∧ c) reaching this eor. The
            # `and`-producer is anywhere strictly before this eor (clang
            # routinely hoists it ahead of the eor⊕and pair to overlap
            # latencies). We resolve bc_and through the eor's regs_read.
            final_eor = None
            bc_and_used = None
            final_dst = None
            bc_name = None
            for k in range(and_idx + 1, min(len(items), and_idx + 1 + MAJ_WIN)):
                tk = _split(items[k].mnemonic)
                if (len(tk) != 4 or tk[0].lower() != "eor"
                        or not all(_reg_re.match(t) for t in tk[1:])):
                    if t_tmp in items[k].regs_write:
                        break
                    continue
                if tk[2] == t_tmp:
                    bc_candidate = tk[3]
                elif tk[3] == t_tmp:
                    bc_candidate = tk[2]
                else:
                    if t_tmp in items[k].regs_write:
                        break
                    continue
                # Find an `and` whose dst == bc_candidate AND whose srcs
                # are {b_r, c_r}, produced before this eor.
                bc_match = None
                for entry in ands_by_dst.get(bc_candidate, []):
                    if entry["ins"].idx >= items[k].idx:
                        continue
                    if set(entry["srcs"]) == {b_r, c_r}:
                        bc_match = entry
                        # Keep scanning to find the latest producer.
                if bc_match is None:
                    if t_tmp in items[k].regs_write:
                        break
                    continue
                final_eor = items[k]
                bc_and_used = bc_match
                final_dst = tk[1]
                bc_name = bc_candidate
                break
            if final_eor is None or bc_and_used is None:
                continue
            if final_eor.pc in seen_pcs:
                continue
            # Build the claim and verify.
            payload = {"op": "MAJ", "dst": final_dst,
                       "src": [and_a, b_r, c_r]}
            try:
                in_state = {
                    and_a: items[and_idx].regs_read[and_a],
                    b_r:   eor0.regs_read[b_r],
                    c_r:   eor0.regs_read[c_r],
                }
            except KeyError:
                continue
            res = self.verifier.check_handler_semantic(
                in_state, payload, final_eor.regs_write,
            )
            checked += 1
            v = res.verdict.value
            if v == "pass":
                passed += 1
                seen_pcs.add(final_eor.pc)
                try:
                    hid = self.submit_hypothesis(
                        kind="handler_semantic",
                        subject=f"maj@{final_eor.pc:#x}",
                        payload={
                            **payload,
                            "rationale": (
                                f"SHA-2 Maj(a,b,c) via "
                                f"eor⊕and⊕eor (BR-8 #2) at "
                                f"{eor0.pc:#x}/{items[and_idx].pc:#x}/"
                                f"{final_eor.pc:#x}. Verifier confirms."
                            ),
                            "anchor_pcs": [eor0.pc,
                                           items[and_idx].pc,
                                           final_eor.pc],
                        },
                        confidence=1.0,
                        source="s5_deterministic",
                        anchors=[(eor0.idx, final_eor.idx)],
                    )
                    conn = open_hypotheses_db(self.work)
                    try:
                        HypTree(conn).mark_verdict(hid, "pass", {
                            "strategy": res.strategy, **res.detail,
                        })
                    finally:
                        conn.close()
                    self.promote_to_finding(
                        hid, verifier_strategy="handler_semantic",
                        stage="s5-verify",
                    )
                    promoted += 1
                except (KeyError, ValueError):
                    pass
            elif v == "fail":
                failed += 1
            else:
                inconclusive += 1

        return {
            "stage":        "s5-verify-maj",
            "checked":      checked,
            "passed":       passed,
            "failed":       failed,
            "inconclusive": inconclusive,
            "promoted":     promoted,
        }

    def verify_and_promote_triton_simplifications(self) -> dict[str, Any]:
        """Triton symbolic-execution verifier path (0526Plan C1).

        Per-instruction Triton process: symbolize the live input registers,
        decode the instruction bytes, ask Triton for the dst register's
        symbolic AST, then evaluate the AST against the trace's concrete
        regs_read. If the result matches ins.regs_write[dst], wrap the AST
        in an eval_fn and run verifier.check_simplification — on PASS,
        promote a finding tagged source="s5_triton".

        This covers ARM shapes the verifier's _BIN_OPS / _UNI_OPS /
        _SRC2_EXTS / _BFX_OPS / _TRI_OPS tables don't model (FP, NEON, the
        MOVZ/MOVK constant-build chains, etc.) by leaning on Triton's
        AArch64 semantic model as ground truth. Dedupes by PC; PCs already
        promoted by the layer-0 passes are still re-checked but only
        contribute Triton-corroboration (the hyp tree's unique constraint
        on subject keeps the finding count honest).

        Returns the standard stage summary dict. When Triton bindings
        aren't importable on the host, the method returns checked=0 with
        `skipped_reason` set rather than raising — keeps script_mode and
        agent_mode wiring uniform across hosts.
        """
        from ..stages import s3_triton_symex

        if not s3_triton_symex.is_available():
            return {
                "stage":          "s5-verify-triton",
                "checked":        0,
                "passed":         0,
                "failed":         0,
                "inconclusive":   0,
                "promoted":       0,
                "skipped_reason": s3_triton_symex.unavailable_reason() or "Triton unavailable",
            }

        from triton import ARCH, MODE, TritonContext  # type: ignore
        from triton import Instruction as TritonInstr  # type: ignore

        checked = passed = promoted = 0
        decode_failed = model_mismatch = no_dst = 0
        seen_pcs: set[int] = set()

        # Triton is the "second cut" of the deterministic batch — its job is
        # to pick up shapes the layer-0 passes (binop/unary/imm/ext/bfx/ch)
        # can't model. Skip PCs they've already promoted so we count only
        # net Triton-coverage gain instead of double-counting.
        layer0_subject_prefixes = ("binop@", "unary@", "imm_binop@",
                                   "ext_binop@", "bfx@", "ch@")
        for h in self.get_hypotheses():
            if h.source != "s5_deterministic":
                continue
            if not h.subject.startswith(layer0_subject_prefixes):
                continue
            pc_str = h.subject.rsplit("@", 1)[-1]
            try:
                seen_pcs.add(int(pc_str, 16))
            except ValueError:
                continue

        for ins in self._items:
            if ins.pc in seen_pcs:
                continue
            if not ins.regs_write or not ins.bytes_:
                continue

            # Fresh per-instruction Triton ctx. Avoids state bleed from
            # earlier instructions; the verifier-level abstraction is
            # "this single instruction's semantics in isolation".
            ctx = TritonContext()
            ctx.setArchitecture(ARCH.AARCH64)
            ctx.setMode(MODE.CONSTANT_FOLDING, False)

            var_map = {}
            for reg_name in ins.regs_read:
                if reg_name in ("wzr", "xzr", "sp"):
                    continue
                try:
                    reg_obj = ctx.getRegister(reg_name)
                except Exception:
                    continue
                try:
                    sv = ctx.symbolizeRegister(reg_obj, f"in_{reg_name}")
                    ctx.setConcreteVariableValue(sv, ins.regs_read[reg_name])
                    var_map[reg_name] = sv
                except Exception:
                    continue

            t_inst = TritonInstr()
            t_inst.setAddress(ins.pc)
            try:
                t_inst.setOpcode(bytes(ins.bytes_))
                ctx.processing(t_inst)
            except Exception:
                decode_failed += 1
                continue

            chosen_dst = None
            for cand in ins.regs_write:
                if cand in ("wzr", "xzr", "sp"):
                    continue
                try:
                    ctx.getRegister(cand)
                except Exception:
                    continue
                chosen_dst = cand
                break
            if chosen_dst is None:
                no_dst += 1
                continue

            try:
                dst_reg = ctx.getRegister(chosen_dst)
                expr = ctx.getSymbolicRegister(dst_reg)
                ast = expr.getAst() if expr is not None else None
            except Exception:
                ast = None
            if ast is None:
                decode_failed += 1
                continue

            expected = ins.regs_write[chosen_dst]
            try:
                got = int(ast.evaluate())
            except Exception:
                decode_failed += 1
                continue
            # Fresh-ctx evaluation can't model status flags / memory
            # state — when the result disagrees with the trace, treat it as
            # "Triton model is missing context for this instruction" and
            # skip silently. (FAIL would be misleading since the trace is
            # ground truth.)
            if got != expected and (got & 0xFFFFFFFF) != (expected & 0xFFFFFFFF):
                model_mismatch += 1
                continue
            checked += 1

            def _eval_fn(reg_values, _ctx=ctx, _vars=var_map, _ast=ast):
                for r, v in reg_values.items():
                    if r in _vars:
                        try:
                            _ctx.setConcreteVariableValue(_vars[r], v)
                        except Exception:
                            pass
                return int(_ast.evaluate())

            res = self.verifier.check_simplification(
                _eval_fn, ins.regs_read, expected,
            )
            if res.verdict.value != "pass":
                # Defensive: we already matched manually, so this would only
                # fire on race/state-corruption. Don't promote anything.
                model_mismatch += 1
                continue
            passed += 1
            seen_pcs.add(ins.pc)
            try:
                hid = self.submit_hypothesis(
                    kind="handler_semantic",
                    subject=f"triton@{ins.pc:#x}",
                    payload={
                        "op":        "TRITON",
                        "dst":       chosen_dst,
                        "src":       sorted(var_map.keys()),
                        "ast":       str(expr)[:512],
                        "rationale": (
                            f"Triton symbolic execution of '{ins.mnemonic}' "
                            f"at pc={ins.pc:#x} matches the trace's "
                            f"regs_write[{chosen_dst}]."
                        ),
                    },
                    confidence=1.0,
                    source="s5_triton",
                    anchors=[(ins.idx, ins.idx)],
                )
                conn = open_hypotheses_db(self.work)
                try:
                    HypTree(conn).mark_verdict(hid, "pass", {
                        "strategy": res.strategy, **res.detail,
                    })
                finally:
                    conn.close()
                self.promote_to_finding(
                    hid, verifier_strategy="simplification",
                    stage="s5-verify-triton",
                )
                promoted += 1
            except (KeyError, ValueError):
                pass

        return {
            "stage":           "s5-verify-triton",
            "checked":         checked,
            "passed":          passed,
            "promoted":        promoted,
            "decode_failed":   decode_failed,
            "model_mismatch":  model_mismatch,
            "no_dst":          no_dst,
        }

    def verify_and_promote_sigma_idioms(self) -> dict[str, Any]:
        """SHA-2 σ / Σ fold-idiom discoverer (0526Plan C4.1 + C4.2).

        Scans the trace for the canonical clang/gcc-emitted σ / Σ form:

          ror|lsr A, x, #N1
          eor     B, A, x, <ror|lsr> #N2
          eor     C, B, x, <ror|lsr> #N3       ;  C may differ from A/B

        The (kind, amount) set of the three operations is matched against
        eight SHA-2 templates (σ0/σ1/Σ0/Σ1 in both 32-bit SHA-256 and 64-bit
        SHA-512 amounts). On match the discoverer also verifies the algebra
        against the trace's concrete x value, then promotes a single
        `kind=fold_idiom` finding tagged source="s5_fold_idiom" and groups
        the three constituent layer-0 findings under it via finding_groups.

        Two phases (BUG_REPORT-6 #4):
        - Phase 1 (contiguous): historical 3-insn back-to-back match.
        - Phase 2 (relaxed, per-BB): allows up to MAX_GAP intervening
          instructions between the three components, as long as the
          accumulator carried into each step is not overwritten. Catches
          ILP-scheduled SHA-2 compression rounds where Σ interleaves with
          Maj. Also relaxes the dst-stability requirement on the final
          instruction (#4a): the algebra check uses the actual dst written
          by inst[2], so a compiler-chosen output register is accepted.
        """
        import re as _re

        from ..stages.s1_segment import read_blocks as _read_s1_blocks
        from ..store import link_finding_group_members

        _reg_re = _re.compile(r"^[wx]\d+$|^wzr$|^xzr$")

        def _split(s: str) -> list[str]:
            return [t for t in _re.split(r"[,\s]+", s.strip()) if t]

        def _parse_imm(tok: str) -> int | None:
            if not tok.startswith("#"):
                return None
            body = tok[1:]
            try:
                return int(body, 16) if body.lower().startswith("0x") else int(body)
            except ValueError:
                return None

        # SHA-2 σ / Σ. Frozenset of (kind, amount) -> idiom name. We match on
        # the unordered set because compilers reorder the three accumulator
        # XORs freely.
        TEMPLATES = {
            frozenset([("ror", 2), ("ror", 13), ("ror", 22)]):  "SHA256.Sigma0",
            frozenset([("ror", 6), ("ror", 11), ("ror", 25)]):  "SHA256.Sigma1",
            frozenset([("ror", 7), ("ror", 18), ("lsr", 3)]):   "SHA256.sigma0",
            frozenset([("ror", 17), ("ror", 19), ("lsr", 10)]): "SHA256.sigma1",
            frozenset([("ror", 28), ("ror", 34), ("ror", 39)]): "SHA512.Sigma0",
            frozenset([("ror", 14), ("ror", 18), ("ror", 41)]): "SHA512.Sigma1",
            frozenset([("ror", 1), ("ror", 8), ("lsr", 7)]):    "SHA512.sigma0",
            frozenset([("ror", 19), ("ror", 61), ("lsr", 6)]):  "SHA512.sigma1",
        }

        MAX_GAP = 8   # tolerated insns between Phase-2 components

        checked = matched = algebra_mismatch = promoted = linked = 0
        # Dedupe across both phases: keyed on sorted (pc_a, pc_b, pc_c) so
        # an unrolled template hit at a unique PC triple isn't double-
        # counted. Pre-loaded from any existing `fold_idiom` findings so a
        # second call to this method (e.g. self_rescan re-running it) is a
        # no-op for already-promoted triples.
        seen_triples: set[tuple[int, ...]] = set()
        f_conn = open_findings_db(self.work)
        try:
            for (ref,) in f_conn.execute(
                "SELECT payload_ref FROM findings WHERE kind='fold_idiom'"
            ).fetchall():
                if not ref:
                    continue
                try:
                    pl = read_payload(f_conn, ref)
                except KeyError:
                    continue
                anchors = pl.get("anchor_pcs") or []
                pcs: list[int] = []
                for p in anchors:
                    if isinstance(p, int):
                        pcs.append(p)
                    elif isinstance(p, str):
                        try:
                            pcs.append(int(p, 16))
                        except ValueError:
                            pass
                if len(pcs) == 3:
                    seen_triples.add(tuple(sorted(pcs)))
        finally:
            f_conn.close()
        items = self._items

        def _parse_seed(ins):
            """ror|lsr dst, x, #N -> (op, dst, x, N) or None."""
            toks = _split(ins.mnemonic)
            if len(toks) != 4:
                return None
            op = toks[0].lower()
            if op not in ("ror", "lsr"):
                return None
            dst, xr = toks[1], toks[2]
            if not (_reg_re.match(dst) and _reg_re.match(xr)):
                return None
            n = _parse_imm(toks[3])
            if n is None:
                return None
            return op, dst, xr, n

        def _parse_eor_shifted(ins):
            """eor dst, lhs, x, ror|lsr #N -> (dst, lhs, x, kind, N) or None."""
            toks = _split(ins.mnemonic)
            if len(toks) != 6 or toks[0].lower() != "eor":
                return None
            kind = toks[4].lower()
            if kind not in ("ror", "lsr"):
                return None
            n = _parse_imm(toks[5])
            if n is None:
                return None
            return toks[1], toks[2], toks[3], kind, n

        def _algebra(kind1, n1, kind2, n2, kind3, n3, x_val, width):
            mask = (1 << width) - 1
            x_val &= mask

            def _apply(kind, n, val):
                if kind == "ror":
                    sh = n & (width - 1)
                    if sh == 0:
                        return val
                    return ((val >> sh) | ((val << (width - sh)) & mask)) & mask
                return (val & mask) >> (n & (width - 1))

            return (_apply(kind1, n1, x_val) ^ _apply(kind2, n2, x_val)
                    ^ _apply(kind3, n3, x_val)) & mask

        def _try_match(ins_a, ins_b, ins_c,
                       op_a, n1, x_reg_a, acc_a,
                       kind2, n2, kind3, n3, final_dst):
            """Attempt to promote one fold-idiom candidate. Returns True on match."""
            nonlocal checked, matched, algebra_mismatch, promoted, linked
            # Canonicalize by sorted PC-tuple so Phase 1 / Phase 2 / Phase 3
            # share a single dedup key, and a hot loop running the same
            # triple every iteration only promotes once.
            triple_key = tuple(sorted((ins_a.pc, ins_b.pc, ins_c.pc)))
            if triple_key in seen_triples:
                return False
            sig = frozenset([(op_a, n1), (kind2, n2), (kind3, n3)])
            idiom = TEMPLATES.get(sig)
            if idiom is None:
                return False
            checked += 1
            try:
                x_val = ins_a.regs_read[x_reg_a]
            except KeyError:
                return False
            width = 32 if acc_a.startswith("w") else 64
            mask = (1 << width) - 1
            computed = _algebra(op_a, n1, kind2, n2, kind3, n3, x_val, width)
            expected = ins_c.regs_write.get(final_dst, 0) & mask
            if computed != expected:
                algebra_mismatch += 1
                return False
            matched += 1
            seen_triples.add(triple_key)

            try:
                hid = self.submit_hypothesis(
                    kind="fold_idiom",
                    subject=f"{idiom}@{ins_c.pc:#x}",
                    payload={
                        "idiom":      idiom,
                        "input_reg":  x_reg_a,
                        "dst_reg":    final_dst,
                        "anchor_pcs": [ins_a.pc, ins_b.pc, ins_c.pc],
                        "components": [
                            {"kind": op_a,  "amount": n1, "pc": f"0x{ins_a.pc:x}"},
                            {"kind": kind2, "amount": n2, "pc": f"0x{ins_b.pc:x}"},
                            {"kind": kind3, "amount": n3, "pc": f"0x{ins_c.pc:x}"},
                        ],
                        "rationale": (
                            f"3-insn σ/Σ idiom matched at "
                            f"{ins_a.pc:#x}/{ins_b.pc:#x}/{ins_c.pc:#x}: "
                            f"{op_a}({x_reg_a},#{n1}) ^ {kind2}({x_reg_a},#{n2}) "
                            f"^ {kind3}({x_reg_a},#{n3}) = {idiom}({x_reg_a}); "
                            f"computed=0x{computed:x} matches trace dst."
                        ),
                    },
                    confidence=1.0,
                    source="s5_fold_idiom",
                    anchors=[(ins_a.idx, ins_c.idx)],
                )
                conn_h = open_hypotheses_db(self.work)
                try:
                    HypTree(conn_h).mark_verdict(hid, "pass", {
                        "strategy": "algebraic_idiom_match",
                        "computed": f"0x{computed:x}",
                        "expected": f"0x{expected:x}",
                    })
                finally:
                    conn_h.close()
                finding_id = self.promote_to_finding(
                    hid, verifier_strategy="algebraic_idiom_match",
                    stage="s5-fold",
                )
                promoted += 1

                # Best-effort: link the three constituent layer-0 findings.
                # Layer-0 promote pre-empts us in script_mode, so they should
                # already be in findings.sqlite under subjects like
                # `imm_binop@<pc>` / `ext_binop@<pc>`.
                f_conn = open_findings_db(self.work)
                try:
                    members = []
                    for ins_e, role in (
                        (ins_a, f"{op_a}_{n1}"),
                        (ins_b, f"{kind2}_{n2}"),
                        (ins_c, f"{kind3}_{n3}"),
                    ):
                        row = f_conn.execute(
                            "SELECT id FROM findings "
                            "WHERE subject LIKE ? AND kind = 'handler_semantic' "
                            "ORDER BY id ASC LIMIT 1",
                            (f"%@0x{ins_e.pc:x}",),
                        ).fetchone()
                        if row:
                            members.append((row[0], role))
                    if members:
                        link_finding_group_members(
                            f_conn,
                            parent_finding_id=finding_id,
                            idiom_name=idiom,
                            members=members,
                        )
                        linked += len(members)
                finally:
                    f_conn.close()
                return True
            except (KeyError, ValueError):
                return False

        # ---- Phase 1: contiguous 3-insn window ----
        # Same shape as the historical matcher, modulo Bug #4a: inst[2] may
        # write a register other than the original accumulator, and the
        # algebra check now uses inst[2]'s actual dst.
        for i in range(len(items) - 2):
            seed = _parse_seed(items[i])
            if seed is None:
                continue
            op_a, acc_a, x_reg, n1 = seed
            e1 = _parse_eor_shifted(items[i + 1])
            if e1 is None:
                continue
            b_dst, b_lhs, b_x, kind2, n2 = e1
            if b_lhs != acc_a or b_x != x_reg:
                continue
            e2 = _parse_eor_shifted(items[i + 2])
            if e2 is None:
                continue
            c_dst, c_lhs, c_x, kind3, n3 = e2
            if c_lhs != b_dst or c_x != x_reg:
                continue
            _try_match(items[i], items[i + 1], items[i + 2],
                       op_a, n1, x_reg, acc_a,
                       kind2, n2, kind3, n3, c_dst)

        # ---- Phase 2: relaxed within-BB scan (BUG_REPORT-6 #4b) ----
        # Allow up to MAX_GAP unrelated instructions between the three σ/Σ
        # components, provided the running accumulator is not overwritten by
        # an intervening instruction. Catches ILP-scheduled SHA-2 rounds
        # where Σ pieces interleave with Maj.
        try:
            blocks = _read_s1_blocks(self.work)
        except FileNotFoundError:
            blocks = []

        for bb in blocks:
            lo = bb.instr_idx_start
            hi = bb.instr_idx_end   # inclusive
            if hi - lo < 2:
                continue
            for i in range(lo, hi - 1):
                seed = _parse_seed(items[i])
                if seed is None:
                    continue
                op_a, acc_a, x_reg, n1 = seed
                j_max = min(hi, i + 1 + MAX_GAP)
                # Walk forward looking for the second component.
                for j in range(i + 1, j_max + 1):
                    insj = items[j]
                    e1 = _parse_eor_shifted(insj)
                    matched_b = False
                    if e1 is not None:
                        b_dst, b_lhs, b_x, kind2, n2 = e1
                        if b_lhs == acc_a and b_x == x_reg:
                            matched_b = True
                            # Walk forward from j looking for the third.
                            k_max = min(hi, j + 1 + MAX_GAP)
                            for k in range(j + 1, k_max + 1):
                                insk = items[k]
                                e2 = _parse_eor_shifted(insk)
                                if e2 is not None:
                                    c_dst, c_lhs, c_x, kind3, n3 = e2
                                    if c_lhs == b_dst and c_x == x_reg:
                                        _try_match(
                                            items[i], items[j], items[k],
                                            op_a, n1, x_reg, acc_a,
                                            kind2, n2, kind3, n3, c_dst,
                                        )
                                        break
                                if b_dst in insk.regs_write:
                                    break  # b_dst clobbered before chain closed
                            break  # done with this (i, j) seed pair
                    if not matched_b and acc_a in insj.regs_write:
                        break  # acc_a clobbered before chain started

        # ---- Phase 3: DFG-grouped (BUG_REPORT-8 #1) ----
        # Phase 1/2 chain along acc_a within a BB; they miss when σ/Σ pieces
        # straddle a BB boundary (TC1 SHA-256 round body), or the final write
        # targets the input register (TC2 SHA-512 σ₁ dst==input). Phase 3
        # collects every rotate/shift acting on a given input register —
        # including the shifted-source operand of `eor d, lhs, x, <rot|shr>` —
        # and looks for (kind, amount)-frozenset triples that match a
        # template. The algebra check + final-dst value act as the falsity
        # filter, so we don't need the accumulator-chain constraint.
        from collections import defaultdict as _defaultdict
        WINDOW = 32   # max idx-span between earliest and latest op of a triple
        ops_by_input: dict[str, list[dict]] = _defaultdict(list)
        for ins in items:
            seed = _parse_seed(ins)
            if seed is not None:
                op_kind, dst_seed, xr_seed, n_seed = seed
                ops_by_input[xr_seed].append({
                    "idx": ins.idx, "ins": ins,
                    "kind": op_kind, "amount": n_seed, "dst": dst_seed,
                })
            eor = _parse_eor_shifted(ins)
            if eor is not None:
                dst_e, _lhs_e, xr_e, kind2_e, n2_e = eor
                ops_by_input[xr_e].append({
                    "idx": ins.idx, "ins": ins,
                    "kind": kind2_e, "amount": n2_e, "dst": dst_e,
                })

        for sig, idiom_name in TEMPLATES.items():
            pieces = list(sig)
            # Templates pin to a width: SHA256.* → 32-bit, SHA512.* → 64-bit.
            # Reject any candidate whose final-dst width disagrees — otherwise
            # a SHA-256 σ₁ amount set could spuriously match a 64-bit dst
            # where the algebra happens to coincide.
            width_hint = 32 if idiom_name.startswith("SHA256") else \
                (64 if idiom_name.startswith("SHA512") else None)
            for xr, ops in ops_by_input.items():
                by_ka: dict[tuple[str, int], list[dict]] = _defaultdict(list)
                for o in ops:
                    by_ka[(o["kind"], o["amount"])].append(o)
                if not all(by_ka.get(p) for p in pieces):
                    continue
                seeds_p = by_ka[pieces[0]]
                for o1 in seeds_p:
                    lo, hi = o1["idx"] - WINDOW, o1["idx"] + WINDOW
                    c2 = [o for o in by_ka[pieces[1]]
                          if o is not o1 and lo <= o["idx"] <= hi]
                    c3 = [o for o in by_ka[pieces[2]]
                          if o is not o1 and lo <= o["idx"] <= hi]
                    if not c2 or not c3:
                        continue
                    for o2 in c2:
                        for o3 in c3:
                            if o3 is o2:
                                continue
                            triple = sorted([o1, o2, o3], key=lambda o: o["idx"])
                            if triple[-1]["idx"] - triple[0]["idx"] > WINDOW:
                                continue
                            # Dedupe by sorted PC-tuple: hot loops (SHA-512
                            # sigma0 runs 256x per probe) execute the same
                            # PC triple over and over. Phase 3 would
                            # otherwise emit one fold_idiom per execution.
                            triple_key_pcs = tuple(sorted(
                                t["ins"].pc for t in triple
                            ))
                            if triple_key_pcs in seen_triples:
                                continue
                            final_op = triple[-1]
                            dst_name = final_op["dst"]
                            width = 32 if dst_name.startswith("w") else 64
                            if width_hint is not None and width_hint != width:
                                continue
                            mask = (1 << width) - 1
                            # x snapshot: earliest op's regs_read; if absent
                            # there (unusual), fall back to the final op's.
                            x_val = triple[0]["ins"].regs_read.get(xr)
                            if x_val is None:
                                x_val = final_op["ins"].regs_read.get(xr)
                            if x_val is None:
                                continue
                            computed = _algebra(
                                triple[0]["kind"], triple[0]["amount"],
                                triple[1]["kind"], triple[1]["amount"],
                                triple[2]["kind"], triple[2]["amount"],
                                x_val, width,
                            )
                            expected = final_op["ins"].regs_write.get(
                                dst_name, 0) & mask
                            checked += 1
                            if computed != expected:
                                algebra_mismatch += 1
                                continue
                            matched += 1
                            seen_triples.add(triple_key_pcs)
                            ins_a, ins_b, ins_c = (triple[0]["ins"],
                                                   triple[1]["ins"],
                                                   triple[2]["ins"])
                            try:
                                hid = self.submit_hypothesis(
                                    kind="fold_idiom",
                                    subject=f"{idiom_name}@{ins_c.pc:#x}",
                                    payload={
                                        "idiom":      idiom_name,
                                        "input_reg":  xr,
                                        "dst_reg":    dst_name,
                                        "anchor_pcs": [ins_a.pc, ins_b.pc, ins_c.pc],
                                        "components": [
                                            {"kind": t["kind"], "amount": t["amount"],
                                             "pc": f"0x{t['ins'].pc:x}"}
                                            for t in triple
                                        ],
                                        "rationale": (
                                            f"σ/Σ DFG-grouped match "
                                            f"(Phase 3, BR-8 #1): {idiom_name}({xr}) "
                                            f"at {ins_a.pc:#x}/{ins_b.pc:#x}/{ins_c.pc:#x}; "
                                            f"computed=0x{computed:x} matches trace dst."
                                        ),
                                    },
                                    confidence=1.0,
                                    source="s5_fold_idiom",
                                    anchors=[(ins_a.idx, ins_c.idx)],
                                )
                                conn_h = open_hypotheses_db(self.work)
                                try:
                                    HypTree(conn_h).mark_verdict(hid, "pass", {
                                        "strategy": "algebraic_idiom_match_dfg",
                                        "computed": f"0x{computed:x}",
                                        "expected": f"0x{expected:x}",
                                    })
                                finally:
                                    conn_h.close()
                                self.promote_to_finding(
                                    hid,
                                    verifier_strategy="algebraic_idiom_match_dfg",
                                    stage="s5-fold",
                                )
                                promoted += 1
                            except (KeyError, ValueError):
                                pass

        return {
            "stage":             "s5-fold-sigma",
            "checked":           checked,
            "matched":           matched,
            "algebra_mismatch":  algebra_mismatch,
            "promoted":          promoted,
            "linked_members":    linked,
        }

    # 0527 preprocess batch: one-call deterministic chain + tagged findings + hints.
    # Pass names are short tokens agents can mix-and-match (skip e.g. "triton" on
    # hosts without bindings, or "sigma"+"algorithm" if the agent does its own
