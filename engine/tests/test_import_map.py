"""#4+#6 — auto PLT/import map + external-function summaries.

Synthetic / artifact-driven (no live binary required for the core cases).
Regression fixtures map 1:1 to spec_tc2_import_map_extern_summary "Regression
fixtures" + the A8④ degenerate paths.
"""

from __future__ import annotations

from engine.import_map import (
    EXTERN_SUMMARIES,
    ExternSummary,
    ImportMap,
    annotate_calls,
    build_import_map,
    extern_summary,
)
from engine.types import Instruction


def _ins(idx, mnem, reads=None, pc=None):
    return Instruction(idx=idx, pc=pc if pc is not None else 0x400000 + idx * 4,
                       bytes_=b"\x00\x00\x00\x00", mnemonic=mnem,
                       regs_read=dict(reads or {}), regs_write={}, mem=())


# Pre-captured objdump / readelf text artifacts (no binary on disk needed).
_DISASM_PLT = """
Disassembly of section .plt:

0000000000400a80 <rand@plt>:
  400a80:\tadrp x16, ...
  400a84:\tbr x17

0000000000400a90 <memcpy@plt>:
  400a90:\tadrp x16, ...
  400a94:\tbr x17

0000000000400aa0 <time@plt>:
  400aa0:\tadrp x16, ...
"""

_RELOCS = """
Relocation section '.rela.plt' at offset 0x500 contains 3 entries:
  Offset          Info           Type           Sym. Value    Sym. Name + Addend
000000411038  000000060402 R_AARCH64_JUMP_SLO 0000000000000000 rand + 0
000000411040  000000070402 R_AARCH64_JUMP_SLO 0000000000000000 memcpy + 0
000000411048  000000080402 R_AARCH64_JUMP_SLO 0000000000000000 time + 0
"""


# --- #4 import map -----------------------------------------------------------

def test_plt_map_resolves_known_stub_to_symbol():
    im = build_import_map(static_artifacts={"disasm": _DISASM_PLT, "relocs": _RELOCS})
    assert im.binary_available
    assert im.by_plt_addr[0x400a80] == "rand"
    assert im.by_plt_addr[0x400a90] == "memcpy"
    assert im.by_got[0x411038] == "rand"
    assert im.symbol_for(0x400a80) == "rand"


def test_annotated_trace_shows_rand_at_plt():
    im = build_import_map(static_artifacts={"disasm": _DISASM_PLT})
    # A direct bl into the rand PLT stub.
    trace = [_ins(0, "bl 0x400a80"), _ins(1, "mov x0, x0")]
    anns = annotate_calls(trace, im)
    assert len(anns) == 1
    assert anns[0]["symbol"] == "rand@plt"
    assert anns[0]["resolved_from"] == "plt"
    assert anns[0]["external_state"] is True
    assert anns[0]["state_kind"] == "prng"


def test_indirect_blr_resolved_from_register_value():
    im = build_import_map(static_artifacts={"disasm": _DISASM_PLT})
    # blr x8 with x8 = the memcpy PLT stub (resolved from the concrete trace).
    trace = [_ins(0, "blr x8", reads={"x8": 0x400a90})]
    anns = annotate_calls(trace, im)
    assert anns[0]["symbol"] == "memcpy@plt"
    assert anns[0]["external_state"] is False


def test_stripped_no_binary_yields_unresolved_unknown_no_fabrication():
    # No binary, no artifacts, no override → binary_available False, empty maps.
    im = build_import_map(binary_path=None)
    assert im.binary_available is False
    trace = [_ins(0, "bl 0x400a80")]
    anns = annotate_calls(trace, im)
    # Target resolved from the trace, but no symbol → unknown@<addr>, never guessed.
    assert anns[0]["symbol"] == "unknown@0x400a80"
    assert anns[0]["resolved_from"] == "no_symbol"
    assert anns[0]["external_state"] is None


def test_missing_binary_path_is_honest_not_a_crash():
    im = build_import_map(binary_path="/no/such/binary/xyz")
    assert im.binary_available is False
    assert "not found" in im.detail


def test_unresolvable_call_target_annotated_unresolved():
    im = build_import_map(static_artifacts={"disasm": _DISASM_PLT})
    # blr x8 with no captured x8 value → target unresolvable.
    trace = [_ins(0, "blr x8")]
    anns = annotate_calls(trace, im)
    assert anns[0]["symbol"] == "unknown@<unresolved>"
    assert anns[0]["resolved_from"] == "unresolved_target"


def test_explicit_plt_map_override_used_verbatim():
    im = build_import_map(plt_map={0x1234: "srand"})
    assert im.source == "explicit_override"
    assert im.symbol_for(0x1234) == "srand"


# --- #6 external summaries ---------------------------------------------------

def test_extern_summary_memcpy_abi_mapping_feeds_5():
    s = extern_summary("memcpy")
    assert s is not None
    assert s.role_reg("dst") == "x0"
    assert s.role_reg("src") == "x1"
    assert s.role_reg("n") == "x2"
    assert s.introduces_external_state is False


def test_extern_summary_rand_introduces_external_prng_state():
    s = extern_summary("rand")
    assert s.introduces_external_state is True
    assert s.state_kind == "prng"


def test_extern_summary_strips_plt_suffix():
    assert extern_summary("rand@plt") is EXTERN_SUMMARIES["rand"]


def test_extern_summary_unknown_symbol_returns_none_no_fabrication():
    assert extern_summary("totally_unknown_fn") is None


def test_time_summary_is_time_state():
    assert extern_summary("time").state_kind == "time"
    assert extern_summary("time").introduces_external_state is True


def test_memset_const_summary_has_c_role_not_src():
    s = extern_summary("memset")
    assert s.role_reg("dst") == "x0"
    assert s.role_reg("c") == "x1"
    assert s.role_reg("src") is None


def test_known_symbol_without_summary_tagged_external_unknown():
    # A symbol resolved by the import map but NOT in the #6 table → external_unknown
    # (introduces external state of UNKNOWN kind), never assumed pure (A8④).
    im = build_import_map(plt_map={0x5000: "some_libc_fn"})
    trace = [_ins(0, "bl 0x5000")]
    anns = annotate_calls(trace, im)
    assert anns[0]["symbol"] == "some_libc_fn@plt"
    assert anns[0]["external_state"] == "external_unknown"
    assert anns[0]["state_kind"] == "unknown"


def test_import_map_to_dict_round_shapes():
    im = build_import_map(static_artifacts={"disasm": _DISASM_PLT, "relocs": _RELOCS})
    d = im.to_dict()
    assert d["by_plt_addr"]["0x400a80"] == "rand"
    assert d["binary_available"] is True
