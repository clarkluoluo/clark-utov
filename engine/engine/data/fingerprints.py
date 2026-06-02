"""Cryptographic constant fingerprint catalog for S1.5 indexing.

Ported from algokiller-plugin tools/search/search.c (Sprint 4 / constscan,
MIT cloudza 2026, see NOTICE).

Each scalar entry: (name, category, confidence, magic_hex)
  - category   : hash | cipher_sym | ecc | crc | mac
  - confidence : STRONG  (unique, RFC/standard-verified, real-trace 0 FP)
                 MEDIUM  (algorithm-specific but somewhat short or shared with
                          adjacent primitives — e.g. SHA-256 IV ↔ BLAKE2s IV)
                 WEAK    (known to overlap with general-purpose code)

References (verified):
  MD5/SHA-1/SHA-256 init  : RFC 6234 §5.3.{1,3,4}
  SHA-512 init            : RFC 6234 §5.3.5
  AES S-box / Te0         : FIPS 197 §5.1.1 + §5.2
  SHA-3 / Keccak RC       : FIPS 202 §3.2.5
  ChaCha20 sigma          : RFC 8439 §2.3
  SM3 IV / T_j            : GM/T 0004-2012
  SM4 FK / CK             : GM/T 0002-2012
  TEA delta               : Wheeler & Needham 1994
  CRC32                   : IEEE 802.3 §3.2.8, zlib
  FNV-1a 64               : http://isthe.com/chongo/tech/comp/fnv/
  P-256                   : FIPS 186-4 §D.1.2.3
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Confidence(str, Enum):
    STRONG = "strong"
    MEDIUM = "medium"
    WEAK = "weak"


@dataclass(frozen=True)
class Fingerprint:
    name: str
    category: str   # hash | cipher_sym | ecc | crc | mac
    confidence: Confidence
    magic: int      # the constant value


@dataclass(frozen=True)
class InstructionPattern:
    """Substring match against disassembly text — for NEON / SIMD constructions
    that the scalar Fingerprint table cannot see (e.g. HMAC pad via movi).
    """
    name: str
    category: str
    confidence: Confidence
    match_text: str            # substring of disasm
    primitive: str             # logical primitive this points to
    interpretation: str        # short human-readable explanation


STRONG = Confidence.STRONG
MEDIUM = Confidence.MEDIUM
WEAK = Confidence.WEAK


# 95 scalar fingerprints.
FINGERPRINTS: tuple[Fingerprint, ...] = (

    # ---- Hash: MD5 init quartet ----
    Fingerprint("MD5.A",          "hash",       STRONG, 0x67452301),
    Fingerprint("MD5.B",          "hash",       STRONG, 0xefcdab89),
    Fingerprint("MD5.C",          "hash",       STRONG, 0x98badcfe),
    Fingerprint("MD5.D",          "hash",       STRONG, 0x10325476),

    # ---- Hash: MD5 T table (RFC 1321 §3.4)
    # IVs fire once per init; T fires 64x per block compression.
    Fingerprint("MD5.T[1]",       "hash",       STRONG, 0xd76aa478),
    Fingerprint("MD5.T[2]",       "hash",       STRONG, 0xe8c7b756),
    Fingerprint("MD5.T[3]",       "hash",       STRONG, 0x242070db),
    Fingerprint("MD5.T[4]",       "hash",       STRONG, 0xc1bdceee),

    # ---- Hash: SHA-1 ----
    Fingerprint("SHA1.h4",        "hash",       STRONG, 0xc3d2e1f0),
    Fingerprint("SHA1.K[0..19]",  "hash",       STRONG, 0x5a827999),
    Fingerprint("SHA1.K[20..39]", "hash",       STRONG, 0x6ed9eba1),
    Fingerprint("SHA1.K[40..59]", "hash",       STRONG, 0x8f1bbcdc),
    Fingerprint("SHA1.K[60..79]", "hash",       STRONG, 0xca62c1d6),

    # ---- Hash: SHA-256 (BLAKE2s IV identical → medium) ----
    Fingerprint("SHA256.h0",      "hash",       MEDIUM, 0x6a09e667),
    Fingerprint("SHA256.h1",      "hash",       MEDIUM, 0xbb67ae85),
    Fingerprint("SHA256.h2",      "hash",       MEDIUM, 0x3c6ef372),
    Fingerprint("SHA256.h3",      "hash",       MEDIUM, 0xa54ff53a),
    Fingerprint("SHA256.h4",      "hash",       MEDIUM, 0x510e527f),
    Fingerprint("SHA256.h5",      "hash",       MEDIUM, 0x9b05688c),
    Fingerprint("SHA256.h6",      "hash",       MEDIUM, 0x1f83d9ab),
    Fingerprint("SHA256.h7",      "hash",       MEDIUM, 0x5be0cd19),

    # ---- Hash: SHA-256 K table (first 8 as sentinel; full K[0..63] in FIPS 180-4 Appx A) ----
    Fingerprint("SHA256.K[0]",    "hash",       STRONG, 0x428a2f98),
    Fingerprint("SHA256.K[1]",    "hash",       STRONG, 0x71374491),
    Fingerprint("SHA256.K[2]",    "hash",       STRONG, 0xb5c0fbcf),
    Fingerprint("SHA256.K[3]",    "hash",       STRONG, 0xe9b5dba5),
    Fingerprint("SHA256.K[4]",    "hash",       STRONG, 0x3956c25b),
    Fingerprint("SHA256.K[5]",    "hash",       STRONG, 0x59f111f1),
    Fingerprint("SHA256.K[6]",    "hash",       STRONG, 0x923f82a4),
    Fingerprint("SHA256.K[7]",    "hash",       STRONG, 0xab1c5ed5),

    # ---- Hash: SHA-512 (BLAKE2b IV identical → medium) ----
    Fingerprint("SHA512.h0",      "hash",       MEDIUM, 0x6a09e667f3bcc908),
    Fingerprint("SHA512.h1",      "hash",       MEDIUM, 0xbb67ae8584caa73b),
    Fingerprint("SHA512.h2",      "hash",       MEDIUM, 0x3c6ef372fe94f82b),
    Fingerprint("SHA512.h3",      "hash",       MEDIUM, 0xa54ff53a5f1d36f1),
    Fingerprint("SHA512.h4",      "hash",       MEDIUM, 0x510e527fade682d1),
    Fingerprint("SHA512.h5",      "hash",       MEDIUM, 0x9b05688c2b3e6c1f),
    Fingerprint("SHA512.h6",      "hash",       MEDIUM, 0x1f83d9abfb41bd6b),
    Fingerprint("SHA512.h7",      "hash",       MEDIUM, 0x5be0cd19137e2179),

    # ---- Hash: SM3 IV (GM/T 0004-2012) ----
    Fingerprint("SM3.IV0",        "hash",       STRONG, 0x7380166f),
    Fingerprint("SM3.IV1",        "hash",       STRONG, 0x4914b2b9),
    Fingerprint("SM3.IV2",        "hash",       STRONG, 0x172442d7),
    Fingerprint("SM3.IV3",        "hash",       STRONG, 0xda8a0600),
    Fingerprint("SM3.IV4",        "hash",       STRONG, 0xa96f30bc),
    Fingerprint("SM3.IV5",        "hash",       STRONG, 0x163138aa),
    Fingerprint("SM3.IV6",        "hash",       STRONG, 0xe38dee4d),
    Fingerprint("SM3.IV7",        "hash",       STRONG, 0xb0fb0e4e),

    # ---- Hash: SM3 round constants T_j ----
    Fingerprint("SM3.T_j[0..15]", "hash",       STRONG, 0x79cc4519),
    Fingerprint("SM3.T_j[16..63]","hash",       STRONG, 0x7a879d8a),

    # ---- Hash: SHA-3 / Keccak round constants (skip RC[0]=0x01, too generic) ----
    Fingerprint("SHA3.RC[1]",     "hash",       STRONG, 0x0000000000008082),
    Fingerprint("SHA3.RC[2]",     "hash",       STRONG, 0x800000000000808a),
    Fingerprint("SHA3.RC[4]",     "hash",       STRONG, 0x000000000000808b),

    # ---- Hash: FNV-1a 64-bit ----
    Fingerprint("FNV1a.prime64",  "hash",       WEAK,   0x100000001b3),
    Fingerprint("FNV1a.offset64", "hash",       WEAK,   0xcbf29ce484222325),

    # ---- Cipher: AES — byte-array S-box (raw bytes) ----
    Fingerprint("AES.sbox_bytes[0..3]", "cipher_sym", MEDIUM, 0x637c777b),
    Fingerprint("AES.sbox_bytes[4..7]", "cipher_sym", MEDIUM, 0xf26b6fc5),
    Fingerprint("AES.inv_sbox_bytes",   "cipher_sym", MEDIUM, 0x52096ad5),

    # ---- Cipher: AES — T-table Te0[0..3] ----
    Fingerprint("AES.Te0[0]",     "cipher_sym", STRONG, 0xc66363a5),
    Fingerprint("AES.Te0[1]",     "cipher_sym", STRONG, 0xf87c7c84),
    Fingerprint("AES.Te0[2]",     "cipher_sym", STRONG, 0xee777799),
    Fingerprint("AES.Te0[3]",     "cipher_sym", STRONG, 0xf67b7b8d),

    # ---- Cipher: SM4 (国密) ----
    Fingerprint("SM4.sbox[0..3]", "cipher_sym", STRONG, 0xd690e9fe),
    Fingerprint("SM4.sbox[4..7]", "cipher_sym", STRONG, 0xcce13db7),
    Fingerprint("SM4.FK0",        "cipher_sym", STRONG, 0xa3b1bac6),
    Fingerprint("SM4.FK1",        "cipher_sym", STRONG, 0x56aa3350),
    Fingerprint("SM4.CK[0]",      "cipher_sym", STRONG, 0x00070e15),
    Fingerprint("SM4.CK[1]",      "cipher_sym", STRONG, 0x1c232a31),
    Fingerprint("SM4.CK[2]",      "cipher_sym", STRONG, 0x383f464d),
    Fingerprint("SM4.CK[3]",      "cipher_sym", STRONG, 0x545b6269),

    # ---- Cipher: ChaCha20 / Salsa20 sigma ----
    Fingerprint("ChaCha20.sigma[0]", "cipher_sym", STRONG, 0x61707865),
    Fingerprint("ChaCha20.sigma[1]", "cipher_sym", STRONG, 0x3320646e),
    Fingerprint("ChaCha20.sigma[2]", "cipher_sym", STRONG, 0x79622d32),
    Fingerprint("ChaCha20.sigma[3]", "cipher_sym", STRONG, 0x6b206574),

    # ---- Cipher: TEA family (delta also used by Knuth hash / xxHash) ----
    Fingerprint("TEA.delta",      "cipher_sym", MEDIUM, 0x9e3779b9),

    # ---- Cipher hint: Whirlpool S-box first 4 bytes ----
    Fingerprint("Whirlpool.S[0..3]", "cipher_sym", WEAK, 0x18233481),

    # ---- Cipher: DES (FIPS 46-3) ----
    # const0/const1/shifted0/shifted1 are library-specific PC / SP-box artefacts,
    # not in FIPS spec — kept WEAK pending real-trace corroboration.
    Fingerprint("DES.const0",     "cipher_sym", WEAK,   0xfee1a2b3),
    Fingerprint("DES.const1",     "cipher_sym", WEAK,   0xd7bef080),
    Fingerprint("DES.shifted0",   "cipher_sym", WEAK,   0x3a322a22),
    Fingerprint("DES.shifted1",   "cipher_sym", WEAK,   0x2a223a32),
    Fingerprint("DES.sbox_word[0]","cipher_sym", MEDIUM, 0x2c1e241b),
    Fingerprint("DES.sbox_word[1]","cipher_sym", MEDIUM, 0x5a7f361d),
    Fingerprint("DES.sbox_word[2]","cipher_sym", MEDIUM, 0x3d4793c6),
    Fingerprint("DES.sbox_word[3]","cipher_sym", MEDIUM, 0x0b0eedf8),

    # ---- MAC: Poly1305 r-mask clamp ----
    Fingerprint("Poly1305.clamp_lo", "mac",     STRONG, 0x0ffffffc0fffffff),
    Fingerprint("Poly1305.clamp_hi", "mac",     STRONG, 0x0ffffffc0ffffffc),

    # ---- MAC: SipHash IV ----
    Fingerprint("SipHash.k0",     "mac",        STRONG, 0x736f6d6570736575),
    Fingerprint("SipHash.k1",     "mac",        STRONG, 0x646f72616e646f6d),

    # ---- MAC: HMAC ipad / opad (scalar form; SIMD form in INSTR_PATTERNS) ----
    Fingerprint("HMAC.ipad",      "mac",        STRONG, 0x36363636),
    Fingerprint("HMAC.opad",      "mac",        STRONG, 0x5c5c5c5c),

    # ---- CRC: 32-bit polynomial constants ----
    Fingerprint("CRC32.poly_reflected", "crc",  STRONG, 0xedb88320),
    Fingerprint("CRC32.poly_normal",    "crc",  STRONG, 0x04c11db7),

    # ---- ECC: NIST P-256 ----
    Fingerprint("P256.order_low[0]", "ecc",     STRONG, 0xbce6faada7179e84),
    Fingerprint("P256.order_low[1]", "ecc",     STRONG, 0xf3b9cac2fc632551),
    Fingerprint("P256.b_lo",      "ecc",        STRONG, 0xcc53b0f63bce3c3e),

    # ---- ECC: secp256k1 ----
    Fingerprint("secp256k1.p_lo", "ecc",        STRONG, 0xfffffffefffffc2f),

    # ---- ECC: Ed25519 d_lo (distinguishes Ed25519 from X25519/Curve25519) ----
    Fingerprint("Ed25519.d_lo",   "ecc",        STRONG, 0x52036cee2b6ffe73),

    # ---- ECC: Curve25519 ladder constant a24 ----
    Fingerprint("Curve25519.a24", "ecc",        MEDIUM, 0x1db41),
)


# NEON / SIMD instruction-pattern fingerprints (substring matches on disasm text).
# These catch constructions the scalar magic table misses.
INSTR_PATTERNS: tuple[InstructionPattern, ...] = (
    InstructionPattern(
        name="HMAC.ipad.simd_movi",
        category="mac",
        confidence=STRONG,
        match_text=".16b, #0x36",
        primitive="HMAC.ipad",
        interpretation=(
            "NEON broadcast of 0x36 across 16 bytes — canonical HMAC-* ipad "
            "initialisation. Each hit corresponds to ONE HMAC inner-pad setup. "
            "Use this as the upper bound on HMAC operation count."
        ),
    ),
    InstructionPattern(
        name="HMAC.opad.simd_movi",
        category="mac",
        confidence=STRONG,
        match_text=".16b, #0x5c",
        primitive="HMAC.opad",
        interpretation=(
            "NEON broadcast of 0x5c across 16 bytes — canonical HMAC-* opad "
            "initialisation. Pair with ipad.simd_movi to confirm full HMAC pad "
            "construction."
        ),
    ),
)


# Per-block primitives: these magic values fire once per compression block, not once per init.
# Used to estimate block count from total_hits when reporting findings.
PER_BLOCK_PRIMITIVE: dict[str, str] = {
    "MD5.T[1]":         "MD5",
    "MD5.T[2]":         "MD5",
    "MD5.T[3]":         "MD5",
    "MD5.T[4]":         "MD5",
    "SHA256.K[0]":      "SHA-256",
    "SHA256.K[1]":      "SHA-256",
    "SHA256.K[2]":      "SHA-256",
    "SHA256.K[3]":      "SHA-256",
    "SHA256.K[4]":      "SHA-256",
    "SHA256.K[5]":      "SHA-256",
    "SHA256.K[6]":      "SHA-256",
    "SHA256.K[7]":      "SHA-256",
    "SM3.T_j[0..15]":   "SM3 (×16 rounds/block)",
    "SM3.T_j[16..63]":  "SM3 (×48 rounds/block)",
}


class Verdict(str, Enum):
    """Evidence quality for a single hit line. Distinguishes 'this is the real
    constant living in code/data' from 'this is an ALU collision'.
    """
    LOAD_IMM = "load_imm"   # movz/movk-built immediate — real signal
    MEM_R    = "mem_r"      # ldr/ldp memory read of preloaded table — real signal
    MEM_W    = "mem_w"      # store going to this address — usually irrelevant
    ALU      = "alu"        # value computed by ALU — likely coincidence


# Targets where SM3 (or other) fingerprints were confirmed outside plugin S1b.
# Each entry: algorithm label, ELF evidence, optional vmtrace structure notes.
VERIFIED_TARGET_PATTERNS: tuple[dict, ...] = (
    {
        "target": "reference-target",
        "algorithm": "SM3",
        "verified_at": "2026-05-28",
        "verified_by": "clark_sm3_fingerprint.py",
        "classification": "confirmed_sm3",
        "evidence": {
            "iv_table_elf_rva": "0x455210",
            "iv_words": [
                "0x7380166f",
                "0x4914b2b9",
                "0x172442d7",
                "0xda8a0600",
                "0xa96f30bc",
                "0x163138aa",
                "0xe38dee4d",
                "0xb0fb0e4e",
            ],
            "vm_rounds": 32,
            "vm_opcodes": ["0x39", "0x54", "0x22", "0x2a"],
            "digest_export": "memcpy@plt 32B @ 0x32350c",
        },
        "notes": "T_j not in rodata (VMP tables); sign uses custom preprocessing vs raw UTF-8 content",
        "partial_archive": "work-tc3-samples/work/legacy/reference-target_partial_archive/",
    },
)
