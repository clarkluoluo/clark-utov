"""Algorithm-spec registry for the Tier 1 pseudocode emitter
(FEATURE-REQUEST-1).

Each entry contains:

  prefix                       fingerprint / fold-idiom subject prefix
                               (matches engine/engine/data/fingerprints.py)
  iv_count                     number of H constants in the algorithm
  k_count                      number of round constants (info-only; the
                               engine may stock fewer K fingerprints)
  msched_t_range               (lo, hi) tuple of message-schedule t-loop
  msched_iters_per_block       iters in the message-schedule loop per block
  compress_t_range             (lo, hi) tuple of compression t-loop
  compress_iters_per_block     iters in compression per block
  sigma_idioms                 small-σ idiom suffixes (e.g. "sigma0")
  Sigma_idioms                 big-Σ idiom suffixes (e.g. "Sigma0")
  pseudocode                   generic algorithm body (string literal)

These values are algorithm SPECs — they don't depend on any particular
target binary. Per-binary facts (anchor PCs, IV values, observed loop
counts) come from `findings.sqlite` at emit time.
"""

from __future__ import annotations

from typing import Any

ALGORITHM_SPECS: dict[str, dict[str, Any]] = {
    "SHA-256": {
        "prefix":                   "SHA256",
        "iv_count":                 8,
        "k_count":                  64,
        "msched_t_range":           (16, 63),
        "msched_iters_per_block":   48,
        "compress_t_range":         (0, 63),
        "compress_iters_per_block": 64,
        "sigma_idioms":             ("sigma0", "sigma1"),
        "Sigma_idioms":             ("Sigma0", "Sigma1"),
        "pseudocode": """\
  // SHA-256 (FIPS 180-4)
  H[0..7] = initial-hash-values (see Constants)
  for each 64-byte message block M:
    W[0..15] = M
    for t in 16..63:
      W[t] = sigma1(W[t-2]) + W[t-7] + sigma0(W[t-15]) + W[t-16]
    (a,b,c,d,e,f,g,h) = H
    for t in 0..63:
      T1 = h + Sigma1(e) + Ch(e,f,g) + K[t] + W[t]
      T2 = Sigma0(a) + Maj(a,b,c)
      h=g; g=f; f=e; e=d+T1; d=c; c=b; b=a; a=T1+T2
    H += (a,b,c,d,e,f,g,h)
  return H  (32 bytes)""",
    },
    "SHA-512": {
        "prefix":                   "SHA512",
        "iv_count":                 8,
        "k_count":                  80,
        "msched_t_range":           (16, 79),
        "msched_iters_per_block":   64,
        "compress_t_range":         (0, 79),
        "compress_iters_per_block": 80,
        "sigma_idioms":             ("sigma0", "sigma1"),
        "Sigma_idioms":             ("Sigma0", "Sigma1"),
        "pseudocode": """\
  // SHA-512 (FIPS 180-4)
  H[0..7] = initial-hash-values (see Constants)
  for each 128-byte message block M:
    W[0..15] = M
    for t in 16..79:
      W[t] = sigma1(W[t-2]) + W[t-7] + sigma0(W[t-15]) + W[t-16]
    (a,b,c,d,e,f,g,h) = H
    for t in 0..79:
      T1 = h + Sigma1(e) + Ch(e,f,g) + K[t] + W[t]
      T2 = Sigma0(a) + Maj(a,b,c)
      h=g; g=f; f=e; e=d+T1; d=c; c=b; b=a; a=T1+T2
    H += (a,b,c,d,e,f,g,h)
  return H  (64 bytes, or truncated per variant)""",
    },
}
