from __future__ import annotations

HEADER = bytes.fromhex("746305100000")
_PM_MOD = 2147483647
_TYPE3_DEG = 31
_TYPE3_SEP = 3
_TYPE3_WARMUP = 10 * _TYPE3_DEG


def _park_miller_step(word: int) -> int:
    hi = word // 127773
    lo = word % 127773
    nxt = 16807 * lo - 2836 * hi
    if nxt <= 0:
        nxt += _PM_MOD
    return nxt


def bionic_rand_words(time_seed: int, count: int = 32) -> list[int]:
    """Reproduce Android bionic's rand()/random() stream after srand(time_seed).

    The recovered native path is:

        seed = time(NULL)
        srand(seed)
        for i in range(32):
            buf[i] = rand() & 0xff

    utov's external summary for ``rand`` identifies bionic's TYPE_3 additive
    feedback generator. This implementation rebuilds the exact 31-word state
    from ``srand(seed)`` and returns the subsequent ``rand()`` words.
    """

    if count < 0:
        raise ValueError("count must be non-negative")
    seed = int(time_seed) & 0xFFFFFFFF
    if seed == 0:
        seed = 1

    state = [0] * _TYPE3_DEG
    state[0] = seed & 0x7FFFFFFF
    word = state[0]
    for idx in range(1, _TYPE3_DEG):
        word = _park_miller_step(word)
        state[idx] = word

    fptr = _TYPE3_SEP
    rptr = 0

    def step() -> int:
        nonlocal fptr, rptr
        word = (state[fptr] + state[rptr]) & 0xFFFFFFFF
        state[fptr] = word
        out = (word >> 1) & 0x7FFFFFFF
        fptr = (fptr + 1) % _TYPE3_DEG
        rptr = (rptr + 1) % _TYPE3_DEG
        return out

    for _ in range(_TYPE3_WARMUP):
        step()
    return [step() for _ in range(count)]


def src32_from_time_seed(time_seed: int) -> bytes:
    return bytes(word & 0xFF for word in bionic_rand_words(time_seed, 32))


def libencryptor_output_from_time_seed(input_bytes: bytes, time_seed: int) -> bytes:
    """Reproduce the visible runner output from explicit external state.

    ``input_bytes`` remains in the signature so the dependency contract stays
    honest: the native execution is selected by the plaintext input, while the
    visible 32-byte output is materialized from the same-execution ``time(NULL)``
    seed via ``srand``/``rand``.
    """

    if not isinstance(input_bytes, (bytes, bytearray)):
        raise TypeError("input_bytes must be bytes-like")
    src32 = src32_from_time_seed(time_seed)
    return HEADER + src32[:26]
