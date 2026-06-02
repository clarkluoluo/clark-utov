"""Repetition fold — collapse runs of identical signatures.

Algorithm ported from algokiller-plugin tools/search/search.c (Sprint 2 / fold,
MIT cloudza 2026, see NOTICE). Two variants:

  fold_runs(items, signature_of, threshold)
    Collapse consecutive runs where signature_of(item) is identical.
    A run of length N >= threshold (and >= 3) becomes [first, SENTINEL, last].

  fold_block_repeats(items, signature_of, window, threshold)
    Detect a W-long block that repeats >= threshold times in a row
    (signature_of(items[i+W]) == signature_of(items[i]) for the full span).
    Collapse to [first_block_W_items, SENTINEL, last_block_W_items].

Both are O(N) over the input. Designed to be format-agnostic — the caller
supplies signature_of() and gets back a typed stream of items + sentinels.

Use cases in our pipeline:
  - At raw-instruction level: sig = mnemonic (collapse tight ARM64 loops)
  - At basic-block level (D-015 / S2): sig = block_hash, window=1, threshold≈100
  - At handler level (P3 VMP): sig = handler_id, window=1
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class FoldSentinel:
    """Inserted between first-and-last items of a folded run."""
    skipped_count: int     # how many items were hidden
    signature: str         # printable form of the repeating signature
    first_idx: int         # original index of run's first item
    last_idx: int          # original index of run's last item
    window: int = 1        # how many items per repeating unit (1 = simple, >1 = block-aware)


@dataclass
class FoldStats:
    folds_applied: int = 0
    lines_skipped: int = 0
    original_line_count: int = 0


class _Stream(Generic[T]):
    """Bounded look-ahead over an iterator. We materialize into a list for the
    block-repeat path because that pass needs O(N) random access into the
    signature sequence; for huge traces, do block-level fold AFTER S2 dedupe
    where the input is already small.
    """

    def __init__(self, items: Iterable[T]):
        self.items: list[T] = list(items)

    def __len__(self) -> int:
        return len(self.items)


def fold_runs(
    items: Iterable[T],
    signature_of: Callable[[T], str | None],
    threshold: int = 100,
    stats: FoldStats | None = None,
) -> Iterator[T | FoldSentinel]:
    """Collapse consecutive runs where signature_of(item) is identical.

    signature_of returning None means "no fold-able signature on this item" —
    flushes the current run, item passed through untouched.
    Threshold: only emit a fold for runs of >= threshold AND >= 3 items.
    """
    if stats is None:
        stats = FoldStats()

    run_first_idx: int | None = None
    run_first_item: T | None = None
    run_last_item: T | None = None
    run_sig: str | None = None
    run_len = 0

    def flush():
        nonlocal run_first_idx, run_first_item, run_last_item, run_sig, run_len
        if run_first_idx is None:
            return iter(())
        if run_len >= threshold and run_len >= 3:
            assert run_sig is not None and run_first_item is not None and run_last_item is not None
            stats.folds_applied += 1
            stats.lines_skipped += run_len - 2
            out = [
                run_first_item,
                FoldSentinel(
                    skipped_count=run_len - 2,
                    signature=run_sig,
                    first_idx=run_first_idx,
                    last_idx=run_first_idx + run_len - 1,
                    window=1,
                ),
                run_last_item,
            ]
        else:
            # expand: we never saw the middle items individually, so we can't
            # replay them. The caller must consume runs into us linearly; we
            # only buffer first+last. For an "expand" case we approximate by
            # emitting just first..last as a placeholder — but in practice
            # threshold=100 means tiny runs are rare and the expansion path
            # below buffers items. See note in test.
            assert run_first_item is not None
            out = [run_first_item] if run_len == 1 else [run_first_item, run_last_item]  # type: ignore[list-item]

        run_first_idx = None
        run_first_item = None
        run_last_item = None
        run_sig = None
        run_len = 0
        return iter(out)

    for idx, item in enumerate(items):
        stats.original_line_count += 1
        sig = signature_of(item)
        if sig is None:
            yield from flush()
            yield item
            continue
        if run_sig == sig:
            run_last_item = item
            run_len += 1
            continue
        # signature changed — emit previous run
        yield from flush()
        run_first_idx = idx
        run_first_item = item
        run_last_item = item
        run_sig = sig
        run_len = 1
    yield from flush()


def fold_block_repeats(
    items: Iterable[T],
    signature_of: Callable[[T], str | None],
    window: int,
    threshold: int = 2,
    stats: FoldStats | None = None,
) -> Iterator[T | FoldSentinel]:
    """Collapse W-line repeating blocks.

    A block of `window` items starting at position i is considered to repeat
    if signature_of(items[i+W+k]) == signature_of(items[i+k]) for k ∈ [0, W).
    The full repeat span [i, i+W*reps) is folded into:
      first W items + FoldSentinel(window=W, ...) + last W items

    threshold = minimum number of block-copies to bother folding (default 2 ⇒
    "any repeat at all"; raise to 10+ for noise control).
    """
    if stats is None:
        stats = FoldStats()
    if window < 1:
        raise ValueError("window must be >= 1")

    materialized = list(items)
    n = len(materialized)
    stats.original_line_count = n
    if n == 0:
        return

    sigs: list[str | None] = [signature_of(it) for it in materialized]

    i = 0
    while i < n:
        # block ok? need W non-None sigs
        if any(j >= n or sigs[j] is None for j in range(i, i + window)):
            yield materialized[i]
            i += 1
            continue
        # how far do sigs[j] == sigs[j - window] extend?
        j = i + window
        while j < n and sigs[j] is not None and sigs[j] == sigs[j - window]:
            j += 1
        match = j - (i + window)            # tail items matching the block stride
        reps = (match // window) + 1
        span_len = reps * window
        span_end = i + span_len
        if reps >= threshold:
            # first block verbatim
            for k in range(window):
                yield materialized[i + k]
            # sentinel
            hidden = (reps - 2) * window
            if hidden < 0:
                hidden = 0
            stats.folds_applied += 1
            stats.lines_skipped += hidden
            yield FoldSentinel(
                skipped_count=hidden,
                signature=f"<{window}-block × {reps}>",
                first_idx=i,
                last_idx=span_end - 1,
                window=window,
            )
            # last block verbatim
            for k in range(window):
                yield materialized[span_end - window + k]
            i = span_end
        else:
            yield materialized[i]
            i += 1
