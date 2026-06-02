package com.clarkutov.runner.contract;

/** A request for {@code size} bytes starting at {@code addr}. Used in {@link ObservePoint}. */
public final class MemRange {
    public final long addr;
    public final int  size;
    public MemRange(long addr, int size) { this.addr = addr; this.size = size; }
}
