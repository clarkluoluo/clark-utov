package com.clarkutov.runner.contract;

/** Memory access record inside an {@link Instruction}. */
public final class MemOp {

    public enum RW { READ, WRITE }

    public final RW rw;
    public final long addr;
    public final long val;
    public final int size;

    public MemOp(RW rw, long addr, long val, int size) {
        this.rw = rw;
        this.addr = addr;
        this.val = val;
        this.size = size;
    }
}
