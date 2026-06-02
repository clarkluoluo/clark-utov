package com.clarkutov.runner.contract;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * One trace record. Mirrors {@code contracts/runner_interface.md §2.1} schema.
 *
 * <p>Per the size-control rules in §2.3:</p>
 * <ul>
 *   <li>{@link #regsRead}/{@link #regsWrite} only contain registers actually
 *       read/written by THIS instruction — NOT a full register snapshot.</li>
 *   <li>{@link #mem} only contains accesses this instruction actually performed.</li>
 * </ul>
 */
public final class Instruction {

    public final long idx;
    public final long pc;
    public final byte[] bytes;
    public final String mnemonic;
    public final Map<String, Long> regsRead;
    public final Map<String, Long> regsWrite;
    public final List<MemOp> mem;

    public Instruction(long idx, long pc, byte[] bytes, String mnemonic,
                       Map<String, Long> regsRead,
                       Map<String, Long> regsWrite,
                       List<MemOp> mem) {
        this.idx = idx;
        this.pc = pc;
        this.bytes = bytes;
        this.mnemonic = mnemonic;
        this.regsRead  = regsRead  != null ? Collections.unmodifiableMap(new LinkedHashMap<>(regsRead))  : Collections.emptyMap();
        this.regsWrite = regsWrite != null ? Collections.unmodifiableMap(new LinkedHashMap<>(regsWrite)) : Collections.emptyMap();
        this.mem = mem != null ? Collections.unmodifiableList(mem) : Collections.emptyList();
    }
}
