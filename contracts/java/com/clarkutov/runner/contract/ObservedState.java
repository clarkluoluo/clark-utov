package com.clarkutov.runner.contract;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * One observed state snapshot returned from {@link Runner#rerun(byte[], java.util.List)}.
 * Matches a corresponding {@link ObservePoint} request by position in the
 * {@link RerunResult#observations} list.
 */
public final class ObservedState {

    /** PC at which the snapshot was taken (== requested {@code ObservePoint.pc}). */
    public final long pc;

    /** Whether snapshot was taken before or after the instruction. */
    public final ObservePoint.When when;

    /** Register name → 64-bit value (extended to long, unsigned semantics). */
    public final Map<String, Long> regs;

    /** Memory: starting address → captured bytes. */
    public final Map<Long, byte[]> mem;

    public ObservedState(long pc, ObservePoint.When when,
                         Map<String, Long> regs, Map<Long, byte[]> mem) {
        this.pc = pc;
        this.when = when;
        this.regs = regs != null ? Collections.unmodifiableMap(new LinkedHashMap<>(regs))
                                  : Collections.emptyMap();
        this.mem  = mem  != null ? Collections.unmodifiableMap(new LinkedHashMap<>(mem))
                                  : Collections.emptyMap();
    }

    public static ObservedState regsOnly(long pc, ObservePoint.When when, Map<String, Long> regs) {
        return new ObservedState(pc, when, regs, null);
    }
}
