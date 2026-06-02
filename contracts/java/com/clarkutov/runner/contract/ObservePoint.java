package com.clarkutov.runner.contract;

import java.util.Collections;
import java.util.EnumSet;
import java.util.List;

/**
 * One observation point requested by the engine in {@link Runner#rerun(byte[], List)}.
 * The runner pauses at {@code pc} (before or after execution), dumps the requested
 * state, and returns an {@link ObservedState}.
 *
 * <p>If {@link #regs} is null/empty, all general-purpose registers are captured
 * (caller convention).</p>
 */
public final class ObservePoint {

    public enum When { BEFORE, AFTER }

    public enum Capture { REGS, MEM }

    /** PC address at which to observe. */
    public final long pc;

    /** Capture state before or after the instruction at {@link #pc} executes. */
    public final When when;

    /** Which categories of state to capture. */
    public final EnumSet<Capture> capture;

    /** Specific registers to capture; null/empty = caller convention. */
    public final List<String> regs;

    /** Memory ranges to dump. */
    public final List<MemRange> mem;

    public ObservePoint(long pc, When when, EnumSet<Capture> capture,
                        List<String> regs, List<MemRange> mem) {
        this.pc = pc;
        this.when = when;
        this.capture = capture;
        this.regs = regs != null ? regs : Collections.emptyList();
        this.mem  = mem  != null ? mem  : Collections.emptyList();
    }

    /** Convenience for the common "regs-only after-execution" case. */
    public static ObservePoint regsAfter(long pc, List<String> regs) {
        return new ObservePoint(pc, When.AFTER, EnumSet.of(Capture.REGS), regs, null);
    }

    /** Convenience for the common "regs-only before-execution" case. */
    public static ObservePoint regsBefore(long pc, List<String> regs) {
        return new ObservePoint(pc, When.BEFORE, EnumSet.of(Capture.REGS), regs, null);
    }
}
