package com.clarkutov.runner.contract;

import java.util.List;

/**
 * Runner — the system-facing interface a target environment must implement.
 *
 * <p>The clark-utov engine consumes a runner; the runner takes care of all
 * "make the .so actually run" concerns: anti-debug, anti-Frida, JNI dynamic
 * registration, unidbg setup, fake stabilization. The engine never touches
 * those.</p>
 *
 * <p>See {@code contracts/runner_interface.md} for the full spec
 * (including PLAN §17 conformance test gate).</p>
 *
 * <p>Three-method surface (PLAN §17):</p>
 * <ul>
 *   <li>{@link #metadata()} — fixed properties of the target (called once)</li>
 *   <li>{@link #getTrace(byte[], long, long)} — produce a normalized trace
 *       for [start, end] PC range under the given input</li>
 *   <li>{@link #rerun(byte[], List)} — execute with observation points and
 *       return outputs + observed state snapshots</li>
 * </ul>
 *
 * <p>Two implementation modes:</p>
 * <ul>
 *   <li><b>Live</b>: all three methods fully implemented; full verifier capability</li>
 *   <li><b>File</b>: only {@link #metadata()} and a pre-baked trace file are
 *       available; {@code getTrace}/{@code rerun} throw {@link UnsupportedOperationException}.
 *       The conformance test C1/C2/C3 SKIP, only C4 runs; verifier degrades.</li>
 * </ul>
 *
 * <p>Contract guarantees (must hold; not just method signatures):</p>
 * <ol>
 *   <li>Deterministic: same input ⇒ bit-identical output across calls.</li>
 *   <li>Fake stable: time / random / device fingerprint fixed on env side.</li>
 *   <li>Observation points reachable and accurate.</li>
 *   <li>Rerun affordable: target single-call latency &lt; 1 s, or supports caching.</li>
 * </ol>
 */
public interface Runner extends AutoCloseable {

    /** Metadata describing the target. Called once at boot. */
    TargetMeta metadata();

    /**
     * Run the target on {@code input}; capture state at each {@code observePoints}
     * entry; return the algorithm output (e.g. digest bytes) plus collected
     * {@link ObservedState} list, in input order.
     *
     * @param input          algorithm input bytes
     * @param observePoints  zero or more observation points; null treated as empty
     * @return non-null {@link RerunResult}; observations list is in same order as input
     * @throws UnsupportedOperationException if this runner is File mode
     */
    RerunResult rerun(byte[] input, List<ObservePoint> observePoints);

    /**
     * Produce a normalized trace for {@code input} over PC range {@code [start, end]}.
     * The returned {@link TraceStream} is the caller's responsibility to close.
     *
     * @param input  algorithm input bytes
     * @param start  trace start anchor (PC). Use {@link TargetMeta#algoEntryPc} typically.
     * @param end    trace end anchor (PC). Use {@link TargetMeta#algoExitPc} typically.
     * @return non-null {@link TraceStream}
     * @throws UnsupportedOperationException if this runner is File mode
     */
    TraceStream getTrace(byte[] input, long start, long end);

    /**
     * Release runner resources (emulator state, file handles, temp dirs).
     * Idempotent. Required by AutoCloseable so try-with-resources works.
     */
    @Override
    void close();
}
