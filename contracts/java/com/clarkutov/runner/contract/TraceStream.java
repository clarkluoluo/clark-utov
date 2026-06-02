package com.clarkutov.runner.contract;

import java.nio.file.Path;
import java.util.Iterator;

/**
 * Returned by {@link Runner#getTrace(byte[], long, long)}.
 *
 * <p>Two consumption modes (caller picks whichever is convenient):</p>
 * <ul>
 *   <li><b>File hand-off</b>: read {@link #path()} and {@link #format()} and parse it
 *       in the engine layer (most common — the Python engine consumes a file path).</li>
 *   <li><b>Java iteration</b>: use {@link #iterator()} to walk parsed
 *       {@link Instruction}s directly in-process.</li>
 * </ul>
 *
 * <p>Implementations must release any underlying file handles / pipes on {@link #close()}.</p>
 */
public interface TraceStream extends Iterable<Instruction>, AutoCloseable {

    /** Path to the trace file the runner produced. Must exist while the stream is open. */
    Path path();

    /**
     * Trace format identifier. Currently one of:
     * <ul>
     *   <li>{@code "jsonl"}        — contracts/runner_interface.md §2.1 schema (preferred)</li>
     *   <li>{@code "unidbg_text"}  — contracts/runner_interface.md §2.2 compat</li>
     * </ul>
     */
    String format();

    @Override
    Iterator<Instruction> iterator();

    @Override
    void close();
}
