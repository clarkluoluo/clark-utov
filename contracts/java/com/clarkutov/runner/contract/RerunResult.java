package com.clarkutov.runner.contract;

import java.util.Collections;
import java.util.List;

/**
 * Return value of {@link Runner#rerun(byte[], java.util.List)}.
 *
 * <ul>
 *   <li>{@link #output} — the algorithm's final byte output (digest/cipher/etc.)</li>
 *   <li>{@link #observations} — one {@link ObservedState} per requested
 *       {@link ObservePoint}, in the same order</li>
 * </ul>
 */
public final class RerunResult {

    public final byte[] output;
    public final List<ObservedState> observations;

    public RerunResult(byte[] output, List<ObservedState> observations) {
        this.output = output;
        this.observations = observations != null
                ? Collections.unmodifiableList(observations)
                : Collections.emptyList();
    }
}
