package com.clarkutov.runner.test;

import com.clarkutov.runner.contract.Instruction;
import com.clarkutov.runner.contract.TraceStream;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Iterator;

/**
 * TraceStream backed by a temp file the runner wrote.
 * <p>For File-handoff consumption (Python engine reads {@link #path()}), no
 * iteration is needed. For Java iteration we return an empty iterator —
 * the actual unidbg-text parser lives on the Python side
 * ({@code engine/runner_client.py::UnidbgTextTraceReader}).</p>
 */
public class FileTraceStream implements TraceStream {

    private final Path path;
    private final String format;

    public FileTraceStream(Path path, String format) {
        this.path = path;
        this.format = format;
    }

    @Override public Path path() { return path; }
    @Override public String format() { return format; }

    @Override
    public Iterator<Instruction> iterator() {
        // Test runner does not implement Java-side parsing; engine consumes via path().
        return Collections.<Instruction>emptyList().iterator();
    }

    @Override
    public void close() {
        // Caller decides whether to delete; the temp file is small and tests
        // benefit from inspection.
    }

    /** Convenience for debugging: line count. */
    public long lineCount() throws IOException {
        try (var s = Files.lines(path)) {
            return s.count();
        }
    }
}
