package com.clarkutov.runner.contract;

import com.github.unidbg.AndroidEmulator;
import com.github.unidbg.Module;
import com.github.unidbg.TraceHook;
import com.github.unidbg.arm.backend.Backend;
import com.github.unidbg.arm.backend.CodeHook;
import com.github.unidbg.arm.backend.UnHook;
import com.github.unidbg.linux.android.AndroidEmulatorBuilder;
import com.github.unidbg.linux.android.AndroidResolver;
import com.github.unidbg.linux.android.dvm.DalvikModule;
import com.github.unidbg.linux.android.dvm.DvmClass;
import com.github.unidbg.linux.android.dvm.DvmObject;
import com.github.unidbg.linux.android.dvm.VM;
import com.github.unidbg.linux.android.dvm.array.ByteArray;
import com.github.unidbg.memory.Memory;
import unicorn.Arm64Const;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Reference base class for an Android JNI native-method Live-mode runner.
 *
 * <p>0527 BUG_REPORT-7: extracted from the duplicated scaffolding in
 * the sample SHA256 runner's {@code Sha256TestRunner} and the reference
 * target's {@code LibEncryptorTestRunner}. ~120 of the ~150 lines each previously
 * carried were verbatim copies of (emulator setup × per-call freshness ×
 * observe-point capture × traceCode redirect × regs-name mapping); this
 * class owns all of that. A subclass for a new Android JNI target only
 * needs to override 5 abstract methods (~12 lines).</p>
 *
 * <pre>{@code
 * public class MyTargetRunner extends JniLiveRunner {
 *     @Override protected String targetName()   { return "mylib.so"; }
 *     @Override protected long   algoEntryPc()  { return 0x40001234L; }
 *     @Override protected long   algoExitPc()   { return 0x40005678L; }
 *     @Override protected int    outputLength() { return 32; }
 *     @Override protected File   soFile()       { return resolveSoFile(); }
 * }
 * }</pre>
 *
 * <p><b>Default JNI ABI</b> for {@link #callArgs}:
 * {@code f(JNIEnv*, jobject thiz, jbyteArray msg, jint len) -> jbyteArray}.
 * Override {@code callArgs(EmuContext, byte[])} for other shapes (e.g.
 * {@code (jstring, jint) -> jstring}, output-via-buffer parameter form,
 * symbol-registered native methods using {@code callStaticJniMethodObject}).</p>
 *
 * <p><b>C5 cross-call independence</b> is enforced by constructing a fresh
 * {@link EmuContext} inside every {@link #rerun} and {@link #getTrace}
 * invocation. Do <i>not</i> subclass and "cache" the context — the first
 * runner author who tried sharing state across calls saw {@code traceCode}
 * hooks fire only on the first invocation and {@code getTrace} return
 * truncated output for the second.</p>
 *
 * <p><b>If Live mode fails for your target</b> — the four most common
 * culprits, in priority order:</p>
 * <ol>
 *   <li>Anti-debug / Frida-detection inside the .so. Runner has not patched
 *       the check; emulation halts on the first SVC. Override the load
 *       step in a custom {@code EmuContext} constructor to patch.</li>
 *   <li>JNI vtable surprise: target relies on a registered native that the
 *       default {@code (JNIEnv*, jobject, jbyteArray, jint)} ABI doesn't
 *       cover. Override {@link #callArgs} + {@link #resolveOutput}.</li>
 *   <li>Output-shape mismatch: the algorithm writes to a caller-supplied
 *       buffer instead of returning a jbyteArray. Override
 *       {@link #resolveOutput}.</li>
 *   <li>Rebased .so: the {@link #algoEntryPc} address you hard-coded was
 *       captured at one base; unidbg loaded it at a different base. The
 *       {@code EmuContext} constructor bounds-checks and throws with a
 *       clear message — recompute the offset and retry.</li>
 * </ol>
 *
 * @see Runner
 * @see TraceStream
 * @see RerunResult
 */
public abstract class JniLiveRunner implements Runner {

    // ── subclass contract ────────────────────────────────────────────────

    /** Human-readable target name (typically the .so basename). */
    protected abstract String targetName();

    /** PC where the algorithm function begins (trace-start anchor). */
    protected abstract long algoEntryPc();

    /** PC where the algorithm function ends (typically a {@code ret}). */
    protected abstract long algoExitPc();

    /** Algorithm output length in bytes (e.g. 32 for SHA-256). */
    protected abstract int outputLength();

    /** The .so file unidbg should load. */
    protected abstract File soFile();

    /**
     * Synthetic class name unidbg uses to construct a dummy {@code thiz}.
     * Default {@code "com/clarkutov/Target"} works for stripped .so targets
     * called via raw-PC ABI. Override for targets that use a real Java
     * class name (e.g. {@code "com/clark/utov/test/Sha256"}).
     */
    protected String dummyJClass() {
        return "com/clarkutov/Target";
    }

    /** Optional input length advertised in metadata; {@code null} = variable. */
    protected Integer inputLength() {
        return null;
    }

    /** Optional algorithm symbol name when the .so isn't stripped; {@code null} otherwise. */
    protected String algoSymbol() {
        return null;
    }

    /** Optional emulator-name string for metadata audit (e.g. {@code "unidbg"}). */
    protected String emulatorName() {
        return "unidbg";
    }

    /** Optional emulator-version string for metadata audit (e.g. {@code "0.9.9"}). */
    protected String emulatorVersion() {
        return null;
    }

    /**
     * Build the argument list for {@code Module.callFunction}.
     *
     * <p>Default: standard Android JNI ABI
     * {@code (JNIEnv*, jobject thiz, jbyteArray msg, jint len)}.</p>
     *
     * <p>Override for other shapes — e.g.:
     * <ul>
     *   <li>{@code (JNIEnv*, jobject, jbyteArray in, jint len, jbyteArray out)}
     *       for buffer-out style</li>
     *   <li>{@code (JNIEnv*, jclass, jstring, jint)} for jstring inputs</li>
     *   <li>Use {@code vm.callStaticJniMethodObject(name, ...)} entirely
     *       (and override {@link #invoke} too) for {@code RegisterNatives}-style
     *       symbol dispatch.</li>
     * </ul>
     */
    protected Object[] callArgs(EmuContext ctx, byte[] input) {
        ByteArray inputArr      = new ByteArray(ctx.vm, input);
        int       thizHandle    = ctx.vm.addLocalObject(ctx.dummyThiz);
        int       byteArrHandle = ctx.vm.addLocalObject(inputArr);
        return new Object[] {
                ctx.vm.getJNIEnv(),
                (long) thizHandle,
                (long) byteArrHandle,
                (long) input.length,
        };
    }

    /**
     * Resolve the function's return value to output bytes.
     *
     * <p>Default: result is a jbyteArray handle; look up its {@link DvmObject}
     * and extract {@code byte[]} value. Override for buffer-out form (read
     * the output bytes from emulator memory instead of the return value).</p>
     */
    protected byte[] resolveOutput(EmuContext ctx, Number result) {
        if (result == null) {
            return new byte[0];
        }
        DvmObject<?> outObj = ctx.vm.getObject((int) result.longValue());
        if (outObj == null) {
            return new byte[0];
        }
        Object v = outObj.getValue();
        return (v instanceof byte[]) ? (byte[]) v : new byte[0];
    }

    // ── Runner interface implementation ──────────────────────────────────

    @Override
    public final TargetMeta metadata() {
        return new TargetMeta(
                targetName(), "arm64",
                algoEntryPc(), algoExitPc(),
                inputLength(), outputLength(),
                algoSymbol(),
                emulatorName(), emulatorVersion());
    }

    @Override
    public final RerunResult rerun(byte[] input, List<ObservePoint> observePoints) {
        try (EmuContext ctx = new EmuContext(soFile(), dummyJClass(), algoEntryPc())) {
            return doRerun(ctx, input != null ? input : new byte[0], observePoints);
        }
    }

    @Override
    public final TraceStream getTrace(byte[] input, long start, long end) {
        Path out;
        try {
            String prefix = targetName().replaceAll("[^a-zA-Z0-9]", "_") + "-trace-";
            out = Files.createTempFile(prefix, ".txt");
        } catch (IOException e) {
            throw new UncheckedIOException("alloc trace temp file", e);
        }
        try (EmuContext ctx = new EmuContext(soFile(), dummyJClass(), algoEntryPc());
             PrintStream traceOut = new PrintStream(out.toFile(), StandardCharsets.UTF_8)) {
            TraceHook traceHook = ctx.emulator.traceCode(
                    ctx.module.base, ctx.module.base + ctx.module.size);
            traceHook.setRedirect(traceOut);
            try {
                invoke(ctx, input != null ? input : new byte[16]);
            } finally {
                traceHook.stopTrace();
                traceOut.flush();
            }
        } catch (IOException e) {
            throw new UncheckedIOException("open trace stream", e);
        }
        return new FileTraceStream(out, "unidbg_text");
    }

    @Override
    public void close() {
        /* per-call EmuContext — nothing global to release. */
    }

    // ── shared internals (subclass-overridable) ──────────────────────────

    /**
     * Invoke the algorithm once and return its output bytes.
     * Default: {@code module.callFunction(emulator, entryOffset, callArgs)}
     * then {@link #resolveOutput}. Override for symbol-registered native
     * methods (use {@code vm.callStaticJniMethodObject}).
     */
    protected byte[] invoke(EmuContext ctx, byte[] input) {
        Number result = ctx.module.callFunction(
                ctx.emulator, ctx.entryOffset, callArgs(ctx, input));
        return resolveOutput(ctx, result);
    }

    private RerunResult doRerun(EmuContext ctx, byte[] input,
                                List<ObservePoint> pts) {
        List<ObservePoint> ops = (pts != null) ? pts : Collections.emptyList();
        List<ObservedState> snaps = new ArrayList<>(ops.size());
        Backend bk = ctx.emulator.getBackend();
        for (ObservePoint p : ops) {
            long hookPc = (p.when == ObservePoint.When.AFTER) ? p.pc + 4 : p.pc;
            bk.hook_add_new(new CodeHook() {
                @Override
                public void hook(Backend bk2, long addr, int size, Object user) {
                    snaps.add(captureState(p, bk2));
                }
                @Override public void onAttach(UnHook u) { /* unused */ }
                @Override public void detach() { /* unused */ }
            }, hookPc, hookPc, null);
        }
        byte[] out = invoke(ctx, input);
        return new RerunResult(out, snaps);
    }

    private static ObservedState captureState(ObservePoint p, Backend bk) {
        Map<String, Long> regs = new LinkedHashMap<>();
        if (p.capture.contains(ObservePoint.Capture.REGS)) {
            List<String> wanted = (p.regs == null || p.regs.isEmpty())
                    ? defaultGpRegs() : p.regs;
            for (String r : wanted) {
                Integer ucId = regNameToUcId(r);
                if (ucId != null) {
                    regs.put(r, bk.reg_read(ucId).longValue());
                }
            }
        }
        Map<Long, byte[]> mems = new LinkedHashMap<>();
        if (p.capture.contains(ObservePoint.Capture.MEM) && p.mem != null) {
            for (MemRange mr : p.mem) {
                mems.put(mr.addr, bk.mem_read(mr.addr, mr.size));
            }
        }
        return new ObservedState(p.pc, p.when, regs, mems);
    }

    private static List<String> defaultGpRegs() {
        List<String> out = new ArrayList<>(33);
        for (int i = 0; i <= 30; i++) out.add("x" + i);
        out.add("sp");
        out.add("pc");
        return out;
    }

    private static Integer regNameToUcId(String name) {
        if (name == null) return null;
        String n = name.toLowerCase(Locale.ROOT);
        switch (n) {
            case "sp": return Arm64Const.UC_ARM64_REG_SP;
            case "pc": return Arm64Const.UC_ARM64_REG_PC;
            case "lr": return Arm64Const.UC_ARM64_REG_LR;
            case "fp": return Arm64Const.UC_ARM64_REG_FP;
            default:
                if (n.startsWith("x")) {
                    try {
                        int idx = Integer.parseInt(n.substring(1));
                        if (idx >= 0 && idx <= 30) {
                            return Arm64Const.UC_ARM64_REG_X0 + idx;
                        }
                    } catch (NumberFormatException ignored) {
                        /* fall through */
                    }
                }
                return null;
        }
    }

    // ── shared EmuContext (constructed fresh per call) ───────────────────

    /**
     * Per-call emulator + DalvikVM + loaded module bundle.
     *
     * <p>Constructed inside every {@link JniLiveRunner#rerun} and
     * {@link JniLiveRunner#getTrace} invocation; closed at the end of that
     * call. Sharing an instance across calls breaks C5 cross-call
     * independence and causes traceCode hooks to misfire — don't do it.</p>
     */
    public static final class EmuContext implements AutoCloseable {
        public final AndroidEmulator emulator;
        public final VM              vm;
        public final Module          module;
        public final DvmClass        dummyThiz;
        public final long            entryOffset;

        EmuContext(File soFile, String dummyJClass, long entryPc) {
            this.emulator = AndroidEmulatorBuilder.for64Bit().build();
            Memory memory = emulator.getMemory();
            memory.setLibraryResolver(new AndroidResolver(23));
            this.vm = emulator.createDalvikVM();
            this.vm.setVerbose(false);
            DalvikModule dm = vm.loadLibrary(soFile, true);
            this.module = dm.getModule();
            if (entryPc < module.base || entryPc >= module.base + module.size) {
                throw new IllegalStateException(String.format(
                        "entry PC 0x%x outside module [0x%x, 0x%x); .so was rebased",
                        entryPc, module.base, module.base + module.size));
            }
            this.entryOffset = entryPc - module.base;
            this.dummyThiz = vm.resolveClass(dummyJClass);
        }

        @Override
        public void close() {
            try {
                emulator.close();
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
    }

    // ── concrete TraceStream impl returned by getTrace ───────────────────

    /**
     * Concrete {@link TraceStream} backed by a temp file holding unidbg's
     * default text trace output. Inlined here so subclasses don't need to
     * ship their own TraceStream implementation.
     */
    protected static final class FileTraceStream implements TraceStream {
        private final Path   path;
        private final String format;

        public FileTraceStream(Path path, String format) {
            this.path   = path;
            this.format = format;
        }

        @Override public Path   path()   { return path; }
        @Override public String format() { return format; }

        @Override
        public Iterator<Instruction> iterator() {
            // The Python engine consumes the path directly; in-process
            // iteration in Java is rare. Subclasses needing it should
            // ship their own parser keyed off `format`.
            return Collections.emptyIterator();
        }

        @Override
        public void close() {
            /* leave temp file in place — engine reads it after close. */
        }
    }
}
