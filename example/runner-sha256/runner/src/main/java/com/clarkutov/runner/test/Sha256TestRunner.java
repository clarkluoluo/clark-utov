package com.clarkutov.runner.test;

import com.clarkutov.runner.contract.*;

import com.github.unidbg.AndroidEmulator;
import com.github.unidbg.Module;
import com.github.unidbg.Symbol;
import com.github.unidbg.TraceHook;
import com.github.unidbg.arm.backend.Backend;
import com.github.unidbg.arm.backend.CodeHook;
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
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Unidbg-backed sample runner for example/runner-sha256/libs/arm64-v8a/libsha256.so.
 *
 * <p>Test-only — not part of the clark-utov engine. Used to validate the
 * Java {@link Runner} contract surface and exercise conformance C1-C5.</p>
 *
 * <p><b>Threading / state model:</b> every call to {@link #rerun} and
 * {@link #getTrace} builds a <i>fresh</i> {@code EmuContext}: a brand-new
 * emulator + DVM + reloaded .so + re-resolved class. This is the simple
 * correct way to satisfy {@code contracts/runner_interface.md §3.2}
 * ("rerun must have no cross-call side effects"). The earlier design that
 * shared one {@link AndroidEmulator} across calls had multiple sharp edges:
 * stale {@link CodeHook} accumulation in unidbg 0.9.9 left newly registered
 * hooks effectively silent after the second-or-later {@code
 * callStaticJniMethodObject}; subsequent {@link #getTrace} would emit a
 * single instruction then stop. A fresh emulator per call costs ~200-500ms
 * but is stable and contract-compliant.</p>
 */
public class Sha256TestRunner implements Runner {

    private static final String TARGET_NAME = "libsha256.so";
    private static final String ALGO_SYMBOL = "Java_com_clark_utov_test_Sha256_hash";
    private static final String JAVA_CLASS  = "com/clark/utov/test/Sha256";
    private static final String JAVA_METHOD = "hash([B)[B";

    private final File soFile;
    // metadata() values are stable across the runner's lifetime; we resolve
    // them once via a throwaway emulator at construction.
    private final long algoEntryPc;
    private final long algoExitPc;

    public Sha256TestRunner(File soFile) {
        this.soFile = soFile;
        // One-shot probe to learn entry/exit PCs. Discarded immediately after.
        try (EmuContext probe = new EmuContext(soFile)) {
            Symbol entry = probe.module.findSymbolByName(ALGO_SYMBOL, false);
            if (entry == null) {
                throw new IllegalStateException(
                    "symbol not found in " + soFile + ": " + ALGO_SYMBOL);
            }
            this.algoEntryPc = entry.getAddress();
            this.algoExitPc = probe.module.base + probe.module.size;
        }
    }

    /** Brand-new emulator + DVM + loaded .so, scoped to one call. */
    private static final class EmuContext implements AutoCloseable {
        final AndroidEmulator emulator;
        final VM vm;
        final Module module;
        final DvmClass shaClass;

        EmuContext(File soFile) {
            this.emulator = AndroidEmulatorBuilder.for64Bit().build();
            Memory memory = emulator.getMemory();
            memory.setLibraryResolver(new AndroidResolver(23));
            this.vm = emulator.createDalvikVM();
            this.vm.setVerbose(false);
            DalvikModule dm = vm.loadLibrary(soFile, true);
            this.module = dm.getModule();
            this.shaClass = vm.resolveClass(JAVA_CLASS);
        }

        @Override
        public void close() {
            try {
                emulator.close();
            } catch (IOException ignored) { /* best-effort */ }
        }
    }

    @Override
    public TargetMeta metadata() {
        String emuVer = System.getProperty("unidbg.version");
        return new TargetMeta(
                TARGET_NAME,
                "arm64",
                algoEntryPc,
                algoExitPc,
                null,
                32,
                ALGO_SYMBOL,
                "unidbg",
                emuVer);
    }

    @Override
    public RerunResult rerun(byte[] input, List<ObservePoint> observePoints) {
        List<ObservePoint> pts = (observePoints != null) ? observePoints : Collections.emptyList();
        List<ObservedState> snaps = new ArrayList<>(pts.size());

        try (EmuContext ctx = new EmuContext(soFile)) {
            Backend bk = ctx.emulator.getBackend();
            for (ObservePoint p : pts) {
                long hookPc = (p.when == ObservePoint.When.AFTER) ? p.pc + 4 : p.pc;
                CodeHook h = new CodeHook() {
                    @Override public void hook(Backend bk2, long address, int size, Object user) {
                        snaps.add(captureState(p, bk2));
                    }
                    @Override public void onAttach(com.github.unidbg.arm.backend.UnHook unHook) { /* keep */ }
                    @Override public void detach() { /* noop */ }
                };
                bk.hook_add_new(h, hookPc, hookPc, null);
            }
            ByteArray inputArr = new ByteArray(ctx.vm, input != null ? input : new byte[0]);
            DvmObject<?> result = ctx.shaClass.callStaticJniMethodObject(ctx.emulator, JAVA_METHOD, inputArr);
            byte[] output = (byte[]) result.getValue();
            return new RerunResult(output, snaps);
        }
    }

    private static ObservedState captureState(ObservePoint p, Backend bk) {
        Map<String, Long> regs = new LinkedHashMap<>();
        if (p.capture.contains(ObservePoint.Capture.REGS)) {
            List<String> wanted = p.regs.isEmpty() ? defaultGpRegs() : p.regs;
            for (String r : wanted) {
                Integer ucId = regNameToUcId(r);
                if (ucId != null) {
                    regs.put(r, bk.reg_read(ucId).longValue());
                }
            }
        }
        Map<Long, byte[]> mems = new LinkedHashMap<>();
        if (p.capture.contains(ObservePoint.Capture.MEM)) {
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
        if (n.equals("sp")) return Arm64Const.UC_ARM64_REG_SP;
        if (n.equals("pc")) return Arm64Const.UC_ARM64_REG_PC;
        if (n.equals("lr")) return Arm64Const.UC_ARM64_REG_LR;
        if (n.equals("fp")) return Arm64Const.UC_ARM64_REG_FP;
        if (n.startsWith("x")) {
            try {
                int idx = Integer.parseInt(n.substring(1));
                if (idx >= 0 && idx <= 30) {
                    return Arm64Const.UC_ARM64_REG_X0 + idx;
                }
            } catch (NumberFormatException ignored) { /* fall through */ }
        }
        return null;
    }

    @Override
    public TraceStream getTrace(byte[] input, long start, long end) {
        Path tracePath;
        try {
            tracePath = Files.createTempFile("sha256-trace-", ".txt");
        } catch (IOException e) {
            throw new RuntimeException("failed to allocate trace file", e);
        }
        TraceHook traceHook = null;
        PrintStream traceOut = null;
        try (EmuContext ctx = new EmuContext(soFile)) {
            traceOut = new PrintStream(tracePath.toFile(), StandardCharsets.UTF_8);
            traceHook = ctx.emulator.traceCode(start, end);
            traceHook.setRedirect(traceOut);
            ByteArray inputArr = new ByteArray(ctx.vm, input != null ? input : new byte[0]);
            ctx.shaClass.callStaticJniMethodObject(ctx.emulator, JAVA_METHOD, inputArr);
        } catch (IOException e) {
            throw new RuntimeException("failed to open trace stream", e);
        } finally {
            if (traceHook != null) traceHook.stopTrace();
            if (traceOut != null) {
                traceOut.flush();
                traceOut.close();
            }
        }
        return new FileTraceStream(tracePath, "unidbg_text");
    }

    @Override
    public void close() {
        // Per-call EmuContext lifecycle means no long-lived emulator to close.
    }
}
