package com.clarkutov.runner.contract;

/**
 * Target metadata returned by {@link Runner#metadata()}.
 * See {@code contracts/runner_interface.md §4}.
 */
public final class TargetMeta {

    /** Human-readable target name, typically the .so basename. */
    public final String targetName;

    /** Architecture string. Currently always {@code "arm64"}. */
    public final String arch;

    /** Trace start anchor — PC where the algorithm function begins. */
    public final long algoEntryPc;

    /** Trace end anchor — typically the {@code ret} of the algorithm function. */
    public final long algoExitPc;

    /** Fixed input length in bytes; {@code null} = variable. */
    public final Integer inputLength;

    /** Output length in bytes. */
    public final int outputLength;

    /** Optional symbol name (if the .so isn't stripped). May be {@code null}. */
    public final String algoSymbol;

    /** Optional: which emulator/VM the runner uses ({@code "unidbg"}, {@code "qiling"},
     *  {@code "frida"}, etc.). {@code null} when N/A or unknown. Recorded in audit. */
    public final String emulatorName;

    /** Optional: emulator version string (e.g. {@code "0.9.9"}). {@code null} unknown.
     *  Useful for cross-run reproducibility tags and rule applicability scoping. */
    public final String emulatorVersion;

    public TargetMeta(String targetName, String arch,
                      long algoEntryPc, long algoExitPc,
                      Integer inputLength, int outputLength,
                      String algoSymbol,
                      String emulatorName, String emulatorVersion) {
        this.targetName  = targetName;
        this.arch        = arch;
        this.algoEntryPc = algoEntryPc;
        this.algoExitPc  = algoExitPc;
        this.inputLength = inputLength;
        this.outputLength = outputLength;
        this.algoSymbol  = algoSymbol;
        this.emulatorName    = emulatorName;
        this.emulatorVersion = emulatorVersion;
    }

    /** Back-compat constructor (no emulator info). */
    public TargetMeta(String targetName, String arch,
                      long algoEntryPc, long algoExitPc,
                      Integer inputLength, int outputLength,
                      String algoSymbol) {
        this(targetName, arch, algoEntryPc, algoExitPc,
             inputLength, outputLength, algoSymbol, null, null);
    }

    @Override
    public String toString() {
        return "TargetMeta{" + targetName + " " + arch
             + " entry=0x" + Long.toHexString(algoEntryPc)
             + " exit=0x"  + Long.toHexString(algoExitPc)
             + " inLen="   + inputLength
             + " outLen="  + outputLength
             + (algoSymbol != null ? " sym=" + algoSymbol : "")
             + (emulatorName != null
                ? " emu=" + emulatorName + (emulatorVersion != null ? "/" + emulatorVersion : "")
                : "")
             + '}';
    }
}
