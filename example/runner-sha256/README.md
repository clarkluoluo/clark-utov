# example/runner-sha256/ — sample fixture, NOT part of the engine

Everything under this directory is an **example** showing how to satisfy the
contracts in [`../../contracts/`](../../contracts/). It exists so we can validate
the engine end-to-end without forcing every user to write a runner first.

**The engine does not depend on anything in here.** It accepts any runner
subprocess that speaks the NDJSON protocol in
[`../../contracts/agent_protocol.md`](../../contracts/agent_protocol.md).

Don't list Java / Maven / Android NDK / unidbg as engine prerequisites just
because this sample uses them — those are sample-specific, optional.

---

## Contents

```
example/runner-sha256/
├── README.md                      ← you are here
├── libs/arm64-v8a/libsha256.so    ← Sample analysis target. OLLVM-style
│                                    SHA-256 .so we built ourselves; passes
│                                    all NIST FIPS 180-2 vectors. Useful as
│                                    a ground-truth target.
├── java/com/clark/utov/test/Sha256.java
│                                  ← Java glue class that loads the .so via
│                                    JNI. Just for reference.
└── runner/                        ← Sample runner (Java + unidbg) that
                                     implements contracts/runner_interface.md.
    ├── pom.xml                      Maven build, thin jar (no unidbg shaded).
    └── src/main/java/...            Sha256TestRunner + Main + FileTraceStream.
```

---

## Using the sample runner

**One-time:**

```bash
# 1. Build the thin jar (~30 KB; unidbg NOT bundled)
cd runner
mvn -DskipTests package

# 2. Populate unidbg locally (see ../../DEPENDENCIES.md §3 for full recipe)
export UNIDBG_HOME=$HOME/.unidbg/0.9.9   # path to dir of unidbg-android-*.jar + transitives
```

**Run it standalone (interactive demo):**

```bash
../../bin/run-runner.sh demo /path/to/libsha256.so
```

**Use it with `utov pipeline`:**

```bash
# The engine just spawns whatever string you pass it. Compose the runner cmd:
utov pipeline \
    --runner-cmd "$(pwd)/../../bin/run-runner.sh serve $(pwd)/libs/arm64-v8a/libsha256.so" \
    --input 616263 --mode frugal --work-root ./work
```

---

## Known issue (intentionally left in)

`Sha256TestRunner.rerun()` installs observation `CodeHook`s and does NOT
detach them. This violates `contracts/runner_interface.md §3.2` ("rerun
must have no cross-call side effects"). The engine's **conformance C5**
catches this exact bug — running `utov pipeline` against this sample with
default settings produces:

```
RuntimeError: Pre-trace conformance C5 (cross-call independence) FAILED.
The runner leaks state across calls; ...
```

That FAILURE is **expected**, and it's the canonical demo of C5 working.
If you want to use this sample to exercise the rest of the engine, either
fix the leak in `Sha256TestRunner.java` (detach hooks after each `rerun`)
or pass `--skip-conformance` (not recommended).

---

## Writing your own runner

Don't fork this. Implement `contracts/runner_interface.md` + speak
`contracts/agent_protocol.md` NDJSON in any language. Examples of valid
runners that are NOT here:

- Frida + python script that hooks the .so on a device
- qiling-based runner in pure Python
- A QEMU-based one
- Static-only File-mode runner (just metadata + a pre-baked trace file)

As long as `metadata()` / `get_trace()` / `rerun()` work over stdio NDJSON
and pass conformance, the engine doesn't care which one you wrote.
