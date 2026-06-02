# Dependencies

The engine is a **pure Python package**. Everything else listed below is for
*optional* sample fixtures and *runner-side* concerns the engine does not
own.

Run `utov doctor` (or `python3 -m engine.doctor`) to scan your machine; it
prints a grouped report and only exits non-zero on **engine** failures —
the other groups are informational.

---

## 1. Engine (the only thing the engine itself needs)

```bash
python3 -m pip install --user clark_utov_engine-0.1.0-py3-none-any.whl
```

| Requirement | Version | Why |
|---|---|---|
| Python | 3.11+ | engine runtime |
| Python packages | as in `engine/pyproject.toml` | pulled automatically by pip |

The Python packages (pulled by pip): `openai`, `python-dotenv`, `jsonschema`,
`click`, `tqdm`, `pyarrow`. Dev-only (you only need these to develop the
engine itself): `pytest`, `ruff`, `mypy`, `build`.

**That is the complete list of engine dependencies. Anything below this line
is OPTIONAL — for sample fixtures or runner-side workflows.**

---

## 2. Runner compatibility (a statement, not a dependency)

The engine consumes any runner subprocess that satisfies
[`contracts/runner_interface.md`](contracts/runner_interface.md) +
[`contracts/agent_protocol.md`](contracts/agent_protocol.md). It is
**runner-agnostic**: language, emulator, and platform are all up to the
runner author.

What we have explicitly contract-tested:

| Runner stack | Status |
|---|---|
| Sample runner (Java + `unidbg-android 0.9.9`, `bin/run-runner.sh`) | ✅ tested |
| `unidbg-android 0.9.8` | ❌ **broken** — Maven POM is invalid, transitive deps don't resolve |
| `unidbg-android ≤ 0.9.7` | ⚠ untested |
| `unidbg-android 1.x.x` | ⚠ untested — API surface may differ |
| qiling / Frida / QEMU / pure-Python emulators | ⚠ untested but should work if contract is met |

If you write a runner using a different emulator/version, expose it in
`metadata()` (the contract has optional `emulator_name` and
`emulator_version` fields). The engine logs that into `meta.json` and
intervention audit so future agents can tell which environment a finding
came from.

---

## 3. Optional: bundled Java sample runner (`example/runner-sha256/runner/`)

`example/runner-sha256/` is **not part of the engine**. It exists only to let you
exercise the engine end-to-end without writing a runner first. See
[`example/runner-sha256/README.md`](example/runner-sha256/README.md). Only relevant if you use it.

| Tool | Why (only for the sample) |
|---|---|
| Java JDK 17+ | runs the sample runner jar |
| Maven 3.9+ | rebuilds the sample runner jar |
| Android NDK 25 / 29 | only if rebuilding `libsha256.so` from `tmp/jni/` |
| `unidbg-android 0.9.9` jars in `$UNIDBG_HOME` | sample runner needs this on classpath (BYO, see below) |

### 3.1 Bringing your own unidbg (sample runner uses it)

```bash
mkdir -p ~/.unidbg/0.9.9 && cd ~/.unidbg/0.9.9
cat > pom.xml <<'POM'
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>local.unidbg</groupId><artifactId>collect</artifactId><version>1</version>
  <repositories>
    <repository><id>jitpack.io</id><url>https://jitpack.io</url></repository>
  </repositories>
  <dependencies>
    <dependency>
      <groupId>com.github.zhkl0228</groupId>
      <artifactId>unidbg-android</artifactId>
      <version>0.9.9</version>
    </dependency>
  </dependencies>
</project>
POM
mvn dependency:copy-dependencies -DoutputDirectory=. -DincludeScope=runtime
rm pom.xml
export UNIDBG_HOME=$HOME/.unidbg/0.9.9
```

`bin/run-runner.sh` reads `$UNIDBG_HOME` and assembles the runtime classpath.

---

## 4. Optional: external API services (only for `--mode aggressive`)

If you run the LLM hypothesis loop (S6) with engine-internal LLM calls:

- **DeepSeek** (default): sign up at https://platform.deepseek.com/ → put
  `DEEPSEEK_API_KEY=sk-...` in `.env`.
- **MiMo self-hosted**: run vLLM/SGLang serving the model and set
  `MIMO_API_KEY=...` + `MIMO_BASE_URL=http://...` + `LLM_BACKEND=mimo`.

When the engine is driven by an external agent (`utov agent-serve`), our
LLM calls go to the agent over NDJSON instead — engine pays $0 and no key
is needed.

---

## 5. Optional: static analysis tools (engine works without them)

`engine/static_tools.py` whitelists `radare2 / r2 / objdump / readelf / nm
/ strings / llvm-objdump / llvm-readelf / llvm-nm`. The engine never invokes
anything outside this set. Install only what your workflow needs.

macOS: `brew install binutils radare2`.

---

## 6. Platform support

| OS | Status |
|---|---|
| macOS arm64 (Apple Silicon) | ✅ developed/tested here |
| macOS x86_64 | ⚠ untested |
| Linux x86_64 | ⚠ untested; pyarrow + Java are platform-neutral |
| Linux ARM | ⚠ pyarrow wheel may need source build |
| Windows | ❌ not supported (bash launchers for the sample) |

---

## 7. Known gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `mvn` warns "POM for unidbg-android:0.9.8 is invalid" | unidbg-android 0.9.8 has a broken POM | use 0.9.9+ |
| `getTrace` returns 1 line | runner leaks state across calls | conformance C5 catches; fix the runner per `contracts/runner_interface.md §3.2` |
| `pyarrow` install hangs on Linux ARM | no prebuilt wheel | `pip install --no-binary pyarrow pyarrow` |
| `utov pipeline` aborts at C5 | runner doesn't satisfy "no cross-call side effects" | fix runner |
| `BudgetExceeded` mid-S6 | hit a `--budget-*` cap | `utov resume <work_dir> --raise-budget-*` |
