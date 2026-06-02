"""Doctor — scan host for clark-utov dependencies and report green/yellow/red.

Usage:
    python3 -m engine.doctor
    utov doctor

Exit code: 0 if no red findings, 1 otherwise. Yellow does NOT fail — those
are warnings (e.g. optional static tools not installed).
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# --- compatibility matrix (D-026 / DEPENDENCIES.md) -------------------------

REQUIRED_PYTHON = (3, 11)
KNOWN_GOOD_UNIDBG = {"0.9.9"}
KNOWN_BAD_UNIDBG  = {"0.9.8": "POM is broken, transitive deps don't resolve"}

REQUIRED_PY_PACKAGES = [
    ("openai",       "1.50.0", "LLM API client"),
    ("dotenv",       "1.0.0",  "env loading"),
    ("jsonschema",   "4.21.0", "LLM output schema"),
    ("click",        "8.1.0",  "CLI"),
    ("tqdm",         "4.66.0", "progress"),
    ("pyarrow",      "15.0.0", "parquet stage outputs (optional but recommended)"),
]

STATIC_TOOL_WHITELIST = [
    ("strings", "binutils — quick static strings extraction"),
    ("nm",      "binutils — symbol listing"),
    ("objdump", "binutils — disassembly"),
    ("readelf", "binutils — ELF inspection"),
    ("radare2", "deeper static analysis (optional)"),
    ("r2",      "alias for radare2"),
    ("llvm-objdump", "LLVM binutils alternative"),
    ("llvm-readelf", "LLVM binutils alternative"),
    ("llvm-nm",      "LLVM binutils alternative"),
]


class Level:
    OK   = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class Finding:
    level: str
    component: str
    detail: str


def _ver_tuple(v: str) -> tuple[int, ...]:
    out: list[int] = []
    for part in v.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits or "0"))
    return tuple(out)


def _check_python() -> Finding:
    have = sys.version_info[:2]
    if have >= REQUIRED_PYTHON:
        return Finding(Level.OK, "python",
                       f"{sys.version.split()[0]} (>= {'.'.join(map(str, REQUIRED_PYTHON))} required)")
    return Finding(Level.FAIL, "python",
                   f"{sys.version.split()[0]} — need >= {'.'.join(map(str, REQUIRED_PYTHON))}")


def _check_py_package(pkg: str, min_ver: str, description: str) -> Finding:
    try:
        # python-dotenv exports as 'dotenv'
        mod = importlib.import_module(pkg)
    except ImportError:
        return Finding(Level.FAIL, f"python:{pkg}",
                       f"not installed — {description}. Install: pip install '{pkg}>={min_ver}'")
    ver = getattr(mod, "__version__", None) or "(unknown)"
    if ver != "(unknown)" and _ver_tuple(ver) < _ver_tuple(min_ver):
        return Finding(Level.WARN, f"python:{pkg}",
                       f"version {ver} — engine pins to >= {min_ver}")
    return Finding(Level.OK, f"python:{pkg}", f"version {ver}")


def _check_command(cmd: str, version_flag: str = "--version") -> tuple[bool, str]:
    path = shutil.which(cmd)
    if not path:
        return False, ""
    try:
        out = subprocess.run([cmd, version_flag], capture_output=True, text=True,
                              timeout=5).stdout or ""
    except Exception:
        out = ""
    return True, out.strip().splitlines()[0] if out else "(version unknown)"


def _check_java() -> Finding:
    ok, ver = _check_command("java", "-version")
    if not ok:
        try:
            r = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=5)
            ver = (r.stderr or "").strip().splitlines()[0] if r.stderr else ""
            ok = bool(ver)
        except Exception:
            ok = False
    if not ok:
        return Finding(Level.FAIL, "java",
                       "not found — install JDK 17+ (e.g. brew install openjdk@21)")
    return Finding(Level.OK, "java", ver)


def _check_maven() -> Finding:
    ok, ver = _check_command("mvn", "--version")
    if not ok:
        return Finding(Level.WARN, "maven",
                       "not found — only needed if rebuilding the test runner; "
                       "install with `brew install maven`")
    return Finding(Level.OK, "maven", ver)


def _check_ndk() -> Finding:
    candidates = [
        Path.home() / "Library/Android/sdk/ndk/29.0.14206865",
        Path.home() / "Library/Android/sdk/ndk/25.1.8937393",
    ]
    found = [p for p in candidates if p.exists()]
    if not found:
        return Finding(Level.WARN, "ndk",
                       "not found — only needed if rebuilding libsha256.so; "
                       "install via Android Studio SDK Manager (NDK 25 or 29)")
    return Finding(Level.OK, "ndk", f"found at {found[0]}")


def _check_unidbg_home() -> list[Finding]:
    home = os.environ.get("UNIDBG_HOME")
    if not home:
        return [Finding(Level.WARN, "unidbg",
                        "UNIDBG_HOME not set — only needed if your runner uses "
                        "unidbg. The engine itself doesn't require it. See "
                        "DEPENDENCIES.md §unidbg for the bring-your-own recipe.")]
    p = Path(home)
    if not p.is_dir():
        return [Finding(Level.FAIL, "unidbg",
                        f"UNIDBG_HOME={home} is not a directory")]
    jars = list(p.glob("unidbg-android-*.jar"))
    if not jars:
        return [Finding(Level.FAIL, "unidbg",
                        f"no unidbg-android-*.jar in {home}; see bin/run-runner.sh "
                        "for the dependency:copy-dependencies recipe")]
    ver = jars[0].stem.removeprefix("unidbg-android-")
    findings = [Finding(Level.OK, "unidbg", f"found {jars[0].name} (UNIDBG_HOME={home})")]
    if ver in KNOWN_BAD_UNIDBG:
        findings.append(Finding(Level.FAIL, "unidbg:version",
                                f"version {ver} is known broken: {KNOWN_BAD_UNIDBG[ver]}"))
    elif ver not in KNOWN_GOOD_UNIDBG:
        findings.append(Finding(Level.WARN, "unidbg:version",
                                f"version {ver} not in tested set ({sorted(KNOWN_GOOD_UNIDBG)})"))
    # Spot-check a few transitive deps that must be there.
    must_have = ["unicorn", "capstone", "jna", "fastjson"]
    missing = [m for m in must_have if not list(p.glob(f"{m}-*.jar"))]
    if missing:
        findings.append(Finding(Level.FAIL, "unidbg:transitives",
                                f"missing transitive jars in {home}: {missing}. "
                                "Re-run mvn dependency:copy-dependencies."))
    return findings


def _check_triton() -> Finding:
    """Triton is an optional symex backend (BR-2 P1.5). Engine works without
    it (the concrete-DFG path is the default), so missing-Triton is a WARN."""
    try:
        from engine.stages import s3_triton_symex
    except Exception as e:
        return Finding(Level.WARN, "triton",
                       f"engine.stages.s3_triton_symex import failed: {e}")
    if s3_triton_symex.is_available():
        return Finding(Level.OK, "triton",
                       "Triton bindings importable — `--symex triton` enabled")
    return Finding(Level.WARN, "triton",
                   f"Triton not installed ({s3_triton_symex.unavailable_reason()}). "
                   f"Engine works without it; install via `pip install triton-library` "
                   f"to enable `--symex triton` for richer S3 ASTs.")


def _check_static_tools() -> list[Finding]:
    out: list[Finding] = []
    for tool, desc in STATIC_TOOL_WHITELIST:
        if shutil.which(tool):
            out.append(Finding(Level.OK, f"tool:{tool}", desc))
        else:
            out.append(Finding(Level.WARN, f"tool:{tool}", f"not installed — {desc}"))
    return out


def _check_env_keys() -> list[Finding]:
    out: list[Finding] = []
    if not os.environ.get("DEEPSEEK_API_KEY"):
        out.append(Finding(Level.WARN, "env:DEEPSEEK_API_KEY",
                           "not set — required only for `--mode aggressive` (LLM S6 loop)"))
    else:
        out.append(Finding(Level.OK, "env:DEEPSEEK_API_KEY", "set"))
    return out


def _check_test_target(sample_dir: Path | None = None) -> list[Finding]:
    """Sample-fixture presence check.

    BR-2 §9: previously scanned hard-coded `<repo-root>/example/runner-sha256/...` paths
    that don't exist when the engine is installed from a wheel — every probe
    yielded a confusing "not built" warning. Now: only run the check when the
    user (or CLI flag) explicitly points to a sample dir.
    """
    if sample_dir is None:
        env_dir = os.environ.get("UTOV_SAMPLE_DIR")
        if env_dir:
            sample_dir = Path(env_dir)
    if sample_dir is None:
        return [Finding(
            Level.OK, "sample",
            "skipped — pass --sample-dir or set UTOV_SAMPLE_DIR to a "
            "example/runner-sha256/ tree to enable bundled-fixture checks",
        )]
    sample_dir = sample_dir.resolve()
    if not sample_dir.is_dir():
        return [Finding(Level.WARN, "sample",
                        f"--sample-dir {sample_dir} is not a directory")]
    out: list[Finding] = []
    jar = sample_dir / "runner/target/sha256-test-runner-0.1.0.jar"
    so  = sample_dir / "libs/arm64-v8a/libsha256.so"
    out.append(Finding(Level.OK if jar.exists() else Level.WARN, "test:runner-jar",
                       str(jar) if jar.exists() else f"not built: {jar}"))
    out.append(Finding(Level.OK if so.exists() else Level.WARN, "test:libsha256",
                       str(so) if so.exists() else f"not built: {so}"))
    return out


def _check_capabilities() -> list[Finding]:
    """Report THIS build's capability tokens (dev-recovery-blockkind §需求2).

    Each token is a feature-milestone marker present in the source ⇔ the build
    provides it. Surfaced so an agent can confirm a stamp's build actually carries
    a feature before trusting a terminal that relies on it (a self-consistent /
    commit_ok build that pre-dates the feature lacks the token →串台 guard)."""
    from .capabilities import (
        TERMINAL_REQUIRES,
        collect_build_capabilities,
        coverage_for_terminal,
    )
    caps = collect_build_capabilities()
    out = [Finding(Level.OK, "capabilities",
                   "build provides: " + ", ".join(sorted(caps)))]
    # Per-terminal coverage: WARN any terminal whose required tokens this build
    # lacks (it would have produced a coverage_ok:false stamp at runtime).
    for terminal in sorted(TERMINAL_REQUIRES):
        ok, missing, warn = coverage_for_terminal(terminal)
        if not ok:
            out.append(Finding(Level.WARN, f"capability:{terminal}",
                               warn or f"missing {sorted(missing)}"))
    return out


# --- main entrypoint ---


def run_doctor(sample_dir: Path | None = None) -> dict[str, list[Finding]]:
    """Returns findings grouped by category:
        engine  — actually required to run the engine. fail = engine broken.
        runner  — required only if you use the OPTIONAL Java sample runner
                  in example/runner-sha256/. Most users write their own runner; for them
                  these are not relevant.
        env     — API keys / config. Only needed for aggressive mode.
        sample  — presence of the bundled sample target + sample runner jar.
        tools   — optional static analysis tools (radare2, objdump, ...).
                  Engine works without them; they enrich some stages.
    """
    return {
        "engine": [
            _check_python(),
            *[_check_py_package(p, v, d) for p, v, d in REQUIRED_PY_PACKAGES],
        ],
        "runner": [
            _check_java(),
            _check_maven(),
            _check_ndk(),
            *_check_unidbg_home(),
        ],
        "env": _check_env_keys(),
        "sample": _check_test_target(sample_dir),
        "symex": [_check_triton()],
        "capabilities": _check_capabilities(),
        "tools": _check_static_tools(),
    }


_SECTION_HEADERS = {
    "engine": "Engine requirements (must all pass for the engine to run)",
    "runner": "Optional: deps if your runner uses Java + unidbg (BYO; engine doesn't ship one)",
    "env":    "Optional: env vars (needed only for --mode aggressive)",
    "sample": "Optional: in-repo test fixtures (won't be present in a wheel install)",
    "symex":  "Optional: symbolic-execution backend (Triton; concrete DFG works without it)",
    "capabilities": "Build capability tokens (which recovery features THIS build's source provides)",
    "tools":  "Optional: static analysis tools (engine works without them)",
}


def main(sample_dir: Path | None = None) -> int:
    # Allow `python3 -m engine.doctor --sample-dir PATH` directly without
    # going through click.
    if sample_dir is None and len(sys.argv) >= 3 and sys.argv[1] == "--sample-dir":
        sample_dir = Path(sys.argv[2])
    groups = run_doctor(sample_dir=sample_dir)
    icon = {Level.OK: "✓", Level.WARN: "!", Level.FAIL: "✗"}
    color = {Level.OK: "\033[32m", Level.WARN: "\033[33m", Level.FAIL: "\033[31m"}
    reset = "\033[0m"
    use_color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    n_ok = n_warn = n_fail_engine = n_fail_other = 0
    for group_name, findings in groups.items():
        if not findings:
            continue
        print(f"\n[{group_name}] {_SECTION_HEADERS[group_name]}")
        for f in findings:
            prefix = (f"{color[f.level]}{icon[f.level]}{reset}"
                      if use_color else icon[f.level])
            print(f"  {prefix}  {f.component:24} {f.detail}")
            if f.level == Level.OK:
                n_ok += 1
            elif f.level == Level.WARN:
                n_warn += 1
            elif group_name == "engine":
                n_fail_engine += 1
            else:
                n_fail_other += 1

    print()
    print(f"  {n_ok} OK, {n_warn} warning, "
          f"{n_fail_engine} engine fail, {n_fail_other} other fail")
    if n_fail_engine == 0 and n_fail_other == 0:
        print("  Engine is ready to run.")
    elif n_fail_engine == 0:
        print("  Engine OK. Some optional groups have failures — only matters "
              "if you intend to use them.")
    else:
        print("  Engine has missing dependencies — fix the [engine] failures first.")
    # Only engine-level failures block usage of the engine itself.
    return 0 if n_fail_engine == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
