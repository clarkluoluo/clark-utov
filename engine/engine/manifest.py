"""Run manifest writer (capability_request.md §P2-2, M9).

Background: a reference target parity run was silently injecting
``-Dheap.trace=true`` and the resulting Layer-2 80 vectors all broke at
``debugger_break``; the JVM flag was not in any manifest so reproducing
"the same hook *without* heap.trace" took days.

Fix: every utov run writes a ``run_manifest.json`` at session boot
listing the JVM flags, jar SHA-256, hook RVAs, and any environment
variable starting with ``UTOV_``. The session refuses to start if a
forbidden flag (default: ``heap.trace=true``) is set without explicit
``DEBUG_HEAP=1`` override.

Place: ``<run_dir>/run_manifest.json``. The conformance gate reads it
back when comparing archival runs to experiment runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FORBIDDEN_JVM_FLAGS: tuple[str, ...] = (
    "-Dheap.trace=true",        # M9 canonical example
)

ALLOWED_OVERRIDE_ENV: dict[str, str] = {
    "-Dheap.trace=true": "DEBUG_HEAP",
}


class ManifestError(RuntimeError):
    """Raised when a forbidden flag is set without an explicit override."""


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Snapshot of the runtime knobs that affect parity output."""
    run_id: str
    target_name: str
    jvm_flags: tuple[str, ...] = field(default_factory=tuple)
    jar_sha256: str | None = None
    runner_cmd: tuple[str, ...] = field(default_factory=tuple)
    hook_rvas: tuple[str, ...] = field(default_factory=tuple)
    utov_env: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id":      self.run_id,
            "target_name": self.target_name,
            "jvm_flags":   list(self.jvm_flags),
            "jar_sha256":  self.jar_sha256,
            "runner_cmd":  list(self.runner_cmd),
            "hook_rvas":   list(self.hook_rvas),
            "utov_env":    dict(self.utov_env),
            "notes":       list(self.notes),
        }


def _hash_jar(jar_path: Path | None) -> str | None:
    if jar_path is None or not jar_path.exists():
        return None
    h = hashlib.sha256()
    with jar_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


_JAR_RE = re.compile(r"^(.+\.jar)$")


def _extract_jar(cmd: list[str] | tuple[str, ...]) -> Path | None:
    if not cmd:
        return None
    for i, a in enumerate(cmd):
        if a in ("-jar", "--jar") and i + 1 < len(cmd):
            return Path(cmd[i + 1])
        m = _JAR_RE.match(a)
        if m:
            return Path(m.group(1))
    return None


def _extract_jvm_flags(cmd: list[str] | tuple[str, ...]) -> list[str]:
    return [a for a in (cmd or []) if isinstance(a, str) and a.startswith("-D")]


def _collect_utov_env(env: dict[str, str] | None = None) -> dict[str, str]:
    src = env if env is not None else os.environ
    return {k: v for k, v in src.items() if k.startswith("UTOV_") or k == "DEBUG_HEAP"}


def guard_forbidden_flags(jvm_flags: list[str] | tuple[str, ...],
                          env: dict[str, str] | None = None) -> None:
    """Raise ManifestError if a forbidden flag is set without its allowed
    override env var. Used at session boot to refuse parity runs that
    would silently produce different output."""
    src_env = env if env is not None else os.environ
    for flag in jvm_flags:
        if flag in FORBIDDEN_JVM_FLAGS:
            override = ALLOWED_OVERRIDE_ENV.get(flag)
            if override and src_env.get(override) == "1":
                continue
            raise ManifestError(
                f"forbidden JVM flag {flag!r} requires "
                f"{override}=1 to be set explicitly (M9)."
            )


def build_manifest(
    *,
    run_id: str,
    target_name: str,
    runner_cmd: list[str] | tuple[str, ...] | None = None,
    hook_rvas: list[int] | tuple[int, ...] | None = None,
    extra_env: dict[str, str] | None = None,
    notes: list[str] | None = None,
) -> RunManifest:
    cmd = tuple(runner_cmd or ())
    flags = tuple(_extract_jvm_flags(cmd))
    guard_forbidden_flags(flags, env=extra_env)
    jar = _extract_jar(cmd)
    jar_hash = _hash_jar(jar)
    rvas = tuple(f"0x{x:x}" for x in (hook_rvas or ()))
    return RunManifest(
        run_id=run_id, target_name=target_name,
        jvm_flags=flags, jar_sha256=jar_hash,
        runner_cmd=cmd, hook_rvas=rvas,
        utov_env=_collect_utov_env(extra_env),
        notes=tuple(notes or ()),
    )


def write_manifest(manifest: RunManifest, run_dir: Path,
                   filename: str = "run_manifest.json") -> Path:
    path = Path(run_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False))
    return path


def read_manifest(run_dir: Path, filename: str = "run_manifest.json") -> dict[str, Any] | None:
    path = Path(run_dir) / filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def diff_manifests(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Return human-readable diffs between two manifests. Empty list =
    identical for parity purposes."""
    out: list[str] = []
    for key in ("jvm_flags", "jar_sha256", "hook_rvas", "utov_env"):
        if a.get(key) != b.get(key):
            out.append(f"{key}: {a.get(key)!r} != {b.get(key)!r}")
    return out
