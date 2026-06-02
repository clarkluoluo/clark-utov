"""capability_request.md §P2-2 / M9 — run_manifest.json tests."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from engine.manifest import (
    ManifestError,
    build_manifest,
    diff_manifests,
    guard_forbidden_flags,
    read_manifest,
    write_manifest,
)


def test_guard_blocks_heap_trace_without_override():
    with pytest.raises(ManifestError, match="heap.trace"):
        guard_forbidden_flags(["-Dheap.trace=true"], env={})


def test_guard_allows_heap_trace_with_explicit_debug():
    guard_forbidden_flags(["-Dheap.trace=true"], env={"DEBUG_HEAP": "1"})


def test_guard_passes_normal_flags():
    guard_forbidden_flags(
        ["-Dfoo.bar=baz", "-Xss4m"], env={})


def test_build_manifest_extracts_flags_and_jar_hash(tmp_path: Path):
    # synthesize a tiny jar
    jar = tmp_path / "runner.jar"
    jar.write_bytes(b"PK\x03\x04hello-jar-bytes")
    expected = hashlib.sha256(jar.read_bytes()).hexdigest()

    m = build_manifest(
        run_id="20260528-aabbcc",
        target_name="libreference.so",
        runner_cmd=["java", "-Dfoo=bar", "-jar", str(jar), "serve"],
        hook_rvas=[0xb7bb0, 0x31c4b0],
        extra_env={"UTOV_VMP_PREIMAGE_BL_RVAS": "0x322c90"},
    )
    assert m.jar_sha256 == expected
    assert m.jvm_flags == ("-Dfoo=bar",)
    assert m.hook_rvas == ("0xb7bb0", "0x31c4b0")
    assert m.utov_env == {"UTOV_VMP_PREIMAGE_BL_RVAS": "0x322c90"}


def test_build_refuses_forbidden_flag(tmp_path: Path):
    with pytest.raises(ManifestError):
        build_manifest(
            run_id="r", target_name="t",
            runner_cmd=["java", "-Dheap.trace=true", "-jar", "x.jar"],
            extra_env={},
        )


def test_write_then_read_roundtrips(tmp_path: Path):
    m = build_manifest(run_id="r", target_name="t",
                       runner_cmd=["java", "-jar", "/nonexistent.jar"])
    path = write_manifest(m, tmp_path)
    assert path == tmp_path / "run_manifest.json"
    rt = read_manifest(tmp_path)
    assert rt is not None
    assert rt["run_id"] == "r"
    assert rt["target_name"] == "t"


def test_diff_manifests_flags_mismatch():
    a = {"jvm_flags": ["-Dx=1"], "jar_sha256": "aaa",
         "hook_rvas": [], "utov_env": {}}
    b = {"jvm_flags": ["-Dx=2"], "jar_sha256": "aaa",
         "hook_rvas": [], "utov_env": {}}
    diffs = diff_manifests(a, b)
    assert any("jvm_flags" in d for d in diffs)


def test_diff_manifests_identical_empty():
    a = {"jvm_flags": [], "jar_sha256": None,
         "hook_rvas": [], "utov_env": {}}
    assert diff_manifests(a, a) == []


def test_missing_jar_yields_null_sha(tmp_path: Path):
    m = build_manifest(run_id="r", target_name="t",
                       runner_cmd=["java", "-jar", str(tmp_path / "missing.jar")])
    assert m.jar_sha256 is None
