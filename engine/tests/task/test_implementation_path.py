"""implementation_path: an OPTIONAL staged-plan field on a task spec.

Pins the shape contract: a brief may declare a light-to-heavy path, may omit
it entirely, and the loader validates only SHAPE (never path quality — the
brief 三审 is held). Also pins that audit-time tree rewrites preserve the field.
"""

from __future__ import annotations

import pytest

from engine.task.audit import _spec_with_children
from engine.task.implementation_path import (
    ImplementationPath,
    ImplementationPathError,
    ImplementationStage,
)
from engine.task.loader import TaskLoadError, parse_task_spec


_MIN = {"id": "t", "goal": "g", "done_criterion": {"kind": "node_closed", "node": "n1"},
        "nodes": [{"id": "n1"}]}


def _spec(**extra):
    return parse_task_spec({**_MIN, **extra})


# --- optional: absent is fine ----------------------------------------------

def test_absent_implementation_path_loads_as_none():
    assert _spec().implementation_path is None


def test_null_implementation_path_loads_as_none():
    assert _spec(implementation_path=None).implementation_path is None


# --- happy path -------------------------------------------------------------

def test_full_staged_path_parses():
    spec = _spec(implementation_path={
        "stages": [
            {"name": "calltrace", "intent": "find crypto entry",
             "upgrade_when": "entry not located by calltrace",
             "compliant_exit": "declare capability_blocked: no symbol"},
            {"name": "hook_io", "intent": "capture I/O",
             "upgrade_when": "I/O shape insufficient to induce formula"},
            {"name": "vmtrace", "intent": "full trace (heavy)"},
        ],
        "note": "light to heavy",
    })
    ip = spec.implementation_path
    assert isinstance(ip, ImplementationPath)
    assert [s.name for s in ip.stages] == ["calltrace", "hook_io", "vmtrace"]
    assert ip.stages[0].compliant_exit.startswith("declare capability_blocked")
    assert ip.stages[1].compliant_exit == ""   # unfilled = "not specified", valid
    assert ip.note == "light to heavy"


def test_empty_stages_is_valid():
    # the field may exist as a marker without stages
    ip = _spec(implementation_path={"stages": []}).implementation_path
    assert ip is not None and ip.stages == ()


# --- shape-only validation (NOT quality) ------------------------------------

def test_stage_requires_a_name():
    with pytest.raises(TaskLoadError):
        _spec(implementation_path={"stages": [{"intent": "no name"}]})


def test_duplicate_stage_name_rejected():
    with pytest.raises(TaskLoadError):
        _spec(implementation_path={"stages": [{"name": "a"}, {"name": "a"}]})


def test_stages_must_be_a_list():
    with pytest.raises(TaskLoadError):
        _spec(implementation_path={"stages": "calltrace"})


def test_stage_field_must_be_string():
    with pytest.raises(TaskLoadError):
        _spec(implementation_path={"stages": [{"name": "a", "intent": 5}]})


def test_quality_is_NOT_judged():
    """A single-stage path with no upgrade/exit, no gradient — would fail a
    quality review, but shape validation accepts it (三审 is held, and must not
    reach into task content)."""
    ip = _spec(implementation_path={"stages": [{"name": "just_vmtrace"}]}).implementation_path
    assert ip is not None and len(ip.stages) == 1


# --- parse() direct contract -----------------------------------------------

def test_parse_rejects_non_object():
    with pytest.raises(ImplementationPathError):
        ImplementationPath.parse(["stages"])


# --- preserved across audit tree rewrites -----------------------------------

def test_implementation_path_preserved_through_tree_rewrite():
    spec = _spec(implementation_path={"stages": [{"name": "calltrace"}]})
    # _spec_with_children claims "all other fields preserved verbatim" — the
    # rewrite that runs on every audit-time insert/replace/delete_child.
    rebuilt = _spec_with_children(spec, ())
    assert rebuilt.implementation_path is not None
    assert rebuilt.implementation_path.stages[0].name == "calltrace"
