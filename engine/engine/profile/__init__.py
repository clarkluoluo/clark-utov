"""Profile layer — declarable judgment semantics (v0.3.0 · PLAN §19).

Two-tier:

- **base profile** (`engine/profiles/base.json`) — mechanism baseline.
  M1 / M3 / `constant_provenance` framework / value-provenance principle /
  watch_first_write / §17 runner conformance. ``mechanism: true`` lock —
  cannot be disabled, overridden, or removed by any subprofile.
- **domain profile** (e.g. ``vmp_algorithm_extraction.json``,
  ``key_extraction.json``) — semantic layer. Evidence-class ordering,
  node-state vocabulary + role bindings, closure-gate composition,
  domain-specific probes, scope rules, routing rules. Freely declarable,
  extensible, overridable.

Step-1 surface (this commit): ``types`` + ``loader`` + ``registry`` +
``lint`` + an empty ``base.json`` shell. Mechanism entries (M1/M3/…)
land in step 2+ as their existing modules migrate to the ``Probe``
interface.
"""

from engine.profile.evidence_class_synth import (
    most_restrictive_class_id,
    synth_node_cap,
)
from engine.profile.gate_runtime import ConjunctiveGate, GateResult
from engine.profile.loader import ProfileLoadError, load_profile_file
from engine.profile.probe_runtime import (
    EvidenceClassCap,
    Probe,
    ProbeContext,
    ProbeResolveError,
    ProbeResult,
    StateView,
    Verdict,
    get_builtin_probe_class,
    list_builtin_probes,
    register_builtin_probe,
    resolve_probe_class,
)
from engine.profile.registry import (
    BASE_PROFILE_NAME,
    PROFILES_DIR,
    MergedProfile,
    ProfileMergeError,
    ProfileRegistry,
)
from engine.profile.routing_runtime import (
    RoutingTable,
    lint_actions_against_known,
)
from engine.profile.state_machine import StateMachine
from engine.profile.types import (
    CapMappingEntry,
    EvidenceClassSpec,
    GateSpec,
    Profile,
    ProbeSpec,
    RoutingRule,
    ScopeRule,
    StateSpec,
    TaskTemplateRef,
)

__all__ = [
    "BASE_PROFILE_NAME",
    "CapMappingEntry",
    "ConjunctiveGate",
    "EvidenceClassCap",
    "EvidenceClassSpec",
    "GateResult",
    "GateSpec",
    "MergedProfile",
    "PROFILES_DIR",
    "Probe",
    "ProbeContext",
    "ProbeResolveError",
    "ProbeResult",
    "ProbeSpec",
    "Profile",
    "ProfileLoadError",
    "ProfileMergeError",
    "ProfileRegistry",
    "RoutingRule",
    "RoutingTable",
    "ScopeRule",
    "StateMachine",
    "StateSpec",
    "StateView",
    "TaskTemplateRef",
    "Verdict",
    "get_builtin_probe_class",
    "lint_actions_against_known",
    "list_builtin_probes",
    "load_profile_file",
    "most_restrictive_class_id",
    "register_builtin_probe",
    "resolve_probe_class",
    "synth_node_cap",
]
