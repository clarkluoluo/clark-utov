"""Builtin probe adapters — one module per probe.

Importing this package fires each module's ``@register_builtin_probe``
decorator. Mechanism probes (base profile, ``mechanism: true``):
``m1``, ``m3``, ``constant_provenance``, ``value_provenance``,
``watch_first_write``. Domain probes (currently only
``length_chain_check`` for ``vmp_algorithm_extraction``) live in the
same package but their profile entries omit the mechanism flag —
making them freely overridable / disable-able per §19.4.
"""

from engine.profile.probes import constant_provenance  # noqa: F401
from engine.profile.probes import length_chain_check  # noqa: F401
from engine.profile.probes import m1  # noqa: F401
from engine.profile.probes import m3  # noqa: F401
from engine.profile.probes import scope_boundary_gate  # noqa: F401
from engine.profile.probes import scope_upscale_gate  # noqa: F401
from engine.profile.probes import use_case_fork  # noqa: F401
from engine.profile.probes import value_provenance  # noqa: F401
from engine.profile.probes import watch_first_write  # noqa: F401

__all__: list[str] = []
