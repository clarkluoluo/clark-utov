"""Core layer facade (PLAN §13, DECISIONS D-019).

Single capability surface for both drivers (script_mode / agent_mode).
Implementation glues stages, verifier, hyp tree, store, and runner_client
together without imposing orchestration policy.
"""


from ._base import *  # noqa: F401,F403
from ._core import Core, open_live, _pick_reader, open_findings  # noqa: F401
