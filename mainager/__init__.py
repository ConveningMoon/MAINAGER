"""MAINAGER — control plane between an autonomous agent and a generative API.

Four planes, one loop:

* ``preflight``  — intent compiler and plan-level cost estimate, before any charge
* ``guard``      — deterministic ceilings, whitelist, loop detection, kill switch
* ``postflight`` — automatic QA of the result and budgeted re-roll
* ``fastpath``   — low-latency inbound channel handling

Provider-specific code lives under ``mainager.providers``; the planes stay
provider-agnostic.
"""

__version__ = "0.1.0"
