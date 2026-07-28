"""MAINAGER — control plane between an autonomous agent and a generative API.

Three planes built, one designed:

* ``preflight``  — intent compiler and plan-level cost estimate, before any charge
* ``guard``      — deterministic ceilings, whitelist, loop detection, kill switch
* ``postflight`` — retrospective cost analysis; result QA is designed, not built

A fourth plane, a low-latency inbound channel, is designed in DESIGN.md and
deliberately not implemented: it cannot be demonstrated without inbound traffic.

Provider-specific code lives under ``mainager.providers``; the planes stay
provider-agnostic.
"""

__version__ = "0.1.0"
