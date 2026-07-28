"""Plane 2 — guard.

Deterministic stop-cock in front of every paid or write action: spend ceilings,
autonomy whitelist, loop detection, semantic dedup, kill switch, audit log.
Checks run cheapest first; a model is consulted only as a last resort.
"""
