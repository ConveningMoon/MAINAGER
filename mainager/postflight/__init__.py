"""Plane 3 — postflight.

Scores a finished generation against what was actually asked for, and decides
whether to accept it or re-roll within a fixed attempt and rouble budget.
Yields the headline metric: cost per *accepted* result.
"""
