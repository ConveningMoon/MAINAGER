"""Plane 4 — fastpath.

Long-polls the inbound message channel and answers inside the window where the
reply is still free, routing any side effects through the guard whitelist.
"""
