"""Adapter for the Vibe Agent API.

Base URL ``https://lk.vibemarketolog.ru/api/agent``, bearer auth, scope-limited.
``GET /capabilities`` is the source of truth for models, parameters and limits;
nothing about the catalog is hardcoded outside a snapshot used for tests.
"""
