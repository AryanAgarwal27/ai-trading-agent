"""Shared pytest fixtures grouped by topic.

Fixtures defined here are re-exported from ``tests/conftest.py`` so
pytest auto-discovers them in any test file under ``tests/`` without
explicit imports. New topic modules (``hitl.py``, future ``freqtrade.py``,
etc.) follow the same pattern: define the fixtures here, re-export
from the root ``conftest.py``.
"""
