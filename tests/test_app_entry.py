"""Lightweight tests for the app package entrypoint.

Ensures the app.main module imports cleanly so coverage includes it.
"""

import importlib


def test_app_module_imports() -> None:
    mod = importlib.import_module("app.main")
    assert hasattr(mod, "app")
