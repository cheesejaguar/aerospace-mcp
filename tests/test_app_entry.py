"""Lightweight tests for the app package entrypoint.

Ensures the app.main module imports cleanly so coverage includes it.
"""


def test_app_module_imports() -> None:
    import importlib

    mod = importlib.import_module("app.main")
    assert hasattr(mod, "app")
