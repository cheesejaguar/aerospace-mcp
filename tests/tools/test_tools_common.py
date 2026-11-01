"""Common helpers for testing MCP tools modules.

The tools are exercised against the real integration modules.  The helpers
here only simulate an *environment* in which an optional dependency (or the
integration module that needs it) cannot be imported, to cover the tools'
"not available" branches.
"""

from __future__ import annotations

import builtins
import sys
from contextlib import contextmanager
from typing import Any


@contextmanager
def import_raising(module_glob: str):
    """Context manager to make imports of a module raise ImportError.

    Args:
        module_glob: Substring to match in module import name.
    """

    original_import = builtins.__import__

    def raising_import(name: str, *args: Any, **kwargs: Any):  # type: ignore[override]
        if module_glob in name:
            raise ImportError(f"forced import error for {name}")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = raising_import  # type: ignore[assignment]
    try:
        yield
    finally:
        builtins.__import__ = original_import  # type: ignore[assignment]


def without_integrations(monkeypatch: Any, *names: str) -> None:
    """Make ``import aerospace_mcp.integrations.<name>`` raise ImportError.

    That is what the tools see when an integration's optional dependencies
    are not installed.  ``monkeypatch`` restores the real modules afterwards.
    """
    for name in names:
        monkeypatch.setitem(sys.modules, f"aerospace_mcp.integrations.{name}", None)
