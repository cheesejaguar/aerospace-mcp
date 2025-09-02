"""Common stubs and helpers for testing MCP tools modules.

These helpers allow simulating the integrations modules without installing
heavy optional dependencies, while covering success and error branches.
"""

from __future__ import annotations

import builtins
import types
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


class StubModel:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)

    def model_dump(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class StubPerf:
    max_altitude_m: float = 1000.0
    max_velocity_ms: float = 100.0
    max_mach: float = 0.3
    apogee_time_s: float = 50.0
    burnout_time_s: float = 10.0
    max_q_pa: float = 12_000.0
    total_impulse_ns: float = 100_000.0
    specific_impulse_s: float = 250.0


@dataclass
class StubPoint:
    time_s: float = 0.0
    altitude_m: float = 0.0
    velocity_ms: float = 0.0
    acceleration_ms2: float = 0.0


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


def make_module(**members: Any) -> types.ModuleType:
    mod = types.ModuleType("stub_module")
    for k, v in members.items():
        setattr(mod, k, v)
    return mod
