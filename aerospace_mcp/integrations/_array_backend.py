"""
Array Backend Abstraction for NumPy/CuPy Compatibility

This module provides a drop-in abstraction layer for array operations.
By default, it uses NumPy, but can be configured to use CuPy for GPU
acceleration when available.

Usage:
    from aerospace_mcp.integrations._array_backend import np, get_backend_info

    # All array operations use `np` which can be numpy or cupy
    arr = np.array([1, 2, 3])
    result = np.sin(arr)

To switch to CuPy (GPU acceleration):
    from aerospace_mcp.integrations._array_backend import set_backend
    set_backend('cupy')

To force NumPy (CPU):
    set_backend('numpy')
"""

import importlib
import importlib.util
from typing import Any

# Default backend
_BACKEND_NAME = "numpy"
_BACKEND_MODULE: Any = None

# Track what's available
_NUMPY_AVAILABLE = False
_CUPY_AVAILABLE = False

# Try to import NumPy
try:
    import numpy

    _NUMPY_AVAILABLE = True
    _BACKEND_MODULE = numpy
except ImportError:
    # NumPy not installed - module will have limited functionality
    pass

# Check if CuPy is available (without importing it to avoid overhead)
_CUPY_AVAILABLE = importlib.util.find_spec("cupy") is not None


def set_backend(backend: str) -> None:
    """
    Set the array backend.

    Args:
        backend: Either 'numpy' or 'cupy'

    Raises:
        ImportError: If the requested backend is not available
        ValueError: If the backend name is invalid
    """
    global _BACKEND_NAME, _BACKEND_MODULE

    backend = backend.lower()

    if backend == "numpy":
        if not _NUMPY_AVAILABLE:
            raise ImportError("NumPy is not installed")
        _BACKEND_NAME = "numpy"
        _BACKEND_MODULE = importlib.import_module("numpy")

    elif backend == "cupy":
        if not _CUPY_AVAILABLE:
            raise ImportError(
                "CuPy is not installed. Install with: pip install cupy-cuda11x "
                "(replace 11x with your CUDA version)"
            )
        _BACKEND_NAME = "cupy"
        _BACKEND_MODULE = importlib.import_module("cupy")

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'numpy' or 'cupy'")


def get_backend_info() -> dict[str, Any]:
    """
    Get information about the current array backend.

    Returns:
        Dictionary with backend name, availability, and version info
    """
    info = {
        "current_backend": _BACKEND_NAME,
        "numpy_available": _NUMPY_AVAILABLE,
        "cupy_available": _CUPY_AVAILABLE,
    }

    if _NUMPY_AVAILABLE:
        import numpy

        info["numpy_version"] = numpy.__version__

    if _CUPY_AVAILABLE:
        try:
            import cupy

            info["cupy_version"] = cupy.__version__
            try:
                info["cuda_version"] = cupy.cuda.runtime.runtimeGetVersion()
            except Exception:
                # CUDA runtime not accessible
                info["cuda_version"] = "unknown"
        except ImportError:
            # CuPy detection was wrong, update flag
            info["cupy_available"] = False

    return info


def to_numpy(arr: Any) -> Any:
    """
    Convert array to NumPy array (useful when CuPy is active).

    Args:
        arr: Array to convert

    Returns:
        NumPy array
    """
    if _BACKEND_NAME == "cupy" and hasattr(arr, "get"):
        return arr.get()
    return arr


def from_numpy(arr: Any) -> Any:
    """
    Convert NumPy array to current backend array.

    Args:
        arr: NumPy array to convert

    Returns:
        Array in current backend format
    """
    if _BACKEND_NAME == "cupy" and _CUPY_AVAILABLE:
        import cupy

        return cupy.asarray(arr)
    return arr


# Expose the backend module as `np` for drop-in compatibility
# This allows: from _array_backend import np
# And then using np.array(), np.sin(), etc.
class _BackendProxy:
    """Proxy class that forwards attribute access to the current backend."""

    def __getattr__(self, name: str) -> Any:
        if _BACKEND_MODULE is None:
            raise AttributeError(
                f"Cannot access '{name}': No array backend available. "
                "Install numpy: pip install numpy"
            )
        return getattr(_BACKEND_MODULE, name)


np = _BackendProxy()

__all__ = [
    "np",
    "set_backend",
    "get_backend_info",
    "to_numpy",
    "from_numpy",
]
