"""Tests for the array backend abstraction layer."""

import numpy as numpy_module
import pytest


class TestArrayBackendBasics:
    """Test basic array backend functionality."""

    def test_default_backend_is_numpy(self):
        """Test that NumPy is the default backend."""
        from aerospace_mcp.integrations._array_backend import get_backend_info

        info = get_backend_info()
        assert info["current_backend"] == "numpy"
        assert info["numpy_available"] is True

    def test_np_proxy_works(self):
        """Test that the np proxy forwards to NumPy correctly."""
        from aerospace_mcp.integrations._array_backend import np

        arr = np.array([1, 2, 3])
        assert isinstance(arr, numpy_module.ndarray)
        assert list(arr) == [1, 2, 3]

    def test_np_proxy_math_functions(self):
        """Test that math functions work through the proxy."""
        from aerospace_mcp.integrations._array_backend import np

        arr = np.array([0, numpy_module.pi / 2, numpy_module.pi])
        sin_result = np.sin(arr)
        assert len(sin_result) == 3
        assert abs(sin_result[0] - 0.0) < 1e-10
        assert abs(sin_result[1] - 1.0) < 1e-10
        assert abs(sin_result[2] - 0.0) < 1e-10

    def test_np_proxy_constants(self):
        """Test that constants are accessible through the proxy."""
        from aerospace_mcp.integrations._array_backend import np

        assert abs(np.pi - numpy_module.pi) < 1e-15
        assert abs(np.e - numpy_module.e) < 1e-15


class TestBackendSwitching:
    """Test backend switching functionality."""

    def test_set_backend_numpy(self):
        """Test switching to NumPy backend."""
        from aerospace_mcp.integrations._array_backend import (
            get_backend_info,
            set_backend,
        )

        set_backend("numpy")
        info = get_backend_info()
        assert info["current_backend"] == "numpy"

    def test_set_backend_case_insensitive(self):
        """Test that backend names are case insensitive."""
        from aerospace_mcp.integrations._array_backend import (
            get_backend_info,
            set_backend,
        )

        set_backend("NumPy")
        info = get_backend_info()
        assert info["current_backend"] == "numpy"

        set_backend("NUMPY")
        info = get_backend_info()
        assert info["current_backend"] == "numpy"

    def test_set_backend_invalid_raises_error(self):
        """Test that invalid backend names raise ValueError."""
        from aerospace_mcp.integrations._array_backend import set_backend

        with pytest.raises(ValueError, match="Unknown backend"):
            set_backend("invalid_backend")

    def test_set_backend_cupy_unavailable_raises_error(self):
        """Test that CuPy backend raises ImportError if unavailable."""
        from aerospace_mcp.integrations._array_backend import (
            _CUPY_AVAILABLE,
            set_backend,
        )

        if not _CUPY_AVAILABLE:
            with pytest.raises(ImportError, match="CuPy is not installed"):
                set_backend("cupy")


class TestConversionFunctions:
    """Test array conversion functions."""

    def test_to_numpy_with_numpy_array(self):
        """Test to_numpy with a NumPy array returns it unchanged."""
        from aerospace_mcp.integrations._array_backend import np, to_numpy

        arr = np.array([1, 2, 3])
        result = to_numpy(arr)
        assert isinstance(result, numpy_module.ndarray)
        assert list(result) == [1, 2, 3]

    def test_to_numpy_with_list(self):
        """Test to_numpy with a list returns it unchanged."""
        from aerospace_mcp.integrations._array_backend import to_numpy

        data = [1, 2, 3]
        result = to_numpy(data)
        assert result == [1, 2, 3]

    def test_from_numpy_with_numpy_array(self):
        """Test from_numpy returns array in current backend format."""
        from aerospace_mcp.integrations._array_backend import (
            from_numpy,
            set_backend,
        )

        # Ensure we're on NumPy
        set_backend("numpy")
        arr = numpy_module.array([1, 2, 3])
        result = from_numpy(arr)
        assert isinstance(result, numpy_module.ndarray)
        assert list(result) == [1, 2, 3]


class TestBackendInfo:
    """Test get_backend_info function."""

    def test_get_backend_info_structure(self):
        """Test that get_backend_info returns expected structure."""
        from aerospace_mcp.integrations._array_backend import get_backend_info

        info = get_backend_info()
        assert "current_backend" in info
        assert "numpy_available" in info
        assert "cupy_available" in info

    def test_get_backend_info_numpy_version(self):
        """Test that NumPy version is included when available."""
        from aerospace_mcp.integrations._array_backend import get_backend_info

        info = get_backend_info()
        if info["numpy_available"]:
            assert "numpy_version" in info
            assert info["numpy_version"] == numpy_module.__version__


class TestBackendProxyErrors:
    """Test error handling in _BackendProxy."""

    def test_proxy_attribute_error_message(self):
        """Test that AttributeError has helpful message."""
        from aerospace_mcp.integrations import _array_backend

        # Save original state
        original_module = _array_backend._BACKEND_MODULE

        try:
            # Simulate no backend available
            _array_backend._BACKEND_MODULE = None

            with pytest.raises(AttributeError, match="No array backend available"):
                _ = _array_backend.np.array
        finally:
            # Restore original state
            _array_backend._BACKEND_MODULE = original_module
