"""Pytest configuration for tools tests.

Provides fixtures to properly isolate module stubbing when running tests
in parallel with pytest-xdist.
"""

import sys

import pytest

# List of integration modules that may be stubbed during tests
STUBBED_MODULES = [
    "aerospace_mcp.integrations.rockets",
    "aerospace_mcp.integrations.atmosphere",
    "aerospace_mcp.integrations.aero",
    "aerospace_mcp.integrations.frames",
    "aerospace_mcp.integrations.orbits",
    "aerospace_mcp.integrations.propellers",
    "aerospace_mcp.integrations.trajopt",
]


@pytest.fixture(autouse=True)
def restore_stubbed_modules():
    """Save and restore all integration modules that may be stubbed.

    This fixture runs automatically before and after each test to ensure
    module stubs don't leak between tests when running in parallel with
    pytest-xdist.
    """
    # Save original modules
    originals = {}
    for mod_name in STUBBED_MODULES:
        if mod_name in sys.modules:
            originals[mod_name] = sys.modules[mod_name]

    yield

    # Restore original modules
    for mod_name in STUBBED_MODULES:
        if mod_name in originals:
            sys.modules[mod_name] = originals[mod_name]
        elif mod_name in sys.modules:
            # Module was added during test but wasn't there before
            del sys.modules[mod_name]
