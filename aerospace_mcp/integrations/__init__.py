"""
Aerospace MCP Integrations

This package contains specialized modules for different aerospace domains.
Each module provides wrappers around external libraries with graceful fallbacks
when optional dependencies are not available.

Available domains:
- atmosphere: ISA/COESA atmosphere models and wind profiles
- frames: Coordinate frame transformations (ECI/ECEF/ITRF)
- aero: Aircraft aerodynamics (VLM, airfoils, polars)
- propellers: Propeller analysis and UAV energy estimation
- rockets: Rocket performance, trajectories, and optimization
- orbits: Orbital mechanics, transfers, and propagation
- gnc: Guidance, navigation, and control tools
- trajopt: Trajectory optimization utilities
"""

from typing import Dict, List, Optional

# Availability flags - set by each module during import
AVAILABILITY_FLAGS: dict[str, bool] = {
    "atmosphere": True,  # Core module, always available
    "frames": False,
    "aero": False,
    "propellers": False,
    "rockets": False,
    "orbits": False,
    "gnc": False,
    "trajopt": False,
}

# Library version tracking
LIBRARY_VERSIONS: dict[str, str | None] = {}


def get_domain_status() -> dict[str, dict[str, any]]:
    """Get availability status and versions for all domains."""
    return {
        domain: {
            "available": available,
            "libraries": {
                lib: ver
                for lib, ver in LIBRARY_VERSIONS.items()
                if lib.startswith(domain)
            },
        }
        for domain, available in AVAILABILITY_FLAGS.items()
    }


def update_availability(domain: str, available: bool, libraries: dict[str, str] = None):
    """Update availability status for a domain."""
    AVAILABILITY_FLAGS[domain] = available
    if libraries:
        for lib, version in libraries.items():
            LIBRARY_VERSIONS[f"{domain}_{lib}"] = version


# Domain-specific imports (lazy loaded)
__all__ = [
    "AVAILABILITY_FLAGS",
    "LIBRARY_VERSIONS",
    "get_domain_status",
    "update_availability",
]
