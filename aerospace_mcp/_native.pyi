"""Type stubs for the native (Rust) core, built from ``rust/src/python.rs``."""

from collections.abc import Callable, Sequence
from typing import Any, Protocol

Vec3 = list[float]

MU_EARTH: float
R_EARTH: float
J2_EARTH: float
OMEGA_EARTH: float
G0: float
__version__: str

class _Geometry(Protocol):
    dry_mass_kg: float
    propellant_mass_kg: float
    diameter_m: float
    cd: float
    thrust_curve: Sequence[Sequence[float]] | None

class AtmosphereTable:
    def __init__(
        self,
        altitudes_m: Sequence[float],
        densities_kg_m3: Sequence[float],
        speeds_of_sound_ms: Sequence[float],
    ) -> None: ...

def thrust_at_time(
    thrust_curve: Sequence[Sequence[float]] | None, time_s: float
) -> float: ...
def rocket_trajectory(
    geometry: _Geometry,
    atmosphere: AtmosphereTable,
    dt_s: float = 0.1,
    max_time_s: float = 300.0,
    launch_angle_deg: float = 90.0,
) -> list[tuple[float, float, float, float, float, float, float, float, float]]: ...
def analyze_performance(
    trajectory: Sequence[Any],
) -> tuple[float, float, float, float, float, float, float, float, float, float]: ...
def golden_section_search(
    objective: Callable[[float], float],
    lower_bound: float,
    upper_bound: float,
    tolerance: float = 1e-3,
    max_iterations: int = 100,
) -> tuple[float, float]: ...
def gradient_descent(
    objective: Callable[[list[float]], float],
    initial_params: Sequence[float],
    param_bounds: Sequence[tuple[float, float]],
    learning_rate: float = 0.01,
    tolerance: float = 1e-3,
    max_iterations: int = 100,
) -> tuple[list[float], float, int, bool]: ...
def optimize_launch_angle(
    geometry: _Geometry,
    atmosphere: AtmosphereTable,
    objective: str,
    lower_deg: float,
    upper_deg: float,
) -> tuple[float, float]: ...
def optimize_thrust_profile(
    geometry: _Geometry,
    atmosphere: AtmosphereTable,
    burn_time_s: float,
    total_impulse_target: float,
    n_segments: int,
    objective: str,
) -> tuple[list[float], float, int, bool, list[tuple[float, float]]]: ...
def kepler_solve(
    mean_anomaly_rad: float,
    eccentricity: float,
    tolerance: float = 1e-8,
    max_iterations: int = 50,
) -> float: ...
def elements_to_state(
    semi_major_axis_m: float,
    eccentricity: float,
    inclination_deg: float,
    raan_deg: float,
    arg_periapsis_deg: float,
    true_anomaly_deg: float,
    mu: float = ...,
) -> tuple[Vec3, Vec3]: ...
def state_to_elements(
    position_m: Sequence[float], velocity_ms: Sequence[float], mu: float = ...
) -> tuple[float, float, float, float, float, float]: ...
def orbit_properties(
    semi_major_axis_m: float,
    eccentricity: float,
    mu: float = ...,
    body_radius_m: float = ...,
) -> tuple[float, float, float, float, float]: ...
def propagate_j2(
    position_m: Sequence[float],
    velocity_ms: Sequence[float],
    time_span_s: float,
    time_step_s: float = 60.0,
    mu: float = ...,
    j2: float = ...,
    equatorial_radius_m: float = ...,
) -> list[tuple[float, Vec3, Vec3]]: ...
def ground_track(
    positions_m: Sequence[Sequence[float]], elapsed_s: Sequence[float]
) -> list[tuple[float, float, float]]: ...
def hohmann_transfer(
    r1_m: float, r2_m: float, mu: float = ...
) -> tuple[float, float, float, float, float]: ...
def lambert_izzo(
    mu: float,
    r1: Sequence[float],
    r2: Sequence[float],
    tof: float,
    revolutions: int = 0,
    prograde: bool = True,
    low_path: bool = True,
    max_iterations: int = 35,
    tolerance: float = 1e-8,
) -> tuple[Vec3, Vec3, float]: ...
