"""Call every registered MCP tool against the real integrations.

Most tool tests stub the integration modules, which is how a whole class of
bugs (wrong call signatures, wrong attribute names, models passed to
json.dumps) went unnoticed: the stubs accepted anything. This module calls
each tool in ``ALL_TOOLS`` with realistic inputs and no stubs, and checks:

* the call returns a non-error result;
* any JSON it returns is strict (no NaN/Infinity);
* selected tools reproduce known answers from independent references.

``test_every_tool_has_a_smoke_case`` fails when a tool is added to the
registry without a case here.
"""

import json
import math

import pytest

from aerospace_mcp.tools.registry import ALL_TOOLS

ROCKET = {
    "dry_mass_kg": 20.0,
    "propellant_mass_kg": 80.0,
    "diameter_m": 0.2,
    "length_m": 3.0,
    "cd": 0.4,
    "thrust_curve": [[0.0, 4000.0], [8.0, 4000.0], [8.1, 0.0]],
}
LEO = {
    "semi_major_axis_m": 6.778e6,
    "eccentricity": 0.001,
    "inclination_deg": 51.6,
    "raan_deg": 0.0,
    "arg_periapsis_deg": 0.0,
    "true_anomaly_deg": 0.0,
    "epoch_utc": "2026-01-01T00:00:00Z",
}
WING = {
    "span_m": 10.0,
    "chord_root_m": 1.8,
    "chord_tip_m": 1.2,
    "sweep_deg": 5.0,
    "airfoil_root": "NACA2412",
}
WAYPOINTS = [
    {"time_s": 0.0, "position_m": [7e6, 0.0, 0.0], "velocity_ms": [0.0, 7546.0, 0.0]},
    {
        "time_s": 600.0,
        "position_m": [6.25e6, 3.15e6, 0.0],
        "velocity_ms": [-3400.0, 6740.0, 0.0],
    },
]

CASES = {
    "search_aerospace_tools": {"query": "orbit propagation"},
    "list_tool_categories": {},
    "search_airports": {"query": "Nice"},
    "plan_flight": {
        "departure": {"city": "San Jose", "iata": "SJC"},
        "arrival": {"city": "Denver", "iata": "DEN"},
        "aircraft": {"ac_type": "A320", "mass_kg": 60000},
    },
    "plan_multi_leg_flight": {
        "waypoints": [{"iata": "SJC"}, {"iata": "DEN"}, {"iata": "JFK"}],
        "aircraft": {"ac_type": "A320"},
    },
    "calculate_distance": {
        "lat1": 37.3626,
        "lon1": -121.929,
        "lat2": 35.7647,
        "lon2": 140.3864,
    },
    "get_aircraft_performance": {"aircraft_type": "A320", "distance_km": 2000},
    "get_aircraft_database": {"search": "A3"},
    "get_system_status": {},
    "convert_units": {"value": 100, "from_unit": "kts", "to_unit": "mps"},
    "get_atmosphere_profile": {"altitudes_m": [0, 1000, 11000, 20000, 47000]},
    "wind_model_simple": {"altitudes_m": [10, 50, 100]},
    "transform_frames": {
        "coordinates": {"x": 6378137.0, "y": 0.0, "z": 0.0},
        "from_frame": "ECEF",
        "to_frame": "GEODETIC",
    },
    "geodetic_to_ecef": {"latitude_deg": 0.0, "longitude_deg": 0.0},
    "ecef_to_geodetic": {"x_m": 0.0, "y_m": 0.0, "z_m": 6356752.314245},
    "wing_vlm_analysis": {
        "wing_config": WING,
        "flight_conditions": {"alpha_deg_list": [0, 2, 4], "mach": 0.2},
    },
    "airfoil_polar_analysis": {"airfoil_name": "NACA2412"},
    "calculate_stability_derivatives": {
        "wing_config": WING,
        "flight_conditions": {"alpha_deg": 2.0},
    },
    "get_airfoil_database": {},
    "propeller_bemt_analysis": {
        "propeller_geometry": {"diameter_m": 0.254, "pitch_m": 0.114, "num_blades": 2},
        "operating_conditions": {"rpm_list": [5000, 7000], "velocity_ms": 10},
    },
    "uav_energy_estimate": {
        "uav_config": {"mass_kg": 2.5, "disk_area_m2": 0.2, "num_motors": 4},
        "battery_config": {
            "capacity_ah": 5.0,
            "voltage_nominal_v": 14.8,
            "mass_kg": 0.6,
        },
    },
    "get_propeller_database": {},
    "rocket_3dof_trajectory": {
        "rocket_geometry": ROCKET,
        "launch_conditions": {"launch_angle_deg": 85},
    },
    "estimate_rocket_sizing": {"target_altitude_m": 10000, "payload_mass_kg": 5},
    "optimize_launch_angle": {"rocket_geometry": ROCKET},
    "elements_to_state_vector": {"orbital_elements": LEO},
    "state_vector_to_elements": {
        "state_vector": {
            "position_m": [7e6, 0.0, 0.0],
            "velocity_ms": [0.0, 7546.0, 0.0],
            "epoch_utc": "2026-01-01T00:00:00Z",
        }
    },
    "propagate_orbit_j2": {
        "initial_state": LEO,
        "propagation_time_s": 5400,
        "time_step_s": 60,
    },
    "calculate_ground_track": {
        "orbital_state": LEO,
        "duration_s": 5400,
        "time_step_s": 60,
    },
    "hohmann_transfer": {"r1_m": 6.778e6, "r2_m": 4.2164e7},
    "orbital_rendezvous_planning": {
        "chaser_elements": LEO,
        "target_elements": dict(LEO, semi_major_axis_m=6.79e6, true_anomaly_deg=10.0),
    },
    "lambert_problem_solver": {
        "r1_m": [5000e3, 10000e3, 2100e3],
        "r2_m": [-14600e3, 2500e3, 7000e3],
        "tof_s": 3600,
    },
    "kalman_filter_state_estimation": {
        "initial_state": [0.0, 0.0, 0.0, 0.0],
        "initial_covariance": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "process_noise": [
            [1e-4, 0, 0, 0],
            [0, 1e-4, 0, 0],
            [0, 0, 1e-4, 0],
            [0, 0, 0, 1e-4],
        ],
        "measurement_noise": [[0.01, 0], [0, 0.01]],
        "measurements": [
            {
                "time": float(k),
                "z": [1.0 * k, 2.0 * k],
                "H": [[1, 0, 0, 0], [0, 1, 0, 0]],
            }
            for k in range(1, 21)
        ],
    },
    "lqr_controller_design": {
        "A_matrix": [[0, 1], [0, 0]],
        "B_matrix": [[0], [1]],
        "Q_matrix": [[1, 0], [0, 1]],
        "R_matrix": [[1]],
    },
    "density_altitude_calculator": {"pressure_altitude_ft": 5000, "temperature_c": 30},
    "true_airspeed_converter": {
        "speed_value": 250,
        "speed_type": "CAS",
        "altitude_ft": 10000,
    },
    "stall_speed_calculator": {
        "weight_kg": 1100,
        "wing_area_m2": 16.2,
        "cl_max_clean": 1.6,
    },
    "weight_and_balance": {
        "basic_empty_weight_kg": 770,
        "basic_empty_arm_m": 1.0,
        "fuel_kg": 100,
        "fuel_arm_m": 1.2,
        "payload_items": [{"name": "pilot", "weight_kg": 80, "arm_m": 0.95}],
    },
    "takeoff_performance": {
        "weight_kg": 70000,
        "pressure_altitude_ft": 0,
        "temperature_c": 15,
    },
    "landing_performance": {
        "weight_kg": 60000,
        "pressure_altitude_ft": 0,
        "temperature_c": 15,
    },
    "fuel_reserve_calculator": {
        "regulation": "FAR_121_FLAG",
        "trip_fuel_kg": 10000,
        "cruise_fuel_flow_kg_hr": 2500,
        "flight_time_min": 240,
        "alternate_fuel_kg": 1500,
    },
    "optimize_thrust_profile": {
        "rocket_geometry": {k: v for k, v in ROCKET.items() if k != "thrust_curve"},
        "burn_time_s": 8.0,
        "total_impulse_target": 32000.0,
        "n_segments": 3,
    },
    "trajectory_sensitivity_analysis": {
        "rocket_geometry": ROCKET,
        "parameter_variations": {"cd": [0.35, 0.45]},
    },
    "genetic_algorithm_optimization": {
        "optimization_problem": {"trajectory": WAYPOINTS},
        "ga_parameters": {"population_size": 10, "generations": 5, "seed": 1},
    },
    "particle_swarm_optimization": {
        "optimization_problem": {"trajectory": WAYPOINTS},
        "pso_parameters": {"num_particles": 10, "max_iterations": 5, "seed": 1},
    },
    "porkchop_plot_analysis": {
        "departure_body": "Earth",
        "arrival_body": "Mars",
        "departure_date_range": ["2025-04-01T00:00:00Z", "2025-05-01T00:00:00Z"],
        "arrival_date_range": ["2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"],
    },
    "monte_carlo_uncertainty_analysis": {
        "nominal_trajectory": {"trajectory": WAYPOINTS},
        "uncertainty_parameters": {"position_m": {"std": 10.0}},
        "n_samples": 50,
        "analysis_options": {"seed": 1},
    },
    "format_data_for_tool": {
        "tool_name": "convert_units",
        "user_requirements": "100 knots to m/s",
    },
    "select_aerospace_tool": {"user_task": "convert 100 knots to m/s"},
}

# LLM-backed tools are opt-in (LLM_TOOLS_ENABLED); disabled is a valid result.
LLM_TOOLS = {"format_data_for_tool", "select_aerospace_tool"}

FAILURE_MARKERS = ("error", "not available", "traceback", '"success": false')


def _reject_non_finite(token):
    raise ValueError(f"non-finite number in tool JSON: {token}")


def _last_json_object(text: str):
    """The last top-level JSON object in ``text`` (tools often append one)."""
    try:
        return json.loads(text, parse_constant=_reject_non_finite)
    except json.JSONDecodeError:
        pass
    depth, start, last = 0, None, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                last = (start, i + 1)
    if last is None:
        return None
    return json.loads(text[last[0] : last[1]], parse_constant=_reject_non_finite)


def _run(name: str):
    return ALL_TOOLS[name](**CASES[name])


def test_every_tool_has_a_smoke_case():
    assert set(CASES) == set(ALL_TOOLS)


@pytest.mark.parametrize("name", sorted(CASES))
def test_tool_runs_against_real_integrations(name, monkeypatch):
    if name in LLM_TOOLS:
        # No LLM in CI: pin the documented default rather than inherit
        # whatever an earlier test left in the module.
        monkeypatch.setattr("aerospace_mcp.tools.agents.LLM_TOOLS_ENABLED", False)
    out = _run(name)
    assert isinstance(out, str) and out.strip()
    head = out.strip().splitlines()[0].lower()
    if name in LLM_TOOLS and "disabled" in head:
        return
    assert not any(marker in head for marker in FAILURE_MARKERS), out[:300]
    if "{" in out:
        _last_json_object(out)  # raises on NaN/Infinity or malformed JSON


# ---------------------------------------------------------------------------
# Known answers (independent references, not this codebase's own output)
# ---------------------------------------------------------------------------


def test_great_circle_distance_matches_geographiclib():
    data = _last_json_object(_run("calculate_distance"))
    assert data["distance_km"] == pytest.approx(8294.83, abs=0.01)


def test_unit_conversion():
    data = _last_json_object(_run("convert_units"))
    assert data["result"] == pytest.approx(100 * 1852 / 3600)


def test_geodetic_conversions_hit_wgs84_axes():
    ecef = _last_json_object(_run("geodetic_to_ecef"))["output"]
    assert ecef["x_m"] == pytest.approx(6378137.0)
    pole = _last_json_object(_run("ecef_to_geodetic"))["output"]
    assert pole["latitude_deg"] == pytest.approx(90.0)
    assert pole["altitude_m"] == pytest.approx(0.0, abs=1e-6)


def test_cas_to_tas_matches_isa_compressible_relations():
    # 250 KCAS at 10,000 ft ISA: qc from CAS at sea level, Mach at altitude.
    p0, a0, t0 = 101325.0, 340.294, 288.15
    h = 10000 * 0.3048
    t = t0 - 0.0065 * h
    p = p0 * (t / t0) ** 5.25588
    qc = p0 * ((1 + 0.2 * (250 * 0.514444 / a0) ** 2) ** 3.5 - 1)
    mach = math.sqrt(5 * ((qc / p + 1) ** (2 / 7) - 1))
    tas_kt = mach * a0 * math.sqrt(t / t0) / 0.514444
    out = _run("true_airspeed_converter")
    tas_line = next(
        line for line in out.splitlines() if line.strip().startswith("TAS:")
    )
    assert float(tas_line.split()[1]) == pytest.approx(tas_kt, abs=0.1)


def test_density_altitude_is_the_exact_isa_inversion():
    data = _last_json_object(_run("density_altitude_calculator"))
    assert data["density_altitude_ft"] == pytest.approx(7801, abs=5)


def test_stall_speed_formula():
    # Vs = sqrt(2 W / (rho S CLmax)) at sea level.
    vs = math.sqrt(2 * 1100 * 9.80665 / (1.225 * 16.2 * 1.6)) / 0.514444
    out = _run("stall_speed_calculator")
    line = next(line for line in out.splitlines() if "VS1 (Clean)" in line)
    assert float(line.split(":")[1].split()[0]) == pytest.approx(vs, abs=0.1)


def test_far_121_flag_reserve():
    # 14 CFR 121.645: 10% of flight time at cruise flow + alternate + 30 min hold.
    data = _last_json_object(_run("fuel_reserve_calculator"))
    text = json.dumps(data)
    assert "1000" in text  # 0.10 * 4 h * 2500 kg/h contingency


def test_airport_resolution_prefers_the_main_airport():
    assert "NCE" in _run("search_airports")


def test_lqr_double_integrator_gain():
    data = _last_json_object(_run("lqr_controller_design"))
    assert data["lqr_solution"]["gain_matrix_K"][0] == pytest.approx(
        [1.0, math.sqrt(3.0)], abs=1e-6
    )
    assert data["stability_analysis"]["closed_loop_stable"] is True


def test_kalman_recovers_velocity_from_position_only_measurements():
    data = _last_json_object(_run("kalman_filter_state_estimation"))
    vx, vy = data["output"]["final_state"][2:]
    assert (vx, vy) == pytest.approx((1.0, 2.0), abs=0.01)


def test_hohmann_leo_to_geo():
    data = _last_json_object(_run("hohmann_transfer"))
    assert data["total_delta_v_ms"] == pytest.approx(3854.0, abs=1.0)


def test_lambert_curtis_example_5_2():
    data = _last_json_object(_run("lambert_problem_solver"))
    assert data["v1_ms"] == pytest.approx([-5992.5, 1925.4, 3245.6], abs=0.5)


def test_orbit_element_round_trip_through_tools():
    state = _last_json_object(ALL_TOOLS["elements_to_state_vector"](LEO))
    state["epoch_utc"] = LEO["epoch_utc"]
    back = _last_json_object(ALL_TOOLS["state_vector_to_elements"](state))
    assert back["semi_major_axis_m"] == pytest.approx(
        LEO["semi_major_axis_m"], rel=1e-9
    )
    assert back["inclination_deg"] == pytest.approx(LEO["inclination_deg"], rel=1e-9)


def test_thrust_profile_optimum_is_no_worse_than_constant_thrust():
    data = _last_json_object(_run("optimize_thrust_profile"))
    assert data["optimal_objective"] >= data["initial_objective"]


def test_us76_at_11_km_geometric():
    # US Standard Atmosphere 1976: z = 11 km geometric is H = 10,981 m
    # geopotential, so T = 288.15 - 6.5e-3 * 10981 and p = 22,700 Pa.
    data = _last_json_object(ALL_TOOLS["get_atmosphere_profile"]([11000.0]))
    point = data[0] if isinstance(data, list) else data
    assert point["temperature_k"] == pytest.approx(216.774, abs=0.01)
    assert point["pressure_pa"] == pytest.approx(22700.0, rel=1e-3)


def test_power_law_wind_profile():
    out = ALL_TOOLS["wind_model_simple"]([5.0], 10.0, 270.0, "power_law")
    row = next(line for line in out.splitlines() if line.strip().startswith("5 "))
    assert float(row.split()[1]) == pytest.approx(10.0 * 0.5 ** (1 / 7), abs=0.05)


def test_propeller_efficiency_is_bounded_by_momentum_theory():
    out = ALL_TOOLS["propeller_bemt_analysis"](
        {"diameter_m": 0.254, "pitch_m": 0.114, "num_blades": 2},
        {"rpm_list": [7000.0], "velocity_ms": 10.0, "altitude_m": 0.0},
    )
    data = _last_json_object(out)
    point = data[0] if isinstance(data, list) else data
    # Ideal (actuator disk) efficiency 2 / (1 + sqrt(1 + T / (q A))).
    area = math.pi * 0.127**2
    q = 0.5 * 1.225 * 10.0**2
    ideal = 2 / (1 + math.sqrt(1 + point["thrust_n"] / (q * area)))
    assert point["thrust_n"] > 1.0
    assert 0.3 < point["efficiency"] < ideal
