# Aerospace MCP Tool Ideas

This document contains 100 ideas for new tools for the Aerospace MCP, stack-ranked by impact and feasibility.

## Implementation Status

**Top 10 tools have been implemented!** (January 2026)

| Tool | Status | Location |
|------|--------|----------|
| `lambert_problem_solver` | ✅ Implemented | `tools/orbits.py` |
| `density_altitude_calculator` | ✅ Implemented | `tools/performance.py` |
| `true_airspeed_converter` | ✅ Implemented | `tools/performance.py` |
| `weight_and_balance` | ✅ Implemented | `tools/performance.py` |
| `kalman_filter_state_estimation` | ✅ Implemented | `tools/gnc.py` |
| `stall_speed_calculator` | ✅ Implemented | `tools/performance.py` |
| `takeoff_performance` | ✅ Implemented | `tools/performance.py` |
| `landing_performance` | ✅ Implemented | `tools/performance.py` |
| `fuel_reserve_calculator` | ✅ Implemented | `tools/performance.py` |
| `lqr_controller_design` | ✅ Implemented | `tools/gnc.py` |

All 10 tools have comprehensive test coverage in `tests/tools/`.

---

## Part 1: Complete List of 100 Tool Ideas

### Category: GNC (Guidance, Navigation, Control) - 12 Ideas

| # | Tool Name | Description | Feasibility | Impact |
|---|-----------|-------------|-------------|--------|
| 1 | `kalman_filter_state_estimation` | Extended Kalman Filter for sensor fusion and state estimation | High (filterpy integrated) | Very High |
| 2 | `lqr_controller_design` | Linear Quadratic Regulator synthesis for optimal control | High (control lib ready) | Very High |
| 3 | `pid_tuning_ziegler_nichols` | PID controller tuning using Ziegler-Nichols method | High | High |
| 4 | `autopilot_gains_calculation` | Calculate autopilot gains for altitude/heading hold | Medium | High |
| 5 | `trajectory_tracking_controller` | Design controller for following a reference trajectory | Medium | High |
| 6 | `sensor_noise_simulation` | Simulate IMU, GPS, and altimeter sensor noise models | High | Medium |
| 7 | `navigation_filter_design` | Design complementary or Kalman navigation filter | Medium | High |
| 8 | `attitude_controller_quaternion` | Quaternion-based attitude controller design | Medium | Medium |
| 9 | `guidance_law_proportional_nav` | Proportional navigation guidance law for intercept | Medium | Medium |
| 10 | `waypoint_following_dubins` | Dubins path planning for waypoint following | Medium | High |
| 11 | `landing_guidance_ils` | ILS glideslope/localizer guidance simulation | Low | Medium |
| 12 | `trim_condition_solver` | Solve for aircraft trim conditions at given flight state | Medium | High |

### Category: Advanced Aerodynamics - 14 Ideas

| # | Tool Name | Description | Feasibility | Impact |
|---|-----------|-------------|-------------|--------|
| 13 | `full_aircraft_drag_buildup` | Complete drag breakdown (zero-lift, induced, parasitic) | High | Very High |
| 14 | `control_surface_effectiveness` | Elevator/aileron/rudder control power estimation | High | Very High |
| 15 | `fuselage_aerodynamics` | Body lift, drag, and moment calculations | Medium | High |
| 16 | `tail_sizing_analysis` | Horizontal/vertical tail sizing for stability | Medium | High |
| 17 | `high_lift_device_analysis` | Flap and slat effectiveness estimation | Medium | High |
| 18 | `transonic_drag_rise` | Wave drag estimation for transonic flight | Medium | Medium |
| 19 | `supersonic_aerodynamics` | Shock-expansion theory for supersonic vehicles | Low | Medium |
| 20 | `ground_effect_analysis` | Aerodynamic changes near ground during takeoff/landing | Medium | Medium |
| 21 | `downwash_calculation` | Wing downwash effects on tail | High | Medium |
| 22 | `induced_drag_analysis` | Span loading optimization for minimum induced drag | High | High |
| 23 | `compressibility_correction` | Prandtl-Glauert correction for high-speed flight | High | Medium |
| 24 | `stall_speed_calculator` | Calculate stall speeds for different configurations | High | Very High |
| 25 | `oswald_efficiency_estimation` | Estimate Oswald efficiency factor from geometry | High | High |
| 26 | `wetted_area_calculator` | Calculate aircraft wetted area for drag estimation | High | High |

### Category: Propulsion - 10 Ideas

| # | Tool Name | Description | Feasibility | Impact |
|---|-----------|-------------|-------------|--------|
| 27 | `turbofan_cycle_analysis` | On-design turbofan thermodynamic cycle | Medium | Very High |
| 28 | `engine_off_design_performance` | Off-design thrust and SFC at various conditions | Low | High |
| 29 | `propeller_selection_tool` | Match propeller to motor/engine for given mission | High | High |
| 30 | `electric_motor_sizing` | Size electric motor for UAV/eVTOL application | High | Very High |
| 31 | `battery_discharge_model` | Model battery voltage vs SoC and current draw | High | High |
| 32 | `fuel_tank_capacity_analysis` | Fuel volume and CG shift during flight | Medium | Medium |
| 33 | `engine_inlet_sizing` | Size engine inlet for given mass flow and Mach | Medium | Medium |
| 34 | `nozzle_expansion_analysis` | Rocket/jet nozzle expansion and thrust | Medium | Medium |
| 35 | `specific_impulse_calculator` | Calculate Isp for various propellant combinations | High | High |
| 36 | `throttle_response_model` | Model engine throttle lag and response dynamics | Low | Low |

### Category: Orbital Mechanics (Advanced) - 12 Ideas

| # | Tool Name | Description | Feasibility | Impact |
|---|-----------|-------------|-------------|--------|
| 37 | `lambert_problem_solver` | Two-body trajectory determination for given TOF | High (stubbed exists) | Very High |
| 38 | `bielliptic_transfer` | Calculate bi-elliptic transfer for large orbit changes | High | High |
| 39 | `plane_change_maneuver` | Non-coplanar orbit transfer calculations | High | High |
| 40 | `low_thrust_spiral_trajectory` | Ion/electric propulsion spiral trajectory | Medium | High |
| 41 | `lunar_transfer_trajectory` | Trans-lunar injection and lunar orbit insertion | Medium | High |
| 42 | `gravity_assist_analysis` | Planetary flyby trajectory design | Medium | Very High |
| 43 | `station_keeping_budget` | GEO/LEO station keeping delta-v budget | High | High |
| 44 | `eclipse_timing_prediction` | Satellite eclipse entry/exit time calculation | High | High |
| 45 | `satellite_visibility_pass` | Ground station visibility and pass prediction | High | Very High |
| 46 | `constellation_design_walker` | Walker constellation configuration analysis | Medium | High |
| 47 | `drag_decay_lifetime` | LEO satellite lifetime due to atmospheric drag | High | High |
| 48 | `solar_radiation_pressure` | SRP perturbation effects on orbit | Medium | Medium |

### Category: Mission Planning - 10 Ideas

| # | Tool Name | Description | Feasibility | Impact |
|---|-----------|-------------|-------------|--------|
| 49 | `multi_leg_flight_plan` | Optimize multi-stop flight routing | Medium | Very High |
| 50 | `fuel_reserve_calculator` | Calculate required fuel reserves per regulations | High | Very High |
| 51 | `alternate_airport_selection` | Find suitable alternate airports with weather | Medium | High |
| 52 | `flight_envelope_protection` | Check if flight profile stays within envelope | High | High |
| 53 | `weight_and_balance` | Aircraft W&B calculation for loading | High | Very High |
| 54 | `takeoff_performance` | Takeoff distance and climb gradient calculation | High | Very High |
| 55 | `landing_performance` | Landing distance for given conditions | High | Very High |
| 56 | `cruise_altitude_optimizer` | Find optimal cruise altitude for given weight | High | High |
| 57 | `cost_index_optimization` | Optimize speed for fuel/time cost tradeoff | Medium | High |
| 58 | `etops_planning` | Extended operations planning and diversion analysis | Low | Medium |

### Category: Atmospheric Science - 8 Ideas

| # | Tool Name | Description | Feasibility | Impact |
|---|-----------|-------------|-------------|--------|
| 59 | `turbulence_spectrum_dryden` | Dryden turbulence model for flight simulation | High | High |
| 60 | `turbulence_spectrum_von_karman` | Von Kármán turbulence spectrum model | High | High |
| 61 | `jet_stream_model` | Simplified jet stream location and strength model | Medium | Medium |
| 62 | `icing_conditions_check` | Check for potential icing based on conditions | High | High |
| 63 | `density_altitude_calculator` | Calculate density altitude from pressure/temp | High | Very High |
| 64 | `true_airspeed_converter` | Convert between IAS, CAS, EAS, TAS, Mach | High | Very High |
| 65 | `thermosphere_model_satellite` | Upper atmosphere model for satellite drag | Medium | High |
| 66 | `humidity_psychrometric` | Psychrometric calculations for engine performance | Medium | Low |

### Category: Structures & Materials - 10 Ideas

| # | Tool Name | Description | Feasibility | Impact |
|---|-----------|-------------|-------------|--------|
| 67 | `beam_bending_stress` | Simple beam bending stress analysis | High | High |
| 68 | `buckling_analysis_panel` | Thin panel buckling critical load | High | High |
| 69 | `fatigue_life_estimation` | S-N curve fatigue life calculation | Medium | High |
| 70 | `thermal_expansion_analysis` | Thermal stress from temperature gradients | High | Medium |
| 71 | `composite_laminate_strength` | Classical laminate theory analysis | Medium | High |
| 72 | `material_selection_aerospace` | Material property comparison for aerospace | High | High |
| 73 | `weight_estimation_raymer` | Statistical weight estimation methods | High | Very High |
| 74 | `flutter_speed_estimation` | Critical flutter speed estimation | Low | Medium |
| 75 | `load_factor_envelope` | V-n diagram construction | High | High |
| 76 | `gust_load_calculation` | Gust load factors per FAR 25 | Medium | High |

### Category: Systems Engineering - 8 Ideas

| # | Tool Name | Description | Feasibility | Impact |
|---|-----------|-------------|-------------|--------|
| 77 | `power_budget_analysis` | Electrical power budget for avionics/payload | High | High |
| 78 | `thermal_budget_satellite` | Spacecraft thermal balance analysis | Medium | High |
| 79 | `link_budget_calculator` | RF communication link budget | High | Very High |
| 80 | `reliability_mtbf_analysis` | Mean time between failure calculation | High | Medium |
| 81 | `failure_mode_checklist` | Common failure mode identification | High | Medium |
| 82 | `mass_margin_tracking` | Mass growth tracking and margin analysis | High | High |
| 83 | `data_rate_calculation` | Payload data rate and storage requirements | High | Medium |
| 84 | `solar_array_sizing` | Size solar arrays for spacecraft power | High | High |

### Category: Environment & Safety - 8 Ideas

| # | Tool Name | Description | Feasibility | Impact |
|---|-----------|-------------|-------------|--------|
| 85 | `noise_footprint_estimation` | Ground-level noise contour estimation | Medium | High |
| 86 | `sonic_boom_calculator` | Supersonic boom overpressure estimation | Low | Medium |
| 87 | `emissions_index_calculator` | CO2, NOx emissions estimation | High | High |
| 88 | `bird_strike_energy` | Bird strike impact energy calculation | High | Medium |
| 89 | `debris_collision_probability` | Space debris collision risk assessment | Medium | High |
| 90 | `reentry_heating_estimation` | Ballistic reentry peak heating estimation | Medium | High |
| 91 | `radiation_dose_estimate` | High-altitude/space radiation exposure | Medium | Medium |
| 92 | `contrail_formation_check` | Check conditions for contrail formation | High | Low |

### Category: Simulation & Analysis - 8 Ideas

| # | Tool Name | Description | Feasibility | Impact |
|---|-----------|-------------|-------------|--------|
| 93 | `6dof_state_propagation` | Full 6-DOF rigid body dynamics propagation | Medium | Very High |
| 94 | `linearized_aircraft_model` | Generate linearized state-space aircraft model | Medium | High |
| 95 | `modal_analysis_simple` | Simple structural modal analysis | Low | Medium |
| 96 | `cfd_mesh_sizing` | CFD mesh sizing recommendations | High | Medium |
| 97 | `wind_tunnel_scaling` | Reynolds/Mach scaling for wind tunnel tests | High | Medium |
| 98 | `flight_test_card_generator` | Generate flight test points and sequence | Medium | Medium |
| 99 | `measurement_uncertainty` | Propagate measurement uncertainties | High | High |
| 100 | `unit_conversion_aerospace` | Aerospace-specific unit conversions | High | High |

---

## Part 2: Stack Ranking by Overall Impact Score

**Scoring Methodology:**
- **Feasibility (0-10):** Based on existing integrations, libraries available, complexity
- **Usefulness (0-10):** How frequently users would need this tool
- **SEO/Visibility (0-10):** Search potential, unique value proposition, keyword relevance
- **Gap Fill (0-10):** How much this addresses a missing capability

**Total Score = Feasibility × 0.25 + Usefulness × 0.35 + SEO × 0.20 + Gap Fill × 0.20**

### Top 25 Tools by Impact Score

| Rank | Tool Name | Feas. | Use | SEO | Gap | Total |
|------|-----------|-------|-----|-----|-----|-------|
| 1 | `lambert_problem_solver` | 9 | 9 | 9 | 10 | 9.20 |
| 2 | `density_altitude_calculator` | 10 | 10 | 9 | 7 | 9.15 |
| 3 | `true_airspeed_converter` | 10 | 10 | 9 | 7 | 9.15 |
| 4 | `weight_and_balance` | 9 | 10 | 9 | 8 | 9.15 |
| 5 | `kalman_filter_state_estimation` | 9 | 9 | 8 | 10 | 9.00 |
| 6 | `stall_speed_calculator` | 10 | 9 | 9 | 7 | 8.90 |
| 7 | `takeoff_performance` | 9 | 10 | 8 | 7 | 8.85 |
| 8 | `landing_performance` | 9 | 10 | 8 | 7 | 8.85 |
| 9 | `fuel_reserve_calculator` | 9 | 10 | 7 | 8 | 8.80 |
| 10 | `lqr_controller_design` | 9 | 8 | 8 | 10 | 8.70 |
| 11 | `control_surface_effectiveness` | 8 | 9 | 8 | 9 | 8.65 |
| 12 | `full_aircraft_drag_buildup` | 8 | 9 | 9 | 8 | 8.60 |
| 13 | `link_budget_calculator` | 9 | 8 | 9 | 8 | 8.50 |
| 14 | `satellite_visibility_pass` | 9 | 8 | 9 | 8 | 8.50 |
| 15 | `electric_motor_sizing` | 9 | 9 | 9 | 6 | 8.45 |
| 16 | `multi_leg_flight_plan` | 7 | 10 | 8 | 8 | 8.40 |
| 17 | `gravity_assist_analysis` | 7 | 8 | 10 | 8 | 8.15 |
| 18 | `weight_estimation_raymer` | 9 | 8 | 8 | 7 | 8.10 |
| 19 | `turbulence_spectrum_dryden` | 9 | 8 | 7 | 8 | 8.05 |
| 20 | `propeller_selection_tool` | 9 | 8 | 8 | 6 | 7.95 |
| 21 | `cruise_altitude_optimizer` | 8 | 9 | 7 | 7 | 7.95 |
| 22 | `unit_conversion_aerospace` | 10 | 8 | 7 | 6 | 7.85 |
| 23 | `specific_impulse_calculator` | 10 | 7 | 9 | 6 | 7.75 |
| 24 | `icing_conditions_check` | 9 | 8 | 7 | 7 | 7.75 |
| 25 | `bielliptic_transfer` | 9 | 7 | 8 | 8 | 7.70 |

---

## Part 3: Top 10 Tools for Implementation

Based on the stack ranking, here are the top 10 tools recommended for implementation:

### 1. `lambert_problem_solver`
**Score: 9.20 | Category: Orbital Mechanics**

Solves the Lambert orbital boundary value problem - given two position vectors and time-of-flight, determine the orbit connecting them. This is foundational for interplanetary mission design and currently exists as a stub raising `NotImplementedError`.

**Parameters:**
- `r1_m`: Initial position vector [x, y, z] in meters
- `r2_m`: Final position vector [x, y, z] in meters
- `tof_s`: Time of flight in seconds
- `direction`: "prograde" or "retrograde"
- `num_revolutions`: Number of complete revolutions (default 0)

**Returns:** Transfer orbit velocity vectors, orbital elements, delta-v requirements

---

### 2. `density_altitude_calculator`
**Score: 9.15 | Category: Atmospheric Science**

Calculates density altitude from pressure altitude, temperature, and humidity. Essential for performance calculations, especially at high-altitude airports.

**Parameters:**
- `pressure_altitude_ft`: Pressure altitude in feet
- `temperature_c`: Outside air temperature in Celsius
- `dewpoint_c`: Optional dewpoint for humidity correction

**Returns:** Density altitude in feet, air density, pressure ratio

---

### 3. `true_airspeed_converter`
**Score: 9.15 | Category: Atmospheric Science**

Converts between indicated airspeed (IAS), calibrated airspeed (CAS), equivalent airspeed (EAS), true airspeed (TAS), and Mach number.

**Parameters:**
- `speed_value`: Input speed value
- `speed_type`: Input type ("IAS", "CAS", "EAS", "TAS", "MACH")
- `altitude_ft`: Pressure altitude in feet
- `temperature_c`: Outside air temperature in Celsius (optional, uses ISA if not provided)
- `position_error_kts`: Position error correction in knots (optional)

**Returns:** All five speed representations plus dynamic pressure

---

### 4. `weight_and_balance`
**Score: 9.15 | Category: Mission Planning**

Calculates aircraft center of gravity position and verifies it's within limits for given loading.

**Parameters:**
- `aircraft_type`: Aircraft type code
- `basic_empty_weight_kg`: Basic empty weight
- `basic_empty_arm_m`: Basic empty weight CG arm
- `fuel_kg`: Fuel load in kg
- `fuel_arm_m`: Fuel tank CG arm
- `payload_items`: List of {weight_kg, arm_m, name} for passengers/cargo

**Returns:** Total weight, CG position, CG as % MAC, forward/aft limit check, moment

---

### 5. `kalman_filter_state_estimation`
**Score: 9.00 | Category: GNC**

Implements Extended Kalman Filter for aircraft/spacecraft state estimation from noisy sensor measurements. Leverages existing filterpy integration.

**Parameters:**
- `initial_state`: Initial state vector estimate
- `initial_covariance`: Initial state covariance matrix
- `process_noise`: Process noise covariance (Q matrix)
- `measurement_noise`: Measurement noise covariance (R matrix)
- `measurements`: Time-series of sensor measurements
- `dynamics_model`: "constant_velocity", "constant_acceleration", or "orbital"

**Returns:** Filtered state estimates, covariances, innovation sequences

---

### 6. `stall_speed_calculator`
**Score: 8.90 | Category: Aerodynamics**

Calculates stall speeds for different aircraft configurations (clean, takeoff, landing) and load factors.

**Parameters:**
- `weight_kg`: Aircraft weight
- `wing_area_m2`: Wing reference area
- `cl_max_clean`: Maximum lift coefficient (clean configuration)
- `cl_max_takeoff`: Optional CL_max with takeoff flaps
- `cl_max_landing`: Optional CL_max with landing flaps
- `altitude_ft`: Pressure altitude for density
- `load_factor`: Load factor (default 1.0)

**Returns:** Stall speeds (VS0, VS1, etc.) in knots/m/s, stall Mach number

---

### 7. `takeoff_performance`
**Score: 8.85 | Category: Mission Planning**

Calculates takeoff field length, V-speeds, and climb gradient for given conditions.

**Parameters:**
- `aircraft_type`: Aircraft type or custom performance data
- `weight_kg`: Takeoff weight
- `pressure_altitude_ft`: Airport pressure altitude
- `temperature_c`: Outside air temperature
- `wind_kts`: Headwind (+) or tailwind (-) in knots
- `runway_slope_pct`: Runway slope in percent
- `runway_condition`: "dry", "wet", "contaminated"

**Returns:** V1, VR, V2 speeds, ground roll distance, total takeoff distance, climb gradient

---

### 8. `landing_performance`
**Score: 8.85 | Category: Mission Planning**

Calculates landing distance for given conditions including approach speed and runway factors.

**Parameters:**
- `aircraft_type`: Aircraft type or custom performance data
- `weight_kg`: Landing weight
- `pressure_altitude_ft`: Airport pressure altitude
- `temperature_c`: Outside air temperature
- `wind_kts`: Headwind (+) or tailwind (-) in knots
- `runway_slope_pct`: Runway slope in percent
- `runway_condition`: "dry", "wet", "contaminated"
- `approach_speed_kts`: Reference approach speed (Vref)

**Returns:** Air distance, ground roll, total landing distance, VREF, threshold speed

---

### 9. `fuel_reserve_calculator`
**Score: 8.80 | Category: Mission Planning**

Calculates required fuel reserves per aviation regulations (FAR/JAR).

**Parameters:**
- `regulation`: "FAR_121", "FAR_91", "JAR_OPS", "ICAO"
- `trip_fuel_kg`: Planned trip fuel
- `alternate_fuel_kg`: Fuel to fly to alternate (if required)
- `flight_time_min`: Planned flight time in minutes
- `cruise_fuel_flow_kg_hr`: Cruise fuel flow rate
- `holding_altitude_ft`: Expected holding altitude

**Returns:** Contingency fuel, alternate fuel, final reserve, total required fuel, breakdown

---

### 10. `lqr_controller_design`
**Score: 8.70 | Category: GNC**

Designs Linear Quadratic Regulator (LQR) optimal controller given state-space system matrices. Leverages existing control library integration.

**Parameters:**
- `A_matrix`: State matrix (n x n)
- `B_matrix`: Input matrix (n x m)
- `Q_matrix`: State weighting matrix (n x n)
- `R_matrix`: Input weighting matrix (m x m)
- `state_names`: Optional names for states
- `input_names`: Optional names for control inputs

**Returns:** Optimal gain matrix K, closed-loop eigenvalues, controllability check, cost function value

---

## Part 4: Implementation Guide for Claude Code

This section provides step-by-step instructions for implementing the top 10 tools.

### Prerequisites

Before implementing any tool:

1. **Read the existing codebase patterns:**
   - `aerospace_mcp/tools/orbits.py` - Example of well-structured orbital tools
   - `aerospace_mcp/tools/atmosphere.py` - Example of simple calculation tools
   - `aerospace_mcp/integrations/gnc.py` - Existing GNC integration code

2. **Understand the tool registration pattern:**
   ```python
   # In aerospace_mcp/fastmcp_server.py
   mcp.tool(your_new_tool)
   ```

3. **Follow the return value pattern:**
   - Return formatted strings for human readability
   - Include JSON-parseable data for programmatic use
   - Always handle errors gracefully with descriptive messages

---

### Implementation Steps for Each Tool

#### Tool 1: `lambert_problem_solver`

**File locations:**
- Create: `aerospace_mcp/tools/orbits.py` (add to existing file)
- Modify: `aerospace_mcp/integrations/orbits.py` (add core function)
- Modify: `aerospace_mcp/fastmcp_server.py` (register tool)
- Update: `aerospace_mcp/tools/tool_search.py` (add to registry)

**Step 1: Implement the integration function**
```python
# In aerospace_mcp/integrations/orbits.py

def solve_lambert(
    r1: np.ndarray,
    r2: np.ndarray,
    tof: float,
    mu: float = 3.986004418e14,  # Earth GM
    direction: str = "prograde",
    num_revs: int = 0
) -> dict:
    """
    Solve Lambert's problem using iterative algorithm.

    Uses Izzo's algorithm or Gooding's method for robust convergence.
    """
    # Implementation using poliastro if available, else manual algorithm
    try:
        from poliastro.iod import izzo
        v1, v2 = izzo.lambert(
            k=mu * u.m**3 / u.s**2,
            r0=r1 * u.m,
            r=r2 * u.m,
            tof=tof * u.s,
            M=num_revs,
            prograde=direction == "prograde"
        )
        return {
            "v1_ms": v1.to(u.m/u.s).value.tolist(),
            "v2_ms": v2.to(u.m/u.s).value.tolist(),
            "success": True
        }
    except ImportError:
        # Fallback to manual implementation
        return _lambert_manual(r1, r2, tof, mu, direction, num_revs)
```

**Step 2: Create the tool wrapper**
```python
# In aerospace_mcp/tools/orbits.py

def lambert_problem_solver(
    r1_m: list[float],
    r2_m: list[float],
    tof_s: float,
    direction: str = "prograde",
    num_revolutions: int = 0,
    central_body: str = "earth"
) -> str:
    """Solve Lambert's orbital boundary value problem."""
    try:
        from ..integrations.orbits import solve_lambert, MU_BODIES

        mu = MU_BODIES.get(central_body.lower(), MU_BODIES["earth"])
        result = solve_lambert(
            np.array(r1_m),
            np.array(r2_m),
            tof_s,
            mu,
            direction,
            num_revolutions
        )

        # Calculate delta-v if departure/arrival velocities known
        # Format output
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error solving Lambert problem: {str(e)}"
```

**Step 3: Register in fastmcp_server.py**
```python
from .tools.orbits import lambert_problem_solver
mcp.tool(lambert_problem_solver)
```

**Step 4: Add to tool_search.py registry**
```python
ToolMetadata(
    name="lambert_problem_solver",
    description="Solve Lambert's orbital boundary value problem for transfer trajectories",
    category="orbits",
    parameters={
        "r1_m": "Initial position vector [x, y, z] in meters",
        "r2_m": "Final position vector [x, y, z] in meters",
        "tof_s": "Time of flight in seconds",
        "direction": "'prograde' or 'retrograde'",
        "num_revolutions": "Number of complete revolutions (default 0)",
    },
    keywords=["lambert", "transfer", "trajectory", "orbit", "interplanetary", "rendezvous"],
),
```

**Step 5: Write tests**
```python
# In tests/tools/test_orbits.py

def test_lambert_problem_solver_basic():
    """Test basic Lambert problem - LEO to GEO transfer."""
    r1 = [7000e3, 0, 0]  # 7000 km circular LEO
    r2 = [42164e3, 0, 0]  # GEO
    tof = 5 * 3600  # 5 hour transfer

    result = json.loads(lambert_problem_solver(r1, r2, tof))

    assert result["success"]
    assert "v1_ms" in result
    assert "v2_ms" in result
```

---

#### Tool 2: `density_altitude_calculator`

**File locations:**
- Modify: `aerospace_mcp/tools/atmosphere.py`
- Modify: `aerospace_mcp/integrations/atmosphere.py`
- Update: `aerospace_mcp/fastmcp_server.py`
- Update: `aerospace_mcp/tools/tool_search.py`

**Step 1: Implement in integrations/atmosphere.py**
```python
def calculate_density_altitude(
    pressure_alt_ft: float,
    temperature_c: float,
    dewpoint_c: float | None = None
) -> dict:
    """Calculate density altitude from pressure altitude and temperature."""
    # Convert to SI
    pressure_alt_m = pressure_alt_ft * 0.3048
    temp_k = temperature_c + 273.15

    # ISA temperature at pressure altitude
    isa_temp_k = 288.15 - 0.0065 * pressure_alt_m

    # Temperature deviation from ISA
    delta_isa = temp_k - isa_temp_k

    # Density altitude approximation
    # Each 1°C above ISA ≈ 120 ft increase in density altitude
    density_alt_ft = pressure_alt_ft + (delta_isa * 118.8)

    # Calculate actual density
    p = 101325 * (1 - 0.0065 * pressure_alt_m / 288.15) ** 5.2561
    rho = p / (287.05 * temp_k)
    rho_sl = 1.225  # kg/m³ at sea level ISA

    return {
        "density_altitude_ft": round(density_alt_ft, 0),
        "density_altitude_m": round(density_alt_ft * 0.3048, 0),
        "air_density_kg_m3": round(rho, 4),
        "density_ratio_sigma": round(rho / rho_sl, 4),
        "isa_deviation_c": round(delta_isa, 1),
        "pressure_ratio_delta": round(p / 101325, 4),
    }
```

**Step 2: Create tool wrapper**
```python
# In aerospace_mcp/tools/atmosphere.py

def density_altitude_calculator(
    pressure_altitude_ft: float,
    temperature_c: float,
    dewpoint_c: float | None = None
) -> str:
    """
    Calculate density altitude from pressure altitude and temperature.

    Density altitude is the altitude in the standard atmosphere at which
    the air density equals the actual air density at the given conditions.
    Essential for aircraft performance calculations.
    """
    from ..integrations.atmosphere import calculate_density_altitude

    result = calculate_density_altitude(
        pressure_altitude_ft,
        temperature_c,
        dewpoint_c
    )

    output = f"""
DENSITY ALTITUDE CALCULATION
============================
Input Conditions:
  Pressure Altitude: {pressure_altitude_ft:,.0f} ft
  Temperature: {temperature_c:.1f}°C
  ISA Deviation: {result['isa_deviation_c']:+.1f}°C

Results:
  Density Altitude: {result['density_altitude_ft']:,.0f} ft ({result['density_altitude_m']:,.0f} m)
  Air Density: {result['air_density_kg_m3']:.4f} kg/m³
  Density Ratio (σ): {result['density_ratio_sigma']:.4f}
  Pressure Ratio (δ): {result['pressure_ratio_delta']:.4f}

{json.dumps(result, indent=2)}
"""
    return output.strip()
```

---

#### Tool 3: `true_airspeed_converter`

**File location:** `aerospace_mcp/tools/atmosphere.py`

**Implementation:**
```python
def true_airspeed_converter(
    speed_value: float,
    speed_type: str,
    altitude_ft: float,
    temperature_c: float | None = None,
    position_error_kts: float = 0
) -> str:
    """
    Convert between IAS, CAS, EAS, TAS, and Mach number.

    Parameters:
        speed_value: Input speed value
        speed_type: One of "IAS", "CAS", "EAS", "TAS", "MACH"
        altitude_ft: Pressure altitude in feet
        temperature_c: OAT (if None, uses ISA)
        position_error_kts: Position error correction
    """
    # Get atmospheric conditions
    from ..integrations.atmosphere import get_isa_conditions

    conditions = get_isa_conditions(altitude_ft * 0.3048)
    p = conditions["pressure_pa"]
    rho = conditions["density_kg_m3"]
    a = conditions["speed_of_sound_ms"]

    p0 = 101325  # Sea level pressure
    rho0 = 1.225  # Sea level density
    a0 = 340.29  # Sea level speed of sound

    # Conversion logic based on input type
    # ... (full implementation with compressibility corrections)

    return json.dumps({
        "IAS_kts": ias,
        "CAS_kts": cas,
        "EAS_kts": eas,
        "TAS_kts": tas,
        "TAS_ms": tas * 0.5144,
        "MACH": mach,
        "dynamic_pressure_pa": q,
        "altitude_ft": altitude_ft,
        "temperature_c": temp_c,
    }, indent=2)
```

---

#### Tool 4: `weight_and_balance`

**File location:** Create `aerospace_mcp/tools/performance.py`

**Implementation approach:**
1. Create aircraft database with empty weight, CG, and envelope limits
2. Implement moment calculation for loading items
3. Check CG against forward/aft limits
4. Return W&B summary with visual CG envelope position

---

#### Tool 5: `kalman_filter_state_estimation`

**File location:** Create `aerospace_mcp/tools/gnc.py`

**Implementation:**
```python
def kalman_filter_state_estimation(
    initial_state: list[float],
    initial_covariance: list[list[float]],
    process_noise: list[list[float]],
    measurement_noise: list[list[float]],
    measurements: list[dict],
    dynamics_model: str = "constant_velocity"
) -> str:
    """
    Extended Kalman Filter for state estimation.

    Uses filterpy library when available, falls back to manual implementation.
    """
    try:
        from filterpy.kalman import ExtendedKalmanFilter
        # Implementation using filterpy
    except ImportError:
        # Manual EKF implementation
        pass
```

---

#### Tool 6: `stall_speed_calculator`

**File location:** `aerospace_mcp/tools/aerodynamics.py`

**Implementation:**
```python
def stall_speed_calculator(
    weight_kg: float,
    wing_area_m2: float,
    cl_max_clean: float,
    cl_max_takeoff: float | None = None,
    cl_max_landing: float | None = None,
    altitude_ft: float = 0,
    load_factor: float = 1.0
) -> str:
    """Calculate stall speeds for different configurations."""
    from ..integrations.atmosphere import get_isa_conditions

    rho = get_isa_conditions(altitude_ft * 0.3048)["density_kg_m3"]
    weight_n = weight_kg * 9.81

    def calc_vs(cl_max):
        return math.sqrt((2 * weight_n * load_factor) / (rho * wing_area_m2 * cl_max))

    vs1 = calc_vs(cl_max_clean)  # Clean stall speed
    vs0 = calc_vs(cl_max_landing or cl_max_clean * 1.3)  # Landing config

    return json.dumps({
        "VS1_clean_ms": round(vs1, 2),
        "VS1_clean_kts": round(vs1 * 1.944, 1),
        "VS0_landing_ms": round(vs0, 2),
        "VS0_landing_kts": round(vs0 * 1.944, 1),
        # ... more configurations
    }, indent=2)
```

---

#### Tools 7 & 8: `takeoff_performance` and `landing_performance`

**File location:** Create `aerospace_mcp/tools/performance.py`

These are complex tools requiring:
1. Aircraft performance database or user-provided data
2. V-speed calculation algorithms
3. Ground roll physics (friction, thrust, drag)
4. Regulatory factors (1.15 for wet, 1.25 for contaminated)

**Implementation skeleton:**
```python
def takeoff_performance(
    aircraft_type: str | None = None,
    weight_kg: float = None,
    pressure_altitude_ft: float = 0,
    temperature_c: float = 15,
    wind_kts: float = 0,
    runway_slope_pct: float = 0,
    runway_condition: str = "dry",
    custom_data: dict | None = None
) -> str:
    """
    Calculate takeoff field length and V-speeds.

    Uses OpenAP aircraft data when available, or custom_data.
    """
    # 1. Get aircraft performance data
    # 2. Calculate density altitude
    # 3. Calculate V-speeds (V1, VR, V2)
    # 4. Calculate ground roll distance
    # 5. Calculate air distance to 35ft
    # 6. Apply factors for conditions
    # 7. Calculate climb gradient
```

---

#### Tool 9: `fuel_reserve_calculator`

**File location:** `aerospace_mcp/tools/performance.py`

**Implementation:**
```python
def fuel_reserve_calculator(
    regulation: str,
    trip_fuel_kg: float,
    alternate_fuel_kg: float = 0,
    flight_time_min: float = 0,
    cruise_fuel_flow_kg_hr: float = 0,
    holding_altitude_ft: float = 1500
) -> str:
    """
    Calculate required fuel reserves per aviation regulations.

    Regulations supported: FAR_121, FAR_91, JAR_OPS, ICAO
    """
    reserves = {}

    if regulation == "FAR_121":
        # 10% of trip fuel for contingency
        reserves["contingency_kg"] = trip_fuel_kg * 0.10
        # Fuel to alternate
        reserves["alternate_kg"] = alternate_fuel_kg
        # 45 minutes at normal cruise
        reserves["final_reserve_kg"] = (cruise_fuel_flow_kg_hr / 60) * 45
    elif regulation == "FAR_91":
        # Day VFR: 30 min at cruise
        # Night VFR / IFR: 45 min at cruise
        reserves["final_reserve_kg"] = (cruise_fuel_flow_kg_hr / 60) * 45
    # ... other regulations

    total = trip_fuel_kg + sum(reserves.values())

    return json.dumps({
        "regulation": regulation,
        "trip_fuel_kg": trip_fuel_kg,
        **reserves,
        "total_required_kg": round(total, 1),
    }, indent=2)
```

---

#### Tool 10: `lqr_controller_design`

**File location:** Create `aerospace_mcp/tools/gnc.py`

**Implementation:**
```python
def lqr_controller_design(
    A_matrix: list[list[float]],
    B_matrix: list[list[float]],
    Q_matrix: list[list[float]],
    R_matrix: list[list[float]],
    state_names: list[str] | None = None,
    input_names: list[str] | None = None
) -> str:
    """
    Design Linear Quadratic Regulator (LQR) optimal controller.

    Computes optimal state-feedback gain K that minimizes
    J = integral(x'Qx + u'Ru) dt
    """
    try:
        import control
        import numpy as np

        A = np.array(A_matrix)
        B = np.array(B_matrix)
        Q = np.array(Q_matrix)
        R = np.array(R_matrix)

        # Check controllability
        ctrb = control.ctrb(A, B)
        rank = np.linalg.matrix_rank(ctrb)
        n = A.shape[0]
        controllable = rank == n

        if not controllable:
            return f"Error: System is not controllable (rank {rank} < {n})"

        # Solve LQR
        K, S, E = control.lqr(A, B, Q, R)

        return json.dumps({
            "gain_matrix_K": K.tolist(),
            "cost_matrix_S": S.tolist(),
            "closed_loop_eigenvalues": [complex(e).real for e in E],
            "controllable": controllable,
            "controllability_rank": rank,
        }, indent=2)

    except ImportError:
        return "Error: control library not available. Install with: pip install control"
```

---

### General Implementation Checklist

For each tool, ensure you complete these steps:

- [ ] **Step 1:** Implement core logic in `aerospace_mcp/integrations/<domain>.py`
- [ ] **Step 2:** Create tool wrapper in `aerospace_mcp/tools/<domain>.py`
- [ ] **Step 3:** Register tool in `aerospace_mcp/fastmcp_server.py`
- [ ] **Step 4:** Add metadata to `aerospace_mcp/tools/tool_search.py`
- [ ] **Step 5:** Write unit tests in `tests/tools/test_<domain>.py`
- [ ] **Step 6:** Update docstrings with clear parameter descriptions
- [ ] **Step 7:** Test with MCP inspector or Claude Code
- [ ] **Step 8:** Update CLAUDE.md if new patterns introduced

### Testing Strategy

1. **Unit tests** for pure calculation functions
2. **Integration tests** for tool wrappers with mocked dependencies
3. **End-to-end tests** with actual MCP calls
4. **Edge case tests** for error handling (missing deps, invalid inputs)

### Documentation Updates

After implementing tools, update:
- `README.md` - Add new tools to feature list
- `CLAUDE.md` - Update tool count and categories
- Tool docstrings - Include usage examples

---

## Summary

This document provides:
1. **100 tool ideas** organized by aerospace domain
2. **Stack ranking** based on feasibility, usefulness, SEO, and gap-filling value
3. **Top 10 tools** identified for implementation priority (**ALL IMPLEMENTED**)
4. **Step-by-step implementation guide** for Claude Code

The top 10 tools have been implemented and are available in the aerospace-mcp package:
- **Performance tools** (`tools/performance.py`): density_altitude_calculator, true_airspeed_converter, stall_speed_calculator, weight_and_balance, takeoff_performance, landing_performance, fuel_reserve_calculator
- **GNC tools** (`tools/gnc.py`): kalman_filter_state_estimation, lqr_controller_design
- **Orbital mechanics** (`tools/orbits.py`): lambert_problem_solver

The remaining 90 tool ideas are available for future implementation, with the next highest-ranked tools being:
- `control_surface_effectiveness` (8.65)
- `full_aircraft_drag_buildup` (8.60)
- `link_budget_calculator` (8.50)
- `satellite_visibility_pass` (8.50)
