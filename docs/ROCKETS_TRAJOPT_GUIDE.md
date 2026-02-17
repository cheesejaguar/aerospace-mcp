# Rocket Trajectory & Optimization Guide

This guide covers Phase 3 of the aerospace-mcp project: rocket trajectory analysis and trajectory optimization tools.

## Overview

The Phase 3 implementation adds comprehensive rocket trajectory modeling and optimization capabilities to the aerospace MCP server. These tools enable:

- 3DOF rocket trajectory simulation with atmosphere integration
- Rocket sizing estimation for mission planning
- Launch angle optimization for maximum performance
- Thrust profile optimization using gradient descent
- Sensitivity analysis for design parameter studies

## Available MCP Tools

### Rocket Trajectory Tools

#### `rocket_3dof_trajectory`
Simulates 3-degree-of-freedom rocket trajectories using numerical integration.

**Parameters:**
- `geometry`: Rocket geometry and mass properties
  - `dry_mass_kg`: Rocket dry mass (0.1-100000 kg)
  - `propellant_mass_kg`: Initial propellant mass (0.1-500000 kg)
  - `diameter_m`: Rocket diameter (0.01-10 m)
  - `length_m`: Total rocket length (0.1-100 m)
  - `cd`: Drag coefficient (0.1-2.0, default 0.3)
  - `thrust_curve`: Array of [time_s, thrust_N] points (optional)
- `dt_s`: Integration time step (0.01-1.0 s, default 0.1 s)
- `max_time_s`: Maximum simulation time (10-1000 s, default 300 s)
- `launch_angle_deg`: Launch angle from horizontal (45-90°, default 90°)

**Returns:** Trajectory analysis with performance metrics including max altitude, velocity, Mach number, and dynamic pressure.

#### `estimate_rocket_sizing`
Estimates rocket sizing requirements for target altitude and payload using the rocket equation.

**Parameters:**
- `target_altitude_m`: Target apogee altitude (100-100000 m)
- `payload_mass_kg`: Payload mass (0.01-10000 kg)
- `propellant_type`: "solid" or "liquid" (default "solid")

**Returns:** Sizing estimates including total mass, propellant mass, structure mass, geometry, and performance parameters.

### Trajectory Optimization Tools

#### `optimize_launch_angle`
Optimizes rocket launch angle for maximum altitude or range using golden section search.

**Parameters:**
- `geometry`: Rocket geometry (same as trajectory tool)
- `objective`: "max_altitude" or "max_range" (default "max_altitude")
- `angle_bounds`: [min_deg, max_deg] search range (default [80, 90])

**Returns:** Optimization results with optimal launch angle, objective value, and performance data.

#### `optimize_thrust_profile`
Optimizes rocket thrust profile for better performance using multi-segment optimization.

**Parameters:**
- `geometry`: Base rocket geometry (thrust curve will be optimized)
- `burn_time_s`: Total burn time (1-300 s)
- `total_impulse_target`: Target total impulse (100-10000000 N·s)
- `n_segments`: Number of thrust segments (3-10, default 5)
- `objective`: "max_altitude", "min_max_q", or "min_gravity_loss"

**Returns:** Optimized thrust profile with segment multipliers and performance analysis.

#### `trajectory_sensitivity_analysis`
Performs sensitivity analysis on rocket trajectory parameters to identify critical design variables.

**Parameters:**
- `base_geometry`: Baseline rocket geometry
- `parameter_variations`: Dictionary of parameter names and variation ranges
- `objective`: "max_altitude", "max_velocity", or "specific_impulse"

**Returns:** Sensitivity analysis with parameter rankings and sensitivity coefficients.

## Technical Implementation

### Trajectory Simulation

The 3DOF trajectory simulation uses Euler integration with the following physics:

**Forces:**
- Thrust: From user-defined thrust curve with linear interpolation
- Weight: mg with varying mass due to propellant consumption
- Drag: ½ρv²SCD with altitude-dependent atmospheric properties

**Mass Model:**
- Constant mass flow rate during burn phase
- Mass decreases from initial to dry mass over burn time

**Atmosphere Integration:**
- Uses ISA atmosphere model from Phase 1 tools
- Density and speed of sound vary with altitude
- Fallback exponential model for extreme altitudes

### Optimization Algorithms

#### Golden Section Search (1D)
- Used for launch angle optimization
- Convergence tolerance: 0.1°
- Maximum iterations: 50
- Robust for single-parameter optimization

#### Gradient Descent (Multi-dimensional)
- Used for thrust profile optimization
- Numerical gradient estimation with central differences
- Parameter bounds enforcement
- Learning rate: 0.05, tolerance: 0.001

### Rocket Sizing Algorithm

Uses the rocket equation with simplified loss models:

1. **Delta-V Estimation:**
   - Potential energy: mgh_target
   - Gravity losses: ~1.5x theoretical
   - Drag losses: ~0.3x theoretical
   - Total factor: 1.8x theoretical

2. **Mass Breakdown:**
   - Structure mass = structural_ratio × propellant_mass
   - Mass ratio from rocket equation
   - Geometry from propellant volume and L/D assumptions

## Usage Examples

### Basic Trajectory Simulation

```json
{
  "tool": "rocket_3dof_trajectory",
  "arguments": {
    "geometry": {
      "dry_mass_kg": 25.0,
      "propellant_mass_kg": 75.0,
      "diameter_m": 0.2,
      "length_m": 2.0,
      "cd": 0.4,
      "thrust_curve": [[0.0, 2000.0], [8.0, 2000.0], [8.1, 0.0]]
    },
    "dt_s": 0.1,
    "max_time_s": 200.0,
    "launch_angle_deg": 90.0
  }
}
```

### Mission Sizing

```json
{
  "tool": "estimate_rocket_sizing",
  "arguments": {
    "target_altitude_m": 10000.0,
    "payload_mass_kg": 5.0,
    "propellant_type": "solid"
  }
}
```

### Launch Angle Optimization

```json
{
  "tool": "optimize_launch_angle",
  "arguments": {
    "geometry": {
      "dry_mass_kg": 20.0,
      "propellant_mass_kg": 80.0,
      "diameter_m": 0.18,
      "length_m": 1.8,
      "thrust_curve": [[0.0, 1800.0], [6.0, 1800.0], [6.1, 0.0]]
    },
    "objective": "max_altitude",
    "angle_bounds": [85.0, 90.0]
  }
}
```

### Thrust Profile Optimization

```json
{
  "tool": "optimize_thrust_profile",
  "arguments": {
    "geometry": {
      "dry_mass_kg": 30.0,
      "propellant_mass_kg": 120.0,
      "diameter_m": 0.25,
      "length_m": 2.5,
      "cd": 0.35
    },
    "burn_time_s": 10.0,
    "total_impulse_target": 20000.0,
    "n_segments": 5,
    "objective": "max_altitude"
  }
}
```

### Sensitivity Analysis

```json
{
  "tool": "trajectory_sensitivity_analysis",
  "arguments": {
    "base_geometry": {
      "dry_mass_kg": 25.0,
      "propellant_mass_kg": 75.0,
      "diameter_m": 0.2,
      "length_m": 2.0,
      "thrust_curve": [[0.0, 2000.0], [8.0, 2000.0], [8.1, 0.0]]
    },
    "parameter_variations": {
      "dry_mass_kg": [20.0, 25.0, 30.0],
      "propellant_mass_kg": [65.0, 75.0, 85.0],
      "cd": [0.3, 0.4, 0.5]
    },
    "objective": "max_altitude"
  }
}
```

## Performance Characteristics

### Computational Performance
- Trajectory simulation: ~50-200ms per run
- Launch angle optimization: ~2-5 seconds
- Thrust profile optimization: ~10-30 seconds
- Sensitivity analysis: ~1-10 seconds per parameter

### Accuracy Expectations
- Trajectory altitude: ±10-20% vs high-fidelity models
- Optimization convergence: typically within 50-100 iterations
- Sizing estimates: ±30-50% for preliminary design

### Limitations
- 3DOF only (no attitude dynamics)
- Point mass assumption
- Simplified atmosphere/wind models
- No staging capability
- Basic propellant consumption model

## Integration with Other Tools

### Phase 1 Integration (Atmosphere)
- Automatic ISA atmosphere profile lookup
- Density and sound speed for drag/Mach calculations
- Graceful fallback for extreme altitudes

### Phase 2 Integration (Aerodynamics)
- Compatible coordinate systems and units
- Complementary analysis workflows
- Shared atmospheric property calculations

## Validation and Testing

The implementation includes comprehensive test suites covering:

### Rocket Trajectory Tests
- Basic trajectory physics validation
- Mass consumption accuracy
- Thrust curve interpolation
- Performance analysis consistency
- Edge cases and error handling

### Optimization Tests
- Algorithm convergence properties
- Parameter bound enforcement
- Objective function evaluation
- Sensitivity calculation accuracy
- Repeated optimization consistency

### Integration Tests
- Sizing-to-trajectory consistency
- Multi-parameter optimization
- Cross-tool data compatibility
- Error propagation handling

## Future Enhancements

Potential Phase 4 improvements:
- 6DOF attitude dynamics
- Multi-stage trajectory modeling
- Advanced optimization algorithms (genetic algorithms, particle swarm)
- Monte Carlo uncertainty analysis
- Aerodynamic interaction with Phase 2 tools
- Real-time trajectory guidance algorithms

## References

1. Sutton, G.P. & Biblarz, O. "Rocket Propulsion Elements" 8th Edition
2. Cornelisse, J.W. "Rocket Propulsion and Spaceflight Dynamics"
3. Turner, M.J.L. "Rocket and Spacecraft Propulsion" 3rd Edition
4. Betts, J.T. "Practical Methods for Optimal Control and Estimation Using Nonlinear Programming"
5. NASA Launch Vehicle Design Process Guidelines

---

**Note:** This is Phase 3 of the aerospace-mcp implementation. Phase 4 will add space dynamics and advanced trajectory optimization capabilities.
