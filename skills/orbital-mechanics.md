---
name: orbital-mechanics
description: Orbital mechanics calculations including transfers, propagation, and rendezvous planning
---

# Orbital Mechanics

Tools for orbital mechanics calculations: Keplerian element conversions, orbit propagation with J2 perturbations, Hohmann transfers, Lambert problem solving, ground track calculation, and rendezvous planning.

## Available Tools

| Tool | Description |
|------|-------------|
| `hohmann_transfer` | Calculate Hohmann transfer between two circular orbits |
| `lambert_problem_solver` | Solve Lambert's boundary value problem for transfer trajectories |
| `elements_to_state_vector` | Convert classical orbital elements to position/velocity state vector |
| `state_vector_to_elements` | Convert state vector to classical orbital elements |
| `propagate_orbit_j2` | Propagate orbit with J2 Earth oblateness perturbations |
| `calculate_ground_track` | Calculate satellite ground track (lat/lon path) |
| `orbital_rendezvous_planning` | Plan rendezvous maneuvers between spacecraft |

## CLI Examples

```bash
# Hohmann transfer from LEO (400km) to GEO
aerospace-mcp-cli run hohmann_transfer --r1_m 6778000 --r2_m 42164000

# Lambert problem: find transfer orbit between two positions in 1 hour
aerospace-mcp-cli run lambert_problem_solver \
  --r1_m '[7000000,0,0]' \
  --r2_m '[0,7000000,0]' \
  --tof_s 3600

# Convert orbital elements (ISS-like orbit) to state vector
aerospace-mcp-cli run elements_to_state_vector \
  --orbital_elements '{"semi_major_axis_m":6793000,"eccentricity":0.0001,"inclination_deg":51.6,"raan_deg":0,"arg_periapsis_deg":0,"true_anomaly_deg":0}'

# Propagate orbit for 1 orbit period (~90 min)
aerospace-mcp-cli run propagate_orbit_j2 \
  --initial_state '{"semi_major_axis_m":6793000,"eccentricity":0.0001,"inclination_deg":51.6,"raan_deg":0,"arg_periapsis_deg":0,"true_anomaly_deg":0}' \
  --propagation_time_s 5400 \
  --time_step_s 60

# Calculate ground track for 2 hours
aerospace-mcp-cli run calculate_ground_track \
  --orbital_state '{"semi_major_axis_m":6793000,"eccentricity":0.0001,"inclination_deg":51.6,"raan_deg":0,"arg_periapsis_deg":0,"true_anomaly_deg":0}' \
  --duration_s 7200 \
  --time_step_s 30
```

## Programmatic Usage

```python
from aerospace_mcp.tools.orbits import (
    hohmann_transfer,
    lambert_problem_solver,
    elements_to_state_vector,
    propagate_orbit_j2,
)

# Hohmann transfer LEO -> GEO
result = hohmann_transfer(r1_m=6778000, r2_m=42164000)
print(result)

# Lambert problem
result = lambert_problem_solver(
    r1_m=[7000000, 0, 0],
    r2_m=[0, 7000000, 0],
    tof_s=3600,
)
print(result)

# Orbital elements to state vector
result = elements_to_state_vector(orbital_elements={
    "semi_major_axis_m": 6793000,
    "eccentricity": 0.0001,
    "inclination_deg": 51.6,
    "raan_deg": 0,
    "arg_periapsis_deg": 0,
    "true_anomaly_deg": 0,
})
print(result)
```

## Parameter Reference

### hohmann_transfer
- `--r1_m` (float, required): Initial circular orbit radius in meters
- `--r2_m` (float, required): Final circular orbit radius in meters

### lambert_problem_solver
- `--r1_m` (list[float]/JSON, required): Initial position vector `[x, y, z]` in meters
- `--r2_m` (list[float]/JSON, required): Final position vector `[x, y, z]` in meters
- `--tof_s` (float, required): Time of flight in seconds
- `--direction` (one of: prograde, retrograde; default: prograde)
- `--num_revolutions` (int, default: 0)
- `--central_body` (str, default: earth)

### elements_to_state_vector
- `--orbital_elements` (dict/JSON, required): `semi_major_axis_m`, `eccentricity`, `inclination_deg`, `raan_deg`, `arg_periapsis_deg`, `true_anomaly_deg`

### state_vector_to_elements
- `--state_vector` (dict/JSON, required): `position_m` `[x,y,z]`, `velocity_ms` `[vx,vy,vz]`

### propagate_orbit_j2
- `--initial_state` (dict/JSON, required): Orbital elements or state vector
- `--propagation_time_s` (float, required): Propagation duration in seconds
- `--time_step_s` (float, default: 60.0): Integration time step

### calculate_ground_track
- `--orbital_state` (dict/JSON, required): Orbital state
- `--duration_s` (float, required): Ground track duration in seconds
- `--time_step_s` (float, default: 60.0): Time step for track points

### orbital_rendezvous_planning
- `--chaser_elements` (dict/JSON, required): Chaser spacecraft orbital elements
- `--target_elements` (dict/JSON, required): Target spacecraft orbital elements
- `--rendezvous_options` (dict/JSON, optional): Planning options
