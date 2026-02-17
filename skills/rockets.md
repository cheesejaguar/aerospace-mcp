---
name: rockets
description: Simulate rocket trajectories, estimate sizing, and optimize launch angles
---

# Rocket Analysis

Tools for rocket trajectory simulation (3DOF), rocket sizing estimation, and launch angle optimization.

## Available Tools

| Tool | Description |
|------|-------------|
| `rocket_3dof_trajectory` | 3 Degrees of Freedom rocket trajectory simulation with atmospheric effects |
| `estimate_rocket_sizing` | Estimate rocket mass requirements for a target altitude and payload |
| `optimize_launch_angle` | Optimize launch angle for maximum altitude or range |

## CLI Examples

```bash
# Estimate rocket sizing for 100km altitude with 10kg payload
aerospace-mcp-cli run estimate_rocket_sizing \
  --target_altitude_m 100000 \
  --payload_mass_kg 10 \
  --propellant_type solid

# Liquid rocket sizing with design margin
aerospace-mcp-cli run estimate_rocket_sizing \
  --target_altitude_m 50000 \
  --payload_mass_kg 5 \
  --propellant_type liquid \
  --design_margin 1.3

# 3DOF trajectory simulation
aerospace-mcp-cli run rocket_3dof_trajectory \
  --rocket_geometry '{"length_m":2.0,"diameter_m":0.15,"dry_mass_kg":10,"propellant_mass_kg":5,"thrust_n":500,"burn_time_s":10,"cd":0.4,"reference_area_m2":0.018}' \
  --launch_conditions '{"launch_angle_deg":85,"launch_altitude_m":0}'

# Optimize launch angle for max altitude
aerospace-mcp-cli run optimize_launch_angle \
  --rocket_geometry '{"length_m":2.0,"diameter_m":0.15,"dry_mass_kg":10,"propellant_mass_kg":5,"thrust_n":500,"burn_time_s":10,"cd":0.4,"reference_area_m2":0.018}' \
  --optimize_for altitude

# Optimize for max range
aerospace-mcp-cli run optimize_launch_angle \
  --rocket_geometry '{"length_m":2.0,"diameter_m":0.15,"dry_mass_kg":10,"propellant_mass_kg":5,"thrust_n":500,"burn_time_s":10}' \
  --target_range_m 50000 \
  --optimize_for range
```

## Programmatic Usage

```python
from aerospace_mcp.tools.rockets import (
    estimate_rocket_sizing,
    rocket_3dof_trajectory,
    optimize_launch_angle,
)

# Quick sizing estimate
result = estimate_rocket_sizing(
    target_altitude_m=100000,
    payload_mass_kg=10,
    propellant_type="solid",
)
print(result)

# Full trajectory simulation
result = rocket_3dof_trajectory(
    rocket_geometry={
        "length_m": 2.0,
        "diameter_m": 0.15,
        "dry_mass_kg": 10,
        "propellant_mass_kg": 5,
        "thrust_n": 500,
        "burn_time_s": 10,
        "cd": 0.4,
        "reference_area_m2": 0.018,
    },
    launch_conditions={"launch_angle_deg": 85},
)
print(result)
```

## Parameter Reference

### rocket_3dof_trajectory
- `--rocket_geometry` (dict/JSON, required): `length_m`, `diameter_m`, `dry_mass_kg`, `propellant_mass_kg`, `thrust_n`, `burn_time_s`, `cd`, `reference_area_m2`
- `--launch_conditions` (dict/JSON, required): `launch_angle_deg`, `launch_altitude_m`
- `--simulation_options` (dict/JSON, optional): `dt_s`, `max_time_s`

### estimate_rocket_sizing
- `--target_altitude_m` (float, required): Target altitude in meters
- `--payload_mass_kg` (float, required): Payload mass in kg
- `--propellant_type` (one of: solid, liquid; default: solid)
- `--design_margin` (float, default: 1.2): Design safety margin multiplier

### optimize_launch_angle
- `--rocket_geometry` (dict/JSON, required): Rocket geometry parameters
- `--target_range_m` (float, optional): Target range for range optimization
- `--optimize_for` (one of: altitude, range; default: altitude)
- `--angle_bounds_deg` (list/JSON, default: [45.0, 90.0]): Bounds for launch angle search
