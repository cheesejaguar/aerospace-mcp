---
name: rockets
description: Simulate rocket trajectories, estimate sizing, and optimize launch angles
---

# Rocket Analysis

Tools for rocket trajectory simulation (3DOF), rocket sizing estimation, and launch angle optimization.

## Setup

The examples below call `aerospace-mcp-cli`. If that command is not on `PATH`,
run each one through [uv](https://docs.astral.sh/uv/) instead, which downloads a
prebuilt package (Linux and macOS, x86_64 and arm64; no Rust or C compiler
needed) and caches it after the first run:

```bash
uvx --from aerospace-mcp aerospace-mcp-cli list
```

Install uv with `curl -LsSf https://astral.sh/uv/install.sh | sh` (Linux/macOS);
it fetches Python 3.11+ itself if the system has none. `uv tool install
aerospace-mcp` puts `aerospace-mcp-cli` on `PATH` permanently. When the
aerospace MCP server is connected, the same tools are also available as MCP
tools with the same names and parameters.

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
  --rocket_geometry '{"length_m":2.0,"diameter_m":0.15,"dry_mass_kg":10,"propellant_mass_kg":5,"cd":0.4,"thrust_curve":[[0,500],[10,500],[10.1,0]]}' \
  --launch_conditions '{"launch_angle_deg":85}'

# Optimize launch angle for max altitude
aerospace-mcp-cli run optimize_launch_angle \
  --rocket_geometry '{"length_m":2.0,"diameter_m":0.15,"dry_mass_kg":10,"propellant_mass_kg":5,"cd":0.4,"thrust_curve":[[0,500],[10,500],[10.1,0]]}' \
  --optimize_for altitude

# Optimize for max range
aerospace-mcp-cli run optimize_launch_angle \
  --rocket_geometry '{"length_m":2.0,"diameter_m":0.15,"dry_mass_kg":10,"propellant_mass_kg":5,"thrust_curve":[[0,500],[10,500],[10.1,0]]}' \
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
        "cd": 0.4,
        # [[time_s, thrust_n], ...]: 500 N for 10 s
        "thrust_curve": [[0, 500], [10, 500], [10.1, 0]],
    },
    launch_conditions={"launch_angle_deg": 85},
)
print(result)
```

## Parameter Reference

### rocket_3dof_trajectory
- `--rocket_geometry` (dict/JSON, required): `length_m`, `diameter_m`, `dry_mass_kg`, `propellant_mass_kg`, optional `cd` (default 0.3) and `thrust_curve` as `[[time_s, thrust_n], ...]`; the reference area is the body cross-section from `diameter_m`
- `--launch_conditions` (dict/JSON, required): `launch_angle_deg` (default 90)
- `--simulation_options` (dict/JSON, optional): `dt_s`, `max_time_s`

### estimate_rocket_sizing
- `--target_altitude_m` (float, required): Target altitude in meters
- `--payload_mass_kg` (float, required): Payload mass in kg
- `--propellant_type` (one of: solid, liquid; default: solid)
- `--design_margin` (float, default: 1.2): Design safety margin multiplier

### optimize_launch_angle
- `--rocket_geometry` (dict/JSON, required): Rocket geometry parameters
- `--target_range_m` (float, optional): Target range; accepted but not used yet (downrange distance is not modelled)
- `--optimize_for` (one of: altitude, range; default: altitude)
- `--angle_bounds_deg` (list/JSON, default: [45.0, 90.0]): Bounds for launch angle search
