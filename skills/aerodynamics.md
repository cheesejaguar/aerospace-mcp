---
name: aerodynamics
description: Analyze wing aerodynamics, airfoil polars, and stability derivatives
---

# Aerodynamics Analysis

Tools for wing aerodynamic analysis using the Vortex Lattice Method, airfoil polar generation, stability derivative calculations, and airfoil database lookup.

## Available Tools

| Tool | Description |
|------|-------------|
| `wing_vlm_analysis` | Vortex Lattice Method wing analysis (CL, CD, CM, L/D) |
| `airfoil_polar_analysis` | Generate airfoil polar data (CL, CD, CM vs alpha) |
| `calculate_stability_derivatives` | Calculate longitudinal stability derivatives |
| `get_airfoil_database` | List available airfoils and their data |

## CLI Examples

```bash
# Analyze a NACA 2412 airfoil
aerospace-mcp-cli run airfoil_polar_analysis --airfoil_name NACA2412 --reynolds_number 1000000

# NACA 0012 at specific Mach and alpha range
aerospace-mcp-cli run airfoil_polar_analysis \
  --airfoil_name NACA0012 \
  --reynolds_number 500000 \
  --mach_number 0.3 \
  --alpha_range_deg '[-5, 15, 1]'

# VLM wing analysis
aerospace-mcp-cli run wing_vlm_analysis \
  --wing_config '{"span_m":10,"chord_root_m":2,"chord_tip_m":1,"sweep_deg":5,"dihedral_deg":3,"twist_deg":-2}' \
  --flight_conditions '{"alpha_deg_list":[0,2,4,6,8,10],"mach":0.2,"reynolds":1000000}'

# Stability derivatives
aerospace-mcp-cli run calculate_stability_derivatives \
  --wing_config '{"span_m":10,"chord_root_m":2,"chord_tip_m":1}' \
  --flight_conditions '{"alpha_deg_list":[5],"mach":0.2}'

# Browse airfoil database
aerospace-mcp-cli run get_airfoil_database
```

## Programmatic Usage

```python
from aerospace_mcp.tools.aerodynamics import (
    airfoil_polar_analysis,
    wing_vlm_analysis,
    get_airfoil_database,
)

# Airfoil polar
result = airfoil_polar_analysis("NACA0012", reynolds_number=1e6)
print(result)

# Wing VLM analysis
result = wing_vlm_analysis(
    wing_config={"span_m": 10, "chord_root_m": 2, "chord_tip_m": 1},
    flight_conditions={"alpha_deg_list": [0, 5, 10], "mach": 0.2, "reynolds": 1e6},
)
print(result)
```

## Parameter Reference

### wing_vlm_analysis
- `--wing_config` (dict/JSON, required): Wing geometry — `span_m`, `chord_root_m`, `chord_tip_m`, `sweep_deg`, `dihedral_deg`, `twist_deg`, `airfoil_root`, `airfoil_tip`
- `--flight_conditions` (dict/JSON, required): `alpha_deg_list`, `mach`, `reynolds`
- `--analysis_options` (dict/JSON, optional): Additional analysis options

### airfoil_polar_analysis
- `--airfoil_name` (str, required): Airfoil designation (e.g., `NACA0012`, `NACA2412`)
- `--reynolds_number` (float, default: 1000000): Reynolds number
- `--mach_number` (float, default: 0.1): Mach number
- `--alpha_range_deg` (list[float]/JSON, optional): Angle of attack range `[start, end, step]`

### calculate_stability_derivatives
- `--wing_config` (dict/JSON, required): Wing geometry parameters
- `--flight_conditions` (dict/JSON, required): Flight conditions

### get_airfoil_database
No parameters.
