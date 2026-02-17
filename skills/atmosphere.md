---
name: atmosphere
description: Calculate atmospheric profiles and wind models at various altitudes
---

# Atmospheric Calculations

Tools for atmospheric modeling including ISA standard atmosphere profiles and wind profile models at various altitudes.

## Available Tools

| Tool | Description |
|------|-------------|
| `get_atmosphere_profile` | Calculate pressure, temperature, density, and speed of sound at altitudes |
| `wind_model_simple` | Calculate wind speed/direction profiles using logarithmic or power-law models |

## CLI Examples

```bash
# Get ISA atmosphere profile at multiple altitudes
aerospace-mcp-cli run get_atmosphere_profile --altitudes_m '[0,1000,5000,10000,15000]'

# Get enhanced atmosphere profile
aerospace-mcp-cli run get_atmosphere_profile --altitudes_m '[0,5000,10000]' --model_type enhanced

# Simple wind profile with default surface conditions
aerospace-mcp-cli run wind_model_simple --altitudes_m '[10,50,100,500,1000]'

# Wind profile with custom surface wind
aerospace-mcp-cli run wind_model_simple \
  --altitudes_m '[10,100,500,1000]' \
  --surface_wind_speed_ms 10.0 \
  --surface_wind_direction_deg 180.0 \
  --model_type power_law
```

## Programmatic Usage

```python
from aerospace_mcp.tools.atmosphere import get_atmosphere_profile, wind_model_simple

# ISA atmosphere at sea level, 5km, and 10km
result = get_atmosphere_profile([0, 5000, 10000])
print(result)

# Wind profile with logarithmic model
result = wind_model_simple(
    altitudes_m=[10, 100, 500, 1000],
    surface_wind_speed_ms=8.0,
    surface_wind_direction_deg=270.0,
)
print(result)
```

## Parameter Reference

### get_atmosphere_profile
- `--altitudes_m` (list[float]/JSON, required): List of altitudes in meters, e.g. `[0,5000,10000]`
- `--model_type` (one of: ISA, enhanced; default: ISA): Atmospheric model type

### wind_model_simple
- `--altitudes_m` (list[float]/JSON, required): List of altitudes in meters
- `--surface_wind_speed_ms` (float, default: 5.0): Surface wind speed in m/s
- `--surface_wind_direction_deg` (float, default: 270.0): Surface wind direction in degrees
- `--model_type` (one of: logarithmic, power_law; default: logarithmic): Wind profile model
- `--roughness_length_m` (float, default: 0.03): Surface roughness length in meters
