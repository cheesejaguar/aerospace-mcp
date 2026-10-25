---
name: atmosphere
description: Calculate atmospheric profiles and wind models at various altitudes
---

# Atmospheric Calculations

Tools for atmospheric modeling including ISA standard atmosphere profiles and wind profile models at various altitudes.

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
| `get_atmosphere_profile` | Calculate pressure, temperature, density, and speed of sound at altitudes |
| `wind_model_simple` | Calculate wind speed/direction profiles using logarithmic or power-law models |

## CLI Examples

```bash
# Get ISA atmosphere profile at multiple altitudes
aerospace-mcp-cli run get_atmosphere_profile --altitudes_m '[0,1000,5000,10000,15000]'

# COESA model name (same US Standard Atmosphere 1976, valid to 86 km)
aerospace-mcp-cli run get_atmosphere_profile --altitudes_m '[0,20000,50000,80000]' --model_type COESA

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
- `--model_type` (one of: ISA, COESA; default: ISA): Atmospheric model type (both are US Standard Atmosphere 1976, geometric altitudes 0-86 km)

### wind_model_simple
- `--altitudes_m` (list[float]/JSON, required): List of altitudes in meters
- `--surface_wind_speed_ms` (float, default: 5.0): Surface wind speed in m/s
- `--surface_wind_direction_deg` (float, default: 270.0): Surface wind direction in degrees
- `--model_type` (one of: logarithmic, power_law; default: logarithmic): Wind profile model
- `--roughness_length_m` (float, default: 0.03): Surface roughness length in meters
