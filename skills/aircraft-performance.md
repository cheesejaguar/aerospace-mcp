---
name: aircraft-performance
description: Calculate density altitude, airspeeds, stall speeds, weight and balance, takeoff/landing distances, and fuel reserves
---

# Aircraft Performance

Practical aircraft performance calculations for flight operations: density altitude, airspeed conversions, stall speed calculations, weight & balance, takeoff/landing field lengths, and fuel reserve requirements.

**Safety disclaimer:** For educational/research purposes only. Never use for real flight planning.

## Available Tools

| Tool | Description |
|------|-------------|
| `density_altitude_calculator` | Calculate density altitude from pressure altitude and temperature |
| `true_airspeed_converter` | Convert between IAS, CAS, EAS, TAS, and Mach |
| `stall_speed_calculator` | Calculate stall speeds for clean/takeoff/landing configurations |
| `weight_and_balance` | Calculate CG position and verify within limits |
| `takeoff_performance` | Calculate takeoff V-speeds and field lengths |
| `landing_performance` | Calculate landing distances and approach speeds |
| `fuel_reserve_calculator` | Calculate required fuel reserves per aviation regulations |

## CLI Examples

```bash
# Density altitude at 5000ft pressure altitude, 30C
aerospace-mcp-cli run density_altitude_calculator --pressure_altitude_ft 5000 --temperature_c 30

# Convert 250 KCAS to TAS at FL350
aerospace-mcp-cli run true_airspeed_converter --speed_value 250 --speed_type CAS --altitude_ft 35000

# Convert Mach 0.78 at FL380
aerospace-mcp-cli run true_airspeed_converter --speed_value 0.78 --speed_type MACH --altitude_ft 38000

# Stall speeds for a light aircraft
aerospace-mcp-cli run stall_speed_calculator \
  --weight_kg 1200 \
  --wing_area_m2 16.2 \
  --cl_max_clean 1.6 \
  --cl_max_takeoff 2.0 \
  --cl_max_landing 2.4

# Weight and balance calculation
aerospace-mcp-cli run weight_and_balance \
  --basic_empty_weight_kg 1200 \
  --basic_empty_arm_m 2.0 \
  --fuel_kg 200 \
  --fuel_arm_m 2.5 \
  --payload_items '[{"weight_kg":80,"arm_m":2.2,"name":"Pilot"},{"weight_kg":75,"arm_m":2.2,"name":"Copilot"},{"weight_kg":30,"arm_m":3.5,"name":"Baggage"}]'

# Takeoff performance at a hot/high airport
aerospace-mcp-cli run takeoff_performance \
  --weight_kg 70000 \
  --pressure_altitude_ft 5000 \
  --temperature_c 35 \
  --wind_kts 10

# Landing on a wet runway
aerospace-mcp-cli run landing_performance \
  --weight_kg 60000 \
  --pressure_altitude_ft 1000 \
  --temperature_c 15 \
  --runway_condition wet

# Fuel reserves for FAR 121 operations
aerospace-mcp-cli run fuel_reserve_calculator \
  --regulation FAR_121 \
  --trip_fuel_kg 15000 \
  --cruise_fuel_flow_kg_hr 2500 \
  --flight_time_min 360 \
  --alternate_fuel_kg 3000
```

## Programmatic Usage

```python
from aerospace_mcp.tools.performance import (
    density_altitude_calculator,
    true_airspeed_converter,
    takeoff_performance,
    fuel_reserve_calculator,
)

# Density altitude
result = density_altitude_calculator(pressure_altitude_ft=5000, temperature_c=30)
print(result)

# TAS from CAS at altitude
result = true_airspeed_converter(speed_value=250, speed_type="CAS", altitude_ft=35000)
print(result)

# Takeoff performance
result = takeoff_performance(weight_kg=70000, pressure_altitude_ft=2000, temperature_c=25)
print(result)

# Fuel reserves
result = fuel_reserve_calculator(
    regulation="FAR_121",
    trip_fuel_kg=15000,
    cruise_fuel_flow_kg_hr=2500,
    flight_time_min=360,
)
print(result)
```

## Parameter Reference

### density_altitude_calculator
- `--pressure_altitude_ft` (float, required): Pressure altitude in feet
- `--temperature_c` (float, required): Outside air temperature in Celsius
- `--dewpoint_c` (float, optional): Dewpoint for humidity correction

### true_airspeed_converter
- `--speed_value` (float, required): Speed value (knots or Mach number)
- `--speed_type` (one of: IAS, CAS, EAS, TAS, MACH; required)
- `--altitude_ft` (float, required): Pressure altitude in feet
- `--temperature_c` (float, optional): OAT in Celsius (uses ISA if omitted)
- `--position_error_kts` (float, default: 0): Position error correction in knots

### stall_speed_calculator
- `--weight_kg` (float, required): Aircraft weight in kg
- `--wing_area_m2` (float, required): Wing reference area in m^2
- `--cl_max_clean` (float, required): Max lift coefficient clean config
- `--cl_max_takeoff` (float, optional): Max CL with takeoff flaps
- `--cl_max_landing` (float, optional): Max CL with landing flaps
- `--altitude_ft` (float, default: 0): Pressure altitude in feet
- `--load_factor` (float, default: 1.0)

### weight_and_balance
- `--basic_empty_weight_kg` (float, required): Basic empty weight in kg
- `--basic_empty_arm_m` (float, required): Basic empty weight CG arm in meters
- `--fuel_kg` (float, required): Fuel load in kg
- `--fuel_arm_m` (float, required): Fuel tank CG arm in meters
- `--payload_items` (list[dict]/JSON, required): `[{"weight_kg":..., "arm_m":..., "name":"..."}]`
- `--forward_cg_limit_m` (float, optional): Forward CG limit
- `--aft_cg_limit_m` (float, optional): Aft CG limit
- `--max_takeoff_weight_kg` (float, optional)

### takeoff_performance
- `--weight_kg` (float, required): Takeoff weight in kg
- `--pressure_altitude_ft` (float, required): Airport pressure altitude
- `--temperature_c` (float, required): Outside air temperature
- `--wind_kts` (float, default: 0): Headwind (+) or tailwind (-)
- `--runway_slope_pct` (float, default: 0)
- `--runway_condition` (one of: dry, wet, contaminated; default: dry)

### landing_performance
- `--weight_kg` (float, required): Landing weight in kg
- `--pressure_altitude_ft` (float, required)
- `--temperature_c` (float, required)
- `--wind_kts` (float, default: 0)
- `--runway_slope_pct` (float, default: 0)
- `--runway_condition` (one of: dry, wet, contaminated; default: dry)

### fuel_reserve_calculator
- `--regulation` (one of: FAR_91, FAR_121, JAR_OPS, ICAO; required)
- `--trip_fuel_kg` (float, required): Planned trip fuel in kg
- `--cruise_fuel_flow_kg_hr` (float, required): Cruise fuel flow rate
- `--flight_time_min` (float, required): Planned flight time in minutes
- `--alternate_fuel_kg` (float, default: 0): Fuel to fly to alternate
- `--holding_altitude_ft` (float, default: 1500)
