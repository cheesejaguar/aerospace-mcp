---
name: flight-planning
description: Plan flights, search airports, calculate distances, and estimate aircraft performance
---

# Flight Planning

Core flight planning tools for airport search, route planning, great-circle distance calculation, and aircraft performance estimation using OpenAP.

**Safety disclaimer:** For educational/research purposes only. Never use for real navigation or flight planning.

## Available Tools

| Tool | Description |
|------|-------------|
| `search_airports` | Search airports by IATA code or city name |
| `plan_flight` | Plan a flight route between two airports with performance estimates |
| `calculate_distance` | Calculate great-circle distance between two coordinate points |
| `get_aircraft_performance` | Get climb/cruise/descent performance for an aircraft type |
| `get_system_status` | Check system capabilities and OpenAP availability |

## CLI Examples

```bash
# Search for an airport by IATA code
aerospace-mcp-cli run search_airports --query "SFO"

# Search by city name with country filter
aerospace-mcp-cli run search_airports --query "London" --country "GB"

# Calculate distance between two coordinate points
aerospace-mcp-cli run calculate_distance --lat1 37.36 --lon1 -121.93 --lat2 35.76 --lon2 140.39

# Plan a flight from San Jose to Tokyo on a B777
aerospace-mcp-cli run plan_flight \
  --departure '{"city":"San Jose","iata":"SJC"}' \
  --arrival '{"city":"Tokyo","iata":"NRT"}' \
  --aircraft '{"ac_type":"B777","cruise_alt_ft":39000}'

# Get A320 performance for a 2500km flight
aerospace-mcp-cli run get_aircraft_performance --aircraft_type A320 --distance_km 2500

# Check system status
aerospace-mcp-cli run get_system_status
```

## Programmatic Usage

```python
from aerospace_mcp.tools.core import search_airports, plan_flight, calculate_distance

# Search airports
result = search_airports("SFO")
print(result)

# Plan a flight
result = plan_flight(
    departure={"city": "San Jose", "iata": "SJC"},
    arrival={"city": "Tokyo", "iata": "NRT"},
    aircraft={"ac_type": "B777", "cruise_alt_ft": 39000},
)
print(result)

# Calculate distance
result = calculate_distance(lat1=37.36, lon1=-121.93, lat2=35.76, lon2=140.39)
print(result)
```

## Parameter Reference

### search_airports
- `--query` (str, required): IATA code (e.g., `SJC`) or city name (e.g., `San Jose`)
- `--country` (str, optional): ISO country code filter (e.g., `US`, `JP`)
- `--query_type` (one of: iata, city, auto; default: auto)

### plan_flight
- `--departure` (dict/JSON, required): `{"city": "...", "country": "...", "iata": "..."}`
- `--arrival` (dict/JSON, required): `{"city": "...", "country": "...", "iata": "..."}`
- `--aircraft` (dict/JSON, optional): `{"ac_type": "A320", "cruise_alt_ft": 35000}`
- `--route_options` (dict/JSON, optional): `{"step_km": 100}`

### calculate_distance
- `--lat1` (float, required): Latitude of first point in degrees
- `--lon1` (float, required): Longitude of first point in degrees
- `--lat2` (float, required): Latitude of second point in degrees
- `--lon2` (float, required): Longitude of second point in degrees

### get_aircraft_performance
- `--aircraft_type` (str, required): ICAO aircraft type code (e.g., `A320`, `B737`, `B777`)
- `--distance_km` (float, required): Flight distance in kilometers
- `--cruise_altitude_ft` (float, default: 35000): Cruise altitude in feet

### get_system_status
No parameters.
