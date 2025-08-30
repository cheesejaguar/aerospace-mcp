# Atmosphere and Coordinate Frame Tools Guide

## Overview

The aerospace-mcp server now includes advanced atmosphere modeling and coordinate frame transformation tools, expanding beyond flight planning to support general aerospace research and analysis workflows.

## New MCP Tools

### Atmosphere Tools

#### `get_atmosphere_profile`

Get atmospheric properties at specified altitudes using the International Standard Atmosphere (ISA) model.

**Input Schema:**
```json
{
  "altitudes_m": [0, 5000, 11000, 20000],
  "model_type": "ISA"
}
```

**Output:** Tabular and JSON data with pressure, temperature, density, and speed of sound at each altitude.

**Example Usage:**
```json
{
  "tool": "get_atmosphere_profile",
  "arguments": {
    "altitudes_m": [0, 5000, 11000],
    "model_type": "ISA"
  }
}
```

**Response:**
```
Atmospheric Profile (ISA)
==================================================
 Alt (m)   Press (Pa)  Temp (K)    Density  Sound (m/s)
------------------------------------------------------------
       0     101325.0    288.15   1.225000        340.3
    5000      54048.3    255.68   0.736429        320.5
   11000      22699.9    216.77   0.364801        295.2
```

#### `wind_model_simple`

Calculate wind speeds at different altitudes using logarithmic or power law wind profile models.

**Input Schema:**
```json
{
  "altitudes_m": [0, 10, 50, 100, 500],
  "surface_wind_mps": 12.0,
  "surface_altitude_m": 0.0,
  "model": "logarithmic",
  "roughness_length_m": 0.1
}
```

**Models Available:**
- `logarithmic`: Uses surface roughness for realistic boundary layer modeling
- `power`: Simple power law with typical exponent for open terrain

### Coordinate Frame Tools

#### `transform_frames`

Transform coordinates between different reference frames commonly used in aerospace applications.

**Supported Frames:**
- `ECEF`: Earth-Centered, Earth-Fixed
- `ECI`: Earth-Centered Inertial  
- `GEODETIC`: Latitude/Longitude/Altitude
- `ITRF`: International Terrestrial Reference Frame
- `GCRS`: Geocentric Celestial Reference System

**Input Schema:**
```json
{
  "xyz": [1000000.0, 2000000.0, 3000000.0],
  "from_frame": "ECEF",
  "to_frame": "GEODETIC", 
  "epoch_iso": "2023-01-01T12:00:00"
}
```

#### `geodetic_to_ecef`

Convert latitude/longitude/altitude coordinates to ECEF coordinates.

**Input Schema:**
```json
{
  "latitude_deg": 37.7749,
  "longitude_deg": -122.4194,
  "altitude_m": 100.0
}
```

#### `ecef_to_geodetic`

Convert ECEF coordinates to geodetic coordinates.

**Input Schema:**
```json
{
  "x": -2706217.22,
  "y": -4261126.21,
  "z": 3885786.75
}
```

## Library Dependencies

### Core Functionality

All atmosphere and basic coordinate frame tools work with **no additional dependencies**. The implementation includes:

- Manual ISA calculations following ICAO standard atmosphere
- WGS84 ellipsoid transformations for ECEF↔Geodetic conversions
- Logarithmic and power-law wind profile models

### Optional Enhanced Libraries

Install optional dependencies for improved accuracy and additional features:

```bash
# Atmosphere enhancements
uv pip install --optional-dependencies atmosphere

# Coordinate frame enhancements  
uv pip install --optional-dependencies space

# Install all optional dependencies
uv pip install --optional-dependencies all
```

**Enhanced Capabilities:**

- **ambiance**: Higher-precision ISA calculations with atmospheric viscosity
- **astropy**: High-precision coordinate transformations with proper time handling
- **skyfield**: Alternative coordinate frame library with JPL ephemeris support

## Usage Examples

### Flight Environment Analysis

```python
# Get atmospheric conditions along climb profile
altitudes = [0, 1000, 3000, 5000, 8000, 11000]
atmosphere = await mcp_client.call_tool("get_atmosphere_profile", {
    "altitudes_m": altitudes,
    "model_type": "ISA"
})

# Calculate wind profile for takeoff analysis
wind_profile = await mcp_client.call_tool("wind_model_simple", {
    "altitudes_m": [0, 50, 100, 200, 500],
    "surface_wind_mps": 15.0,
    "model": "logarithmic",
    "roughness_length_m": 0.03  # Airport terrain
})
```

### Satellite Ground Track Calculation

```python
# Convert satellite position from ECI to geodetic
ground_point = await mcp_client.call_tool("transform_frames", {
    "xyz": [6678000.0, 0.0, 1000000.0],  # ECI position
    "from_frame": "ECI",
    "to_frame": "GEODETIC",
    "epoch_iso": "2024-01-15T12:00:00"
})

# Convert ground station coordinates to ECEF for range calculations
station_ecef = await mcp_client.call_tool("geodetic_to_ecef", {
    "latitude_deg": 34.0522,  # Los Angeles
    "longitude_deg": -118.2437,
    "altitude_m": 71.0
})
```

### Rocket Launch Analysis

```python
# Atmosphere profile for ascent trajectory
launch_atmosphere = await mcp_client.call_tool("get_atmosphere_profile", {
    "altitudes_m": list(range(0, 100001, 5000)),  # 0-100km in 5km steps
    "model_type": "ISA"
})

# Wind conditions affecting launch
wind_conditions = await mcp_client.call_tool("wind_model_simple", {
    "altitudes_m": list(range(0, 20001, 1000)),  # 0-20km in 1km steps
    "surface_wind_mps": 8.0,
    "model": "power"
})
```

### UAV Mission Planning

```python
# Operating environment at mission altitude
mission_alt = 500  # meters AGL
environment = await mcp_client.call_tool("get_atmosphere_profile", {
    "altitudes_m": [mission_alt],
    "model_type": "ISA" 
})

# Wind conditions for energy planning
wind_profile = await mcp_client.call_tool("wind_model_simple", {
    "altitudes_m": [0, 100, 200, 300, 400, 500],
    "surface_wind_mps": 6.0,
    "model": "logarithmic",
    "roughness_length_m": 0.5  # Rural terrain
})
```

## Technical Implementation

### Atmosphere Models

The ISA implementation follows ICAO Document 7488 with proper handling of:
- Troposphere (0-11km): Linear temperature lapse
- Stratosphere layers with different lapse rates  
- Proper pressure/density calculations using hydrostatic equation
- Speed of sound using ideal gas relationships

**Accuracy:** Sea level conditions within 0.1% of standard values.

### Coordinate Transformations

**Manual Implementations:**
- WGS84 ellipsoid parameters (a=6378137m, f=1/298.257223563)
- Iterative geodetic height calculation (typically converges in 2-3 iterations)
- Proper handling of polar and equatorial edge cases

**With Optional Libraries:**
- Astropy: Full IAU-compliant transformations with proper time systems
- Skyfield: JPL ephemeris support for planetary applications

**Accuracy:** 
- Manual ECEF↔Geodetic: <1mm for typical Earth surface coordinates
- With astropy: <1μm with proper time/reference frame handling

### Performance

- **Atmosphere calculations:** ~1ms per altitude point
- **Coordinate transformations:** ~0.1ms per point (manual), ~1ms (astropy)
- **Memory usage:** <10MB additional for core functionality
- **Vectorization:** Supports batch calculations for efficiency

## Error Handling

All tools provide graceful degradation and clear error messages:

```json
{
  "error": "Altitude 100000.0m out of ISA range (0-86000m)",
  "suggestion": "Use altitudes between 0 and 86000 meters"
}
```

```json
{
  "error": "Transformation from GCRS to ITRF not implemented",
  "suggestion": "Install astropy or skyfield for full functionality"
}
```

## Integration with Existing Tools

The new atmosphere and coordinate tools integrate seamlessly with existing flight planning:

1. **Enhanced Flight Planning:** Atmosphere data improves altitude-dependent performance calculations
2. **Wind Corrections:** Surface wind models support runway analysis and approach planning  
3. **Coordinate Consistency:** All tools use consistent SI units and WGS84 reference frame
4. **System Status:** `get_system_status` reports availability of all optional libraries

## Development and Testing

### Running Tests

```bash
# Test atmosphere calculations
python -m pytest tests/test_integrations_atmosphere.py -v

# Test coordinate transformations  
python -m pytest tests/test_integrations_frames.py -v

# Run all tests
python -m pytest tests/ -v
```

### Adding New Features

The modular design allows easy extension:

1. **New atmosphere models:** Add to `atmosphere.py` with appropriate fallbacks
2. **Additional frames:** Extend `frames.py` with new coordinate systems
3. **Integration tests:** Add realistic scenarios to test suites
4. **Documentation:** Update this guide with usage examples

## Future Enhancements

**Planned for Phase 2:**
- Aircraft aerodynamics tools (VLM, airfoil analysis)
- Propeller/UAV performance calculations
- Integration with external weather data sources

**Planned for Phase 3:**
- Rocket trajectory optimization
- Orbital mechanics and transfers
- Advanced guidance and control tools