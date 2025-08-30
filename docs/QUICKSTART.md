# Quick Start Guide - Aerospace MCP

Get up and running with Aerospace MCP in under 5 minutes! This guide covers the fastest path to start planning flights and exploring aviation data.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [5-Minute Setup](#5-minute-setup)
- [First Flight Plan](#first-flight-plan)
- [Common Use Cases](#common-use-cases)
- [MCP Quick Setup](#mcp-quick-setup)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

## 🔧 Prerequisites

**Required:**
- Python 3.11+ installed ([Download here](https://www.python.org/downloads/))
- Internet connection (for initial setup and airport data)
- Terminal/Command Prompt access

**Recommended:**
- UV package manager ([Install guide](https://astral.sh/uv/))
- 1GB+ RAM available
- Modern terminal with curl support

**Check your Python version:**
```bash
python --version
# Should show 3.11.0 or higher
```

## 🚀 5-Minute Setup

### Option A: UV (Fastest)

```bash
# 1. Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clone the repository
git clone https://github.com/username/aerospace-mcp.git
cd aerospace-mcp

# 3. Setup environment (30 seconds)
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 4. Install dependencies (60 seconds)
uv add fastapi uvicorn airportsdata geographiclib openap

# 5. Start the server (5 seconds)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Option B: Pip (Traditional)

```bash
# 1. Clone the repository
git clone https://github.com/username/aerospace-mcp.git
cd aerospace-mcp

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install fastapi uvicorn[standard] airportsdata geographiclib openap

# 4. Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Option C: Docker (Containerized)

```bash
# 1. Clone and build (2 minutes)
git clone https://github.com/username/aerospace-mcp.git
cd aerospace-mcp
docker build -t aerospace-mcp .

# 2. Run container
docker run -p 8000:8000 aerospace-mcp

# Server is now running on http://localhost:8000
```

## ✈️ First Flight Plan

Once your server is running, test it with these commands:

### 1. Health Check

```bash
curl "http://localhost:8000/health"
```

**Expected Output:**
```json
{
  "status": "ok",
  "openap": true,
  "airports_count": 28756
}
```

### 2. Find Airports

```bash
# Find airports in New York
curl "http://localhost:8000/airports/by_city?city=New%20York"

# Find airports in London, UK specifically
curl "http://localhost:8000/airports/by_city?city=London&country=GB"
```

### 3. Plan Your First Flight

```bash
# Simple domestic flight
curl -X POST "http://localhost:8000/plan" \
  -H "Content-Type: application/json" \
  -d '{
    "depart_city": "San Francisco",
    "arrive_city": "Los Angeles",
    "ac_type": "A320",
    "backend": "openap"
  }'
```

**Expected Output:**
```json
{
  "engine": "openap",
  "depart": {
    "iata": "SFO",
    "name": "San Francisco International Airport",
    "city": "San Francisco",
    "country": "US",
    "lat": 37.621311,
    "lon": -122.378958
  },
  "arrive": {
    "iata": "LAX", 
    "name": "Los Angeles International Airport",
    "city": "Los Angeles",
    "country": "US",
    "lat": 33.943001,
    "lon": -118.410042
  },
  "distance_km": 559.1,
  "distance_nm": 301.9,
  "estimates": {
    "block": {
      "time_min": 78.5,
      "fuel_kg": 2140.2
    }
  }
}
```

### 4. International Flight

```bash
# Transatlantic flight
curl -X POST "http://localhost:8000/plan" \
  -H "Content-Type: application/json" \
  -d '{
    "depart_city": "New York",
    "arrive_city": "London", 
    "prefer_depart_iata": "JFK",
    "prefer_arrive_iata": "LHR",
    "ac_type": "B777",
    "cruise_alt_ft": 39000,
    "backend": "openap"
  }'
```

## 🎯 Common Use Cases

### Route Analysis

```python
# Save this as test_flight.py and run: python test_flight.py
import requests

def analyze_route(departure, arrival, aircraft="A320"):
    """Analyze a flight route and print key metrics."""
    response = requests.post("http://localhost:8000/plan", json={
        "depart_city": departure,
        "arrive_city": arrival,
        "ac_type": aircraft,
        "cruise_alt_ft": 35000,
        "backend": "openap"
    })
    
    if response.status_code == 200:
        plan = response.json()
        print(f"\n=== {departure} → {arrival} ({aircraft}) ===")
        print(f"Distance: {plan['distance_nm']:.0f} NM ({plan['distance_km']:.0f} km)")
        print(f"Flight Time: {plan['estimates']['block']['time_min']:.0f} minutes")
        print(f"Fuel Required: {plan['estimates']['block']['fuel_kg']:,.0f} kg")
        print(f"Departure: {plan['depart']['iata']} - {plan['depart']['name']}")
        print(f"Arrival: {plan['arrive']['iata']} - {plan['arrive']['name']}")
    else:
        print(f"Error: {response.json()}")

# Test various routes
analyze_route("Chicago", "Denver", "B737")
analyze_route("Tokyo", "Singapore", "A350")
analyze_route("Miami", "Sao Paulo", "B787")
```

### Aircraft Comparison

```python
# Compare different aircraft on the same route
import requests

def compare_aircraft(departure, arrival, aircraft_types):
    """Compare multiple aircraft on the same route."""
    print(f"\n=== Aircraft Comparison: {departure} → {arrival} ===")
    print(f"{'Aircraft':<8} {'Time (min)':<10} {'Fuel (kg)':<10} {'Range (NM)':<12}")
    print("-" * 50)
    
    for aircraft in aircraft_types:
        response = requests.post("http://localhost:8000/plan", json={
            "depart_city": departure,
            "arrive_city": arrival,
            "ac_type": aircraft,
            "cruise_alt_ft": 35000,
            "backend": "openap"
        })
        
        if response.status_code == 200:
            plan = response.json()
            time_min = plan['estimates']['block']['time_min']
            fuel_kg = plan['estimates']['block']['fuel_kg']
            distance_nm = plan['distance_nm']
            print(f"{aircraft:<8} {time_min:<10.0f} {fuel_kg:<10.0f} {distance_nm:<12.0f}")
        else:
            print(f"{aircraft:<8} {'ERROR':<10}")

# Compare narrow-body aircraft
compare_aircraft("Boston", "Seattle", ["A320", "A321", "B737", "B738"])
```

### Multi-City Journey

```python
# Plan a multi-leg journey
def plan_journey(cities, aircraft="A320"):
    """Plan a multi-city journey."""
    total_distance = 0
    total_time = 0
    total_fuel = 0
    
    print(f"\n=== Multi-City Journey ({aircraft}) ===")
    
    for i in range(len(cities) - 1):
        departure = cities[i]
        arrival = cities[i + 1]
        
        response = requests.post("http://localhost:8000/plan", json={
            "depart_city": departure,
            "arrive_city": arrival,
            "ac_type": aircraft,
            "cruise_alt_ft": 35000,
            "backend": "openap"
        })
        
        if response.status_code == 200:
            plan = response.json()
            distance = plan['distance_nm']
            time = plan['estimates']['block']['time_min']
            fuel = plan['estimates']['block']['fuel_kg']
            
            total_distance += distance
            total_time += time
            total_fuel += fuel
            
            print(f"Leg {i+1}: {departure} → {arrival}")
            print(f"  {distance:.0f} NM, {time:.0f} min, {fuel:.0f} kg fuel")
    
    print(f"\nTotals:")
    print(f"  Distance: {total_distance:.0f} NM")
    print(f"  Time: {total_time:.0f} min ({total_time/60:.1f} hours)")
    print(f"  Fuel: {total_fuel:.0f} kg")

# Example: Grand tour of major cities
plan_journey([
    "New York", "London", "Dubai", "Singapore", "Tokyo", "Los Angeles", "New York"
], "B777")
```

## 🤖 MCP Quick Setup

For AI assistant integration (Claude, etc.):

### Claude Desktop

1. **Find your Claude Desktop config file:**
   - **Windows**: `%APPDATA%\Claude\config.json`
   - **macOS**: `~/Library/Application Support/Claude/config.json`
   - **Linux**: `~/.config/Claude/config.json`

2. **Add aerospace-mcp server:**

```json
{
  "mcpServers": {
    "aerospace-mcp": {
      "command": "python",
      "args": ["-m", "aerospace_mcp.server"],
      "cwd": "/absolute/path/to/aerospace-mcp",
      "env": {
        "PYTHONPATH": "/absolute/path/to/aerospace-mcp"
      }
    }
  }
}
```

3. **Restart Claude Desktop**

4. **Test with natural language:**
   - "Search for airports in Tokyo"
   - "Plan a flight from San Francisco to New York using an A320"
   - "What's the distance between Paris and Rome?"

### VS Code Continue

Add to your Continue config (`~/.continue/config.json`):

```json
{
  "models": [...],
  "mcpServers": [
    {
      "name": "aerospace-mcp",
      "command": "python",
      "args": ["-m", "aerospace_mcp.server"],
      "workingDirectory": "/absolute/path/to/aerospace-mcp"
    }
  ]
}
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Server Won't Start

**Problem**: `ModuleNotFoundError` or import errors

**Solutions**:
```bash
# Check Python version
python --version  # Must be 3.11+

# Verify virtual environment is activated
which python  # Should point to your venv

# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Test individual imports
python -c "import fastapi; print('FastAPI OK')"
python -c "import airportsdata; print('AirportsData OK')"
```

#### 2. OpenAP Not Available

**Problem**: Health check shows `"openap": false`

**Solutions**:
```bash
# Install OpenAP
pip install openap

# If that fails, try:
pip install openap --no-cache-dir
conda install -c conda-forge openap

# Verify installation
python -c "import openap; print('OpenAP OK')"
```

**Workaround**: The system works without OpenAP, but performance estimates won't be available.

#### 3. Airport Not Found

**Problem**: `404 Not Found` when searching for airports

**Solutions**:
```bash
# Try different city name variations
curl "http://localhost:8000/airports/by_city?city=NYC"          # Doesn't work
curl "http://localhost:8000/airports/by_city?city=New%20York"   # Works

# Use country filter for common city names
curl "http://localhost:8000/airports/by_city?city=Paris&country=FR"  # France
curl "http://localhost:8000/airports/by_city?city=Paris&country=US"  # Texas, USA

# Use IATA codes directly
curl -X POST "http://localhost:8000/plan" -H "Content-Type: application/json" \
  -d '{"depart_city": "Any", "arrive_city": "Any", "prefer_depart_iata": "JFK", "prefer_arrive_iata": "LAX", "ac_type": "A320"}'
```

#### 4. Performance Issues

**Problem**: Slow response times

**Solutions**:
```bash
# Increase route step size for faster processing
curl -X POST "http://localhost:8000/plan" -H "Content-Type: application/json" \
  -d '{
    "depart_city": "Tokyo", 
    "arrive_city": "Sydney",
    "ac_type": "B777",
    "route_step_km": 100.0
  }'

# Monitor system resources
htop  # Linux/macOS
# Task Manager on Windows

# Check server logs
uvicorn main:app --reload --log-level debug
```

#### 5. Docker Issues

**Problem**: Container won't start or can't connect

**Solutions**:
```bash
# Check container status
docker ps -a

# View container logs
docker logs aerospace-mcp

# Rebuild image
docker build --no-cache -t aerospace-mcp .

# Check port binding
docker run -p 8000:8000 aerospace-mcp
netstat -an | grep 8000  # Check if port is in use
```

#### 6. MCP Integration Issues

**Problem**: Claude Desktop doesn't recognize the server

**Solutions**:

1. **Check paths are absolute:**
```json
{
  "mcpServers": {
    "aerospace-mcp": {
      "command": "python",
      "args": ["-m", "aerospace_mcp.server"],
      "cwd": "/Users/yourname/aerospace-mcp"
    }
  }
}
```

2. **Verify MCP server works standalone:**
```bash
cd /path/to/aerospace-mcp
python -m aerospace_mcp.server
# Should show MCP server starting up
```

3. **Check Claude Desktop logs:**
   - **macOS**: `~/Library/Logs/Claude/`
   - **Windows**: `%APPDATA%\Claude\Logs\`

### Performance Troubleshooting

#### Memory Usage

```bash
# Monitor memory usage
python -c "
import psutil
import main
print(f'Memory usage: {psutil.virtual_memory().percent}%')
"
```

#### Response Time Analysis

```python
import time
import requests

def benchmark_endpoint(url, data=None, iterations=5):
    """Benchmark an API endpoint."""
    times = []
    
    for i in range(iterations):
        start = time.time()
        if data:
            response = requests.post(url, json=data)
        else:
            response = requests.get(url)
        end = time.time()
        
        if response.status_code == 200:
            times.append(end - start)
        else:
            print(f"Error {response.status_code}: {response.text}")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"Average response time: {avg_time:.3f}s")
        print(f"Min: {min(times):.3f}s, Max: {max(times):.3f}s")

# Test different endpoints
benchmark_endpoint("http://localhost:8000/health")
benchmark_endpoint("http://localhost:8000/airports/by_city?city=London")
benchmark_endpoint("http://localhost:8000/plan", {
    "depart_city": "New York", 
    "arrive_city": "Los Angeles", 
    "ac_type": "A320", 
    "backend": "openap"
})
```

## 🎯 Next Steps

Once you have the system running:

### 1. Explore the Documentation
- [**API Reference**](API.md) - Complete endpoint documentation
- [**Architecture Guide**](ARCHITECTURE.md) - System design and internals
- [**Integration Guide**](INTEGRATION.md) - Client libraries and examples

### 2. Try Advanced Features

```bash
# Advanced flight planning with custom parameters
curl -X POST "http://localhost:8000/plan" \
  -H "Content-Type: application/json" \
  -d '{
    "depart_city": "Frankfurt",
    "arrive_city": "Tokyo",
    "prefer_depart_iata": "FRA",
    "prefer_arrive_iata": "NRT", 
    "ac_type": "A350",
    "cruise_alt_ft": 41000,
    "mass_kg": 280000,
    "route_step_km": 50.0,
    "backend": "openap"
  }'
```

### 3. Build Client Applications

Start with the Python client examples above, then explore:
- Web frontend development
- Mobile app integration
- Batch processing scripts
- Data analysis pipelines

### 4. Contribute to the Project

- Report bugs and suggest features on GitHub
- Add support for new aircraft types
- Improve documentation
- Optimize performance
- Add new backends (weather, NOTAM, etc.)

### 5. Production Deployment

Ready for production? See:
- [**Deployment Guide**](DEPLOYMENT.md) - Production setup
- [**Contributing Guide**](CONTRIBUTING.md) - Development workflow
- **Security considerations** - Authentication, rate limiting, etc.

## 📚 Learning Resources

### Aviation Background
- **ICAO Aircraft Codes**: [ICAO Doc 8643](https://www.icao.int/publications/DOC8643/)
- **IATA Airport Codes**: [IATA Airport Codes](https://www.iata.org/en/publications/directories/code-search/)
- **Great Circle Navigation**: [Navigation principles](https://en.wikipedia.org/wiki/Great-circle_navigation)

### Technical Resources
- **FastAPI Documentation**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com/)
- **OpenAP Project**: [TU Delft OpenAP](https://github.com/TUDelft-CNS-ATM/openap)
- **Model Context Protocol**: [MCP Specification](https://modelcontextprotocol.io/)

---

**🎉 Congratulations!** You now have a fully functional flight planning system. Start exploring and building amazing aviation applications!

**Need help?** Check our [GitHub Issues](https://github.com/username/aerospace-mcp/issues) or join our [Discord community](https://discord.gg/aerospace-mcp).