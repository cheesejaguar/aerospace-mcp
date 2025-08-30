# Integration Guide

This guide provides comprehensive instructions for integrating the Aerospace MCP project with various clients and applications.

## Table of Contents

- [MCP Client Configuration](#mcp-client-configuration)
- [FastAPI Direct Integration](#fastapi-direct-integration)
- [Authentication & Security](#authentication--security)
- [Rate Limiting](#rate-limiting)
- [Common Integration Patterns](#common-integration-patterns)
- [Troubleshooting](#troubleshooting)

## MCP Client Configuration

The Aerospace MCP project provides flight planning capabilities through both MCP (Model Context Protocol) and direct FastAPI access. Here are configuration examples for popular MCP clients.

### Claude Desktop

Add the following configuration to your Claude Desktop settings:

#### Basic Configuration

```json
{
  "mcpServers": {
    "aerospace-mcp": {
      "command": "python",
      "args": ["-m", "uvicorn", "main:app", "--host", "localhost", "--port", "8000"],
      "cwd": "/path/to/aerospace-mcp"
    }
  }
}
```

#### Advanced Configuration with Environment Variables

```json
{
  "mcpServers": {
    "aerospace-mcp": {
      "command": "python",
      "args": ["-m", "uvicorn", "main:app", "--host", "localhost", "--port", "8000", "--reload"],
      "cwd": "/path/to/aerospace-mcp",
      "env": {
        "DEBUG": "true",
        "LOG_LEVEL": "info",
        "OPENAP_AVAILABLE": "true"
      }
    }
  }
}
```

### VS Code with MCP Extension

Add to your VS Code settings.json:

```json
{
  "mcp.servers": [
    {
      "name": "aerospace-mcp",
      "command": "python",
      "args": ["-m", "uvicorn", "main:app", "--host", "localhost", "--port", "8000"],
      "cwd": "/path/to/aerospace-mcp",
      "autoStart": true
    }
  ]
}
```

### Continue.dev

Add to your `config.json`:

```json
{
  "models": [...],
  "mcpServers": [
    {
      "name": "aerospace-mcp",
      "serverPath": "python",
      "args": ["-m", "uvicorn", "main:app", "--host", "localhost", "--port", "8000"],
      "workingDirectory": "/path/to/aerospace-mcp"
    }
  ]
}
```

### Generic MCP Client

For any MCP client that supports server configuration:

```bash
# Start the server
python -m uvicorn main:app --host localhost --port 8000

# The server will be available at:
# HTTP: http://localhost:8000
# MCP: mcp://localhost:8000
```

## FastAPI Direct Integration

### Using cURL

#### Health Check

```bash
# Basic health check
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "ok",
#   "openap": true,
#   "airports_count": 28000
# }
```

#### Find Airports by City

```bash
# Find airports in a specific city
curl "http://localhost:8000/airports/by_city?city=San%20Francisco"

# Find airports with country filter
curl "http://localhost:8000/airports/by_city?city=London&country=GB"

# Expected response:
# [
#   {
#     "iata": "SFO",
#     "icao": "KSFO",
#     "name": "San Francisco International Airport",
#     "city": "San Francisco",
#     "country": "US",
#     "lat": 37.621311,
#     "lon": -122.378958,
#     "tz": "America/Los_Angeles"
#   }
# ]
```

#### Flight Planning

```bash
# Basic flight plan
curl -X POST "http://localhost:8000/plan" \
  -H "Content-Type: application/json" \
  -d '{
    "depart_city": "San Francisco",
    "arrive_city": "New York",
    "ac_type": "A320",
    "cruise_alt_ft": 35000,
    "route_step_km": 50.0,
    "backend": "openap"
  }'

# Flight plan with specific airports
curl -X POST "http://localhost:8000/plan" \
  -H "Content-Type: application/json" \
  -d '{
    "depart_city": "San Francisco",
    "arrive_city": "New York",
    "prefer_depart_iata": "SFO",
    "prefer_arrive_iata": "JFK",
    "ac_type": "B738",
    "cruise_alt_ft": 37000,
    "mass_kg": 65000,
    "route_step_km": 25.0,
    "backend": "openap"
  }'
```

### Using HTTPie

HTTPie provides a more user-friendly interface for API testing:

#### Installation

```bash
pip install httpie
```

#### Examples

```bash
# Health check
http GET localhost:8000/health

# Airport search
http GET localhost:8000/airports/by_city city=="San Francisco" country=="US"

# Flight planning
http POST localhost:8000/plan \
  depart_city="Los Angeles" \
  arrive_city="Chicago" \
  ac_type="A321" \
  cruise_alt_ft:=36000 \
  route_step_km:=30.0 \
  backend="openap"
```

### Using Python Requests

```python
import requests
import json

# Base URL for the API
BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(f"Health: {response.json()}")

# Find airports
airports = requests.get(
    f"{BASE_URL}/airports/by_city",
    params={"city": "Tokyo", "country": "JP"}
)
print(f"Tokyo airports: {airports.json()}")

# Plan a flight
plan_request = {
    "depart_city": "Tokyo",
    "arrive_city": "Sydney",
    "ac_type": "B777",
    "cruise_alt_ft": 39000,
    "route_step_km": 100.0,
    "backend": "openap"
}

response = requests.post(
    f"{BASE_URL}/plan",
    headers={"Content-Type": "application/json"},
    json=plan_request
)

if response.status_code == 200:
    plan = response.json()
    print(f"Flight distance: {plan['distance_nm']:.1f} NM")
    print(f"Estimated block time: {plan['estimates']['block']['time_min']:.1f} minutes")
    print(f"Estimated fuel: {plan['estimates']['block']['fuel_kg']:.1f} kg")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### JavaScript/TypeScript Integration

```typescript
interface PlanRequest {
  depart_city: string;
  arrive_city: string;
  depart_country?: string;
  arrive_country?: string;
  prefer_depart_iata?: string;
  prefer_arrive_iata?: string;
  ac_type: string;
  cruise_alt_ft?: number;
  mass_kg?: number;
  route_step_km?: number;
  backend: "openap";
}

interface Airport {
  iata: string;
  icao: string;
  name: string;
  city: string;
  country: string;
  lat: number;
  lon: number;
  tz?: string;
}

class AerospaceMCP {
  constructor(private baseUrl: string = "http://localhost:8000") {}

  async health(): Promise<{status: string, openap: boolean, airports_count: number}> {
    const response = await fetch(`${this.baseUrl}/health`);
    return response.json();
  }

  async findAirports(city: string, country?: string): Promise<Airport[]> {
    const params = new URLSearchParams({ city });
    if (country) params.append("country", country);
    
    const response = await fetch(`${this.baseUrl}/airports/by_city?${params}`);
    return response.json();
  }

  async planFlight(request: PlanRequest): Promise<any> {
    const response = await fetch(`${this.baseUrl}/plan`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Planning failed: ${response.statusText}`);
    }

    return response.json();
  }
}

// Usage example
const client = new AerospaceMCP();

async function example() {
  try {
    // Check service health
    const health = await client.health();
    console.log("Service status:", health.status);

    // Find airports
    const airports = await client.findAirports("Paris", "FR");
    console.log("Paris airports:", airports.map(a => a.iata));

    // Plan a flight
    const plan = await client.planFlight({
      depart_city: "Paris",
      arrive_city: "London",
      ac_type: "A319",
      cruise_alt_ft: 35000,
      backend: "openap"
    });

    console.log(`Flight planned: ${plan.distance_nm.toFixed(1)} NM`);
  } catch (error) {
    console.error("Error:", error);
  }
}
```

## Authentication & Security

### Current Status

The current version of Aerospace MCP does not implement authentication. This is suitable for:
- Local development
- Internal corporate networks
- Proof-of-concept deployments

### Future Authentication Options

For production deployments, consider implementing:

#### API Key Authentication

```python
# Example implementation (not currently in main.py)
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header()):
    if x_api_key != "your-secret-api-key":
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

@app.post("/plan", dependencies=[Depends(verify_api_key)])
def plan(req: PlanRequest):
    # Implementation
    pass
```

Usage with authentication:

```bash
curl -X POST "http://localhost:8000/plan" \
  -H "X-API-Key: your-secret-api-key" \
  -H "Content-Type: application/json" \
  -d '{"depart_city": "NYC", "arrive_city": "LAX", "ac_type": "A320"}'
```

#### OAuth 2.0 / JWT

For more sophisticated authentication:

```bash
# Get token
TOKEN=$(curl -X POST "http://localhost:8000/token" \
  -d "username=user&password=pass" | jq -r '.access_token')

# Use token
curl -X POST "http://localhost:8000/plan" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"depart_city": "NYC", "arrive_city": "LAX", "ac_type": "A320"}'
```

### HTTPS/TLS

For production deployment:

```bash
# Run with HTTPS (requires SSL certificates)
uvicorn main:app --host 0.0.0.0 --port 443 \
  --ssl-keyfile=key.pem --ssl-certfile=cert.pem
```

## Rate Limiting

### Basic Rate Limiting

Consider implementing rate limiting for production use:

```python
# Example with slowapi (not currently implemented)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/plan")
@limiter.limit("10/minute")
def plan(request: Request, req: PlanRequest):
    # Implementation
    pass
```

### Client-Side Rate Limiting

```python
import time
import asyncio
from typing import AsyncGenerator

class RateLimitedClient:
    def __init__(self, requests_per_minute: int = 60):
        self.min_interval = 60.0 / requests_per_minute
        self.last_request = 0.0

    async def request(self, method, url, **kwargs):
        # Rate limiting logic
        current_time = time.time()
        time_since_last = current_time - self.last_request
        
        if time_since_last < self.min_interval:
            await asyncio.sleep(self.min_interval - time_since_last)
        
        self.last_request = time.time()
        
        # Make the actual request
        # Implementation depends on your HTTP client
        pass
```

## Common Integration Patterns

### Batch Flight Planning

```python
import asyncio
import aiohttp

async def plan_multiple_flights(flight_requests: list) -> list:
    """Plan multiple flights concurrently with rate limiting."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for req in flight_requests:
            task = plan_single_flight(session, req)
            tasks.append(task)
            # Add small delay to avoid overwhelming the server
            await asyncio.sleep(0.1)
        
        return await asyncio.gather(*tasks, return_exceptions=True)

async def plan_single_flight(session, request):
    async with session.post(
        "http://localhost:8000/plan",
        json=request
    ) as response:
        return await response.json()

# Usage
requests = [
    {"depart_city": "NYC", "arrive_city": "LAX", "ac_type": "A320"},
    {"depart_city": "LAX", "arrive_city": "CHI", "ac_type": "B737"},
    {"depart_city": "CHI", "arrive_city": "MIA", "ac_type": "A321"},
]

results = asyncio.run(plan_multiple_flights(requests))
```

### Caching Integration

```python
import redis
import json
import hashlib

class CachedFlightPlanner:
    def __init__(self, redis_url="redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.base_url = "http://localhost:8000"
    
    def _cache_key(self, request_data: dict) -> str:
        """Generate cache key from request parameters."""
        request_str = json.dumps(request_data, sort_keys=True)
        return f"flight_plan:{hashlib.md5(request_str.encode()).hexdigest()}"
    
    def plan_flight(self, request_data: dict, cache_ttl: int = 3600):
        """Plan flight with Redis caching."""
        cache_key = self._cache_key(request_data)
        
        # Check cache first
        cached_result = self.redis.get(cache_key)
        if cached_result:
            return json.loads(cached_result)
        
        # Make API request
        response = requests.post(
            f"{self.base_url}/plan",
            json=request_data
        )
        
        if response.status_code == 200:
            result = response.json()
            # Cache the result
            self.redis.setex(
                cache_key,
                cache_ttl,
                json.dumps(result)
            )
            return result
        else:
            response.raise_for_status()
```

### WebSocket Integration (Future Enhancement)

```python
# Example WebSocket endpoint for real-time flight tracking
from fastapi import WebSocket

@app.websocket("/ws/track/{flight_id}")
async def websocket_flight_tracking(websocket: WebSocket, flight_id: str):
    await websocket.accept()
    try:
        while True:
            # Send flight updates
            update = await get_flight_update(flight_id)
            await websocket.send_json(update)
            await asyncio.sleep(30)  # Update every 30 seconds
    except WebSocketDisconnect:
        pass
```

## Troubleshooting

### Common Issues

#### 1. OpenAP Import Errors

**Problem**: `ModuleNotFoundError: No module named 'openap'`

**Solution**:
```bash
pip install openap
# or
uv add openap
```

**Graceful Handling**: The application will still work without OpenAP, but performance estimates will be unavailable.

#### 2. Airport Not Found

**Problem**: City name not recognized

**Solutions**:
- Use more specific city names
- Include country code parameter
- Use IATA code directly with `prefer_depart_iata` or `prefer_arrive_iata`

```bash
# Instead of:
curl "http://localhost:8000/airports/by_city?city=Paris"

# Try:
curl "http://localhost:8000/airports/by_city?city=Paris&country=FR"
```

#### 3. Connection Refused

**Problem**: `Connection refused` when accessing the API

**Solutions**:
- Ensure the server is running: `uvicorn main:app --reload`
- Check the correct port (default: 8000)
- Verify firewall settings
- For Docker: ensure port mapping is correct

#### 4. Performance Estimation Errors

**Problem**: Aircraft type not recognized by OpenAP

**Solutions**:
- Use standard ICAO aircraft codes (e.g., "A320", "B738", "B777")
- Check OpenAP documentation for supported aircraft
- Verify aircraft type spelling

#### 5. Large Response Times

**Problem**: Slow API responses

**Solutions**:
- Increase `route_step_km` to reduce polyline resolution
- Use caching for repeated requests
- Consider pagination for large result sets
- Monitor server resources

### Debug Mode

Enable debug logging:

```bash
# Set environment variable
export DEBUG=true
export LOG_LEVEL=debug

# Or run with debug flags
uvicorn main:app --reload --log-level debug
```

### Health Checks

Implement health checks in your integration:

```python
def check_service_health():
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        health_data = response.json()
        
        if health_data["status"] != "ok":
            raise Exception("Service unhealthy")
            
        if not health_data["openap"]:
            print("Warning: OpenAP not available")
            
        return True
    except Exception as e:
        print(f"Health check failed: {e}")
        return False
```

### Performance Monitoring

```python
import time
from functools import wraps

def monitor_api_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            raise
        finally:
            duration = time.time() - start_time
            print(f"API call {func.__name__}: {duration:.2f}s, success: {success}")
        return result
    return wrapper

@monitor_api_calls
def plan_flight(request_data):
    return requests.post("http://localhost:8000/plan", json=request_data)
```

### Contact & Support

For integration issues:
- Check the [API documentation](API.md)
- Review the [architecture guide](ARCHITECTURE.md)
- Create an issue on GitHub
- Check existing discussions and issues

Remember to include:
- Your integration environment (Python version, OS, etc.)
- Complete error messages
- Minimal reproduction example
- Expected vs actual behavior