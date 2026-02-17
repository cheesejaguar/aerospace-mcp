---
name: aerospace-tools
description: Discover, search, and explore all 44 aerospace-mcp tools across 11 domains
---

# Aerospace Tool Discovery

Meta-skill for discovering and searching across all 44 aerospace-mcp tools organized into 11 domains. Use this when unsure which tool to use for a given task.

## Tool Categories

| Category | Tools | Description |
|----------|-------|-------------|
| core | 5 | Flight planning, airport search, distances |
| atmosphere | 2 | ISA atmosphere profiles, wind models |
| frames | 3 | Coordinate frame transformations (ECEF, ECI, geodetic) |
| aerodynamics | 4 | Wing VLM analysis, airfoil polars, stability derivatives |
| propellers | 3 | Propeller BEMT, UAV energy estimation |
| rockets | 3 | 3DOF trajectory, sizing, launch angle optimization |
| orbits | 7 | Hohmann transfers, Lambert problem, orbit propagation, ground tracks |
| gnc | 2 | Kalman filters, LQR controller design |
| performance | 7 | Density altitude, airspeeds, stall, W&B, takeoff/landing, fuel |
| optimization | 6 | Thrust profiles, sensitivity, GA, PSO, porkchop plots, Monte Carlo |
| agents | 2 | LLM-assisted tool selection and data formatting |

## CLI Examples

```bash
# List all available tools
aerospace-mcp-cli list

# List tools in a specific category
aerospace-mcp-cli list --category orbits
aerospace-mcp-cli list --category performance

# Search for tools by keyword
aerospace-mcp-cli search "fuel"
aerospace-mcp-cli search "orbit transfer"
aerospace-mcp-cli search "trajectory" --category rockets

# Get detailed info about a specific tool
aerospace-mcp-cli info hohmann_transfer
aerospace-mcp-cli info takeoff_performance
aerospace-mcp-cli info search_airports

# Run the tool search/discovery tools directly
aerospace-mcp-cli run search_aerospace_tools --query "trajectory optimization"
aerospace-mcp-cli run search_aerospace_tools --query ".*orbit.*" --category orbits
aerospace-mcp-cli run list_tool_categories
```

## Programmatic Usage

```python
from aerospace_mcp.tools.tool_search import search_aerospace_tools, list_tool_categories

# Search for tools matching a query
result = search_aerospace_tools("orbital mechanics")
print(result)

# Search with regex
result = search_aerospace_tools(".*fuel.*", search_type="regex")
print(result)

# List all categories
categories = list_tool_categories()
print(categories)
```

## Using the CLI

The `aerospace-mcp-cli` command provides four subcommands:

### `list` — Browse available tools
```bash
aerospace-mcp-cli list                    # All 44 tools
aerospace-mcp-cli list -c core            # Only core tools
```

### `search` — Find tools by keyword
```bash
aerospace-mcp-cli search "distance"       # Text search
aerospace-mcp-cli search "fuel" -n 10     # Up to 10 results
```

### `info` — Tool parameter details
```bash
aerospace-mcp-cli info calculate_distance # Shows params, types, defaults
```

### `run` — Execute a tool
```bash
# Simple scalar args
aerospace-mcp-cli run hohmann_transfer --r1_m 6778000 --r2_m 42164000

# JSON dict/list args
aerospace-mcp-cli run plan_flight \
  --departure '{"city":"London","iata":"LHR"}' \
  --arrival '{"city":"New York","iata":"JFK"}'

# No-arg tools
aerospace-mcp-cli run get_system_status
```

## Parameter Reference

### search_aerospace_tools
- `--query` (str, required): Search query — text or regex pattern (e.g., `".*orbit.*"`)
- `--search_type` (one of: regex, text, auto; default: auto)
- `--max_results` (int, default: 5): Maximum results to return (max 10)
- `--category` (str, optional): Filter by category name

### list_tool_categories
No parameters. Returns all categories with tool counts.
