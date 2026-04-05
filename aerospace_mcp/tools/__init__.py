"""Tool modules for the Aerospace MCP server.

This package contains all MCP tool implementations organized by aerospace domain:

- **core**: Flight planning, airport search, distance calculation, and aircraft performance.
- **atmosphere**: ISA atmospheric modeling and wind profile calculations.
- **frames**: Coordinate frame transformations (ECEF, ECI, geodetic, etc.).
- **aerodynamics**: VLM wing analysis, airfoil polars, and stability derivatives.
- **propellers**: BEMT propeller analysis and UAV energy estimation.
- **rockets**: 3DOF rocket trajectory simulation, sizing, and launch optimization.
- **orbits**: Orbital mechanics including Lambert solver, Hohmann transfers, and propagation.
- **gnc**: Guidance, Navigation, and Control (Kalman filtering, LQR design).
- **optimization**: Trajectory optimization (GA, PSO, Monte Carlo, porkchop plots).
- **agents**: LLM-assisted tool selection and data formatting helpers.
- **tool_search**: Dynamic tool discovery via regex and natural language search.
- **performance**: Aircraft performance (density altitude, airspeed, stall, W&B, fuel).

WARNING: This module is for educational and research purposes only.
Do NOT use for real flight planning, navigation, or aircraft operations.
"""
